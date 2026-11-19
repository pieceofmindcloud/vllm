# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The executor of the last stage of a stream-joined pipeline.

The last stage holds the engine (scheduler, API server) and one local worker;
the earlier stages are separate processes (``vllm.v1.executor.pipeline_stage``)
reached over the stage streams (``vllm.distributed.stream_pipeline``). Every
collective call goes to all stages: it is sent back along the chain, run on
each stage in order, and the replies come forward, first stage first, so a
call answers one value per rank as a multi-worker executor does.

A last stage with several tensor-parallel ranks runs them as worker processes
(``StreamPipelineMultiprocExecutor``); its rank 0 holds the stream and sends
each call back before running its own part.
"""

from __future__ import annotations

import functools
import pickle
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import cloudpickle

from vllm.distributed.parallel_state import get_pp_group
from vllm.distributed.stream_pipeline import StageCalls
from vllm.logger import init_logger
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.serial_utils import run_method

logger = init_logger(__name__)


def initialize_kv_cache(worker: Any, kv_cache_configs: list[Any]) -> None:
    """``initialize_from_config`` on a worker of a stream stage: its config is
    the one at its place in the whole pipeline (``pipeline_rank``), which its
    own rank does not give when the stages have different tensor-parallel
    sizes."""
    from vllm.config import set_current_vllm_config

    config = kv_cache_configs[get_pp_group().pipeline_rank]
    with set_current_vllm_config(worker.vllm_config):
        worker.worker.initialize_from_config(config)


class _StreamKvInit:
    def initialize_from_config(self, kv_cache_configs: list[Any]) -> None:
        self.collective_rpc(initialize_kv_cache, args=(kv_cache_configs,))  # type: ignore[attr-defined]


def encode_call(method: str | Callable, args: tuple, kwargs: dict) -> bytes:
    if not isinstance(method, str):
        method = cloudpickle.dumps(method)
    return pickle.dumps((method, args, kwargs), protocol=pickle.HIGHEST_PROTOCOL)


def decode_call(payload: bytes) -> tuple[str | bytes, tuple, dict]:
    return pickle.loads(payload)


class StreamPipelineExecutor(_StreamKvInit, UniProcExecutor):
    """One local worker (the last stage) plus the earlier stages over the
    streams. Calls run one after the other, in the order they are made, on a
    dedicated thread, so a non-blocking call returns at once and the pipeline
    keeps ``pipeline_parallel_size`` batches in flight."""

    def _init_executor(self) -> None:
        self._order = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pp-stream-call")
        self._lock = threading.Lock()
        super()._init_executor()
        stage = self.parallel_config.pipeline_stream_stage
        self.driver_worker.global_rank = stage
        self._calls = StageCalls(get_pp_group())
        logger.info(
            "stream pipeline: stage %d of %d drives the earlier stages",
            stage,
            self.parallel_config.pipeline_parallel_size,
        )

    def _all_stages(
        self, method: str | Callable, args: tuple, kwargs: dict
    ) -> list[Any]:
        """Run on every stage; the earlier stages' results first."""
        payload = encode_call(method, args, kwargs)
        remote: Future = self._order_remote(payload)
        local = run_method(self.driver_worker, method, args, kwargs)
        if isinstance(local, AsyncModelRunnerOutput):
            local = local.get_output()
        earlier = [pickle.loads(reply) for reply in remote.result()]
        for reply in earlier:
            if isinstance(reply, BaseException):
                raise reply
        return [*earlier, local]

    def _order_remote(self, payload: bytes) -> Future:
        future: Future = Future()

        def send() -> None:
            try:
                future.set_result(self._calls.call(payload, want_reply=True))
            except BaseException as error:  # noqa: BLE001
                future.set_exception(error)

        threading.Thread(target=send, daemon=True).start()
        return future

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        single_value: bool = False,
    ) -> Any:
        kwargs = kwargs or {}
        if self.parallel_config.pipeline_parallel_size == 1:
            return super().collective_rpc(
                method, timeout, args, kwargs, non_block, single_value
            )

        def run() -> Any:
            with self._lock:
                results = self._all_stages(method, args, kwargs)
            # The last stage's value is the pipeline's output.
            return results[-1] if single_value else results

        future = self._order.submit(run)
        if non_block:
            return future
        return future.result(timeout)

    def shutdown(self) -> None:
        self._order.shutdown(wait=False, cancel_futures=True)
        super().shutdown()

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        return False


class StreamStageWorkers(MultiprocExecutor):
    """The tensor-parallel workers of one stream stage: the stage's world is
    its own ranks, not the whole pipeline's."""

    def _get_parallel_sizes(self) -> tuple[int, int, int]:
        tp_size, _, pcp_size = super()._get_parallel_sizes()
        return tp_size, 1, pcp_size


def _run_on_stages(worker: Any, payload: bytes, want_earlier: bool) -> Any:
    """On each worker of the last stage: the leader sends the call back
    along the chain, every rank runs it, and the leader collects the earlier
    stages' replies (one per rank, first stage first)."""
    method, args, kwargs = decode_call(payload)
    group = get_pp_group()
    calls = StageCalls(group) if group.is_leader else None
    if calls is not None:
        calls.send_call(payload, want_reply=True)
    try:
        local = run_method(worker, method, args, kwargs)
        if isinstance(local, AsyncModelRunnerOutput):
            local = local.get_output()
    finally:
        replies = calls.recv_replies() if calls is not None else []
    earlier = [pickle.loads(reply) for reply in replies]
    for reply in earlier:
        if isinstance(reply, BaseException):
            raise reply
    if not want_earlier:
        return local
    return (earlier if calls is not None else None), local


def _then(future: Future, apply: Callable[[Any], Any]) -> Future:
    out: Future = Future()

    def done(source: Future) -> None:
        try:
            out.set_result(apply(source.result()))
        except BaseException as error:  # noqa: BLE001
            out.set_exception(error)

    future.add_done_callback(done)
    return out


class StreamPipelineMultiprocExecutor(_StreamKvInit, StreamStageWorkers):
    """The last stage with several tensor-parallel ranks: every call runs on
    its workers and, through its rank 0, on the earlier stages."""

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: Any = None,
        ec_output_aggregator: Any = None,
    ) -> Any:
        payload = encode_call(method, args, kwargs or {})
        # A call that answers one rank's value (or an aggregate) leaves the
        # earlier stages' values out; one that answers every rank's puts them
        # first, in rank order.
        want_earlier = (
            unique_reply_rank is None
            and kv_output_aggregator is None
            and ec_output_aggregator is None
        )
        result = super().collective_rpc(
            functools.partial(_run_on_stages, payload=payload, want_earlier=want_earlier),
            timeout=timeout,
            non_block=non_block,
            unique_reply_rank=unique_reply_rank,
            kv_output_aggregator=kv_output_aggregator,
            ec_output_aggregator=ec_output_aggregator,
        )
        if not want_earlier:
            return result

        def flatten(results: list[tuple[list[Any] | None, Any]]) -> list[Any]:
            earlier = results[0][0] or []
            return [*earlier, *(local for _, local in results)]

        return _then(result, flatten) if non_block else flatten(result)

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        return False
