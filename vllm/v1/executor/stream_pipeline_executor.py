# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The executor of the last stage of a stream-joined pipeline.

The last stage holds the engine (scheduler, API server) and one local worker;
the earlier stages are separate processes (``vllm.v1.executor.pipeline_stage``)
reached over the stage streams (``vllm.distributed.stream_pipeline``). Every
collective call goes to all stages: it is sent back along the chain, run on
each stage in order, and the replies come forward, first stage first, so a
call answers one value per stage as a multi-worker executor does.
"""

from __future__ import annotations

import pickle
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import cloudpickle

from vllm.distributed.parallel_state import get_pp_group
from vllm.distributed.stream_pipeline import StageCalls
from vllm.logger import init_logger
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.serial_utils import run_method

logger = init_logger(__name__)


def encode_call(method: str | Callable, args: tuple, kwargs: dict) -> bytes:
    if not isinstance(method, str):
        method = cloudpickle.dumps(method)
    return pickle.dumps((method, args, kwargs), protocol=pickle.HIGHEST_PROTOCOL)


def decode_call(payload: bytes) -> tuple[str | bytes, tuple, dict]:
    return pickle.loads(payload)


class StreamPipelineExecutor(UniProcExecutor):
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
