# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An earlier stage of a stream-joined pipeline, as a process of its own.

    python -m vllm.v1.executor.pipeline_stage <model> \\
        --pipeline-parallel-size 2 --pipeline-stream-stage 0 \\
        --pipeline-stream-listen 127.0.0.1:47000 [engine options]

It loads its layers (``VLLM_PP_LAYER_PARTITION`` or the even split), joins
its neighbours (``vllm.distributed.stream_pipeline``) and runs the calls the
last stage sends (``StreamPipelineExecutor``) until the stream closes. It
takes the same engine options as the last stage, so every stage builds the
same configuration. With ``--tensor-parallel-size`` above 1 the stage runs
one worker process per rank (``StreamStageWorkers``), each serving the calls
in step with the others.
"""

from __future__ import annotations

import pickle
import sys
from typing import Any
from multiprocessing import Lock

from vllm.distributed.parallel_state import get_pp_group
from vllm.distributed.stream_pipeline import StageCalls
from vllm.engine.arg_utils import EngineArgs
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import get_file_store_init_method
from vllm.v1.executor.stream_pipeline_executor import decode_call
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.serial_utils import run_method
from vllm.v1.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


def check_stage(vllm_config: Any) -> None:
    parallel = vllm_config.parallel_config
    if not parallel.uses_pipeline_stream:
        raise SystemExit("a pipeline stage needs --pipeline-stream-stage")
    if parallel.pipeline_stream_stage == parallel.pipeline_parallel_size - 1:
        raise SystemExit("the last stage runs the engine (vllm serve), not this")


def build_worker(vllm_config: Any) -> WorkerWrapperBase:
    check_stage(vllm_config)
    parallel = vllm_config.parallel_config
    device = str(vllm_config.device_config.device).split(":")
    local_rank = int(device[1]) if len(device) > 1 else 0
    worker = WorkerWrapperBase(rpc_rank=0, global_rank=parallel.pipeline_stream_stage)
    worker.init_worker(
        all_kwargs=[
            dict(
                vllm_config=vllm_config,
                local_rank=local_rank,
                rank=0,
                distributed_init_method=get_file_store_init_method(),
                is_driver_worker=True,
                shared_worker_lock=Lock(),
            )
        ]
    )
    worker.init_device()
    worker.load_model()
    return worker


def serve(worker: Any) -> None:
    """Runs the last stage's calls on this worker until the stream closes;
    every tensor-parallel rank of the stage runs it in step."""
    calls = StageCalls(get_pp_group())

    def run(payload: bytes) -> bytes:
        try:
            method, args, kwargs = decode_call(payload)
            result = run_method(worker, method, args, kwargs)
            if isinstance(result, AsyncModelRunnerOutput):
                result = result.get_output()
        except BaseException as error:  # noqa: BLE001
            logger.exception("pipeline stage call failed")
            result = RuntimeError(f"pipeline stage call failed: {error!r}")
        try:
            return pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:  # noqa: BLE001
            return pickle.dumps(None)

    while calls.serve_one(run):
        pass
    logger.info("the next stage closed the stream; this stage stops")


def main(argv: list[str] | None = None) -> None:
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("model_tag", nargs="?", help="the model (same as --model)")
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args(argv)
    if args.model_tag:
        args.model = args.model_tag
    engine_args = EngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config()
    if vllm_config.parallel_config.world_size == 1:
        serve(build_worker(vllm_config))
        return
    check_stage(vllm_config)
    from vllm.v1.executor.stream_pipeline_executor import StreamStageWorkers

    workers = StreamStageWorkers(vllm_config)
    try:
        workers.collective_rpc(serve)
    finally:
        workers.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:])
