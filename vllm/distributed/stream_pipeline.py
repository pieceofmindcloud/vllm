# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline parallelism over byte streams between adjacent stages.

Each stage of a pipeline is its own process group (its tensor-parallel ranks
only); adjacent stages are joined by one bidirectional byte stream (a TCP
connection). Everything that crosses a stage boundary travels as a frame on
that stream, staged through host memory, so stages need no shared
``torch.distributed`` world and may run on different accelerator vendors:

- forward (stage ``r`` to ``r + 1``): the intermediate tensors of each step
  (``TENSORS``) and the replies of the earlier stages to a collective call
  (``REPLY``);
- backward (stage ``r + 1`` to ``r``): the collective calls the driving stage
  makes (``CALL``) and the sampled tokens it broadcasts (``BROADCAST``).

A stage with several tensor-parallel ranks has one stream per neighbour, held
by its tensor-parallel rank 0 (the leader); the leader relays what crosses
the stream to the other ranks of its stage (``StageLocal``).

A frame is a little-endian ``u32`` length followed by that many bytes; the
first byte of the payload is its kind. A tensor payload is a ``u32`` header
length, a JSON header (names, dtypes, shapes, the wire dtype) and the raw
bytes of each tensor in order.
"""

from __future__ import annotations

import json
import queue
import socket
import struct
import threading
from collections.abc import Callable
from typing import Any

import torch

TENSORS = 1
BROADCAST = 2
CALL = 3
REPLY = 4
REDUCE_UP = 5
REDUCE_DOWN = 6
# At startup, from each stage to the next: how many ranks come before it.
OFFSET = 7

_KINDS = (TENSORS, BROADCAST, CALL, REPLY, REDUCE_UP, REDUCE_DOWN, OFFSET)

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "bool": torch.bool,
    "uint8": torch.uint8,
}


def _dtype_name(dtype: torch.dtype) -> str:
    for name, value in _DTYPES.items():
        if value == dtype:
            return name
    raise TypeError(f"unsupported tensor dtype on a pipeline stream: {dtype}")


def encode_tensors(
    tensors: dict[str, torch.Tensor], wire_dtype: torch.dtype | None = None
) -> bytes:
    """Serialize named tensors. Floating tensors are cast to ``wire_dtype``
    when given (the receiver restores their original dtype)."""
    header = []
    blobs = []
    for name, tensor in tensors.items():
        original = tensor.dtype
        sent = tensor.detach()
        if wire_dtype is not None and sent.is_floating_point():
            sent = sent.to(wire_dtype)
        sent = sent.contiguous().cpu()
        raw = sent.view(torch.uint8).numpy().tobytes() if sent.numel() else b""
        header.append(
            {
                "name": name,
                "dtype": _dtype_name(original),
                "wire": _dtype_name(sent.dtype),
                "shape": list(sent.shape),
                "bytes": len(raw),
            }
        )
        blobs.append(raw)
    head = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<I", len(head)) + head + b"".join(blobs)


def decode_tensors(
    payload: bytes | memoryview, device: torch.device | str = "cpu"
) -> dict[str, torch.Tensor]:
    view = memoryview(payload)
    (head_len,) = struct.unpack_from("<I", view, 0)
    header = json.loads(bytes(view[4 : 4 + head_len]))
    offset = 4 + head_len
    out: dict[str, torch.Tensor] = {}
    for entry in header:
        size = entry["bytes"]
        wire = _DTYPES[entry["wire"]]
        shape = entry["shape"]
        if size:
            flat = torch.frombuffer(bytearray(view[offset : offset + size]), dtype=wire)
        else:
            flat = torch.empty(0, dtype=wire)
        offset += size
        tensor = flat.reshape(shape).to(device)
        out[entry["name"]] = tensor.to(_DTYPES[entry["dtype"]])
    return out


class FramedStream:
    """One bidirectional stream: framed writes, and a reader thread that
    sorts incoming frames by kind into queues."""

    def __init__(self, sock: socket.socket):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock
        self._send_lock = threading.Lock()
        self._queues: dict[int, queue.Queue] = {kind: queue.Queue() for kind in _KINDS}
        self._closed = threading.Event()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def send(self, kind: int, payload: bytes) -> None:
        frame = struct.pack("<IB", len(payload) + 1, kind) + payload
        with self._send_lock:
            self._sock.sendall(frame)

    def recv(self, kind: int, timeout: float | None = None) -> bytes:
        item = self._queues[kind].get(timeout=timeout)
        if item is None:
            raise ConnectionError("pipeline stream closed")
        return item

    def close(self) -> None:
        self._closed.set()
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()

    def _read_exact(self, size: int) -> bytes | None:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = self._sock.recv(min(size - len(chunks), 1 << 20))
            if not chunk:
                return None
            chunks += chunk
        return bytes(chunks)

    def _read_loop(self) -> None:
        try:
            while not self._closed.is_set():
                head = self._read_exact(4)
                if head is None:
                    break
                (length,) = struct.unpack("<I", head)
                body = self._read_exact(length)
                if body is None or not body:
                    break
                kind = body[0]
                if kind in self._queues:
                    self._queues[kind].put(body[1:])
        except OSError:
            pass
        finally:
            for q in self._queues.values():
                q.put(None)


class StageLocal:
    """The tensor-parallel ranks of one stage. Rank 0, the leader, holds the
    streams; what crosses them reaches the other ranks over gloo groups of
    the stage's own world: ``tokens`` for the sampled-token broadcast (it runs
    on the receiver thread), ``control`` for the calls and the CPU
    collectives (on the main thread), so the two never interleave."""

    def __init__(self, rank: int, size: int):
        import torch.distributed as dist

        ranks = list(range(dist.get_world_size()))
        assert len(ranks) == size, "a stream stage's world is its tensor-parallel ranks"
        self.rank = rank
        self.size = size
        self.tokens = dist.new_group(ranks, backend="gloo")
        self.control = dist.new_group(ranks, backend="gloo")
        self.leader = dist.get_global_rank(self.control, 0)

    @property
    def is_leader(self) -> bool:
        return self.rank == 0

    def share_tensor(self, tensor: torch.Tensor, group: Any) -> None:
        """The leader's ``tensor`` into the same tensor of every rank."""
        import torch.distributed as dist

        if self.is_leader:
            staged = tensor.detach().contiguous().cpu()
        else:
            staged = torch.empty(tensor.shape, dtype=tensor.dtype)
        dist.broadcast(staged, src=self.leader, group=group)
        if not self.is_leader:
            tensor.copy_(staged.reshape(tensor.shape))

    def share_object(self, value: Any, group: Any) -> Any:
        import torch.distributed as dist

        box = [value]
        dist.broadcast_object_list(box, src=self.leader, group=group)
        return box[0]

    def gather_objects(self, value: Any, group: Any) -> list[Any] | None:
        """Every rank's ``value`` on the leader, in rank order; None elsewhere."""
        import torch.distributed as dist

        out: list[Any] | None = [None] * self.size if self.is_leader else None
        dist.gather_object(value, out, dst=self.leader, group=group)
        return out


class StreamPipelineGroup:
    """The pipeline group of a stage joined to its neighbours by streams.

    Duck-types the part of ``GroupCoordinator`` the pipeline path uses:
    ``rank_in_group``, ``world_size``, ``is_first_rank``, ``is_last_rank``,
    ``first_rank``, ``last_rank``, ``ranks``, the tensor-dict send/receive
    and ``broadcast_tensor`` for the sampled-token broadcast.
    """

    def __init__(
        self,
        stage: int,
        num_stages: int,
        prev: FramedStream | None,
        next: FramedStream | None,
        device: torch.device | str = "cpu",
        wire_dtype: torch.dtype | None = None,
        local: StageLocal | None = None,
        rank_offset: int = 0,
    ):
        assert 0 <= stage < num_stages
        if local is None or local.is_leader:
            assert (prev is None) == (stage == 0)
            assert (next is None) == (stage == num_stages - 1)
        else:
            assert prev is None and next is None, "only the leader holds the streams"
        self.rank_in_group = stage
        self.world_size = num_stages
        self.ranks = list(range(num_stages))
        self.first_rank = 0
        self.last_rank = num_stages - 1
        self.prev = prev
        self.next = next
        self.device = device
        self.wire_dtype = wire_dtype
        self.local = local
        # The ranks of the earlier stages: this stage's ranks follow them in
        # the engine's per-worker lists (one per rank of the whole pipeline).
        self.rank_offset = rank_offset
        self.use_stream_transport = True

    @property
    def pipeline_rank(self) -> int:
        """This process's position in the pipeline's per-worker lists."""
        return self.rank_offset + (self.local.rank if self.local is not None else 0)

    def destroy(self) -> None:
        for stream in (self.prev, self.next):
            if stream is not None:
                stream.close()
        self.prev = self.next = None

    @property
    def is_leader(self) -> bool:
        return self.local is None or self.local.is_leader

    @property
    def is_first_rank(self) -> bool:
        return self.rank_in_group == 0

    @property
    def is_last_rank(self) -> bool:
        return self.rank_in_group == self.last_rank

    # Intermediate tensors, forward.
    def send_tensor_dict(self, tensor_dict: dict[str, Any], dst: int | None = None, **_):
        assert not self.is_last_rank, "the last stage sends no intermediate tensors"
        if not self.is_leader:
            # every rank holds the same tensors; the leader sends them
            return None
        assert self.next is not None
        tensors = {k: v for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)}
        self.next.send(TENSORS, encode_tensors(tensors, self.wire_dtype))
        return None

    def isend_tensor_dict(self, tensor_dict: dict[str, Any], dst: int | None = None, **_):
        self.send_tensor_dict(tensor_dict)
        return []

    def recv_tensor_dict(self, src: int | None = None, **_) -> dict[str, torch.Tensor]:
        assert not self.is_first_rank, "the first stage receives no intermediate tensors"
        tensors = None
        if self.is_leader:
            assert self.prev is not None
            tensors = decode_tensors(self.prev.recv(TENSORS), self.device)
        if self.local is None:
            return tensors
        # the stage's other ranks get them from the leader over the device
        from vllm.distributed.parallel_state import get_tp_group

        return get_tp_group().broadcast_tensor_dict(tensors, src=0)

    def irecv_tensor_dict(self, src: int | None = None, **_):
        return self.recv_tensor_dict(), [], None

    # Sampled tokens, backward from the last stage.
    def broadcast_tensor(self, tensor: torch.Tensor) -> None:
        """In place on every stage: the last stage's ``tensor`` is copied into
        the same-shaped ``tensor`` of every earlier stage."""
        if self.is_last_rank:
            # every rank of the last stage sampled the same tokens
            if self.prev is not None:
                self.prev.send(BROADCAST, encode_tensors({"t": tensor}))
            return
        if self.is_leader:
            assert self.next is not None
            payload = self.next.recv(BROADCAST)
            if self.prev is not None:
                self.prev.send(BROADCAST, payload)
            received = decode_tensors(payload, tensor.device)["t"]
            tensor.copy_(received.reshape(tensor.shape))
        if self.local is not None:
            self.local.share_tensor(tensor, self.local.tokens)

    def make_sibling_device_group(self, group_desc: str | None = None):
        return self

    # Small CPU collectives every stage makes at the same point (warm-up
    # decisions): summed forward to the last stage, the sum sent back.
    def cpu_all_reduce_(self, tensor: torch.Tensor) -> torch.Tensor:
        total = tensor.clone()
        if self.is_leader:
            total = self._reduce_over_streams(total)
        if self.local is not None:
            self.local.share_tensor(total, self.local.control)
        tensor.copy_(total)
        return tensor

    def _reduce_over_streams(self, total: torch.Tensor) -> torch.Tensor:
        """Summed forward to the last stage, the sum sent back."""
        shape = total.shape
        if self.prev is not None:
            total += decode_tensors(self.prev.recv(REDUCE_UP))["t"].reshape(shape)
        if self.next is not None:
            self.next.send(REDUCE_UP, encode_tensors({"t": total}))
            total = decode_tensors(self.next.recv(REDUCE_DOWN))["t"].reshape(shape)
        if self.prev is not None:
            self.prev.send(REDUCE_DOWN, encode_tensors({"t": total}))
        return total

    def barrier(self) -> None:
        self.cpu_all_reduce_(torch.zeros(1, dtype=torch.int32))


class StageCalls:
    """Collective calls driven by the last stage over the streams.

    The last stage sends a call backward; each stage forwards it further back
    before running it, and replies travel forward, each stage appending its
    own to those of the earlier ones. Payloads are opaque bytes; the caller
    chooses the serializer (the engine's msgpack encoder).
    """

    def __init__(self, group: StreamPipelineGroup):
        self.group = group

    def call(self, payload: bytes, want_reply: bool) -> list[bytes]:
        """From the last stage: ask every earlier stage to run ``payload``.
        Returns the earlier stages' replies, first stage first, one per
        tensor-parallel rank."""
        self.send_call(payload, want_reply)
        return self.recv_replies() if want_reply else []

    def send_call(self, payload: bytes, want_reply: bool) -> None:
        """The first half of ``call``: the last stage sends the call and runs
        its own part before it waits for the replies."""
        assert self.group.is_last_rank and self.group.is_leader
        if self.group.prev is not None:
            self.group.prev.send(CALL, bytes([1 if want_reply else 0]) + payload)

    def recv_replies(self) -> list[bytes]:
        assert self.group.is_last_rank and self.group.is_leader
        if self.group.prev is None:
            return []
        return _split_replies(self.group.prev.recv(REPLY))

    def serve_one(self, run: Callable[[bytes], bytes]) -> bool:
        """On an earlier stage: take one call, pass it back, run it, reply.
        Every tensor-parallel rank of the stage calls this in step: the
        leader reads the call and shares it, and gathers the ranks' replies.
        Returns False once the stream from the next stage closes."""
        group = self.group
        local = group.local
        assert not group.is_last_rank
        if group.is_leader:
            assert group.next is not None
            try:
                frame: bytes | None = group.next.recv(CALL)
            except ConnectionError:
                frame = None
            if local is not None:
                local.share_object(frame, local.control)
            if frame is None:
                return False
            if group.prev is not None:
                group.prev.send(CALL, frame)
        else:
            assert local is not None
            frame = local.share_object(None, local.control)
            if frame is None:
                return False
        want_reply, payload = frame[0] == 1, frame[1:]
        reply = run(payload)
        if not want_reply:
            return True
        replies = [reply] if local is None else local.gather_objects(reply, local.control)
        if group.is_leader:
            assert replies is not None and group.next is not None
            earlier = (
                _split_replies(group.prev.recv(REPLY)) if group.prev is not None else []
            )
            group.next.send(REPLY, _join_replies([*earlier, *replies]))
        return True


def _join_replies(replies: list[bytes]) -> bytes:
    out = bytearray(struct.pack("<I", len(replies)))
    for reply in replies:
        out += struct.pack("<I", len(reply)) + reply
    return bytes(out)


def _split_replies(blob: bytes) -> list[bytes]:
    view = memoryview(blob)
    (count,) = struct.unpack_from("<I", view, 0)
    offset = 4
    out = []
    for _ in range(count):
        (size,) = struct.unpack_from("<I", view, offset)
        offset += 4
        out.append(bytes(view[offset : offset + size]))
        offset += size
    return out


def connect(addr: str, timeout: float = 600.0) -> FramedStream:
    """The stream to the previous stage, which listens at ``host:port``.
    Retries until ``timeout``: the previous stage may still be starting."""
    import time

    host, _, port = addr.rpartition(":")
    deadline = time.monotonic() + timeout
    while True:
        try:
            sock = socket.create_connection((host.strip("[]"), int(port)), timeout=10)
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.5)
    sock.settimeout(None)
    return FramedStream(sock)


def open_stage_group(parallel_config: Any, local_rank: int = 0) -> "StreamPipelineGroup":
    """This process's pipeline group from its parallel config: connect to the
    previous stage first, then wait for the next one (so the chain forms from
    the first stage on, whatever order the processes start in)."""
    if parallel_config.prefill_context_parallel_size != 1:
        raise NotImplementedError("stream pipeline stages take no context parallelism")
    stage = parallel_config.pipeline_stream_stage
    stages = parallel_config.pipeline_parallel_size
    local = None
    if parallel_config.tensor_parallel_size > 1:
        from vllm.distributed.parallel_state import get_tp_group

        # every rank makes the stage's gloo groups, before the leader blocks
        # on the streams
        local = StageLocal(
            get_tp_group().rank_in_group, parallel_config.tensor_parallel_size
        )
    leader = local is None or local.is_leader
    prev = connect(parallel_config.pipeline_stream_prev) if leader and stage > 0 else None
    nxt = (
        accept(parallel_config.pipeline_stream_listen)
        if leader and stage < stages - 1
        else None
    )
    # The ranks before this stage, counted along the chain.
    offset = 0
    if leader:
        if prev is not None:
            (offset,) = struct.unpack("<I", prev.recv(OFFSET))
        if nxt is not None:
            nxt.send(OFFSET, struct.pack("<I", offset + parallel_config.tensor_parallel_size))
    if local is not None:
        offset = local.share_object(offset, local.control)
    wire = parallel_config.pipeline_stream_wire_dtype
    return StreamPipelineGroup(
        stage,
        stages,
        prev,
        nxt,
        device=torch.device("cuda", local_rank) if torch.cuda.is_available() else "cpu",
        wire_dtype=_DTYPES[wire] if wire else None,
        local=local,
        rank_offset=offset,
    )


def accept(addr: str, timeout: float | None = None) -> FramedStream:
    """Listen at ``host:port`` for the next stage and return its stream."""
    host, _, port = addr.rpartition(":")
    family = socket.AF_INET6 if ":" in host.strip("[]") else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host.strip("[]"), int(port)))
        server.listen(1)
        server.settimeout(timeout)
        sock, _ = server.accept()
    sock.settimeout(None)
    return FramedStream(sock)
