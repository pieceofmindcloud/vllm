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

_KINDS = (TENSORS, BROADCAST, CALL, REPLY)

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
    ):
        assert 0 <= stage < num_stages
        assert (prev is None) == (stage == 0)
        assert (next is None) == (stage == num_stages - 1)
        self.rank_in_group = stage
        self.world_size = num_stages
        self.ranks = list(range(num_stages))
        self.first_rank = 0
        self.last_rank = num_stages - 1
        self.prev = prev
        self.next = next
        self.device = device
        self.wire_dtype = wire_dtype
        self.use_stream_transport = True

    @property
    def is_first_rank(self) -> bool:
        return self.rank_in_group == 0

    @property
    def is_last_rank(self) -> bool:
        return self.rank_in_group == self.last_rank

    # Intermediate tensors, forward.
    def send_tensor_dict(self, tensor_dict: dict[str, Any], dst: int | None = None, **_):
        assert self.next is not None, "the last stage sends no intermediate tensors"
        tensors = {k: v for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)}
        self.next.send(TENSORS, encode_tensors(tensors, self.wire_dtype))
        return None

    def isend_tensor_dict(self, tensor_dict: dict[str, Any], dst: int | None = None, **_):
        self.send_tensor_dict(tensor_dict)
        return []

    def recv_tensor_dict(self, src: int | None = None, **_) -> dict[str, torch.Tensor]:
        assert self.prev is not None, "the first stage receives no intermediate tensors"
        return decode_tensors(self.prev.recv(TENSORS), self.device)

    def irecv_tensor_dict(self, src: int | None = None, **_):
        return self.recv_tensor_dict(), [], None

    # Sampled tokens, backward from the last stage.
    def broadcast_tensor(self, tensor: torch.Tensor) -> None:
        """In place on every stage: the last stage's ``tensor`` is copied into
        the same-shaped ``tensor`` of every earlier stage."""
        if self.is_last_rank:
            if self.prev is not None:
                self.prev.send(BROADCAST, encode_tensors({"t": tensor}))
            return
        assert self.next is not None
        payload = self.next.recv(BROADCAST)
        if self.prev is not None:
            self.prev.send(BROADCAST, payload)
        received = decode_tensors(payload, tensor.device)["t"]
        tensor.copy_(received.reshape(tensor.shape))

    def make_sibling_device_group(self, group_desc: str | None = None):
        return self


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
        Returns the earlier stages' replies, first stage first."""
        assert self.group.is_last_rank
        if self.group.prev is None:
            return []
        self.group.prev.send(CALL, bytes([1 if want_reply else 0]) + payload)
        if not want_reply:
            return []
        return _split_replies(self.group.prev.recv(REPLY))

    def serve_one(self, run: Callable[[bytes], bytes]) -> bool:
        """On an earlier stage: take one call, pass it back, run it, reply.
        Returns False once the stream from the next stage closes."""
        assert not self.group.is_last_rank and self.group.next is not None
        try:
            frame = self.group.next.recv(CALL)
        except ConnectionError:
            return False
        want_reply, payload = frame[0] == 1, frame[1:]
        if self.group.prev is not None:
            self.group.prev.send(CALL, frame)
        reply = run(payload)
        if want_reply:
            earlier = (
                _split_replies(self.group.prev.recv(REPLY))
                if self.group.prev is not None
                else []
            )
            self.group.next.send(REPLY, _join_replies([*earlier, reply]))
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
    """The stream to the previous stage, which listens at ``host:port``."""
    host, _, port = addr.rpartition(":")
    sock = socket.create_connection((host.strip("[]"), int(port)), timeout=timeout)
    sock.settimeout(None)
    return FramedStream(sock)


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
