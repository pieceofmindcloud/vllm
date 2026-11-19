# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import pathlib
import socket
import threading

import torch

_PATH = pathlib.Path(__file__).parents[2] / "vllm/distributed/stream_pipeline.py"
_spec = importlib.util.spec_from_file_location("stream_pipeline", _PATH)
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)


def _chain(n, wire_dtype=None):
    """n stages joined by socket pairs: groups[i].next <-> groups[i+1].prev."""
    pairs = [socket.socketpair() for _ in range(n - 1)]
    groups = []
    for i in range(n):
        prev = sp.FramedStream(pairs[i - 1][1]) if i > 0 else None
        nxt = sp.FramedStream(pairs[i][0]) if i < n - 1 else None
        groups.append(sp.StreamPipelineGroup(i, n, prev, nxt, wire_dtype=wire_dtype))
    return groups


def test_tensors_round_trip_with_a_narrower_wire_dtype():
    tensors = {
        "hidden_states": torch.randn(3, 8, dtype=torch.float32),
        "residual": torch.randn(3, 8, dtype=torch.bfloat16),
        "ids": torch.arange(5, dtype=torch.int64),
        "empty": torch.empty(0, 4),
    }
    out = sp.decode_tensors(sp.encode_tensors(tensors, torch.float16))
    assert out.keys() == tensors.keys()
    assert out["hidden_states"].dtype == torch.float32
    assert torch.allclose(out["hidden_states"], tensors["hidden_states"], atol=1e-2)
    assert out["residual"].dtype == torch.bfloat16
    assert torch.equal(out["ids"], tensors["ids"])
    assert out["empty"].shape == (0, 4)


def test_intermediate_tensors_move_forward_stage_by_stage():
    a, b, c = _chain(3)
    hidden = torch.randn(2, 4)
    a.isend_tensor_dict({"hidden_states": hidden, "note": "not a tensor"})
    got, handles, post = b.irecv_tensor_dict()
    assert handles == [] and post is None
    assert set(got) == {"hidden_states"} and torch.equal(got["hidden_states"], hidden)
    b.send_tensor_dict({"hidden_states": got["hidden_states"] * 2})
    assert torch.equal(c.recv_tensor_dict()["hidden_states"], hidden * 2)
    assert a.is_first_rank and c.is_last_rank and not b.is_first_rank
    assert b.make_sibling_device_group("pp_broadcast") is b


def test_the_last_stage_broadcast_reaches_every_earlier_stage():
    groups = _chain(3)
    sampled = torch.tensor([[7], [9]], dtype=torch.int64)
    targets = [torch.zeros(2, 1, dtype=torch.int64) for _ in range(2)]
    threads = [
        threading.Thread(target=groups[i].broadcast_tensor, args=(targets[i],))
        for i in range(2)
    ]
    for t in threads:
        t.start()
    groups[2].broadcast_tensor(sampled)
    for t in threads:
        t.join(5)
    assert all(torch.equal(t, sampled) for t in targets)


def test_calls_run_on_every_earlier_stage_and_replies_come_back_in_order():
    groups = _chain(3)
    calls = [sp.StageCalls(g) for g in groups]
    ran = []

    def serve(i):
        calls[i].serve_one(lambda payload: (ran.append((i, payload)), f"r{i}".encode())[1])

    threads = [threading.Thread(target=serve, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    replies = calls[2].call(b"load_model", want_reply=True)
    for t in threads:
        t.join(5)
    assert replies == [b"r0", b"r1"]
    assert sorted(ran) == [(0, b"load_model"), (1, b"load_model")]


def test_a_closed_stream_ends_the_call_loop():
    a, b = _chain(2)
    b.prev.close()
    assert sp.StageCalls(a).serve_one(lambda payload: b"") is False


def test_a_cpu_all_reduce_sums_over_every_stage():
    groups = _chain(3)
    values = [torch.tensor([i + 1], dtype=torch.int32) for i in range(3)]
    threads = [threading.Thread(target=g.cpu_all_reduce_, args=(v,)) for g, v in zip(groups, values)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert [int(v.item()) for v in values] == [6, 6, 6]


def test_a_stage_takes_its_place_after_the_earlier_stages_ranks():
    # stage 0 runs two tensor-parallel ranks, so stage 1's rank comes third
    first, last = _chain(2)
    last.rank_offset = 2
    assert first.pipeline_rank == 0 and last.pipeline_rank == 2
    first.next.send(sp.OFFSET, (2).to_bytes(4, "little"))
    assert int.from_bytes(last.prev.recv(sp.OFFSET, timeout=5), "little") == 2
    last.destroy()
    assert last.prev is None and last.next is None
