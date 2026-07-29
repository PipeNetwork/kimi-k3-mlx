#!/usr/bin/env python3
"""Prove separate JACCL payload and control groups on two RDMA ranks."""

import mlx.core as mx

try:
    from scripts.distributed_groups import init_distributed_groups
except ModuleNotFoundError:  # Direct ``python scripts/dual_group_smoke.py``.
    from distributed_groups import init_distributed_groups


def main() -> int:
    payload, control = init_distributed_groups("jaccl")

    rank = payload.rank()
    if (payload.size(), control.size(), control.rank()) != (2, 2, rank):
        raise RuntimeError("payload/control group mismatch")

    mx.eval(
        mx.distributed.all_sum(
            mx.ones((10,), dtype=mx.float32), group=control, stream=mx.cpu
        )
    )
    shape = (5, 1, 64, 7168)
    if rank == 1:
        packet = mx.full(shape, 1.25, dtype=mx.bfloat16)
        transferred = mx.distributed.send(
            packet, 0, group=payload, stream=mx.cpu
        )
    else:
        transferred = mx.distributed.recv_like(
            mx.zeros(shape, dtype=mx.bfloat16),
            1,
            group=payload,
            stream=mx.cpu,
        )
    mx.eval(transferred)
    mx.eval(
        mx.distributed.all_sum(
            mx.ones((10,), dtype=mx.float32), group=control, stream=mx.cpu
        )
    )
    error = mx.max(mx.abs(transferred.astype(mx.float32) - 1.25)).item()
    gathered = mx.distributed.all_gather(
        mx.array([rank], dtype=mx.int32), group=control, stream=mx.cpu
    )
    mx.eval(gathered)
    if error != 0 or gathered.tolist() != [0, 1]:
        raise AssertionError(f"rank={rank} error={error} gathered={gathered.tolist()}")
    print(f"[rank {rank}] dual JACCL groups: boundary_error={error}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
