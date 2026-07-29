#!/usr/bin/env python3
"""Stress JACCL collectives while large MLX buffers remain resident."""

from __future__ import annotations

import argparse
import resource
import time

import mlx.core as mx

try:
    from scripts.distributed_groups import (
        init_distributed_groups,
        prime_distributed_groups,
    )
except ModuleNotFoundError:  # Direct ``python scripts/jaccl_resident_stress.py``.
    from distributed_groups import init_distributed_groups, prime_distributed_groups


GIB = 1 << 30
MIB = 1 << 20


def allocate_resident(gib: float, block_mib: int, rank: int) -> list[mx.array]:
    """Materialize and retain deterministic MLX buffers totaling ``gib``."""
    target = round(gib * GIB)
    block_bytes = block_mib * MIB
    resident = []
    allocated = 0
    next_report = 32 * GIB
    while allocated < target:
        size = min(block_bytes, target - allocated)
        block = mx.full((size,), rank + 1, dtype=mx.uint8)
        mx.eval(block)
        resident.append(block)
        allocated += size
        if allocated >= next_report or allocated == target:
            print(
                f"[rank {rank}] resident={allocated / GIB:.1f} GiB "
                f"active={mx.get_active_memory() / GIB:.1f} GiB",
                flush=True,
            )
            next_report += 32 * GIB
    return resident


def transfer(group, control, operation: str, length: int) -> tuple[float, float]:
    rank = group.rank()
    shape = (5, 1, length, 7168)
    packet = (
        mx.full(shape, 1.25, dtype=mx.bfloat16)
        if rank == 1
        else mx.zeros(shape, dtype=mx.bfloat16)
    )
    packet = mx.contiguous(packet)
    mx.eval(packet)

    started = time.perf_counter()
    if operation == "p2p":
        if rank == 1:
            transferred = mx.distributed.send(
                packet, 0, group=group, stream=mx.cpu
            )
        else:
            transferred = mx.distributed.recv_like(
                packet, 1, group=group, stream=mx.cpu
            )
    elif operation == "all-sum":
        transferred = mx.distributed.all_sum(packet, group=group, stream=mx.cpu)
    else:
        gathered = mx.distributed.all_gather(packet, group=group, stream=mx.cpu)
        mx.eval(gathered)
        transferred = gathered[shape[0] :]
    mx.eval(transferred)
    transfer_seconds = time.perf_counter() - started

    error = mx.max(mx.abs(transferred.astype(mx.float32) - 1.25)).item()
    synchronized = mx.distributed.all_gather(
        mx.array([rank], dtype=mx.int32), group=control, stream=mx.cpu
    )
    mx.eval(synchronized)
    if error != 0 or synchronized.tolist() != [0, 1]:
        raise AssertionError(
            f"rank={rank} error={error} synchronized={synchronized.tolist()}"
        )
    return error, transfer_seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resident-gib", type=float, default=0)
    parser.add_argument("--block-mib", type=int, default=512)
    parser.add_argument(
        "--operation", choices=("p2p", "all-sum", "all-gather"), default="p2p"
    )
    parser.add_argument("--length", type=int, default=64)
    args = parser.parse_args()
    if args.resident_gib < 0 or args.block_mib < 1 or args.length < 1:
        raise ValueError("resident size, block size, and length must be valid")

    group, control = init_distributed_groups("jaccl")
    if (group.size(), control.size()) != (2, 2):
        raise RuntimeError("resident stress requires exactly two JACCL ranks")
    prime_distributed_groups(group, control)
    rank = group.rank()

    old_limit = mx.set_wired_limit(
        mx.device_info()["max_recommended_working_set_size"]
    )
    try:
        resident = allocate_resident(args.resident_gib, args.block_mib, rank)
        error, seconds = transfer(group, control, args.operation, args.length)
        # Keep buffers live through the final collective and integrity check.
        if resident:
            first = resident[0][0].item()
            last = resident[-1][-1].item()
            if first != rank + 1 or last != rank + 1:
                raise AssertionError("resident buffer changed during JACCL transfer")
        # macOS reports ru_maxrss in bytes (Linux reports KiB).
        rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GIB
        print(
            f"[rank {rank}] resident JACCL {args.operation}: error={error:.3g} "
            f"seconds={seconds:.6f} rss={rss_gib:.1f} GiB",
            flush=True,
        )
    finally:
        mx.synchronize()
        mx.set_wired_limit(old_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
