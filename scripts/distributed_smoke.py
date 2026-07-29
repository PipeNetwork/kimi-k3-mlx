#!/usr/bin/env python3
"""Compare a two-rank pipeline forward/decode with the unsharded reference."""

import argparse
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model

try:
    from scripts.distributed_groups import (
        init_distributed_groups,
        prime_distributed_groups,
    )
except ModuleNotFoundError:  # Direct ``python scripts/distributed_smoke.py``.
    from distributed_groups import init_distributed_groups, prime_distributed_groups


def load_distributed(stage: Path, group, control_group=None):
    model, _ = load_model(stage, lazy=True, strict=True)
    model.model.pipeline(group, control_group or group)
    mx.eval(model.parameters())
    return model


def run_steps(model, prefill_length: int, decode_steps: int):
    cache = model.make_cache()
    prefill_tokens = (mx.arange(prefill_length, dtype=mx.int32) % 127 + 1)[None]
    prefill = model(prefill_tokens, cache=cache)
    mx.eval(prefill)
    outputs = [prefill]
    for step in range(decode_steps):
        token = mx.array([[(prefill_length + step) % 127 + 1]], dtype=mx.int32)
        output = model(token, cache=cache)
        mx.eval(output)
        outputs.append(output)
    return outputs


def run_boundary_probe(model, group, length: int, hidden_size: int) -> float:
    """Exercise the production-size acknowledged point-to-point boundary."""
    rank = group.rank()
    # Kimi-K3 rank 1 sends h plus four AttnRes blocks at the 47-layer split.
    shape = (5, 1, length, hidden_size)
    if rank == 1:
        packet = mx.full(shape, 1.25, dtype=mx.bfloat16)
        transferred = model.model._pipeline_send(packet, 0)
    else:
        template = mx.zeros(shape, dtype=mx.bfloat16)
        transferred = model.model._pipeline_recv(template, 1)
    error = mx.max(mx.abs(transferred.astype(mx.float32) - 1.25)).item()
    print(
        f"[rank {rank}] boundary probe: shape={shape}, error={error:.3g}",
        flush=True,
    )
    return error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("work/tiny-uvmax-pipeline-2"))
    parser.add_argument(
        "--backend",
        choices=("ring", "jaccl"),
        default="ring",
        help="Distributed backend to initialize (default: ring for local CI).",
    )
    parser.add_argument("--prefill-length", type=int, default=3)
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument(
        "--production-boundary-probe",
        action="store_true",
        help="Transfer a 5x1xLx7168 BF16 packet before model parity.",
    )
    args = parser.parse_args()
    if args.prefill_length < 1 or args.decode_steps < 1:
        raise ValueError("prefill length and decode steps must be positive")
    group, control_group = init_distributed_groups(args.backend)
    if group.size() != 2:
        raise RuntimeError(f"expected two ranks, got {group.size()}")
    prime_distributed_groups(group, control_group)
    rank = group.rank()
    stage = args.root.resolve() / f"rank{rank}"

    distributed = load_distributed(stage, group, control_group)
    if args.production_boundary_probe:
        boundary_error = run_boundary_probe(distributed, group, args.prefill_length, 7168)
        if boundary_error != 0:
            raise AssertionError(f"production boundary mismatch: {boundary_error}")
    actual = run_steps(distributed, args.prefill_length, args.decode_steps)

    reference, _ = load_model(stage, lazy=False, strict=True)
    expected = run_steps(reference, args.prefill_length, args.decode_steps)
    errors = [
        mx.max(mx.abs(actual_output - expected_output)).item()
        for actual_output, expected_output in zip(actual, expected)
    ]
    prefill_error = errors[0]
    decode_error = max(errors[1:])
    if prefill_error > 2e-4 or decode_error > 2e-4:
        raise AssertionError(
            f"pipeline mismatch: prefill={prefill_error}, decode={decode_error}"
        )
    print(
        f"[rank {rank}] pipeline parity: prefill={prefill_error:.3g}, "
        f"decode={decode_error:.3g} ({args.decode_steps} steps)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
