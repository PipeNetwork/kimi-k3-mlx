#!/usr/bin/env python3
"""Compare a two-rank pipeline forward/decode with the unsharded reference."""

import argparse
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model


def load_distributed(stage: Path, group):
    model, _ = load_model(stage, lazy=True, strict=True)
    model.model.pipeline(group)
    mx.eval(model.parameters())
    return model


def run_two_steps(model):
    cache = model.make_cache()
    prefill = model(mx.array([[1, 2, 3]], dtype=mx.int32), cache=cache)
    mx.eval(prefill)
    decode = model(mx.array([[4]], dtype=mx.int32), cache=cache)
    mx.eval(decode)
    return prefill, decode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("work/tiny-uvmax-pipeline-2"))
    parser.add_argument(
        "--backend",
        choices=("ring", "jaccl"),
        default="ring",
        help="Distributed backend to initialize (default: ring for local CI).",
    )
    args = parser.parse_args()
    group = mx.distributed.init(strict=True, backend=args.backend)
    if group.size() != 2:
        raise RuntimeError(f"expected two ranks, got {group.size()}")
    rank = group.rank()
    stage = args.root.resolve() / f"rank{rank}"

    distributed = load_distributed(stage, group)
    actual_prefill, actual_decode = run_two_steps(distributed)

    reference, _ = load_model(stage, lazy=False, strict=True)
    expected_prefill, expected_decode = run_two_steps(reference)
    prefill_error = mx.max(mx.abs(actual_prefill - expected_prefill)).item()
    decode_error = mx.max(mx.abs(actual_decode - expected_decode)).item()
    if prefill_error > 2e-4 or decode_error > 2e-4:
        raise AssertionError(
            f"pipeline mismatch: prefill={prefill_error}, decode={decode_error}"
        )
    print(
        f"[rank {rank}] pipeline parity: prefill={prefill_error:.3g}, "
        f"decode={decode_error:.3g}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
