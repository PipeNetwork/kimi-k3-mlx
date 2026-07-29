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
    args = parser.parse_args()
    if args.prefill_length < 1 or args.decode_steps < 1:
        raise ValueError("prefill length and decode steps must be positive")
    group = mx.distributed.init(strict=True, backend=args.backend)
    if group.size() != 2:
        raise RuntimeError(f"expected two ranks, got {group.size()}")
    rank = group.rank()
    stage = args.root.resolve() / f"rank{rank}"

    distributed = load_distributed(stage, group)
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
