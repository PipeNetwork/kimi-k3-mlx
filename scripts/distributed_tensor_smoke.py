#!/usr/bin/env python3
"""Compare two-rank tensor parallel forward/decode with an unsharded model."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model

try:
    from scripts.tensor_stage import load_tensor_stage
except ModuleNotFoundError:
    from tensor_stage import load_tensor_stage


def run_steps(model, prefill_length: int, decode_steps: int):
    cache = model.make_cache()
    tokens = (mx.arange(prefill_length, dtype=mx.int32) % 127 + 1)[None]
    outputs = [model(tokens, cache=cache)]
    mx.eval(outputs[-1])
    for step in range(decode_steps):
        token = mx.array([[(prefill_length + step) % 127 + 1]], dtype=mx.int32)
        outputs.append(model(token, cache=cache))
        mx.eval(outputs[-1])
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("work/tiny-tp"))
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("work/tiny-uvmax-pipeline-2/source"),
    )
    parser.add_argument("--backend", choices=("ring", "jaccl"), default="ring")
    parser.add_argument("--prefill-length", type=int, default=64)
    parser.add_argument("--decode-steps", type=int, default=32)
    args = parser.parse_args()

    group = mx.distributed.init(strict=True, backend=args.backend)
    if group.size() != 2:
        raise RuntimeError(f"expected two ranks, got {group.size()}")
    mx.eval(mx.distributed.all_sum(mx.ones((10,)), group=group, stream=mx.cpu))
    rank = group.rank()
    actual_model, _ = load_tensor_stage(
        args.root / f"rank{rank}", group, tokenizer=False
    )
    actual = run_steps(actual_model, args.prefill_length, args.decode_steps)

    reference_model, _ = load_model(args.reference, lazy=False, strict=True)
    expected = run_steps(reference_model, args.prefill_length, args.decode_steps)
    errors = [
        mx.max(mx.abs(got - want)).item() for got, want in zip(actual, expected)
    ]
    prefill_error, decode_error = errors[0], max(errors[1:])
    if prefill_error > 2e-4 or decode_error > 2e-4:
        raise AssertionError(
            f"tensor parity mismatch: prefill={prefill_error}, decode={decode_error}"
        )
    print(
        f"[rank {rank}] tensor parity: prefill={prefill_error:.3g}, "
        f"decode={decode_error:.3g} ({args.decode_steps} steps)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
