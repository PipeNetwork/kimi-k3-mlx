#!/usr/bin/env python3
"""Pipelined K3 must equal single-rank K3, token for token.

Run under mlx.launch with >=2 ranks:

  mlx.launch --backend ring --hostfile hosts.json --cwd <dir> -- \
      <venv>/bin/python tests/test_pipeline_parity.py

[K3-2] The load-bearing part is the AttnRes stack. Every sub-layer mixes over a
growing stack of block residuals, so a pipeline boundary has to carry the stack
as well as the hidden state. Sending only the hidden state still produces
fluent-looking output -- the stack silently restarts empty on each rank -- which
is exactly the failure this asserts against.
"""
import os
import sys

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from test_kimi_k3 import kimi_k3, tiny_args  # noqa: E402

TOKENS = mx.array([[3, 14, 15, 92, 65, 35, 89, 79]])

# bf16 by default. A float32 model is float32 end to end and cannot catch a
# boundary that sizes its recv template from the local h.dtype. The real model
# holds bf16 weights, so h enters the stack as bf16 but comes back from the
# KDA/MLA layers as float32 (the delta-rule state is float32). A receiver
# templating on bf16 then reads half the bytes and gets NaN -- not an error.
BF16 = os.environ.get("K3_PARITY_BF16", "1") != "0"


def build(seed):
    mx.random.seed(seed)
    args = tiny_args()
    model = kimi_k3.Model(args)
    if BF16:
        model.apply(lambda p: p.astype(mx.bfloat16) if p.dtype == mx.float32 else p)
    model.eval()
    mx.eval(model.parameters())
    return model


def main():
    group = mx.distributed.init()
    rank, size = group.rank(), group.size()

    # identical seed on every rank -> identical weights, so the pipelined run is
    # comparable to the single-rank one tensor for tensor
    ref = build(0)
    ref_out = ref(TOKENS)
    mx.eval(ref_out)

    pipe = build(0)
    pipe.model.pipeline(group)
    n_local = len(pipe.layers)
    print(f"rank {rank}/{size}: layers[{pipe.model.start_idx}:{pipe.model.end_idx}] "
          f"= {n_local}, stack_in={pipe.model._blocks_before_slice}, "
          f"bf16={BF16}", flush=True)

    pipe_out = pipe(TOKENS, cache=pipe.make_cache())
    mx.eval(pipe_out)

    if rank == 0:
        a = ref_out.astype(mx.float32)
        b = pipe_out.astype(mx.float32)
        maxdiff = float(mx.abs(a - b).max())
        fa, fb = a.flatten(), b.flatten()
        cos = float(mx.sum(fa * fb) /
                    (mx.sqrt(mx.sum(fa * fa)) * mx.sqrt(mx.sum(fb * fb)) + 1e-12))
        finite = bool(mx.isfinite(b).all())
        # boundaries round-trip through float32, so a bf16 model rounds
        # differently than the single-rank path; cosine is the structural check
        # and max-abs just bounds the rounding
        tol = 5e-2 if BF16 else 1e-3
        ok = finite and cos > 0.9999 and maxdiff < tol
        print(f"\nsingle-rank vs {size}-rank pipeline: max abs diff {maxdiff:.3e} "
              f"(tol {tol:.0e}), cosine {cos:.7f}, finite={finite}")
        print(f"  ref  {ref_out.shape} mean|y| {float(mx.abs(a).mean()):.6f}")
        print(f"  pipe {pipe_out.shape} mean|y| {float(mx.abs(b).mean()):.6f}")
        print("PIPELINE PARITY " + ("OK" if ok else "FAIL"))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
