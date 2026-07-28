#!/usr/bin/env python3
"""Wire model memory before timing anything, or the number will be wrong.

MLX's default wired limit leaves a multi-hundred-GB model unwired, so every
decode step faults its weights back in from SSD instead of reading them from
RAM. Measured on this repo's own published artifact:

    Kimi-K3-REAP80-MLX-mxfp4-q8, 350 GB, M3 Ultra / 512 GiB
      default wired limit   0.20 tok/s     17 GB/s effective
      wired to 480 GB       5.42 tok/s    472 GB/s effective  (58% of 819 peak)

Every tok/s figure published from this repo before 2026-07-28 was measured the
first way and understates the model by up to 27x. An independent user reported
5.608 tok/s on that same repo, which is what surfaced it.

The reason it hid so long: PREFILL is compute-bound and touches each page once,
so it looked sane (~73 tok/s) while decode was 50x below the bandwidth ceiling.
Only decode re-reads every weight per token, so only decode is destroyed by an
unwired model.

IMPORTANT -- this is a harness bug, not a user-facing one. mlx-lm already does
this: stream_generate() wraps its loop in a `wired_limit` context manager, and
generate(), the mlx_lm.generate CLI, BatchGenerator and server.py all inherit
it. Anyone using the normal APIs gets the fast path for free. What does NOT get
it is a hand-rolled decode loop over raw generate_step() or model(...) -- which
is exactly what scripts/smoke.py does, and why this repo's own published numbers
were the only ones that were wrong. Any such loop must call wire() before
loading, and the limit used here is the same one mlx-lm picks so the numbers are
directly comparable.
"""

import mlx.core as mx


def recommended_limit() -> int:
    """The value mlx-lm itself wires to -- Metal's own working-set recommendation."""
    return mx.device_info()["max_recommended_working_set_size"]


def wire(limit: int = None, verbose: bool = True) -> int:
    """Wire model memory. -> bytes actually requested, or 0 if MLX refused.

    Returns 0 and warns rather than raising: a slow measurement beats a crashed
    one, but it must be visible in the log, because a silently unwired run is
    exactly the failure this module exists to prevent.
    """
    if not mx.metal.is_available():
        return 0
    limit = recommended_limit() if limit is None else limit
    try:
        mx.set_wired_limit(limit)
    except Exception as e:
        print(f"[mem] WARNING: set_wired_limit({limit/1e9:.0f} GB) failed "
              f"({type(e).__name__}: {e}); tok/s from this run is NOT comparable "
              f"and is likely many times too low", flush=True)
        return 0
    if verbose:
        print(f"[mem] wired limit {limit/1e9:.0f} GB (Metal recommendation, "
              f"same value mlx-lm uses)", flush=True)
    return limit
