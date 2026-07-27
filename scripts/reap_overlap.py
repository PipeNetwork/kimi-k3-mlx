#!/usr/bin/env python3
"""Do different languages use different experts in Kimi-K3?

Consumes the per-source saliency from a tagged calibration run and reports, for
every pair of corpora, how much their top-N expert sets overlap.

Why this matters: if English-only and Chinese-only calibration select the *same*
experts, there is no separable language structure and a targeted build buys
nothing. If they select different experts, then at a fixed size a build
calibrated on your actual distribution is meaningfully better than a generic
one -- not smaller, better.

Baselines are the point of comparison. Two random top-N picks out of NE experts
overlap by ~N/NE just by chance (27% at N=242, NE=896). Overlap only counts as
"shared structure" well above that, and as "separable" well below it.

Usage: scripts/reap_overlap.py --saliency out/reap_saliency_perlang.npz --keep 242
"""

import argparse
import itertools

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--saliency", required=True)
    p.add_argument("--keep", type=int, default=242)
    p.add_argument("--group", action="store_true",
                   help="also report code / english / cjk / other groupings")
    a = p.parse_args()

    z = np.load(a.saliency)
    labels = [str(x) for x in z["source_labels"]]
    per = z["per_source"]                       # (src, layer, expert)
    moe = [int(i) for i in z["moe_layers"]]
    toks = z["source_tokens"] if "source_tokens" in z.files else None
    NE = int(z["num_experts"])
    N = a.keep

    if len(labels) == 1:
        raise SystemExit("saliency has a single bucket -- rerun calibration with "
                         "a corpus that has a .sources.json manifest")

    per = per[:, moe, :]                        # drop dense layers
    S, L, _ = per.shape
    print(f"per-language saliency: {S} sources x {L} MoE layers x {NE} experts")
    if toks is not None:
        for i, lab in enumerate(labels):
            print(f"  {lab:<14} {int(toks[i]):>9,} tokens")
    print(f"\nchance overlap for top-{N} of {NE}: {N/NE:.1%}\n")

    # Normalise each (source, layer) to unit mass: raw magnitude differs by layer
    # depth and by how many tokens a source contributed, neither of which is an
    # importance signal.
    mass = per.sum(-1, keepdims=True)
    norm = np.divide(per, mass, out=np.zeros_like(per), where=mass > 0)

    # top-N expert set per (source, layer)
    top = np.argsort(-norm, axis=-1)[:, :, :N]
    sets = [[set(top[s, l].tolist()) for l in range(L)] for s in range(S)]

    print(f"pairwise top-{N} overlap (mean over layers)")
    hdr = "                " + " ".join(f"{l[:7]:>8}" for l in labels)
    print(hdr)
    M = np.zeros((S, S))
    for i in range(S):
        row = []
        for j in range(S):
            ov = np.mean([len(sets[i][l] & sets[j][l]) / N for l in range(L)])
            M[i, j] = ov
            row.append(f"{ov:>7.1%} ")
        print(f"  {labels[i]:<14}" + "".join(row))

    off = M[~np.eye(S, dtype=bool)]
    print(f"\n  off-diagonal: min {off.min():.1%}  mean {off.mean():.1%}  max {off.max():.1%}")
    ratio = off.mean() / (N / NE)
    print(f"  vs chance ({N/NE:.1%}): {ratio:.2f}x")
    if ratio > 1.6:
        print("  -> languages largely SHARE experts; a targeted build buys little.")
    elif ratio < 1.15:
        print("  -> languages select largely DIFFERENT experts; targeted builds "
              "should be meaningfully better at the same size.")
    else:
        print("  -> partial separation; targeted builds buy something, not a lot.")

    # what a targeted build gains: saliency retained on your own distribution
    # The honest baseline is the ACTUAL blended corpus that was calibrated on --
    # token-weighted, not an equal average of the 11 sources (which would
    # over-weight the 1k-token languages and flatter targeted builds).
    total = z["saliency"][moe]                  # token-weighted sum over all sources
    tmass = total.sum(-1, keepdims=True)
    tnorm = np.divide(total, tmass, out=np.zeros_like(total), where=tmass > 0)
    mixed_top = np.argsort(-tnorm, axis=-1)[:, :N]

    print(f"\nsaliency retained at top-{N}: own-distribution vs the ACTUAL mixed build")
    print(f"  {'source':<14} {'own':>7} {'mixed':>7} {'gain':>9}")
    for si, lab in enumerate(labels):
        own = np.mean([norm[si, l, top[si, l]].sum() for l in range(L)])
        mix = np.mean([norm[si, l, mixed_top[l]].sum() for l in range(L)])
        print(f"  {lab:<14} {own:>6.1%} {mix:>7.1%} {100*(own-mix):>+7.1f} pts")


if __name__ == "__main__":
    main()
