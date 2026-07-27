#!/usr/bin/env python3
"""Summarise REAP saliency: how prunable is this model, and where?

reap_plan.py answers "given a target, which experts go". This answers the prior
question -- how concentrated the saliency actually is, which is what decides
whether an aggressive prune ratio is plausible at all.

Key readings:
  concentration   fraction of a layer's saliency mass held by its top-N experts.
                  Concentrated layers prune cheaply; flat layers do not.
  near-dead       experts that are essentially never routed, or routed with
                  negligible gate*norm. These are free.
  layer spread    if layers differ a lot, `global` planning beats `uniform`.
  retention curve saliency mass kept vs experts kept, the honest proxy for
                  quality available without running the pruned model.

Usage: scripts/reap_report.py --saliency out/reap_saliency.npz
"""

import argparse

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--saliency", required=True)
    p.add_argument("--budget-gb", type=float, default=460.0)
    a = p.parse_args()

    z = np.load(a.saliency)
    sal_raw, counts = z["saliency"], z["counts"]
    n_tokens = float(z["n_tokens"])
    NE, TOPK = int(z["num_experts"]), int(z["top_k"])
    moe = [int(i) for i in z["moe_layers"]]
    sal = sal_raw[moe] / n_tokens          # (L, NE), REAP normalisation
    cnt = counts[moe]
    L = len(moe)

    print(f"REAP saliency — {L} MoE layers x {NE} experts, top-{TOPK}, "
          f"{n_tokens:,.0f} calibration tokens\n")

    # ---- routing coverage
    exp_tokens = cnt.sum(1)
    print("routing")
    print(f"  (token, slot) pairs per layer : {exp_tokens[0]:,.0f} "
          f"(= {n_tokens:,.0f} x {TOPK})")
    print(f"  mean pairs per expert         : {cnt.mean():,.0f}")
    never = (cnt == 0).sum()
    print(f"  never routed                  : {never} / {L*NE} ({never/(L*NE):.3%})")
    rare = (cnt < cnt.mean() * 0.1).sum()
    print(f"  routed <10% of average        : {rare} / {L*NE} ({rare/(L*NE):.2%})")
    load = cnt / cnt.mean(1, keepdims=True)
    print(f"  load imbalance (max/mean)     : {load.max():.1f}x   "
          f"(min {load.min():.3f}x)")

    # ---- concentration
    print("\nconcentration (share of layer saliency held by top-N experts)")
    srt = -np.sort(-sal, axis=1)
    tot = srt.sum(1, keepdims=True)
    cum = np.cumsum(srt, axis=1) / np.maximum(tot, 1e-30)
    for frac in (0.10, 0.25, 0.375, 0.50, 0.75):
        n = max(1, int(round(NE * frac)))
        col = cum[:, n - 1]
        print(f"  top {frac:>5.1%} ({n:>3} experts): "
              f"mean {col.mean():>6.2%}  min {col.min():>6.2%}  max {col.max():>6.2%}")

    # a flat layer would give top-x% ~= x%; report the excess
    flat_gap = cum[:, int(NE * 0.25) - 1].mean() - 0.25
    print(f"  -> top-25% holds {flat_gap:+.1%} more mass than a uniform layer would")

    # ---- per-layer spread
    #
    # CAREFUL: raw layer mass is NOT an importance signal. Saliency is
    # gate * ||expert_output||, so it scales with each layer's activation
    # magnitude, which on K3 grows with depth. Worse, K3's LatentMoE applies
    # RMSNorm to the *combined* expert output before up_proj, so that magnitude
    # is renormalised away downstream -- a layer with 10x larger expert outputs
    # is not 10x more important. Ranking (layer, expert) pairs across layers
    # (`global` mode) would chase this artifact and starve early layers.
    # Judge cross-layer differences on NORMALISED concentration instead.
    print("\nper-layer variation (does `global` planning help?)")
    layer_mass = sal.sum(1)
    depth_r = np.corrcoef(np.arange(L), layer_mass)[0, 1]
    print(f"  raw layer mass spread : {layer_mass.max()/max(layer_mass.min(),1e-30):.0f}x "
          f"(corr with depth {depth_r:+.2f}) <- activation scale, NOT importance")
    c25 = cum[:, int(NE * 0.25) - 1]
    print(f"  normalised top-25% retention: min {c25.min():.2%}  max {c25.max():.2%}  "
          f"(spread {c25.max()-c25.min():.1%})")
    if c25.max() - c25.min() < 0.30:
        print("  -> layers are similarly prunable once scale is removed; prefer `uniform`.")
    else:
        print("  -> genuinely different concentration by layer; `global` may help.")
    hardest = [moe[i] for i in np.argsort(c25)[:5]]
    easiest = [moe[i] for i in np.argsort(-c25)[:5]]
    print(f"  least prunable layers: {hardest}")
    print(f"  most prunable layers : {easiest}")

    # ---- retention curve + sizing
    EXPERT_PARAMS = 3 * 3584 * 3072
    NONEXP = 57.23e9
    print("\nretention curve (uniform keep, mxfp4 experts + 8-bit rest)")
    print(f"  {'keep':>6} {'experts':>8} {'saliency kept':>14} {'size':>8}")
    for frac in (0.75, 0.50, 0.375, 0.27, 0.25, 0.1875, 0.125):
        n = max(1, int(round(NE * frac)))
        kept = cum[:, n - 1].mean()
        gb = (L * n * EXPERT_PARAMS * 4.25 / 8 + NONEXP * 8.5 / 8) / 1e9
        flag = " *" if gb <= a.budget_gb else ""
        print(f"  {frac:>5.0%} {n:>8} {kept:>13.2%} {gb:>7.0f}G{flag}")
    print(f"  * fits {a.budget_gb:.0f} GB")


if __name__ == "__main__":
    main()
