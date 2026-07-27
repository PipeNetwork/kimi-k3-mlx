#!/usr/bin/env python3
"""Turn REAP saliency into a prune plan for Kimi-K3.

Modes
-----
uniform   keep the same number of experts in every MoE layer (what Cerebras'
          REAP does, and what the published REAP models use)
global    rank every (layer, expert) pair together and keep the top N overall,
          so layers that carry more saliency mass keep more experts
graded    three tiers instead of keep/drop -- most-salient experts at mxfp4
          (free for K3, since that is the source encoding), mid experts at a
          lower bit width, tail dropped. Fits ~30% more experts in the same
          memory than a binary cut.

          NOT BUILDABLE TODAY. MLX's QuantizedSwitchLinear holds one weight
          tensor per layer with scalar bits/group_size/mode, and gather_qmm
          takes them as scalars, so every expert in a layer shares a width.
          Realising graded needs a two-bank SwitchGLU -- either 2x expert
          compute (run both banks, select) or a contiguous partition over the
          already-sorted routing indices. scripts/convert.py rejects graded
          plans rather than silently mis-building them; --mode graded is kept
          because its sizing output is what justifies that work.

`global` and `graded` both need a floor: a layer stripped below top_k experts
cannot route at all, and one near top_k routes degenerately. `--min-experts`
defaults to 4*top_k.

Usage:
  scripts/reap_plan.py --saliency out/reap_saliency.npz --keep 0.27
  scripts/reap_plan.py --saliency out/reap_saliency.npz --mode graded \
      --hi 0.15 --lo 0.20 --out out/reap_plan.json
"""

import argparse
import json
import os

import numpy as np

EXPERT_PARAMS = 3 * 3584 * 3072   # w1 + w2 + w3 for one K3 expert
NONEXPERT_PARAMS = 57.23e9
BPW = {"mxfp4": 4.25, "8bit": 8.5, "6bit": 6.5, "4bit": 4.5, "3bit": 3.5, "2bit": 2.5}


def load(path):
    z = np.load(path)
    n_tokens = float(z["n_tokens"])
    # REAP normalizes by the calibration set size, not by per-expert hit count:
    # an expert that fires rarely *should* score low. Dividing by its own count
    # would measure average magnitude when used and hide the frequency signal.
    sal = z["saliency"] / max(n_tokens, 1.0)
    return sal, z["counts"], z, n_tokens


def size_gb(keep_per_layer, bpw_per_layer, nonexpert_bpw=8.5):
    exp_bits = sum(n * EXPERT_PARAMS * b for n, b in zip(keep_per_layer, bpw_per_layer))
    return (exp_bits / 8 + NONEXPERT_PARAMS * nonexpert_bpw / 8) / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--saliency", required=True)
    p.add_argument("--mode", choices=["uniform", "global", "graded"], default="uniform")
    p.add_argument("--keep", type=float, default=0.27, help="fraction of experts kept")
    p.add_argument("--hi", type=float, default=0.15, help="graded: mxfp4 tier fraction")
    p.add_argument("--lo", type=float, default=0.20, help="graded: low-bit tier fraction")
    p.add_argument("--lo-bits", default="2bit", choices=list(BPW))
    p.add_argument("--min-experts", type=int, default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    sal, counts, z, n_tokens = load(a.saliency)
    NE = int(z["num_experts"])
    TOPK = int(z["top_k"])
    moe_layers = [int(i) for i in z["moe_layers"]]
    floor = a.min_experts if a.min_experts is not None else 4 * TOPK

    print(f"saliency: {len(moe_layers)} MoE layers x {NE} experts, "
          f"{n_tokens:,.0f} calibration tokens, top-{TOPK}")
    dead = int((counts[moe_layers] == 0).sum())
    print(f"never-routed experts: {dead} / {len(moe_layers)*NE} "
          f"({dead/(len(moe_layers)*NE):.2%}) -- free to drop")

    plan = {"mode": a.mode, "top_k": TOPK, "num_experts": NE,
            "min_experts": floor, "layers": {}}

    if a.mode in ("uniform", "global"):
        n_keep = max(floor, int(round(NE * a.keep)))
        if a.mode == "uniform":
            per_layer = {L: n_keep for L in moe_layers}
        else:
            flat = [(sal[L, e], L, e) for L in moe_layers for e in range(NE)]
            flat.sort(reverse=True)
            budget = n_keep * len(moe_layers)
            per_layer = {L: 0 for L in moe_layers}
            # seed the floor so no layer can be starved below routable width
            for L in moe_layers:
                for e in np.argsort(-sal[L])[:floor]:
                    per_layer[L] += 1
            taken = sum(per_layer.values())
            ranks = {L: list(np.argsort(-sal[L])) for L in moe_layers}
            for s, L, e in flat:
                if taken >= budget:
                    break
                if ranks[L].index(e) < per_layer[L]:
                    continue
                per_layer[L] += 1
                taken += 1

        for L in moe_layers:
            order = np.argsort(-sal[L])
            keep = sorted(int(x) for x in order[: per_layer[L]])
            plan["layers"][str(L)] = {"keep": keep, "bits": {"all": "mxfp4"}}

        kept = [per_layer[L] for L in moe_layers]
        print(f"\n{a.mode}: keep {min(kept)}-{max(kept)} experts/layer "
              f"(mean {np.mean(kept):.0f}), density top-{TOPK}/{int(np.mean(kept))} "
              f"= {TOPK/np.mean(kept):.1%}")
        print(f"size @ mxfp4: {size_gb(kept, [BPW['mxfp4']]*len(kept)):.0f} GB")

    else:  # graded
        n_hi = max(1, int(round(NE * a.hi)))
        n_lo = max(0, int(round(NE * a.lo)))
        if n_hi + n_lo < floor:
            raise SystemExit(f"hi+lo = {n_hi+n_lo} experts is below the "
                             f"routable floor of {floor}")
        for L in moe_layers:
            order = [int(x) for x in np.argsort(-sal[L])]
            hi, lo = order[:n_hi], order[n_hi : n_hi + n_lo]
            plan["layers"][str(L)] = {
                "keep": sorted(hi + lo),
                "bits": {**{str(e): "mxfp4" for e in hi},
                         **{str(e): a.lo_bits for e in lo}},
            }
        nl = len(moe_layers)
        gb = (n_hi * EXPERT_PARAMS * BPW["mxfp4"] + n_lo * EXPERT_PARAMS * BPW[a.lo_bits]) \
             * nl / 8 / 1e9 + NONEXPERT_PARAMS * 8.5 / 8 / 1e9
        print(f"\ngraded: {n_hi} experts @ mxfp4 + {n_lo} @ {a.lo_bits} "
              f"= {n_hi+n_lo} kept/layer ({(n_hi+n_lo)/NE:.1%}), "
              f"{1-(n_hi+n_lo)/NE:.0%} dropped")
        print(f"density top-{TOPK}/{n_hi+n_lo} = {TOPK/(n_hi+n_lo):.1%}")
        print(f"size: {gb:.0f} GB")
        print("NOTE: graded plans are sizing-only -- scripts/convert.py cannot "
              "build them\n      (MLX pins one bit width per layer's expert "
              "tensor). See the module docstring.")
        kept = [n_hi + n_lo] * nl

    # how much saliency mass survives -- the honest quality proxy available
    # without running the pruned model
    total = sum(sal[L].sum() for L in moe_layers)
    surv = sum(sal[L][plan["layers"][str(L)]["keep"]].sum() for L in moe_layers)
    print(f"saliency mass retained: {surv/total:.2%}")

    params = np.mean(kept) * len(moe_layers) * EXPERT_PARAMS + NONEXPERT_PARAMS
    print(f"pruned model: {params/1e9:.0f} B params "
          f"(from {NE*len(moe_layers)*EXPERT_PARAMS/1e9 + NONEXPERT_PARAMS/1e9:.0f} B)")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        json.dump(plan, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
