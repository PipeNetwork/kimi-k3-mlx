#!/usr/bin/env python3
"""Derive a targeted saliency file from a tagged per-language calibration.

A tagged run already recorded gate*||output|| separately per corpus, so a
domain-targeted build needs no second calibration pass: sum the buckets you care
about. That is strictly better than re-calibrating on a filtered corpus -- same
tokens, same layer states, same conditions, so any difference between the
resulting builds is attributable to the target distribution and nothing else.

Usage:
  scripts/reap_subset.py --saliency out/reap_saliency_perlang.npz \\
      --keep-sources code-multi,code-python,web-en --out out/reap_saliency_encode.npz
"""

import argparse

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--saliency", required=True)
    p.add_argument("--keep-sources", required=True,
                   help="comma-separated source labels to sum")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    z = np.load(a.saliency)
    labels = [str(x) for x in z["source_labels"]]
    want = [s.strip() for s in a.keep_sources.split(",")]
    missing = [w for w in want if w not in labels]
    if missing:
        raise SystemExit(f"unknown source(s) {missing}; available: {labels}")
    idx = [labels.index(w) for w in want]

    per, per_cnt = z["per_source"], z["per_source_counts"]
    toks = z["source_tokens"]
    sal = per[idx].sum(0)
    cnt = per_cnt[idx].sum(0)
    n_tok = int(toks[idx].sum())

    kept_frac = n_tok / int(z["n_tokens"])
    print(f"summing {len(idx)} sources: {want}")
    for i, w in zip(idx, want):
        print(f"  {w:<14} {int(toks[i]):>8,} tokens")
    print(f"  total          {n_tok:>8,} tokens ({kept_frac:.1%} of the full corpus)")

    # How different is the resulting ranking from the full-corpus one?
    moe = [int(i) for i in z["moe_layers"]]
    NE = int(z["num_experts"])
    N = 242
    a_top = np.argsort(-sal[moe], axis=-1)[:, :N]
    b_top = np.argsort(-z["saliency"][moe], axis=-1)[:, :N]
    ov = np.mean([len(set(a_top[l]) & set(b_top[l])) / N for l in range(len(moe))])
    print(f"\ntop-{N} overlap with the full-corpus ranking: {ov:.1%}")
    print(f"  -> {int(N*(1-ov))} of {N} experts per layer differ from the mixed build")

    np.savez(
        a.out,
        saliency=sal, counts=cnt,
        n_tokens=np.array(n_tok),
        moe_layers=z["moe_layers"],
        num_experts=z["num_experts"],
        top_k=z["top_k"],
        num_hidden_layers=z["num_hidden_layers"],
        source_labels=np.array(want),
        derived_from=np.array([a.saliency]),
    )
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
