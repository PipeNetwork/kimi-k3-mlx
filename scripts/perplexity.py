#!/usr/bin/env python3
"""Held-out perplexity for a converted Kimi-K3 tier, bucketed by source corpus.

Three prompts of greedy decode cannot separate two quantizations that differ by
a fraction of a percent in weight space: any perturbation eventually flips an
argmax, and once it does the sequences diverge completely, so "the output reads
better" is indistinguishable from noise. Perplexity over held-out text is the
measurement that can actually resolve it.

HELD-OUT MEANS HELD OUT. reap_calibrate.py consumes the FIRST seqs x seqlen
tokens of the corpus, so any tier built from a plan calibrated on this file has
already seen that prefix -- scoring it would flatter every build that was fitted
to it. --skip-tokens starts the evaluation past that prefix. It is required
rather than defaulted, because the correct value depends on which calibration
produced the plan and guessing it wrong invalidates the number silently.

Per-token NLL is written to the npz. Both arms of an A/B see byte-identical
token sequences, so the paired difference has far lower variance than either
perplexity alone -- that is what makes a sub-1% effect measurable at this size.

Usage:
  scripts/perplexity.py --path out/Kimi-K3-REAP73-MLX-2bit --src Kimi-K3-src \\
      --calib-text out/calib.txt --skip-tokens 200000 --seqs 32 --seqlen 2048 \\
      --out out/ppl_2bit.npz
"""

import argparse
import json
import os
import pathlib
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def load_heldout(a, enc):
    """-> (seqs, seqlen) ids and matching source ids, starting past --skip-tokens.

    Tokenizes document by document against the .sources.json manifest, exactly
    as reap_calibrate.load_calibration does, so the source labels here mean the
    same thing they do in the saliency tables.
    """
    text = open(a.calib_text, errors="ignore").read()
    need = a.seqs * a.seqlen
    side = a.calib_text + ".sources.json"

    labels, ids, src_ids = [], [], []
    if os.path.exists(side):
        man = json.load(open(side))
        labels = sorted(set(man["order"]))
        lut = {l: i for i, l in enumerate(labels)}
        pos = 0
        for lab, n in zip(man["order"], man["chars"]):
            if len(ids) >= a.skip_tokens + need:
                break
            doc = enc.encode_ordinary(text[pos:pos + n])
            pos += n
            ids.extend(doc)
            src_ids.extend([lut[lab]] * len(doc))
    else:
        ids = enc.encode_ordinary(text)
        src_ids = [0] * len(ids)
        labels = ["all"]

    if len(ids) < a.skip_tokens + need:
        raise SystemExit(
            f"corpus has {len(ids):,} tokens; need {a.skip_tokens:,} skipped + "
            f"{need:,} evaluated = {a.skip_tokens + need:,}."
        )
    ids = np.array(ids[a.skip_tokens: a.skip_tokens + need], np.int32)
    sids = np.array(src_ids[a.skip_tokens: a.skip_tokens + need], np.int32)
    print(f"[ppl] held-out {need:,} tokens starting at offset {a.skip_tokens:,} "
          f"({len(labels)} sources)", flush=True)
    return ids.reshape(a.seqs, a.seqlen), sids.reshape(a.seqs, a.seqlen), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--src", default="Kimi-K3-src", help="for tiktoken.model")
    ap.add_argument("--calib-text", required=True)
    ap.add_argument("--skip-tokens", type=int, required=True,
                    help="tokens the calibration already consumed; evaluation "
                         "starts after these. NOT optional -- see module docstring")
    ap.add_argument("--seqs", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from mlx_lm.utils import load_model
    from reap_calibrate import build_tokenizer

    tok_src = a.path if os.path.exists(os.path.join(a.path, "tiktoken.model")) else a.src
    enc = build_tokenizer(tok_src)
    ids, sids, labels = load_heldout(a, enc)

    t0 = time.time()
    print(f"[ppl] loading {a.path} ...", flush=True)
    model, _ = load_model(pathlib.Path(a.path), lazy=False)
    mx.eval(model.parameters())
    peak = mx.get_peak_memory() / 1e9 if hasattr(mx, "get_peak_memory") else float("nan")
    print(f"[ppl] loaded in {time.time()-t0:.0f}s | peak {peak:.0f} GB", flush=True)

    nll = np.zeros((a.seqs, a.seqlen - 1), np.float64)
    t0 = time.time()
    for s in range(a.seqs):
        x = mx.array(ids[s][None, :])
        cache = model.make_cache()
        logits = model(x, cache=cache).astype(mx.float32)[0]        # (seqlen, vocab)
        # predict token t+1 from position t
        lse = mx.logsumexp(logits[:-1], axis=-1)
        tgt = mx.array(ids[s][1:].astype(np.int32))
        picked = mx.take_along_axis(logits[:-1], tgt[:, None], axis=-1)[:, 0]
        row = lse - picked
        mx.eval(row)
        nll[s] = np.array(row, np.float64)
        del logits, cache, x, lse, picked, row
        el = time.time() - t0
        print(f"[ppl] seq {s+1:3d}/{a.seqs}  ppl so far "
              f"{np.exp(nll[:s+1].mean()):8.3f}  {el/(s+1):5.1f}s/seq  "
              f"elapsed {el/60:5.1f}m", flush=True)

    tgt_src = sids[:, 1:]
    overall = float(np.exp(nll.mean()))
    print(f"\n[ppl] {a.path}")
    print(f"[ppl] overall perplexity {overall:.4f} over {nll.size:,} tokens")
    per = {}
    for i, lab in enumerate(labels):
        m = tgt_src == i
        if m.sum() == 0:
            continue
        per[lab] = float(np.exp(nll[m].mean()))
        print(f"[ppl]   {lab:14s} {per[lab]:10.4f}   ({int(m.sum()):,} tokens)")

    np.savez(a.out, nll=nll, source_ids=tgt_src, labels=np.array(labels),
             ids=ids, overall=np.array(overall), skip_tokens=np.array(a.skip_tokens))
    print(f"[ppl] wrote {a.out}")
    print("[ppl] PPL-DONE")


if __name__ == "__main__":
    main()
