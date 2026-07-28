#!/usr/bin/env python3
"""Re-balance an existing calibration corpus into a pre-tokenized .npy.

make_calib.py targets token shares when it BUILDS a corpus, but anything that
reads only a prefix of the result gets whatever mix happens to sit at the front
of the file. The AWQ scale search consumed the first 65,536 tokens of
out/calib.txt and got 36.4% Chinese, 28.6% code-multi, and zero French or
Russian -- and the resulting scales improved Chinese perplexity 5.7% while
regressing Arabic 6.6%. A single scale vector per layer cannot serve every
domain, so which domains it serves is decided by this file.

Two things this guarantees that slicing a prefix does not:

  * equal token counts per source, not equal bytes and not "whatever came first"
  * documents round-robined across sources, so every sequence carries a mixed
    context. Concatenating by source would give single-domain sequences, and
    activation statistics are context-dependent even though the accumulator is
    not.

--skip-tokens exists to keep the output disjoint from regions that must stay
held out (the REAP calibration prefix, the perplexity eval window). It counts
tokens from the start of the corpus, matching how reap_calibrate.py consumes it.

Usage:
  scripts/make_balanced_calib.py --corpus out/calib.txt --src Kimi-K3-src \\
      --skip-tokens 265536 --tokens 65536 --seqlen 2048 \\
      --out out/calib_balanced.npy
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--src", default="Kimi-K3-src", help="for tiktoken.model")
    p.add_argument("--skip-tokens", type=int, default=0,
                   help="ignore documents falling within this many tokens of the "
                        "corpus start; use it to stay clear of held-out regions")
    p.add_argument("--tokens", type=int, default=65536)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    from reap_calibrate import build_tokenizer
    enc = build_tokenizer(a.src)

    side = a.corpus + ".sources.json"
    if not os.path.exists(side):
        raise SystemExit(f"need {side} to know which document is which source")
    man = json.load(open(side))
    text = open(a.corpus, errors="ignore").read()
    labels = sorted(set(man["order"]))

    # bucket documents by source, skipping the held-out prefix
    buckets = {l: [] for l in labels}
    pos = seen = 0
    for lab, n in zip(man["order"], man["chars"]):
        ids = enc.encode_ordinary(text[pos:pos + n])
        pos += n
        if seen >= a.skip_tokens:
            buckets[lab].append(ids)
        seen += len(ids)
    print(f"[bal] corpus {seen:,} tokens; using documents beyond {a.skip_tokens:,}")

    # Size the quotas to the exact sequence count instead of trimming the stream
    # afterwards. Trimming looks harmless but is not: a source with short
    # documents fills its quota across every round-robin pass, including the
    # last ones, so a tail cut lands almost entirely on it -- code-multi came
    # out at 6.2% against 9.4% for everyone else that way.
    n_seq = a.tokens // a.seqlen
    total = n_seq * a.seqlen
    n_src = len(labels)
    quotas = {l: total // n_src for l in labels}
    for l in labels[: total % n_src]:
        quotas[l] += 1
    short = [l for l in labels if sum(len(d) for d in buckets[l]) < quotas[l]]
    if short:
        quota = min(quotas.values())
        raise SystemExit(
            f"sources {short} have fewer than {quota:,} tokens available past "
            f"--skip-tokens; lower --tokens or --skip-tokens."
        )

    # round-robin documents so sequences are mixed, not blocked by source
    taken = {l: 0 for l in labels}
    cursor = {l: 0 for l in labels}
    stream, sids = [], []
    lut = {l: i for i, l in enumerate(labels)}
    while any(taken[l] < quotas[l] for l in labels):
        progressed = False
        for l in labels:
            if taken[l] >= quotas[l] or cursor[l] >= len(buckets[l]):
                continue
            doc = buckets[l][cursor[l]]
            cursor[l] += 1
            doc = doc[: quotas[l] - taken[l]]
            stream.extend(doc)
            sids.extend([lut[l]] * len(doc))
            taken[l] += len(doc)
            progressed = True
        if not progressed:
            break

    if len(stream) != total:
        raise SystemExit(f"built {len(stream):,} tokens, expected {total:,}")
    arr = np.array(stream, np.int32).reshape(n_seq, a.seqlen)
    sarr = np.array(sids, np.int32).reshape(n_seq, a.seqlen)
    np.save(a.out, arr)
    np.save(a.out.replace(".npy", "") + ".sources.npy", sarr)

    print(f"[bal] wrote {a.out}  {arr.shape} = {arr.size:,} tokens")
    print(f"[bal] {'source':14s} {'tokens':>9s} {'share':>7s}")
    for i, l in enumerate(labels):
        n = int((sarr == i).sum())
        print(f"[bal] {l:14s} {n:9,d} {n/sarr.size*100:6.2f}%")
    spread = max((sarr == i).sum() for i in range(len(labels))) / \
             max(1, min((sarr == i).sum() for i in range(len(labels))))
    print(f"[bal] max/min source ratio {spread:.3f} (1.000 = perfectly even)")


if __name__ == "__main__":
    main()
