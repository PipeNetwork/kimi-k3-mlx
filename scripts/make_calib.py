#!/usr/bin/env python3
"""Assemble a calibration corpus for REAP saliency on Kimi-K3.

Calibration decides which experts survive pruning, so the mix is a real
modelling choice, not a formality. Calibrate on English code alone and you will
faithfully prune away whatever handles Chinese prose -- with no error, no
warning, and a model that looks fine until someone writes in Chinese.

K3 is pitched for long-horizon coding, agentic knowledge work and multilingual
use, and ships a 163,840-token tiktoken vocab, so the default mix is:

    40%  code          multi-language (OpenCoder annealing) + real Python files
    30%  English web   FineWeb
    15%  Chinese       C4 zh -- given an explicit share because C4's pooled
                       "multilingual" stream skews European and left CJK at
                       0.03% of the corpus, which would have quietly pruned
                       whatever handles Chinese
    15%  multilingual  C4 per-language: ja, ru, ko, de, fr, es, ar. Named
                       explicitly because C4's pooled "multilingual" stream
                       measured 97% Latin-script over its first 200 documents
                       -- effectively a second helping of English, leaving the
                       Japanese/Korean/Cyrillic/Arabic experts unprotected.

Sources are **interleaved**, not concatenated. `reap_calibrate.py` consumes the
first `seqs * seqlen` tokens of this file, so a concatenated corpus would
calibrate entirely on whichever source happened to be written first.

Deterministic: fixed seed, fixed take order, no shuffling of the stream itself.

Usage:
  scripts/make_calib.py --out out/calib.txt --mb 12
"""

import argparse
import itertools
import os
import random
import sys

def _cjk_frac(s):
    return sum(1 for c in s if "\u4e00" <= c <= "\u9fff") / max(len(s), 1)


# C4's zh split contains some double-encoded documents (mojibake renders as CJK
# codepoints, so a plain "has CJK" check passes them). Requiring a healthy CJK
# share filters them without hand-maintaining a blocklist.
def _is_real_chinese(s):
    return _cjk_frac(s) > 0.35


MIX = [
    # (label, loader kwargs, text field, share, doc filter)
    ("code-multi", dict(path="OpenCoder-LLM/opc-annealing-corpus",
                        name="algorithmic_corpus", split="train"), "text", 0.25, None),
    ("code-python", dict(path="codeparrot/codeparrot-clean-valid",
                         split="train"), "content", 0.15, None),
    ("web-en", dict(path="HuggingFaceFW/fineweb",
                    name="sample-10BT", split="train"), "text", 0.30, None),
    ("chinese", dict(path="allenai/c4", name="zh", split="train"),
     "text", 0.15, _is_real_chinese),
] + [
    (f"lang-{lg}", dict(path="allenai/c4", name=lg, split="train"),
     "text", 0.15 / 7, None)
    for lg in ("ja", "ru", "ko", "de", "fr", "es", "ar")
]

MIN_DOC_CHARS = 400          # skip stubs; they waste sequence slots
MAX_DOC_CHARS = 20000        # keep any single document from dominating


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mb", type=float, default=12.0, help="target corpus size in MB")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from datasets import load_dataset

    target = int(a.mb * 1024 * 1024)
    rng = random.Random(a.seed)

    print(f"[calib] target {a.mb:.0f} MB")
    pools = {}
    for label, kw, field, share, keep in MIX:
        want = int(target * share)
        ds = load_dataset(**kw, streaming=True)
        docs, got = [], 0
        for row in ds:
            t = row.get(field) or ""
            if not isinstance(t, str) or len(t) < MIN_DOC_CHARS:
                continue
            if keep is not None and not keep(t):
                continue
            t = t[:MAX_DOC_CHARS]
            docs.append(t)
            got += len(t)
            if got >= want:
                break
        pools[label] = docs
        print(f"[calib]   {label:<13} {got/1e6:5.2f} MB  {len(docs):>5} docs "
              f"(target {want/1e6:.2f} MB)")

    # Interleave proportionally so any prefix of the file reflects the whole mix.
    order = []
    for label, _, _, share, _f in MIX:
        order += [label] * len(pools[label])
    rng.shuffle(order)

    cursors = {k: 0 for k in pools}
    written = 0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for label in order:
            i = cursors[label]
            if i >= len(pools[label]):
                continue
            cursors[label] = i + 1
            doc = pools[label][i]
            f.write(doc)
            f.write("\n\n")
            written += len(doc) + 2
            if written >= target:
                break

    print(f"[calib] wrote {written/1e6:.2f} MB -> {a.out}")
    for label in pools:
        print(f"[calib]   used {cursors[label]}/{len(pools[label])} {label} docs")


if __name__ == "__main__":
    main()
