#!/usr/bin/env python3
"""Load a converted Kimi-K3 build and actually generate.

Uses the bundled tiktoken BPE directly rather than transformers' AutoTokenizer:
the shipped tokenization_kimi.py targets transformers 4.56 and does not import
under 5.x, and that is a packaging detail, not a reason to skip the only test
that proves the model works.

Usage: scripts/smoke.py --path out/Kimi-K3-REAP73-MLX-mxfp4 [--max-tokens 96]
"""

import argparse
import os
import pathlib
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--src", default="Kimi-K3-src", help="for tiktoken.model")
    a = ap.parse_args()

    from mlx_lm.utils import load_model
    from reap_calibrate import build_tokenizer
    import mlxmem

    # Must precede load_model: an unwired model faults its weights from SSD on
    # every decode step and reports ~1/27th of its real tok/s.
    mlxmem.wire()

    tok_src = a.path if os.path.exists(os.path.join(a.path, "tiktoken.model")) else a.src
    enc = build_tokenizer(tok_src)
    print(f"[smoke] tokenizer ok (n_vocab {enc.n_vocab})", flush=True)

    t0 = time.time()
    print(f"[smoke] loading {a.path} ...", flush=True)
    model, cfg = load_model(pathlib.Path(a.path), lazy=False)
    mx.eval(model.parameters())
    load_s = time.time() - t0
    peak = mx.get_peak_memory() / 1e9 if hasattr(mx, "get_peak_memory") else float("nan")
    print(f"[smoke] loaded in {load_s:.0f}s  | peak mem {peak:.0f} GB", flush=True)

    prompts = [
        "def merge_intervals(intervals):\n    \"\"\"Merge overlapping intervals.\"\"\"\n",
        "The capital of France is",
        "机器学习的基本原理是",
    ]

    for p in prompts:
        ids = enc.encode_ordinary(p)
        x = mx.array([ids])
        cache = model.make_cache()

        t0 = time.time()
        logits = model(x, cache=cache)
        mx.eval(logits)
        prefill_s = time.time() - t0

        print(f"\n[smoke] prompt: {p!r}")
        print(f"[smoke] prefill {len(ids)} tok in {prefill_s:.1f}s", flush=True)
        print("[smoke] --> ", end="", flush=True)

        out, tok, shown = [], int(mx.argmax(logits[0, -1])), ""
        t0 = time.time()
        for i in range(a.max_tokens):
            out.append(tok)
            # Re-decode the whole prefix and print the delta: decoding a single
            # id can split a multi-byte CJK character and emit replacement chars.
            text = enc.decode(out)
            if len(text) > len(shown):
                print(text[len(shown):], end="", flush=True)
                shown = text
            logits = model(mx.array([[tok]]), cache=cache)
            mx.eval(logits)
            tok = int(mx.argmax(logits[0, -1]))
            if tok in (163584, 163586):          # BOS / end_of_msg
                break
            if i == 0:
                print(f"\n[smoke]     (first token in {time.time()-t0:.1f}s)\n[smoke] --> {shown}",
                      end="", flush=True)
        gen_s = time.time() - t0
        print(f"\n[smoke] decode {len(out)} tok in {gen_s:.1f}s = "
              f"{len(out)/max(gen_s,1e-9):.2f} tok/s", flush=True)

    print("\n[smoke] SMOKE-DONE")


if __name__ == "__main__":
    main()
