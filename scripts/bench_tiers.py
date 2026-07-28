#!/usr/bin/env python3
"""Measure decode tok/s for converted tiers, one per subprocess.

Each tier is benchmarked in a fresh process: a 450 GB model does not get fully
released inside one interpreter reliably enough to load the next one, and a
half-released predecessor is exactly the kind of memory pressure that produces
the bogus numbers this script exists to replace.

Reports effective GB/s alongside tok/s. That is the number that says whether a
result is believable: an M3 Ultra peaks near 819 GB/s, so anything reading a
couple of percent of that is a misconfiguration, not a model property.

Usage:
  scripts/bench_tiers.py out/Kimi-K3-REAP80-MLX-mxfp4-q8 out/... [--tokens 48]
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

CHILD = r'''
import json, pathlib, sys, time
import mlx.core as mx
sys.path.insert(0, %(here)r); sys.path.insert(0, %(root)r)
import mlxmem
from mlx_lm.utils import load_model
from reap_calibrate import build_tokenizer

path, ntok = sys.argv[1], int(sys.argv[2])
mlxmem.wire(verbose=False)
enc = build_tokenizer(path if pathlib.Path(path, "tiktoken.model").exists()
                      else %(src)r)
t0 = time.time()
model, cfg = load_model(pathlib.Path(path), lazy=False)
mx.eval(model.parameters())
load_s = time.time() - t0
peak = mx.get_peak_memory() / 1e9

ids = enc.encode_ordinary("Write a Python function that merges overlapping intervals.")
cache = model.make_cache()
t0 = time.time(); lg = model(mx.array([ids]), cache=cache); mx.eval(lg)
prefill_s = time.time() - t0
tok = int(mx.argmax(lg[0, -1]))

t0 = time.time()
for _ in range(ntok):
    lg = model(mx.array([[tok]]), cache=cache); mx.eval(lg)
    tok = int(mx.argmax(lg[0, -1]))
dec = time.time() - t0

tc = cfg.get("text_config", cfg)
ec = tc.get("expert_counts") or []
kept = max(ec) if ec else tc.get("n_routed_experts", 0)
print("RESULT " + json.dumps(dict(
    path=path, load_s=round(load_s, 1), peak_gb=round(peak, 1),
    prefill_tok=len(ids), prefill_s=round(prefill_s, 2),
    tok_s=round(ntok / dec, 2), ms_tok=round(dec / ntok * 1000), kept=kept)))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--src", default=os.path.join(os.path.dirname(HERE), "Kimi-K3-src"))
    ap.add_argument("--out", default=None, help="write results as json")
    a = ap.parse_args()

    src = CHILD % dict(here=HERE, root=os.path.dirname(HERE), src=a.src)
    rows = []
    for p in a.paths:
        if not os.path.isdir(p):
            print(f"[bench] skip {p} (not found)", flush=True)
            continue
        print(f"[bench] {os.path.basename(p)} ...", flush=True)
        r = subprocess.run([sys.executable, "-c", src, p, str(a.tokens)],
                           capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
        if not line:
            print(f"[bench]   FAILED: {r.stderr.strip().splitlines()[-1:] or r.stdout[-300:]}",
                  flush=True)
            continue
        d = json.loads(line[len("RESULT "):])
        rows.append(d)
        print(f"[bench]   {d['tok_s']:6.2f} tok/s  {d['ms_tok']:5d} ms/tok  "
              f"peak {d['peak_gb']:6.1f} GB  load {d['load_s']:5.1f}s  "
              f"prefill {d['prefill_tok']}tok/{d['prefill_s']}s", flush=True)

    print(f"\n{'tier':46s} {'experts':>7s} {'GB':>7s} {'tok/s':>7s} {'ms/tok':>7s}")
    for d in rows:
        print(f"{os.path.basename(d['path']):46s} {d['kept']:7d} {d['peak_gb']:7.1f} "
              f"{d['tok_s']:7.2f} {d['ms_tok']:7d}")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=2)
        print(f"\n[bench] wrote {a.out}")
    print("[bench] BENCH-TIERS-DONE")


if __name__ == "__main__":
    main()
