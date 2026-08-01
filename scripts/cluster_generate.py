#!/usr/bin/env python3
"""Run Kimi-K3 pipelined across the GB10 cluster.

  mlx.launch --backend ring --hostfile hosts4.json --cwd <dir> -- \
      <venv>/bin/python cluster_generate.py --prompt "..."

Reports per-rank memory before and after load: rank 0 carries the extra layer
(~118 GB against ~127 GB free), so if anything OOMs it will be there, and the
pre/post numbers say whether it was the weights or the KV cache.
"""
import argparse
import socket
import sys
import time

import mlx.core as mx

from mlx_lm import stream_generate
from mlx_lm.utils import sharded_load

REPO = "pipenetwork/Kimi-K3-REAP73-MLX-mxfp4-q8"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=REPO)
    ap.add_argument("--prompt", default="Write a Python function that merges overlapping intervals.")
    ap.add_argument("--max-tokens", "-m", type=int, default=64)
    ap.add_argument("--raw", action="store_true", help="skip the chat template")
    ap.add_argument("--probe", action="store_true",
                    help="one forward pass; report logit health instead of generating")
    a = ap.parse_args()

    group = mx.distributed.init()
    rank, size = group.rank(), group.size()
    host = socket.gethostname()

    def rprint(*x, **k):
        if rank == 0:
            print(*x, **k)

    import os
    mode = os.environ.get("K3_PIPELINE_EXTRA", "low")
    free0 = mx.device_info()["free_memory"] / 1e9
    print(f"[rank {rank}/{size}] {host}: {free0:.0f} GB free before load "
          f"(extra={mode})", flush=True)

    t = time.time()
    model, tokenizer = sharded_load(a.model, group, None)
    load_s = time.time() - t

    info = mx.device_info()
    free1 = info["free_memory"] / 1e9
    n_layers = len(model.model.pipeline_layers)
    print(f"[rank {rank}/{size}] {host}: loaded layers"
          f"[{model.model.start_idx}:{model.model.end_idx}] ({n_layers}) in {load_s:.0f}s | "
          f"free {free0:.0f} -> {free1:.0f} GB (used {free0-free1:.0f} GB) | "
          f"mlx active {mx.get_active_memory()/1e9:.0f} GB", flush=True)

    # every rank must reach here before generation starts
    mx.eval(mx.distributed.all_sum(mx.array(1.0), group=group, stream=mx.cpu))

    if a.raw:
        prompt = tokenizer.encode(a.prompt)
    else:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": a.prompt}], add_generation_prompt=True
        )

    if a.probe:
        # single forward: is the model producing usable logits at all?
        toks = mx.array([list(prompt)])
        t = time.time()
        logits = model(toks)
        mx.eval(logits)
        dt = time.time() - t
        last = logits[0, -1].astype(mx.float32)
        finite = bool(mx.isfinite(last).all().item())
        nan_n = int(mx.sum(mx.isnan(last)).item())
        inf_n = int(mx.sum(mx.isinf(last)).item())
        print(f"[rank {rank}] forward {tuple(logits.shape)} in {dt:.1f}s | "
              f"finite={finite} nan={nan_n} inf={inf_n}", flush=True)
        if rank == 0:
            order = mx.argsort(-last)[:5]
            mx.eval(order)
            print(f"  logits: min={last.min().item():.4f} max={last.max().item():.4f} "
                  f"mean={last.mean().item():.4f} std={last.std().item():.4f}")
            for i in order.tolist():
                print(f"    id {i:>6} logit {last[i].item():>10.4f} "
                      f"{tokenizer.decode([i])!r}")
            print("PROBE-DONE")
        return

    rprint(f"\n=== prompt ({len(prompt)} tokens) ===\n{a.prompt}\n=== response ===")
    resp = None
    t = time.time()
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=a.max_tokens):
        rprint(resp.text, end="", flush=True)
    wall = time.time() - t

    if rank == 0 and resp is not None:
        print("\n" + "=" * 60)
        print(f"prompt     : {resp.prompt_tokens} tok @ {resp.prompt_tps:.2f} tok/s")
        print(f"generation : {resp.generation_tokens} tok @ {resp.generation_tps:.3f} tok/s")
        print(f"wall       : {wall:.1f}s")
        print(f"peak memory: {resp.peak_memory:.1f} GB (rank 0)")
        print("CLUSTER-GENERATE-DONE")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
