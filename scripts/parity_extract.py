#!/usr/bin/env python3
"""Pull one layer's non-expert tensors out of the source shards.

Both sides of the parity test load *these exact bytes* in the source's own
naming, so neither implementation can drift on weight loading -- the thing being
compared is the arithmetic, not the plumbing.

  scripts/parity_extract.py --layer 13 --src Kimi-K3-src --out out/parity
"""
import argparse
import gzip
import json
import os

import mlx.core as mx

PREFIX = "language_model.model.layers"


def layer_keys(index_map, layer, include_experts):
    want = f"{PREFIX}.{layer}."
    for k in index_map:
        if not k.startswith(want):
            continue
        if not include_experts and ".experts." in k:
            continue
        yield k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--src", default="Kimi-K3-src")
    ap.add_argument("--out", default="out/parity")
    ap.add_argument("--index", default="reference/model.safetensors.index.json.gz")
    ap.add_argument("--experts", action="store_true",
                    help="include the routed-expert bank (~16 GB)")
    args = ap.parse_args()

    opener = gzip.open if args.index.endswith(".gz") else open
    index_map = json.load(opener(args.index))["weight_map"]

    keys = sorted(layer_keys(index_map, args.layer, args.experts))
    if not keys:
        raise SystemExit(f"no tensors for layer {args.layer}")

    by_shard = {}
    for k in keys:
        by_shard.setdefault(index_map[k], []).append(k)

    tensors, total = {}, 0
    for shard, ks in sorted(by_shard.items()):
        # mx.load mmaps the shard; only the keys we touch are materialised.
        shard_tensors = mx.load(os.path.join(args.src, shard))
        for k in ks:
            t = shard_tensors[k]
            # strip the layer prefix so the consumer keys off module paths
            tensors[k[len(f"{PREFIX}.{args.layer}."):]] = t
            total += t.nbytes
        mx.eval(list(tensors.values()))

    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, f"layer{args.layer}_src.safetensors")
    mx.save_safetensors(dst, tensors, metadata={"layer": str(args.layer)})
    print(f"layer {args.layer}: {len(tensors)} tensors, {total/1e9:.3f} GB -> {dst}")
    for k in sorted(tensors)[:40]:
        print(f"   {k:<52} {str(tensors[k].dtype):<10} {tensors[k].shape}")
    if len(tensors) > 40:
        print(f"   ... and {len(tensors)-40} more")


if __name__ == "__main__":
    main()
