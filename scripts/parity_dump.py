#!/usr/bin/env python3
"""MLX half of the layer-parity test: run one layer's attention and record I/O.

Writes a self-contained bundle -- the source-format weights, the exact inputs,
and MLX's outputs -- so the torch side needs no access to the shards and cannot
diverge on weight loading. Compare with scripts/parity_check.py on a CUDA box.

  scripts/install_model.sh
  scripts/parity_dump.py --layer 13 --seqlen 64
"""
import argparse
import json
import os
import sys

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_args(config_path):
    from mlx_lm.models.kimi_k3 import ModelArgs

    cfg = json.load(open(config_path))
    text = cfg.get("text_config", cfg)
    text.setdefault("model_type", "kimi_k3")
    return ModelArgs.from_dict(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--seqlen", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--src-config", default="Kimi-K3-src/config.json")
    ap.add_argument("--dir", default="out/parity")
    args = ap.parse_args()

    from mlx_lm.models.kimi_k3 import (
        KimiDeltaAttention,
        KimiMLAAttention,
        is_kda_layer,
        remap_checkpoint,
    )

    margs = build_args(args.src_config)
    L, B, T, H = args.layer, args.batch, args.seqlen, margs.hidden_size
    kda = is_kda_layer(margs, L)
    print(f"layer {L}: {'KDA' if kda else 'MLA'}  hidden={H}  B={B} T={T}")

    src_path = os.path.join(args.dir, f"layer{L}_src.safetensors")
    src = mx.load(src_path)
    print(f"loaded {len(src)} source tensors from {src_path}")

    # Route through the model's own remap so this exercises the real load path.
    prefixed = {f"model.layers.{L}.{k}": v for k, v in src.items()}
    remapped = remap_checkpoint(prefixed, margs, layer_indices=[L], stack_experts=False)
    attn_prefix = f"model.layers.{L}.self_attn."
    attn_weights = {
        k[len(attn_prefix):]: v for k, v in remapped.items() if k.startswith(attn_prefix)
    }

    module = KimiDeltaAttention(margs, L) if kda else KimiMLAAttention(margs)
    module.load_weights(list(attn_weights.items()), strict=True)
    module.eval()
    print(f"built {type(module).__name__}, loaded {len(attn_weights)} tensors")

    mx.random.seed(args.seed)
    x = mx.random.normal((B, T, H)).astype(mx.bfloat16)
    mx.eval(x)

    # Prefill path (mask=None -> causal inside the module), no cache.
    y = module(x, None, None)
    mx.eval(y)

    out = {
        "__in_x": x,
        "__out_y": y,
        **{f"__w_{k}": v for k, v in src.items() if k.startswith("self_attn.")},
    }
    meta = {
        "layer": str(L), "kind": "kda" if kda else "mla", "batch": str(B),
        "seqlen": str(T), "seed": str(args.seed), "hidden_size": str(H),
    }
    dst = os.path.join(args.dir, f"layer{L}_mlx.safetensors")
    mx.save_safetensors(dst, out, metadata=meta)

    ay = mx.abs(y.astype(mx.float32))
    print(f"out {tuple(y.shape)} {y.dtype}  mean|y|={ay.mean().item():.6f}  "
          f"max|y|={ay.max().item():.6f}  finite={bool(mx.isfinite(y).all().item())}")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
