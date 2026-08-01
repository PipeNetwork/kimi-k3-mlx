#!/usr/bin/env python3
"""Run one real KDA layer through MLX on this backend and compare to the bundle.

kimi_k3.py picks its delta-rule implementation with

    use_kernel = mx.default_device() == mx.gpu and mx.metal.is_available()

so on CUDA every one of the 69 KDA layers silently takes `gated_delta_ops`, the
sequential ops fallback, instead of the fused kernel. That path is only ever
exercised on CPU on a Mac. This runs the real layer-13 weights through whichever
path this backend selects and diffs against the Metal reference in the bundle.

  kda_backend_check.py --bundle layer13_mlx.safetensors
"""
import argparse

import mlx.core as mx
from mlx_lm.models import kimi_k3
from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--layer", type=int, default=13)
    a = ap.parse_args()

    dev = mx.device_info()
    metal = mx.metal.is_available() if hasattr(mx, "metal") else False
    print(f"backend {dev.get('device_name')} / {dev.get('architecture')} | "
          f"metal.is_available()={metal} -> "
          f"{'gated_delta_kernel' if metal else 'gated_delta_ops (fallback)'}")

    import json
    cfg = json.load(open(a.config))
    text = cfg.get("text_config", cfg)
    text.setdefault("model_type", "kimi_k3")
    margs = kimi_k3.ModelArgs.from_dict(text)

    b = mx.load(a.bundle)
    src = {k[len("__w_"):]: v for k, v in b.items() if k.startswith("__w_")}
    prefixed = {f"model.layers.{a.layer}.{k}": v for k, v in src.items()}
    remapped = kimi_k3.remap_checkpoint(prefixed, margs, layer_indices=[a.layer],
                                        stack_experts=False)
    pre = f"model.layers.{a.layer}.self_attn."
    w = {k[len(pre):]: v for k, v in remapped.items() if k.startswith(pre)}

    mod = kimi_k3.KimiDeltaAttention(margs, a.layer)
    mod.load_weights(list(w.items()), strict=True)
    mod.eval()

    x = b["__in_x"]
    ref = b["__out_y"].astype(mx.float32)
    y = mod(x, None, None)
    mx.eval(y)
    y32 = y.astype(mx.float32)

    finite = bool(mx.isfinite(y32).all().item())
    d = mx.abs(ref - y32)
    fw, fg = ref.flatten(), y32.flatten()
    cos = (mx.sum(fw * fg) /
           (mx.sqrt(mx.sum(fw * fw)) * mx.sqrt(mx.sum(fg * fg)) + 1e-12)).item()
    print(f"  out {tuple(y.shape)} finite={finite} "
          f"nan={int(mx.sum(mx.isnan(y32)).item())}")
    print(f"  mean|ref|={mx.abs(ref).mean().item():.6f} "
          f"mean|got|={mx.abs(y32).mean().item():.6f}")
    print(f"  max abs diff {d.max().item():.3e}   cosine {cos:.7f}")
    print("KDA BACKEND OK" if finite and cos > 0.999 else "KDA BACKEND MISMATCH")

    # isolate the delta-rule op itself from the rest of the layer
    B, T, H = 1, 16, margs.linear_attn_config["head_dim"]
    nh = margs.linear_attn_config["num_heads"]
    mx.random.seed(0)
    q = mx.random.normal((B, T, nh, H)).astype(mx.bfloat16)
    k = mx.random.normal((B, T, nh, H)).astype(mx.bfloat16)
    v = mx.random.normal((B, T, nh, H)).astype(mx.bfloat16)
    g = mx.ones((B, T, nh, H)) * 0.9
    beta = mx.ones((B, T, nh)) * 0.5
    st = mx.zeros((B, nh, H, H), dtype=mx.float32)
    o, s2 = gated_delta_ops(q, k, v, g, beta, st, None)
    mx.eval(o, s2)
    print(f"  gated_delta_ops standalone: finite={bool(mx.isfinite(o).all().item())} "
          f"mean|o|={mx.abs(o.astype(mx.float32)).mean().item():.6f}")


if __name__ == "__main__":
    main()
