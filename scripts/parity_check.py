#!/usr/bin/env python3
"""CUDA half of the layer-parity test: run Moonshot's reference layer, compare.

Runs on a box with real Triton kernels (fla), which is the whole point -- the
text tower's KDA path can never be exercised against the reference on Apple
silicon, so `fla` gets stubbed there and the comparison never happens.

  .venv/bin/python parity_check.py --bundle layer13_mlx.safetensors --config config.json
"""
import argparse
import json
import sys

import torch
from safetensors.torch import load_file


def build_config(config_path):
    from ref.configuration_kimi_k3 import KimiLinearConfig

    cfg = json.load(open(config_path))
    text = cfg.get("text_config", cfg)
    text.pop("architectures", None)
    text.pop("auto_map", None)
    return KimiLinearConfig(**text)


def load_into(module, weights, label):
    """Assign checkpoint tensors onto the module, reporting any shape the
    reference's own __init__ disagrees with -- that disagreement is a finding,
    not an error to paper over."""
    own = dict(module.named_parameters())
    own.update(dict(module.named_buffers()))
    missing = sorted(set(own) - set(weights))
    unexpected = sorted(set(weights) - set(own))
    mismatched = []
    for name, tensor in weights.items():
        if name not in own:
            continue
        target = own[name]
        if tuple(target.shape) != tuple(tensor.shape):
            mismatched.append((name, tuple(target.shape), tuple(tensor.shape)))
        # checkpoint shape wins: the released weights are authoritative
        parent = module
        *path, leaf = name.split(".")
        for part in path:
            parent = getattr(parent, part)
        value = tensor.to(target.dtype)
        if isinstance(getattr(parent, leaf), torch.nn.Parameter):
            setattr(parent, leaf, torch.nn.Parameter(value, requires_grad=False))
        else:
            setattr(parent, leaf, value)

    print(f"[{label}] loaded {len(weights)-len(unexpected)}/{len(own)} tensors")
    if missing:
        print(f"[{label}] MISSING from checkpoint: {missing}")
    if unexpected:
        print(f"[{label}] not in module: {unexpected}")
    for name, want, got in mismatched:
        print(f"[{label}] SHAPE reference.__init__={want} checkpoint={got}  <- {name}")
    return mismatched


def compare(name, a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    diff = (a - b).abs()
    denom = a.abs().clamp_min(1e-6)
    rel = (diff / denom).median().item()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    print(f"\n=== {name} ===")
    print(f"  max abs diff   {diff.max().item():.3e}")
    print(f"  mean abs diff  {diff.mean().item():.3e}")
    print(f"  median rel     {rel:.3e}")
    print(f"  cosine         {cos:.8f}")
    print(f"  ref mean|y|    {b.abs().mean().item():.6f}")
    print(f"  mlx mean|y|    {a.abs().mean().item():.6f}")
    return cos, diff.max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                    help="compute dtype for the reference module")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    from ref.modeling_kimi_linear import KimiDeltaAttention, KimiMLAAttention

    bundle = load_file(args.bundle)
    meta = {}
    with open(args.bundle, "rb") as f:
        import struct
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n)).get("__metadata__", {})
    layer = int(meta["layer"])
    kind = meta["kind"]
    print(f"bundle: layer {layer} ({kind}), seqlen {meta['seqlen']}, seed {meta['seed']}")

    cfg = build_config(args.config)
    # eager keeps this dependency-free; the MLA path branches on it
    cfg._attn_implementation = "eager"
    torch.set_grad_enabled(False)
    dev = torch.device(args.device)

    module = (KimiDeltaAttention if kind == "kda" else KimiMLAAttention)(cfg, layer)
    module = module.to(dev).to(dtype).eval()

    weights = {
        k[len("__w_self_attn."):]: v.to(dev)
        for k, v in bundle.items() if k.startswith("__w_self_attn.")
    }
    load_into(module, weights, "attn")

    x = bundle["__in_x"].to(dev).to(dtype)
    mlx_y = bundle["__out_y"].to(dev)
    print(f"input {tuple(x.shape)} {x.dtype}")

    if kind == "kda":
        out = module(hidden_states=x, attention_mask=None, cache_params=None)
    else:
        pos = torch.arange(x.shape[1], device=dev).unsqueeze(0)
        out = module(hidden_states=x, attention_mask=None, position_ids=pos,
                     past_key_values=None)
    ref_y = out[0] if isinstance(out, tuple) else out
    torch.cuda.synchronize()
    print(f"ref out {tuple(ref_y.shape)} {ref_y.dtype} finite={torch.isfinite(ref_y).all().item()}")

    cos, maxdiff = compare(f"layer {layer} {kind} attention: MLX vs torch reference",
                           mlx_y, ref_y)
    ok = cos > 0.999
    print("\n" + ("PARITY OK" if ok else "PARITY FAIL") + f"  (cosine {cos:.6f})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
