#!/usr/bin/env python3
"""AWQ scale search for Kimi-K3's routed experts.

Plain affine quantization minimises error in the WEIGHTS. What matters is error
in the OUTPUT, and those differ whenever some input channels carry far more
activation than others: a channel the model barely uses gets the same number of
levels as one it depends on. AWQ fixes this by scaling salient input channels up
before quantizing and folding the inverse into the preceding matrix --
mathematically an identity, so only the quantization error moves.

K3 makes this unusually clean. Its LatentMoE routes every expert in a layer
through the SAME 3584-d latent, so one scale vector per layer covers all 896
experts, and the inverse folds exactly into `routed_expert_down_proj`:

    w1' = w1 * s          w3' = w3 * s          (scale input channels)
    down_proj' = down_proj / s                  (fold the inverse)

s_j = mean|x_j| ** alpha, with alpha grid-searched per layer to minimise
activation-weighted reconstruction error of the quantized experts. alpha=0 is
no-op; alpha=1 fully equalises. The search is what makes this calibrated rather
than a heuristic.

Usage:
  scripts/awq.py --saliency out/reap_awq.npz --src Kimi-K3-src \\
      --bits 2 --out out/awq_scales.npz
"""

import argparse
import json
import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import convert  # noqa: E402
from mlx_lm.models import kimi_k3  # noqa: E402

ALPHAS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]


def recon_error(w, s, act, bits, group_size):
    """Activation-weighted reconstruction error of w after scale-then-quantize.

    w:   (out, in) expert matrix
    s:   (in,)     candidate scale
    act: (in,)     mean |x| per input channel -- the weighting that makes this
                   an output-error proxy rather than a weight-error one
    """
    ws = w * s
    q, sc, b = mx.quantize(ws, group_size=group_size, bits=bits, mode="affine")
    deq = mx.dequantize(q, sc, b, group_size=group_size, bits=bits, mode="affine") / s
    err = mx.abs(deq - w) * act          # channels the model uses count more
    return float(mx.mean(err))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--saliency", required=True, help="npz with act_mag from calibration")
    p.add_argument("--src", required=True)
    p.add_argument("--bits", type=int, default=2)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--probe-experts", type=int, default=4,
                   help="experts per layer used to score each alpha")
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    z = np.load(a.saliency)
    if "act_mag" not in z.files:
        raise SystemExit("no act_mag in this npz -- rerun calibration with the "
                         "latent stats hook (scripts/reap_calibrate.py)")
    act_mag, act_n = z["act_mag"], z["act_n"]
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    args = kimi_k3.ModelArgs.from_dict(cfg)
    index = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    reader = convert.ShardReader(a.src, index, keep=2)

    moe = [i for i in range(args.num_hidden_layers) if kimi_k3.is_moe_layer(args, i)]
    if a.layers:
        moe = moe[: a.layers]
    lat = args.routed_expert_hidden_size

    print(f"AWQ scale search: {len(moe)} MoE layers, {a.bits}-bit, "
          f"{a.probe_experts} probe experts/layer, alphas {ALPHAS}")

    scales = np.ones((args.num_hidden_layers, lat), np.float32)
    chosen = {}
    rng = np.random.default_rng(0)

    for n, L in enumerate(moe):
        act = act_mag[L] / max(act_n[L], 1.0)
        act = act / (act.mean() + 1e-12)                # normalise, keeps s ~O(1)
        act_mx = mx.array(act.astype(np.float32))

        base = f"language_model.model.layers.{L}.block_sparse_moe.experts"
        probes = rng.choice(args.num_experts, a.probe_experts, replace=False)
        ws = []
        for e in probes:
            packed = reader.get(f"{base}.{int(e)}.w1.weight_packed").view(mx.uint32)
            sc = reader.get(f"{base}.{int(e)}.w1.weight_scale")
            ws.append(mx.dequantize(packed, sc, group_size=32, bits=4,
                                    mode="mxfp4").astype(mx.float32))

        best_a, best_e = 0.0, None
        for alpha in ALPHAS:
            s = mx.power(act_mx, alpha)
            s = s / mx.sqrt(mx.max(s) * mx.min(s))      # centre so s ~ 1
            err = float(np.mean([recon_error(w, s, act_mx, a.bits, a.group_size)
                                 for w in ws]))
            if best_e is None or err < best_e:
                best_a, best_e = alpha, err
        s = np.array(mx.power(act_mx, best_a) /
                     mx.sqrt(mx.max(mx.power(act_mx, best_a)) *
                             mx.min(mx.power(act_mx, best_a))))
        scales[L] = s
        chosen[L] = best_a
        del ws
        if n % 10 == 0 or n == len(moe) - 1:
            print(f"  layer {L:3d}  alpha {best_a:.2f}  err {best_e:.3e}  "
                  f"s range [{s.min():.3f}, {s.max():.3f}]", flush=True)

    vals = list(chosen.values())
    print(f"\nalpha chosen: mean {np.mean(vals):.2f}  "
          f"{{{', '.join(f'{x}:{vals.count(x)}' for x in sorted(set(vals)))}}}")
    if all(v == 0.0 for v in vals):
        print("  WARNING: alpha=0 everywhere -- AWQ finds no benefit at this bit width")

    np.savez(a.out, scales=scales, alpha=np.array([chosen.get(i, 0.0)
                                                   for i in range(args.num_hidden_layers)]),
             bits=np.array(a.bits), group_size=np.array(a.group_size))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
