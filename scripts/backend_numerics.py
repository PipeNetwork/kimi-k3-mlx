#!/usr/bin/env python3
"""Do MLX's quantized kernels agree between Metal and CUDA on identical bytes?

scripts/mlx_cuda_gate.py established that every op K3 needs *runs* on sm_121 and
returns finite values. It never checked the values were right. A dequant or qmm
that is merely plausible produces a model that loads, runs, and emits garbage --
which is what the cluster did.

  # on the Mac (Metal)
  backend_numerics.py --dump out/parity/backend_ref.safetensors
  # on a GB10 (CUDA), same file
  backend_numerics.py --check backend_ref.safetensors
"""
import argparse

import mlx.core as mx
import mlx.nn as nn

E_IN, E_OUT, H, TOPK, N_EXPERTS = 3584, 3072, 7168, 16, 64
MODES = (("mxfp4", 4, 32), ("affine", 8, 64), ("affine", 4, 64))


def build(seed=0):
    """Identical quantized weights and inputs for both backends."""
    mx.random.seed(seed)
    t = {}
    t["x_lin"] = mx.random.normal((1, 64, H)).astype(mx.bfloat16)
    # gather_qmm batches over x's leading dims, which must match rhs_indices:
    # one (token, slot) pair per row, 8 tokens x top-16
    t["x_moe"] = mx.random.normal((8 * TOPK, 1, E_IN)).astype(mx.bfloat16)
    t["idx"] = mx.broadcast_to(mx.arange(TOPK).reshape(1, TOPK), (8, TOPK)).reshape(-1)
    for mode, bits, gs in MODES:
        tag = f"{mode}{bits}"
        # mxfp4 returns (weight, scales); affine returns (weight, scales, biases)
        w = mx.random.normal((H, H)).astype(mx.bfloat16)
        q = mx.quantize(w, group_size=gs, bits=bits, mode=mode)
        t[f"{tag}.w"], t[f"{tag}.s"] = q[0], q[1]
        if len(q) > 2:
            t[f"{tag}.b"] = q[2]
        we = mx.random.normal((N_EXPERTS, E_OUT, E_IN)).astype(mx.bfloat16)
        qe = mx.quantize(we, group_size=gs, bits=bits, mode=mode)
        t[f"{tag}.ew"], t[f"{tag}.es"] = qe[0], qe[1]
        if len(qe) > 2:
            t[f"{tag}.eb"] = qe[2]
    mx.eval(t)
    return t


def compute(t):
    """Run the ops K3 depends on; returns name -> result."""
    out = {}
    for mode, bits, gs in MODES:
        tag = f"{mode}{bits}"
        w, s = t[f"{tag}.w"], t[f"{tag}.s"]
        b = t.get(f"{tag}.b")
        kw = dict(group_size=gs, bits=bits, mode=mode)

        out[f"{tag}.dequant"] = mx.dequantize(w, s, b, **kw)
        out[f"{tag}.qmm"] = mx.quantized_matmul(
            t["x_lin"], w, s, b, transpose=True, **kw)

        ew, es = t[f"{tag}.ew"], t[f"{tag}.es"]
        eb = t.get(f"{tag}.eb")
        out[f"{tag}.gather_qmm"] = mx.gather_qmm(
            t["x_moe"], ew, es, eb,
            rhs_indices=t["idx"], transpose=True, sorted_indices=True, **kw)
    mx.eval(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump")
    ap.add_argument("--check")
    a = ap.parse_args()

    dev = mx.device_info()
    print(f"backend: {dev.get('device_name')} / {dev.get('architecture')}  mlx {mx.__version__}")

    if a.dump:
        t = build()
        res = compute(t)
        mx.save_safetensors(a.dump, {**t, **{f"out.{k}": v for k, v in res.items()}})
        for k, v in res.items():
            av = mx.abs(v.astype(mx.float32))
            print(f"  {k:<26} mean|y|={av.mean().item():.6f} max={av.max().item():.4f}")
        print(f"wrote {a.dump}")
        return

    ref = mx.load(a.check)
    t = {k: v for k, v in ref.items() if not k.startswith("out.")}
    res = compute(t)
    bad = 0
    print(f"{'op':<26} {'max abs diff':>13} {'rel':>10} {'cosine':>11}")
    print("-" * 64)
    for k, got in res.items():
        want = ref[f"out.{k}"].astype(mx.float32)
        got = got.astype(mx.float32)
        d = mx.abs(want - got)
        denom = mx.maximum(mx.abs(want).mean(), mx.array(1e-9))
        rel = (d.mean() / denom).item()
        gw, gg = want.flatten(), got.flatten()
        cos = (mx.sum(gw * gg) / (mx.sqrt(mx.sum(gw * gw)) * mx.sqrt(mx.sum(gg * gg)) + 1e-12)).item()
        flag = "" if cos > 0.999 else "   <-- MISMATCH"
        if cos <= 0.999:
            bad += 1
        print(f"{k:<26} {d.max().item():>13.3e} {rel:>10.3e} {cos:>11.7f}{flag}")
    print("\n" + ("BACKEND NUMERICS OK" if bad == 0 else f"BACKEND NUMERICS FAIL ({bad} ops)"))
    raise SystemExit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
