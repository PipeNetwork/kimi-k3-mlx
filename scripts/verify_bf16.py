"""Verify a bf16 build's expert tensors against an independent dequant of the source.

The converter dequantizes MXFP4 experts one at a time and stacks them. That stacking
is where an off-by-one would hide: every tensor would still be dense bf16 of the right
shape, just holding the wrong expert. So this checks *identity*, not just format —
each stacked slice must equal the source expert it claims to be, and must NOT equal
its neighbours (which would mean the index is being ignored).
"""
import argparse, glob, json, os, sys
import mlx.core as mx
from safetensors import safe_open

FP4_GROUP = 32


def src_expert(src, wm, layer, e, w):
    base = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{e}.{w}"
    out = {}
    for suf in ("weight_packed", "weight_scale"):
        k = f"{base}.{suf}"
        with safe_open(os.path.join(src, wm[k]), framework="np") as f:
            out[suf] = mx.array(f.get_tensor(k))
    return mx.dequantize(out["weight_packed"].view(mx.uint32), out["weight_scale"],
                         group_size=FP4_GROUP, bits=4, mode="mxfp4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--experts", type=int, default=4, help="how many experts to spot-check")
    a = ap.parse_args()

    wm = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    built = {}
    for f in glob.glob(os.path.join(a.build, "*.safetensors")):
        with safe_open(f, framework="np") as h:
            for k in h.keys():
                if f".layers.{a.layer}.block_sparse_moe.switch_mlp." in k:
                    built[k] = (f, k)
    if not built:
        sys.exit(f"no expert tensors for layer {a.layer} in {a.build}")

    fails = 0
    for key, (f, k) in sorted(built.items()):
        if key.endswith((".scales", ".biases")):
            sys.exit(f"FAIL: bf16 build contains a quantized leaf: {key}")
        # numpy has no bfloat16, so the build side is read with MLX's loader
        # (mmap-backed: slices materialize per expert, not the whole 60 GB stack).
        stack = mx.load(f)[k]
        # converted name -> source expert weight name
        dst = key.split(".switch_mlp.")[1].rsplit(".weight", 1)[0]
        w = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}[dst]
        if stack.dtype != mx.bfloat16:
            print(f"FAIL {key}: dtype {stack.dtype}, expected bfloat16"); fails += 1; continue
        n = stack.shape[0]
        picks = sorted({0, n - 1, *(i * n // a.experts for i in range(a.experts))})
        for e in picks:
            ref = src_expert(a.src, wm, a.layer, e, w).astype(mx.bfloat16)
            got = stack[e]
            if got.shape != ref.shape:
                print(f"FAIL {key}[{e}]: shape {got.shape} vs {ref.shape}"); fails += 1; continue
            exact = bool(mx.all(got == ref).item())
            # identity guard: a neighbour must NOT also match, or the index is inert
            nb = (e + 1) % n
            alias = bool(mx.all(got == src_expert(a.src, wm, a.layer, nb, w)
                                .astype(mx.bfloat16)).item())
            status = "OK " if (exact and not alias) else "FAIL"
            if status == "FAIL":
                fails += 1
                print(f"{status} {key}[{e}] exact={exact} matches_neighbour={alias}")
        print(f"  checked {key}  shape={tuple(stack.shape)}  experts={picks}")
    print(("PASS: every checked expert is bit-exact and uniquely indexed"
           if not fails else f"FAILED: {fails} check(s)"))
    sys.exit(1 if fails else 0)


main()
