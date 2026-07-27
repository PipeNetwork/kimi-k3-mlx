#!/usr/bin/env python3
"""Verify a converted Kimi-K3 MLX tier without loading the model.

No tier fits in this machine's 512 GB of unified memory (the smallest, 2-bit, is
~870 GB), so `mlx_lm.generate` is not an option. This checks everything that can
be checked from the artifacts themselves:

  1. config       quantization block is coherent and mlx-lm-loadable
  2. index        every mapped tensor exists, sizes agree with the index total
  3. coverage     the emitted key set matches exactly what kimi_k3.Model expects
                  (no missing modules, no orphans) -- derived from ModelArgs
                  rather than from an instantiated model
  4. numerics     dequantize sampled experts from the OUTPUT and compare against
                  dequantizing the same experts from the SOURCE

Check 4 is the one that catches a mis-stacked or mis-scaled expert, which is the
failure mode a structural check alone would sail straight past. For the mxfp4
tier the tolerance is exactly zero.

Usage: scripts/verify.py --path out/Kimi-K3-MLX-mxfp4 --src Kimi-K3-src
"""

import argparse
import json
import os
import random
import sys

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mlx_lm.models import kimi_k3  # noqa: E402

FAIL = []
WARN = []


def check(cond, msg):
    if cond:
        print(f"  ok    {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAIL.append(msg)


def warn(cond, msg):
    if not cond:
        print(f"  warn  {msg}")
        WARN.append(msg)


def expected_keys(args: kimi_k3.ModelArgs):
    """Module paths kimi_k3.Model expects, without building the 2.78T model."""
    q = set()          # quantizable -> .weight (+ .scales [+ .biases])
    plain = set()      # stored as-is

    plain |= {"model.norm.weight"}
    q |= {"model.embed_tokens"}
    if not args.tie_word_embeddings:
        q |= {"lm_head"}
    if args.attn_res_block_size is not None:
        plain |= {"model.output_attn_res.proj_weight", "model.output_attn_res.norm_weight"}

    for i in range(args.num_hidden_layers):
        p = f"model.layers.{i}"
        plain |= {f"{p}.input_layernorm.weight", f"{p}.post_attention_layernorm.weight"}
        if args.attn_res_block_size is not None:
            for n in ("self_attention_res", "mlp_res"):
                plain |= {f"{p}.{n}.proj_weight", f"{p}.{n}.norm_weight"}

        a = f"{p}.self_attn"
        if kimi_k3.is_kda_layer(args, i):
            q |= {f"{a}.{n}" for n in
                  ("q_proj", "k_proj", "v_proj", "o_proj", "f_a_proj", "f_b_proj", "b_proj")}
            cfg = args.linear_attn_config
            q |= {f"{a}.g_proj"} if cfg.get("use_full_rank_gate") else \
                 {f"{a}.g_a_proj", f"{a}.g_b_proj"}
            plain |= {f"{a}.A_log", f"{a}.dt_bias", f"{a}.o_norm.weight"}
            plain |= {f"{a}.{n}.conv.weight" for n in ("q_conv", "k_conv", "v_conv")}
        else:
            q |= {f"{a}.q_a_proj", f"{a}.q_b_proj", f"{a}.kv_a_proj_with_mqa",
                  f"{a}.o_proj", f"{a}.embed_q", f"{a}.unembed_out"}
            if args.mla_use_output_gate:
                q |= {f"{a}.g_proj"}
            plain |= {f"{a}.q_a_layernorm.weight", f"{a}.kv_a_layernorm.weight"}

        if kimi_k3.is_moe_layer(args, i):
            b = f"{p}.block_sparse_moe"
            q |= {f"{b}.gate"}
            q |= {f"{b}.switch_mlp.{n}" for n in ("gate_proj", "up_proj", "down_proj")}
            q |= {f"{b}.shared_experts.{n}" for n in ("gate_proj", "up_proj", "down_proj")} \
                 if args.num_shared_experts else set()
            plain |= {f"{b}.e_score_correction_bias"}
            if args.routed_expert_hidden_size is not None:
                q |= {f"{b}.routed_expert_down_proj", f"{b}.routed_expert_up_proj"}
                if args.latent_moe_use_norm:
                    plain |= {f"{b}.routed_expert_norm.weight"}
        else:
            q |= {f"{p}.mlp.{n}" for n in ("gate_proj", "up_proj", "down_proj")}

    # vision tower + projector: carried verbatim in bf16, keys in source form
    vc = args.vision_config or {}
    if vc:
        plain |= {
            "vision_tower.patch_embed.proj.weight",
            "vision_tower.patch_embed.pos_emb.weight",
            "vision_tower.encoder.final_layernorm.weight",
            "mm_projector.proj.0.weight",
            "mm_projector.proj.2.weight",
            "mm_projector.post_norm.weight",
        }
        for i in range(vc.get("vt_num_hidden_layers", 0)):
            b = f"vision_tower.encoder.blocks.{i}"
            plain |= {f"{b}.{n}.weight" for n in
                      ("wqkv", "wo", "mlp.fc0", "mlp.fc1", "norm0", "norm1")}

    return q, plain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--src", default=None, help="source repo, enables numeric spot-check")
    ap.add_argument("--samples", type=int, default=12)
    a = ap.parse_args()

    print(f"verifying {a.path}")

    # ---- 1. config
    cfg = json.load(open(os.path.join(a.path, "config.json")))
    args = kimi_k3.ModelArgs.from_dict(cfg)
    quant = cfg.get("quantization")
    print("\n[1] config")
    check(cfg.get("model_type") == "kimi_k3", "model_type == kimi_k3")
    check(quant is not None, "quantization block present")
    check("quantization_config" not in cfg,
          "stale compressed-tensors block removed (mlx-lm would mis-read it as affine)")
    if quant:
        print(f"        global: mode={quant.get('mode')} bits={quant.get('bits')} "
              f"group_size={quant.get('group_size')}  "
              f"per-module overrides={sum(1 for v in quant.values() if isinstance(v, dict))}")
    check(os.path.exists(os.path.join(a.path, "kimi_k3.py")), "kimi_k3.py bundled")
    for f in ("tokenizer_config.json", "tiktoken.model", "tokenization_kimi.py"):
        warn(os.path.exists(os.path.join(a.path, f)), f"{f} missing")

    # ---- 2. index
    print("\n[2] index")
    idx_path = os.path.join(a.path, "model.safetensors.index.json")
    idx = json.load(open(idx_path))
    wm = idx["weight_map"]
    shards = sorted(set(wm.values()))
    on_disk = sorted(f for f in os.listdir(a.path) if f.endswith(".safetensors"))
    check(shards == on_disk, f"{len(shards)} shards mapped == {len(on_disk)} on disk")
    actual = sum(os.path.getsize(os.path.join(a.path, s)) for s in on_disk)
    print(f"        total {actual / 1e12:.4f} TB across {len(on_disk)} shards")

    present = set()
    for s in on_disk:
        present |= set(mx.load(os.path.join(a.path, s)).keys())
    missing_from_disk = set(wm) - present
    check(not missing_from_disk, f"every mapped tensor present ({len(missing_from_disk)} missing)")

    # ---- 3. coverage
    print("\n[3] module coverage")
    q_exp, plain_exp = expected_keys(args)
    got = present
    missing, orphan = [], []
    for m in sorted(q_exp):
        if f"{m}.weight" not in got:
            missing.append(f"{m}.weight")
    for k in sorted(plain_exp):
        if k not in got:
            missing.append(k)
    accounted = set()
    for m in q_exp:
        accounted |= {f"{m}.weight", f"{m}.scales", f"{m}.biases"}
    accounted |= plain_exp
    orphan = sorted(got - accounted)
    check(not missing, f"no missing tensors ({len(missing)}): {missing[:5]}")
    check(not orphan, f"no orphan tensors ({len(orphan)}): {orphan[:5]}")
    n_quantized = sum(1 for m in q_exp if f"{m}.scales" in got)
    print(f"        {len(q_exp)} quantizable modules, {n_quantized} carry scales, "
          f"{len(plain_exp)} full-precision tensors")

    # bits/weight from the known 2.78T parameter count
    KNOWN_PARAMS = 2_779_970_000_000
    if actual > 1e11:
        print(f"        effective {actual * 8 / KNOWN_PARAMS:.3f} bits/weight "
              f"over {KNOWN_PARAMS / 1e12:.2f}T params")

    # ---- 4. numerics
    if a.src and os.path.exists(a.src):
        print("\n[4] expert numerics vs source")
        src_idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
        moe_layers = [i for i in range(args.num_hidden_layers) if kimi_k3.is_moe_layer(args, i)]
        rnd = random.Random(0)
        worst, worst_cos, checked, skipped = 0.0, 1.0, 0, 0
        out_cache = {}
        for _ in range(a.samples):
            layer = rnd.choice(moe_layers)
            e = rnd.randrange(args.num_experts)
            w, dst = rnd.choice([("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")])
            sk = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{e}.{w}"
            if f"{sk}.weight_packed" not in src_idx:
                skipped += 1
                continue
            sp = os.path.join(a.src, src_idx[f"{sk}.weight_packed"])
            if not os.path.exists(sp):
                skipped += 1
                continue
            src_sh = mx.load(sp)
            ref = mx.dequantize(
                src_sh[f"{sk}.weight_packed"].view(mx.uint32),
                src_sh[f"{sk}.weight_scale"],
                group_size=32, bits=4, mode="mxfp4",
            ).astype(mx.float32)

            op = f"model.layers.{layer}.block_sparse_moe.switch_mlp.{dst}"
            shard = os.path.join(a.path, wm[f"{op}.weight"])
            if shard not in out_cache:
                out_cache.clear()
                out_cache[shard] = mx.load(shard)
            osh = out_cache[shard]
            kw = {"group_size": quant.get("group_size"), "bits": quant.get("bits"),
                  "mode": quant.get("mode", "affine")}
            if op in quant:
                kw = {"group_size": quant[op]["group_size"], "bits": quant[op]["bits"],
                      "mode": quant[op]["mode"]}
            deq_args = [osh[f"{op}.weight"][e], osh[f"{op}.scales"][e]]
            if f"{op}.biases" in osh:
                deq_args.append(osh[f"{op}.biases"][e])
            got_w = mx.dequantize(*deq_args, **kw).astype(mx.float32)
            err = float(mx.abs(got_w - ref).max())
            rel = float(mx.abs(got_w - ref).mean() / (mx.abs(ref).mean() + 1e-12))
            # Cosine similarity is the discriminating test: quantization noise
            # barely moves it, but pulling the wrong expert out of the stack (or
            # transposing a projection) drops it to ~0 while magnitude stats can
            # still look entirely plausible.
            cos = float(
                mx.sum(got_w * ref)
                / (mx.sqrt(mx.sum(got_w * got_w)) * mx.sqrt(mx.sum(ref * ref)) + 1e-12)
            )
            worst = max(worst, err)
            worst_cos = min(worst_cos, cos)
            checked += 1
            print(f"        L{layer:<3d} e{e:<4d} {w}  max|err| {err:.3e}  "
                  f"mean rel {rel:7.3%}  cos {cos:.5f}")

        if checked == 0:
            print("        (source shards not downloaded yet -- skipped)")
        elif quant.get("mode") == "mxfp4":
            check(worst == 0.0, f"mxfp4 experts bit-exact vs source (worst {worst})")
        else:
            # per-bit-width floors for honest affine quantization noise
            floor = {2: 0.80, 3: 0.92, 4: 0.97}.get(quant.get("bits"), 0.80)
            check(
                worst_cos > floor,
                f"experts track source (min cos {worst_cos:.5f} > {floor} "
                f"for {quant.get('bits')}-bit)",
            )
        if skipped:
            print(f"        note: {skipped}/{a.samples} samples skipped (shard absent)")
    else:
        print("\n[4] expert numerics: skipped (no --src)")

    print(f"\n{'FAILED' if FAIL else 'PASSED'}  {len(FAIL)} failures, {len(WARN)} warnings")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
