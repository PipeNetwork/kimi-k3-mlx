#!/usr/bin/env python3
"""Streaming Kimi-K3 -> MLX converter.

`mlx_lm convert` is not usable here: it materialises the whole model, and K3 is
2.78T parameters (5.6 TB in bf16). This converter walks the checkpoint one layer
at a time and never holds more than a few tens of GB.

Profiles
--------
  mxfp4   experts copied BYTE-FOR-BYTE from the source (lossless); rest bf16
  3bit    experts affine 3-bit g64; rest 4-bit; router gate 8-bit
  2bit    experts affine 2-bit g64; rest 4-bit; router gate 8-bit
  mixed2  experts affine 2-bit g64; attention/shared/latent/embed 4-bit;
          router gate 8-bit  (the tier most likely to stay coherent at ~2 bits)

The source ships routed experts as MXFP4 (`weight_packed` u8 + e8m0
`weight_scale`, group 32) and everything else as bf16. MLX's native `mxfp4`
quantization mode uses the identical encoding — low-nibble-first codes, e8m0
scales — so the `mxfp4` profile is a reinterpretation, not a requantization:
`tests/test_mxfp4_passthrough.py` asserts bit-exactness against the reference
dequant. Every other profile must dequantize to bf16 first and then affine
quantize, which is a genuine second lossy pass on top of the source's first.

Usage:
  scripts/convert.py --src Kimi-K3-src --out out/Kimi-K3-MLX-mxfp4 --profile mxfp4
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mlx_lm.models import kimi_k3  # noqa: E402

FP4_GROUP = 32
SHARD_BYTES = 5 * 1024**3

# Copied verbatim from the source repo into every output.
PASSTHROUGH_FILES = [
    "tokenizer_config.json", "tokenization_kimi.py", "tiktoken.model",
    "generation_config.json", "preprocessor_config.json", "LICENSE",
    "configuration_kimi_k3.py", "kimi_k3_processor.py",
    "kimi_k3_vision_processing.py", "media_utils.py", "encoding_k3.py",
]


# --------------------------------------------------------------- profiles


def profile_spec(profile: str, nonexpert_bits: Optional[int] = None) -> Dict[str, object]:
    if profile == "mxfp4":
        # Non-experts default to bf16 (114 GB on K3 -- 2% of params but 23% of a
        # REAP'd build's footprint). --nonexpert-bits trades that down; they were
        # never MXFP4, so 8-bit there is a mild first quantization, unlike the
        # experts where anything but mxfp4 is a lossy second pass.
        other = (
            None if nonexpert_bits is None
            else {"mode": "affine", "group_size": 64, "bits": nonexpert_bits}
        )
        return dict(
            expert={"mode": "mxfp4", "group_size": 32, "bits": 4},
            other=other,
            gate=other,
            global_q={"mode": "mxfp4", "group_size": 32, "bits": 4},
        )
    if profile in ("2bit", "3bit"):
        bits = int(profile[0])
        return dict(
            expert={"mode": "affine", "group_size": 64, "bits": bits},
            other={"mode": "affine", "group_size": 64, "bits": 4},
            gate={"mode": "affine", "group_size": 64, "bits": 8},
            global_q={"mode": "affine", "group_size": 64, "bits": bits},
        )
    if profile == "mixed2":
        return dict(
            expert={"mode": "affine", "group_size": 64, "bits": 2},
            other={"mode": "affine", "group_size": 64, "bits": 4},
            gate={"mode": "affine", "group_size": 64, "bits": 8},
            global_q={"mode": "affine", "group_size": 64, "bits": 2},
        )
    raise SystemExit(f"unknown profile {profile!r}")


def classify(path: str) -> str:
    """-> 'expert' | 'gate' | 'other' | 'skip' for a *remapped* module path."""
    if path.endswith("block_sparse_moe.gate"):
        return "gate"
    if ".switch_mlp." in path:
        return "expert"
    if path.endswith(
        (
            "input_layernorm", "post_attention_layernorm", "norm", "o_norm",
            "kv_a_layernorm", "q_a_layernorm", "routed_expert_norm",
        )
    ):
        return "skip"
    if path.endswith(("q_conv.conv", "k_conv.conv", "v_conv.conv")):
        return "skip"  # Conv1d
    if path.endswith(("self_attention_res", "mlp_res", "output_attn_res")):
        return "skip"  # rank-1 AttnRes score directions -> keep full precision
    return "other"


# ----------------------------------------------------------- source access


class ShardReader:
    """mmap-backed shard access with a tiny LRU so we touch each file once."""

    def __init__(self, src: str, index: Dict[str, str], keep: int = 3):
        self.src = src
        self.index = index
        self.keep = keep
        self._open: Dict[str, Dict[str, mx.array]] = {}
        self._order: List[str] = []

    def _shard(self, name: str) -> Dict[str, mx.array]:
        if name not in self._open:
            self._open[name] = mx.load(os.path.join(self.src, name))
            self._order.append(name)
            while len(self._order) > self.keep:
                old = self._order.pop(0)
                self._open.pop(old, None)
                gc.collect()
        return self._open[name]

    def get(self, key: str) -> mx.array:
        return self._shard(self.index[key])[key]

    def has(self, key: str) -> bool:
        return key in self.index


# ------------------------------------------------------------- output writer


class ShardWriter:
    def __init__(self, out: str, limit: int = SHARD_BYTES):
        self.out = out
        self.limit = limit
        self.buf: Dict[str, mx.array] = {}
        self.nbytes = 0
        self.shards: List[Tuple[str, List[str]]] = []
        self.weight_map: Dict[str, str] = {}
        self.total = 0
        os.makedirs(out, exist_ok=True)

    def add(self, key: str, val: mx.array):
        mx.eval(val)
        self.buf[key] = val
        self.nbytes += val.nbytes
        self.total += val.nbytes
        if self.nbytes >= self.limit:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        name = f"model-{len(self.shards) + 1:05d}.safetensors"
        mx.save_safetensors(os.path.join(self.out, name), self.buf, {"format": "mlx"})
        for k in self.buf:
            self.weight_map[k] = name
        self.shards.append((name, list(self.buf)))
        self.buf, self.nbytes = {}, 0
        gc.collect()

    def finalize(self):
        self.flush()
        n = len(self.shards)
        # rename to the conventional -of-NNNNN form now that the count is known
        remap = {}
        for i, (old, keys) in enumerate(self.shards, 1):
            new = f"model-{i:05d}-of-{n:05d}.safetensors"
            os.rename(os.path.join(self.out, old), os.path.join(self.out, new))
            remap[old] = new
        self.weight_map = {k: remap[v] for k, v in self.weight_map.items()}
        with open(os.path.join(self.out, "model.safetensors.index.json"), "w") as f:
            json.dump(
                {"metadata": {"total_size": self.total}, "weight_map": self.weight_map},
                f,
                indent=2,
            )


# ------------------------------------------------------------- quantization


def quantizable(w: mx.array, spec: Optional[Dict]) -> bool:
    """MLX needs >=2 dims with the last divisible by group_size."""
    return (
        spec is not None
        and w.ndim >= 2
        and w.shape[-1] % spec["group_size"] == 0
    )


def quantize_tensor(w: mx.array, spec: Optional[Dict]) -> Dict[str, mx.array]:
    """-> {'weight':..., 'scales':..., ['biases':...]} or {'weight': w} if bf16.

    Falls back to bf16 rather than raising when a tensor cannot be grouped;
    mlx-lm's loader keys off the presence of `<path>.scales`, so an unquantized
    tensor round-trips correctly with no config entry.
    """
    if not quantizable(w, spec):
        return {"weight": w.astype(mx.bfloat16)}
    res = mx.quantize(
        w, group_size=spec["group_size"], bits=spec["bits"], mode=spec["mode"]
    )
    if len(res) == 3:
        return {"weight": res[0], "scales": res[1], "biases": res[2]}
    return {"weight": res[0], "scales": res[1]}


def expert_stack_passthrough(
    reader: ShardReader, layer: int, w: str, n_experts, keep=None
) -> Dict[str, mx.array]:
    """MXFP4 -> MLX mxfp4 with no arithmetic: reinterpret bytes, stack, done."""
    base = f"language_model.model.layers.{layer}.block_sparse_moe.experts"
    order = keep if keep is not None else range(n_experts)
    packed, scales = [], []
    for e in order:
        packed.append(reader.get(f"{base}.{e}.{w}.weight_packed").view(mx.uint32))
        scales.append(reader.get(f"{base}.{e}.{w}.weight_scale"))
    out = {"weight": mx.stack(packed), "scales": mx.stack(scales)}
    mx.eval(out["weight"], out["scales"])
    return out


def expert_stack_requant(
    reader: ShardReader, layer: int, w: str, n_experts, spec: Dict, keep=None,
    awq_scale=None
) -> Dict[str, mx.array]:
    """MXFP4 -> bf16 -> affine N-bit, one expert at a time (bounded memory).

    With `awq_scale` (a per-input-channel vector for this layer's shared latent),
    w1/w3 are scaled before quantizing; the inverse is folded into
    routed_expert_down_proj by the caller, so the product is unchanged and only
    the quantization error moves. w2 reads the per-expert intermediate, not the
    shared latent, so it is left alone.
    """
    base = f"language_model.model.layers.{layer}.block_sparse_moe.experts"
    order = keep if keep is not None else range(n_experts)
    qs, ss, bs = [], [], []
    for e in order:
        packed = reader.get(f"{base}.{e}.{w}.weight_packed").view(mx.uint32)
        scale = reader.get(f"{base}.{e}.{w}.weight_scale")
        dense = mx.dequantize(packed, scale, group_size=FP4_GROUP, bits=4, mode="mxfp4")
        if awq_scale is not None and w in ("w1", "w3"):
            dense = dense * awq_scale
        q, s, b = mx.quantize(
            dense, group_size=spec["group_size"], bits=spec["bits"], mode=spec["mode"]
        )
        mx.eval(q, s, b)
        qs.append(q)
        ss.append(s)
        bs.append(b)
        del dense, packed, scale
    out = {"weight": mx.stack(qs), "scales": mx.stack(ss), "biases": mx.stack(bs)}
    mx.eval(*out.values())
    return out


# -------------------------------------------------------------------- main


def load_prune_plan(path: Optional[str], args: kimi_k3.ModelArgs):
    """-> {layer_idx: [kept expert ids, ascending]} or None.

    A REAP plan renumbers experts: new index i is old expert keep[i]. The router
    gate rows and e_score_correction_bias must be reordered to match, or the
    model routes to the wrong experts while looking structurally perfect.
    """
    if not path:
        return None, None
    plan = json.load(open(path))
    top_k = args.num_experts_per_token
    keep = {}
    for k, v in plan["layers"].items():
        idx = int(k)
        ids = sorted(int(e) for e in v["keep"])
        if len(ids) < top_k:
            raise SystemExit(
                f"layer {idx} keeps {len(ids)} experts but top-k is {top_k}; "
                f"the router cannot select that many. Raise --min-experts."
            )
        if len(set(ids)) != len(ids) or ids[-1] >= args.num_experts:
            raise SystemExit(f"layer {idx} has duplicate or out-of-range expert ids")
        keep[idx] = ids

    # Graded plans: experts within a layer carry two different bit widths, held
    # by kimi_k3.TwoBankSwitchGLU. Experts are reordered so the high-precision
    # bank is the leading contiguous range, which is what lets the two-bank
    # forward split the sorted routing pairs with a single slice.
    banks = {}
    for k, v in plan["layers"].items():
        idx = int(k)
        bits = v.get("bits", {})
        widths = set(bits.values())
        if len(widths) <= 1:
            banks[idx] = None
            continue
        if len(widths) > 2:
            raise SystemExit(
                f"layer {idx} uses {len(widths)} bit widths; TwoBankSwitchGLU "
                f"supports exactly two"
            )
        hi = sorted(int(e) for e, b in bits.items() if b == "mxfp4")
        lo = sorted(int(e) for e, b in bits.items() if b != "mxfp4")
        if not hi or not lo:
            raise SystemExit(f"layer {idx}: graded needs a non-empty mxfp4 tier")
        if sorted(hi + lo) != keep[idx]:
            raise SystemExit(
                f"layer {idx}: bits cover {len(hi)+len(lo)} experts but keep "
                f"lists {len(keep[idx])}"
            )
        lo_bits = {b for b in bits.values() if b != "mxfp4"}
        if len(lo_bits) != 1:
            raise SystemExit(f"layer {idx}: low tier mixes {lo_bits}")
        # hi first -> new indices [0, len(hi)) are the high-precision bank
        keep[idx] = hi + lo
        banks[idx] = (len(hi), lo_bits.pop())
    return keep, banks


def convert(
    src: str,
    out: str,
    profile: str,
    limit_layers: Optional[int] = None,
    prune_plan: Optional[str] = None,
    nonexpert_bits: Optional[int] = None,
    awq_scales: Optional[str] = None,
):
    spec = profile_spec(profile, nonexpert_bits)
    raw_cfg = json.load(open(os.path.join(src, "config.json")))
    args = kimi_k3.ModelArgs.from_dict(raw_cfg)
    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    awq = None
    if awq_scales:
        _z = __import__("numpy").load(awq_scales)
        awq = mx.array(_z["scales"])
        print(f"[{profile}] AWQ: per-layer input scales, alpha mean "
              f"{float(_z['alpha'].mean()):.2f}")
    keep_map, bank_map = load_prune_plan(prune_plan, args)
    if awq is not None and bank_map:
        # AWQ is an identity only if EVERY consumer of the latent is scaled: the
        # down_proj is divided by s once per layer, so an expert whose w1/w3 was
        # not multiplied by s silently reads latent/s. The graded hi bank is a
        # byte copy of the source MXFP4 -- scaling it would destroy the
        # bit-exactness that is its entire reason to exist. So the two are
        # mutually exclusive, and s ~ 1 +/- 5% means the corruption would degrade
        # quietly rather than fail loudly. Refuse instead.
        raise SystemExit(
            "--awq-scales cannot be combined with a graded (two-bank) plan: the "
            "mxfp4 hi bank is passed through unscaled, so folding 1/s into "
            "down_proj would break the identity for exactly those experts. Use a "
            "uniform lossy profile (2bit/mixed2) for AWQ."
        )
    expert_counts = bank_splits = None
    if keep_map is not None:
        if bank_map and any(bank_map.values()):
            bank_splits = [
                (bank_map[i][0] if bank_map.get(i) else 0)
                for i in range(args.num_hidden_layers)
            ]
            n_graded = sum(1 for v in bank_splits if v)
            print(f"[{profile}] graded: {n_graded} layers use a two-bank "
                  f"switch_mlp (mxfp4 + low tier)")
        expert_counts = [
            len(keep_map[i]) if i in keep_map else 0
            for i in range(args.num_hidden_layers)
        ]
        kept = [c for c in expert_counts if c]
        print(f"[{profile}] REAP: keeping {min(kept)}-{max(kept)} of {args.num_experts} "
              f"experts per layer (mean {sum(kept)/len(kept):.0f}, "
              f"{sum(kept)/(len(kept)*args.num_experts):.1%})")

    reader = ShardReader(src, index)
    writer = ShardWriter(out)
    overrides: Dict[str, Dict] = {}
    n_layers = limit_layers or args.num_hidden_layers
    t0 = time.time()

    def emit(module_path: str, tensors: Dict[str, mx.array]):
        for suffix, val in tensors.items():
            writer.add(f"{module_path}.{suffix}" if suffix else module_path, val)

    def emit_quantized(module_path: str, w: mx.array, kind: str):
        s = spec[kind]
        emit(module_path, quantize_tensor(w, s))
        if quantizable(w, s) and s != spec["global_q"]:
            overrides[module_path] = dict(s)

    # ---- embeddings
    emb = reader.get("language_model.model.embed_tokens.weight")
    emit_quantized("model.embed_tokens", emb, "other")
    del emb

    # ---- layers
    for i in range(n_layers):
        lt0 = time.time()
        prefix = f"language_model.model.layers.{i}"
        raw = {
            k: reader.get(k)
            for k in index
            if k.startswith(prefix + ".") and ".experts." not in k
        }
        mapped = kimi_k3.remap_checkpoint(raw, args, layer_indices=[i], stack_experts=False)
        del raw

        # Renumber the router to the kept experts. gate.weight is
        # (num_experts, hidden) and e_score_correction_bias is (num_experts,);
        # new row i must be old row keep[i]. Miss this and the model loads,
        # runs, and routes every token to the wrong expert.
        if keep_map is not None and i in keep_map:
            sel = mx.array(keep_map[i])
            moe = f"model.layers.{i}.block_sparse_moe"
            # NOTE: remap_checkpoint has already dropped the `.gate.` segment
            # from the correction bias. These are asserted rather than guarded
            # with `if key in mapped` -- a silent no-op here does not fail, it
            # ships a model that routes every token to the wrong expert.
            for key in (f"{moe}.gate.weight", f"{moe}.e_score_correction_bias"):
                assert key in mapped, f"router tensor {key} missing; cannot renumber"
                mapped[key] = mx.take(mapped[key], sel, axis=0)

        if awq is not None and kimi_k3.is_moe_layer(args, i):
            dk = f"model.layers.{i}.block_sparse_moe.routed_expert_down_proj.weight"
            if dk in mapped:
                # down_proj: (latent, hidden). Its row j produces latent channel
                # j, which is exactly the channel w1/w3 scaled by s_j -- so
                # dividing row j by s_j makes the pair an exact identity.
                mapped[dk] = (mapped[dk].astype(mx.float32)
                              / awq[i].reshape(-1, 1)).astype(mx.bfloat16)

        for key, val in mapped.items():
            assert key.endswith(".weight") or key.endswith((".proj_weight", ".norm_weight", ".A_log", ".dt_bias", ".e_score_correction_bias")), key
            if key.endswith(".weight"):
                module_path, suffix = key[: -len(".weight")], "weight"
            else:
                module_path, suffix = key.rsplit(".", 1)[0], key.rsplit(".", 1)[1]

            kind = classify(module_path) if suffix == "weight" else "skip"
            if kind == "skip":
                writer.add(key, val.astype(mx.bfloat16) if val.dtype == mx.float32 and suffix == "weight" else val)
            else:
                emit_quantized(module_path, val, kind)
        del mapped

        # ---- routed experts (the 97.9%)
        if kimi_k3.is_moe_layer(args, i):
            moe = f"model.layers.{i}.block_sparse_moe.switch_mlp"
            keep = keep_map[i] if keep_map is not None else None
            bank = bank_map.get(i) if bank_map else None
            for w, dst in (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")):
                es = spec["expert"]
                if bank is None:
                    if es["mode"] == "mxfp4":
                        t = expert_stack_passthrough(reader, i, w, args.num_experts, keep)
                    else:
                        t = expert_stack_requant(reader, i, w, args.num_experts, es,
                                                 keep, awq[i] if awq is not None else None)
                    emit(f"{moe}.{dst}", t)
                    if es != spec["global_q"]:
                        overrides[f"{moe}.{dst}"] = dict(es)
                    del t
                else:
                    # two banks: leading `n_hi` kept experts at mxfp4 (a byte
                    # copy of the source), the remainder requantized.
                    n_hi, lo_name = bank
                    lo_spec = {"mode": "affine", "group_size": 64,
                               "bits": int(lo_name[0])}
                    hi_t = expert_stack_passthrough(reader, i, w, args.num_experts,
                                                    keep[:n_hi])
                    lo_t = expert_stack_requant(reader, i, w, args.num_experts,
                                                lo_spec, keep[n_hi:],
                                                awq[i] if awq is not None else None)
                    emit(f"{moe}.bank_hi.{dst}", hi_t)
                    emit(f"{moe}.bank_lo.{dst}", lo_t)
                    overrides[f"{moe}.bank_hi.{dst}"] = {
                        "mode": "mxfp4", "group_size": 32, "bits": 4}
                    overrides[f"{moe}.bank_lo.{dst}"] = dict(lo_spec)
                    del hi_t, lo_t
                gc.collect()

        print(
            f"[{profile}] layer {i:3d}/{n_layers - 1}  "
            f"{time.time() - lt0:6.1f}s  written {writer.total / 1e9:8.1f} GB  "
            f"elapsed {(time.time() - t0) / 60:6.1f}m",
            flush=True,
        )

    # ---- vision tower + projector
    #
    # Copied verbatim in bf16. 447 M params (~0.9 GB) against a ~1 TB artifact,
    # so quantizing them would save nothing measurable while putting the only
    # visual pathway at risk. Keys stay in source form: kimi_k3.Model.sanitize
    # drops them (mlx-lm is text-only) and kimi_k3_vision.VisionModel.sanitize
    # consumes them, so there is exactly one place that maps these names.
    vis_keys = sorted(
        (k for k in index if k.startswith(("vision_tower.", "mm_projector."))),
        key=lambda k: (index[k], k),  # group by shard so the LRU sees each once
    )
    n_vis = 0
    for key in vis_keys:
        writer.add(key, reader.get(key).astype(mx.bfloat16))
        n_vis += 1
    print(f"[{profile}] vision tower: {n_vis} tensors copied bf16", flush=True)

    # ---- tail
    for key in (
        "language_model.model.norm.weight",
        "language_model.model.output_attn_res_proj.weight",
        "language_model.model.output_attn_res_norm.weight",
    ):
        if reader.has(key):
            m = kimi_k3.remap_checkpoint({key: reader.get(key)}, args, layer_indices=[])
            for k, v in m.items():
                writer.add(k, v)
    if not args.tie_word_embeddings:
        emit_quantized("lm_head", reader.get("language_model.lm_head.weight"), "other")

    writer.finalize()

    # ---- config + aux files
    cfg = dict(raw_cfg)
    cfg.pop("quantization_config", None)
    if "text_config" in cfg:
        cfg["text_config"] = dict(cfg["text_config"])
        cfg["text_config"].pop("quantization_config", None)
    q = dict(spec["global_q"])
    q.update(overrides)
    cfg["quantization"] = q
    if expert_counts is not None:
        tc = cfg.setdefault("text_config", cfg)
        tc["expert_counts"] = expert_counts
        # num_experts is the widest layer; per-layer widths come from
        # expert_counts, which kimi_k3.experts_in_layer() reads.
        tc["num_experts"] = max(expert_counts)
        if bank_splits is not None:
            tc["expert_bank_split"] = bank_splits
        cfg["reap"] = {
            "plan": os.path.basename(prune_plan),
            "source_num_experts": args.num_experts,
            "kept_mean": sum(c for c in expert_counts if c) / sum(1 for c in expert_counts if c),
            # new expert index i == source expert keep[i]. Recorded so the
            # artifact is self-describing: without it nothing downstream can
            # check a converted expert against the source it came from.
            "keep": {str(k): v for k, v in sorted(keep_map.items())},
        }
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    here = os.path.dirname(__file__)
    for mod in ("kimi_k3.py", "kimi_k3_vision.py"):
        shutil.copy(os.path.join(here, "..", mod), os.path.join(out, mod))
    # mlx-vlm wrapper package (text tower + vision tower + placeholder merge)
    vl_src = os.path.join(here, "..", "kimi_k3_vl")
    vl_dst = os.path.join(out, "kimi_k3_vl")
    shutil.rmtree(vl_dst, ignore_errors=True)
    shutil.copytree(vl_src, vl_dst, ignore=shutil.ignore_patterns("__pycache__"))
    for name in PASSTHROUGH_FILES:
        p = os.path.join(src, name)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(out, name))

    print(f"[{profile}] DONE  {writer.total / 1e12:.3f} TB  "
          f"{len(writer.shards)} shards  {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", required=True,
                    choices=["mxfp4", "3bit", "2bit", "mixed2"])
    ap.add_argument("--limit-layers", type=int, default=None,
                    help="convert only the first N layers (smoke testing)")
    ap.add_argument("--nonexpert-bits", type=int, default=None,
                    help="quantize non-expert tensors to N bits (mxfp4 profile "
                         "keeps them bf16 by default: 114 GB on K3)")
    ap.add_argument("--awq-scales", default=None,
                    help="npz from scripts/awq.py; scales w1/w3 input channels "
                         "and folds the inverse into routed_expert_down_proj")
    ap.add_argument("--prune-plan", default=None,
                    help="REAP plan from scripts/reap_plan.py; prunes experts "
                         "and renumbers the router to match")
    a = ap.parse_args()
    convert(a.src, a.out, a.profile, a.limit_layers, a.prune_plan, a.nonexpert_bits, a.awq_scales)
