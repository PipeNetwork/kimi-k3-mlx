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


def profile_spec(profile: str) -> Dict[str, object]:
    if profile == "mxfp4":
        return dict(
            expert={"mode": "mxfp4", "group_size": 32, "bits": 4},
            other=None,  # bf16
            gate=None,
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
    reader: ShardReader, layer: int, w: str, n_experts: int
) -> Dict[str, mx.array]:
    """MXFP4 -> MLX mxfp4 with no arithmetic: reinterpret bytes, stack, done."""
    base = f"language_model.model.layers.{layer}.block_sparse_moe.experts"
    packed, scales = [], []
    for e in range(n_experts):
        packed.append(reader.get(f"{base}.{e}.{w}.weight_packed").view(mx.uint32))
        scales.append(reader.get(f"{base}.{e}.{w}.weight_scale"))
    out = {"weight": mx.stack(packed), "scales": mx.stack(scales)}
    mx.eval(out["weight"], out["scales"])
    return out


def expert_stack_requant(
    reader: ShardReader, layer: int, w: str, n_experts: int, spec: Dict
) -> Dict[str, mx.array]:
    """MXFP4 -> bf16 -> affine N-bit, one expert at a time (bounded memory)."""
    base = f"language_model.model.layers.{layer}.block_sparse_moe.experts"
    qs, ss, bs = [], [], []
    for e in range(n_experts):
        packed = reader.get(f"{base}.{e}.{w}.weight_packed").view(mx.uint32)
        scale = reader.get(f"{base}.{e}.{w}.weight_scale")
        dense = mx.dequantize(packed, scale, group_size=FP4_GROUP, bits=4, mode="mxfp4")
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


def convert(src: str, out: str, profile: str, limit_layers: Optional[int] = None):
    spec = profile_spec(profile)
    raw_cfg = json.load(open(os.path.join(src, "config.json")))
    args = kimi_k3.ModelArgs.from_dict(raw_cfg)
    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]

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
            for w, dst in (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")):
                es = spec["expert"]
                if es["mode"] == "mxfp4":
                    t = expert_stack_passthrough(reader, i, w, args.num_experts)
                else:
                    t = expert_stack_requant(reader, i, w, args.num_experts, es)
                emit(f"{moe}.{dst}", t)
                if es != spec["global_q"]:
                    overrides[f"{moe}.{dst}"] = dict(es)
                del t
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
    a = ap.parse_args()
    convert(a.src, a.out, a.profile, a.limit_layers)
