"""End-to-end converter test on a synthetic mini-K3 in the *source* format.

Nothing about the real 2.78T checkpoint is loadable on a 512 GB machine, so this
builds a structurally faithful miniature — `language_model.*` key prefixes,
routed experts stored as MXFP4 `weight_packed`/`weight_scale` pairs, everything
else bf16, sharded with an index.json — and pushes it through the full pipeline:

    fixture -> scripts/convert.py -> mlx_lm.load -> forward pass

It asserts the thing that actually matters for the mxfp4 tier: the experts come
out the far end *bit-identical* to the source, so that profile adds no error on
top of Moonshot's own quantization.

Run:  python3 tests/test_convert_roundtrip.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mlx_lm.models import kimi_k3  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONVERT = os.path.join(ROOT, "scripts", "convert.py")

HID, LAT, MOE_I, NE, NHEAD, HDIM = 128, 64, 64, 8, 4, 64
NLAYERS, VOCAB = 4, 128
KDA_LAYERS = [1, 2, 4]  # 1-based -> tensor layers 0,1,3 ; layer 2 (idx 2) is MLA
MERGE_K = 2


def tiny_config():
    return {
        "model_type": "kimi_k3",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "text_config": {
            "vocab_size": VOCAB,
            "hidden_size": HID,
            "num_hidden_layers": NLAYERS,
            "num_attention_heads": NHEAD,
            "num_key_value_heads": NHEAD,
            "intermediate_size": 128,
            "rms_norm_eps": 1e-5,
            "linear_attn_config": {
                "head_dim": HDIM,
                "num_heads": NHEAD,
                "short_conv_kernel_size": 4,
                "kda_layers": KDA_LAYERS,
                "full_attn_layers": [3],
                "use_full_rank_gate": True,
                "gate_lower_bound": -5.0,
            },
            "num_experts": NE,
            "moe_intermediate_size": MOE_I,
            "kv_lora_rank": 16,
            "q_lora_rank": 24,
            "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
            "mla_use_nope": True,
            "mla_use_output_gate": True,
            "num_experts_per_token": 2,
            "num_shared_experts": 2,
            "first_k_dense_replace": 1,
            "hidden_act": "situ",
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "attn_res_block_size": 2,
            "routed_expert_hidden_size": LAT,
            "latent_moe_use_norm": True,
            "tie_word_embeddings": False,
            "moe_layer_freq": 1,
            "num_expert_group": 1,
            "topk_group": 1,
            "routed_scaling_factor": 1.0,
            "moe_router_activation_func": "sigmoid",
            "moe_renormalize": True,
        },
        "vision_config": {
            "vt_hidden_size": 32, "vt_intermediate_size": 64, "vt_num_hidden_layers": 2,
            "vt_num_attention_heads": 2, "qkv_hidden_size": 16, "patch_size": 4,
            "init_pos_emb_height": 8, "init_pos_emb_width": 8, "init_pos_emb_time": 4,
            "merge_kernel_size": [MERGE_K, MERGE_K], "mm_hidden_size": 32,
            "text_hidden_size": HID, "projector_ln_eps": 1e-5,
            "pos_emb_interpolation_mode": "bilinear", "merge_type": "sd2_tpool",
            "mm_projector_type": "patchmergerv2", "norm_type": "rmsnorm",
        },
    }


KE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def make_mxfp4(rng, out_dim, in_dim):
    """Emit a random but *valid* MXFP4 tensor pair, plus its exact dense value."""
    assert in_dim % 32 == 0
    codes = rng.integers(0, 16, size=(out_dim, in_dim), dtype=np.uint8)
    scale = rng.integers(120, 132, size=(out_dim, in_dim // 32), dtype=np.uint8)
    lo, hi = codes[:, 0::2], codes[:, 1::2]
    packed = (lo | (hi << 4)).astype(np.uint8)
    vals = KE[(codes & 0x07)] * np.where((codes & 0x08).astype(bool), -1.0, 1.0)
    exp = (scale.astype(np.int32) - 127).repeat(32, axis=1)
    dense = (vals * np.exp2(exp.astype(np.float32))).astype(np.float32)
    return packed, scale, dense


def build_fixture(path):
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(0)
    cfg = tiny_config()
    tc = cfg["text_config"]
    W, dense_experts = {}, {}

    def bf16(*shape, scale=0.02):
        return mx.array((rng.standard_normal(shape) * scale).astype(np.float32)).astype(mx.bfloat16)

    P = "language_model."
    W[P + "model.embed_tokens.weight"] = bf16(VOCAB, HID)
    W[P + "lm_head.weight"] = bf16(VOCAB, HID)
    W[P + "model.norm.weight"] = bf16(HID, scale=1.0)
    W[P + "model.output_attn_res_proj.weight"] = bf16(1, HID, scale=1.0)
    W[P + "model.output_attn_res_norm.weight"] = bf16(HID, scale=1.0)

    for i in range(NLAYERS):
        lp = f"{P}model.layers.{i}."
        for n in ("input_layernorm", "post_attention_layernorm",
                  "self_attention_res_norm", "mlp_res_norm"):
            W[lp + n + ".weight"] = bf16(HID, scale=1.0)
        for n in ("self_attention_res_proj", "mlp_res_proj"):
            W[lp + n + ".weight"] = bf16(1, HID, scale=1.0)

        pdim = NHEAD * HDIM
        if (i + 1) in KDA_LAYERS:
            for n in ("q_proj", "k_proj", "v_proj", "g_proj"):
                W[lp + f"self_attn.{n}.weight"] = bf16(pdim, HID)
            W[lp + "self_attn.o_proj.weight"] = bf16(HID, pdim)
            W[lp + "self_attn.f_a_proj.weight"] = bf16(HDIM, HID)
            W[lp + "self_attn.f_b_proj.weight"] = bf16(pdim, HDIM)
            W[lp + "self_attn.b_proj.weight"] = bf16(NHEAD, HID)
            for n in ("q_conv1d", "k_conv1d", "v_conv1d"):
                W[lp + f"self_attn.{n}.weight"] = mx.array(
                    (rng.standard_normal((pdim, 1, 4)) * 0.1).astype(np.float32))
            W[lp + "self_attn.A_log"] = mx.array(
                np.log(rng.uniform(1, 16, HDIM)).astype(np.float32))
            W[lp + "self_attn.dt_bias"] = mx.array(
                rng.standard_normal(pdim).astype(np.float32) * 0.1)
            W[lp + "self_attn.o_norm.weight"] = mx.array(np.ones(HDIM, np.float32))
        else:
            qh = tc["qk_nope_head_dim"] + tc["qk_rope_head_dim"]
            W[lp + "self_attn.q_a_proj.weight"] = bf16(tc["q_lora_rank"], HID)
            W[lp + "self_attn.q_a_layernorm.weight"] = bf16(tc["q_lora_rank"], scale=1.0)
            W[lp + "self_attn.q_b_proj.weight"] = bf16(NHEAD * qh, tc["q_lora_rank"])
            W[lp + "self_attn.kv_a_proj_with_mqa.weight"] = bf16(
                tc["kv_lora_rank"] + tc["qk_rope_head_dim"], HID)
            W[lp + "self_attn.kv_a_layernorm.weight"] = bf16(tc["kv_lora_rank"], scale=1.0)
            W[lp + "self_attn.kv_b_proj.weight"] = bf16(
                NHEAD * (tc["qk_nope_head_dim"] + tc["v_head_dim"]), tc["kv_lora_rank"])
            W[lp + "self_attn.o_proj.weight"] = bf16(HID, NHEAD * tc["v_head_dim"])
            W[lp + "self_attn.g_proj.weight"] = bf16(NHEAD * tc["v_head_dim"], HID)

        if i < tc["first_k_dense_replace"]:
            W[lp + "mlp.gate_proj.weight"] = bf16(tc["intermediate_size"], HID)
            W[lp + "mlp.up_proj.weight"] = bf16(tc["intermediate_size"], HID)
            W[lp + "mlp.down_proj.weight"] = bf16(HID, tc["intermediate_size"])
        else:
            bp = lp + "block_sparse_moe."
            W[bp + "gate.weight"] = bf16(NE, HID)
            W[bp + "gate.e_score_correction_bias"] = mx.array(
                rng.standard_normal(NE).astype(np.float32) * 0.01)
            W[bp + "routed_expert_down_proj.weight"] = bf16(LAT, HID)
            W[bp + "routed_expert_up_proj.weight"] = bf16(HID, LAT)
            W[bp + "routed_expert_norm.weight"] = bf16(LAT, scale=1.0)
            sh = MOE_I * tc["num_shared_experts"]
            W[bp + "shared_experts.gate_proj.weight"] = bf16(sh, HID)
            W[bp + "shared_experts.up_proj.weight"] = bf16(sh, HID)
            W[bp + "shared_experts.down_proj.weight"] = bf16(HID, sh)
            for e in range(NE):
                for w, (o, n) in (("w1", (MOE_I, LAT)), ("w3", (MOE_I, LAT)),
                                  ("w2", (LAT, MOE_I))):
                    pk, sc, dn = make_mxfp4(rng, o, n)
                    W[bp + f"experts.{e}.{w}.weight_packed"] = mx.array(pk)
                    W[bp + f"experts.{e}.{w}.weight_scale"] = mx.array(sc)
                    dense_experts[f"{i}.{e}.{w}"] = dn

    # vision tower + projector, in source key form (converter copies verbatim)
    vc = cfg["vision_config"]
    VH, VQ, VI, VL = vc["vt_hidden_size"], vc["qkv_hidden_size"], vc["vt_intermediate_size"], vc["vt_num_hidden_layers"]
    W["vision_tower.patch_embed.proj.weight"] = bf16(VH, 3, vc["patch_size"], vc["patch_size"])
    W["vision_tower.patch_embed.pos_emb.weight"] = bf16(
        vc["init_pos_emb_height"], vc["init_pos_emb_width"], VH, scale=1.0)
    W["vision_tower.encoder.final_layernorm.weight"] = bf16(VH, scale=1.0)
    for i in range(VL):
        b = f"vision_tower.encoder.blocks.{i}"
        W[f"{b}.wqkv.weight"] = bf16(VQ * 3, VH)
        W[f"{b}.wo.weight"] = bf16(VH, VQ)
        W[f"{b}.mlp.fc0.weight"] = bf16(VI, VH)
        W[f"{b}.mlp.fc1.weight"] = bf16(VH, VI)
        W[f"{b}.norm0.weight"] = bf16(VH, scale=1.0)
        W[f"{b}.norm1.weight"] = bf16(VH, scale=1.0)
    inner = vc["mm_hidden_size"] * MERGE_K * MERGE_K
    W["mm_projector.proj.0.weight"] = bf16(inner, inner)
    W["mm_projector.proj.2.weight"] = bf16(HID, inner)
    W["mm_projector.post_norm.weight"] = bf16(HID, scale=1.0)

    # shard it, like the real repo
    keys = list(W)
    per = max(1, len(keys) // 3)
    weight_map, total = {}, 0
    chunks = [keys[i:i + per] for i in range(0, len(keys), per)]
    for si, chunk in enumerate(chunks, 1):
        name = f"model-{si:05d}-of-{len(chunks):05d}.safetensors"
        mx.save_safetensors(os.path.join(path, name), {k: W[k] for k in chunk})
        for k in chunk:
            weight_map[k] = name
            total += W[k].nbytes
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
              open(os.path.join(path, "model.safetensors.index.json"), "w"))
    json.dump(cfg, open(os.path.join(path, "config.json"), "w"), indent=2)
    return dense_experts


class TestConverterRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="k3fix-")
        cls.src = os.path.join(cls.tmp, "src")
        cls.dense_experts = build_fixture(cls.src)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _convert(self, profile):
        out = os.path.join(self.tmp, profile)
        r = subprocess.run(
            # --overwrite: several tests convert the same profile into the same
            # path, and the converter now refuses to write over a build unasked.
            [sys.executable, CONVERT, "--src", self.src, "--out", out,
             "--profile", profile, "--overwrite"],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return out

    def test_mxfp4_experts_are_bit_exact(self):
        out = self._convert("mxfp4")
        cfg = json.load(open(os.path.join(out, "config.json")))
        self.assertEqual(cfg["quantization"]["mode"], "mxfp4")
        self.assertEqual(cfg["quantization"]["group_size"], 32)
        self.assertNotIn("quantization_config", cfg)

        idx = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]
        shards = {}
        for k, sh in idx.items():
            shards.setdefault(sh, []).append(k)
        W = {}
        for sh in shards:
            W.update(mx.load(os.path.join(out, sh)))

        # experts must dequantize to exactly the source values
        worst = 0.0
        for layer in range(1, NLAYERS):
            for w, dst in (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")):
                p = f"model.layers.{layer}.block_sparse_moe.switch_mlp.{dst}"
                deq = mx.dequantize(W[p + ".weight"], W[p + ".scales"],
                                    group_size=32, bits=4, mode="mxfp4")
                for e in range(NE):
                    ref = self.dense_experts[f"{layer}.{e}.{w}"]
                    err = float(mx.abs(deq[e].astype(mx.float32) - mx.array(ref)).max())
                    worst = max(worst, err)
        self.assertEqual(worst, 0.0, f"mxfp4 passthrough is lossy: max err {worst}")

    def test_all_profiles_load_and_run(self):
        results = {}
        for profile in ("mxfp4", "2bit", "3bit", "mixed2"):
            out = self._convert(profile)
            model = self._load_model_only(out)
            toks = mx.array([[5, 9, 17, 33, 4, 8]])
            logits = model(toks)
            mx.eval(logits)
            self.assertEqual(logits.shape, (1, 6, VOCAB))
            self.assertFalse(bool(mx.any(mx.isnan(logits))), f"{profile} produced NaNs")
            results[profile] = logits
            size = sum(
                os.path.getsize(os.path.join(out, f))
                for f in os.listdir(out) if f.endswith(".safetensors")
            )
            print(f"  {profile:7s} {size / 1e6:8.2f} MB  logits ok")

        # mxfp4 is lossless on experts, so it should track a plain bf16 reference
        # far more closely than the affine tiers do.
        ref = results["mxfp4"]
        for p in ("2bit", "3bit", "mixed2"):
            d = float(mx.abs(results[p] - ref).mean())
            print(f"  mean |{p} - mxfp4| = {d:.4f}")

    def test_profiles_are_pairwise_distinct(self):
        """No two profiles may describe the same quantization.

        `2bit` and `mixed2` were byte-identical: both quantized experts to 2-bit
        and non-experts to 4-bit, so build_all.sh wrote the same ~880 GB model
        twice under two names while the README documented them as separate
        tiers at different bit widths. Nothing structural catches that -- both
        builds convert, verify and load perfectly -- so it is asserted here.
        """
        # the other tests shell out to convert.py; this one needs the specs
        # themselves, so import it directly.
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import convert
        specs = {p: convert.profile_spec(p)
                 for p in ("mxfp4", "2bit", "3bit", "mixed2")}
        seen = {}
        for name, spec in specs.items():
            key = repr(sorted((k, repr(v)) for k, v in spec.items()))
            self.assertNotIn(
                key, seen,
                f"profiles {seen.get(key)!r} and {name!r} are identical: "
                f"{spec}. Two names for one tier wastes a full build."
            )
            seen[key] = name

        # and the specific distinction that was missing: mixed2 keeps
        # non-experts at full precision, 2bit does not.
        self.assertIsNone(specs["mixed2"]["other"],
                          "mixed2 must leave non-experts bf16")
        self.assertEqual(specs["2bit"]["other"]["bits"], 4)
        self.assertEqual(specs["2bit"]["expert"]["bits"],
                         specs["mixed2"]["expert"]["bits"],
                         "both are 2-bit-expert tiers; only non-experts differ")

    def test_vision_survives_conversion_and_loads(self):
        """Vision tensors must reach the output in bf16 and load into VisionModel.

        The text model deliberately drops them (mlx-lm is text-only), so without
        this the tower could silently vanish while every other test still passed.
        """
        import kimi_k3_vision as mlxv

        out = self._convert("2bit")
        idx = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]
        vis_keys = [k for k in idx if k.startswith(("vision_tower.", "mm_projector."))]
        vc = tiny_config()["vision_config"]
        expected = 3 + 1 + 2 + vc["vt_num_hidden_layers"] * 6  # patch(2)+final_ln+proj(3)
        self.assertEqual(len(vis_keys), expected, sorted(vis_keys))

        W = {}
        for sh in sorted(set(idx.values())):
            W.update(mx.load(os.path.join(out, sh)))
        for k in vis_keys:
            self.assertEqual(W[k].dtype, mx.bfloat16, f"{k} not bf16")
            self.assertNotIn(k + ".scales", W, f"{k} was quantized")

        model = mlxv.VisionModel(mlxv.VisionConfig.from_dict(vc))
        model.load_weights(list(model.sanitize(W).items()))
        mx.eval(model.parameters())

        grids = [(1, 4, 4), (2, 4, 4)]
        n = sum(t * h * w for t, h, w in grids)
        px = mx.random.normal((n, 3, vc["patch_size"], vc["patch_size"]))
        feats = model(px, grids)
        mx.eval(feats)
        self.assertEqual(len(feats), 2)
        for f in feats:
            self.assertEqual(f.shape, (4, HID))  # (4/2)*(4/2) tokens -> text dim
            self.assertFalse(bool(mx.any(mx.isnan(f))))
        self.assertTrue(os.path.exists(os.path.join(out, "kimi_k3_vision.py")))
        for f in ("__init__.py", "config.py", "kimi_k3_vl.py", "language.py", "vision.py"):
            self.assertTrue(os.path.exists(os.path.join(out, "kimi_k3_vl", f)), f)

    def _load_model_only(self, path):
        """mlx_lm.load also wants a tokenizer; the fixture has none."""
        import pathlib

        from mlx_lm.utils import load_model

        model, _config = load_model(pathlib.Path(path), lazy=False)
        return model


if __name__ == "__main__":
    unittest.main(verbosity=2)
