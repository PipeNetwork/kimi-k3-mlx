"""Numerical parity: kimi_k3_vision.py (MLX) vs Moonshot's reference torch code.

This runs the *actual* `modeling_kimi_k3.py` from the model repo on CPU, with
identical random weights, and compares every stage of the vision tower. `fla` is
stubbed out because it is a CUDA/Triton package needed only by the text tower's
KDA kernels, which this test never touches.

The vision tower is 0.49 B params -- small enough to test at real width if we
wanted, and tiny at the reduced dims used here -- so unlike the 2.78T text tower
this part of the port can be validated end to end for real.

Run:  python3 tests/test_vision_parity.py
"""

import os
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---- stub `fla` before importing the reference package -----------------------
for name in ("fla", "fla.modules", "fla.ops", "fla.ops.kda", "fla.ops.utils",
             "fla.ops.utils.index", "fla.utils"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["fla.modules"].FusedRMSNormGated = object
sys.modules["fla.modules"].ShortConvolution = object
sys.modules["fla.ops.kda"].chunk_kda = None
sys.modules["fla.ops.kda"].fused_recurrent_kda = None
sys.modules["fla.ops.utils.index"].prepare_cu_seqlens_from_mask = None
sys.modules["fla.ops.utils.index"].prepare_lens_from_mask = None
sys.modules["fla.utils"].tensor_cache = lambda f: f

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import mlx.core as mx  # noqa: E402

# make reference/ importable as a package
REF_DIR = os.path.join(os.path.dirname(__file__), "..", "reference")
_pkg = types.ModuleType("k3ref")
_pkg.__path__ = [os.path.abspath(REF_DIR)]
sys.modules["k3ref"] = _pkg

# The reference targets transformers 4.56; 5.x moved several private helpers, so
# modeling_kimi_linear (text tower) no longer imports. Stub it -- the vision
# classes we exercise never reference it.
_txt = types.ModuleType("k3ref.modeling_kimi_linear")
_txt.KimiLinearForCausalLM = object
sys.modules["k3ref.modeling_kimi_linear"] = _txt

from k3ref import modeling_kimi_k3 as ref  # noqa: E402

import kimi_k3_vision as mlxv  # noqa: E402

# torch.compile is semantics-preserving; drop it so the test runs on CPU without
# a compiler backend. F.interpolate itself is the ground truth we care about.
ref.get_rope_shape = lambda org, interpolation_mode, shape: (
    F.interpolate(org.permute(2, 0, 1).unsqueeze(0), size=shape, mode=interpolation_mode)
    .squeeze(0).permute(1, 2, 0).flatten(end_dim=1)
)

HID, QKV, HEADS, INTER, LAYERS = 64, 96, 12, 128, 2
PATCH, POSH, POSW, POST = 4, 8, 8, 4
TEXT_HID = 32
MERGE = (2, 2)


def cfg_mlx():
    return mlxv.VisionConfig(
        vt_hidden_size=HID, vt_intermediate_size=INTER, vt_num_hidden_layers=LAYERS,
        vt_num_attention_heads=HEADS, qkv_hidden_size=QKV, patch_size=PATCH,
        init_pos_emb_height=POSH, init_pos_emb_width=POSW, init_pos_emb_time=POST,
        merge_kernel_size=MERGE, mm_hidden_size=HID, text_hidden_size=TEXT_HID,
        pos_emb_interpolation_mode="bilinear", rope_max_size=64,
    )


def t2m(t):
    return mx.array(t.detach().float().numpy())


class TestPrimitives(unittest.TestCase):
    def test_bilinear_matches_torch(self):
        rng = np.random.default_rng(0)
        for (ih, iw), (oh, ow) in (((8, 8), (5, 7)), ((8, 8), (16, 3)),
                                   ((64, 64), (23, 41)), ((8, 8), (8, 8))):
            w = rng.standard_normal((ih, iw, HID)).astype(np.float32)
            ref_out = (
                F.interpolate(torch.from_numpy(w).permute(2, 0, 1).unsqueeze(0),
                              size=(oh, ow), mode="bilinear")
                .squeeze(0).permute(1, 2, 0).numpy()
            )
            got = np.array(mlxv.bilinear_interpolate(mx.array(w), oh, ow))
            err = np.abs(got - ref_out).max()
            self.assertLess(err, 2e-5, f"{ih}x{iw}->{oh}x{ow} err {err}")

    def test_sincos_matches_reference(self):
        want = ref.get_1d_sincos_pos_embed(HID, POST)
        got = np.array(mlxv.sincos_1d(HID, POST))
        self.assertLess(np.abs(got - want).max(), 1e-5)

    def test_rope_freqs_match_reference(self):
        dim = QKV // HEADS
        r = ref.Rope2DPosEmbRepeated(dim, 64, 64)
        grids = [(2, 6, 4), (1, 8, 8)]
        want = r.get_freqs_cis(torch.tensor(grids), device="cpu").numpy()
        got = np.array(mlxv.Rope2DPosEmbRepeated(dim, 64, 64)(grids))
        self.assertEqual(got.shape, want.shape)
        self.assertLess(np.abs(got - want).max(), 1e-5)

    def test_rope_repeats_per_frame(self):
        """t frames must share identical spatial rope -- [K3-V1]."""
        dim = QKV // HEADS
        f = np.array(mlxv.Rope2DPosEmbRepeated(dim, 64, 64)([(3, 4, 4)]))
        self.assertEqual(f.shape[0], 3 * 16)
        self.assertLess(np.abs(f[:16] - f[16:32]).max(), 1e-6)
        self.assertLess(np.abs(f[:16] - f[32:48]).max(), 1e-6)


class TestFullTowerParity(unittest.TestCase):
    """Build both towers, copy weights, compare stage by stage."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.grids = [(1, 8, 6), (2, 4, 4)]
        cls.ntok = sum(t * h * w for t, h, w in cls.grids)

        block_cfg = dict(
            num_heads=HEADS, hidden_dim=HID, qkv_hidden_size=QKV, mlp_dim=INTER,
            norm_type="rmsnorm", mlp_type="mlp2", activation=ref.PytorchGELUTanh(),
            attn_bias=False, linear_bias=False, attn_implementation="eager",
        )
        cls.r_patch = ref.MoonVision3dPatchEmbed(
            out_dim=HID, patch_size=PATCH, pos_emb_height=POSH, pos_emb_width=POSW,
            pos_emb_time=POST, pos_emb_type="divided_fixed",
            patch_embed_proj_bias=False, pos_emb_interpolation_mode="bilinear",
        ).eval()
        cls.r_enc = ref.MoonViT3dEncoder(HID, LAYERS, block_cfg).eval()

        class PCfg:
            projector_ln_eps = 1e-5
            mm_hidden_size = HID
            merge_kernel_size = MERGE
            hidden_size = TEXT_HID
        cls.r_proj = ref.PatchMergerMLPV2(PCfg()).eval()

        # randomise everything (defaults leave several norms at exactly 1.0,
        # which would hide weight-mapping mistakes)
        for m in (cls.r_patch, cls.r_enc, cls.r_proj):
            for p in m.parameters():
                with torch.no_grad():
                    p.copy_(torch.randn_like(p) * 0.1)

        cls.m = mlxv.VisionModel(cfg_mlx())
        w = {}
        sd = cls.r_patch.state_dict()
        w["vision_tower.patch_embed.proj.weight"] = t2m(sd["proj.weight"])
        w["vision_tower.patch_embed.pos_emb.weight"] = t2m(sd["pos_emb.weight"])
        for k, v in cls.r_enc.state_dict().items():
            w[f"vision_tower.encoder.{k}"] = t2m(v)
        for k, v in cls.r_proj.state_dict().items():
            w[f"mm_projector.{k}"] = t2m(v)
        cls.m.load_weights(list(cls.m.sanitize(w).items()))
        mx.eval(cls.m.parameters())

        cls.pixels = torch.randn(cls.ntok, 3, PATCH, PATCH)
        cls.grid_t = torch.tensor(cls.grids)

    def test_patch_embed(self):
        want = self.r_patch(self.pixels, self.grid_t).detach().numpy()
        got = np.array(self.m.patch_embed(t2m(self.pixels), self.grids))
        self.assertEqual(got.shape, want.shape)
        err = np.abs(got - want).max() / (np.abs(want).max() + 1e-9)
        self.assertLess(err, 2e-4, f"patch_embed rel err {err}")

    def test_encoder(self):
        h = self.r_patch(self.pixels, self.grid_t)
        want = self.r_enc(h, self.grid_t).detach().numpy()
        got = np.array(self.m.encoder(t2m(h), self.grids))
        self.assertEqual(got.shape, want.shape)
        err = np.abs(got - want).max() / (np.abs(want).max() + 1e-9)
        self.assertLess(err, 5e-3, f"encoder rel err {err}")

    def test_merger(self):
        h = self.r_enc(self.r_patch(self.pixels, self.grid_t), self.grid_t)
        want = ref.tpool_patch_merger(h, self.grid_t, MERGE)
        got = mlxv.tpool_patch_merger(t2m(h), self.grids, MERGE)
        self.assertEqual(len(got), len(want))
        for g, wv in zip(got, want):
            self.assertEqual(tuple(g.shape), tuple(wv.shape))
            self.assertLess(np.abs(np.array(g) - wv.detach().numpy()).max(), 2e-4)

    def test_end_to_end(self):
        h = self.r_enc(self.r_patch(self.pixels, self.grid_t), self.grid_t)
        merged = ref.tpool_patch_merger(h, self.grid_t, MERGE)
        want = [x.detach().numpy() for x in self.r_proj(merged)]

        got = self.m(t2m(self.pixels), self.grids)
        self.assertEqual(len(got), len(want))
        for g, wv in zip(got, want):
            g = np.array(g)
            self.assertEqual(g.shape, wv.shape)
            err = np.abs(g - wv).max() / (np.abs(wv).max() + 1e-9)
            self.assertLess(err, 1e-2, f"projected rel err {err}")
        # output must be in text-embedding space
        self.assertEqual(got[0].shape[-1], TEXT_HID)
        # token count halves in each spatial dim, frames collapse entirely
        for (t, hh, ww), g in zip(self.grids, got):
            self.assertEqual(g.shape[0], (hh // 2) * (ww // 2))


class TestRealDimensions(unittest.TestCase):
    """Parity at K3's actual vision dimensions, all 27 layers, 447 M params.

    The reduced-dim tests above can hide errors that only appear at the real
    head_dim (128), the real 64x64 position grid, or the real 14px patch. The
    vision tower is small enough to just run for real, so we do.
    """

    def test_real_config_end_to_end(self):
        import json

        vc = json.load(open(os.path.join(REF_DIR, "config.json")))["vision_config"]
        cfg = mlxv.VisionConfig.from_dict(vc)
        self.assertEqual((cfg.vt_hidden_size, cfg.qkv_hidden_size, cfg.head_dim), (1024, 1536, 128))
        self.assertEqual(cfg.vt_num_hidden_layers, 27)

        torch.manual_seed(0)
        block_cfg = dict(
            num_heads=cfg.vt_num_attention_heads, hidden_dim=cfg.vt_hidden_size,
            qkv_hidden_size=cfg.qkv_hidden_size, mlp_dim=cfg.vt_intermediate_size,
            norm_type="rmsnorm", mlp_type="mlp2", activation=ref.PytorchGELUTanh(),
            attn_bias=False, linear_bias=False, attn_implementation="eager",
        )
        r_patch = ref.MoonVision3dPatchEmbed(
            out_dim=cfg.vt_hidden_size, patch_size=cfg.patch_size,
            pos_emb_height=cfg.init_pos_emb_height, pos_emb_width=cfg.init_pos_emb_width,
            pos_emb_time=cfg.init_pos_emb_time, pos_emb_type="divided_fixed",
            patch_embed_proj_bias=False, pos_emb_interpolation_mode="bilinear").eval()
        r_enc = ref.MoonViT3dEncoder(cfg.vt_hidden_size, cfg.vt_num_hidden_layers, block_cfg).eval()

        class PC:
            projector_ln_eps = cfg.projector_ln_eps
            mm_hidden_size = cfg.mm_hidden_size
            merge_kernel_size = cfg.merge_kernel_size
            hidden_size = cfg.text_hidden_size
        r_proj = ref.PatchMergerMLPV2(PC()).eval()
        for m_ in (r_patch, r_enc, r_proj):
            for p in m_.parameters():
                with torch.no_grad():
                    p.copy_(torch.randn_like(p) * 0.02)

        m = mlxv.VisionModel(cfg)
        w = {
            "vision_tower.patch_embed.proj.weight": t2m(r_patch.state_dict()["proj.weight"]),
            "vision_tower.patch_embed.pos_emb.weight": t2m(r_patch.state_dict()["pos_emb.weight"]),
        }
        for k, v in r_enc.state_dict().items():
            w[f"vision_tower.encoder.{k}"] = t2m(v)
        for k, v in r_proj.state_dict().items():
            w[f"mm_projector.{k}"] = t2m(v)
        m.load_weights(list(m.sanitize(w).items()))
        mx.eval(m.parameters())

        grids = [(1, 24, 32), (2, 16, 16)]  # a still image and a 2-frame clip
        ntok = sum(t * h * wd for t, h, wd in grids)
        px = torch.randn(ntok, 3, cfg.patch_size, cfg.patch_size)
        gt = torch.tensor(grids)

        h = r_enc(r_patch(px, gt), gt)
        want = [x.detach().numpy() for x in
                r_proj(ref.tpool_patch_merger(h, gt, cfg.merge_kernel_size))]
        got = [np.array(g) for g in m(t2m(px), grids)]

        for i, (g, wv) in enumerate(zip(got, want)):
            self.assertEqual(g.shape, wv.shape)
            rel = np.abs(g - wv).max() / (np.abs(wv).max() + 1e-9)
            self.assertLess(rel, 1e-4, f"image {i} rel err {rel:.3e}")
        # 2x2 spatial merge; frames collapse via temporal pooling
        self.assertEqual(got[0].shape, (12 * 16, cfg.text_hidden_size))
        self.assertEqual(got[1].shape, (8 * 8, cfg.text_hidden_size))


if __name__ == "__main__":
    unittest.main(verbosity=2)
