"""Structural + numerical tests for the Kimi-K3 MLX port.

No weights download required — the checkpoint key/shape coverage test reads the
bundled `reference/model.safetensors.index.json.gz` plus a shape table captured
from the real safetensors headers.

Run:  python3 tests/test_kimi_k3.py
"""

import gzip
import json
import os
import re
import sys
import unittest

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mlx_lm.models import kimi_k3  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
REF_CONFIG = os.path.join(ROOT, "reference", "config.json")
REF_INDEX = os.path.join(ROOT, "reference", "model.safetensors.index.json.gz")


def tiny_args(**over):
    """A 6-layer / 8-expert K3 that keeps every architectural feature.

    Layers 1..6 (1-based) -> kda for all but layer 4, matching K3's 1-based
    config convention. attn_res_block_size=3 exercises two block pushes.
    """
    cfg = dict(
        model_type="kimi_k3",
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        rms_norm_eps=1e-5,
        linear_attn_config={
            # head_dim must stay >= 64: mlx's gated_delta Metal kernel computes
            # n_per_t = head_dim / 4 / 4 and emits a zero-length C++ array below
            # that. Real K3 uses 128, so this is a test-fixture constraint only.
            "head_dim": 64,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [1, 2, 3, 5, 6],
            "full_attn_layers": [4],
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        num_experts=8,
        moe_intermediate_size=32,
        kv_lora_rank=16,
        q_lora_rank=24,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        mla_use_nope=True,
        mla_use_output_gate=True,
        num_experts_per_token=2,
        num_shared_experts=2,
        first_k_dense_replace=1,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        attn_res_block_size=3,
        routed_expert_hidden_size=32,
        latent_moe_use_norm=True,
        tie_word_embeddings=False,
    )
    cfg.update(over)
    return kimi_k3.ModelArgs.from_dict(cfg)


class TestConfigParsing(unittest.TestCase):
    def test_parses_nested_text_config(self):
        raw = json.load(open(REF_CONFIG))
        args = kimi_k3.ModelArgs.from_dict(raw)
        self.assertEqual(args.model_type, "kimi_k3")
        self.assertEqual(args.hidden_size, 7168)
        self.assertEqual(args.num_hidden_layers, 93)
        self.assertEqual(args.num_experts, 896)
        self.assertEqual(args.num_experts_per_token, 16)
        self.assertEqual(args.num_shared_experts, 2)
        self.assertEqual(args.routed_expert_hidden_size, 3584)
        self.assertEqual(args.moe_intermediate_size, 3072)
        self.assertEqual(args.attn_res_block_size, 12)
        self.assertEqual(args.q_lora_rank, 1536)
        self.assertEqual(args.kv_lora_rank, 512)
        self.assertTrue(args.mla_use_nope)
        self.assertTrue(args.mla_use_output_gate)
        self.assertEqual(args.activation_situ_beta, 4.0)
        self.assertEqual(args.activation_situ_linear_beta, 25.0)

    def test_layer_type_split_is_one_based(self):
        """config full_attn_layers=[4,8,...] must map to tensor layers 3,7,..."""
        args = kimi_k3.ModelArgs.from_dict(json.load(open(REF_CONFIG)))
        kda = args.linear_attn_config["kda_layers"]
        linear = [(i + 1) in kda for i in range(args.num_hidden_layers)]
        mla_idx = [i for i, is_lin in enumerate(linear) if not is_lin]
        self.assertEqual(
            mla_idx,
            [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71,
             75, 79, 83, 87, 91, 92],
        )
        self.assertEqual(sum(linear), 69)
        self.assertEqual(len(mla_idx), 24)


class TestSitu(unittest.TestCase):
    def test_matches_reference_formula(self):
        beta, lbeta = 4.0, 25.0
        act = kimi_k3.SituGLU(beta, lbeta)
        mx.random.seed(0)
        gate = mx.random.normal((5, 17)) * 6.0
        up = mx.random.normal((5, 17)) * 6.0
        got = act(up, gate)
        g32, u32 = gate.astype(mx.float32), up.astype(mx.float32)
        want = (
            beta * mx.tanh(g32 / beta) * mx.sigmoid(g32)
        ) * (lbeta * mx.tanh(u32 / lbeta))
        self.assertLess(float(mx.abs(got - want).max()), 1e-6)

    def test_saturates(self):
        act = kimi_k3.SituGLU(4.0, 25.0)
        big = mx.full((1, 4), 1e4)
        out = act(big, big)
        self.assertLess(float(mx.abs(out - 4.0 * 25.0).max()), 1e-3)


class TestAttnRes(unittest.TestCase):
    def test_is_convex_combination(self):
        """Output must be a softmax-weighted average of the K+1 candidates."""
        N, K, H = 3, 4, 32
        mx.random.seed(1)
        prefix = mx.random.normal((N, H))
        blocks = mx.random.normal((N, K, H))
        proj = mx.random.normal((1, H))
        norm = mx.random.normal((H,))
        out = kimi_k3._apply_attn_res(prefix, blocks, proj, norm, 1e-5)
        self.assertEqual(out.shape, (N, H))

        cand = mx.concatenate([blocks, mx.expand_dims(prefix, 1)], axis=1)
        var = mx.mean(cand * cand, axis=-1, keepdims=True)
        k = cand * mx.rsqrt(var + 1e-5)
        scores = mx.sum(k * (norm * proj.reshape(-1)), axis=-1)
        probs = mx.softmax(scores, axis=-1)
        want = (mx.expand_dims(probs, 1) @ cand).squeeze(1)
        self.assertLess(float(mx.abs(out - want).max()), 1e-4)
        # probabilities sum to 1 -> output lies in the convex hull
        self.assertLess(float(mx.abs(probs.sum(-1) - 1.0).max()), 1e-5)

    def test_single_candidate_is_identity(self):
        N, H = 2, 16
        prefix = mx.random.normal((N, H))
        empty = mx.zeros((N, 0, H))
        out = kimi_k3._apply_attn_res(prefix, empty, mx.ones((1, H)), mx.ones((H,)), 1e-5)
        self.assertLess(float(mx.abs(out - prefix).max()), 1e-4)


class TestKdaGate(unittest.TestCase):
    def test_A_log_broadcasts_per_head(self):
        """[K3-5] A_log is stored as [head_dim] but the decay is PER-HEAD.

        fla/ops/kda/gate.py loads it with `tl.load(A_log + i_h)` over a grid whose
        second dimension is H = g.shape[-2] = num_heads, then broadcasts that
        scalar across head_dim. The checkpoint pads the tensor out to head_dim
        with zeros, which is what makes a per-channel reading look plausible from
        the shape alone; scripts/parity_probe_gate2.py scores this form at
        cosine 0.99999 against the kernel and every alternative below 0.95.
        """
        args = tiny_args()
        kda = kimi_k3.KimiDeltaAttention(args, 0)
        self.assertEqual(kda.A_log.shape, (kda.head_dim,))
        # s must be nonzero: the gate is sigmoid(exp(A_log) * s), so at s == 0 it
        # collapses to sigmoid(0) for every head and A_log drops out entirely.
        a = mx.ones((2, 3, kda.num_heads, kda.head_dim))
        self.assertEqual(kda._compute_g(a).shape, (2, 3, kda.num_heads, kda.head_dim))

        # distinct value per head, zero-padded exactly as the checkpoint ships it
        per_head = (mx.arange(kda.num_heads) + 1).astype(mx.float32) * 0.3
        pad = mx.zeros((kda.head_dim - kda.num_heads,))
        kda.A_log = mx.concatenate([per_head, pad])
        kda.dt_bias = mx.zeros((kda.projection_dim,))
        g = kda._compute_g(a)

        # constant across channels within a head, varying across heads
        self.assertLess(float(mx.abs(g[:, :, 0, :] - g[:, :, 0, :1]).max()), 1e-6)
        self.assertGreater(float(mx.abs(g[0, 0, 0, 0] - g[0, 0, 1, 0])), 1e-3)

    def test_A_log_padding_is_ignored(self):
        """Entries [num_heads:] are padding; touching them must change nothing."""
        args = tiny_args()
        kda = kimi_k3.KimiDeltaAttention(args, 0)
        kda.dt_bias = mx.zeros((kda.projection_dim,))
        a = mx.ones((1, 2, kda.num_heads, kda.head_dim))

        base = mx.concatenate([
            mx.ones((kda.num_heads,)) * 0.5, mx.zeros((kda.head_dim - kda.num_heads,))
        ])
        kda.A_log = base
        before = kda._compute_g(a)
        # garbage in the padded tail
        kda.A_log = mx.concatenate([
            base[: kda.num_heads], mx.ones((kda.head_dim - kda.num_heads,)) * 9.0
        ])
        after = kda._compute_g(a)
        self.assertLess(float(mx.abs(before - after).max()), 1e-7)

    def test_gate_lower_bound_saturates(self):
        """fla's safe-gate branch is `lower_bound * sigmoid(exp(A_log) * s)`.

        Not a clamp on the softplus form -- there is no softplus in that branch at
        all. It approaches lower_bound by saturation, so g bottoms out at exp(-5).
        """
        args = tiny_args()
        kda = kimi_k3.KimiDeltaAttention(args, 0)
        kda.A_log = mx.full((kda.head_dim,), 5.0)  # exp(5) ~ 148
        kda.dt_bias = mx.full((kda.projection_dim,), 5.0)
        g = kda._compute_g(mx.zeros((1, 1, kda.num_heads, kda.head_dim)))
        self.assertAlmostEqual(float(g.min()), float(mx.exp(mx.array(-5.0))), places=5)

        # and the other way: a large negative argument saturates the sigmoid to 0,
        # leaving the decay at 1.0 rather than the softplus form's exp(-something)
        kda.dt_bias = mx.full((kda.projection_dim,), -5.0)
        g = kda._compute_g(mx.zeros((1, 1, kda.num_heads, kda.head_dim)))
        self.assertAlmostEqual(float(g.max()), 1.0, places=5)


class TestPipelineSpan(unittest.TestCase):
    """The split must cover every layer exactly once, for both remainder modes.

    mlx-lm's PipelineMixin does not: for 93 layers over 4 ranks it yields
    [0:23] [23:46] [46:69] [72:93], dropping 69..71 with no error. Nothing in a
    forward pass notices -- the model just quietly skips three layers.
    """

    def _check(self, n, size, mode):
        spans = [kimi_k3.pipeline_span(n, size, r, mode) for r in range(size)]
        for lo, hi in spans:
            self.assertGreaterEqual(hi, lo)
        front = sorted(spans)
        self.assertEqual(front[0][0], 0, f"{mode} {n}/{size}: gap at start")
        self.assertEqual(front[-1][1], n, f"{mode} {n}/{size}: gap at end")
        for (a_lo, a_hi), (b_lo, b_hi) in zip(front, front[1:]):
            self.assertEqual(a_hi, b_lo, f"{mode} {n}/{size}: gap/overlap {a_hi}!={b_lo}")
        self.assertEqual(sum(hi - lo for lo, hi in spans), n)
        return spans

    def test_gapless_both_modes(self):
        for n, size in ((93, 4), (93, 2), (93, 3), (93, 5), (6, 4), (8, 3), (4, 4), (1, 1)):
            for mode in ("low", "high"):
                self._check(n, size, mode)

    def test_extra_moves_off_rank0(self):
        """K3_PIPELINE_EXTRA=high must take the 24th layer off rank 0."""
        low = [kimi_k3.pipeline_span(93, 4, r, "low") for r in range(4)]
        high = [kimi_k3.pipeline_span(93, 4, r, "high") for r in range(4)]
        self.assertEqual(low[0][1] - low[0][0], 24)
        self.assertEqual(high[0][1] - high[0][0], 23)
        self.assertEqual(high[3][1] - high[3][0], 24)

    def test_env_var_selects_mode(self):
        import os as _os
        prev = _os.environ.get("K3_PIPELINE_EXTRA")
        try:
            _os.environ["K3_PIPELINE_EXTRA"] = "high"
            self.assertEqual(kimi_k3.pipeline_span(93, 4, 0), (70, 93))
            _os.environ["K3_PIPELINE_EXTRA"] = "low"
            self.assertEqual(kimi_k3.pipeline_span(93, 4, 0), (69, 93))
        finally:
            if prev is None:
                _os.environ.pop("K3_PIPELINE_EXTRA", None)
            else:
                _os.environ["K3_PIPELINE_EXTRA"] = prev


class TestForward(unittest.TestCase):
    def test_prefill_and_decode_agree(self):
        args = tiny_args()
        model = kimi_k3.Model(args)
        model.eval()
        mx.eval(model.parameters())

        toks = mx.array([[3, 14, 15, 92, 65, 35, 89, 79]])
        full = model(toks)
        self.assertEqual(full.shape, (1, toks.shape[1], args.vocab_size))
        self.assertFalse(bool(mx.any(mx.isnan(full))))

        cache = model.make_cache()
        outs = []
        for i in range(toks.shape[1]):
            outs.append(model(toks[:, i : i + 1], cache=cache))
        stepwise = mx.concatenate(outs, axis=1)

        # AttnRes + KDA recurrence must give the same answer either way.
        diff = float(mx.abs(full - stepwise).max())
        self.assertLess(diff, 2e-2, f"prefill/decode mismatch: {diff}")

    def test_attn_res_pushes_expected_blocks(self):
        args = tiny_args()
        model = kimi_k3.Model(args)
        pushes = [i for i in range(args.num_hidden_layers) if i % args.attn_res_block_size == 0]
        self.assertEqual(pushes, [0, 3])
        mx.eval(model.parameters())
        out = model(mx.array([[1, 2, 3]]))
        self.assertFalse(bool(mx.any(mx.isnan(out))))


class TestCheckpointCoverage(unittest.TestCase):
    """Every one of the 497,220 checkpoint tensors must land somewhere."""

    @classmethod
    def setUpClass(cls):
        cls.args = kimi_k3.ModelArgs.from_dict(json.load(open(REF_CONFIG)))
        with gzip.open(REF_INDEX) as f:
            cls.keys = list(json.load(f)["weight_map"])

    def test_key_count(self):
        self.assertEqual(len(self.keys), 497220)

    def test_every_key_is_consumed_or_deliberately_skipped(self):
        args = self.args
        kda = set(args.linear_attn_config["kda_layers"])
        n_layers = args.num_hidden_layers

        consumed, skipped, unknown = 0, 0, []
        for k in self.keys:
            if k.startswith(("vision_tower.", "mm_projector.")):
                skipped += 1
                continue
            self.assertTrue(k.startswith("language_model."), k)
            t = k[len("language_model.") :]
            if t in (
                "lm_head.weight",
                "model.embed_tokens.weight",
                "model.norm.weight",
                "model.output_attn_res_proj.weight",
                "model.output_attn_res_norm.weight",
            ):
                consumed += 1
                continue
            m = re.match(r"model\.layers\.(\d+)\.(.+)", t)
            self.assertIsNotNone(m, k)
            idx, rest = int(m.group(1)), m.group(2)
            self.assertLess(idx, n_layers, k)
            is_linear = (idx + 1) in kda

            ok = False
            if rest in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attention_res_proj.weight",
                "self_attention_res_norm.weight",
                "mlp_res_proj.weight",
                "mlp_res_norm.weight",
            ):
                ok = True
            elif rest.startswith("self_attn."):
                name = rest[len("self_attn.") :]
                linear_names = {
                    "q_proj.weight", "k_proj.weight", "v_proj.weight",
                    "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight",
                    "f_a_proj.weight", "f_b_proj.weight", "b_proj.weight",
                    "g_proj.weight", "o_proj.weight", "o_norm.weight",
                    "A_log", "dt_bias",
                }
                mla_names = {
                    "q_a_proj.weight", "q_b_proj.weight", "q_a_layernorm.weight",
                    "kv_a_proj_with_mqa.weight", "kv_b_proj.weight",
                    "kv_a_layernorm.weight", "g_proj.weight", "o_proj.weight",
                }
                ok = name in (linear_names if is_linear else mla_names)
            elif rest.startswith("mlp."):
                ok = idx < args.first_k_dense_replace and rest in (
                    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"
                )
            elif rest.startswith("block_sparse_moe."):
                name = rest[len("block_sparse_moe.") :]
                if re.match(r"experts\.\d+\.w[123]\.weight(_packed|_scale)?$", name):
                    e = int(name.split(".")[1])
                    ok = e < args.num_experts
                else:
                    ok = name in (
                        "gate.weight", "gate.e_score_correction_bias",
                        "routed_expert_down_proj.weight", "routed_expert_up_proj.weight",
                        "routed_expert_norm.weight",
                        "shared_experts.gate_proj.weight",
                        "shared_experts.up_proj.weight",
                        "shared_experts.down_proj.weight",
                    )
            if ok:
                consumed += 1
            else:
                unknown.append(k)

        self.assertEqual(unknown[:20], [], f"{len(unknown)} unmapped keys")
        self.assertEqual(consumed + skipped, len(self.keys))
        self.assertEqual(skipped, 168)  # 27 blocks x 6 + patch_embed x2 + final_ln + proj x3

    def test_mxfp4_experts_are_packed_pairs(self):
        packed = sum(1 for k in self.keys if k.endswith(".weight_packed"))
        scales = sum(1 for k in self.keys if k.endswith(".weight_scale"))
        self.assertEqual(packed, scales)
        self.assertEqual(packed, 92 * 896 * 3)  # 92 MoE layers x 896 experts x w1/w2/w3


class TestTwoBankSwitchGLU(unittest.TestCase):
    """The two-bank MoE must be a pure refactor of SwitchGLU.

    Graded REAP splits a layer's experts across two banks so they can carry
    different bit widths (MLX pins one width per expert tensor). Give both banks
    the SAME weights and precision and the result must match a single bank
    exactly -- that isolates the partition/unsort machinery from quantization
    error, so a bug there cannot hide behind "the low tier is lossy".
    """

    D, H, NE, K, N_HI = 64, 32, 8, 3, 5

    def _pair(self):
        from mlx_lm.models.switch_layers import SwitchGLU

        mx.random.seed(0)
        act = kimi_k3.SituGLU(4.0, 25.0)
        single = SwitchGLU(self.D, self.H, self.NE, activation=act)
        mx.eval(single.parameters())
        two = kimi_k3.TwoBankSwitchGLU(
            self.D, self.H, self.N_HI, self.NE - self.N_HI, activation=act
        )
        p = single.parameters()
        two.bank_hi.update({k: {kk: vv[: self.N_HI] for kk, vv in v.items()}
                            for k, v in p.items()})
        two.bank_lo.update({k: {kk: vv[self.N_HI:] for kk, vv in v.items()}
                            for k, v in p.items()})
        mx.eval(two.parameters())
        return single, two

    def test_matches_single_bank(self):
        single, two = self._pair()
        for B, T in ((2, 64), (1, 1), (1, 4), (4, 128)):
            x = mx.random.normal((B, T, self.D))
            idx = mx.random.randint(0, self.NE, (B, T, self.K))
            a, b = single(x, idx), two(x, idx)
            mx.eval(a, b)
            err = float(mx.abs(a - b).max())
            # Exact where single-bank also sorts (>=64 routed pairs). Below that
            # it skips its sort while we always sort, so summation order differs
            # by float32 epsilon.
            tol = 0.0 if idx.size >= 64 else 1e-6
            self.assertLessEqual(err, tol, f"({B},{T}) pairs={idx.size} err={err}")

    def test_handles_all_pairs_in_one_bank(self):
        single, two = self._pair()
        x = mx.random.normal((1, 8, self.D))
        for lo, hi, label in ((0, self.N_HI, "all-hi"), (self.N_HI, self.NE, "all-lo")):
            idx = mx.random.randint(lo, hi, (1, 8, self.K))
            a, b = single(x, idx), two(x, idx)
            mx.eval(a, b)
            self.assertLess(float(mx.abs(a - b).max()), 1e-6, label)

    def test_both_partition_paths_agree(self):
        """Host partition (small) and device partition (large) must agree."""
        _, two = self._pair()
        x = mx.random.normal((1, 400, self.D))
        idx = mx.random.randint(0, self.NE, (1, 400, self.K))
        self.assertGreater(idx.size, two.PARTITION_ON_HOST)
        dev = two(x, idx)
        saved = two.PARTITION_ON_HOST
        try:
            two.PARTITION_ON_HOST = 10**9          # force the host path
            host = two(x, idx)
            mx.eval(dev, host)
            self.assertEqual(float(mx.abs(dev - host).max()), 0.0)
        finally:
            two.PARTITION_ON_HOST = saved

    def test_rejects_empty_bank(self):
        with self.assertRaises(ValueError):
            kimi_k3.TwoBankSwitchGLU(8, 8, 4, 0, activation=kimi_k3.SituGLU(4.0, 25.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
