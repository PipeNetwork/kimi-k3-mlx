"""Tests for the streaming REAP calibration harness.

Structural checks (shapes, routing counts) are cheap and would pass even if the
saliency numbers were nonsense, so the load-bearing test here is
`test_saliency_tracks_expert_magnitude`: it deliberately amplifies one expert in
the checkpoint and asserts that expert's score rises. Without that, a harness
that silently recorded, say, gates only -- or the wrong axis of `y` -- would look
perfectly healthy.

Amplification is done by bumping the MXFP4 e8m0 scale byte, which multiplies the
expert's weights by an exact power of two without touching the packed codes.

Run:  python3 tests/test_reap.py
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import mlx.core as mx  # noqa: E402
from test_convert_roundtrip import build_fixture, tiny_config  # noqa: E402

CALIB = os.path.join(ROOT, "scripts", "reap_calibrate.py")
PLAN = os.path.join(ROOT, "scripts", "reap_plan.py")
SEQS, SEQLEN = 4, 32


def run_calibration(src, out, extra=()):
    toks = np.random.default_rng(0).integers(0, 128, (SEQS, SEQLEN)).astype(np.int32)
    tp = os.path.join(os.path.dirname(out), "toks.npy")
    np.save(tp, toks)
    r = subprocess.run(
        [sys.executable, CALIB, "--src", src, "--out", out, "--calib-tokens", tp,
         "--seqs", str(SEQS), "--seqlen", str(SEQLEN), "--batch", "2", *extra],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise AssertionError(r.stdout + r.stderr)
    return np.load(out)


def amplify_expert(src, layer, expert, shift=4):
    """Multiply one expert's weights by 2**shift via its e8m0 scale bytes."""
    idx_path = os.path.join(src, "model.safetensors.index.json")
    idx = json.load(open(idx_path))["weight_map"]
    touched = set()
    for w in ("w1", "w2", "w3"):
        key = (f"language_model.model.layers.{layer}.block_sparse_moe"
               f".experts.{expert}.{w}.weight_scale")
        touched.add(idx[key])
    for shard in touched:
        p = os.path.join(src, shard)
        W = dict(mx.load(p))
        for w in ("w1", "w2", "w3"):
            key = (f"language_model.model.layers.{layer}.block_sparse_moe"
                   f".experts.{expert}.{w}.weight_scale")
            if key in W:
                # e8m0: byte = exponent + 127, so +shift multiplies by 2**shift.
                # Clip well below 255 so nothing saturates to inf.
                W[key] = mx.minimum(
                    W[key].astype(mx.int32) + shift, mx.array(200, mx.int32)
                ).astype(mx.uint8)
        # mx.load is mmap-backed: writing straight back to `p` would pull the
        # rug out from under the very arrays being serialized (it silently
        # scrambled the packed weights). Serialize to a sibling, then rename.
        mx.eval(list(W.values()))
        tmp_path = p[: -len(".safetensors")] + ".tmp.safetensors"
        mx.save_safetensors(tmp_path, W)
        os.replace(tmp_path, p)


class TestCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="reap-test-")
        cls.src = os.path.join(cls.tmp, "src")
        build_fixture(cls.src)
        cls.z = run_calibration(cls.src, os.path.join(cls.tmp, "sal.npz"))
        cls.cfg = tiny_config()

    def test_shapes_and_layer_split(self):
        z = self.z
        tc = self.cfg["text_config"]
        self.assertEqual(z["saliency"].shape, (tc["num_hidden_layers"], tc["num_experts"]))
        self.assertEqual(int(z["n_tokens"]), SEQS * SEQLEN)
        # layer 0 is dense (first_k_dense_replace=1): no experts, no saliency
        self.assertEqual(list(z["moe_layers"]), [1, 2, 3])
        self.assertEqual(z["saliency"][0].sum(), 0.0)
        self.assertEqual(z["counts"][0].sum(), 0.0)

    def test_routing_counts_are_exact(self):
        z = self.z
        expected = int(z["n_tokens"]) * int(z["top_k"])
        for L in z["moe_layers"]:
            self.assertEqual(int(z["counts"][L].sum()), expected,
                             f"layer {L} routed a wrong number of (token, slot) pairs")

    def test_saliency_is_nonnegative_and_finite(self):
        s = self.z["saliency"]
        self.assertTrue(np.all(np.isfinite(s)))
        self.assertTrue(np.all(s >= 0), "gate * ||output|| cannot be negative")

    def test_unrouted_experts_score_zero(self):
        z = self.z
        s, c = z["saliency"], z["counts"]
        never = (c == 0) & np.isin(np.arange(s.shape[0])[:, None], z["moe_layers"])
        if never.any():
            self.assertTrue(np.all(s[never] == 0.0))

    def test_deterministic(self):
        z2 = run_calibration(self.src, os.path.join(self.tmp, "sal2.npz"))
        np.testing.assert_allclose(self.z["saliency"], z2["saliency"], rtol=1e-5)


class TestSaliencyIsMeaningful(unittest.TestCase):
    """Amplify one expert; its saliency must rise and it must survive pruning."""

    def test_saliency_tracks_expert_magnitude(self):
        tmp = tempfile.mkdtemp(prefix="reap-amp-")
        src = os.path.join(tmp, "src")
        build_fixture(src)

        base = run_calibration(src, os.path.join(tmp, "before.npz"))
        LAYER, EXPERT = 2, 5
        before = base["saliency"][LAYER, EXPERT]
        rank_before = int(np.argsort(-base["saliency"][LAYER]).tolist().index(EXPERT))

        amplify_expert(src, LAYER, EXPERT, shift=4)  # 16x weights
        after_z = run_calibration(src, os.path.join(tmp, "after.npz"))
        after = after_z["saliency"][LAYER, EXPERT]
        rank_after = int(np.argsort(-after_z["saliency"][LAYER]).tolist().index(EXPERT))

        self.assertGreater(after, before * 2.0,
                           f"16x weights only moved saliency {before:.3e} -> {after:.3e}")
        self.assertEqual(rank_after, 0, "amplified expert should rank most salient")
        self.assertLessEqual(rank_after, rank_before)

        # untouched layers must be unaffected -- confirms the stat is per-layer
        for L in after_z["moe_layers"]:
            if int(L) < LAYER:
                np.testing.assert_allclose(
                    base["saliency"][L], after_z["saliency"][L], rtol=1e-4,
                    err_msg=f"layer {L} changed but is upstream of the edit")


class TestPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="reap-plan-")
        src = os.path.join(cls.tmp, "src")
        build_fixture(src)
        cls.npz_path = os.path.join(cls.tmp, "sal.npz")
        cls.z = run_calibration(src, cls.npz_path)

    def _plan(self, *args):
        out = os.path.join(self.tmp, f"plan_{abs(hash(args))}.json")
        r = subprocess.run([sys.executable, PLAN, "--saliency", self.npz_path,
                            "--out", out, "--min-experts", "2", *args],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return json.load(open(out)), r.stdout

    def test_uniform_keeps_highest_saliency(self):
        plan, _ = self._plan("--mode", "uniform", "--keep", "0.5")
        NE = int(self.z["num_experts"])
        for L, entry in plan["layers"].items():
            keep = entry["keep"]
            self.assertEqual(len(keep), NE // 2)
            sal = self.z["saliency"][int(L)]
            kept_min = sal[keep].min()
            dropped_max = sal[[e for e in range(NE) if e not in keep]].max()
            self.assertGreaterEqual(kept_min, dropped_max,
                                    f"layer {L} dropped a more salient expert than it kept")

    def test_graded_assigns_precision_by_rank(self):
        plan, out = self._plan("--mode", "graded", "--hi", "0.25", "--lo", "0.25")
        for L, entry in plan["layers"].items():
            sal = self.z["saliency"][int(L)]
            hi = [int(e) for e, b in entry["bits"].items() if b == "mxfp4"]
            lo = [int(e) for e, b in entry["bits"].items() if b == "2bit"]
            self.assertTrue(hi and lo)
            # every mxfp4 expert must outrank every 2-bit expert
            self.assertGreaterEqual(sal[hi].min(), sal[lo].max())
            self.assertEqual(sorted(hi + lo), entry["keep"])
        self.assertIn("saliency mass retained", out)

    def test_min_experts_floor_is_respected(self):
        plan, _ = self._plan("--mode", "uniform", "--keep", "0.01")
        for entry in plan["layers"].values():
            self.assertGreaterEqual(len(entry["keep"]), 2)

    def test_reports_retained_saliency_mass(self):
        _, out = self._plan("--mode", "uniform", "--keep", "0.5")
        line = [x for x in out.splitlines() if "saliency mass retained" in x][0]
        pct = float(line.split(":")[1].strip().rstrip("%"))
        # keeping the top half by saliency must retain well over half the mass
        self.assertGreater(pct, 50.0)
        self.assertLessEqual(pct, 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
