"""Tests for applying a REAP plan in scripts/convert.py.

The dangerous failure here is not a crash. Pruning renumbers experts -- new
index i is old expert keep[i] -- so the router's gate rows and
e_score_correction_bias have to be reordered to match. Forget that and the
model loads, runs, and produces fluent text while routing every token to the
wrong expert. Nothing structural catches it.

So the central test builds a fixture whose experts are *identifiable* (each
expert's weights encode its own index), prunes it, and checks that the surviving
experts compute what the originals computed and that the router still points at
them.

Run:  python3 tests/test_prune_apply.py
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from mlx_lm.models import kimi_k3  # noqa: E402
from test_convert_roundtrip import build_fixture, tiny_config, NE, NLAYERS  # noqa: E402

CONVERT = os.path.join(ROOT, "scripts", "convert.py")


def make_plan(path, keep_per_layer, top_k=2, num_experts=NE):
    plan = {"mode": "test", "top_k": top_k, "num_experts": num_experts,
            "min_experts": top_k, "layers": {}}
    for L, keep in keep_per_layer.items():
        plan["layers"][str(L)] = {"keep": sorted(keep),
                                  "bits": {"all": "mxfp4"}}
    json.dump(plan, open(path, "w"))
    return plan


def run_convert(src, out, profile="mxfp4", plan=None, expect_fail=False):
    cmd = [sys.executable, CONVERT, "--src", src, "--out", out, "--profile", profile]
    if plan:
        cmd += ["--prune-plan", plan]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if expect_fail:
        return r
    if r.returncode != 0:
        raise AssertionError(r.stdout + r.stderr)
    return r


def load_all(path):
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    W = {}
    for sh in sorted(set(idx.values())):
        W.update(mx.load(os.path.join(path, sh)))
    return W


class TestPruneApply(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="prune-")
        cls.src = os.path.join(cls.tmp, "src")
        build_fixture(cls.src)
        cls.moe_layers = [L for L in range(NLAYERS)
                          if L >= tiny_config()["text_config"]["first_k_dense_replace"]]
        cls.keep = {L: [0, 2, 4, 6] for L in cls.moe_layers}   # 4 of 8
        cls.plan_path = os.path.join(cls.tmp, "plan.json")
        make_plan(cls.plan_path, cls.keep)
        cls.out = os.path.join(cls.tmp, "pruned")
        run_convert(cls.src, cls.out, plan=cls.plan_path)
        cls.W = load_all(cls.out)
        cls.cfg = json.load(open(os.path.join(cls.out, "config.json")))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_expert_tensors_are_narrowed(self):
        for L in self.moe_layers:
            for dst in ("gate_proj", "up_proj", "down_proj"):
                w = self.W[f"model.layers.{L}.block_sparse_moe.switch_mlp.{dst}.weight"]
                self.assertEqual(w.shape[0], 4, f"layer {L}.{dst} kept {w.shape[0]}")

    def test_router_rows_follow_the_kept_experts(self):
        """gate.weight[i] must equal the ORIGINAL gate.weight[keep[i]]."""
        srcW = load_all_source(self.src)
        for L in self.moe_layers:
            orig = srcW[f"language_model.model.layers.{L}.block_sparse_moe.gate.weight"]
            got = self.W[f"model.layers.{L}.block_sparse_moe.gate.weight"]
            self.assertEqual(got.shape[0], 4)
            for new_i, old_i in enumerate(self.keep[L]):
                self.assertTrue(
                    bool(mx.all(got[new_i] == orig[old_i])),
                    f"layer {L}: gate row {new_i} is not old row {old_i}",
                )

    def test_correction_bias_follows_too(self):
        srcW = load_all_source(self.src)
        for L in self.moe_layers:
            orig = srcW[f"language_model.model.layers.{L}.block_sparse_moe"
                        f".gate.e_score_correction_bias"]
            got = self.W[f"model.layers.{L}.block_sparse_moe.e_score_correction_bias"]
            want = mx.take(orig, mx.array(self.keep[L]), axis=0)
            self.assertTrue(bool(mx.all(got == want)), f"layer {L} bias misordered")

    def test_expert_weights_are_the_kept_ones_bit_exact(self):
        """mxfp4 passthrough must survive pruning unchanged."""
        srcW = load_all_source(self.src)
        for L in self.moe_layers:
            for w, dst in (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")):
                got = self.W[f"model.layers.{L}.block_sparse_moe.switch_mlp.{dst}.weight"]
                gs = self.W[f"model.layers.{L}.block_sparse_moe.switch_mlp.{dst}.scales"]
                for new_i, old_i in enumerate(self.keep[L]):
                    base = (f"language_model.model.layers.{L}.block_sparse_moe"
                            f".experts.{old_i}.{w}")
                    op = srcW[f"{base}.weight_packed"].view(mx.uint32)
                    os_ = srcW[f"{base}.weight_scale"]
                    self.assertTrue(bool(mx.all(got[new_i] == op)),
                                    f"L{L} {dst}[{new_i}] != source expert {old_i}")
                    self.assertTrue(bool(mx.all(gs[new_i] == os_)))

    def test_config_records_per_layer_counts(self):
        tc = self.cfg.get("text_config", self.cfg)
        counts = tc["expert_counts"]
        self.assertEqual(len(counts), NLAYERS)
        for L in range(NLAYERS):
            self.assertEqual(counts[L], 4 if L in self.moe_layers else 0)
        self.assertEqual(tc["num_experts"], 4)
        self.assertEqual(self.cfg["reap"]["source_num_experts"], NE)

    def test_pruned_model_loads_and_runs(self):
        from mlx_lm.utils import load_model
        import pathlib

        model, _ = load_model(pathlib.Path(self.out), lazy=False)
        out = model(mx.array([[3, 4, 5, 6, 7]]))
        mx.eval(out)
        self.assertEqual(out.shape[-1], tiny_config()["text_config"]["vocab_size"])
        self.assertFalse(bool(mx.any(mx.isnan(out))))
        # every MoE layer really is narrower now
        args = kimi_k3.ModelArgs.from_dict(self.cfg)
        for L in self.moe_layers:
            self.assertEqual(kimi_k3.experts_in_layer(args, L), 4)


def load_all_source(src):
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    W = {}
    for sh in sorted(set(idx.values())):
        W.update(mx.load(os.path.join(src, sh)))
    return W


class TestVariableWidthLayers(unittest.TestCase):
    """`global` planning mode keeps different counts per layer."""

    def test_uneven_layers_convert_and_run(self):
        tmp = tempfile.mkdtemp(prefix="prune-var-")
        try:
            src = os.path.join(tmp, "src")
            build_fixture(src)
            fk = tiny_config()["text_config"]["first_k_dense_replace"]
            moe = [L for L in range(NLAYERS) if L >= fk]
            widths = {}
            for n, L in enumerate(moe):
                widths[L] = list(range(2 + n * 2))  # 2, 4, 6, ...
            plan = os.path.join(tmp, "plan.json")
            make_plan(plan, widths)
            out = os.path.join(tmp, "pruned")
            run_convert(src, out, plan=plan)

            cfg = json.load(open(os.path.join(out, "config.json")))
            counts = cfg.get("text_config", cfg)["expert_counts"]
            for L in moe:
                self.assertEqual(counts[L], len(widths[L]))
            self.assertEqual(cfg.get("text_config", cfg)["num_experts"],
                             max(len(v) for v in widths.values()))

            from mlx_lm.utils import load_model
            import pathlib

            model, _ = load_model(pathlib.Path(out), lazy=False)
            o = model(mx.array([[1, 2, 3, 4]]))
            mx.eval(o)
            self.assertFalse(bool(mx.any(mx.isnan(o))))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestPruneEquivalence(unittest.TestCase):
    """Behavioural checks, not just structural ones."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="prune-eq-")
        cls.src = os.path.join(cls.tmp, "src")
        build_fixture(cls.src)
        cls.fk = tiny_config()["text_config"]["first_k_dense_replace"]
        cls.moe = [L for L in range(NLAYERS) if L >= cls.fk]
        cls.full = os.path.join(cls.tmp, "full")
        run_convert(cls.src, cls.full)
        cls.toks = mx.array([[3, 11, 27, 5, 9, 40]])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _load(self, path):
        import pathlib

        from mlx_lm.utils import load_model

        m, _ = load_model(pathlib.Path(path), lazy=False)
        return m

    def test_identity_prune_is_a_no_op(self):
        """Keeping every expert must reproduce the unpruned model exactly."""
        plan = os.path.join(self.tmp, "identity.json")
        make_plan(plan, {L: list(range(NE)) for L in self.moe})
        out = os.path.join(self.tmp, "ident")
        run_convert(self.src, out, plan=plan)

        a = self._load(self.full)(self.toks)
        b = self._load(out)(self.toks)
        mx.eval(a, b)
        self.assertEqual(float(mx.abs(a - b).max()), 0.0,
                         "identity prune changed the model")

    def test_pruned_equals_full_with_dropped_experts_masked(self):
        """The semantic definition of pruning.

        Dropping experts is exactly equivalent to making the router never
        select them. Push the correction bias of the dropped experts to -inf in
        the full model and the two must agree: the surviving gates renormalize
        identically (moe_renormalize=True), and the kept experts' weights are
        untouched. This is what catches a wrong keep->row mapping, which no
        shape check can see.
        """
        keep = [0, 3, 5, 7]
        dropped = [e for e in range(NE) if e not in keep]
        plan = os.path.join(self.tmp, "subset.json")
        make_plan(plan, {L: keep for L in self.moe})
        out = os.path.join(self.tmp, "subset")
        run_convert(self.src, out, plan=plan)

        pruned = self._load(out)
        full = self._load(self.full)
        for L in self.moe:
            moe = full.model.layers[L].block_sparse_moe
            bias = np.array(moe.e_score_correction_bias, copy=True)
            bias[dropped] = -1e9          # unselectable
            moe.e_score_correction_bias = mx.array(bias)
        mx.eval(full.parameters())

        a = pruned(self.toks)
        b = full(self.toks)
        mx.eval(a, b)
        rel = float(mx.abs(a - b).max() / (mx.abs(b).max() + 1e-9))
        self.assertLess(rel, 1e-5,
                        f"pruned model disagrees with masked full model: rel {rel:.2e}")

    def test_wrong_router_mapping_would_be_caught(self):
        """Guard on the guard: shuffle the router rows and the check must fail.

        If this passed, the equivalence test above would be vacuous.
        """
        keep = [0, 3, 5, 7]
        plan = os.path.join(self.tmp, "subset2.json")
        make_plan(plan, {L: keep for L in self.moe})
        out = os.path.join(self.tmp, "subset2")
        run_convert(self.src, out, plan=plan)

        good = self._load(out)
        bad = self._load(out)
        for L in self.moe:                      # rotate router rows by one
            moe = bad.model.layers[L].block_sparse_moe
            w = moe.gate.weight                 # bf16: stay in MLX, numpy can't view it
            moe.gate.weight = mx.concatenate([w[-1:], w[:-1]], axis=0)
        mx.eval(bad.parameters())

        a, b = good(self.toks), bad(self.toks)
        mx.eval(a, b)
        self.assertGreater(float(mx.abs(a - b).max()), 1e-4,
                           "shuffling router rows changed nothing -- the "
                           "equivalence test cannot detect misrouting")


class TestPlanValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="prune-val-")
        self.src = os.path.join(self.tmp, "src")
        build_fixture(self.src)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_rejects_layer_narrower_than_top_k(self):
        fk = tiny_config()["text_config"]["first_k_dense_replace"]
        plan = os.path.join(self.tmp, "bad.json")
        make_plan(plan, {L: [0] for L in range(fk, NLAYERS)})  # 1 expert, top_k=2
        r = run_convert(self.src, os.path.join(self.tmp, "o"), plan=plan, expect_fail=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("top-k", r.stdout + r.stderr)

    def test_rejects_out_of_range_expert(self):
        fk = tiny_config()["text_config"]["first_k_dense_replace"]
        plan = os.path.join(self.tmp, "bad2.json")
        make_plan(plan, {L: [0, 1, NE + 5] for L in range(fk, NLAYERS)})
        r = run_convert(self.src, os.path.join(self.tmp, "o2"), plan=plan, expect_fail=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("out-of-range", r.stdout + r.stderr)

    def test_rejects_graded_plan_with_an_explanation(self):
        fk = tiny_config()["text_config"]["first_k_dense_replace"]
        plan_path = os.path.join(self.tmp, "graded.json")
        plan = make_plan(plan_path, {L: [0, 1, 2, 3] for L in range(fk, NLAYERS)})
        for entry in plan["layers"].values():
            entry["bits"] = {"0": "mxfp4", "1": "mxfp4", "2": "2bit", "3": "2bit"}
        json.dump(plan, open(plan_path, "w"))
        r = run_convert(self.src, os.path.join(self.tmp, "o3"), plan=plan_path,
                        expect_fail=True)
        self.assertNotEqual(r.returncode, 0)
        msg = r.stdout + r.stderr
        self.assertIn("bit widths", msg)
        self.assertIn("gather_qmm", msg)

    def test_prune_works_with_lossy_profiles_too(self):
        fk = tiny_config()["text_config"]["first_k_dense_replace"]
        plan = os.path.join(self.tmp, "p.json")
        make_plan(plan, {L: [1, 3, 5, 7] for L in range(fk, NLAYERS)})
        out = os.path.join(self.tmp, "o3bit")
        run_convert(self.src, out, profile="3bit", plan=plan)
        W = load_all(out)
        for L in range(fk, NLAYERS):
            w = W[f"model.layers.{L}.block_sparse_moe.switch_mlp.gate_proj.weight"]
            self.assertEqual(w.shape[0], 4)
            self.assertIn(f"model.layers.{L}.block_sparse_moe.switch_mlp.gate_proj.biases", W)


if __name__ == "__main__":
    unittest.main(verbosity=2)
