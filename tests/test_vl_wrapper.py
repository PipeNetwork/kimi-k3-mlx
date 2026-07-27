"""Tests for the Kimi-K3 mlx-vlm wrapper.

The load-bearing behaviour is the *expanding* merge: K3 puts exactly one
`<|media_pad|>` (163605) in the prompt per image, and it must become that
image's whole feature block. A same-length scatter -- the arrangement most
LLaVA-style models and mlx-vlm's own Kimi-VL glue use -- would keep one token
per image and silently discard the rest, which does not crash and does not
produce obviously broken text. So it is asserted directly, including against a
hand-built expected sequence.

Run:  python3 tests/test_vl_wrapper.py
"""

import json
import os
import sys
import unittest

import mlx.core as mx
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import kimi_k3_vl  # noqa: E402
from kimi_k3_vl.kimi_k3_vl import merge_image_features  # noqa: E402

PH = 163605  # <|media_pad|>


class TestMerge(unittest.TestCase):
    def test_expands_one_placeholder_into_many(self):
        D = 8
        ids = mx.array([[10, 11, PH, 12, 13]])
        emb = mx.arange(5 * D, dtype=mx.float32).reshape(1, 5, D)
        feat = mx.full((7, D), -1.0)

        merged, mask = merge_image_features(emb, ids, [feat], PH)
        # 5 tokens - 1 placeholder + 7 image tokens
        self.assertEqual(merged.shape, (1, 11, D))
        self.assertEqual(mask.shape, (1, 11))

        want = mx.concatenate([emb[0, :2], feat, emb[0, 3:]], axis=0)
        self.assertTrue(bool(mx.all(merged[0] == want)))
        self.assertTrue(bool(mx.all(mask == 1)))

    def test_multiple_images_keep_prompt_order(self):
        D = 4
        ids = mx.array([[1, PH, 2, PH, 3]])
        emb = mx.random.normal((1, 5, D))
        f0 = mx.full((3, D), 7.0)
        f1 = mx.full((2, D), 9.0)

        merged, _ = merge_image_features(emb, ids, [f0, f1], PH)
        self.assertEqual(merged.shape, (1, 5 - 2 + 5, D))
        self.assertTrue(bool(mx.all(merged[0, 1:4] == 7.0)))
        self.assertTrue(bool(mx.all(merged[0, 5:7] == 9.0)))
        self.assertTrue(bool(mx.all(merged[0, 0] == emb[0, 0])))
        self.assertTrue(bool(mx.all(merged[0, 4] == emb[0, 2])))
        self.assertTrue(bool(mx.all(merged[0, 7] == emb[0, 4])))

    def test_placeholder_at_sequence_edges(self):
        D = 4
        emb = mx.random.normal((1, 3, D))
        f = mx.full((2, D), 5.0)
        for ids in (mx.array([[PH, 1, 2]]), mx.array([[1, 2, PH]])):
            merged, _ = merge_image_features(emb, ids, [f], PH)
            self.assertEqual(merged.shape, (1, 4, D))

    def test_no_placeholder_is_identity(self):
        D = 4
        ids = mx.array([[1, 2, 3]])
        emb = mx.random.normal((1, 3, D))
        merged, mask = merge_image_features(emb, ids, [], PH)
        self.assertTrue(bool(mx.all(merged == emb)))
        self.assertTrue(bool(mx.all(mask == 1)))

    def test_count_mismatch_raises(self):
        D = 4
        ids = mx.array([[1, PH, 2]])
        emb = mx.random.normal((1, 3, D))
        with self.assertRaises(ValueError):
            merge_image_features(emb, ids, [], PH)
        with self.assertRaises(ValueError):
            merge_image_features(emb, ids, [mx.zeros((2, D))] * 2, PH)

    def test_batch_is_left_padded(self):
        D = 4
        ids = mx.array([[1, PH, 2], [1, PH, 2]])
        emb = mx.random.normal((2, 3, D))
        f0 = mx.full((4, D), 1.0)   # row 0 -> 2 + 4 = 6
        f1 = mx.full((1, D), 2.0)   # row 1 -> 2 + 1 = 3
        merged, mask = merge_image_features(emb, ids, [f0, f1], PH, pad_token_id=0)
        self.assertEqual(merged.shape, (2, 6, D))
        # shorter row is padded on the LEFT, and padding is zeroed
        self.assertEqual(mask[1].tolist(), [0, 0, 0, 1, 1, 1])
        self.assertTrue(bool(mx.all(merged[1, :3] == 0.0)))
        self.assertEqual(mask[0].tolist(), [1] * 6)


class TestEndToEnd(unittest.TestCase):
    """Tiny full multimodal model: pixels -> vision -> merge -> text -> logits."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.join(ROOT, "tests"))
        from test_convert_roundtrip import tiny_config

        cls.cfg_dict = tiny_config()
        cls.cfg = kimi_k3_vl.ModelConfig.from_dict(cls.cfg_dict)
        cls.model = kimi_k3_vl.Model(cls.cfg)
        mx.eval(cls.model.parameters())

    def test_config_picks_up_placeholder(self):
        # the fixture omits it, so the K3 default must be used
        self.assertEqual(self.cfg.media_placeholder_token_id, PH)
        real = json.load(open(os.path.join(ROOT, "reference", "config.json")))
        self.assertEqual(
            kimi_k3_vl.ModelConfig.from_dict(real).media_placeholder_token_id, 163605
        )

    def test_text_only_path(self):
        ids = mx.array([[3, 4, 5, 6]])
        out = self.model(ids)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 4, self.cfg_dict["text_config"]["vocab_size"]))
        self.assertFalse(bool(mx.any(mx.isnan(out))))

    def test_multimodal_path_grows_sequence(self):
        vc = self.cfg_dict["vision_config"]
        grids = [(1, 4, 4)]
        ntok = sum(t * h * w for t, h, w in grids)
        px = mx.random.normal((ntok, 3, vc["patch_size"], vc["patch_size"]))

        ids = mx.array([[3, 4, PH, 5]])
        out = self.model(ids, pixel_values=px, grid_thws=grids)
        mx.eval(out)

        # image contributes (4/2)*(4/2) = 4 tokens; 4 - 1 + 4 = 7
        self.assertEqual(out.shape[1], 7)
        self.assertFalse(bool(mx.any(mx.isnan(out))))

    def test_image_actually_changes_logits(self):
        """A guard against the image path being silently inert."""
        vc = self.cfg_dict["vision_config"]
        grids = [(1, 4, 4)]
        ntok = 16
        ids = mx.array([[3, 4, PH, 5]])
        a = self.model(ids, pixel_values=mx.random.normal((ntok, 3, vc["patch_size"], vc["patch_size"]), key=mx.random.key(0)), grid_thws=grids)
        b = self.model(ids, pixel_values=mx.random.normal((ntok, 3, vc["patch_size"], vc["patch_size"]), key=mx.random.key(1)) * 5.0, grid_thws=grids)
        mx.eval(a, b)
        self.assertEqual(a.shape, b.shape)
        self.assertGreater(float(mx.abs(a - b).max()), 1e-4)

    def test_two_images_expand_independently(self):
        vc = self.cfg_dict["vision_config"]
        grids = [(1, 4, 4), (2, 4, 4)]   # 4 tokens each after merge/tpool
        ntok = sum(t * h * w for t, h, w in grids)
        px = mx.random.normal((ntok, 3, vc["patch_size"], vc["patch_size"]))
        ids = mx.array([[3, PH, 4, PH, 5]])
        out = self.model(ids, pixel_values=px, grid_thws=grids)
        mx.eval(out)
        self.assertEqual(out.shape[1], 5 - 2 + 4 + 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
