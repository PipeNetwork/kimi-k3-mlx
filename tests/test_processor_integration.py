"""Kimi-K3's real image processor -> the MLX vision tower.

Everything else tests the port against the reference *model*. This tests it
against the reference *processor*: the actual `KimiK3VisionProcessor` from the
model repo turns a PIL image into `(L, 3, 14, 14)` patches and a `(t, h, w)`
grid, and those must drop into the MLX tower with no reshaping and produce
exactly the token count the processor itself predicts.

That last equality is the real check. `media_tokens_calculator` is what decides
how many `<|media_pad|>`-expanded slots the prompt reserves, so if the tower
disagreed the merge would fail or, worse, silently misalign text and image.

Skipped unless Kimi-K3-src/ has been downloaded.

Run:  python3 tests/test_processor_integration.py
"""

import json
import os
import sys
import types
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

SRC = os.path.join(ROOT, "Kimi-K3-src")
NEEDED = ["kimi_k3_vision_processing.py", "media_utils.py", "preprocessor_config.json"]
HAVE_SRC = all(os.path.exists(os.path.join(SRC, f)) for f in NEEDED)

import mlx.core as mx  # noqa: E402
import kimi_k3_vision as mlxv  # noqa: E402


@unittest.skipUnless(HAVE_SRC, "Kimi-K3-src not downloaded")
class TestProcessorIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PIL import Image

        cls.Image = Image
        if "k3src" not in sys.modules:
            pkg = types.ModuleType("k3src")
            pkg.__path__ = [SRC]
            sys.modules["k3src"] = pkg
        from k3src import kimi_k3_vision_processing as k3vp

        cfg = json.load(open(os.path.join(SRC, "preprocessor_config.json")))
        kw = {k: v for k, v in cfg.items()
              if k not in ("processor_class", "image_processor_type", "auto_map")}
        cls.proc = k3vp.KimiK3VisionProcessor(**kw)

        vc = json.load(open(os.path.join(ROOT, "reference", "config.json")))["vision_config"]
        cls.vcfg = mlxv.VisionConfig.from_dict(vc)
        cls.tower = mlxv.VisionModel(cls.vcfg)
        mx.eval(cls.tower.parameters())

    def _run(self, w, h):
        rng = np.random.default_rng(w * h)
        img = self.Image.fromarray((rng.random((h, w, 3)) * 255).astype(np.uint8))
        out = self.proc.preprocess([{"type": "image", "image": img}])
        px, gt = out["pixel_values"], out["grid_thws"]
        grids = [tuple(int(v) for v in g) for g in gt.tolist()]
        predicted = self.proc.media_tokens_calculator({"type": "image", "image": img})
        feats = self.tower(mx.array(px.numpy()), grids)
        mx.eval(feats)
        return px, grids, predicted, feats

    def test_patch_layout_is_what_the_tower_expects(self):
        px, grids, _, _ = self._run(611, 437)
        t, h, w = grids[0]
        self.assertEqual(px.shape[1:], (3, self.vcfg.patch_size, self.vcfg.patch_size))
        self.assertEqual(px.shape[0], t * h * w)

    def test_token_count_matches_processor_across_aspect_ratios(self):
        for w, h in ((611, 437), (224, 224), (1920, 1080), (300, 900)):
            with self.subTest(size=f"{w}x{h}"):
                _, grids, predicted, feats = self._run(w, h)
                t, gh, gw = grids[0]
                kh, kw = self.vcfg.merge_kernel_size
                self.assertEqual(feats[0].shape[0], (gh // kh) * (gw // kw))
                self.assertEqual(feats[0].shape[0], predicted)
                self.assertEqual(feats[0].shape[1], self.vcfg.text_hidden_size)
                self.assertFalse(bool(mx.any(mx.isnan(feats[0]))))

    def test_multiple_images_in_one_batch(self):
        rng = np.random.default_rng(7)
        medias = []
        for w, h in ((224, 224), (448, 224)):
            arr = (rng.random((h, w, 3)) * 255).astype(np.uint8)
            medias.append({"type": "image", "image": self.Image.fromarray(arr)})
        out = self.proc.preprocess(medias)
        grids = [tuple(int(v) for v in g) for g in out["grid_thws"].tolist()]
        feats = self.tower(mx.array(out["pixel_values"].numpy()), grids)
        mx.eval(feats)
        self.assertEqual(len(feats), 2)
        kh, kw = self.vcfg.merge_kernel_size
        for (t, gh, gw), f in zip(grids, feats):
            self.assertEqual(f.shape[0], (gh // kh) * (gw // kw))
        # different-sized images must yield different token counts
        self.assertNotEqual(feats[0].shape[0], feats[1].shape[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
