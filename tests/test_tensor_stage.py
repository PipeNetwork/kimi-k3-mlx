"""Rank-local tensor checkpoint conversion and strict loader tests."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models import kimi_k3_uvmax
from mlx_lm.utils import save_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.make_tiny_uvmax_fixture import config
from scripts.shard_uvmax_file import OfflineGroup, sha256_file, shard_file
from scripts.tensor_stage import build_index, load_tensor_stage


class TensorStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="kimi-tensor-test-")
        cls.root = Path(cls.temporary.name)
        cls.source = cls.root / "source"
        cfg = config()
        mx.random.seed(7)
        model = kimi_k3_uvmax.Model(kimi_k3_uvmax.ModelArgs.from_dict(cfg))
        mx.eval(model.parameters())
        save_model(cls.source, model)
        (cls.source / "config.json").write_text(json.dumps(cfg) + "\n")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_each_rank_round_trips_strictly(self):
        source_file = self.source / "model.safetensors"
        digest = sha256_file(source_file)
        for rank in (0, 1):
            with self.subTest(rank=rank):
                stage = self.root / f"rank{rank}"
                stage.mkdir()
                shutil.copy2(self.source / "config.json", stage / "config.json")
                result = shard_file(
                    self.source,
                    source_file,
                    stage / "model.safetensors",
                    rank,
                    2,
                    digest,
                )
                self.assertEqual(result["source_sha256"], digest)
                index = build_index(stage)
                loaded, _ = load_tensor_stage(
                    stage, OfflineGroup(rank, 2), tokenizer=False
                )
                self.assertEqual(
                    set(index["weight_map"]),
                    {name for name, _ in tree_flatten(loaded.parameters())},
                )


if __name__ == "__main__":
    unittest.main()
