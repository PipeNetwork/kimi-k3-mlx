"""The bf16 (no-quant) profile and the converter's resume journal.

Both are covered here because both failed in ways that produce a *loadable* build
rather than a crash: a bf16 build that still carries a `quantization` block makes
the loader build quantized modules and then hunt for scales that were never
written, and a resume that keeps orphaned shards writes a second copy of a layer
that nothing in the index points at.
"""

import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import unittest

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_spec = importlib.util.spec_from_file_location(
    "convert", os.path.join(os.path.dirname(__file__), "..", "scripts", "convert.py"))
convert = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert)


class TestBf16Profile(unittest.TestCase):
    def test_quantizes_nothing(self):
        """Every tier must be None -- one non-None tier silently quantizes a slice."""
        spec = convert.profile_spec("bf16", None)
        self.assertEqual(spec, {"expert": None, "other": None, "gate": None,
                                "global_q": None})

    def test_other_profiles_still_quantize(self):
        """Guards against a bf16 short-circuit that swallows the real profiles."""
        for p in ("mxfp4", "3bit", "2bit"):
            with self.subTest(profile=p):
                self.assertIsNotNone(convert.profile_spec(p, None)["expert"])


class TestBf16BuildLoads(unittest.TestCase):
    """A bf16 build writes no `quantization` key. mlx-lm keys the whole quantized
    path off that key's presence, so its absence is the entire contract -- and it
    is only observable by loading a build that omits it."""

    def test_round_trips_through_load_model(self):
        import dataclasses
        from mlx_lm.models import kimi_k3
        from mlx_lm.utils import load_model
        sys.path.insert(0, os.path.dirname(__file__))
        from test_kimi_k3 import tiny_args

        args = tiny_args()
        model = kimi_k3.Model(args)
        mx.eval(model.parameters())
        ids = mx.array([[3, 9, 27, 81]])
        want = model(ids)
        mx.eval(want)

        d = tempfile.mkdtemp()
        from mlx.utils import tree_flatten
        w = dict(tree_flatten(model.parameters()))
        mx.save_safetensors(os.path.join(d, "model.safetensors"), w, {"format": "mlx"})
        cfg = dataclasses.asdict(args)
        self.assertNotIn("quantization", cfg)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(cfg, f)

        loaded, _ = load_model(pathlib.Path(d), lazy=False)
        got = loaded(ids)
        mx.eval(got)
        self.assertTrue(bool(mx.all(got == want).item()),
                        "reloaded bf16 build does not reproduce the source model")


class TestShardWriterJournal(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def _writer(self):
        return convert.ShardWriter(self.dir, limit=1 << 20)

    def test_checkpoint_round_trips_state(self):
        w = self._writer()
        w.add("a", mx.zeros((4, 4), dtype=mx.bfloat16))
        w.checkpoint({"done_layers": 3, "overrides": {"m": {"bits": 4}}})

        w2 = self._writer()
        st = w2.resume_from()
        self.assertEqual(st["done_layers"], 3)
        self.assertEqual(st["overrides"], {"m": {"bits": 4}})
        self.assertEqual(w2.weight_map, w.weight_map)
        self.assertEqual(w2.total, w.total)

    def test_resume_deletes_shards_written_after_the_checkpoint(self):
        """A size-flush past the last checkpoint leaves a shard the journal never
        names. Its layer gets replayed, so keeping the file duplicates tensors."""
        w = self._writer()
        w.add("a", mx.zeros((4, 4), dtype=mx.bfloat16))
        w.checkpoint({"done_layers": 1})
        w.add("b", mx.zeros((1024, 1024), dtype=mx.bfloat16))  # 2 MB > limit -> flushes
        self.assertEqual(len(w.shards), 2)

        w2 = self._writer()
        st = w2.resume_from()
        self.assertEqual(st["done_layers"], 1)
        self.assertEqual(len(w2.shards), 1)
        self.assertNotIn("b", w2.weight_map)
        on_disk = [f for f in os.listdir(self.dir) if f.endswith(".safetensors")]
        self.assertEqual(len(on_disk), 1)

    def test_resume_without_a_journal_refuses_a_finished_build(self):
        w = self._writer()
        w.add("a", mx.zeros((4, 4), dtype=mx.bfloat16))
        w.finalize()
        self.assertFalse(os.path.exists(os.path.join(self.dir, convert.PROGRESS)),
                         "finalize must clear the journal")
        with self.assertRaises(SystemExit):
            self._writer().resume_from()

    def test_fresh_run_refuses_a_populated_directory(self):
        w = self._writer()
        w.add("a", mx.zeros((4, 4), dtype=mx.bfloat16))
        w.flush()
        with self.assertRaises(SystemExit):
            self._writer().guard_fresh()

    def test_fresh_run_accepts_an_empty_directory(self):
        self._writer().guard_fresh()

    def test_resume_of_a_never_started_run_is_not_an_error(self):
        self.assertIsNone(self._writer().resume_from())

    def test_journal_survives_a_write_that_dies_midway(self):
        """checkpoint() swaps atomically, so a torn write cannot orphan the run."""
        w = self._writer()
        w.add("a", mx.zeros((4, 4), dtype=mx.bfloat16))
        w.checkpoint({"done_layers": 1})
        jp = os.path.join(self.dir, convert.PROGRESS)
        with open(jp) as f:
            good = f.read()
        with open(os.path.join(self.dir, convert.PROGRESS + ".tmp"), "w") as f:
            f.write('{"done_layers": 9, "shar')          # interrupted write
        with open(jp) as f:
            self.assertEqual(f.read(), good)
        self.assertEqual(self._writer().resume_from()["done_layers"], 1)


if __name__ == "__main__":
    unittest.main()
