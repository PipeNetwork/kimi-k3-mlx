import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.distributed_groups import init_distributed_groups, secondary_coordinator


class FakeGroup:
    def __init__(self, rank=0, size=2):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


class TestDistributedGroups(unittest.TestCase):
    def test_secondary_coordinator_uses_adjacent_port(self):
        self.assertEqual(secondary_coordinator("10.0.0.1:32323"), "10.0.0.1:32324")
        with self.assertRaises(ValueError):
            secondary_coordinator("10.0.0.1:65535")

    def test_jaccl_initializes_independent_control_backend(self):
        payload, control = FakeGroup(), FakeGroup()
        addresses = []

        def initialize(*, strict, backend):
            self.assertTrue(strict)
            self.assertEqual(backend, "jaccl")
            addresses.append(os.environ["MLX_JACCL_COORDINATOR"])
            return payload if len(addresses) == 1 else control

        environment = {
            "MLX_JACCL_COORDINATOR": "10.0.0.1:32323",
            "KIMI_JACCL_CONTROL_COORDINATOR": "10.0.0.1:32324",
        }
        with patch.dict(os.environ, environment, clear=False):
            with patch("scripts.distributed_groups.mx.distributed.init", initialize):
                result = init_distributed_groups("jaccl")
            self.assertEqual(
                os.environ["MLX_JACCL_COORDINATOR"], "10.0.0.1:32323"
            )
        self.assertEqual(result, (payload, control))
        self.assertEqual(addresses, ["10.0.0.1:32323", "10.0.0.1:32324"])

    def test_ring_reuses_one_group(self):
        group = FakeGroup()
        with patch("scripts.distributed_groups.mx.distributed.init", return_value=group):
            self.assertEqual(init_distributed_groups("ring"), (group, group))


if __name__ == "__main__":
    unittest.main()
