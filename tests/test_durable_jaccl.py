import json
import tempfile
import unittest
from pathlib import Path

from scripts.durable_jaccl import SAFE_RUN_ID, load_jaccl_hostfile, shell_worker


class TestDurableJaccl(unittest.TestCase):
    def test_hostfile_extracts_exact_rdma_matrix(self):
        value = {
            "backend": "jaccl",
            "hosts": [
                {
                    "ssh": "localhost",
                    "ips": ["10.0.0.1"],
                    "rdma": [None, "rdma_en3"],
                },
                {
                    "ssh": "beast2.local",
                    "ips": ["10.0.0.2"],
                    "rdma": ["rdma_en3", None],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hosts.json"
            path.write_text(json.dumps(value))
            loaded, devices = load_jaccl_hostfile(path)
        self.assertEqual(loaded, value)
        self.assertEqual(devices, [[None, "rdma_en3"], ["rdma_en3", None]])

    def test_hostfile_rejects_non_jaccl_or_unpaired_rdma(self):
        invalid = {
            "backend": "ring",
            "hosts": [
                {"ips": ["10.0.0.1"], "rdma": [None, None]},
                {"ssh": "beast2.local", "rdma": [None, None]},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hosts.json"
            path.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                load_jaccl_hostfile(path)

    def test_worker_quotes_command_environment_and_exit_marker(self):
        script = shell_worker(
            Path("/tmp/repo with spaces"),
            ["/bin/echo", "hello world"],
            {"MLX_RANK": "0", "VALUE": "a'b"},
            Path("/tmp/run/rank0.exit"),
        )
        self.assertIn("cd '/tmp/repo with spaces'", script)
        self.assertIn("export MLX_RANK=0;", script)
        self.assertIn("export VALUE='a'\"'\"'b';", script)
        self.assertIn("/bin/echo 'hello world'", script)
        self.assertIn("rank0.exit.tmp", script)

    def test_run_id_policy(self):
        self.assertIsNotNone(SAFE_RUN_ID.fullmatch("83646ef-canonical.1"))
        self.assertIsNone(SAFE_RUN_ID.fullmatch("../escape"))
        self.assertIsNone(SAFE_RUN_ID.fullmatch(""))


if __name__ == "__main__":
    unittest.main()
