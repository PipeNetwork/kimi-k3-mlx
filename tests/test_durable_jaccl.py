import json
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

from scripts.durable_jaccl import (
    SAFE_RUN_ID,
    load_jaccl_hostfile,
    shell_worker,
    wait_for_local_coordinator,
    wait_for_remote_pid,
)


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

    @patch("scripts.durable_jaccl.subprocess.run")
    def test_coordinator_readiness_does_not_connect(self, run):
        run.return_value = subprocess.CompletedProcess([], 0)
        process = Mock()
        process.poll.return_value = None
        wait_for_local_coordinator("10.0.0.1", 32323, process, timeout=0.1)
        run.assert_called_once_with(
            [
                "lsof",
                "-nP",
                "-iTCP@10.0.0.1:32323",
                "-sTCP:LISTEN",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    @patch("scripts.durable_jaccl.remote_text", return_value="1234")
    def test_remote_pid_is_published_by_detached_rank(self, remote_text):
        path = Path("/tmp/rank1.pid")
        self.assertEqual(wait_for_remote_pid("beast2.local", path), 1234)
        remote_text.assert_called_once_with("beast2.local", path)


if __name__ == "__main__":
    unittest.main()
