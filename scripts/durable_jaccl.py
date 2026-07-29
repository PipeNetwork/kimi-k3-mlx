#!/usr/bin/env python3
"""Launch and monitor two durable JACCL ranks without SSH-owned worker PTYs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def load_jaccl_hostfile(path: Path) -> tuple[dict, list[list[str | None]]]:
    value = json.loads(path.read_text())
    hosts = value.get("hosts")
    if value.get("backend") != "jaccl" or not isinstance(hosts, list) or len(hosts) != 2:
        raise ValueError("durable launcher requires a two-host JACCL hostfile")
    devices = [host.get("rdma") for host in hosts]
    if (
        any(not isinstance(row, list) or len(row) != 2 for row in devices)
        or devices[0][0] is not None
        or devices[1][1] is not None
        or not devices[0][1]
        or not devices[1][0]
    ):
        raise ValueError(f"malformed two-rank RDMA matrix: {devices!r}")
    if not hosts[0].get("ips") or not hosts[1].get("ssh"):
        raise ValueError("hostfile needs a rank-0 IP and rank-1 SSH hostname")
    return value, devices


def shell_worker(
    repo: Path,
    command: list[str],
    environment: dict[str, str],
    exit_path: Path,
    pid_path: Path | None = None,
) -> str:
    exports = " ".join(
        f"export {name}={shlex.quote(value)};"
        for name, value in sorted(environment.items())
    )
    argv = shlex.join(command)
    publish_pid = ""
    if pid_path is not None:
        publish_pid = (
            f"printf '%s\\n' \"$$\" > {shlex.quote(str(pid_path))}.tmp; "
            f"mv {shlex.quote(str(pid_path))}.tmp {shlex.quote(str(pid_path))}; "
        )
    return (
        f"cd {shlex.quote(str(repo))}; "
        "trap '' HUP; "
        f"{publish_pid}"
        f"{exports} "
        "set +e; "
        f"{argv}; "
        "rank_exit=$?; "
        f"printf '%s\\n' \"$rank_exit\" > {shlex.quote(str(exit_path))}.tmp; "
        f"mv {shlex.quote(str(exit_path))}.tmp {shlex.quote(str(exit_path))}; "
        "exit \"$rank_exit\""
    )


def launch_ssh_worker(remote: str, script: str, log: Path) -> subprocess.Popen:
    """Run a rank under the forced-PTY SSH context proven by mlx.launch."""
    handle = log.open("ab", buffering=0)
    try:
        return subprocess.Popen(
            [
                "ssh",
                "-tt",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                remote,
                f"stty -echo; {script}",
            ],
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        handle.close()


def ssh(remote: str, script: str, *, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", remote, script],
        check=True,
        text=True,
        capture_output=capture,
    )


def remote_text(remote: str, path: Path) -> str | None:
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            remote,
            f"test -f {shlex.quote(str(path))} && /bin/cat {shlex.quote(str(path))}",
        ],
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def wait_for_remote_pid(
    remote: str,
    path: Path,
    *,
    timeout: float = 10,
) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = remote_text(remote, path)
        if value is not None:
            try:
                pid = int(value)
            except ValueError as error:
                raise RuntimeError(f"invalid remote rank PID in {path}: {value!r}") from error
            if pid > 0:
                return pid
            raise RuntimeError(f"invalid remote rank PID in {path}: {pid}")
        time.sleep(0.1)
    raise TimeoutError(f"remote rank did not publish {path} within {timeout:g} seconds")


def rank_alive(remote: str | None, pid: int) -> bool:
    if remote is None:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            remote,
            f"kill -0 {pid} 2>/dev/null",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # A transient SSH failure is not evidence that a durable rank died.
    return result.returncode in (0, 255)


def wait_for_local_coordinator(
    host: str,
    port: int,
    process: subprocess.Popen,
    *,
    timeout: float = 60,
) -> None:
    """Wait for rank 0 to listen without consuming JACCL's one peer socket."""
    deadline = time.monotonic() + timeout
    endpoint = f"TCP@{host}:{port}"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"rank 0 exited with status {process.returncode} before "
                "opening the JACCL coordinator"
            )
        listener = subprocess.run(
            ["lsof", "-nP", f"-i{endpoint}", "-sTCP:LISTEN"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if listener.returncode == 0:
            return
        time.sleep(0.1)
    raise TimeoutError(
        f"rank 0 did not listen on {host}:{port} within {timeout:g} seconds"
    )


def launch(args: argparse.Namespace) -> int:
    if not SAFE_RUN_ID.fullmatch(args.run_id):
        raise ValueError(f"unsafe run id: {args.run_id!r}")
    hostfile, devices = load_jaccl_hostfile(args.hostfile.resolve())
    repo = args.repo.resolve()
    python = args.python.resolve()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("missing rank command after --")

    run_dir = repo / "work" / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    remote = hostfile["hosts"][1]["ssh"]
    remote_run_dir = Path(args.remote_repo) / "work" / "runs" / args.run_id
    ssh(
        remote,
        f"test ! -e {shlex.quote(str(remote_run_dir))} && "
        f"mkdir -p {shlex.quote(str(remote_run_dir))}",
    )

    device_path = run_dir / "ibv-devices.json"
    device_path.write_text(json.dumps(devices) + "\n")
    remote_device_path = remote_run_dir / "ibv-devices.json"
    subprocess.run(
        ["scp", "-q", str(device_path), f"{remote}:{remote_device_path}"],
        check=True,
    )

    coordinator = f"{hostfile['hosts'][0]['ips'][0]}:{args.port}"
    hostfile_sha256 = hashlib.sha256(args.hostfile.read_bytes()).hexdigest()
    common = {
        "KIMI_HOSTFILE_SHA256": hostfile_sha256,
        "KIMI_RUN_ID": args.run_id,
        "MLX_JACCL_COORDINATOR": coordinator,
        "MLX_METAL_FAST_SYNCH": "1",
    }
    local_log = run_dir / "rank0.log"
    local_exit = run_dir / "rank0.exit"
    local_pid_path = run_dir / "rank0.pid"
    local_env = {
        **common,
        "MLX_IBV_DEVICES": str(device_path),
        "MLX_RANK": "0",
    }
    local_script = shell_worker(
        repo, command, local_env, local_exit, local_pid_path
    )
    local_process = launch_ssh_worker(
        hostfile["hosts"][0]["ssh"], local_script, local_log
    )

    remote_log = remote_run_dir / "rank1.log"
    remote_exit = remote_run_dir / "rank1.exit"
    remote_env = {
        **common,
        "MLX_IBV_DEVICES": str(remote_device_path),
        "MLX_RANK": "1",
    }
    remote_script = shell_worker(
        Path(args.remote_repo), command, remote_env, remote_exit
    )
    remote_pid_path = remote_run_dir / "rank1.pid"
    screen_name = f"kimi-{hashlib.sha256(args.run_id.encode()).hexdigest()[:16]}"
    remote_bootstrap = (
        f"exec </dev/null >>{shlex.quote(str(remote_log))} 2>&1; "
        f"printf '%s\\n' \"$$\" > {shlex.quote(str(remote_pid_path))}.tmp; "
        f"mv {shlex.quote(str(remote_pid_path))}.tmp "
        f"{shlex.quote(str(remote_pid_path))}; "
        f"{remote_script}"
    )
    try:
        ssh(
            remote,
            f"/usr/bin/screen -DmS {shlex.quote(screen_name)} "
            f"/bin/bash -lc {shlex.quote(remote_bootstrap)}",
        )
        remote_pid = wait_for_remote_pid(remote, remote_pid_path)
    except Exception:
        os.killpg(local_process.pid, signal.SIGTERM)
        raise

    metadata = {
        "schema": 1,
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo),
        "remote_repo": args.remote_repo,
        "remote": remote,
        "python": str(python),
        "command": command,
        "backend": "jaccl",
        "transport": "thunderbolt-rdma",
        "coordinator": coordinator,
        "hostfile": str(args.hostfile.resolve()),
        "hostfile_sha256": hostfile_sha256,
        "devices": devices,
        "ranks": [
            {
                "rank": 0,
                "pid": local_process.pid,
                "log": str(local_log),
                "exit": str(local_exit),
            },
            {
                "rank": 1,
                "pid": remote_pid,
                "screen": screen_name,
                "log": str(remote_log),
                "exit": str(remote_exit),
            },
        ],
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[durable-jaccl] launched {args.run_id}: "
        f"rank0 pid={local_process.pid}, rank1 {remote} pid={remote_pid}"
    )
    print(f"[durable-jaccl] metadata={run_dir / 'metadata.json'}")
    return 0


def load_metadata(repo: Path, run_id: str) -> dict:
    path = repo.resolve() / "work" / "runs" / run_id / "metadata.json"
    value = json.loads(path.read_text())
    if value.get("run_id") != run_id or value.get("backend") != "jaccl":
        raise ValueError(f"invalid durable run metadata: {path}")
    return value


def wait(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    metadata = load_metadata(repo, args.run_id)
    remote = metadata["remote"]
    local_exit = Path(metadata["ranks"][0]["exit"])
    remote_exit = Path(metadata["ranks"][1]["exit"])
    local_summary = repo / "work" / "benchmarks" / (
        f"{args.run_id}-summary-rank0.json"
    )
    remote_summary = Path(metadata["remote_repo"]) / "work" / "benchmarks" / (
        f"{args.run_id}-summary-rank1.json"
    )
    deadline = time.monotonic() + args.timeout
    next_report = 0.0
    while time.monotonic() < deadline:
        local_done = local_summary.is_file() and local_summary.stat().st_size > 0
        remote_done = (
            subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=5",
                    remote,
                    f"test -s {shlex.quote(str(remote_summary))}",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
        if args.completion == "summaries" and local_done and remote_done:
            print(f"[durable-jaccl] both ranks completed {args.run_id}")
            return 0

        local_code = local_exit.read_text().strip() if local_exit.is_file() else None
        remote_code = remote_text(remote, remote_exit)
        if any(code not in (None, "0") for code in (local_code, remote_code)):
            print(
                f"[durable-jaccl] rank exits: "
                f"rank0={local_code}, rank1={remote_code}",
                file=sys.stderr,
            )
            return 1
        if local_code == remote_code == "0" and args.completion == "exit":
            print(f"[durable-jaccl] both ranks exited successfully for {args.run_id}")
            return 0
        if local_code == remote_code == "0" and not (local_done and remote_done):
            print(
                "[durable-jaccl] ranks exited without both summaries",
                file=sys.stderr,
            )
            return 1

        now = time.monotonic()
        if now >= next_report:
            local_alive = rank_alive(None, int(metadata["ranks"][0]["pid"]))
            remote_alive = rank_alive(remote, int(metadata["ranks"][1]["pid"]))
            print(
                f"[durable-jaccl] waiting {args.run_id}: "
                f"alive={int(local_alive)}/{int(remote_alive)} "
                f"summaries={int(local_done)}/{int(remote_done)}",
                flush=True,
            )
            if not local_alive or not remote_alive:
                return 1
            next_report = now + args.report_interval
        time.sleep(args.poll_interval)
    print(f"[durable-jaccl] timed out waiting for {args.run_id}", file=sys.stderr)
    return 124


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--run-id", required=True)
    launch_parser.add_argument("--hostfile", type=Path, required=True)
    launch_parser.add_argument("--repo", type=Path, required=True)
    launch_parser.add_argument("--remote-repo", required=True)
    launch_parser.add_argument("--python", type=Path, required=True)
    launch_parser.add_argument("--port", type=int, default=32323)
    launch_parser.add_argument("command", nargs=argparse.REMAINDER)

    wait_parser = subparsers.add_parser("wait")
    wait_parser.add_argument("--run-id", required=True)
    wait_parser.add_argument("--repo", type=Path, required=True)
    wait_parser.add_argument("--timeout", type=float, default=7200)
    wait_parser.add_argument("--poll-interval", type=float, default=2)
    wait_parser.add_argument("--report-interval", type=float, default=30)
    wait_parser.add_argument(
        "--completion",
        choices=("summaries", "exit"),
        default="summaries",
        help="Require benchmark summaries (production) or only zero rank exits.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return launch(args) if args.action == "launch" else wait(args)


if __name__ == "__main__":
    raise SystemExit(main())
