#!/usr/bin/env python3
"""Stream one host's source shards into both rank-local TP checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SHARD = re.compile(r"model-(\d{5})-of-00185\.safetensors$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def append_journal(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as output:
        output.write(json.dumps(entry, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def run_sharder(
    python: Path,
    source_root: Path,
    source: Path,
    output: Path,
    rank: int,
    digest: str,
) -> dict:
    command = [
        str(python),
        str(Path(__file__).with_name("shard_uvmax_file.py")),
        "--source-root",
        str(source_root),
        "--source-file",
        str(source),
        "--output",
        str(output),
        "--rank",
        str(rank),
        "--trusted-source-sha256",
        digest,
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            f"sharder failed for {source.name} rank {rank}:\n"
            f"{completed.stdout}{completed.stderr}"
        )
    print(completed.stdout, end="", flush=True)
    return json.loads(completed.stdout.strip().splitlines()[-1])


def initialize_stage(source_root: Path, output: Path, rank: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for source in source_root.iterdir():
        if not source.is_file() or source.suffix == ".safetensors":
            continue
        if source.name in {"model.safetensors.index.json", "stage-manifest.json"}:
            continue
        target = output / source.name
        if not target.exists():
            shutil.copy2(source, target)
    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("distributed_pipeline", None)
    config["distributed_tensor"] = {
        "rank": rank,
        "world_size": 2,
        "source_repo": "kernelpool/Kimi-K3-2bit-UVMAX",
        "source_revision": "edb5113218df612f4a92f95145680f3f8eacd375",
    }
    temporary = output / ".config.json.partial"
    temporary.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    temporary.replace(config_path)


def remote_file_state(host: str, python: str, path: Path) -> dict | None:
    code = """import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1])
if not p.is_file(): raise SystemExit(3)
d=hashlib.sha256()
with p.open('rb') as f:
    while True:
        c=f.read(8<<20)
        if not c: break
        d.update(c)
print(json.dumps({'size':p.stat().st_size,'sha256':d.hexdigest()}))
"""
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, python, "-", str(path)],
        input=code,
        text=True,
        capture_output=True,
    )
    if completed.returncode == 3:
        return None
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip())
    return json.loads(completed.stdout)


def install_local(temporary: Path, target: Path, result: dict) -> None:
    if target.exists():
        state = {"size": target.stat().st_size, "sha256": sha256_file(target)}
        if state != {
            "size": result["output_bytes"],
            "sha256": result["output_sha256"],
        }:
            raise RuntimeError(f"existing local output differs: {target}")
        temporary.unlink()
    else:
        os.replace(temporary, target)


def install_remote(
    host: str,
    python: str,
    temporary: Path,
    target: Path,
    result: dict,
) -> None:
    state = remote_file_state(host, python, target)
    expected = {
        "size": result["output_bytes"],
        "sha256": result["output_sha256"],
    }
    if state is not None:
        if state != expected:
            raise RuntimeError(f"existing peer output differs: {host}:{target}")
        temporary.unlink()
        return
    peer_partial = target.with_name(f".{target.name}.incoming.{os.getpid()}")
    subprocess.run(
        ["scp", "-q", str(temporary), f"{host}:{peer_partial}"], check=True
    )
    code = """import hashlib,os,pathlib,sys
p,t=map(pathlib.Path,sys.argv[1:3]); want_size=int(sys.argv[3]); want=sys.argv[4]
if p.stat().st_size != want_size: raise SystemExit('peer transfer size mismatch')
d=hashlib.sha256()
with p.open('rb') as f:
    while True:
        c=f.read(8<<20)
        if not c: break
        d.update(c)
if d.hexdigest() != want: raise SystemExit('peer transfer digest mismatch')
if t.exists(): raise SystemExit('peer target appeared during transfer')
os.replace(p,t)
"""
    subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            python,
            "-",
            str(peer_partial),
            str(target),
            str(expected["size"]),
            expected["sha256"],
        ],
        input=code,
        text=True,
        check=True,
    )
    temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--peer-host", required=True)
    parser.add_argument("--peer-output", type=Path, required=True)
    parser.add_argument("--peer-python", required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--init-only", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output = args.output.resolve()
    peer_output = args.peer_output
    local_rank = args.local_rank
    peer_rank = 1 - local_rank
    # Do not resolve the venv interpreter symlink into Homebrew's bare Python.
    python = Path(sys.executable).absolute()
    initialize_stage(source_root, output, local_rank)
    subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            args.peer_host,
            "mkdir",
            "-p",
            str(peer_output),
        ],
        check=True,
    )
    if args.init_only:
        return 0

    source_manifest = json.loads((source_root / "stage-manifest.json").read_text())
    entries = {item["name"]: item for item in source_manifest["weights"]["files"]}
    journal = output / ".tensor-conversion.jsonl"
    temporary_root = Path("work/.tensor-convert-tmp").resolve()
    temporary_root.mkdir(parents=True, exist_ok=True)

    for number in range(args.first, args.last + 1):
        name = f"model-{number:05d}-of-00185.safetensors"
        if not SHARD.fullmatch(name) or name not in entries:
            raise RuntimeError(f"unsafe or unmanifested source shard: {name}")
        source = source_root / name
        entry = entries[name]
        if not source.is_file():
            append_journal(journal, {"source": name, "phase": "already-deleted"})
            continue
        if source.stat().st_size != entry["size"]:
            raise RuntimeError(f"source size mismatch: {source}")
        digest = sha256_file(source)
        if digest != entry["sha256"]:
            raise RuntimeError(f"source digest mismatch: {source}")

        print(f"[owner rank {local_rank}] {name}: verified source", flush=True)
        peer_temp = temporary_root / f"{name}.rank{peer_rank}.safetensors"
        peer_temp.unlink(missing_ok=True)
        peer_result = run_sharder(
            python, source_root, source, peer_temp, peer_rank, digest
        )
        install_remote(
            args.peer_host,
            args.peer_python,
            peer_temp,
            peer_output / name,
            peer_result,
        )
        append_journal(
            journal,
            {"source": name, "phase": "peer", "result": peer_result},
        )

        local_temp = temporary_root / f"{name}.rank{local_rank}.safetensors"
        local_temp.unlink(missing_ok=True)
        local_result = run_sharder(
            python, source_root, source, local_temp, local_rank, digest
        )
        install_local(local_temp, output / name, local_result)
        append_journal(
            journal,
            {"source": name, "phase": "local", "result": local_result},
        )

        # Exact target, pinned manifest entry, verified source and both outputs.
        # The immutable Hugging Face revision remains the recovery source.
        source.unlink()
        append_journal(journal, {"source": name, "phase": "source-deleted"})
        print(f"[owner rank {local_rank}] {name}: committed and released", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
