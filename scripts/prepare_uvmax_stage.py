#!/usr/bin/env python3
"""Prepare one disk-local Kimi-K3 UVMAX pipeline stage from Hugging Face.

The upstream checkpoint is 760.68 GiB, but its shards follow decoder-layer
order.  A two-rank pipeline therefore needs only 384.5 GiB on either host.
This downloader pins the proven checkpoint revision, selects the rank-local
files, validates disk headroom, resumes safely, and records exact provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Hugging Face Hub 1.x routes large files through hf-xet. Its documented high
# performance mode increases range concurrency and is appropriate for these
# internal NVMe SSDs. It must be set before importing huggingface_hub constants.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import HfApi, hf_hub_download


DEFAULT_REPO = "kernelpool/Kimi-K3-2bit-UVMAX"
DEFAULT_REVISION = "edb5113218df612f4a92f95145680f3f8eacd375"
INDEX_FILE = "model.safetensors.index.json"
LAYER_KEY = re.compile(r"language_model\.model\.layers\.(\d+)\.")


def pipeline_bounds(num_layers: int, size: int, rank: int) -> tuple[int, int]:
    if size < 1 or not 0 <= rank < size:
        raise ValueError(f"invalid pipeline size/rank: size={size}, rank={rank}")
    logical_stage = size - rank - 1
    base, extra = divmod(num_layers, size)
    counts = [base + (stage < extra) for stage in range(size)]
    start = sum(counts[:logical_stage])
    return start, start + counts[logical_stage]


def select_stage_files(
    weight_map: dict[str, str], start: int, end: int
) -> set[str]:
    """Select local text weights, duplicating the small common endpoints.

    Embedding, final norm, final AttnRes, and LM head are retained on both
    stages because released mlx-lm generation samples on every rank. Vision
    tensors are deliberately excluded from this text-inference checkpoint.
    """
    selected: set[str] = set()
    for key, filename in weight_map.items():
        match = LAYER_KEY.match(key)
        if match:
            if start <= int(match.group(1)) < end:
                selected.add(filename)
        elif key.startswith("language_model."):
            selected.add(filename)
    return selected


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w") as tmp:
            json.dump(value, tmp, indent=2, sort_keys=True)
            tmp.write("\n")
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def file_metadata(info) -> dict[str, dict[str, Any]]:
    result = {}
    for sibling in info.siblings:
        lfs = getattr(sibling, "lfs", None)
        result[sibling.rfilename] = {
            "size": int(getattr(sibling, "size", None) or 0),
            "sha256": getattr(lfs, "sha256", None) if lfs else None,
        }
    return result


def download_one(
    repo: str, revision: str, output: Path, filename: str, expected_size: int
) -> str:
    path = Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            revision=revision,
            local_dir=output,
        )
    )
    actual = path.stat().st_size
    if actual != expected_size:
        raise IOError(f"{filename}: expected {expected_size} bytes, got {actual}")
    return filename


def download_metadata(
    repo: str, revision: str, output: Path, filename: str, expected_size: int
) -> None:
    """Keep an immutable metadata mirror separate from the stage config.

    The stage's ``config.json`` changes ``model_type`` to select the local
    loader. Re-downloading through the same local directory would otherwise
    return that intentionally modified file and fail remote-size validation on
    resume.
    """
    mirror = output / ".source-metadata"
    source = Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            revision=revision,
            local_dir=mirror,
        )
    )
    actual = source.stat().st_size
    if actual != expected_size:
        raise IOError(f"{filename}: expected {expected_size} bytes, got {actual}")
    destination = output / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=96.0,
        help="Free disk space that must remain after pending downloads.",
    )
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    start, end = pipeline_bounds(93, args.world_size, args.rank)
    output = args.output or Path(
        f"weights/Kimi-K3-2bit-UVMAX-pipeline-{args.world_size}/rank{args.rank}"
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    print(
        f"[stage] rank {args.rank}/{args.world_size}: layers [{start}, {end})",
        flush=True,
    )
    print(f"[stage] destination: {output}", flush=True)

    api = HfApi()
    info = api.model_info(args.repo, revision=args.revision, files_metadata=True)
    if info.sha != args.revision:
        raise RuntimeError(f"revision mismatch: requested {args.revision}, got {info.sha}")
    metadata = file_metadata(info)
    if INDEX_FILE not in metadata or "config.json" not in metadata:
        raise RuntimeError("checkpoint metadata is incomplete")

    small_files = sorted(
        name for name in metadata if not name.endswith(".safetensors")
    )
    for number, name in enumerate(small_files, 1):
        download_metadata(
            args.repo, args.revision, output, name, metadata[name]["size"]
        )
        print(f"[metadata {number}/{len(small_files)}] {name}", flush=True)

    index = json.loads((output / INDEX_FILE).read_text())
    weight_map = index["weight_map"]
    num_layers = 1 + max(
        int(match.group(1))
        for key in weight_map
        if (match := LAYER_KEY.match(key))
    )
    if num_layers != 93:
        raise RuntimeError(f"expected 93 decoder layers, found {num_layers}")

    selected = select_stage_files(weight_map, start, end)
    missing_metadata = selected - metadata.keys()
    if missing_metadata:
        raise RuntimeError(f"missing file metadata: {sorted(missing_metadata)}")

    required_bytes = sum(metadata[name]["size"] for name in selected)
    complete = {
        name
        for name in selected
        if (output / name).is_file()
        and (output / name).stat().st_size == metadata[name]["size"]
    }
    pending = sorted(selected - complete)
    pending_bytes = sum(metadata[name]["size"] for name in pending)
    free_bytes = shutil.disk_usage(output).free
    reserve_bytes = int(args.reserve_gib * 2**30)

    print(
        f"[stage] {len(selected)} weight shards, {required_bytes / 2**30:.3f} GiB total; "
        f"{len(pending)} pending, {pending_bytes / 2**30:.3f} GiB",
        flush=True,
    )
    print(
        f"[stage] disk free {free_bytes / 2**30:.3f} GiB; "
        f"reserve {args.reserve_gib:.3f} GiB",
        flush=True,
    )
    if pending_bytes + reserve_bytes > free_bytes:
        short = pending_bytes + reserve_bytes - free_bytes
        raise OSError(
            f"insufficient disk headroom by {short / 2**30:.3f} GiB "
            f"(pending + reserve exceeds free space)"
        )

    manifest = {
        "format": 1,
        "source": {"repo": args.repo, "revision": args.revision},
        "pipeline": {
            "rank": args.rank,
            "world_size": args.world_size,
            "layer_start": start,
            "layer_end": end,
        },
        "weights": {
            "bytes": required_bytes,
            "gib": required_bytes / 2**30,
            "files": [
                {
                    "name": name,
                    "size": metadata[name]["size"],
                    "sha256": metadata[name]["sha256"],
                }
                for name in sorted(selected)
            ],
        },
        "complete": not pending,
    }
    atomic_json(output / "stage-manifest.json", manifest)

    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    config["model_type"] = "kimi_k3_uvmax"
    config["distributed_pipeline"] = {
        **manifest["pipeline"],
        "source_repo": args.repo,
        "source_revision": args.revision,
        "checkpoint_bytes": required_bytes,
    }
    atomic_json(config_path, config)

    if args.dry_run or args.metadata_only:
        mode = "dry run" if args.dry_run else "metadata-only"
        print(f"[stage] {mode}; weights were not downloaded", flush=True)
        return 0

    if pending:
        done_bytes = required_bytes - pending_bytes
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    download_one,
                    args.repo,
                    args.revision,
                    output,
                    name,
                    metadata[name]["size"],
                ): name
                for name in pending
            }
            for number, future in enumerate(as_completed(futures), 1):
                name = future.result()
                done_bytes += metadata[name]["size"]
                print(
                    f"[weights {number}/{len(pending)}] {name} "
                    f"({done_bytes / 2**30:.3f}/{required_bytes / 2**30:.3f} GiB)",
                    flush=True,
                )

    invalid = [
        name
        for name in selected
        if not (output / name).is_file()
        or (output / name).stat().st_size != metadata[name]["size"]
    ]
    if invalid:
        raise IOError(f"stage validation failed for {invalid}")
    manifest["complete"] = True
    atomic_json(output / "stage-manifest.json", manifest)
    print(
        f"[stage] complete: rank {args.rank}, layers [{start}, {end}), "
        f"{required_bytes / 2**30:.3f} GiB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[stage] interrupted; downloads are resumable", file=sys.stderr)
        raise SystemExit(130)
