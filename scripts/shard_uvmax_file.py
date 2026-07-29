#!/usr/bin/env python3
"""Write one rank-local TP safetensor from one UVMAX source shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.utils import load_model
from safetensors import safe_open


class OfflineGroup:
    """The rank/size surface used by MLX's deterministic sharding helpers."""

    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def source_language_keys(path: Path) -> tuple[list[str], int]:
    with safe_open(path, framework="numpy") as source:
        keys = list(source.keys())
    language = [key for key in keys if key.startswith("language_model.")]
    return language, len(keys) - len(language)


def validate_output(path: Path, arrays: dict[str, mx.array]) -> None:
    with safe_open(path, framework="numpy") as output:
        keys = list(output.keys())
        if keys != sorted(arrays):
            raise RuntimeError(f"output key mismatch in {path}")
        for key, array in arrays.items():
            tensor = output.get_slice(key)
            if tuple(tensor.get_shape()) != tuple(array.shape):
                raise RuntimeError(f"output shape mismatch for {key}")


def shard_file(
    source_root: Path,
    source_file: Path,
    output: Path,
    rank: int,
    world_size: int,
) -> dict:
    source_root = source_root.resolve()
    source_file = source_file.resolve()
    output = output.resolve()
    if not source_file.is_file() or source_file.parent != source_root:
        raise FileNotFoundError(f"source shard is not in source root: {source_file}")
    config_path = source_root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    language_keys, skipped = source_language_keys(source_file)
    if not language_keys:
        raise RuntimeError(f"source shard contains no language tensors: {source_file}")

    with tempfile.TemporaryDirectory(prefix="kimi-tp-template-") as template_name:
        template = Path(template_name)
        shutil.copy2(config_path, template / "config.json")
        os.symlink(source_file, template / source_file.name)
        model, _ = load_model(template, lazy=True, strict=False)
        model.shard(OfflineGroup(rank, world_size))
        parameters = dict(tree_flatten(model.parameters()))
        missing = [key for key in language_keys if key not in parameters]
        if missing:
            raise RuntimeError(
                f"{source_file.name}: {len(missing)} tensors missing after sharding: "
                + ", ".join(missing[:5])
            )
        arrays = {key: parameters[key] for key in language_keys}

        output.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.name}.partial.",
            suffix=".safetensors",
        )
        os.close(handle)
        temporary = Path(temporary_name)
        # MLX's safetensors writer expects to create the destination itself.
        temporary.unlink()
        try:
            mx.save_safetensors(temporary, arrays)
            validate_output(temporary, arrays)
            os.replace(temporary, output)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    result = {
        "format": 1,
        "rank": rank,
        "world_size": world_size,
        "source": source_file.name,
        "source_bytes": source_file.stat().st_size,
        "source_sha256": sha256_file(source_file),
        "output": output.name,
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
        "language_tensors": len(language_keys),
        "skipped_non_language_tensors": skipped,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    args = parser.parse_args()
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("invalid TP rank/world size")
    shard_file(
        args.source_root,
        args.source_file,
        args.output,
        args.rank,
        args.world_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
