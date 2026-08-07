#!/usr/bin/env python3
"""Validate, hash, index, and seal one complete Kimi-K3 TP stage."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from mlx.utils import tree_flatten
from mlx_lm.utils import load_model
from safetensors import safe_open

try:
    from scripts.shard_uvmax_file import OfflineGroup, sha256_file
    from scripts.tensor_stage import build_index
except ModuleNotFoundError:
    from shard_uvmax_file import OfflineGroup, sha256_file
    from tensor_stage import build_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    stage = args.stage.resolve()
    files = sorted(stage.glob("model-?????-of-00185.safetensors"))
    if len(files) != 185:
        raise RuntimeError(f"expected 185 TP files, found {len(files)} in {stage}")

    with tempfile.TemporaryDirectory(prefix="kimi-tp-finalize-") as name:
        template = Path(name)
        shutil.copy2(stage / "config.json", template / "config.json")
        model, _ = load_model(template, lazy=True, strict=False)
        model.shard(OfflineGroup(args.rank, 2))
        expected = dict(tree_flatten(model.parameters()))

    seen: dict[str, tuple[int, ...]] = {}
    records = []
    for number, path in enumerate(files, 1):
        if path.name != f"model-{number:05d}-of-00185.safetensors":
            raise RuntimeError(f"noncanonical TP filename: {path.name}")
        with safe_open(path, framework="numpy") as source:
            for key in source.keys():
                if key in seen:
                    raise RuntimeError(f"duplicate tensor: {key}")
                seen[key] = tuple(source.get_slice(key).get_shape())
        print(f"[finalize rank {args.rank}] hashing {path.name}", flush=True)
        records.append(
            {"name": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
        )
    missing = sorted(expected.keys() - seen.keys())
    extra = sorted(seen.keys() - expected.keys())
    bad_shapes = sorted(
        key for key in expected.keys() & seen.keys() if tuple(expected[key].shape) != seen[key]
    )
    if missing or extra or bad_shapes:
        raise RuntimeError(
            f"stage mismatch missing={missing[:5]} extra={extra[:5]} "
            f"bad_shapes={bad_shapes[:5]}"
        )

    index = build_index(stage)
    source_manifest = json.loads(args.source_manifest.read_text())
    total = sum(record["size"] for record in records)
    manifest = {
        "format": 1,
        "source": source_manifest["source"],
        "tensor": {"rank": args.rank, "world_size": 2},
        "weights": {
            "bytes": total,
            "gib": total / (1024**3),
            "files": records,
            "sha256_verified": True,
            "tensor_count": len(index["weight_map"]),
        },
        "complete": True,
    }
    temporary = stage / ".stage-manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(stage / "stage-manifest.json")
    print(
        json.dumps(
            {
                "stage": str(stage),
                "rank": args.rank,
                "files": len(records),
                "tensors": len(seen),
                "gib": manifest["weights"]["gib"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
