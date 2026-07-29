#!/usr/bin/env python3
"""Load and validate a rank-local tensor-parallel Kimi-K3 checkpoint."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.utils import load_model, load_tokenizer
from safetensors import safe_open


def weight_files(stage: Path) -> list[Path]:
    index_path = stage / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        names = sorted(set(index["weight_map"].values()))
        files = [stage / name for name in names]
    else:
        files = sorted(stage.glob("model*.safetensors"))
    if not files or any(not path.is_file() for path in files):
        raise FileNotFoundError(f"incomplete tensor stage: {stage}")
    return files


def load_tensor_stage(stage: Path, group, *, tokenizer: bool = True):
    """Shard an empty architecture first, then load rank-local tensor slices.

    MLX's normal loader attaches checkpoint tensors before ``Model.shard``.
    Rank-local checkpoints need the inverse order so strict shape validation is
    performed against the already-sharded module tree.
    """
    stage = stage.resolve()
    config_path = stage / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    with tempfile.TemporaryDirectory(prefix="kimi-tp-architecture-") as name:
        template = Path(name)
        shutil.copy2(config_path, template / "config.json")
        config = json.loads(config_path.read_text())
        if model_file := config.get("model_file"):
            shutil.copy2(stage / model_file, template / model_file)
        model, loaded_config = load_model(template, lazy=True, strict=False)

    if not hasattr(model, "shard"):
        raise TypeError("the selected loader does not support tensor parallelism")
    model.shard(group)

    weights: dict[str, mx.array] = {}
    for path in weight_files(stage):
        shard = mx.load(path)
        overlap = weights.keys() & shard.keys()
        if overlap:
            raise RuntimeError(f"duplicate tensors in {path}: {sorted(overlap)[:5]}")
        weights.update(shard)

    expected = dict(tree_flatten(model.parameters()))
    missing = sorted(expected.keys() - weights.keys())
    extra = sorted(weights.keys() - expected.keys())
    if missing or extra:
        raise RuntimeError(
            f"tensor stage key mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    if not tokenizer:
        return model, loaded_config
    tok = load_tokenizer(
        stage,
        {"trust_remote_code": True},
        eos_token_ids=loaded_config.get("eos_token_id"),
    )
    return model, tok


def build_index(stage: Path) -> dict:
    """Write an exact index for all safetensors currently in ``stage``."""
    stage = stage.resolve()
    mapping: dict[str, str] = {}
    total_size = 0
    files = sorted(stage.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no model safetensors in {stage}")
    for path in files:
        total_size += path.stat().st_size
        with safe_open(path, framework="numpy") as source:
            for key in source.keys():
                if key in mapping:
                    raise RuntimeError(f"duplicate tensor {key} in {path}")
                mapping[key] = path.name
    index = {"metadata": {"total_size": total_size}, "weight_map": mapping}
    temporary = stage / ".model.safetensors.index.json.partial"
    temporary.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    temporary.replace(stage / "model.safetensors.index.json")
    return index
