#!/usr/bin/env python3
"""Generate and benchmark Kimi-K3 over a mandatory two-node JACCL pipeline."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load_model, load_tokenizer
from mlx.utils import tree_flatten


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("weights/Kimi-K3-2bit-UVMAX-pipeline-2"),
        help="Directory containing rank0/ and rank1/ stage checkpoints.",
    )
    parser.add_argument("--prompt", default="Who is Albert Einstein?")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Do not wrap the prompt in the checkpoint's chat template.",
    )
    return parser.parse_args()


def load_manifest(stage: Path, rank: int, size: int) -> dict:
    path = stage / "stage-manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing stage manifest: {path}")
    manifest = json.loads(path.read_text())
    pipeline = manifest.get("pipeline", {})
    if not manifest.get("complete"):
        raise RuntimeError(f"stage is incomplete: {path}")
    if pipeline.get("rank") != rank or pipeline.get("world_size") != size:
        raise RuntimeError(
            f"stage rank mismatch: manifest={pipeline}, runtime rank={rank}/{size}"
        )
    for entry in manifest["weights"]["files"]:
        weight = stage / entry["name"]
        if not weight.is_file() or weight.stat().st_size != entry["size"]:
            raise RuntimeError(f"missing or truncated stage weight: {weight}")
    return manifest


def load_pipeline(stage: Path, group):
    """Load only files present in a rank-local stage, then prune lazy layers.

    This is the local-checkpoint half of ``mlx_lm.utils.sharded_load`` without
    its repository-discovery pass.  Avoiding that first pass matters for a
    384-GiB stage and keeps the file policy explicit and auditable.
    """
    model, config = load_model(stage, lazy=True, strict=False)
    if not hasattr(model, "model") or not hasattr(model.model, "pipeline"):
        raise TypeError("the selected loader does not support pipeline inference")
    model.model.pipeline(group)
    weight_map = json.loads(
        (stage / "model.safetensors.index.json").read_text()
    )["weight_map"]
    missing = []
    for name, _ in tree_flatten(model.parameters()):
        filename = weight_map.get(name)
        if filename is None or not (stage / filename).is_file():
            missing.append((name, filename))
    if missing:
        preview = ", ".join(f"{name} -> {filename}" for name, filename in missing[:8])
        raise RuntimeError(
            f"rank-local checkpoint does not cover {len(missing)} parameters: {preview}"
        )
    mx.eval(model.parameters())
    mx.eval(mx.distributed.all_sum(mx.array(1.0), group=group, stream=mx.cpu))
    tokenizer = load_tokenizer(
        stage,
        {"trust_remote_code": True},
        eos_token_ids=config.get("eos_token_id"),
    )
    return model, tokenizer


def main() -> int:
    args = parse_args()
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")

    # Mandatory by design: this refuses to silently fall back to TCP ring/MPI.
    group = mx.distributed.init(strict=True, backend="jaccl")
    rank, size = group.rank(), group.size()
    if size != 2:
        raise RuntimeError(f"this deployment requires exactly two JACCL ranks, got {size}")

    model_root = args.model_root.resolve()
    stage = model_root / f"rank{rank}"
    manifest = load_manifest(stage, rank, size)
    run_id = args.run_id or os.environ.get("KIMI_RUN_ID") or time.strftime(
        "%Y%m%dT%H%M%S"
    )
    host = socket.gethostname()
    print(
        f"[rank {rank}] host={host} backend=jaccl "
        f"layers=[{manifest['pipeline']['layer_start']},"
        f"{manifest['pipeline']['layer_end']}) stage={stage}",
        flush=True,
    )

    mx.random.seed(args.seed)
    load_started = time.perf_counter()
    model, tokenizer = load_pipeline(stage, group)
    load_seconds = time.perf_counter() - load_started
    print(f"[rank {rank}] model loaded in {load_seconds:.3f}s", flush=True)

    if args.raw_prompt:
        prompt = args.prompt
    else:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True,
        )
    sampler = make_sampler(
        temp=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        top_k=args.top_k,
    )

    generated = []
    response = None
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=args.max_tokens,
        sampler=sampler,
    ):
        generated.append(response.text)
        if rank == 0:
            print(response.text, end="", flush=True)
    if response is None:
        raise RuntimeError("generation produced no response")
    if rank == 0:
        print(flush=True)

    record = {
        "schema": 1,
        "run_id": run_id,
        "rank": rank,
        "world_size": size,
        "backend": "jaccl",
        "transport": "thunderbolt-rdma",
        "host": host,
        "platform": platform.platform(),
        "versions": {"mlx": version("mlx"), "mlx_lm": version("mlx-lm")},
        "checkpoint": manifest["source"],
        "stage": manifest["pipeline"],
        "stage_gib": manifest["weights"]["gib"],
        "load_seconds": load_seconds,
        "prompt": args.prompt,
        "raw_prompt": args.raw_prompt,
        "seed": args.seed,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "min_p": args.min_p,
            "top_k": args.top_k,
        },
        "max_tokens": args.max_tokens,
        "prompt_tokens": response.prompt_tokens,
        "prompt_tps": response.prompt_tps,
        "generation_tokens": response.generation_tokens,
        "generation_tps": response.generation_tps,
        "peak_memory_gb": response.peak_memory,
        "text": "".join(generated),
    }
    output = Path("work/benchmarks") / f"{run_id}-rank{rank}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        f"[rank {rank}] prompt={response.prompt_tokens} "
        f"({response.prompt_tps:.3f} tok/s), generation={response.generation_tokens} "
        f"({response.generation_tps:.3f} tok/s), "
        f"peak={response.peak_memory:.3f} GB, record={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
