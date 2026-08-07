#!/usr/bin/env python3
"""Generate and benchmark Kimi-K3 with mandatory two-node JACCL tensor parallelism."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx

try:
    from scripts.distributed_generate import (
        benchmark_cases,
        generate_once,
        rdma_state,
        repository_state,
    )
    from scripts.tensor_stage import load_tensor_stage
except ModuleNotFoundError:
    from distributed_generate import (
        benchmark_cases,
        generate_once,
        rdma_state,
        repository_state,
    )
    from tensor_stage import load_tensor_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("weights/Kimi-K3-2bit-UVMAX-tensor-2"),
    )
    parser.add_argument("--prompt", default="Who is Albert Einstein?")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--run-id")
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup-tokens", type=int, default=0)
    parser.add_argument("--raw-prompt", action="store_true")
    return parser.parse_args()


def load_manifest(stage: Path, rank: int, size: int) -> dict:
    path = stage / "stage-manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing stage manifest: {path}")
    manifest = json.loads(path.read_text())
    tensor = manifest.get("tensor", {})
    if not manifest.get("complete"):
        raise RuntimeError(f"stage is incomplete: {path}")
    if manifest.get("weights", {}).get("sha256_verified") is not True:
        raise RuntimeError(f"stage has no full SHA-256 proof: {path}")
    if tensor.get("rank") != rank or tensor.get("world_size") != size:
        raise RuntimeError(
            f"stage rank mismatch: manifest={tensor}, runtime={rank}/{size}"
        )
    files = manifest["weights"]["files"]
    if len(files) != 185:
        raise RuntimeError(f"stage manifest has {len(files)} files, not 185")
    for entry in files:
        weight = stage / entry["name"]
        if not weight.is_file() or weight.stat().st_size != entry["size"]:
            raise RuntimeError(f"missing or truncated stage weight: {weight}")
    return manifest


def main() -> int:
    args = parse_args()
    if args.max_tokens < 1 or args.repetitions < 1 or args.warmup_tokens < 0:
        raise ValueError("invalid token or repetition count")
    cases = benchmark_cases(args)

    # Deliberately no TCP/ring fallback: every model collective uses this mesh.
    group = mx.distributed.init(strict=True, backend="jaccl")
    rank, size = group.rank(), group.size()
    if size != 2:
        raise RuntimeError(f"this deployment requires exactly two JACCL ranks, got {size}")
    mx.eval(
        mx.distributed.all_sum(
            mx.ones((10,), dtype=mx.float32), group=group, stream=mx.cpu
        )
    )
    rdma = rdma_state()

    stage = args.model_root.resolve() / f"rank{rank}"
    manifest = load_manifest(stage, rank, size)
    run_id = args.run_id or os.environ.get("KIMI_RUN_ID") or time.strftime(
        "%Y%m%dT%H%M%S"
    )
    host = socket.gethostname()
    hardware = {**mx.device_info(), "machine": platform.machine()}
    print(
        f"[rank {rank}] host={host} backend=jaccl "
        f"rdma={','.join(rdma['active_ports'])} strategy=tensor-parallel stage={stage}",
        flush=True,
    )

    load_started = time.perf_counter()
    model, tokenizer = load_tensor_stage(stage, group)
    load_seconds = time.perf_counter() - load_started
    print(f"[rank {rank}] model loaded in {load_seconds:.3f}s", flush=True)

    if args.warmup_tokens:
        mx.random.seed(args.seed)
        print(f"[rank {rank}] warm-up: {args.warmup_tokens} tokens", flush=True)
        generate_once(
            model,
            tokenizer,
            cases[0],
            args,
            rank,
            max_tokens=args.warmup_tokens,
            show_text=False,
        )

    output_dir = Path("work/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    multiple = len(cases) > 1 or args.repetitions > 1
    repo = repository_state()
    for repetition in range(1, args.repetitions + 1):
        for case in cases:
            mx.random.seed(args.seed + repetition - 1)
            if hasattr(mx, "reset_peak_memory"):
                mx.reset_peak_memory()
            if rank == 0:
                print(
                    f"\n=== {case['id']} repetition {repetition}/{args.repetitions} ===",
                    flush=True,
                )
            response, generated_text = generate_once(model, tokenizer, case, args, rank)
            suffix = f"-{case['id']}-r{repetition}" if multiple else ""
            record_id = f"{run_id}{suffix}"
            record = {
                "schema": 1,
                "run_id": record_id,
                "suite_run_id": run_id,
                "case_id": case["id"],
                "repetition": repetition,
                "rank": rank,
                "world_size": size,
                "strategy": "tensor-parallel",
                "backend": "jaccl",
                "transport": "thunderbolt-rdma",
                "communication": {
                    "collective_group": "jaccl-rdma",
                    "mode": "tensor-parallel-all-reduce",
                },
                "rdma": rdma,
                "jaccl_hostfile_sha256": os.environ.get("KIMI_HOSTFILE_SHA256"),
                "jaccl_ring": os.environ.get("MLX_JACCL_RING") == "1",
                "mlx_metal_fast_synch": os.environ.get("MLX_METAL_FAST_SYNCH"),
                "host": host,
                "platform": platform.platform(),
                "hardware": hardware,
                "code": repo,
                "versions": {"mlx": version("mlx"), "mlx_lm": version("mlx-lm")},
                "checkpoint": manifest["source"],
                "stage": manifest["tensor"],
                "stage_gib": manifest["weights"]["gib"],
                "load_seconds": load_seconds,
                "warmup_tokens": args.warmup_tokens,
                "prompt": case["prompt"],
                "raw_prompt": args.raw_prompt,
                "seed": args.seed + repetition - 1,
                "sampling": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "min_p": args.min_p,
                    "top_k": args.top_k,
                },
                "max_tokens": case["max_tokens"],
                "prompt_tokens": response.prompt_tokens,
                "prompt_tps": response.prompt_tps,
                "generation_tokens": response.generation_tokens,
                "generation_tps": response.generation_tps,
                "peak_memory_gb": response.peak_memory,
                "text": generated_text,
            }
            output = output_dir / f"{record_id}-rank{rank}.json"
            output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            records.append(record)
            print(
                f"[rank {rank}] prompt={response.prompt_tokens} "
                f"({response.prompt_tps:.3f} tok/s), generation={response.generation_tokens} "
                f"({response.generation_tps:.3f} tok/s), peak={response.peak_memory:.3f} GB, "
                f"record={output}",
                flush=True,
            )

    summary = {
        "schema": 1,
        "run_id": run_id,
        "rank": rank,
        "strategy": "tensor-parallel",
        "backend": "jaccl",
        "transport": "thunderbolt-rdma",
        "records": [record["run_id"] for record in records],
        "generation_tps": {
            "median": statistics.median(r["generation_tps"] for r in records),
            "min": min(r["generation_tps"] for r in records),
            "max": max(r["generation_tps"] for r in records),
        },
        "prompt_tps": {
            "median": statistics.median(r["prompt_tps"] for r in records),
            "min": min(r["prompt_tps"] for r in records),
            "max": max(r["prompt_tps"] for r in records),
        },
        "peak_memory_gb": max(r["peak_memory_gb"] for r in records),
    }
    summary_path = output_dir / f"{run_id}-summary-rank{rank}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"[rank {rank}] median generation={summary['generation_tps']['median']:.3f} "
        f"tok/s, range={summary['generation_tps']['min']:.3f}-"
        f"{summary['generation_tps']['max']:.3f}, summary={summary_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
