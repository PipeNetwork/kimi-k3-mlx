#!/usr/bin/env python3
"""Generate and benchmark Kimi-K3 over a mandatory two-node JACCL pipeline."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load_model, load_tokenizer
from mlx.utils import tree_flatten


def parse_active_rdma_ports(output: str) -> list[str]:
    """Extract active HCA names from macOS ``ibv_devinfo`` output."""
    active = []
    hca = None
    for line in output.splitlines():
        key, separator, value = line.strip().partition(":")
        if not separator:
            continue
        if key == "hca_id":
            hca = value.strip()
        elif key == "state" and "PORT_ACTIVE" in value and hca:
            active.append(hca)
    return active


def rdma_state() -> dict:
    try:
        output = subprocess.check_output(
            ["/usr/bin/ibv_devinfo"], text=True, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot inspect macOS RDMA devices: {error}") from error
    active = parse_active_rdma_ports(output)
    if not active:
        raise RuntimeError("JACCL run has no locally active Thunderbolt RDMA port")
    return {"active_ports": active}


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
        "--suite",
        type=Path,
        help="JSON benchmark suite; loads the model once and runs every prompt.",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=0,
        help="Unrecorded generation on the first prompt after model load.",
    )
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
    if manifest.get("weights", {}).get("sha256_verified") is not True:
        raise RuntimeError(
            f"stage has no full SHA-256 verification proof: {path}; "
            "rerun prepare_uvmax_stage.py with --verify-sha256"
        )
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


def benchmark_cases(args) -> list[dict]:
    if args.suite is None:
        return [{"id": "prompt", "prompt": args.prompt, "max_tokens": args.max_tokens}]
    value = json.loads(args.suite.read_text())
    cases = value.get("prompts") if isinstance(value, dict) else value
    if not isinstance(cases, list) or not cases:
        raise ValueError("benchmark suite must contain a non-empty prompts list")
    result = []
    for number, case in enumerate(cases, 1):
        if not isinstance(case, dict) or not isinstance(case.get("prompt"), str):
            raise ValueError(f"invalid benchmark case {number}: {case!r}")
        case_id = str(case.get("id", f"prompt-{number}"))
        if not case_id or any(not (c.isalnum() or c in "-_") for c in case_id):
            raise ValueError(f"unsafe benchmark case id: {case_id!r}")
        max_tokens = int(case.get("max_tokens", args.max_tokens))
        if max_tokens < 1:
            raise ValueError(f"case {case_id!r} has invalid max_tokens={max_tokens}")
        result.append({"id": case_id, "prompt": case["prompt"], "max_tokens": max_tokens})
    return result


def prompt_tokens(tokenizer, prompt: str, raw_prompt: bool):
    if raw_prompt:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True
    )


def generate_once(
    model,
    tokenizer,
    case: dict,
    args,
    rank: int,
    *,
    max_tokens: int | None = None,
    show_text: bool = True,
):
    prompt = prompt_tokens(tokenizer, case["prompt"], args.raw_prompt)
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
        max_tokens=max_tokens or case["max_tokens"],
        sampler=sampler,
    ):
        generated.append(response.text)
        if rank == 0 and show_text:
            print(response.text, end="", flush=True)
    if response is None:
        raise RuntimeError("generation produced no response")
    if rank == 0 and show_text:
        print(flush=True)
    return response, "".join(generated)


def repository_state() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def main() -> int:
    args = parse_args()
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive")
    if args.warmup_tokens < 0:
        raise ValueError("--warmup-tokens cannot be negative")
    cases = benchmark_cases(args)

    # Mandatory by design: this refuses to silently fall back to TCP ring/MPI.
    group = mx.distributed.init(strict=True, backend="jaccl")
    rank, size = group.rank(), group.size()
    if size != 2:
        raise RuntimeError(f"this deployment requires exactly two JACCL ranks, got {size}")
    rdma = rdma_state()

    model_root = args.model_root.resolve()
    stage = model_root / f"rank{rank}"
    manifest = load_manifest(stage, rank, size)
    run_id = args.run_id or os.environ.get("KIMI_RUN_ID") or time.strftime(
        "%Y%m%dT%H%M%S"
    )
    host = socket.gethostname()
    print(
        f"[rank {rank}] host={host} backend=jaccl "
        f"rdma={','.join(rdma['active_ports'])} "
        f"layers=[{manifest['pipeline']['layer_start']},"
        f"{manifest['pipeline']['layer_end']}) stage={stage}",
        flush=True,
    )

    load_started = time.perf_counter()
    model, tokenizer = load_pipeline(stage, group)
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
            response, text = generate_once(model, tokenizer, case, args, rank)
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
                "backend": "jaccl",
                "transport": "thunderbolt-rdma",
                "rdma": rdma,
                "jaccl_hostfile_sha256": os.environ.get("KIMI_HOSTFILE_SHA256"),
                "mlx_metal_fast_synch": os.environ.get("MLX_METAL_FAST_SYNCH"),
                "host": host,
                "platform": platform.platform(),
                "code": repo,
                "versions": {"mlx": version("mlx"), "mlx_lm": version("mlx-lm")},
                "checkpoint": manifest["source"],
                "stage": manifest["pipeline"],
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
                "text": text,
            }
            output = output_dir / f"{record_id}-rank{rank}.json"
            output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            records.append(record)
            print(
                f"[rank {rank}] prompt={response.prompt_tokens} "
                f"({response.prompt_tps:.3f} tok/s), "
                f"generation={response.generation_tokens} "
                f"({response.generation_tps:.3f} tok/s), "
                f"peak={response.peak_memory:.3f} GB, record={output}",
                flush=True,
            )

    summary = {
        "schema": 1,
        "run_id": run_id,
        "rank": rank,
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
