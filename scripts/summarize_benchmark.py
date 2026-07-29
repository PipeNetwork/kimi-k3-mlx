#!/usr/bin/env python3
"""Validate paired rank records and create an auditable system-level summary."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="Suite run id used with distributed_generate.py")
    parser.add_argument("--input", type=Path, default=Path("work/benchmarks"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def validate_pair(rank0: dict, rank1: dict) -> dict:
    """Validate one synchronized generation and return its system metrics."""
    for rank, record in enumerate((rank0, rank1)):
        if record.get("rank") != rank:
            raise ValueError(f"expected rank {rank}, got {record.get('rank')}")
        if record.get("world_size") != 2 or record.get("backend") != "jaccl":
            raise ValueError(f"rank {rank} is not a two-rank JACCL result")
        if record.get("transport") != "thunderbolt-rdma":
            raise ValueError(f"rank {rank} does not declare Thunderbolt RDMA")
        if not record.get("rdma", {}).get("active_ports"):
            raise ValueError(f"rank {rank} has no active RDMA-port evidence")
        if record.get("code", {}).get("dirty") is not False:
            raise ValueError(f"rank {rank} was not run from a clean worktree")
        if record.get("mlx_metal_fast_synch") != "1":
            raise ValueError(f"rank {rank} did not enable MLX_METAL_FAST_SYNCH")
        hardware = record.get("hardware", {})
        if not hardware.get("device_name") or not hardware.get("memory_size"):
            raise ValueError(f"rank {rank} has incomplete hardware evidence")

    equal_fields = (
        "run_id",
        "suite_run_id",
        "case_id",
        "repetition",
        "world_size",
        "backend",
        "transport",
        "strategy",
        "code",
        "versions",
        "checkpoint",
        "jaccl_hostfile_sha256",
        "jaccl_ring",
        "prompt",
        "seed",
        "sampling",
        "max_tokens",
        "prompt_tokens",
        "generation_tokens",
        "text",
    )
    for field in equal_fields:
        if rank0.get(field) != rank1.get(field):
            raise ValueError(f"rank mismatch in {field!r}")
    if not rank0.get("jaccl_hostfile_sha256"):
        raise ValueError("missing JACCL hostfile hash")
    if rank0.get("host") == rank1.get("host"):
        raise ValueError("rank records came from the same hostname")

    # Both ranks synchronize on every token. The slower observed rank is the
    # conservative end-to-end system throughput for a paired record.
    return {
        "run_id": rank0["run_id"],
        "case_id": rank0["case_id"],
        "repetition": rank0["repetition"],
        "generation_tokens": rank0["generation_tokens"],
        "generation_tps": min(rank0["generation_tps"], rank1["generation_tps"]),
        "prompt_tps": min(rank0["prompt_tps"], rank1["prompt_tps"]),
        "peak_memory_gb": max(rank0["peak_memory_gb"], rank1["peak_memory_gb"]),
        "rank_generation_tps": [rank0["generation_tps"], rank1["generation_tps"]],
        "rank_prompt_tps": [rank0["prompt_tps"], rank1["prompt_tps"]],
        "rank_hardware": [rank0["hardware"], rank1["hardware"]],
    }


def summarize(run_id: str, input_dir: Path) -> dict:
    summaries = [
        read_json(input_dir / f"{run_id}-summary-rank{rank}.json")
        for rank in (0, 1)
    ]
    if summaries[0].get("records") != summaries[1].get("records"):
        raise ValueError("rank summaries list different records")
    record_ids = summaries[0].get("records", [])
    if not record_ids:
        raise ValueError("rank summaries contain no records")
    pairs = [
        validate_pair(
            read_json(input_dir / f"{record_id}-rank0.json"),
            read_json(input_dir / f"{record_id}-rank1.json"),
        )
        for record_id in record_ids
    ]
    decode = [pair["generation_tps"] for pair in pairs]
    prefill = [pair["prompt_tps"] for pair in pairs]
    return {
        "schema": 1,
        "run_id": run_id,
        "validated_rank_agreement": True,
        "records": pairs,
        "generation_tps": {
            "median": statistics.median(decode),
            "min": min(decode),
            "max": max(decode),
        },
        "prompt_tps": {
            "median": statistics.median(prefill),
            "min": min(prefill),
            "max": max(prefill),
        },
        "peak_memory_gb": max(pair["peak_memory_gb"] for pair in pairs),
    }


def main() -> int:
    args = parse_args()
    summary = summarize(args.run_id, args.input)
    output = args.output or args.input / f"{args.run_id}-validated.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    generation = summary["generation_tps"]
    print(
        f"validated {len(summary['records'])} paired records: "
        f"median generation={generation['median']:.3f} tok/s, "
        f"range={generation['min']:.3f}-{generation['max']:.3f}; {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
