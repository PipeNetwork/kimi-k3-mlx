#!/usr/bin/env python3
"""Benchmark the resident Kimi-K3 service at several request concurrencies."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def parse_concurrencies(value: str) -> list[int]:
    result = [int(item) for item in value.split(",")]
    if not result or any(item < 1 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("concurrencies must be unique positive integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:8080/v1/chat/completions"
    )
    parser.add_argument("--concurrencies", type=parse_concurrencies, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--prompt", default="Explain why the sky appears blue in two sentences.")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def request_once(endpoint: str, body: bytes, barrier: threading.Barrier, timeout: float):
    barrier.wait()
    request = urllib.request.Request(
        endpoint, body, {"content-type": "application/json"}
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    return value, time.perf_counter() - started


def run_trial(args: argparse.Namespace, concurrency: int, trial: int) -> dict:
    body = json.dumps(
        {
            "model": "Kimi-K3-2bit-UVMAX",
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
        }
    ).encode()
    barrier = threading.Barrier(concurrency)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        responses = list(
            pool.map(
                lambda _: request_once(args.endpoint, body, barrier, args.timeout),
                range(concurrency),
            )
        )
    wall_seconds = time.perf_counter() - started
    values = [item[0] for item in responses]
    latencies = [item[1] for item in responses]
    completion_tokens = sum(item["usage"]["completion_tokens"] for item in values)
    distributed = [item["distributed"] for item in values]
    return {
        "concurrency": concurrency,
        "trial": trial,
        "wall_seconds": wall_seconds,
        "completion_tokens": completion_tokens,
        "aggregate_generation_tps": completion_tokens / wall_seconds,
        "request_latency_seconds": {
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
        },
        "server_batch_sizes": sorted({item["batch_size"] for item in distributed}),
        "server_batch_generation_tps": sorted(
            {item["batch_generation_tps"] for item in distributed}
        ),
        "parity_sha256": sorted({item["parity_sha256"] for item in distributed}),
    }


def main() -> int:
    args = parse_args()
    if args.trials < 1 or args.max_tokens < 1:
        raise ValueError("trials and max-tokens must be positive")
    run_id = datetime.now(timezone.utc).strftime("server-benchmark-%Y%m%dT%H%M%SZ")
    records = []
    for concurrency in args.concurrencies:
        for trial in range(1, args.trials + 1):
            record = run_trial(args, concurrency, trial)
            records.append(record)
            print(
                f"concurrency={concurrency} trial={trial}/{args.trials} "
                f"aggregate={record['aggregate_generation_tps']:.3f} tok/s "
                f"batches={record['server_batch_sizes']} "
                f"latency={record['request_latency_seconds']['median']:.3f}s",
                flush=True,
            )
    summary = {}
    for concurrency in args.concurrencies:
        values = [
            record["aggregate_generation_tps"]
            for record in records
            if record["concurrency"] == concurrency
        ]
        summary[str(concurrency)] = {
            "median_aggregate_generation_tps": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    result = {
        "schema": 1,
        "run_id": run_id,
        "endpoint": args.endpoint,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "trials": args.trials,
        "records": records,
        "summary": summary,
    }
    output = args.output or Path("work/server-benchmarks") / f"{run_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"record={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
