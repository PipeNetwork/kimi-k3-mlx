#!/usr/bin/env python3
"""Persistent OpenAI-compatible Kimi-K3 service over two-node JACCL tensor parallelism."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import signal
import socket
import struct
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm.generate import BatchGenerator, BatchStats
from mlx_lm.sample_utils import make_sampler

try:
    from scripts.distributed_generate import prompt_tokens, rdma_state
    from scripts.tensor_generate import load_manifest
    from scripts.tensor_stage import load_tensor_stage
except ModuleNotFoundError:
    from distributed_generate import prompt_tokens, rdma_state
    from tensor_generate import load_manifest
    from tensor_stage import load_tensor_stage


MAX_CONTROL_MESSAGE = 64 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("weights/Kimi-K3-2bit-UVMAX-tensor-2"),
    )
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--decode-concurrency", type=int, default=8)
    parser.add_argument("--prompt-concurrency", type=int, default=4)
    parser.add_argument("--batch-window-ms", type=float, default=20.0)
    parser.add_argument("--max-request-tokens", type=int, default=4096)
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument(
        "--allow-test-fixture",
        action="store_true",
        help="Use the explicit tiny-test tokenizer and skip production manifest checks.",
    )
    return parser.parse_args()


def parse_endpoint(value: str) -> tuple[str, int]:
    host, separator, port = value.rpartition(":")
    if not separator or not host:
        raise ValueError(f"invalid control endpoint: {value!r}")
    return host, int(port)


class JsonChannel:
    """Length-prefixed JSON control channel; inference payloads remain on JACCL."""

    def __init__(self, connection: socket.socket):
        self.connection = connection

    def send(self, value: dict[str, Any]) -> None:
        body = json.dumps(value, separators=(",", ":")).encode()
        if len(body) > MAX_CONTROL_MESSAGE:
            raise ValueError("control message is too large")
        self.connection.sendall(struct.pack("!I", len(body)) + body)

    def receive(self) -> dict[str, Any] | None:
        header = self._read_exact(4)
        if header is None:
            return None
        size = struct.unpack("!I", header)[0]
        if size > MAX_CONTROL_MESSAGE:
            raise ValueError(f"refusing {size}-byte control message")
        body = self._read_exact(size)
        if body is None:
            raise ConnectionError("control connection closed mid-message")
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError("control message must be a JSON object")
        return value

    def _read_exact(self, size: int) -> bytes | None:
        chunks = []
        remaining = size
        while remaining:
            chunk = self.connection.recv(remaining)
            if not chunk:
                return None if remaining == size else b"".join(chunks)
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.connection.close()


def connect_control(rank: int) -> JsonChannel:
    endpoint = os.environ.get("KIMI_JACCL_CONTROL_COORDINATOR")
    if not endpoint:
        raise RuntimeError("KIMI_JACCL_CONTROL_COORDINATOR is required")
    host, port = parse_endpoint(endpoint)
    if rank == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, port))
            listener.listen(1)
            listener.settimeout(180)
            connection, peer = listener.accept()
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[rank 0] control worker connected from {peer[0]}:{peer[1]}", flush=True)
        return JsonChannel(connection)

    deadline = time.monotonic() + 180
    while True:
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            connection.connect((host, port))
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return JsonChannel(connection)
        except OSError:
            connection.close()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"could not connect to control coordinator {endpoint}")
            time.sleep(0.25)


def output_digest(tokens: list[list[int]], finishes: list[str]) -> str:
    encoded = json.dumps(
        {"tokens": tokens, "finishes": finishes},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class FixtureTokenizer:
    """Small deterministic tokenizer used only by explicit transport tests."""

    eos_token_ids = [127]

    @staticmethod
    def encode(value: str) -> list[int]:
        return [byte % 126 + 1 for byte in value.encode()]

    def apply_chat_template(
        self, messages: list[dict[str, Any]], *, tokenize: bool, add_generation_prompt: bool
    ) -> list[int]:
        if not tokenize:
            raise ValueError("fixture tokenizer only supports tokenize=True")
        rendered = "\n".join(str(message.get("content", "")) for message in messages)
        if add_generation_prompt:
            rendered += "\nassistant:"
        return self.encode(rendered)

    @staticmethod
    def decode(tokens: list[int]) -> str:
        return " ".join(str(token) for token in tokens)


def run_batch(model, tokenizer, command: dict[str, Any]) -> dict[str, Any]:
    prompts = command["prompts"]
    max_tokens = command["max_tokens"]
    sampling = command["sampling"]
    mx.random.seed(command["seed"])
    sampler = make_sampler(
        temp=sampling["temperature"],
        top_p=sampling["top_p"],
        min_p=sampling["min_p"],
        top_k=sampling["top_k"],
    )
    generator = BatchGenerator(
        model,
        max_tokens=max(max_tokens),
        stop_tokens=[[token] for token in tokenizer.eos_token_ids],
        sampler=sampler,
        completion_batch_size=command["decode_concurrency"],
        prefill_batch_size=command["prompt_concurrency"],
    )
    uids = generator.insert(prompts, max_tokens=max_tokens)
    results = {uid: [] for uid in uids}
    finish_reasons = {uid: "length" for uid in uids}
    stats = BatchStats()
    try:
        with generator.stats(stats):
            while responses := generator.next_generated():
                for response in responses:
                    if response.finish_reason is not None:
                        finish_reasons[response.uid] = response.finish_reason
                    if response.finish_reason != "stop":
                        results[response.uid].append(int(response.token))
    finally:
        generator.close()
    tokens = [results[uid] for uid in uids]
    finishes = [finish_reasons[uid] for uid in uids]
    return {
        "digest": output_digest(tokens, finishes),
        "tokens": tokens,
        "finishes": finishes,
        "stats": asdict(stats),
    }


@dataclass
class Request:
    request_id: str
    prompt: list[int]
    max_tokens: int
    sampling: dict[str, float | int]
    seed: int | None
    created: float = field(default_factory=time.time)
    done: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: BaseException | None = None

    @property
    def batch_key(self) -> tuple:
        return (*self.sampling.values(), self.seed)


class InferenceEngine:
    def __init__(self, model, tokenizer, channel: JsonChannel, args: argparse.Namespace):
        self.model = model
        self.tokenizer = tokenizer
        self.channel = channel
        self.args = args
        self.requests: queue.Queue[Request | None] = queue.Queue()
        self.deferred: list[Request] = []
        self.thread = threading.Thread(target=self._run, name="inference", daemon=True)
        self.fatal: str | None = None
        self.started = time.time()
        self.batch_count = 0
        self.request_count = 0
        self.generated_tokens = 0
        self._seed = 0
        self._metrics_lock = threading.Lock()
        self.thread.start()

    def submit(self, request: Request) -> dict[str, Any]:
        if self.fatal:
            raise RuntimeError(self.fatal)
        self.requests.put(request)
        request.done.wait()
        if request.error:
            raise RuntimeError(str(request.error)) from request.error
        assert request.result is not None
        return request.result

    def snapshot(self) -> dict[str, Any]:
        with self._metrics_lock:
            return {
                "ready": self.fatal is None and self.thread.is_alive(),
                "fatal": self.fatal,
                "uptime_seconds": time.time() - self.started,
                "batches": self.batch_count,
                "requests": self.request_count,
                "generated_tokens": self.generated_tokens,
                "queued": self.requests.qsize() + len(self.deferred),
                "decode_concurrency": self.args.decode_concurrency,
                "prompt_concurrency": self.args.prompt_concurrency,
                "batch_window_ms": self.args.batch_window_ms,
            }

    def stop(self) -> None:
        self.requests.put(None)
        self.thread.join(timeout=30)

    def _next(self, timeout: float | None = None) -> Request | None:
        if self.deferred:
            return self.deferred.pop(0)
        return self.requests.get(timeout=timeout)

    def _collect_batch(self, first: Request) -> list[Request]:
        batch = [first]
        deadline = time.monotonic() + self.args.batch_window_ms / 1000
        while len(batch) < self.args.decode_concurrency:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                candidate = self.requests.get(timeout=remaining)
            except queue.Empty:
                break
            if candidate is None:
                self.requests.put(None)
                break
            if candidate.batch_key == first.batch_key:
                batch.append(candidate)
            else:
                self.deferred.append(candidate)
        return batch

    def _run(self) -> None:
        current: list[Request] = []
        try:
            while True:
                first = self._next()
                if first is None:
                    self.channel.send({"op": "shutdown"})
                    return
                current = self._collect_batch(first)
                self._execute(current)
                current = []
        except BaseException as error:
            self.fatal = f"distributed inference failed: {error}"
            for request in current + self.deferred:
                request.error = error
                request.done.set()
            self.deferred.clear()
            while True:
                try:
                    request = self.requests.get_nowait()
                except queue.Empty:
                    break
                if request is not None:
                    request.error = error
                    request.done.set()

    def _execute(self, requests: list[Request]) -> None:
        seed = requests[0].seed
        if seed is None:
            seed = self._seed
            self._seed = (self._seed + 1) % (2**31)
        command = {
            "op": "generate",
            "batch_id": self.batch_count,
            "prompts": [request.prompt for request in requests],
            "max_tokens": [request.max_tokens for request in requests],
            "sampling": requests[0].sampling,
            "seed": seed,
            "decode_concurrency": self.args.decode_concurrency,
            "prompt_concurrency": min(self.args.prompt_concurrency, len(requests)),
        }
        started = time.perf_counter()
        self.channel.send(command)
        local = run_batch(self.model, self.tokenizer, command)
        remote = self.channel.receive()
        if remote is None or remote.get("op") != "result":
            raise RuntimeError(f"rank 1 returned invalid batch result: {remote!r}")
        if remote.get("digest") != local["digest"]:
            raise RuntimeError(
                f"rank output mismatch: rank0={local['digest']} rank1={remote.get('digest')}"
            )
        elapsed = time.perf_counter() - started
        stats = local["stats"]
        for index, request in enumerate(requests):
            token_ids = local["tokens"][index]
            request.result = {
                "text": self.tokenizer.decode(token_ids),
                "token_ids": token_ids,
                "finish_reason": local["finishes"][index],
                "batch_size": len(requests),
                "batch_seconds": elapsed,
                "batch_generation_tps": stats["generation_tps"],
                "batch_prompt_tps": stats["prompt_tps"],
                "prompt_tokens": len(request.prompt),
                "completion_tokens": len(token_ids),
                "parity_sha256": local["digest"],
            }
            request.done.set()
        with self._metrics_lock:
            self.batch_count += 1
            self.request_count += len(requests)
            self.generated_tokens += sum(len(tokens) for tokens in local["tokens"])


def worker_loop(model, tokenizer, channel: JsonChannel) -> int:
    while command := channel.receive():
        operation = command.get("op")
        if operation == "shutdown":
            return 0
        if operation != "generate":
            raise ValueError(f"unknown control operation: {operation!r}")
        result = run_batch(model, tokenizer, command)
        channel.send(
            {
                "op": "result",
                "batch_id": command["batch_id"],
                "digest": result["digest"],
                "stats": result["stats"],
            }
        )
    return 0


def json_response(handler: BaseHTTPRequestHandler, status: int, value: dict) -> None:
    body = json.dumps(value, separators=(",", ":")).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(engine: InferenceEngine, tokenizer, args: argparse.Namespace):
    class Handler(BaseHTTPRequestHandler):
        server_version = "KimiK3MLXDistributed/1"

        def log_message(self, format: str, *values: Any) -> None:
            print(f"[http] {self.address_string()} {format % values}", flush=True)

        def do_GET(self) -> None:
            if self.path == "/health":
                state = engine.snapshot()
                json_response(self, 200 if state["ready"] else 503, state)
            elif self.path == "/v1/models":
                json_response(
                    self,
                    200,
                    {"object": "list", "data": [{"id": "Kimi-K3-2bit-UVMAX", "object": "model"}]},
                )
            else:
                json_response(self, 404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 16 * 1024 * 1024:
                    raise ValueError("invalid request size")
                body = json.loads(self.rfile.read(length))
                if body.get("stream"):
                    raise ValueError("streaming is not yet supported; use stream=false")
                if self.path == "/v1/chat/completions":
                    messages = body.get("messages")
                    if not isinstance(messages, list) or not messages:
                        raise ValueError("messages must be a non-empty list")
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                elif self.path == "/v1/completions":
                    value = body.get("prompt")
                    if not isinstance(value, str):
                        raise ValueError("prompt must be a string")
                    prompt = tokenizer.encode(value)
                else:
                    json_response(self, 404, {"error": {"message": "not found"}})
                    return
                if hasattr(prompt, "tolist"):
                    prompt = prompt.tolist()
                if prompt and isinstance(prompt[0], list):
                    prompt = prompt[0]
                if not prompt:
                    raise ValueError("prompt tokenization produced no tokens")
                max_tokens = int(body.get("max_tokens", 256))
                if not 1 <= max_tokens <= args.max_request_tokens:
                    raise ValueError(
                        f"max_tokens must be in [1, {args.max_request_tokens}]"
                    )
                sampling = {
                    "temperature": float(body.get("temperature", 0.0)),
                    "top_p": float(body.get("top_p", 0.0)),
                    "min_p": float(body.get("min_p", 0.0)),
                    "top_k": int(body.get("top_k", 0)),
                }
                seed = body.get("seed")
                if seed is not None:
                    seed = int(seed)
                request_id = f"chatcmpl-{uuid.uuid4().hex}"
                result = engine.submit(
                    Request(request_id, [int(token) for token in prompt], max_tokens, sampling, seed)
                )
                response = {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": body.get("model", "Kimi-K3-2bit-UVMAX"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": result["text"]},
                            "finish_reason": result["finish_reason"],
                        }
                    ],
                    "usage": {
                        "prompt_tokens": result["prompt_tokens"],
                        "completion_tokens": result["completion_tokens"],
                        "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
                    },
                    "distributed": {
                        key: result[key]
                        for key in (
                            "batch_size",
                            "batch_seconds",
                            "batch_generation_tps",
                            "batch_prompt_tps",
                            "parity_sha256",
                        )
                    },
                }
                json_response(self, 200, response)
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                json_response(self, 400, {"error": {"message": str(error)}})
            except BaseException as error:
                json_response(self, 500, {"error": {"message": str(error)}})

    return Handler


def main() -> int:
    args = parse_args()
    if args.decode_concurrency < 1 or args.prompt_concurrency < 1:
        raise ValueError("batch concurrencies must be positive")
    if args.batch_window_ms < 0:
        raise ValueError("--batch-window-ms cannot be negative")

    group = mx.distributed.init(strict=True, backend="jaccl")
    rank, size = group.rank(), group.size()
    if size != 2:
        raise RuntimeError(f"this service requires exactly two JACCL ranks, got {size}")
    mx.eval(
        mx.distributed.all_sum(
            mx.ones((10,), dtype=mx.float32), group=group, stream=mx.cpu
        )
    )
    rdma = rdma_state()
    stage = args.model_root.resolve() / f"rank{rank}"
    if args.allow_test_fixture:
        if not (stage / "config.json").is_file():
            raise FileNotFoundError(stage / "config.json")
    else:
        load_manifest(stage, rank, size)
    print(
        f"[rank {rank}] JACCL ready, RDMA={','.join(rdma['active_ports'])}, stage={stage}",
        flush=True,
    )
    started = time.perf_counter()
    if args.allow_test_fixture:
        model, _ = load_tensor_stage(stage, group, tokenizer=False)
        tokenizer = FixtureTokenizer()
    else:
        model, tokenizer = load_tensor_stage(stage, group)
    print(f"[rank {rank}] model loaded in {time.perf_counter() - started:.3f}s", flush=True)
    channel = connect_control(rank)
    if rank == 1:
        try:
            return worker_loop(model, tokenizer, channel)
        finally:
            channel.close()

    engine = InferenceEngine(model, tokenizer, channel, args)
    server = ThreadingHTTPServer((args.bind, args.port), make_handler(engine, tokenizer, args))
    server.daemon_threads = True
    run_id = os.environ.get("KIMI_RUN_ID", "server")
    ready_dir = Path("work/server") / run_id
    ready_dir.mkdir(parents=True, exist_ok=True)
    ready_path = ready_dir / "ready.json"
    ready_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "pid": os.getpid(),
                "bind": args.bind,
                "port": args.port,
                "backend": "jaccl",
                "transport": "thunderbolt-rdma",
                "rdma": rdma,
                **engine.snapshot(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        f"[rank 0] ready at http://{args.bind}:{args.port}; proof={ready_path}",
        flush=True,
    )

    def request_shutdown(_signum=None, _frame=None):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        engine.stop()
        channel.close()
        ready_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
