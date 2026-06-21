#!/usr/bin/env python3
"""Replay a JSONL workload against vLLM OpenAI-compatible endpoints."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import random
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml


LOGGER = logging.getLogger("online_workload_client")
VALID_MODES = (
    "colocated_4replica",
    "aggregated_tp4",
    "real_pd_1p3d",
    "real_pd_2p2d",
    "real_pd_3p1d",
)
FIELDS = [
    "run_id", "request_id", "workload_name", "workload_mode", "request_type",
    "serving_mode", "result_type", "route_target", "prompt_len_target",
    "output_len_target", "prompt_len_actual", "output_len_actual", "arrival_rate",
    "concurrency", "arrival_time", "send_time", "ttft_ms", "tpot_ms",
    "total_latency_ms", "success", "error_msg", "prefix_caching_enabled",
    "ttft_method", "data_source", "seed",
]


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/real4gpu.yaml"))
    parser.add_argument(
        "--workload", type=Path,
        default=Path("data/processed/workload_mixed_50p50d_synthetic_unique.jsonl"),
    )
    parser.add_argument("--mode", choices=VALID_MODES, default="colocated_4replica")
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/metrics/real4gpu/online_mixed_50p50d_colocated_4replica.csv"),
    )
    parser.add_argument(
        "--routing", choices=("round_robin", "random", "least_outstanding"),
        default="round_robin",
    )
    parser.add_argument("--arrival-rate", type=float, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--stream", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--request-timeout-s", type=float, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return data


def load_workload(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            missing = [key for key in ("request_id", "prompt", "prompt_len", "output_len", "arrival_time") if key not in row]
            if missing:
                raise ValueError(f"{path}:{line_no}: missing fields {missing}")
            if int(row["prompt_len"]) <= 0 or int(row["output_len"]) <= 0:
                raise ValueError(f"{path}:{line_no}: token lengths must be positive")
            rows.append(row)
            if len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no requests loaded from {path}")
    return sorted(rows, key=lambda row: float(row["arrival_time"]))


def endpoints_for_mode(config: dict[str, Any], mode: str) -> list[str]:
    ports = config["ports"]
    if mode == "colocated_4replica":
        selected = ports["colocated"]
    elif mode == "aggregated_tp4":
        selected = [ports["aggregated_tp4"]]
    else:
        selected = [ports.get("pd_proxy", 8500)]
    return [f"http://127.0.0.1:{int(port)}/v1/completions" for port in selected]


def result_type_for_mode(mode: str) -> str:
    if mode == "colocated_4replica":
        return "real_colocated"
    if mode == "aggregated_tp4":
        return "real_aggregated_tp"
    return "real_disaggregated_pd"


def make_arrivals(
    requests: list[dict[str, Any]], arrival_rate: float | None, seed: int
) -> tuple[list[float], float]:
    if arrival_rate is not None:
        if not math.isfinite(arrival_rate) or arrival_rate <= 0:
            raise ValueError("arrival-rate must be positive")
        rng = random.Random(seed)
        values = [0.0]
        for _ in range(1, len(requests)):
            values.append(values[-1] + rng.expovariate(arrival_rate))
        return values, arrival_rate
    original = [float(row["arrival_time"]) for row in requests]
    base = min(original)
    values = [value - base for value in original]
    span = max(values) - min(values)
    inferred = (len(values) - 1) / span if len(values) > 1 and span > 0 else math.nan
    return values, inferred


def load_token_counter(name: str) -> Callable[[str], int] | None:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name)
        return lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    except Exception as exc:
        LOGGER.warning(
            "Tokenizer %s could not be loaded; API usage fields are required for actual "
            "token counts: %s", name, exc,
        )
        return None


@dataclass
class HTTPResult:
    ttft_ms: float | None
    total_latency_ms: float
    prompt_tokens: int | None
    output_tokens: int | None
    generated_text: str


def post_completion(
    endpoint: str,
    model: str,
    prompt: str,
    output_len: int,
    stream: bool,
    timeout_s: float,
    token_counter: Callable[[str], int] | None,
) -> HTTPResult:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": int(output_len),
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    pieces: list[str] = []
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            if stream:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    content = line[5:].strip()
                    if not content or content == "[DONE]":
                        continue
                    chunk = json.loads(content)
                    usage = chunk.get("usage") or {}
                    if usage.get("prompt_tokens") is not None:
                        prompt_tokens = int(usage["prompt_tokens"])
                    if usage.get("completion_tokens") is not None:
                        output_tokens = int(usage["completion_tokens"])
                    for choice in chunk.get("choices") or []:
                        text = choice.get("text")
                        if text is None:
                            text = (choice.get("delta") or {}).get("content")
                        if text:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            pieces.append(text)
            else:
                body = json.loads(response.read().decode("utf-8"))
                usage = body.get("usage") or {}
                prompt_tokens = usage.get("prompt_tokens")
                output_tokens = usage.get("completion_tokens")
                for choice in body.get("choices") or []:
                    if choice.get("text"):
                        pieces.append(choice["text"])
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:1000]}") from exc
    ended = time.perf_counter()
    generated = "".join(pieces)
    if output_tokens is None and token_counter is not None:
        output_tokens = token_counter(generated)
    return HTTPResult(
        ttft_ms=(first_token_at - started) * 1000 if first_token_at else None,
        total_latency_ms=(ended - started) * 1000,
        prompt_tokens=int(prompt_tokens) if prompt_tokens is not None else None,
        output_tokens=int(output_tokens) if output_tokens is not None else None,
        generated_text=generated,
    )


class Router:
    def __init__(self, endpoints: list[str], policy: str, seed: int):
        self.endpoints = endpoints
        self.policy = policy
        self.rng = random.Random(seed)
        self.cursor = 0
        self.outstanding = {endpoint: 0 for endpoint in endpoints}
        self.lock = asyncio.Lock()

    async def acquire(self) -> str:
        async with self.lock:
            if self.policy == "random":
                endpoint = self.rng.choice(self.endpoints)
            elif self.policy == "least_outstanding":
                minimum = min(self.outstanding.values())
                candidates = [x for x in self.endpoints if self.outstanding[x] == minimum]
                endpoint = candidates[self.cursor % len(candidates)]
                self.cursor += 1
            else:
                endpoint = self.endpoints[self.cursor % len(self.endpoints)]
                self.cursor += 1
            self.outstanding[endpoint] += 1
            return endpoint

    async def release(self, endpoint: str) -> None:
        async with self.lock:
            self.outstanding[endpoint] = max(0, self.outstanding[endpoint] - 1)


async def run_benchmark(
    requests: list[dict[str, Any]], arrivals: list[float], config: dict[str, Any],
    args: argparse.Namespace, seed: int, run_id: str, effective_rate: float,
    token_counter: Callable[[str], int] | None,
) -> list[dict[str, Any]]:
    endpoints = endpoints_for_mode(config, args.mode)
    router = Router(endpoints, args.routing, seed)
    semaphore = asyncio.Semaphore(args.concurrency)
    run_started = time.perf_counter()
    result_type = result_type_for_mode(args.mode)
    timeout_s = args.request_timeout_s or float(config.get("request_timeout_s", 300))

    async def one(index: int) -> dict[str, Any]:
        request = requests[index]
        target_time = run_started + arrivals[index]
        await asyncio.sleep(max(0.0, target_time - time.perf_counter()))
        endpoint = ""
        send_time = math.nan
        async with semaphore:
            endpoint = await router.acquire()
            send_time = time.perf_counter() - run_started
            try:
                result = await asyncio.to_thread(
                    post_completion, endpoint, str(config["model_name"]),
                    str(request["prompt"]), int(request["output_len"]), args.stream,
                    timeout_s, token_counter,
                )
                if args.stream and result.ttft_ms is None:
                    raise RuntimeError("stream completed without a non-empty token/chunk")
                output_actual = result.output_tokens
                tpot = None
                if output_actual is not None and output_actual > 1 and result.ttft_ms is not None:
                    tpot = (result.total_latency_ms - result.ttft_ms) / (output_actual - 1)
                return {
                    "run_id": run_id, "request_id": request["request_id"],
                    "workload_name": request.get("workload_name", args.workload.stem.removeprefix("workload_")),
                    "workload_mode": request.get("workload_mode", "unknown"),
                    "request_type": request.get("request_type", "unknown"),
                    "serving_mode": args.mode, "result_type": result_type,
                    "route_target": endpoint, "prompt_len_target": int(request["prompt_len"]),
                    "output_len_target": int(request["output_len"]),
                    "prompt_len_actual": result.prompt_tokens,
                    "output_len_actual": output_actual, "arrival_rate": effective_rate,
                    "concurrency": args.concurrency, "arrival_time": arrivals[index],
                    "send_time": send_time, "ttft_ms": result.ttft_ms,
                    "tpot_ms": tpot, "total_latency_ms": result.total_latency_ms,
                    "success": True, "error_msg": "",
                    "prefix_caching_enabled": bool(config.get("enable_prefix_caching", False)),
                    "ttft_method": "streaming_measured" if args.stream else "unavailable",
                    "data_source": "measured", "seed": seed,
                }
            except Exception as exc:
                LOGGER.exception("Request %s failed", request["request_id"])
                return {
                    "run_id": run_id, "request_id": request["request_id"],
                    "workload_name": request.get("workload_name", "unknown"),
                    "workload_mode": request.get("workload_mode", "unknown"),
                    "request_type": request.get("request_type", "unknown"),
                    "serving_mode": args.mode, "result_type": result_type,
                    "route_target": endpoint, "prompt_len_target": int(request["prompt_len"]),
                    "output_len_target": int(request["output_len"]),
                    "prompt_len_actual": None, "output_len_actual": None,
                    "arrival_rate": effective_rate, "concurrency": args.concurrency,
                    "arrival_time": arrivals[index], "send_time": send_time,
                    "ttft_ms": None, "tpot_ms": None, "total_latency_ms": None,
                    "success": False, "error_msg": f"{type(exc).__name__}: {exc}"[:2000],
                    "prefix_caching_enabled": bool(config.get("enable_prefix_caching", False)),
                    "ttft_method": "streaming_measured" if args.stream else "unavailable",
                    "data_source": "measured_attempt", "seed": seed,
                }
            finally:
                if endpoint:
                    await router.release(endpoint)

    tasks = [asyncio.create_task(one(index)) for index in range(len(requests))]
    return await asyncio.gather(*tasks)


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(path, mode="w", encoding="utf-8")],
    )


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    max_requests = int(args.max_requests or config.get("max_requests", 200))
    warmup = int(args.warmup if args.warmup is not None else config.get("warmup_requests", 10))
    if args.concurrency <= 0 or max_requests <= 0 or warmup < 0:
        raise ValueError("concurrency/max-requests must be positive and warmup non-negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.log_file or Path("outputs/logs/real4gpu") / f"client_{args.output.stem}.log"
    configure_logging(log_path)
    requests = load_workload(args.workload, max_requests)
    arrivals, effective_rate = make_arrivals(requests, args.arrival_rate, seed)
    run_id = args.run_id or f"{args.mode}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    token_counter = load_token_counter(str(config.get("tokenizer_name", config["model_name"])))
    endpoints = endpoints_for_mode(config, args.mode)
    LOGGER.info("Run %s: %d requests, endpoints=%s", run_id, len(requests), endpoints)

    for index in range(min(warmup, len(requests))):
        endpoint = endpoints[index % len(endpoints)]
        try:
            post_completion(
                endpoint, str(config["model_name"]), str(requests[index]["prompt"]),
                int(requests[index]["output_len"]), args.stream,
                args.request_timeout_s or float(config.get("request_timeout_s", 300)),
                token_counter,
            )
        except Exception as exc:
            LOGGER.error("Warmup %d/%d failed: %s", index + 1, warmup, exc)
            raise RuntimeError("warmup failed; measured run was not started") from exc
    LOGGER.info("Completed %d warmup requests", min(warmup, len(requests)))

    rows = asyncio.run(
        run_benchmark(requests, arrivals, config, args, seed, run_id, effective_rate, token_counter)
    )
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    successes = sum(bool(row["success"]) for row in rows)
    LOGGER.info("Wrote %s: %d success, %d failed", args.output, successes, len(rows) - successes)
    if successes == 0:
        raise RuntimeError("all measured requests failed; failure rows were preserved")


if __name__ == "__main__":
    main()
