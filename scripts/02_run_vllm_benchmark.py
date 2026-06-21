#!/usr/bin/env python3
"""Run measured offline LLM benchmarks on a synthetic workload JSONL file."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml


LOGGER = logging.getLogger("run_vllm_benchmark")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RUN_LOG_PATH: Path | None = None

REQUEST_FIELDS = [
    "request_id",
    "workload_name",
    "workload_mode",
    "request_type",
    "cache_group_id",
    "intended_prefix_reuse",
    "unique_marker",
    "prompt_len_target",
    "output_len_target",
    "prompt_len_actual",
    "output_len_actual",
    "ttft_ms",
    "tpot_ms",
    "total_latency_ms",
    "tokens_per_second",
    "num_cached_tokens",
    "ttft_method",
    "backend",
    "model_name",
    "success",
    "error_msg",
    "data_source",
]

SUMMARY_FIELDS = [
    "timestamp_utc",
    "workload_name",
    "workload_mode",
    "workload_file",
    "backend",
    "model_name",
    "attempted_requests",
    "successful_requests",
    "failed_requests",
    "wall_time_s",
    "requests_per_second",
    "output_tokens",
    "tokens_per_second",
    "ttft_method",
    "prefix_caching_enabled",
    "intended_prefix_reuse_ratio",
    "cache_hit_requests",
    "cache_hit_request_ratio",
    "cached_prompt_tokens",
    "cached_prompt_token_ratio",
    "avg_num_cached_tokens",
    "valid_metrics",
    "seed",
    "warmup_requests",
    "data_source",
]

GPU_TRACE_FIELDS = ["timestamp", "gpu_util", "memory_used_mb", "memory_total_mb"]


@dataclass(frozen=True)
class WorkloadRequest:
    request_id: str
    workload_name: str
    workload_mode: str
    request_type: str
    cache_group_id: str | None
    intended_prefix_reuse: bool
    unique_marker: str
    prompt_len: int
    output_len: int
    prompt: str


@dataclass
class BackendResult:
    rows: list[dict[str, Any]]
    wall_time_s: float
    backend: str
    ttft_method: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model.yaml"),
        help="Model YAML file (default: configs/model.yaml).",
    )
    parser.add_argument(
        "--workload",
        type=Path,
        default=Path(
            "data/processed/workload_mixed_50p50d_synthetic_unique.jsonl"
        ),
        help="Input workload JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/metrics/benchmark_mixed_50p50d_synthetic_unique.csv"
        ),
        help="Per-request output CSV.",
    )
    parser.add_argument(
        "--backend",
        choices=("vllm", "hf"),
        default="vllm",
        help="Inference backend (default: vllm).",
    )
    parser.add_argument(
        "--max-requests", type=int, default=200, help="Maximum measured requests."
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Number of unrecorded warmup requests."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics" / "benchmark_summary.csv",
        help="Aggregate benchmark summary CSV.",
    )
    parser.add_argument(
        "--gpu-sample-interval",
        type=float,
        default=0.2,
        help="pynvml sample interval in seconds (default: 0.2).",
    )
    parser.add_argument(
        "--disable-gpu-monitor",
        action="store_true",
        help="Disable pynvml utilization and memory sampling.",
    )
    parser.add_argument(
        "--no-hf-fallback",
        action="store_true",
        help="Do not try HuggingFace when vLLM cannot be initialized.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Run log path (default: outputs/logs/benchmark_<workload>.log).",
    )
    return parser.parse_args()


def ensure_run_file_handler() -> None:
    """Attach our logger after third-party logging reconfiguration."""

    if _RUN_LOG_PATH is None:
        return
    for handler in list(LOGGER.handlers):
        if getattr(handler, "_pca_run_file_handler", False):
            LOGGER.removeHandler(handler)
            handler.close()
    handler = logging.FileHandler(_RUN_LOG_PATH, mode="a", encoding="utf-8")
    handler._pca_run_file_handler = True  # type: ignore[attr-defined]
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = True


def configure_run_logging(log_path: Path, backend: str) -> None:
    """Capture both benchmark and vLLM/EngineCore logs in one file."""

    global _RUN_LOG_PATH
    _RUN_LOG_PATH = log_path.resolve()
    _RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RUN_LOG_PATH.write_text("", encoding="utf-8")
    ensure_run_file_handler()

    if backend != "vllm":
        return
    vllm_logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "pca": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "pca",
                "level": "INFO",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "pca",
                "level": "INFO",
                "filename": str(_RUN_LOG_PATH),
                "mode": "a",
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "vllm": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            }
        },
    }
    config_path = Path("/tmp") / f"pca_vllm_logging_{os.getpid()}.json"
    config_path.write_text(
        json.dumps(vllm_logging_config, ensure_ascii=False), encoding="utf-8"
    )
    os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
    os.environ["VLLM_LOGGING_CONFIG_PATH"] = str(config_path)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def validate_model_config(config: dict[str, Any]) -> None:
    required = {
        "model_name",
        "tokenizer_name",
        "dtype",
        "max_model_len",
        "gpu_memory_utilization",
        "tensor_parallel_size",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Model config is missing fields: {', '.join(missing)}")
    if not isinstance(config["max_model_len"], int) or config["max_model_len"] <= 0:
        raise ValueError("max_model_len must be a positive integer")
    if not 0 < float(config["gpu_memory_utilization"]) <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if int(config["tensor_parallel_size"]) <= 0:
        raise ValueError("tensor_parallel_size must be positive")
    if not isinstance(config.get("enable_prefix_caching", False), bool):
        raise ValueError("enable_prefix_caching must be true or false")


def load_workload(path: Path, max_requests: int) -> list[WorkloadRequest]:
    if not path.is_file():
        raise FileNotFoundError(f"Workload does not exist: {path}")
    requests: list[WorkloadRequest] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if len(requests) >= max_requests:
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            required = {
                "request_id",
                "workload_name",
                "request_type",
                "prompt_len",
                "output_len",
                "prompt",
            }
            missing = required - record.keys()
            if missing:
                raise ValueError(
                    f"Missing fields at {path}:{line_number}: {sorted(missing)}"
                )
            request_id = str(record["request_id"])
            if request_id in seen_ids:
                raise ValueError(f"Duplicate request_id at line {line_number}: {request_id}")
            seen_ids.add(request_id)
            prompt_len = record["prompt_len"]
            output_len = record["output_len"]
            if (
                not isinstance(prompt_len, int)
                or isinstance(prompt_len, bool)
                or prompt_len <= 0
                or not isinstance(output_len, int)
                or isinstance(output_len, bool)
                or output_len <= 0
            ):
                raise ValueError(f"Invalid token lengths at {path}:{line_number}")
            if not isinstance(record["prompt"], str) or not record["prompt"]:
                raise ValueError(f"Prompt must be a non-empty string at {path}:{line_number}")
            requests.append(
                WorkloadRequest(
                    request_id=request_id,
                    workload_name=str(record["workload_name"]),
                    workload_mode=str(record.get("workload_mode", "legacy")),
                    request_type=str(record["request_type"]),
                    cache_group_id=(
                        str(record["cache_group_id"])
                        if record.get("cache_group_id") is not None
                        else None
                    ),
                    intended_prefix_reuse=bool(
                        record.get("intended_prefix_reuse", False)
                    ),
                    unique_marker=str(record.get("unique_marker", "")),
                    prompt_len=prompt_len,
                    output_len=output_len,
                    prompt=record["prompt"],
                )
            )
    if not requests:
        raise ValueError(f"No requests loaded from {path}")
    workload_names = {request.workload_name for request in requests}
    if len(workload_names) != 1:
        raise ValueError(f"Expected one workload_name per file, found {workload_names}")
    workload_modes = {request.workload_mode for request in requests}
    if len(workload_modes) != 1:
        raise ValueError(f"Expected one workload_mode per file, found {workload_modes}")
    return requests


def cache_roots() -> list[Path]:
    roots: list[Path] = []
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return list(dict.fromkeys(roots))


def resolve_local_snapshot(name_or_path: str) -> str:
    """Prefer an existing path or cached main snapshot; never downloads here."""

    supplied = Path(name_or_path).expanduser()
    if supplied.exists():
        return str(supplied.resolve())
    repository_dirname = "models--" + name_or_path.replace("/", "--")
    for root in cache_roots():
        repository = root / repository_dirname
        main_ref = repository / "refs" / "main"
        if main_ref.is_file():
            revision = main_ref.read_text(encoding="utf-8").strip()
            snapshot = repository / "snapshots" / revision
            if (snapshot / "config.json").is_file():
                LOGGER.info("Resolved %s to local snapshot %s", name_or_path, snapshot)
                return str(snapshot)
        for snapshot in sorted((repository / "snapshots").glob("*")):
            if (snapshot / "config.json").is_file():
                LOGGER.info("Resolved %s to local snapshot %s", name_or_path, snapshot)
                return str(snapshot)
    LOGGER.warning(
        "No local snapshot found for %s; the backend may attempt remote resolution",
        name_or_path,
    )
    return name_or_path


def nan() -> float:
    return float("nan")


def failed_row(
    request: WorkloadRequest, backend: str, model_name: str, error: str
) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "workload_name": request.workload_name,
        "workload_mode": request.workload_mode,
        "request_type": request.request_type,
        "cache_group_id": request.cache_group_id,
        "intended_prefix_reuse": request.intended_prefix_reuse,
        "unique_marker": request.unique_marker,
        "prompt_len_target": request.prompt_len,
        "output_len_target": request.output_len,
        "prompt_len_actual": nan(),
        "output_len_actual": 0,
        "ttft_ms": nan(),
        "tpot_ms": nan(),
        "total_latency_ms": nan(),
        "tokens_per_second": nan(),
        "num_cached_tokens": nan(),
        "ttft_method": "unavailable",
        "backend": backend,
        "model_name": model_name,
        "success": False,
        "error_msg": error,
        "data_source": "measured",
    }


class GpuMonitor:
    """Best-effort pynvml monitor; failures never abort inference."""

    def __init__(self, output_path: Path, interval_s: float, num_gpus: int) -> None:
        self.output_path = output_path
        self.interval_s = interval_s
        self.num_gpus = num_gpus
        self.samples: list[dict[str, float]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml: Any | None = None
        self._handles: list[Any] = []
        self._start_time = 0.0

    def _visible_physical_indices(self, device_count: int) -> list[int]:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            parsed: list[int] = []
            for value in visible.split(","):
                value = value.strip()
                if value.isdigit():
                    parsed.append(int(value))
            if parsed:
                return [index for index in parsed[: self.num_gpus] if index < device_count]
        return list(range(min(self.num_gpus, device_count)))

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            indices = self._visible_physical_indices(device_count)
            if not indices:
                raise RuntimeError("NVML found no GPU selected for this benchmark")
            self._nvml = pynvml
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(index) for index in indices]
            self._start_time = time.perf_counter()
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
            LOGGER.info("GPU monitor started for physical GPU index(es): %s", indices)
        except Exception as exc:
            LOGGER.warning("GPU sampling unavailable; continuing without it: %s", exc)
            self._nvml = None

    def _sample_loop(self) -> None:
        assert self._nvml is not None
        while not self._stop_event.is_set():
            try:
                utilizations: list[float] = []
                used_mb = 0.0
                total_mb = 0.0
                for handle in self._handles:
                    utilization = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                    memory = self._nvml.nvmlDeviceGetMemoryInfo(handle)
                    utilizations.append(float(utilization.gpu))
                    used_mb += float(memory.used) / (1024**2)
                    total_mb += float(memory.total) / (1024**2)
                self.samples.append(
                    {
                        "timestamp": round(time.perf_counter() - self._start_time, 6),
                        "gpu_util": sum(utilizations) / len(utilizations),
                        "memory_used_mb": used_mb,
                        "memory_total_mb": total_mb,
                    }
                )
            except Exception as exc:
                LOGGER.warning("GPU sampling stopped after an NVML error: %s", exc)
                self._stop_event.set()
                break
            self._stop_event.wait(self.interval_s)

    def stop_and_write(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 3))
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception as exc:
                LOGGER.warning("NVML shutdown warning: %s", exc)
        atomic_write_csv(self.output_path, GPU_TRACE_FIELDS, self.samples)
        if self.samples:
            LOGGER.info("Wrote %d GPU samples to %s", len(self.samples), self.output_path)
        else:
            LOGGER.warning("Wrote header-only GPU trace to %s", self.output_path)


def atomic_write_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def upsert_summary(path: Path, row: dict[str, Any]) -> None:
    existing: list[dict[str, Any]] = []
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != SUMMARY_FIELDS:
                LOGGER.warning(
                    "Migrating benchmark summary from legacy fields: %s",
                    reader.fieldnames,
                )
            existing = [
                {field: old.get(field, "") for field in SUMMARY_FIELDS}
                for old in reader
            ]
    key = (row["workload_file"], row["backend"], row["model_name"])
    existing = [
        old
        for old in existing
        if (old["workload_file"], old["backend"], old["model_name"]) != key
    ]
    existing.append(row)
    atomic_write_csv(path, SUMMARY_FIELDS, existing)


def synchronize_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def calculate_tpot(
    total_latency_ms: float, ttft_ms: float, output_len_actual: int
) -> float:
    if output_len_actual <= 1 or not math.isfinite(total_latency_ms) or not math.isfinite(ttft_ms):
        return nan()
    return max(0.0, total_latency_ms - ttft_ms) / (output_len_actual - 1)


def vllm_row(
    request: WorkloadRequest, output: Any, model_name: str
) -> dict[str, Any]:
    prompt_ids = getattr(output, "prompt_token_ids", None)
    prompt_len_actual = len(prompt_ids) if prompt_ids is not None else nan()
    completions = getattr(output, "outputs", None)
    if not completions:
        raise RuntimeError("vLLM returned no completion")
    output_ids = list(completions[0].token_ids)
    output_len_actual = len(output_ids)
    raw_cached_tokens = getattr(output, "num_cached_tokens", None)
    num_cached_tokens = (
        int(raw_cached_tokens) if raw_cached_tokens is not None else nan()
    )
    metrics = getattr(output, "metrics", None)

    ttft_ms = nan()
    total_latency_ms = nan()
    ttft_method = "unavailable"
    if metrics is not None:
        first_token_latency = float(getattr(metrics, "first_token_latency", 0.0) or 0.0)
        first_token_ts = float(getattr(metrics, "first_token_ts", 0.0) or 0.0)
        last_token_ts = float(getattr(metrics, "last_token_ts", 0.0) or 0.0)
        if first_token_latency > 0:
            ttft_ms = first_token_latency * 1000.0
            ttft_method = "offline_proxy"
            if last_token_ts >= first_token_ts > 0:
                total_latency_ms = (
                    first_token_latency + last_token_ts - first_token_ts
                ) * 1000.0

    tpot_ms = calculate_tpot(total_latency_ms, ttft_ms, output_len_actual)
    tokens_per_second = (
        output_len_actual / (total_latency_ms / 1000.0)
        if output_len_actual > 0
        and math.isfinite(total_latency_ms)
        and total_latency_ms > 0
        else nan()
    )
    return {
        "request_id": request.request_id,
        "workload_name": request.workload_name,
        "workload_mode": request.workload_mode,
        "request_type": request.request_type,
        "cache_group_id": request.cache_group_id,
        "intended_prefix_reuse": request.intended_prefix_reuse,
        "unique_marker": request.unique_marker,
        "prompt_len_target": request.prompt_len,
        "output_len_target": request.output_len,
        "prompt_len_actual": prompt_len_actual,
        "output_len_actual": output_len_actual,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "total_latency_ms": total_latency_ms,
        "tokens_per_second": tokens_per_second,
        "num_cached_tokens": num_cached_tokens,
        "ttft_method": ttft_method,
        "backend": "vllm",
        "model_name": model_name,
        "success": True,
        "error_msg": "",
        "data_source": "measured",
    }


def split_generate_vllm(
    llm: Any,
    jobs: list[tuple[int, WorkloadRequest, Any]],
    model_name: str,
    results: dict[int, dict[str, Any]],
) -> None:
    """Run a batch and recursively isolate request-specific failures."""

    if not jobs:
        return
    try:
        outputs = llm.generate(
            [job[1].prompt for job in jobs],
            sampling_params=[job[2] for job in jobs],
            use_tqdm=True,
        )
        if len(outputs) != len(jobs):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(jobs)} inputs")
        for (index, request, _), output in zip(jobs, outputs, strict=True):
            try:
                results[index] = vllm_row(request, output, model_name)
            except Exception as exc:
                LOGGER.exception("Could not parse output for request %s", request.request_id)
                results[index] = failed_row(request, "vllm", model_name, str(exc))
    except Exception as exc:
        if len(jobs) == 1:
            index, request, _ = jobs[0]
            LOGGER.exception("vLLM generation failed for request %s", request.request_id)
            results[index] = failed_row(request, "vllm", model_name, str(exc))
            return
        midpoint = len(jobs) // 2
        LOGGER.warning(
            "vLLM batch of %d failed (%s); splitting into %d and %d requests",
            len(jobs),
            exc,
            midpoint,
            len(jobs) - midpoint,
        )
        split_generate_vllm(llm, jobs[:midpoint], model_name, results)
        split_generate_vllm(llm, jobs[midpoint:], model_name, results)


def reset_prefix_cache_after_warmup(
    llm: Any, timeout_s: float = 10.0, retry_interval_s: float = 0.1
) -> None:
    """Ensure warmup KV blocks cannot be reused by measured requests."""

    reset_method = getattr(llm, "reset_prefix_cache", None)
    if not callable(reset_method):
        raise RuntimeError(
            "This vLLM version has no reset_prefix_cache API; refusing to measure "
            "with potentially warm prefix-cache state"
        )
    deadline = time.monotonic() + timeout_s
    attempts = 0
    while True:
        attempts += 1
        if bool(reset_method()):
            LOGGER.info(
                "Prefix cache reset after warmup succeeded (attempts=%d)", attempts
            )
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Prefix cache reset did not succeed within {timeout_s:.1f}s"
            )
        time.sleep(retry_interval_s)


def initialize_vllm(config: dict[str, Any], seed: int) -> tuple[Any, Callable[[], None]]:
    from vllm import LLM

    ensure_run_file_handler()
    model_path = resolve_local_snapshot(str(config["model_name"]))
    tokenizer_path = resolve_local_snapshot(str(config["tokenizer_name"]))
    LOGGER.info("Initializing vLLM model=%s", model_path)
    LOGGER.info(
        "Prefix caching explicitly configured as %s",
        bool(config.get("enable_prefix_caching", False)),
    )
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_path,
        dtype=str(config["dtype"]),
        max_model_len=int(config["max_model_len"]),
        gpu_memory_utilization=float(config["gpu_memory_utilization"]),
        tensor_parallel_size=int(config["tensor_parallel_size"]),
        enable_prefix_caching=bool(config.get("enable_prefix_caching", False)),
        disable_log_stats=False,
        seed=seed,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )

    def cleanup() -> None:
        nonlocal llm
        del llm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return llm, cleanup


def run_vllm(
    llm: Any,
    requests: list[WorkloadRequest],
    warmup_count: int,
    model_name: str,
    max_model_len: int,
    seed: int,
    on_measurement_start: Callable[[], None],
) -> BackendResult:
    from vllm import SamplingParams

    def params(request: WorkloadRequest) -> Any:
        return SamplingParams(
            temperature=0.0,
            max_tokens=request.output_len,
            min_tokens=request.output_len,
            ignore_eos=True,
            seed=seed,
        )

    warmups = requests[:warmup_count]
    if warmups:
        LOGGER.info("Running %d unrecorded vLLM warmup requests", len(warmups))
        try:
            llm.generate(
                [request.prompt for request in warmups],
                sampling_params=[params(request) for request in warmups],
                use_tqdm=True,
            )
        except Exception as exc:
            LOGGER.warning("Warmup failed; measured requests will still be attempted: %s", exc)
        # APC may retain exact warmup prompts. Clear only the prefix KV cache;
        # compiled kernels and CUDA graphs remain warm for fair measurement.
        reset_prefix_cache_after_warmup(llm)

    results: dict[int, dict[str, Any]] = {}
    jobs: list[tuple[int, WorkloadRequest, Any]] = []
    for index, request in enumerate(requests):
        if request.prompt_len + request.output_len > max_model_len:
            message = (
                f"Target prompt+output length {request.prompt_len + request.output_len} "
                f"exceeds max_model_len={max_model_len}"
            )
            results[index] = failed_row(request, "vllm", model_name, message)
        else:
            jobs.append((index, request, params(request)))

    LOGGER.info("Starting measured vLLM batch with %d valid requests", len(jobs))
    on_measurement_start()
    synchronize_cuda()
    started = time.perf_counter()
    split_generate_vllm(llm, jobs, model_name, results)
    synchronize_cuda()
    wall_time_s = time.perf_counter() - started
    rows = [results[index] for index in range(len(requests))]
    methods = {row["ttft_method"] for row in rows if row["success"]}
    method = methods.pop() if len(methods) == 1 else "unavailable"
    return BackendResult(rows, wall_time_s, "vllm", method)


def initialize_hf(config: dict[str, Any], seed: int) -> tuple[Any, Any, Callable[[], None]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ensure_run_file_handler()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model_path = resolve_local_snapshot(str(config["model_name"]))
    tokenizer_path = resolve_local_snapshot(str(config["tokenizer_name"]))
    dtype_name = str(config["dtype"]).lower()
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype_name)
    if dtype is None:
        raise ValueError(f"Unsupported HuggingFace dtype: {dtype_name}")
    LOGGER.info("Initializing HuggingFace fallback model=%s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    def cleanup() -> None:
        nonlocal model
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model, tokenizer, cleanup


def run_hf(
    model: Any,
    tokenizer: Any,
    requests: list[WorkloadRequest],
    warmup_count: int,
    model_name: str,
    max_model_len: int,
    on_measurement_start: Callable[[], None],
) -> BackendResult:
    import torch

    device = next(model.parameters()).device

    def prepare(request: WorkloadRequest) -> Any:
        return tokenizer(request.prompt, return_tensors="pt", add_special_tokens=True).to(
            device
        )

    def generate(inputs: Any, new_tokens: int) -> Any:
        return model.generate(
            **inputs,
            do_sample=False,
            min_new_tokens=new_tokens,
            max_new_tokens=new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    for request in requests[:warmup_count]:
        try:
            inputs = prepare(request)
            with torch.inference_mode():
                generate(inputs, request.output_len)
        except Exception as exc:
            LOGGER.warning("HuggingFace warmup request %s failed: %s", request.request_id, exc)

    rows: list[dict[str, Any]] = []
    on_measurement_start()
    measured_start = time.perf_counter()
    for request in requests:
        try:
            inputs = prepare(request)
            prompt_len_actual = int(inputs["input_ids"].shape[-1])
            if prompt_len_actual + request.output_len > max_model_len:
                raise ValueError(
                    f"Actual prompt+output length {prompt_len_actual + request.output_len} "
                    f"exceeds max_model_len={max_model_len}"
                )

            # A separate one-token call is only a first-output proxy, not streaming TTFT.
            synchronize_cuda()
            proxy_start = time.perf_counter()
            with torch.inference_mode():
                generate(inputs, 1)
            synchronize_cuda()
            ttft_ms = (time.perf_counter() - proxy_start) * 1000.0

            synchronize_cuda()
            generation_start = time.perf_counter()
            with torch.inference_mode():
                generated = generate(inputs, request.output_len)
            synchronize_cuda()
            total_latency_ms = (time.perf_counter() - generation_start) * 1000.0
            output_len_actual = int(generated.shape[-1]) - prompt_len_actual
            tpot_ms = calculate_tpot(total_latency_ms, ttft_ms, output_len_actual)
            token_rate = (
                output_len_actual / (total_latency_ms / 1000.0)
                if output_len_actual > 0 and total_latency_ms > 0
                else nan()
            )
            rows.append(
                {
                    "request_id": request.request_id,
                    "workload_name": request.workload_name,
                    "workload_mode": request.workload_mode,
                    "request_type": request.request_type,
                    "cache_group_id": request.cache_group_id,
                    "intended_prefix_reuse": request.intended_prefix_reuse,
                    "unique_marker": request.unique_marker,
                    "prompt_len_target": request.prompt_len,
                    "output_len_target": request.output_len,
                    "prompt_len_actual": prompt_len_actual,
                    "output_len_actual": output_len_actual,
                    "ttft_ms": ttft_ms,
                    "tpot_ms": tpot_ms,
                    "total_latency_ms": total_latency_ms,
                    "tokens_per_second": token_rate,
                    "num_cached_tokens": nan(),
                    "ttft_method": "offline_proxy",
                    "backend": "hf",
                    "model_name": model_name,
                    "success": True,
                    "error_msg": "",
                    "data_source": "measured",
                }
            )
        except Exception as exc:
            LOGGER.exception("HuggingFace generation failed for %s", request.request_id)
            rows.append(failed_row(request, "hf", model_name, str(exc)))
    wall_time_s = time.perf_counter() - measured_start
    return BackendResult(rows, wall_time_s, "hf", "offline_proxy")


def make_summary(
    result: BackendResult,
    requests: list[WorkloadRequest],
    workload_path: Path,
    model_name: str,
    seed: int,
    warmup_count: int,
    prefix_caching_enabled: bool,
) -> dict[str, Any]:
    successful = [row for row in result.rows if row["success"]]
    output_tokens = sum(int(row["output_len_actual"]) for row in successful)
    wall_time = result.wall_time_s
    cached_values = [
        float(row["num_cached_tokens"])
        for row in successful
        if math.isfinite(float(row["num_cached_tokens"]))
    ]
    cache_hit_requests = sum(value > 0 for value in cached_values)
    cached_prompt_tokens = sum(cached_values)
    prompt_tokens = sum(
        float(row["prompt_len_actual"])
        for row in successful
        if math.isfinite(float(row["prompt_len_actual"]))
    )
    valid_metrics = bool(successful) and all(
        math.isfinite(float(row["ttft_ms"]))
        and math.isfinite(float(row["total_latency_ms"]))
        for row in successful
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "workload_name": requests[0].workload_name,
        "workload_mode": requests[0].workload_mode,
        "workload_file": str(workload_path),
        "backend": result.backend,
        "model_name": model_name,
        "attempted_requests": len(result.rows),
        "successful_requests": len(successful),
        "failed_requests": len(result.rows) - len(successful),
        "wall_time_s": wall_time,
        "requests_per_second": len(successful) / wall_time if wall_time > 0 else nan(),
        "output_tokens": output_tokens,
        "tokens_per_second": output_tokens / wall_time if wall_time > 0 else nan(),
        "ttft_method": result.ttft_method,
        "prefix_caching_enabled": prefix_caching_enabled,
        "intended_prefix_reuse_ratio": sum(
            request.intended_prefix_reuse for request in requests
        )
        / len(requests),
        "cache_hit_requests": cache_hit_requests if cached_values else nan(),
        "cache_hit_request_ratio": (
            cache_hit_requests / len(cached_values) if cached_values else nan()
        ),
        "cached_prompt_tokens": cached_prompt_tokens if cached_values else nan(),
        "cached_prompt_token_ratio": (
            cached_prompt_tokens / prompt_tokens
            if cached_values and prompt_tokens > 0
            else nan()
        ),
        "avg_num_cached_tokens": (
            cached_prompt_tokens / len(cached_values) if cached_values else nan()
        ),
        "valid_metrics": valid_metrics,
        "seed": seed,
        "warmup_requests": warmup_count,
        "data_source": "measured",
    }


def release_backend(cleanup: Callable[[], None] | None) -> None:
    if cleanup is not None:
        try:
            cleanup()
        except Exception as exc:
            LOGGER.warning("Backend cleanup warning: %s", exc)


def run(args: argparse.Namespace) -> bool:
    if args.max_requests <= 0:
        raise ValueError("--max-requests must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative")
    if not math.isfinite(args.gpu_sample_interval) or args.gpu_sample_interval <= 0:
        raise ValueError("--gpu-sample-interval must be positive and finite")

    config = load_yaml(args.model_config)
    validate_model_config(config)
    requests = load_workload(args.workload, args.max_requests)
    model_name = str(config["model_name"])
    workload_name = requests[0].workload_name
    workload_file_label = args.workload.stem.removeprefix("workload_")
    gpu_trace_path = (
        PROJECT_ROOT
        / "outputs"
        / "metrics"
        / f"gpu_trace_{workload_file_label}.csv"
    )
    LOGGER.info(
        "Loaded %d requests from %s (workload=%s)",
        len(requests),
        args.workload,
        workload_name,
    )

    backend_name = args.backend
    cleanup: Callable[[], None] | None = None
    backend_objects: tuple[Any, ...] = ()
    initialization_errors: list[str] = []
    try:
        if backend_name == "vllm":
            try:
                llm, cleanup = initialize_vllm(config, args.seed)
                backend_objects = (llm,)
            except Exception as exc:
                LOGGER.exception("vLLM initialization failed")
                initialization_errors.append(f"vLLM: {type(exc).__name__}: {exc}")
                release_backend(cleanup)
                cleanup = None
                if args.no_hf_fallback:
                    raise
                LOGGER.warning("Attempting HuggingFace fallback")
                backend_name = "hf"
        if backend_name == "hf" and not backend_objects:
            try:
                model, tokenizer, cleanup = initialize_hf(config, args.seed)
                backend_objects = (model, tokenizer)
            except Exception as exc:
                LOGGER.exception("HuggingFace initialization failed")
                initialization_errors.append(f"HuggingFace: {type(exc).__name__}: {exc}")
                raise
    except Exception:
        error = " | ".join(initialization_errors) or "Backend initialization failed"
        rows = [failed_row(request, backend_name, model_name, error) for request in requests]
        atomic_write_csv(args.output, REQUEST_FIELDS, rows)
        result = BackendResult(rows, 0.0, backend_name, "unavailable")
        upsert_summary(
            args.summary_output,
            make_summary(
                result,
                requests,
                args.workload,
                model_name,
                args.seed,
                args.warmup,
                bool(config.get("enable_prefix_caching", False)),
            ),
        )
        atomic_write_csv(gpu_trace_path, GPU_TRACE_FIELDS, [])
        LOGGER.error("No benchmark measurements were produced; failure rows were saved")
        return False

    monitor = GpuMonitor(
        gpu_trace_path,
        args.gpu_sample_interval,
        int(config["tensor_parallel_size"]),
    )
    on_measurement_start = (
        monitor.start if not args.disable_gpu_monitor else lambda: None
    )
    try:
        try:
            if backend_name == "vllm":
                result = run_vllm(
                    backend_objects[0],
                    requests,
                    min(args.warmup, len(requests)),
                    model_name,
                    int(config["max_model_len"]),
                    args.seed,
                    on_measurement_start,
                )
            else:
                result = run_hf(
                    backend_objects[0],
                    backend_objects[1],
                    requests,
                    min(args.warmup, len(requests)),
                    model_name,
                    int(config["max_model_len"]),
                    on_measurement_start,
                )
        except Exception as exc:
            LOGGER.exception("Unexpected backend execution failure")
            error = f"{type(exc).__name__}: {exc}"
            rows = [
                failed_row(request, backend_name, model_name, error)
                for request in requests
            ]
            result = BackendResult(rows, 0.0, backend_name, "unavailable")
    finally:
        if not args.disable_gpu_monitor:
            monitor.stop_and_write()
        else:
            atomic_write_csv(gpu_trace_path, GPU_TRACE_FIELDS, [])
            LOGGER.info("GPU monitoring disabled; wrote header-only trace %s", gpu_trace_path)
        release_backend(cleanup)

    atomic_write_csv(args.output, REQUEST_FIELDS, result.rows)
    summary = make_summary(
        result,
        requests,
        args.workload,
        model_name,
        args.seed,
        args.warmup,
        bool(config.get("enable_prefix_caching", False)),
    )
    upsert_summary(args.summary_output, summary)
    LOGGER.info("Wrote per-request metrics: %s", args.output)
    LOGGER.info("Updated benchmark summary: %s", args.summary_output)
    LOGGER.info(
        "Benchmark finished: success=%d, failed=%d, req/s=%.4f, tok/s=%.4f",
        summary["successful_requests"],
        summary["failed_requests"],
        summary["requests_per_second"],
        summary["tokens_per_second"],
    )
    successful_rows = [row for row in result.rows if row["success"]]
    latency_rows = [
        row
        for row in successful_rows
        if math.isfinite(float(row["ttft_ms"]))
        and math.isfinite(float(row["total_latency_ms"]))
    ]
    if successful_rows and not latency_rows:
        LOGGER.error(
            "Generation succeeded, but every request is missing TTFT/total latency; "
            "the run is incomplete and will return a non-zero status"
        )
        return False
    if len(latency_rows) < len(successful_rows):
        LOGGER.warning(
            "%d of %d successful requests have incomplete latency metrics",
            len(successful_rows) - len(latency_rows),
            len(successful_rows),
        )
    return bool(successful_rows)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    log_path = args.log_file or (
        PROJECT_ROOT / "outputs" / "logs" / f"benchmark_{args.workload.stem}.log"
    )
    configure_run_logging(log_path, args.backend)
    LOGGER.info("Writing run log to %s", log_path)
    try:
        success = run(args)
    except Exception as exc:
        LOGGER.exception("Benchmark failed before results could be finalized: %s", exc)
        sys.exit(1)
    if not success:
        sys.exit(2)


if __name__ == "__main__":
    main()
