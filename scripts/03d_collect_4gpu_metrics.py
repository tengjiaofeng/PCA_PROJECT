#!/usr/bin/env python3
"""Sample utilization, memory, power, and temperature for selected GPUs."""

from __future__ import annotations

import argparse
import csv
import math
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


FIELDS = [
    "timestamp", "gpu_id", "gpu_util", "memory_used_mb", "memory_total_mb",
    "power_watt", "temperature_c", "run_id", "serving_mode", "workload_name",
    "arrival_rate", "concurrency", "collector_backend",
]
STOP = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--interval-ms", type=int, default=500)
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/metrics/real4gpu/gpu_trace_colocated_4replica_workload.csv"),
    )
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--mode", default="")
    parser.add_argument("--workload-name", default="")
    parser.add_argument("--arrival-rate", type=float, default=math.nan)
    parser.add_argument("--concurrency", type=int, default=0)
    return parser.parse_args()


def stop_handler(_signum: int, _frame: Any) -> None:
    global STOP
    STOP = True


def pynvml_sampler(gpu_ids: list[int]) -> tuple[str, Callable[[], list[dict[str, Any]]], Callable[[], None]]:
    import pynvml

    pynvml.nvmlInit()
    handles = {gpu: pynvml.nvmlDeviceGetHandleByIndex(gpu) for gpu in gpu_ids}

    def sample() -> list[dict[str, Any]]:
        rows = []
        for gpu, handle in handles.items():
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except pynvml.NVMLError:
                power = math.nan
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except pynvml.NVMLError:
                temperature = math.nan
            rows.append({
                "gpu_id": gpu, "gpu_util": float(util.gpu),
                "memory_used_mb": memory.used / 1024**2,
                "memory_total_mb": memory.total / 1024**2,
                "power_watt": power, "temperature_c": temperature,
            })
        return rows

    return "pynvml", sample, pynvml.nvmlShutdown


def nvidia_smi_sampler(gpu_ids: list[int]) -> tuple[str, Callable[[], list[dict[str, Any]]], Callable[[], None]]:
    query = "index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"

    def sample() -> list[dict[str, Any]]:
        command = [
            "nvidia-smi", f"--id={','.join(map(str, gpu_ids))}",
            f"--query-gpu={query}", "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=10
        )
        rows = []
        for line in completed.stdout.splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) != 6:
                raise RuntimeError(f"unexpected nvidia-smi row: {line}")
            values = []
            for value in fields:
                try:
                    values.append(float(value))
                except ValueError:
                    values.append(math.nan)
            rows.append({
                "gpu_id": int(values[0]), "gpu_util": values[1],
                "memory_used_mb": values[2], "memory_total_mb": values[3],
                "power_watt": values[4], "temperature_c": values[5],
            })
        return rows

    return "nvidia-smi", sample, lambda: None


def main() -> None:
    args = parse_args()
    gpu_ids = [int(item.strip()) for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--gpu-ids must contain unique integer IDs")
    if args.interval_ms <= 0 or (args.duration_s is not None and args.duration_s <= 0):
        raise ValueError("sampling interval and duration must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    warning_path = args.output.with_suffix(".warnings.log")
    try:
        backend, sample, shutdown = pynvml_sampler(gpu_ids)
    except Exception as nvml_error:
        warning = (
            f"{datetime.now(timezone.utc).isoformat()} WARNING: pynvml unavailable: "
            f"{type(nvml_error).__name__}: {nvml_error}; falling back to nvidia-smi.\n"
        )
        warning_path.write_text(warning, encoding="utf-8")
        print(warning.strip(), file=sys.stderr)
        try:
            backend, sample, shutdown = nvidia_smi_sampler(gpu_ids)
            sample()  # Fail before entering a long collection loop.
        except Exception as smi_error:
            message = (
                f"GPU sampling unavailable: {type(smi_error).__name__}: {smi_error}. "
                "The online benchmark may continue independently."
            )
            with warning_path.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")
            raise RuntimeError(message) from smi_error

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    started = time.monotonic()
    deadline = started + args.duration_s if args.duration_s is not None else None
    samples = 0
    try:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            while not STOP and (deadline is None or time.monotonic() < deadline):
                timestamp = datetime.now(timezone.utc).isoformat()
                try:
                    current = sample()
                except Exception as exc:
                    message = (
                        f"{timestamp} WARNING: sample failed: {type(exc).__name__}: {exc}"
                    )
                    print(message, file=sys.stderr)
                    with warning_path.open("a", encoding="utf-8") as warnings:
                        warnings.write(message + "\n")
                    time.sleep(args.interval_ms / 1000.0)
                    continue
                for row in current:
                    row.update({
                        "timestamp": timestamp, "run_id": args.run_id,
                        "serving_mode": args.mode, "workload_name": args.workload_name,
                        "arrival_rate": args.arrival_rate, "concurrency": args.concurrency,
                        "collector_backend": backend,
                    })
                    writer.writerow(row)
                    samples += 1
                handle.flush()
                time.sleep(args.interval_ms / 1000.0)
    finally:
        shutdown()
    print(f"Wrote {samples} GPU samples to {args.output} using {backend}")


if __name__ == "__main__":
    main()
