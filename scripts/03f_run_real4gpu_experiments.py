#!/usr/bin/env python3
"""Safely orchestrate real-4GPU server modes and online workload runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "scripts/03b_launch_4gpu_servers.py"
CLIENT = PROJECT_ROOT / "scripts/03c_run_online_workload_client.py"
COLLECTOR = PROJECT_ROOT / "scripts/03d_collect_4gpu_metrics.py"
NIXL_MODES = (
    "real_pd_nixl_1p3d",
    "real_pd_nixl_2p2d",
    "real_pd_nixl_3p1d",
)
FORMAL_MODES = ("colocated_4replica", "aggregated_tp4", *NIXL_MODES)
MAIN_WORKLOADS = (
    "data/processed/workload_mixed_30p70d_synthetic_unique.jsonl",
    "data/processed/workload_mixed_50p50d_synthetic_unique.jsonl",
    "data/processed/workload_mixed_70p30d_synthetic_unique.jsonl",
)
STATUS_FIELDS = (
    "run_id", "preset", "serving_mode", "workload", "arrival_rate", "repeat",
    "seed", "status", "return_code", "elapsed_s", "output_csv", "error_msg",
)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def comma_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("list must not be empty")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/real4gpu.yaml"))
    parser.add_argument(
        "--preset", choices=("pilot", "baseline", "formal"), default="pilot"
    )
    parser.add_argument("--modes", help="Comma-separated override.")
    parser.add_argument("--workloads", help="Comma-separated JSONL path override.")
    parser.add_argument("--arrival-rates", help="Comma-separated req/s override.")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", type=str2bool, default=False)
    parser.add_argument(
        "--print-commands", type=str2bool, default=None,
        help="Print every command; defaults to false for the large formal matrix.",
    )
    parser.add_argument("--skip-existing", type=str2bool, default=True)
    parser.add_argument("--continue-on-error", type=str2bool, default=False)
    parser.add_argument("--collect-gpu-metrics", type=str2bool, default=False)
    parser.add_argument(
        "--gpu-monitoring-approved", type=str2bool, default=False,
        help="Explicit approval because the collector may fall back to nvidia-smi.",
    )
    parser.add_argument("--metrics-dir", type=Path, default=Path("outputs/metrics/real4gpu"))
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/logs/real4gpu"))
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return data


def workload_label(path: Path) -> str:
    label = path.stem.removeprefix("workload_")
    return label.removesuffix("_synthetic_unique")


def rate_label(rate: float) -> str:
    return f"{rate:g}".replace(".", "p")


def successful_csv(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return False
    return bool(rows) and any(
        str(row.get("success", "")).lower() in {"true", "1"} for row in rows
    )


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def run_logged(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {command_text(command)}")
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {command_text(command)}\n")
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return process.wait()


def write_status(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def stop_servers(log_dir: Path, suffix: str) -> int:
    return run_logged(
        [sys.executable, str(LAUNCHER), "--output-log-dir", str(log_dir), "--stop-only"],
        log_dir / f"orchestrator_stop_{suffix}.log",
    )


def resolve_matrix(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    if args.preset == "pilot":
        defaults = {
            "modes": list(NIXL_MODES),
            "workloads": [MAIN_WORKLOADS[1]],
            "arrival_rates": [1.0], "repeats": 1,
            "max_requests": 20, "warmup": 1, "concurrency": 4,
        }
    elif args.preset == "baseline":
        defaults = {
            "modes": ["colocated_4replica", "aggregated_tp4"],
            "workloads": [MAIN_WORKLOADS[1]],
            "arrival_rates": [1.0], "repeats": 1,
            "max_requests": 20, "warmup": 1, "concurrency": 4,
        }
    else:
        defaults = {
            "modes": list(FORMAL_MODES), "workloads": list(MAIN_WORKLOADS),
            "arrival_rates": [float(x) for x in config.get("arrival_rates", [])],
            "repeats": 3, "max_requests": 200, "warmup": 5, "concurrency": 64,
        }
    modes = comma_list(args.modes) or defaults["modes"]
    workloads = comma_list(args.workloads) or defaults["workloads"]
    rate_values = comma_list(args.arrival_rates)
    rates = [float(item) for item in rate_values] if rate_values else defaults["arrival_rates"]
    matrix = {
        "modes": modes, "workloads": workloads, "arrival_rates": rates,
        "repeats": args.repeats if args.repeats is not None else defaults["repeats"],
        "max_requests": (
            args.max_requests if args.max_requests is not None else defaults["max_requests"]
        ),
        "warmup": args.warmup if args.warmup is not None else defaults["warmup"],
        "concurrency": (
            args.concurrency if args.concurrency is not None else defaults["concurrency"]
        ),
    }
    if matrix["repeats"] <= 0 or matrix["max_requests"] <= 0:
        raise ValueError("repeats and max_requests must be positive")
    if matrix["warmup"] < 0 or matrix["concurrency"] <= 0:
        raise ValueError("warmup must be non-negative and concurrency positive")
    if any(rate <= 0 for rate in rates):
        raise ValueError("arrival rates must be positive")
    for workload in workloads:
        if not (PROJECT_ROOT / workload).is_file():
            raise FileNotFoundError(PROJECT_ROOT / workload)
    return matrix


def build_jobs(args: argparse.Namespace, matrix: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for mode in matrix["modes"]:
        for workload_text in matrix["workloads"]:
            workload = Path(workload_text)
            label = workload_label(workload)
            for rate in matrix["arrival_rates"]:
                for repeat in range(1, matrix["repeats"] + 1):
                    seed = args.seed + repeat - 1
                    run_id = (
                        f"{args.preset}-{label}-{mode}-r{rate_label(rate)}-rep{repeat}"
                    )
                    output = args.metrics_dir / f"online_{run_id}.csv"
                    jobs.append({
                        "mode": mode, "workload": workload, "workload_label": label,
                        "arrival_rate": rate, "repeat": repeat, "seed": seed,
                        "run_id": run_id, "output": output,
                    })
    return jobs


def launch_command(args: argparse.Namespace, mode: str) -> list[str]:
    command = [
        sys.executable, str(LAUNCHER), "--config", str(args.config),
        "--mode", mode, "--output-log-dir", str(args.log_dir),
        "--stop-existing", "--dry-run", "false",
    ]
    if mode.startswith("real_pd_"):
        command.extend(["--allow-experimental-pd", "true"])
    return command


def client_command(
    args: argparse.Namespace, matrix: dict[str, Any], job: dict[str, Any]
) -> list[str]:
    return [
        sys.executable, str(CLIENT), "--config", str(args.config),
        "--mode", job["mode"], "--workload", str(job["workload"]),
        "--output", str(job["output"]), "--run-id", job["run_id"],
        "--arrival-rate", str(job["arrival_rate"]),
        "--concurrency", str(matrix["concurrency"]),
        "--max-requests", str(matrix["max_requests"]),
        "--warmup", str(matrix["warmup"]),
        "--request-timeout-s", str(args.request_timeout_s),
        "--stream", "true", "--seed", str(job["seed"]),
        "--log-file", str(args.log_dir / f"client_{job['run_id']}.log"),
    ]


def start_collector(
    args: argparse.Namespace, matrix: dict[str, Any], job: dict[str, Any]
) -> tuple[subprocess.Popen[str], Any]:
    output = args.metrics_dir / f"gpu_trace_{job['run_id']}.csv"
    command = [
        sys.executable, str(COLLECTOR), "--gpu-ids", "0,1,2,3",
        "--interval-ms", "500", "--output", str(output),
        "--run-id", job["run_id"], "--mode", job["mode"],
        "--workload-name", job["workload_label"],
        "--arrival-rate", str(job["arrival_rate"]),
        "--concurrency", str(matrix["concurrency"]),
    ]
    log_handle = (args.log_dir / f"collector_{job['run_id']}.log").open(
        "w", encoding="utf-8"
    )
    process = subprocess.Popen(
        command, cwd=PROJECT_ROOT, stdout=log_handle, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
    )
    return process, log_handle


def main() -> None:
    args = parse_args()
    if args.collect_gpu_metrics and not args.gpu_monitoring_approved:
        raise RuntimeError(
            "GPU metrics require explicit --gpu-monitoring-approved true because "
            "the collector may invoke nvidia-smi as a fallback."
        )
    os.chdir(PROJECT_ROOT)
    config = load_yaml(args.config)
    matrix = resolve_matrix(args, config)
    jobs = build_jobs(args, matrix)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable, "preset": args.preset, "execute": args.execute,
        "matrix": matrix,
        "jobs": [{**job, "workload": str(job["workload"]), "output": str(job["output"])} for job in jobs],
    }
    plan_path = args.log_dir / f"experiment_plan_{args.preset}.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(
        f"Plan: {len(matrix['modes'])} modes, {len(jobs)} runs; "
        f"written to {plan_path}"
    )
    if not args.execute:
        print_commands = (
            args.print_commands
            if args.print_commands is not None
            else args.preset != "formal"
        )
        for mode in matrix["modes"]:
            mode_jobs = [item for item in jobs if item["mode"] == mode]
            if print_commands:
                print(f"\n[launch {mode}] {command_text(launch_command(args, mode))}")
                for job in mode_jobs:
                    print(f"[run] {command_text(client_command(args, matrix, job))}")
                print(f"[stop {mode}] {sys.executable} {LAUNCHER} --stop-only")
            else:
                print(f"  {mode}: {len(mode_jobs)} runs")
        print("\nPlan only: pass --execute true to run it.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    status_path = args.metrics_dir / f"experiment_status_{args.preset}_{stamp}.csv"
    status_rows: list[dict[str, Any]] = []
    try:
        for mode in matrix["modes"]:
            launch_rc = run_logged(
                launch_command(args, mode), args.log_dir / f"orchestrator_launch_{mode}.log"
            )
            if launch_rc != 0:
                raise RuntimeError(f"server launch failed for {mode}, rc={launch_rc}")
            try:
                for job in (item for item in jobs if item["mode"] == mode):
                    base = {
                        "run_id": job["run_id"], "preset": args.preset,
                        "serving_mode": mode, "workload": job["workload_label"],
                        "arrival_rate": job["arrival_rate"], "repeat": job["repeat"],
                        "seed": job["seed"], "output_csv": str(job["output"]),
                    }
                    if args.skip_existing and successful_csv(job["output"]):
                        status_rows.append({**base, "status": "skipped_existing_success", "return_code": 0, "elapsed_s": 0.0, "error_msg": ""})
                        write_status(status_path, status_rows)
                        continue
                    collector = None
                    collector_log = None
                    started = time.monotonic()
                    try:
                        if args.collect_gpu_metrics:
                            collector, collector_log = start_collector(args, matrix, job)
                            time.sleep(1.0)
                        rc = run_logged(
                            client_command(args, matrix, job),
                            args.log_dir / f"orchestrator_client_{job['run_id']}.log",
                        )
                    finally:
                        if collector is not None:
                            os.killpg(collector.pid, signal.SIGINT)
                            try:
                                collector.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                collector.terminate()
                                collector.wait(timeout=5)
                        if collector_log is not None:
                            collector_log.close()
                    elapsed = time.monotonic() - started
                    valid = rc == 0 and successful_csv(job["output"])
                    status_rows.append({
                        **base, "status": "success" if valid else "failed",
                        "return_code": rc, "elapsed_s": round(elapsed, 3),
                        "error_msg": "" if valid else "client failed or no successful CSV row",
                    })
                    write_status(status_path, status_rows)
                    if not valid and not args.continue_on_error:
                        raise RuntimeError(f"experiment failed: {job['run_id']}")
            finally:
                stop_servers(args.log_dir, mode)
    except KeyboardInterrupt:
        print("Interrupted; stopping managed server processes.", file=sys.stderr)
        raise
    finally:
        stop_servers(args.log_dir, "final")
    print(f"Completed {len(status_rows)} runs; status: {status_path}")


if __name__ == "__main__":
    main()
