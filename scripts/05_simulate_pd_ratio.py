#!/usr/bin/env python3
"""Simulate trace-driven Prefill–Decode disaggregation resource ratios."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


LOGGER = logging.getLogger("simulate_pd_ratio")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUEST_FIELDS = [
    "workload_name",
    "request_id",
    "ratio_p",
    "ratio_d",
    "arrival_rate",
    "kv_transfer_ms",
    "arrival_time",
    "prefill_start",
    "prefill_end",
    "decode_ready_time",
    "decode_start",
    "decode_end",
    "prefill_queue_wait_ms",
    "decode_queue_wait_ms",
    "ttft_ms",
    "tpot_ms",
    "total_latency_ms",
    "success",
    "data_source",
]

RESULT_FIELDS = [
    "workload_name",
    "ratio_p",
    "ratio_d",
    "arrival_rate",
    "kv_transfer_ms",
    "num_requests",
    "avg_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "avg_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "avg_total_latency_ms",
    "p95_total_latency_ms",
    "throughput_req_s",
    "throughput_tok_s",
    "goodput_req_s",
    "prefill_util",
    "decode_util",
    "avg_prefill_queue_wait_ms",
    "avg_decode_queue_wait_ms",
    "data_source",
]


@dataclass(frozen=True)
class Request:
    request_id: str
    workload_name: str
    prompt_len: int
    output_len: int
    original_arrival_time_s: float
    trace_order: int


@dataclass
class SimulatedRequest:
    request: Request
    arrival_time_s: float
    prefill_service_ms: float
    decode_service_ms: float
    prefill_start_s: float = 0.0
    prefill_end_s: float = 0.0
    decode_ready_time_s: float = 0.0
    decode_start_s: float = 0.0
    decode_end_s: float = 0.0
    prefill_queue_wait_ms: float = 0.0
    decode_queue_wait_ms: float = 0.0


@dataclass
class ServerPool:
    """Identical FCFS service lanes represented by earliest availability."""

    size: int
    _available: list[tuple[float, int]] = field(init=False, repr=False)
    busy_time_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("ServerPool size must be positive")
        self._available = [(0.0, server_id) for server_id in range(self.size)]
        heapq.heapify(self._available)

    def schedule(self, ready_time_s: float, service_time_ms: float) -> tuple[float, float]:
        if not math.isfinite(service_time_ms) or service_time_ms < 0:
            raise ValueError(f"Invalid service time: {service_time_ms}")
        available_time_s, server_id = heapq.heappop(self._available)
        start_s = max(ready_time_s, available_time_s)
        end_s = start_s + service_time_ms / 1000.0
        heapq.heappush(self._available, (end_s, server_id))
        self.busy_time_ms += service_time_ms
        return start_s, end_s


class StageProfile:
    """Piecewise-linear prefill and bilinear decode service-time lookup."""

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Stage profile does not exist: {path}")
        frame = pd.read_csv(path)
        required = {
            "prompt_len",
            "output_len",
            "ttft_ms_mean",
            "decode_total_ms_mean",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Stage profile is missing columns: {', '.join(missing)}")
        for column in required:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[list(required)].isna().any().any():
            raise ValueError("Stage profile contains non-numeric required values")
        if frame.duplicated(["prompt_len", "output_len"]).any():
            raise ValueError("Stage profile has duplicate prompt/output points")

        self.prompt_knots = np.sort(frame["prompt_len"].unique().astype(float))
        self.output_knots = np.sort(frame["output_len"].unique().astype(float))
        prefill = frame[frame["output_len"] == 1].sort_values("prompt_len")
        if not np.array_equal(
            prefill["prompt_len"].to_numpy(dtype=float), self.prompt_knots
        ):
            raise ValueError("Stage profile needs output_len=1 at every prompt knot")
        self.prefill_costs = prefill["ttft_ms_mean"].to_numpy(dtype=float)

        pivot = frame.pivot(
            index="prompt_len", columns="output_len", values="decode_total_ms_mean"
        ).reindex(index=self.prompt_knots, columns=self.output_knots)
        if pivot.isna().any().any():
            raise ValueError("Decode stage profile must be a complete rectangular grid")
        self.decode_grid = pivot.to_numpy(dtype=float)
        if (self.prefill_costs <= 0).any() or (self.decode_grid < 0).any():
            raise ValueError("Stage service times must be non-negative")

    def validate_range(self, requests: list[Request]) -> None:
        prompt_values = [request.prompt_len for request in requests]
        output_values = [request.output_len for request in requests]
        if min(prompt_values) < self.prompt_knots[0] or max(prompt_values) > self.prompt_knots[-1]:
            raise ValueError(
                f"Workload prompt lengths [{min(prompt_values)}, {max(prompt_values)}] "
                f"exceed profile range [{self.prompt_knots[0]}, {self.prompt_knots[-1]}]"
            )
        if min(output_values) < self.output_knots[0] or max(output_values) > self.output_knots[-1]:
            raise ValueError(
                f"Workload output lengths [{min(output_values)}, {max(output_values)}] "
                f"exceed profile range [{self.output_knots[0]}, {self.output_knots[-1]}]"
            )

    def prefill_service_ms(self, prompt_len: int) -> float:
        return float(np.interp(prompt_len, self.prompt_knots, self.prefill_costs))

    def decode_service_ms(self, prompt_len: int, output_len: int) -> float:
        per_prompt = np.asarray(
            [
                np.interp(output_len, self.output_knots, row)
                for row in self.decode_grid
            ],
            dtype=float,
        )
        return max(0.0, float(np.interp(prompt_len, self.prompt_knots, per_prompt)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        type=Path,
        default=Path("data/processed/workload_mixed_50p50d.jsonl"),
    )
    parser.add_argument(
        "--stage-profile",
        type=Path,
        default=Path("outputs/metrics/clean_stage_profile.csv"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/simulation.yaml"))
    parser.add_argument("--slo", type=Path, default=Path("configs/slo.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics/pd_ratio_sim_results.csv"),
    )
    parser.add_argument(
        "--request-output",
        type=Path,
        default=Path("outputs/metrics/pd_ratio_sim_requests.csv"),
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def load_workload(path: Path) -> list[Request]:
    if not path.is_file():
        raise FileNotFoundError(f"Workload does not exist: {path}")
    requests: list[Request] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            required = {
                "request_id",
                "workload_name",
                "prompt_len",
                "output_len",
                "arrival_time",
            }
            missing = sorted(required - row.keys())
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {missing}")
            request_id = str(row["request_id"])
            if request_id in seen:
                raise ValueError(f"Duplicate request_id: {request_id}")
            seen.add(request_id)
            prompt_len = row["prompt_len"]
            output_len = row["output_len"]
            arrival_time = row["arrival_time"]
            if (
                not isinstance(prompt_len, int)
                or isinstance(prompt_len, bool)
                or prompt_len <= 0
                or not isinstance(output_len, int)
                or isinstance(output_len, bool)
                or output_len <= 0
            ):
                raise ValueError(f"Invalid token lengths at {path}:{line_number}")
            if not isinstance(arrival_time, (int, float)) or not math.isfinite(arrival_time):
                raise ValueError(f"Invalid arrival_time at {path}:{line_number}")
            requests.append(
                Request(
                    request_id=request_id,
                    workload_name=str(row["workload_name"]),
                    prompt_len=prompt_len,
                    output_len=output_len,
                    original_arrival_time_s=float(arrival_time),
                    trace_order=len(requests),
                )
            )
    if not requests:
        raise ValueError("Workload is empty")
    if len({request.workload_name for request in requests}) != 1:
        raise ValueError("One workload file must contain exactly one workload_name")
    requests.sort(key=lambda request: (request.original_arrival_time_s, request.trace_order))
    if any(
        right.original_arrival_time_s <= left.original_arrival_time_s
        for left, right in zip(requests, requests[1:])
    ):
        raise ValueError("Workload arrival_time values must be strictly increasing")
    return requests


def load_scenarios(config: dict[str, Any]) -> tuple[list[tuple[int, int]], list[float], list[float], int]:
    required = {
        "total_gpu_units_list",
        "arrival_rates",
        "kv_transfer_ms_list",
        "seed",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Simulation config is missing: {', '.join(missing)}")
    ratios: list[tuple[int, int]] = []
    for total_gpus in config["total_gpu_units_list"]:
        key = f"ratios_g{int(total_gpus)}"
        if key not in config:
            raise ValueError(f"Simulation config is missing {key}")
        for ratio in config[key]:
            if not isinstance(ratio, list) or len(ratio) != 2:
                raise ValueError(f"Invalid ratio in {key}: {ratio}")
            p, d = int(ratio[0]), int(ratio[1])
            if p <= 0 or d <= 0 or p + d != int(total_gpus):
                raise ValueError(f"Ratio {ratio} does not match G={total_gpus}")
            ratios.append((p, d))
    if len(set(ratios)) != len(ratios):
        raise ValueError("Simulation config contains duplicate P:D ratios")
    arrival_rates = [float(value) for value in config["arrival_rates"]]
    kv_values = [float(value) for value in config["kv_transfer_ms_list"]]
    if not arrival_rates or any(not math.isfinite(value) or value <= 0 for value in arrival_rates):
        raise ValueError("arrival_rates must be positive finite values")
    if not kv_values or any(not math.isfinite(value) or value < 0 for value in kv_values):
        raise ValueError("kv_transfer_ms_list must contain finite non-negative values")
    return ratios, arrival_rates, kv_values, int(config["seed"])


def rescale_arrivals(requests: list[Request], target_rate: float) -> list[float]:
    """Preserve trace interarrival shape while setting its empirical mean rate."""

    if len(requests) == 1:
        return [0.0]
    original = np.asarray(
        [request.original_arrival_time_s for request in requests], dtype=float
    )
    shifted = original - original[0]
    duration = float(shifted[-1])
    if duration <= 0:
        raise ValueError("Original trace duration must be positive")
    target_duration = (len(requests) - 1) / target_rate
    scaled = shifted * (target_duration / duration)
    return scaled.tolist()


def simulate_scenario(
    requests: list[Request],
    arrivals_s: list[float],
    profile: StageProfile,
    ratio_p: int,
    ratio_d: int,
    arrival_rate: float,
    kv_transfer_ms: float,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prefill_pool = ServerPool(ratio_p)
    simulated: list[SimulatedRequest] = []
    for request, arrival_s in zip(requests, arrivals_s, strict=True):
        item = SimulatedRequest(
            request=request,
            arrival_time_s=arrival_s,
            prefill_service_ms=profile.prefill_service_ms(request.prompt_len),
            decode_service_ms=profile.decode_service_ms(
                request.prompt_len, request.output_len
            ),
        )
        item.prefill_start_s, item.prefill_end_s = prefill_pool.schedule(
            arrival_s, item.prefill_service_ms
        )
        item.prefill_queue_wait_ms = (item.prefill_start_s - arrival_s) * 1000.0
        item.decode_ready_time_s = item.prefill_end_s + kv_transfer_ms / 1000.0
        simulated.append(item)

    # Prefill completions can be out of arrival order. Decode FCFS is therefore
    # determined by decode_ready_time, not by the original request ordering.
    decode_pool = ServerPool(ratio_d)
    for item in sorted(
        simulated,
        key=lambda value: (
            value.decode_ready_time_s,
            value.request.trace_order,
        ),
    ):
        item.decode_start_s, item.decode_end_s = decode_pool.schedule(
            item.decode_ready_time_s, item.decode_service_ms
        )
        item.decode_queue_wait_ms = (
            item.decode_start_s - item.decode_ready_time_s
        ) * 1000.0

    scenario_rows: list[dict[str, Any]] = []
    ttft_values: list[float] = []
    tpot_values: list[float] = []
    latency_values: list[float] = []
    good_requests = 0
    for item in sorted(simulated, key=lambda value: value.request.trace_order):
        ttft_ms = (item.decode_start_s - item.arrival_time_s) * 1000.0
        total_latency_ms = (item.decode_end_s - item.arrival_time_s) * 1000.0
        tpot_ms = (
            item.decode_service_ms / (item.request.output_len - 1)
            if item.request.output_len > 1
            else float("nan")
        )
        success = (
            math.isfinite(ttft_ms)
            and math.isfinite(total_latency_ms)
            and math.isfinite(tpot_ms)
            and ttft_ms >= 0
            and total_latency_ms >= ttft_ms
            and tpot_ms >= 0
        )
        if success:
            ttft_values.append(ttft_ms)
            tpot_values.append(tpot_ms)
            latency_values.append(total_latency_ms)
            if ttft_ms <= ttft_slo_ms and tpot_ms <= tpot_slo_ms:
                good_requests += 1
        scenario_rows.append(
            {
                "workload_name": item.request.workload_name,
                "request_id": item.request.request_id,
                "ratio_p": ratio_p,
                "ratio_d": ratio_d,
                "arrival_rate": arrival_rate,
                "kv_transfer_ms": kv_transfer_ms,
                "arrival_time": item.arrival_time_s,
                "prefill_start": item.prefill_start_s,
                "prefill_end": item.prefill_end_s,
                "decode_ready_time": item.decode_ready_time_s,
                "decode_start": item.decode_start_s,
                "decode_end": item.decode_end_s,
                "prefill_queue_wait_ms": item.prefill_queue_wait_ms,
                "decode_queue_wait_ms": item.decode_queue_wait_ms,
                "ttft_ms": ttft_ms,
                "tpot_ms": tpot_ms,
                "total_latency_ms": total_latency_ms,
                "success": success,
                "data_source": "simulated",
            }
        )

    if len(ttft_values) != len(requests):
        raise RuntimeError("Scenario produced invalid request metrics")
    first_arrival_s = min(item.arrival_time_s for item in simulated)
    last_completion_s = max(item.decode_end_s for item in simulated)
    makespan_s = last_completion_s - first_arrival_s
    if makespan_s <= 0:
        raise RuntimeError("Scenario makespan must be positive")
    makespan_ms = makespan_s * 1000.0
    total_output_tokens = sum(item.request.output_len for item in simulated)
    workload_name = requests[0].workload_name
    aggregated = {
        "workload_name": workload_name,
        "ratio_p": ratio_p,
        "ratio_d": ratio_d,
        "arrival_rate": arrival_rate,
        "kv_transfer_ms": kv_transfer_ms,
        "num_requests": len(requests),
        "avg_ttft_ms": float(np.mean(ttft_values)),
        "p95_ttft_ms": float(np.percentile(ttft_values, 95)),
        "p99_ttft_ms": float(np.percentile(ttft_values, 99)),
        "avg_tpot_ms": float(np.mean(tpot_values)),
        "p95_tpot_ms": float(np.percentile(tpot_values, 95)),
        "p99_tpot_ms": float(np.percentile(tpot_values, 99)),
        "avg_total_latency_ms": float(np.mean(latency_values)),
        "p95_total_latency_ms": float(np.percentile(latency_values, 95)),
        "throughput_req_s": len(requests) / makespan_s,
        "throughput_tok_s": total_output_tokens / makespan_s,
        "goodput_req_s": good_requests / makespan_s,
        "prefill_util": prefill_pool.busy_time_ms / (ratio_p * makespan_ms),
        "decode_util": decode_pool.busy_time_ms / (ratio_d * makespan_ms),
        "avg_prefill_queue_wait_ms": float(
            np.mean([item.prefill_queue_wait_ms for item in simulated])
        ),
        "avg_decode_queue_wait_ms": float(
            np.mean([item.decode_queue_wait_ms for item in simulated])
        ),
        "data_source": "simulated",
    }
    for utilization in (aggregated["prefill_util"], aggregated["decode_util"]):
        if utilization < -1e-12 or utilization > 1.0 + 1e-9:
            raise RuntimeError(f"Invalid utilization computed: {utilization}")
    return scenario_rows, aggregated


def atomic_write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def run(args: argparse.Namespace) -> None:
    simulation_config = load_yaml(args.config)
    slo_config = load_yaml(args.slo)
    ratios, arrival_rates, kv_values, seed = load_scenarios(simulation_config)
    # The current model is deterministic. Seed is still validated and logged so
    # stochastic service extensions can preserve the same scenario contract.
    np.random.default_rng(seed)
    for key in ("ttft_slo_ms", "tpot_slo_ms"):
        if key not in slo_config:
            raise ValueError(f"SLO config is missing {key}")
    ttft_slo_ms = float(slo_config["ttft_slo_ms"])
    tpot_slo_ms = float(slo_config["tpot_slo_ms"])
    if ttft_slo_ms <= 0 or tpot_slo_ms <= 0:
        raise ValueError("SLO thresholds must be positive")

    requests = load_workload(args.workload)
    profile = StageProfile(args.stage_profile)
    profile.validate_range(requests)
    request_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    total_scenarios = len(ratios) * len(arrival_rates) * len(kv_values)
    completed = 0
    arrivals_by_rate = {
        rate: rescale_arrivals(requests, rate) for rate in arrival_rates
    }
    for ratio_p, ratio_d in ratios:
        for arrival_rate in arrival_rates:
            for kv_transfer_ms in kv_values:
                scenario_requests, scenario_result = simulate_scenario(
                    requests=requests,
                    arrivals_s=arrivals_by_rate[arrival_rate],
                    profile=profile,
                    ratio_p=ratio_p,
                    ratio_d=ratio_d,
                    arrival_rate=arrival_rate,
                    kv_transfer_ms=kv_transfer_ms,
                    ttft_slo_ms=ttft_slo_ms,
                    tpot_slo_ms=tpot_slo_ms,
                )
                request_rows.extend(scenario_requests)
                result_rows.append(scenario_result)
                completed += 1
                LOGGER.info(
                    "Scenario %d/%d | P:D=%d:%d rate=%.2f kv=%.1fms",
                    completed,
                    total_scenarios,
                    ratio_p,
                    ratio_d,
                    arrival_rate,
                    kv_transfer_ms,
                )
    atomic_write_csv(args.request_output, REQUEST_FIELDS, request_rows)
    atomic_write_csv(args.output, RESULT_FIELDS, result_rows)
    LOGGER.info(
        "Simulation complete: scenarios=%d request_rows=%d results=%s requests=%s seed=%d",
        len(result_rows),
        len(request_rows),
        args.output,
        args.request_output,
        seed,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        LOGGER.exception("PD ratio simulation failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
