#!/usr/bin/env python3
"""Aggregate real four-GPU request and telemetry CSV files."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


SUMMARY_FIELDS = [
    "run_id", "workload_name", "workload_mode", "serving_mode", "result_type",
    "arrival_rate", "concurrency", "num_success", "num_failed", "avg_ttft_ms",
    "p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms", "avg_tpot_ms", "p50_tpot_ms",
    "p95_tpot_ms", "p99_tpot_ms", "avg_total_latency_ms", "p95_total_latency_ms",
    "throughput_req_s", "throughput_tok_s", "goodput_req_s", "avg_gpu_util",
    "avg_memory_used_mb", "max_memory_used_mb", "gpu_util_imbalance",
    "prefix_caching_enabled", "data_source", "request_file", "gpu_trace_file",
]
ALLOWED_RESULT_TYPES = {
    "real_colocated", "real_aggregated_tp", "real_disaggregated_pd",
    "emulated_pd", "simulated_pd",
    "measured_attempt_failed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-dir", type=Path, default=Path("outputs/metrics/real4gpu")
    )
    parser.add_argument(
        "--output-summary", type=Path,
        default=Path("outputs/metrics/real4gpu/real4gpu_summary.csv"),
    )
    parser.add_argument(
        "--output-requests", type=Path,
        default=Path("outputs/metrics/real4gpu/real4gpu_request_metrics.csv"),
    )
    parser.add_argument(
        "--table-output", type=Path,
        default=Path("outputs/tables/table_real4gpu_summary.md"),
    )
    parser.add_argument("--findings", type=Path, default=Path("report/findings.md"))
    parser.add_argument("--slo", type=Path, default=Path("configs/slo.yaml"))
    return parser.parse_args()


def boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def percentile(series: pd.Series, value: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.quantile(value)) if not clean.empty else math.nan


def telemetry_for_run(
    traces: list[tuple[Path, pd.DataFrame]], run_id: str, mode: str,
    workload: str, arrival_rate: float, concurrency: int,
) -> tuple[pd.DataFrame, str]:
    exact = []
    compatible = []
    for path, frame in traces:
        if "run_id" in frame and run_id and (frame["run_id"].astype(str) == run_id).any():
            exact.append((path, frame[frame["run_id"].astype(str) == run_id]))
            continue
        checks = []
        if "serving_mode" in frame and frame["serving_mode"].astype(str).str.len().gt(0).any():
            checks.append((frame["serving_mode"].astype(str) == mode).any())
        if "workload_name" in frame and frame["workload_name"].astype(str).str.len().gt(0).any():
            checks.append((frame["workload_name"].astype(str) == workload).any())
        if "concurrency" in frame and pd.to_numeric(frame["concurrency"], errors="coerce").gt(0).any():
            checks.append((pd.to_numeric(frame["concurrency"], errors="coerce") == concurrency).any())
        if "arrival_rate" in frame and pd.to_numeric(frame["arrival_rate"], errors="coerce").notna().any():
            values = pd.to_numeric(frame["arrival_rate"], errors="coerce")
            checks.append(np.isclose(values, arrival_rate, equal_nan=False).any())
        if checks and all(checks):
            compatible.append((path, frame))
    selected = exact or (compatible if len(compatible) == 1 else [])
    if not selected:
        return pd.DataFrame(), ""
    return pd.concat([frame for _, frame in selected], ignore_index=True), ";".join(str(path) for path, _ in selected)


def gpu_stats(frame: pd.DataFrame) -> dict[str, float]:
    empty = {
        "avg_gpu_util": math.nan, "avg_memory_used_mb": math.nan,
        "max_memory_used_mb": math.nan, "gpu_util_imbalance": math.nan,
    }
    required = {"gpu_id", "gpu_util", "memory_used_mb"}
    if frame.empty or not required.issubset(frame.columns):
        return empty
    util = pd.to_numeric(frame["gpu_util"], errors="coerce")
    memory = pd.to_numeric(frame["memory_used_mb"], errors="coerce")
    per_gpu = frame.assign(_util=util).groupby("gpu_id")["_util"].mean().dropna()
    return {
        "avg_gpu_util": float(util.mean()),
        "avg_memory_used_mb": float(memory.mean()),
        "max_memory_used_mb": float(memory.max()),
        "gpu_util_imbalance": float(per_gpu.max() - per_gpu.min()) if not per_gpu.empty else math.nan,
    }


def summarize_file(
    path: Path, frame: pd.DataFrame, traces: list[tuple[Path, pd.DataFrame]],
    ttft_slo: float, tpot_slo: float,
) -> dict[str, Any]:
    required = {
        "request_id", "workload_name", "serving_mode", "result_type", "success",
        "send_time", "total_latency_ms", "ttft_ms", "tpot_ms", "output_len_actual",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    result_types = set(frame["result_type"].dropna().astype(str))
    if not result_types or not result_types.issubset(ALLOWED_RESULT_TYPES):
        raise ValueError(f"{path}: invalid result_type values {result_types}")
    success_mask = boolean_series(frame["success"])
    successful = frame[success_mask].copy()
    first = frame.iloc[0]
    run_id = str(first.get("run_id", path.stem))
    workload = str(first["workload_name"])
    mode = str(first["serving_mode"])
    arrival_rate = float(pd.to_numeric(pd.Series([first.get("arrival_rate")]), errors="coerce").iloc[0])
    concurrency = int(float(first.get("concurrency", 0)))
    telemetry, trace_paths = telemetry_for_run(
        traces, run_id, mode, workload, arrival_rate, concurrency
    )
    send = pd.to_numeric(successful["send_time"], errors="coerce")
    latency = pd.to_numeric(successful["total_latency_ms"], errors="coerce")
    if successful.empty or send.dropna().empty or latency.dropna().empty:
        wall_time = math.nan
    else:
        wall_time = float((send + latency / 1000).max() - send.min())
    output_tokens = pd.to_numeric(successful["output_len_actual"], errors="coerce").sum(min_count=1)
    ttft = pd.to_numeric(successful["ttft_ms"], errors="coerce")
    tpot = pd.to_numeric(successful["tpot_ms"], errors="coerce")
    latency = pd.to_numeric(successful["total_latency_ms"], errors="coerce")
    good = successful[(ttft <= ttft_slo) & (tpot <= tpot_slo)]
    summary = {
        "run_id": run_id, "workload_name": workload,
        "workload_mode": str(first.get("workload_mode", "unknown")),
        "serving_mode": mode, "result_type": str(first["result_type"]),
        "arrival_rate": arrival_rate, "concurrency": concurrency,
        "num_success": int(success_mask.sum()), "num_failed": int((~success_mask).sum()),
        "avg_ttft_ms": float(ttft.mean()), "p50_ttft_ms": percentile(ttft, 0.50),
        "p95_ttft_ms": percentile(ttft, 0.95), "p99_ttft_ms": percentile(ttft, 0.99),
        "avg_tpot_ms": float(tpot.mean()), "p50_tpot_ms": percentile(tpot, 0.50),
        "p95_tpot_ms": percentile(tpot, 0.95), "p99_tpot_ms": percentile(tpot, 0.99),
        "avg_total_latency_ms": float(latency.mean()),
        "p95_total_latency_ms": percentile(latency, 0.95),
        "throughput_req_s": len(successful) / wall_time if wall_time and wall_time > 0 else math.nan,
        "throughput_tok_s": float(output_tokens) / wall_time if wall_time and wall_time > 0 else math.nan,
        "goodput_req_s": len(good) / wall_time if wall_time and wall_time > 0 else math.nan,
        "prefix_caching_enabled": bool(boolean_series(frame["prefix_caching_enabled"]).iloc[0]) if "prefix_caching_enabled" in frame else False,
        "data_source": "measured" if str(first["result_type"]).startswith("real_") else str(first["result_type"]),
        "request_file": str(path), "gpu_trace_file": trace_paths,
    }
    summary.update(gpu_stats(telemetry))
    return summary


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [
        "workload_name", "serving_mode", "result_type", "arrival_rate", "concurrency",
        "num_success", "num_failed", "p95_ttft_ms", "p95_tpot_ms",
        "throughput_req_s", "goodput_req_s", "avg_gpu_util", "gpu_util_imbalance",
    ]
    if frame.empty:
        return "# Real 4GPU summary\n\nInsufficient data: no `online_*.csv` measured runs were found.\n"
    shown = frame[columns].copy()
    for column in shown.select_dtypes(include=["number"]).columns:
        shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None)]
    return "# Real 4GPU summary\n\n" + "\n".join([header, separator, *rows]) + "\n"


def update_findings(path: Path, summary: pd.DataFrame) -> None:
    begin = "<!-- BEGIN AUTO REAL4GPU -->"
    end = "<!-- END AUTO REAL4GPU -->"
    if summary.empty:
        body = (
            "## Real 4GPU online-serving validation\n\n"
            "Insufficient data: no successful `online_*.csv` run has been analyzed. "
            "No claim about colocated, TP4, or real PD performance is made."
        )
    else:
        real = summary[summary["result_type"].astype(str).str.startswith("real_")]
        lines = [
            "## Real 4GPU online-serving validation", "",
            f"Analyzed {len(real)} measured run(s). Metrics are grouped by run and are not mixed with simulated PD results.", "",
        ]
        for mode, group in real.groupby("serving_mode"):
            lines.append(
                f"- `{mode}`: {int(group['num_success'].sum())} successful and "
                f"{int(group['num_failed'].sum())} failed requests across {len(group)} run(s)."
            )
        pd_real = real[real["result_type"] == "real_disaggregated_pd"]
        if pd_real.empty:
            lines.extend([
                "- Real PD: insufficient data. Ratio conclusions remain trace-driven simulated and must not be reported as measured.",
                "- Fallback: calibrate the simulator with real colocated request/throughput measurements and retain `simulated_pd` labels.",
            ])
        body = "\n".join(lines)
    block = f"{begin}\n{body}\n{end}"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Findings\n"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    updated = pattern.sub(block, existing) if pattern.search(existing) else existing.rstrip() + "\n\n" + block + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated, encoding="utf-8")


def main() -> None:
    args = parse_args()
    slo = yaml.safe_load(args.slo.read_text(encoding="utf-8"))
    ttft_slo = float(slo["ttft_slo_ms"])
    tpot_slo = float(slo["tpot_slo_ms"])
    request_files = sorted(args.metrics_dir.glob("online_*.csv"))
    trace_files = sorted(args.metrics_dir.glob("gpu_trace_*.csv"))
    traces = []
    for path in trace_files:
        try:
            traces.append((path, pd.read_csv(path)))
        except Exception as exc:
            print(f"WARNING: ignoring unreadable GPU trace {path}: {exc}")
    frames = []
    summaries = []
    for path in request_files:
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
        summaries.append(summarize_file(path, frame, traces, ttft_slo, tpot_slo))
    requests = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = pd.DataFrame(summaries, columns=SUMMARY_FIELDS)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_requests.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary, index=False)
    requests.to_csv(args.output_requests, index=False)
    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    args.table_output.write_text(markdown_table(summary), encoding="utf-8")
    update_findings(args.findings, summary)
    print(f"Wrote {len(summary)} run summaries to {args.output_summary}")
    print(f"Wrote {len(requests)} request rows to {args.output_requests}")


if __name__ == "__main__":
    main()
