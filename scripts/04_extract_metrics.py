#!/usr/bin/env python3
"""Normalize benchmark, GPU, and stage-profile results into clean CSV metrics."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("extract_metrics")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_BENCHMARK_FIELDS = [
    "request_id",
    "workload_name",
    "request_type",
    "prompt_len_actual",
    "output_len_actual",
    "ttft_ms",
    "tpot_ms",
    "total_latency_ms",
    "tokens_per_second",
    "success",
    "backend",
    "ttft_method",
]

WORKLOAD_FIELDS = [
    "workload_name",
    "num_success",
    "num_failed",
    "avg_prompt_len",
    "avg_output_len",
    "avg_ttft_ms",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "avg_tpot_ms",
    "p50_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "avg_total_latency_ms",
    "p95_total_latency_ms",
    "throughput_req_s",
    "throughput_tok_s",
    "avg_gpu_util",
    "peak_memory_mb",
]

STAGE_VALUE_FIELDS = [
    "ttft_ms",
    "total_latency_ms",
    "decode_total_ms",
    "tpot_ms",
    "tokens_per_second",
]

CLEAN_STAGE_FIELDS = [
    "prompt_len",
    "output_len",
    "num_repeats",
    "ttft_ms_mean",
    "ttft_ms_std",
    "total_latency_ms_mean",
    "total_latency_ms_std",
    "decode_total_ms_mean",
    "decode_total_ms_std",
    "tpot_ms_mean",
    "tpot_ms_std",
    "tokens_per_second_mean",
    "tokens_per_second_std",
    "backend",
    "ttft_method",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-dir", type=Path, default=Path("outputs/metrics")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/metrics")
    )
    parser.add_argument(
        "--workload-mode",
        default="synthetic_unique",
        help="Select one workload mode to prevent cross-mode aggregation.",
    )
    parser.add_argument(
        "--success-rate-threshold",
        type=float,
        default=0.95,
        help="Warn below this per-workload request success rate.",
    )
    return parser.parse_args()


def parse_bool_series(series: pd.Series, field: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    valid = normalized.isin({"true", "false", "1", "0"})
    if not valid.all():
        bad = sorted(normalized[~valid].unique().tolist())
        raise ValueError(f"Invalid boolean values in {field}: {bad}")
    return normalized.isin({"true", "1"})


def require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {', '.join(missing)}")


def infer_mode(path: Path, frame: pd.DataFrame) -> pd.Series:
    if "workload_mode" in frame.columns:
        return frame["workload_mode"].fillna("legacy").astype(str)
    for mode in ("synthetic_unique", "synthetic_cache_friendly", "real_or_sharegpt"):
        if mode in path.stem:
            return pd.Series(mode, index=frame.index, dtype="object")
    return pd.Series("legacy", index=frame.index, dtype="object")


def load_benchmark_requests(
    metrics_dir: Path, workload_mode: str, warnings: list[str]
) -> pd.DataFrame:
    required = set(CLEAN_BENCHMARK_FIELDS)
    candidates: list[pd.DataFrame] = []
    files = sorted(metrics_dir.glob("benchmark_*.csv"))
    for path in files:
        if path.name == "benchmark_summary.csv":
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            raise ValueError(f"Could not read {path}: {exc}") from exc
        if "request_id" not in frame.columns:
            warnings.append(f"Skipped non-request benchmark CSV: {path.name}")
            continue
        require_columns(frame, required, path)
        frame = frame.copy()
        frame["_workload_mode"] = infer_mode(path, frame)
        frame = frame[frame["_workload_mode"] == workload_mode]
        if frame.empty:
            continue
        frame["_source_file"] = path.name
        frame["_source_mtime"] = path.stat().st_mtime_ns
        frame["_schema_score"] = sum(
            column in frame.columns
            for column in (
                "workload_mode",
                "cache_group_id",
                "intended_prefix_reuse",
                "num_cached_tokens",
            )
        )
        candidates.append(frame)
    if not candidates:
        raise FileNotFoundError(
            f"No benchmark request CSVs found for workload_mode={workload_mode}"
        )

    combined = pd.concat(candidates, ignore_index=True, sort=False)
    combined["success"] = parse_bool_series(combined["success"], "success")
    duplicate_key = ["request_id", "backend"]
    duplicate_count = int(combined.duplicated(duplicate_key, keep=False).sum())
    if duplicate_count:
        duplicate_groups = int(
            combined.loc[
                combined.duplicated(duplicate_key, keep=False), duplicate_key
            ].drop_duplicates().shape[0]
        )
        warnings.append(
            f"Found {duplicate_count} rows in {duplicate_groups} duplicate request keys; "
            "kept the richest/newest source file for each key."
        )
    combined = combined.sort_values(
        ["_schema_score", "_source_mtime", "_source_file"], kind="stable"
    ).drop_duplicates(duplicate_key, keep="last")
    combined = combined.sort_values(
        ["workload_name", "request_id"], kind="stable"
    ).reset_index(drop=True)
    LOGGER.info(
        "Loaded %d unique benchmark requests from %d candidate files",
        len(combined),
        len(candidates),
    )
    return combined


def check_benchmark_data(
    frame: pd.DataFrame, success_rate_threshold: float, warnings: list[str]
) -> None:
    numeric_fields = [
        "prompt_len_actual",
        "output_len_actual",
        "ttft_ms",
        "tpot_ms",
        "total_latency_ms",
        "tokens_per_second",
    ]
    for field in numeric_fields:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")

    successful = frame[frame["success"]]
    too_short = successful[successful["output_len_actual"] <= 1]
    if not too_short.empty:
        warnings.append(
            f"{len(too_short)} successful benchmark requests have output_len_actual <= 1."
        )
    negative_tpot = successful[successful["tpot_ms"] < 0]
    if not negative_tpot.empty:
        warnings.append(f"{len(negative_tpot)} requests have negative tpot_ms.")
    invalid_order = successful[
        successful["total_latency_ms"] < successful["ttft_ms"]
    ]
    if not invalid_order.empty:
        warnings.append(
            f"{len(invalid_order)} requests have total_latency_ms < ttft_ms."
        )
    missing_metrics = successful[
        ["ttft_ms", "tpot_ms", "total_latency_ms", "tokens_per_second"]
    ].isna().any(axis=1)
    if bool(missing_metrics.any()):
        warnings.append(
            f"{int(missing_metrics.sum())} successful requests have missing core metrics."
        )

    for workload_name, group in frame.groupby("workload_name", sort=True):
        rate = float(group["success"].mean())
        if rate < success_rate_threshold:
            warnings.append(
                f"{workload_name}: success rate {rate:.3f} is below "
                f"threshold {success_rate_threshold:.3f}."
            )
        valid = group[group["success"]]
        for field in ("ttft_ms", "tpot_ms", "total_latency_ms"):
            values = valid[field].dropna()
            if len(values) < 5:
                continue
            median = float(values.median())
            mad = float((values - median).abs().median())
            if mad <= 0:
                continue
            extreme = (values - median).abs() > 10.0 * mad
            count = int(extreme.sum())
            if count:
                warnings.append(
                    f"{workload_name}: {count}/{len(values)} possible {field} "
                    "outliers exceed 10 median absolute deviations; retained."
                )


def load_benchmark_summary(
    metrics_dir: Path, workload_mode: str, warnings: list[str]
) -> pd.DataFrame:
    path = metrics_dir / "benchmark_summary.csv"
    if not path.is_file():
        warnings.append("benchmark_summary.csv is missing; throughput fields will be NaN.")
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = {
        "workload_name",
        "requests_per_second",
        "tokens_per_second",
    }
    require_columns(frame, required, path)
    if "workload_mode" in frame.columns:
        frame = frame[frame["workload_mode"].astype(str) == workload_mode]
    else:
        frame = frame[
            frame.get("workload_file", pd.Series("", index=frame.index))
            .astype(str)
            .str.contains(workload_mode, regex=False)
        ]
    if frame.empty:
        warnings.append(
            f"benchmark_summary.csv has no rows for workload_mode={workload_mode}."
        )
        return frame
    if "timestamp_utc" in frame.columns:
        frame = frame.sort_values("timestamp_utc", kind="stable")
    duplicates = int(frame.duplicated("workload_name", keep=False).sum())
    if duplicates:
        warnings.append(
            f"benchmark_summary.csv has {duplicates} duplicate workload rows; kept latest."
        )
    return frame.drop_duplicates("workload_name", keep="last").set_index(
        "workload_name"
    )


def trace_workload_name(path: Path, workload_mode: str) -> str | None:
    stem = path.stem.removeprefix("gpu_trace_")
    suffix = f"_{workload_mode}"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return None


def load_gpu_metrics(
    metrics_dir: Path, workload_mode: str, warnings: list[str]
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for path in sorted(metrics_dir.glob("gpu_trace_*.csv")):
        workload_name = trace_workload_name(path, workload_mode)
        if workload_name is None:
            continue
        frame = pd.read_csv(path)
        require_columns(frame, {"gpu_util", "memory_used_mb"}, path)
        util = pd.to_numeric(frame["gpu_util"], errors="coerce").dropna()
        memory = pd.to_numeric(frame["memory_used_mb"], errors="coerce").dropna()
        if util.empty or memory.empty:
            warnings.append(f"{path.name}: GPU trace is empty or non-numeric.")
            results[workload_name] = {
                "avg_gpu_util": float("nan"),
                "peak_memory_mb": float("nan"),
            }
        else:
            results[workload_name] = {
                "avg_gpu_util": float(util.mean()),
                "peak_memory_mb": float(memory.max()),
            }
    return results


def percentile(series: pd.Series, quantile: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(quantile)) if not values.empty else float("nan")


def mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def aggregate_workloads(
    benchmark: pd.DataFrame,
    summary: pd.DataFrame,
    gpu: dict[str, dict[str, float]],
    warnings: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for workload_name, group in benchmark.groupby("workload_name", sort=True):
        successful = group[group["success"]]
        throughput_req_s = float("nan")
        throughput_tok_s = float("nan")
        if not summary.empty and workload_name in summary.index:
            summary_row = summary.loc[workload_name]
            throughput_req_s = float(summary_row["requests_per_second"])
            throughput_tok_s = float(summary_row["tokens_per_second"])
        else:
            warnings.append(
                f"{workload_name}: no matching benchmark summary; throughput is NaN."
            )
        gpu_values = gpu.get(
            workload_name,
            {"avg_gpu_util": float("nan"), "peak_memory_mb": float("nan")},
        )
        if workload_name not in gpu:
            warnings.append(
                f"{workload_name}: no matching GPU trace; GPU metrics are NaN."
            )
        rows.append(
            {
                "workload_name": workload_name,
                "num_success": int(group["success"].sum()),
                "num_failed": int((~group["success"]).sum()),
                "avg_prompt_len": mean(successful["prompt_len_actual"]),
                "avg_output_len": mean(successful["output_len_actual"]),
                "avg_ttft_ms": mean(successful["ttft_ms"]),
                "p50_ttft_ms": percentile(successful["ttft_ms"], 0.50),
                "p95_ttft_ms": percentile(successful["ttft_ms"], 0.95),
                "p99_ttft_ms": percentile(successful["ttft_ms"], 0.99),
                "avg_tpot_ms": mean(successful["tpot_ms"]),
                "p50_tpot_ms": percentile(successful["tpot_ms"], 0.50),
                "p95_tpot_ms": percentile(successful["tpot_ms"], 0.95),
                "p99_tpot_ms": percentile(successful["tpot_ms"], 0.99),
                "avg_total_latency_ms": mean(successful["total_latency_ms"]),
                "p95_total_latency_ms": percentile(
                    successful["total_latency_ms"], 0.95
                ),
                "throughput_req_s": throughput_req_s,
                "throughput_tok_s": throughput_tok_s,
                "avg_gpu_util": gpu_values["avg_gpu_util"],
                "peak_memory_mb": gpu_values["peak_memory_mb"],
            }
        )
    return pd.DataFrame(rows, columns=WORKLOAD_FIELDS)


def clean_stage_profile(metrics_dir: Path, warnings: list[str]) -> pd.DataFrame:
    path = metrics_dir / "stage_profile.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing stage profile: {path}")
    frame = pd.read_csv(path)
    required = {
        "prompt_len",
        "output_len",
        "success",
        "backend",
        "ttft_method",
        *STAGE_VALUE_FIELDS,
    }
    require_columns(frame, required, path)
    frame["success"] = parse_bool_series(frame["success"], "stage success")
    failures = int((~frame["success"]).sum())
    if failures:
        warnings.append(
            f"stage_profile.csv contains {failures} failed rows; excluded as required."
        )
    valid = frame[frame["success"]].copy()
    if valid.empty:
        raise ValueError("stage_profile.csv has no successful rows")
    for field in ["prompt_len", "output_len", *STAGE_VALUE_FIELDS]:
        valid[field] = pd.to_numeric(valid[field], errors="coerce")
    missing = valid[["ttft_ms", "total_latency_ms", "decode_total_ms"]].isna().any(
        axis=1
    )
    if bool(missing.any()):
        warnings.append(
            f"Excluded {int(missing.sum())} successful stage rows with missing required values."
        )
        valid = valid[~missing]

    rows: list[dict[str, Any]] = []
    for (prompt_len, output_len), group in valid.groupby(
        ["prompt_len", "output_len"], sort=True
    ):
        methods = sorted(group["ttft_method"].astype(str).unique().tolist())
        backends = sorted(group["backend"].astype(str).unique().tolist())
        if len(methods) != 1 or len(backends) != 1:
            warnings.append(
                f"Stage ({prompt_len}, {output_len}) mixes backend or TTFT methods."
            )
        row: dict[str, Any] = {
            "prompt_len": int(prompt_len),
            "output_len": int(output_len),
            "num_repeats": len(group),
        }
        for field in STAGE_VALUE_FIELDS:
            values = group[field].dropna()
            row[f"{field}_mean"] = (
                float(values.mean()) if not values.empty else float("nan")
            )
            row[f"{field}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            )
        row["backend"] = backends[0] if len(backends) == 1 else "mixed"
        row["ttft_method"] = methods[0] if len(methods) == 1 else "mixed"
        rows.append(row)
    return pd.DataFrame(rows, columns=CLEAN_STAGE_FIELDS)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_warnings(path: Path, warnings: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = warnings or ["No warnings detected."]
    content = "\n".join(f"WARNING: {line}" for line in lines) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def run(args: argparse.Namespace) -> None:
    if not args.metrics_dir.is_dir():
        raise FileNotFoundError(f"Metrics directory does not exist: {args.metrics_dir}")
    if not 0 <= args.success_rate_threshold <= 1:
        raise ValueError("--success-rate-threshold must be in [0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    benchmark = load_benchmark_requests(
        args.metrics_dir, args.workload_mode, warnings
    )
    check_benchmark_data(benchmark, args.success_rate_threshold, warnings)
    summary = load_benchmark_summary(args.metrics_dir, args.workload_mode, warnings)
    gpu = load_gpu_metrics(args.metrics_dir, args.workload_mode, warnings)
    workload_metrics = aggregate_workloads(benchmark, summary, gpu, warnings)
    clean_stage = clean_stage_profile(args.metrics_dir, warnings)

    clean_benchmark = benchmark[CLEAN_BENCHMARK_FIELDS].copy()
    atomic_write_csv(
        args.output_dir / "clean_benchmark_metrics.csv", clean_benchmark
    )
    atomic_write_csv(
        args.output_dir / "workload_level_metrics.csv", workload_metrics
    )
    atomic_write_csv(args.output_dir / "clean_stage_profile.csv", clean_stage)
    warning_path = PROJECT_ROOT / "outputs" / "logs" / "metrics_warnings.txt"
    write_warnings(warning_path, warnings)

    for warning in warnings:
        LOGGER.warning(warning)
    LOGGER.info(
        "Wrote clean metrics: requests=%d workloads=%d stage_points=%d warnings=%d",
        len(clean_benchmark),
        len(workload_metrics),
        len(clean_stage),
        len(warnings),
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
        LOGGER.exception("Metric extraction failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
