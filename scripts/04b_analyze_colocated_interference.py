#!/usr/bin/env python3
"""Analyze decode interference from colocated prefill-heavy requests."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("analyze_colocated_interference")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_WORKLOADS = ("decode_heavy_256", "decode_heavy_512")
MIXED_WORKLOADS = ("mixed_30p70d", "mixed_50p50d", "mixed_70p30d")
OUTPUT_FIELDS = [
    "workload_name",
    "prefill_heavy_ratio",
    "decode_heavy_ratio",
    "avg_ttft_ms",
    "p95_ttft_ms",
    "avg_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "avg_total_latency_ms",
    "p95_total_latency_ms",
    "relative_p95_tpot_vs_decode_only",
    "relative_p95_ttft_vs_decode_only",
]
BEGIN_MARKER = "<!-- BEGIN AUTO COLOCATED INTERFERENCE -->"
END_MARKER = "<!-- END AUTO COLOCATED INTERFERENCE -->"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/metrics/clean_benchmark_metrics.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics/colocated_interference.csv"),
    )
    parser.add_argument(
        "--findings",
        type=Path,
        default=PROJECT_ROOT / "report" / "findings.md",
    )
    parser.add_argument(
        "--min-decode-samples",
        type=int,
        default=20,
        help="Minimum successful decode requests required per comparison.",
    )
    return parser.parse_args()


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype(str).str.lower().str.strip()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise ValueError("success contains invalid boolean values")
    return normalized.isin({"true", "1"})


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing clean benchmark metrics: {path}")
    frame = pd.read_csv(path)
    required = {
        "request_id",
        "workload_name",
        "request_type",
        "output_len_actual",
        "ttft_ms",
        "tpot_ms",
        "total_latency_ms",
        "success",
        "backend",
        "ttft_method",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing columns: {', '.join(missing)}")
    frame = frame.copy()
    frame["success"] = parse_bool(frame["success"])
    for field in (
        "output_len_actual",
        "ttft_ms",
        "tpot_ms",
        "total_latency_ms",
    ):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    return frame


def valid_decode_rows(frame: pd.DataFrame) -> pd.DataFrame:
    mask = (
        frame["success"]
        & (frame["request_type"] == "decode_heavy")
        & frame[
            ["output_len_actual", "ttft_ms", "tpot_ms", "total_latency_ms"]
        ].notna().all(axis=1)
        & (frame["output_len_actual"] > 1)
        & (frame["tpot_ms"] >= 0)
        & (frame["total_latency_ms"] >= frame["ttft_ms"])
    )
    return frame[mask].copy()


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    if len(values) == 0 or len(values) != len(weights):
        return float("nan")
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    if len(values) == 0 or float(weights.sum()) <= 0:
        return float("nan")
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cutoff = quantile * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, cutoff, side="left")), len(values) - 1)
    return float(values[index])


def matched_baseline_weights(
    baseline: pd.DataFrame, target: pd.DataFrame
) -> np.ndarray:
    """Reweight baseline output lengths to match a mixed decode subset."""

    target_proportions = target["output_len_actual"].value_counts(normalize=True)
    weights = np.zeros(len(baseline), dtype=float)
    baseline_lengths = baseline["output_len_actual"].to_numpy()
    for output_len, proportion in target_proportions.items():
        indices = np.flatnonzero(baseline_lengths == output_len)
        if len(indices) == 0:
            raise ValueError(
                f"Decode-only baseline has no output_len={output_len} samples"
            )
        weights[indices] = float(proportion) / len(indices)
    return weights


def ordinary_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "avg_ttft_ms": float(frame["ttft_ms"].mean()),
        "p95_ttft_ms": float(frame["ttft_ms"].quantile(0.95)),
        "avg_tpot_ms": float(frame["tpot_ms"].mean()),
        "p95_tpot_ms": float(frame["tpot_ms"].quantile(0.95)),
        "p99_tpot_ms": float(frame["tpot_ms"].quantile(0.99)),
        "avg_total_latency_ms": float(frame["total_latency_ms"].mean()),
        "p95_total_latency_ms": float(
            frame["total_latency_ms"].quantile(0.95)
        ),
    }


def workload_ratios(frame: pd.DataFrame, workload_name: str) -> tuple[float, float]:
    workload = frame[frame["workload_name"] == workload_name]
    if workload.empty:
        return float("nan"), float("nan")
    prefill = float((workload["request_type"] == "prefill_heavy").mean())
    decode = float((workload["request_type"] == "decode_heavy").mean())
    return prefill, decode


def build_analysis(
    frame: pd.DataFrame, min_samples: int
) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    baseline_all = frame[frame["workload_name"].isin(BASELINE_WORKLOADS)]
    baseline = valid_decode_rows(baseline_all)
    missing_baselines = [
        name
        for name in BASELINE_WORKLOADS
        if len(baseline[baseline["workload_name"] == name]) < min_samples
    ]
    if missing_baselines:
        notes.append(
            "insufficient data: decode-only baseline lacks at least "
            f"{min_samples} valid samples for {', '.join(missing_baselines)}."
        )
        return pd.DataFrame(columns=OUTPUT_FIELDS), notes

    baseline_metrics = ordinary_metrics(baseline)
    rows: list[dict[str, Any]] = [
        {
            "workload_name": "decode_only_baseline",
            "prefill_heavy_ratio": 0.0,
            "decode_heavy_ratio": 1.0,
            **baseline_metrics,
            "relative_p95_tpot_vs_decode_only": 1.0,
            "relative_p95_ttft_vs_decode_only": 1.0,
        }
    ]

    for workload_name in MIXED_WORKLOADS:
        full_workload = frame[frame["workload_name"] == workload_name]
        decode = valid_decode_rows(full_workload)
        if len(decode) < min_samples:
            notes.append(
                f"insufficient data: {workload_name} has {len(decode)} valid decode "
                f"requests; at least {min_samples} required."
            )
            continue
        try:
            weights = matched_baseline_weights(baseline, decode)
        except ValueError as exc:
            notes.append(f"insufficient data: {workload_name}: {exc}.")
            continue
        matched_p95_tpot = weighted_quantile(
            baseline["tpot_ms"].to_numpy(dtype=float), weights, 0.95
        )
        matched_p95_ttft = weighted_quantile(
            baseline["ttft_ms"].to_numpy(dtype=float), weights, 0.95
        )
        metrics = ordinary_metrics(decode)
        prefill_ratio, decode_ratio = workload_ratios(frame, workload_name)
        rows.append(
            {
                "workload_name": workload_name,
                "prefill_heavy_ratio": prefill_ratio,
                "decode_heavy_ratio": decode_ratio,
                **metrics,
                "relative_p95_tpot_vs_decode_only": (
                    metrics["p95_tpot_ms"] / matched_p95_tpot
                    if matched_p95_tpot > 0
                    else float("nan")
                ),
                "relative_p95_ttft_vs_decode_only": (
                    metrics["p95_ttft_ms"] / matched_p95_ttft
                    if matched_p95_ttft > 0
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_FIELDS), notes


def format_findings(analysis: pd.DataFrame, notes: list[str]) -> str:
    lines = [
        BEGIN_MARKER,
        "## Colocated Prefill–Decode interference",
        "",
        "Data source: measured vLLM offline-proxy request metrics. Mixed-workload "
        "statistics below include only decode-heavy requests; the decode-only comparator "
        "is reweighted to the same 256/512-token output-length mix.",
        "",
    ]
    mixed = analysis[analysis["workload_name"].isin(MIXED_WORKLOADS)].sort_values(
        "prefill_heavy_ratio"
    )
    if len(mixed) != len(MIXED_WORKLOADS):
        lines.append("**Status: insufficient data for the complete comparison.**")
        lines.append("")
    if notes:
        lines.extend(f"- {note}" for note in notes)
        lines.append("")
    if not mixed.empty:
        lines.extend(
            [
                "| Workload | Prefill ratio | Decode samples P95 TPOT | vs decode-only | "
                "Decode samples P95 TTFT | vs decode-only |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in mixed.iterrows():
            lines.append(
                f"| {row['workload_name']} | {row['prefill_heavy_ratio']:.0%} | "
                f"{row['p95_tpot_ms']:.2f} ms | "
                f"{row['relative_p95_tpot_vs_decode_only']:.2f}× | "
                f"{row['p95_ttft_ms']:.2f} ms | "
                f"{row['relative_p95_ttft_vs_decode_only']:.2f}× |"
            )
        lines.append("")
        tpot_values = mixed["p95_tpot_ms"].to_numpy(dtype=float)
        ttft_values = mixed["p95_ttft_ms"].to_numpy(dtype=float)
        tpot_monotonic = bool(np.all(np.diff(tpot_values) > 0))
        ttft_monotonic = bool(np.all(np.diff(ttft_values) > 0))
        if tpot_monotonic:
            lines.append(
                "- **Decode tail interference:** P95 TPOT worsens monotonically as the "
                "prefill-heavy ratio increases from 30% to 70%."
            )
        else:
            lines.append(
                "- **Decode tail interference:** P95 TPOT is not monotonically worse in "
                "the current measurements; the data do not support that claim."
            )
        if ttft_monotonic:
            lines.append(
                "- **Queueing effect:** decode-request P95 TTFT also rises monotonically, "
                "showing that long prefills delay admission/scheduling before decoding."
            )
        else:
            lines.append(
                "- **Queueing effect:** decode-request P95 TTFT is not monotonic; further "
                "measurements are needed."
            )
        lines.append(
            "- **Why PD separation matters:** colocated prefill kernels consume scheduling "
            "and GPU execution capacity needed by decode token steps. Separate pools can "
            "isolate decode tail latency and let the scheduler provision prefill and decode "
            "resources independently."
        )
        lines.append(
            "- **Scope:** these measurements use a 200-request offline burst and "
            "first-output proxies. They provide controlled evidence of interference, not a "
            "claim that the same multipliers hold for every online arrival process."
        )
    lines.extend(["", END_MARKER])
    return "\n".join(lines) + "\n"


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def update_findings(path: Path, section: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.is_file() else "# Findings\n\n"
    if BEGIN_MARKER in existing and END_MARKER in existing:
        before = existing.split(BEGIN_MARKER, 1)[0].rstrip()
        after = existing.split(END_MARKER, 1)[1].lstrip()
        content = before + "\n\n" + section
        if after:
            content += "\n" + after
    else:
        content = existing.rstrip() + "\n\n" + section
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def run(args: argparse.Namespace) -> None:
    if args.min_decode_samples <= 0:
        raise ValueError("--min-decode-samples must be positive")
    frame = load_metrics(args.input)
    analysis, notes = build_analysis(frame, args.min_decode_samples)
    atomic_write_csv(args.output, analysis)
    update_findings(args.findings, format_findings(analysis, notes))
    if notes:
        for note in notes:
            LOGGER.warning(note)
    LOGGER.info(
        "Wrote colocated interference analysis: rows=%d output=%s findings=%s",
        len(analysis),
        args.output,
        args.findings,
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
        LOGGER.exception("Colocated interference analysis failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
