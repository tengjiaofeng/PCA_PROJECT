#!/usr/bin/env python3
"""Render real-4GPU validation figures as both PNG and PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real4gpu-summary", type=Path,
        default=Path("outputs/metrics/real4gpu/real4gpu_summary.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/figures/real4gpu")
    )
    return parser.parse_args()


def save_both(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(directory / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def labels(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["workload_name"].astype(str).str.replace("mixed_", "m", regex=False)
        + "\nλ=" + frame["arrival_rate"].astype(str)
        + ",c=" + frame["concurrency"].astype(str)
    )


def plot_colocated_tp(frame: pd.DataFrame, metric: str, stem: str, ylabel: str, output: Path) -> bool:
    subset = frame[frame["serving_mode"].isin(["colocated_4replica", "aggregated_tp4"])].copy()
    if subset.empty or subset["serving_mode"].nunique() < 2:
        print(f"WARNING: insufficient colocated/TP4 data for {stem}")
        return False
    subset["scenario"] = labels(subset)
    pivot = subset.pivot_table(index="scenario", columns="serving_mode", values=metric, aggfunc="mean")
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(max(7, len(pivot) * 1.2), 4.5))
    pivot.plot(kind="bar", ax=ax, width=0.75)
    ax.set_xlabel("Workload / offered arrival rate / concurrency")
    ax.set_ylabel(ylabel)
    ax.set_title("Real 4-GPU online serving")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    save_both(fig, output, stem)
    return True


def plot_pd_goodput(frame: pd.DataFrame, output: Path) -> bool:
    subset = frame[frame["result_type"] == "real_disaggregated_pd"].copy()
    if subset.empty or subset["serving_mode"].nunique() < 2:
        print("WARNING: insufficient real PD data; no PD-goodput figure generated")
        return False
    subset["scenario"] = labels(subset)
    pivot = subset.pivot_table(
        index="scenario", columns="serving_mode", values="goodput_req_s", aggfunc="mean"
    ).sort_index()
    fig, ax = plt.subplots(figsize=(max(7, len(pivot) * 1.2), 4.5))
    pivot.plot(kind="bar", ax=ax, width=0.78)
    ax.set_xlabel("Workload / offered arrival rate / concurrency")
    ax.set_ylabel("Goodput (requests/s)")
    ax.set_title("Measured real-PD ratio comparison")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    save_both(fig, output, "fig_real4gpu_pd_ratio_goodput")
    return True


def plot_gpu_util(frame: pd.DataFrame, output: Path) -> bool:
    valid = frame.dropna(subset=["avg_gpu_util"]).copy()
    if valid.empty:
        print("WARNING: no matched GPU telemetry; utilization figure not generated")
        return False
    grouped = valid.groupby("serving_mode", as_index=False).agg(
        avg_gpu_util=("avg_gpu_util", "mean"),
        gpu_util_imbalance=("gpu_util_imbalance", "mean"),
    )
    x = np.arange(len(grouped))
    fig, ax = plt.subplots(figsize=(max(6.5, len(grouped) * 1.3), 4.5))
    ax.bar(x, grouped["avg_gpu_util"], color="#4C78A8", label="Mean utilization")
    ax.scatter(
        x, grouped["gpu_util_imbalance"], color="#E45756", marker="D",
        label="Utilization imbalance", zorder=3,
    )
    ax.set_xticks(x, grouped["serving_mode"], rotation=25, ha="right")
    ax.set_ylabel("GPU utilization / imbalance (percentage points)")
    ax.set_title("Measured 4-GPU utilization")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    save_both(fig, output, "fig_real4gpu_gpu_utilization")
    return True


def main() -> None:
    args = parse_args()
    if not args.real4gpu_summary.is_file():
        raise FileNotFoundError(args.real4gpu_summary)
    frame = pd.read_csv(args.real4gpu_summary)
    if frame.empty:
        print("WARNING: real4gpu_summary.csv is empty; no figures generated")
        return
    generated = 0
    generated += plot_colocated_tp(
        frame, "p95_ttft_ms", "fig_real4gpu_colocated_vs_tp4_ttft",
        "P95 TTFT (ms)", args.output_dir,
    )
    generated += plot_colocated_tp(
        frame, "p95_tpot_ms", "fig_real4gpu_colocated_vs_tp4_tpot",
        "P95 TPOT (ms/token)", args.output_dir,
    )
    generated += plot_pd_goodput(frame, args.output_dir)
    generated += plot_gpu_util(frame, args.output_dir)
    print(f"Generated {generated} figure set(s), each as PNG and PDF")


if __name__ == "__main__":
    main()
