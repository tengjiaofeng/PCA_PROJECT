#!/usr/bin/env python3
"""Profile isolated prefill/decode proxy costs and fit stage cost models."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml


LOGGER = logging.getLogger("profile_stage_time")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_LENS = [128, 256, 512, 1024, 2048, 4096]
OUTPUT_LENS = [1, 32, 64, 128, 256, 512]
CSV_FIELDS = [
    "prompt_len",
    "output_len",
    "repeat_id",
    "ttft_ms",
    "total_latency_ms",
    "decode_total_ms",
    "tpot_ms",
    "tokens_per_second",
    "backend",
    "ttft_method",
    "success",
    "error_msg",
    "data_source",
]

TOPICS = (
    "distributed scheduling",
    "compiler analysis",
    "marine ecology",
    "database systems",
    "robot navigation",
    "renewable energy",
    "numerical optimization",
    "network security",
)
FRAGMENTS = (
    "Experiment {number} studies {topic} under configuration {code}.",
    "The analyst compares measurement {number} with reference {code}.",
    "Explain how {topic} changes when parameter {code} reaches {number}.",
    "Dataset {code} contains observation {number} for {topic}.",
    "A reproducible report should evaluate {topic} in trial {number}.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model.yaml")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics/stage_profile.csv"),
    )
    parser.add_argument("--backend", choices=("vllm", "hf"), default="vllm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Unrecorded 128/1 warmup requests before profiling (default: 1).",
    )
    parser.add_argument(
        "--cost-model-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics" / "stage_cost_model.json",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Override tokenizer_name from model config.",
    )
    parser.add_argument(
        "--allow-remote-model",
        action="store_true",
        help="Allow backend resolution when no complete local snapshot exists.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    required = {
        "model_name",
        "tokenizer_name",
        "dtype",
        "max_model_len",
        "gpu_memory_utilization",
        "tensor_parallel_size",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"Model config is missing: {', '.join(missing)}")
    return data


def cache_roots() -> list[Path]:
    roots: list[Path] = []
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return list(dict.fromkeys(roots))


def resolve_local_snapshot(name_or_path: str, allow_remote: bool) -> str:
    supplied = Path(name_or_path).expanduser()
    if supplied.exists():
        return str(supplied.resolve())
    repository_name = "models--" + name_or_path.replace("/", "--")
    for root in cache_roots():
        repository = root / repository_name
        main_ref = repository / "refs" / "main"
        if main_ref.is_file():
            revision = main_ref.read_text(encoding="utf-8").strip()
            snapshot = repository / "snapshots" / revision
            if (snapshot / "config.json").is_file():
                return str(snapshot)
        for snapshot in sorted((repository / "snapshots").glob("*")):
            if (snapshot / "config.json").is_file():
                return str(snapshot)
    if allow_remote:
        LOGGER.warning("No local snapshot for %s; allowing remote resolution", name_or_path)
        return name_or_path
    raise FileNotFoundError(
        f"No complete local snapshot for {name_or_path}; use --allow-remote-model "
        "only when remote access is intended"
    )


class PromptFactory:
    """Generate exact-length, prefix-diverse prompts for isolated profiling."""

    def __init__(self, tokenizer: Any, rng: np.random.Generator) -> None:
        self.tokenizer = tokenizer
        self.rng = rng
        self.special_overhead = len(
            tokenizer.encode("", add_special_tokens=True)
        )

    def _fragment(self) -> str:
        pattern = str(self.rng.choice(FRAGMENTS))
        return pattern.format(
            number=int(self.rng.integers(1, 1_000_000)),
            topic=str(self.rng.choice(TOPICS)),
            code=bytes(self.rng.bytes(6)).hex(),
        )

    def build(self, target_len: int, marker_index: int) -> str:
        content_budget = target_len - self.special_overhead
        if content_budget <= 0:
            raise ValueError(f"Target prompt length is too small: {target_len}")
        marker = (
            f"{bytes(self.rng.bytes(8)).hex()} Profile-ID: {marker_index}. "
            f"Topic: {self.rng.choice(TOPICS)}. "
        )
        token_ids = list(
            self.tokenizer.encode(marker, add_special_tokens=False)
        )
        if len(token_ids) >= content_budget:
            raise ValueError(f"Profiling marker exceeds prompt budget {target_len}")
        while len(token_ids) < content_budget:
            token_ids.extend(
                self.tokenizer.encode(
                    " " + self._fragment(), add_special_tokens=False
                )
            )
        prompt = self.tokenizer.decode(
            token_ids[:content_budget],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        actual = len(self.tokenizer.encode(prompt, add_special_tokens=True))
        if actual != target_len:
            canonical = list(
                self.tokenizer.encode(prompt, add_special_tokens=False)
            )[:content_budget]
            while len(canonical) < content_budget:
                canonical.extend(
                    self.tokenizer.encode(
                        " " + self._fragment(), add_special_tokens=False
                    )
                )
            prompt = self.tokenizer.decode(
                canonical[:content_budget],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            actual = len(self.tokenizer.encode(prompt, add_special_tokens=True))
        if actual != target_len:
            raise RuntimeError(
                f"Exact prompt construction failed: target={target_len}, actual={actual}"
            )
        return prompt


def synchronize_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def reset_prefix_cache(llm: Any, timeout_s: float = 10.0) -> None:
    method = getattr(llm, "reset_prefix_cache", None)
    if not callable(method):
        raise RuntimeError("vLLM backend does not expose reset_prefix_cache")
    deadline = time.monotonic() + timeout_s
    while not bool(method()):
        if time.monotonic() >= deadline:
            raise TimeoutError("Prefix cache reset timed out")
        time.sleep(0.1)


def initialize_vllm(
    config: dict[str, Any], model_path: str, tokenizer_path: str, seed: int
) -> tuple[Any, Any, Callable[[], None]]:
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=Path(tokenizer_path).exists()
    )
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_path,
        dtype=str(config["dtype"]),
        max_model_len=int(config["max_model_len"]),
        gpu_memory_utilization=float(config["gpu_memory_utilization"]),
        tensor_parallel_size=int(config["tensor_parallel_size"]),
        enable_prefix_caching=bool(config.get("enable_prefix_caching", True)),
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

    return llm, tokenizer, cleanup


def initialize_hf(
    config: dict[str, Any], model_path: str, tokenizer_path: str, seed: int
) -> tuple[Any, Any, Callable[[], None]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(str(config["dtype"]).lower())
    if dtype is None:
        raise ValueError(f"Unsupported HF dtype: {config['dtype']}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=Path(tokenizer_path).exists()
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=Path(model_path).exists(),
    )
    model.eval()

    def cleanup() -> None:
        nonlocal model
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model, tokenizer, cleanup


def empty_row(
    prompt_len: int,
    output_len: int,
    repeat_id: int,
    backend: str,
    error: str,
) -> dict[str, Any]:
    return {
        "prompt_len": prompt_len,
        "output_len": output_len,
        "repeat_id": repeat_id,
        "ttft_ms": float("nan"),
        "total_latency_ms": float("nan"),
        "decode_total_ms": float("nan"),
        "tpot_ms": float("nan"),
        "tokens_per_second": float("nan"),
        "backend": backend,
        "ttft_method": "unavailable",
        "success": False,
        "error_msg": error,
        "data_source": "measured",
    }


def measured_row(
    prompt_len: int,
    output_len: int,
    repeat_id: int,
    ttft_ms: float,
    total_latency_ms: float,
    backend: str,
    ttft_method: str,
) -> dict[str, Any]:
    decode_total_ms = (
        0.0 if output_len <= 1 else max(0.0, total_latency_ms - ttft_ms)
    )
    tpot_ms = decode_total_ms / (output_len - 1) if output_len > 1 else float("nan")
    token_rate = (
        output_len / (total_latency_ms / 1000.0)
        if total_latency_ms > 0
        else float("nan")
    )
    return {
        "prompt_len": prompt_len,
        "output_len": output_len,
        "repeat_id": repeat_id,
        "ttft_ms": ttft_ms,
        "total_latency_ms": total_latency_ms,
        "decode_total_ms": decode_total_ms,
        "tpot_ms": tpot_ms,
        "tokens_per_second": token_rate,
        "backend": backend,
        "ttft_method": ttft_method,
        "success": True,
        "error_msg": "",
        "data_source": "measured",
    }


def measure_vllm_once(
    llm: Any,
    prompt: str,
    prompt_len: int,
    output_len: int,
    repeat_id: int,
    seed: int,
) -> dict[str, Any]:
    from vllm import SamplingParams

    reset_prefix_cache(llm)
    params = SamplingParams(
        temperature=0.0,
        max_tokens=output_len,
        min_tokens=output_len,
        ignore_eos=True,
        seed=seed + repeat_id,
    )
    outputs = llm.generate([prompt], sampling_params=[params], use_tqdm=False)
    if len(outputs) != 1 or not outputs[0].outputs:
        raise RuntimeError("vLLM did not return exactly one completion")
    output = outputs[0]
    actual_prompt_len = len(output.prompt_token_ids or [])
    actual_output_len = len(output.outputs[0].token_ids)
    if actual_prompt_len != prompt_len or actual_output_len != output_len:
        raise RuntimeError(
            "Token length mismatch: "
            f"prompt {actual_prompt_len}/{prompt_len}, output {actual_output_len}/{output_len}"
        )
    cached_tokens = int(getattr(output, "num_cached_tokens", 0) or 0)
    if cached_tokens != 0:
        raise RuntimeError(
            f"Cold-cache stage profile unexpectedly reused {cached_tokens} prompt tokens"
        )
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        raise RuntimeError("vLLM request metrics are unavailable")
    first_latency = float(getattr(metrics, "first_token_latency", 0.0) or 0.0)
    first_ts = float(getattr(metrics, "first_token_ts", 0.0) or 0.0)
    last_ts = float(getattr(metrics, "last_token_ts", 0.0) or 0.0)
    if first_latency <= 0 or first_ts <= 0 or last_ts < first_ts:
        raise RuntimeError("vLLM returned invalid first/last token timestamps")
    ttft_ms = first_latency * 1000.0
    total_latency_ms = (first_latency + last_ts - first_ts) * 1000.0
    return measured_row(
        prompt_len,
        output_len,
        repeat_id,
        ttft_ms,
        total_latency_ms,
        "vllm",
        "offline_proxy",
    )


def measure_hf_once(
    model: Any,
    tokenizer: Any,
    prompt: str,
    prompt_len: int,
    output_len: int,
    repeat_id: int,
) -> dict[str, Any]:
    import torch

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    actual_prompt_len = int(inputs["input_ids"].shape[-1])
    if actual_prompt_len != prompt_len:
        raise RuntimeError(
            f"HF prompt length mismatch: {actual_prompt_len}/{prompt_len}"
        )

    def generate(tokens: int) -> Any:
        return model.generate(
            **inputs,
            do_sample=False,
            min_new_tokens=tokens,
            max_new_tokens=tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    synchronize_cuda()
    proxy_start = time.perf_counter()
    with torch.inference_mode():
        generate(1)
    synchronize_cuda()
    ttft_ms = (time.perf_counter() - proxy_start) * 1000.0

    synchronize_cuda()
    total_start = time.perf_counter()
    with torch.inference_mode():
        generated = generate(output_len)
    synchronize_cuda()
    total_latency_ms = (time.perf_counter() - total_start) * 1000.0
    actual_output_len = int(generated.shape[-1]) - actual_prompt_len
    if actual_output_len != output_len:
        raise RuntimeError(
            f"HF output length mismatch: {actual_output_len}/{output_len}"
        )
    return measured_row(
        prompt_len,
        output_len,
        repeat_id,
        ttft_ms,
        total_latency_ms,
        "hf",
        "offline_proxy",
    )


def warmup_backend(
    backend: str,
    backend_object: Any,
    tokenizer: Any,
    factory: PromptFactory,
    count: int,
    seed: int,
) -> None:
    if count <= 0:
        return
    LOGGER.info("Running %d unrecorded stage-profiler warmup requests", count)
    for index in range(count):
        prompt = factory.build(128, -index - 1)
        if backend == "vllm":
            measure_vllm_once(
                backend_object, prompt, 128, 1, -index - 1, seed
            )
        else:
            measure_hf_once(
                backend_object, tokenizer, prompt, 128, 1, -index - 1
            )
    if backend == "vllm":
        reset_prefix_cache(backend_object)


def profile_matrix(
    backend: str,
    backend_object: Any,
    tokenizer: Any,
    factory: PromptFactory,
    repeat: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    marker_index = 0
    total = len(PROMPT_LENS) * len(OUTPUT_LENS) * repeat
    completed = 0
    for prompt_len in PROMPT_LENS:
        for output_len in OUTPUT_LENS:
            for repeat_id in range(repeat):
                marker_index += 1
                try:
                    prompt = factory.build(prompt_len, marker_index)
                    if backend == "vllm":
                        row = measure_vllm_once(
                            backend_object,
                            prompt,
                            prompt_len,
                            output_len,
                            repeat_id,
                            seed,
                        )
                    else:
                        row = measure_hf_once(
                            backend_object,
                            tokenizer,
                            prompt,
                            prompt_len,
                            output_len,
                            repeat_id,
                        )
                except Exception as exc:
                    LOGGER.exception(
                        "Stage measurement failed: prompt=%d output=%d repeat=%d",
                        prompt_len,
                        output_len,
                        repeat_id,
                    )
                    row = empty_row(
                        prompt_len, output_len, repeat_id, backend, str(exc)
                    )
                rows.append(row)
                completed += 1
                LOGGER.info(
                    "Profile progress %d/%d | prompt=%d output=%d repeat=%d success=%s",
                    completed,
                    total,
                    prompt_len,
                    output_len,
                    repeat_id,
                    row["success"],
                )
    return rows


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = actual - predicted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "r2": r2,
        "mae_ms": float(np.mean(np.abs(residual))),
        "rmse_ms": float(np.sqrt(np.mean(residual**2))),
    }


def fit_cost_models(
    rows: list[dict[str, Any]], model_name: str, backend: str
) -> dict[str, Any]:
    successful = [row for row in rows if row["success"]]
    prefill_rows = [row for row in successful if row["output_len"] == 1]
    decode_rows = [row for row in successful if row["output_len"] > 1]
    result: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "backend": backend,
        "measurement_semantics": {
            "ttft_ms": "offline first-output proxy from an isolated request",
            "prefill_cost_ms": "ttft_ms for output_len=1",
            "decode_total_ms": "max(total_latency_ms - ttft_ms, 0)",
            "tpot_ms": "decode_total_ms / (output_len - 1)",
            "cache_state": "prefix cache reset before every measured request",
        },
        "training_rows": len(successful),
        "failed_rows": len(rows) - len(successful),
    }

    if len(prefill_rows) < 2:
        result["prefill_cost_model"] = {
            "status": "unavailable",
            "reason": "insufficient successful output_len=1 measurements",
            "parameters": None,
            "metrics": None,
        }
    else:
        prompt_values = sorted({int(row["prompt_len"]) for row in prefill_rows})
        knot_costs = [
            float(
                np.mean(
                    [
                        row["ttft_ms"]
                        for row in prefill_rows
                        if row["prompt_len"] == prompt_len
                    ]
                )
            )
            for prompt_len in prompt_values
        ]
        actual = np.asarray([row["ttft_ms"] for row in prefill_rows], dtype=float)
        predicted = np.interp(
            np.asarray([row["prompt_len"] for row in prefill_rows], dtype=float),
            np.asarray(prompt_values, dtype=float),
            np.asarray(knot_costs, dtype=float),
        )
        result["prefill_cost_model"] = {
            "status": "fitted",
            "type": "piecewise_linear",
            "equation": "linear interpolation of mean TTFT at prompt-length knots",
            "parameters": {
                "prompt_len_knots": prompt_values,
                "prefill_cost_ms_at_knots": knot_costs,
            },
            "metrics": regression_metrics(actual, predicted),
            "training_rows": len(prefill_rows),
        }

    if len(decode_rows) < 3:
        result["decode_cost_model"] = {
            "status": "unavailable",
            "reason": "insufficient successful output_len>1 measurements",
            "parameters": None,
            "metrics": None,
        }
    else:
        prompt = np.asarray([row["prompt_len"] for row in decode_rows], dtype=float)
        generated_after_first = np.asarray(
            [row["output_len"] - 1 for row in decode_rows], dtype=float
        )
        actual = np.asarray(
            [row["decode_total_ms"] for row in decode_rows], dtype=float
        )
        # Decode cost grows with generated tokens and context length. The
        # interaction term lets prompt length affect per-token decode cost.
        design = np.column_stack(
            [
                np.ones_like(prompt),
                generated_after_first,
                prompt * generated_after_first,
                generated_after_first**2,
            ]
        )
        coefficients, _, _, _ = np.linalg.lstsq(design, actual, rcond=None)
        predicted = design @ coefficients
        result["decode_cost_model"] = {
            "status": "fitted",
            "type": "multivariate_linear_with_interaction",
            "equation": (
                "decode_total_ms = intercept + alpha*(output_len-1) + "
                "beta*prompt_len*(output_len-1) + gamma*(output_len-1)^2"
            ),
            "prediction_postprocess": "clamp predicted decode_total_ms to >= 0",
            "parameters": {
                "intercept_ms": float(coefficients[0]),
                "alpha_output_token_ms": float(coefficients[1]),
                "beta_prompt_output_ms_per_token2": float(coefficients[2]),
                "gamma_output_quadratic_ms_per_token2": float(coefficients[3]),
            },
            "metrics": regression_metrics(actual, predicted),
            "training_rows": len(decode_rows),
        }
    return result


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def run(args: argparse.Namespace) -> bool:
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative")
    config = load_yaml(args.model_config)
    max_required = max(PROMPT_LENS) + max(OUTPUT_LENS)
    if max_required > int(config["max_model_len"]):
        raise ValueError(
            f"Profiling matrix needs {max_required} tokens but max_model_len="
            f"{config['max_model_len']}"
        )
    model_path = resolve_local_snapshot(
        str(config["model_name"]), args.allow_remote_model
    )
    tokenizer_name = args.tokenizer or str(config["tokenizer_name"])
    tokenizer_path = resolve_local_snapshot(
        tokenizer_name, args.allow_remote_model
    )
    cleanup: Callable[[], None] | None = None
    expected_rows = len(PROMPT_LENS) * len(OUTPUT_LENS) * args.repeat
    rows: list[dict[str, Any]]
    try:
        if args.backend == "vllm":
            backend_object, tokenizer, cleanup = initialize_vllm(
                config, model_path, tokenizer_path, args.seed
            )
        else:
            backend_object, tokenizer, cleanup = initialize_hf(
                config, model_path, tokenizer_path, args.seed
            )
    except Exception as exc:
        LOGGER.exception("Backend initialization failed")
        error = f"{type(exc).__name__}: {exc}"
        rows = [
            empty_row(prompt_len, output_len, repeat_id, args.backend, error)
            for prompt_len in PROMPT_LENS
            for output_len in OUTPUT_LENS
            for repeat_id in range(args.repeat)
        ]
    else:
        try:
            rng = np.random.default_rng(args.seed)
            factory = PromptFactory(tokenizer, rng)
            warmup_backend(
                args.backend,
                backend_object,
                tokenizer,
                factory,
                args.warmup,
                args.seed,
            )
            rows = profile_matrix(
                args.backend,
                backend_object,
                tokenizer,
                factory,
                args.repeat,
                args.seed,
            )
        except Exception as exc:
            LOGGER.exception("Stage profiling aborted unexpectedly")
            error = f"{type(exc).__name__}: {exc}"
            rows = [
                empty_row(prompt_len, output_len, repeat_id, args.backend, error)
                for prompt_len in PROMPT_LENS
                for output_len in OUTPUT_LENS
                for repeat_id in range(args.repeat)
            ]
        finally:
            if cleanup is not None:
                cleanup()

    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} profile rows, got {len(rows)}")
    atomic_write_csv(args.output, rows)
    model = fit_cost_models(rows, str(config["model_name"]), args.backend)
    # JSON forbids non-standard NaN. Metrics containing undefined R2 become null.
    def replace_nonfinite(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: replace_nonfinite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace_nonfinite(item) for item in value]
        return value

    atomic_write_json(args.cost_model_output, replace_nonfinite(model))
    successful = sum(bool(row["success"]) for row in rows)
    LOGGER.info(
        "Stage profile complete: success=%d failed=%d output=%s model=%s",
        successful,
        len(rows) - successful,
        args.output,
        args.cost_model_output,
    )
    return successful == len(rows)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    try:
        success = run(args)
    except Exception as exc:
        LOGGER.exception("Stage profiler failed before outputs were finalized: %s", exc)
        sys.exit(1)
    if not success:
        sys.exit(2)


if __name__ == "__main__":
    main()
