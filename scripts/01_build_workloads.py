#!/usr/bin/env python3
"""Build prefix-controlled workload traces for LLM profiling and simulation."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


LOGGER = logging.getLogger("build_workloads")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODES = ("synthetic_unique", "synthetic_cache_friendly", "real_or_sharegpt")

TOPICS = (
    "distributed systems",
    "marine ecology",
    "compiler optimization",
    "renewable energy",
    "ancient astronomy",
    "database indexing",
    "robot navigation",
    "public transportation",
    "protein folding",
    "digital typography",
    "supply chain planning",
    "numerical linear algebra",
    "urban hydrology",
    "computer security",
    "language education",
    "satellite imaging",
)
ENTITIES = (
    "Orchid Laboratory",
    "Northbridge Institute",
    "Atlas Research Group",
    "Juniper Systems",
    "Silver River Observatory",
    "Cedar Analytics",
    "Kepler Workshop",
    "Harbor Data Cooperative",
    "Nimbus Engineering",
    "Pioneer Archive",
)
TASKS = (
    "compare the competing design choices",
    "identify hidden bottlenecks",
    "write a concise technical explanation",
    "derive a reproducible evaluation plan",
    "summarize the evidence and its limitations",
    "propose three falsifiable hypotheses",
    "analyze the latency-throughput tradeoff",
    "construct a step-by-step validation checklist",
    "explain the result to a graduate student",
    "evaluate the robustness of the proposed method",
)
SENTENCE_PATTERNS = (
    "For {topic}, {entity} recorded measurement {number} during trial {trial}.",
    "Dataset {code} concerns {topic}; its observed value was {number} units.",
    "Please {task} using evidence from {entity} and scenario {trial}.",
    "A reviewer studying {topic} assigned case {code} a score of {number}.",
    "In experiment {trial}, {entity} changed parameter {code} to {number}.",
    "The next section must {task}, with special attention to {topic}.",
    "Entity {entity} reported that sample {code} remained stable for {number} cycles.",
    "Question {trial} asks how {topic} affects metric {code} at level {number}.",
)

REQUEST_FIELDS = (
    "request_id",
    "workload_name",
    "workload_mode",
    "request_type",
    "prompt_len",
    "output_len",
    "prompt",
    "arrival_time",
    "cache_group_id",
    "intended_prefix_reuse",
    "unique_marker",
)

OVERLAP_FIELDS = (
    "workload_name",
    "workload_mode",
    "num_requests",
    "avg_prompt_len",
    "p50_lcp_tokens",
    "p95_lcp_tokens",
    "max_lcp_tokens",
    "avg_lcp_tokens",
    "prefix_reuse_ratio_estimated",
)

SUMMARY_FIELDS = (
    "workload_name",
    "workload_mode",
    "num_requests",
    "avg_prompt_len",
    "avg_output_len",
    "prefill_heavy_ratio",
    "decode_heavy_ratio",
    "avg_lcp_tokens",
    "p95_lcp_tokens",
    "max_lcp_tokens",
)


@dataclass(frozen=True)
class RequestSpec:
    prompt_len: int
    output_len: int
    request_type: str


@dataclass(frozen=True)
class RealSample:
    source_id: str
    prompt: str
    prompt_len: int
    output_len: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/workloads.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=MODES, default="synthetic_unique")
    parser.add_argument(
        "--tokenizer",
        "--tokenizer-name",
        dest="tokenizer",
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--arrival-rate", type=float, default=1.0)
    parser.add_argument("--common-prefix-len", type=int, default=0)
    parser.add_argument("--prefix-reuse-ratio", type=float, default=0.0)
    parser.add_argument(
        "--real-data",
        type=Path,
        default=None,
        help="ShareGPT or JSON/JSONL input required by real_or_sharegpt mode.",
    )
    parser.add_argument(
        "--allow-tokenizer-download",
        action="store_true",
        help="Allow remote tokenizer resolution; local snapshots are preferred.",
    )
    parser.add_argument(
        "--lcp-sample-pairs",
        type=int,
        default=2000,
        help="Maximum randomly sampled request pairs per workload.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def classify_request(name: str) -> str:
    if name == "short_short":
        return "short_short"
    if name == "long_long":
        return "long_long"
    if name.startswith("prefill_heavy_"):
        return "prefill_heavy"
    if name.startswith("decode_heavy_"):
        return "decode_heavy"
    raise ValueError(f"Cannot infer request_type from workload name: {name}")


def validate_workloads(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    workloads = config.get("workloads")
    if not isinstance(workloads, dict) or not workloads:
        raise ValueError("Config must contain a non-empty workloads mapping")
    fixed: set[str] = set()
    for name, definition in workloads.items():
        if not isinstance(definition, dict):
            raise ValueError(f"Workload {name} must be a mapping")
        if definition.get("type") == "mixture":
            continue
        classify_request(name)
        for field in ("prompt_len", "output_len"):
            value = definition.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name}.{field} must be a positive integer")
        fixed.add(name)

    for name, definition in workloads.items():
        if definition.get("type") != "mixture":
            continue
        components = definition.get("components")
        if not isinstance(components, list) or not components:
            raise ValueError(f"{name}.components must be a non-empty list")
        total = 0.0
        types: set[str] = set()
        for component in components:
            referenced = component.get("workload") if isinstance(component, dict) else None
            probability = component.get("probability") if isinstance(component, dict) else None
            if referenced not in fixed:
                raise ValueError(f"{name} references unknown fixed workload: {referenced}")
            if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
                raise ValueError(f"Invalid probability in {name}: {probability}")
            total += float(probability)
            types.add(classify_request(str(referenced)))
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(f"{name} component probabilities sum to {total}, not 1")
        if types != {"prefill_heavy", "decode_heavy"}:
            raise ValueError(f"{name} must combine prefill-heavy and decode-heavy requests")
    return workloads


def cache_roots() -> list[Path]:
    roots: list[Path] = []
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return list(dict.fromkeys(roots))


def find_cached_tokenizer(name: str) -> Path | None:
    supplied = Path(name).expanduser()
    if supplied.exists():
        return supplied.resolve()
    repository_name = "models--" + name.replace("/", "--")
    for root in cache_roots():
        repository = root / repository_name
        main_ref = repository / "refs" / "main"
        if main_ref.is_file():
            revision = main_ref.read_text(encoding="utf-8").strip()
            snapshot = repository / "snapshots" / revision
            if (snapshot / "tokenizer_config.json").is_file():
                return snapshot
        for snapshot in sorted((repository / "snapshots").glob("*")):
            if (snapshot / "tokenizer_config.json").is_file():
                return snapshot
    return None


def load_tokenizer(name: str, allow_download: bool) -> Any | None:
    try:
        from transformers import AutoTokenizer

        cached = find_cached_tokenizer(name)
        if cached is not None:
            LOGGER.info("Loading tokenizer %s from local snapshot %s", name, cached)
            return AutoTokenizer.from_pretrained(str(cached), local_files_only=True)
        LOGGER.info(
            "Loading tokenizer %s (%s)",
            name,
            "downloads allowed" if allow_download else "local cache only",
        )
        return AutoTokenizer.from_pretrained(name, local_files_only=not allow_download)
    except Exception as exc:
        message = (
            f"Could not load tokenizer '{name}' ({type(exc).__name__}: {exc}). "
            "Falling back to whitespace token estimates; prefix-overlap values are approximate."
        )
        warnings.warn(message, RuntimeWarning)
        LOGGER.warning(message)
        return None


class TokenAdapter:
    """Common token operations for a real tokenizer or explicit word fallback."""

    def __init__(self, tokenizer: Any | None) -> None:
        self.tokenizer = tokenizer
        self.uses_word_fallback = tokenizer is None
        self.special_overhead = (
            0 if tokenizer is None else len(tokenizer.encode("", add_special_tokens=True))
        )

    def encode_content(self, text: str) -> list[Any]:
        if self.tokenizer is None:
            return text.split()
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def encode_full(self, text: str) -> list[Any]:
        if self.tokenizer is None:
            return text.split()
        return list(self.tokenizer.encode(text, add_special_tokens=True))

    def decode_content(self, tokens: list[Any]) -> str:
        if self.tokenizer is None:
            return " ".join(str(token) for token in tokens)
        return self.tokenizer.decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def output_length(self, text: str) -> int:
        return len(self.encode_content(text))


def random_marker(rng: np.random.Generator, index: int) -> str:
    # Two independent 48-bit values are ample for a 200-request trace while
    # leaving room for a controlled common prefix even in 128-token prompts.
    request_uuid = uuid.UUID(bytes=bytes(rng.bytes(16))).hex[:12]
    random_key = bytes(rng.bytes(6)).hex()
    topic = str(rng.choice(TOPICS))
    # The random key is the first token-bearing text, minimizing shared prefixes.
    return (
        f"{random_key} Request-ID: {request_uuid}. Sequence: {index}. "
        f"Topic: {topic}. "
    )


def random_fragment(rng: np.random.Generator) -> str:
    pattern = str(rng.choice(SENTENCE_PATTERNS))
    return pattern.format(
        topic=str(rng.choice(TOPICS)),
        entity=str(rng.choice(ENTITIES)),
        task=str(rng.choice(TASKS)),
        number=int(rng.integers(1, 1_000_000)),
        trial=int(rng.integers(1, 100_000)),
        code=bytes(rng.bytes(5)).hex(),
    )


def fill_random_tokens(
    adapter: TokenAdapter,
    initial_tokens: list[Any],
    target_content_len: int,
    rng: np.random.Generator,
) -> list[Any]:
    tokens = list(initial_tokens)
    while len(tokens) < target_content_len:
        tokens.extend(adapter.encode_content(" " + random_fragment(rng)))
    return tokens[:target_content_len]


def make_shared_prefix(
    adapter: TokenAdapter,
    token_length: int,
    workload_name: str,
    rng: np.random.Generator,
) -> list[Any]:
    if token_length <= 0:
        return []
    header = (
        f"Shared serving context for cache group {workload_name}. "
        "All requests in this group must use the following reference material. "
    )
    tokens = adapter.encode_content(header)
    while len(tokens) < token_length:
        tokens.extend(adapter.encode_content(" " + random_fragment(rng)))
    return tokens[:token_length]


def build_synthetic_prompt(
    adapter: TokenAdapter,
    target_len: int,
    marker: str,
    shared_prefix: list[Any],
    rng: np.random.Generator,
) -> tuple[str, int, int]:
    content_budget = target_len - adapter.special_overhead
    if content_budget <= 0:
        raise ValueError(
            f"Target prompt length {target_len} is not larger than special-token overhead"
        )
    marker_tokens = adapter.encode_content(marker)
    if len(marker_tokens) >= content_budget:
        raise ValueError(
            f"Unique marker requires {len(marker_tokens)} tokens but budget is {content_budget}"
        )
    # Always leave room for the marker plus a small request-specific body.
    max_prefix_len = max(0, content_budget - len(marker_tokens) - 8)
    effective_prefix = shared_prefix[:max_prefix_len]
    initial = effective_prefix + marker_tokens
    content_tokens = fill_random_tokens(adapter, initial, content_budget, rng)
    prompt = adapter.decode_content(content_tokens)
    actual_len = len(adapter.encode_full(prompt))
    if actual_len != target_len:
        # Tokenizer decode/encode is normally idempotent. A second canonical
        # pass handles tokenizers that normalize whitespace at boundaries.
        canonical = adapter.encode_content(prompt)[:content_budget]
        canonical = fill_random_tokens(adapter, canonical, content_budget, rng)
        prompt = adapter.decode_content(canonical)
        actual_len = len(adapter.encode_full(prompt))
    if actual_len != target_len:
        raise RuntimeError(
            f"Could not construct exact prompt length: target={target_len}, actual={actual_len}"
        )
    return prompt, actual_len, len(effective_prefix)


def fixed_spec(name: str, definition: dict[str, Any]) -> RequestSpec:
    return RequestSpec(
        prompt_len=int(definition["prompt_len"]),
        output_len=int(definition["output_len"]),
        request_type=classify_request(name),
    )


def build_specs(
    workload_name: str,
    definition: dict[str, Any],
    workloads: dict[str, dict[str, Any]],
    num_requests: int,
    rng: np.random.Generator,
) -> list[RequestSpec]:
    if definition.get("type") != "mixture":
        return [fixed_spec(workload_name, definition)] * num_requests

    probabilities = {"prefill_heavy": 0.0, "decode_heavy": 0.0}
    for component in definition["components"]:
        category = classify_request(str(component["workload"]))
        probabilities[category] += float(component["probability"])
    prefill_count = round(num_requests * probabilities["prefill_heavy"])
    categories = ["prefill_heavy"] * prefill_count + ["decode_heavy"] * (
        num_requests - prefill_count
    )
    pools: dict[str, list[str]] = {"prefill_heavy": [], "decode_heavy": []}
    for candidate, candidate_definition in workloads.items():
        if candidate_definition.get("type") == "mixture":
            continue
        category = classify_request(candidate)
        if category in pools:
            pools[category].append(candidate)
    if not all(pools.values()):
        raise ValueError("Mixed workloads require fixed prefill and decode pools")
    specs = [
        fixed_spec(source := str(rng.choice(pools[category])), workloads[source])
        for category in categories
    ]
    rng.shuffle(specs)
    return specs


def poisson_arrivals(
    count: int, arrival_rate: float, rng: np.random.Generator
) -> np.ndarray:
    return np.cumsum(rng.exponential(scale=1.0 / arrival_rate, size=count))


def build_synthetic_records(
    workload_name: str,
    specs: list[RequestSpec],
    mode: str,
    adapter: TokenAdapter,
    common_prefix_len: int,
    prefix_reuse_ratio: float,
    arrival_rate: float,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[list[Any]]]:
    count = len(specs)
    reused_indices: set[int] = set()
    shared_prefix: list[Any] = []
    if mode == "synthetic_cache_friendly":
        reuse_count = round(count * prefix_reuse_ratio)
        if reuse_count:
            reused_indices = set(
                int(index)
                for index in rng.choice(count, size=reuse_count, replace=False)
            )
            shared_prefix = make_shared_prefix(
                adapter, common_prefix_len, workload_name, rng
            )
    arrivals = poisson_arrivals(count, arrival_rate, rng)
    records: list[dict[str, Any]] = []
    encoded_prompts: list[list[Any]] = []
    clipped_prefix_lengths: list[int] = []
    for index, (spec, arrival) in enumerate(zip(specs, arrivals, strict=True)):
        marker = random_marker(rng, index)
        intended_reuse = index in reused_indices
        prefix = shared_prefix if intended_reuse else []
        prompt, actual_len, used_prefix_len = build_synthetic_prompt(
            adapter, spec.prompt_len, marker, prefix, rng
        )
        if mode == "synthetic_unique":
            marker_head = marker.split(" Request-ID:", maxsplit=1)[0]
            if marker_head not in prompt[: max(128, len(marker_head) + 8)]:
                raise RuntimeError(
                    f"Unique marker is not at the start of request {workload_name}-{index}"
                )
        if intended_reuse:
            clipped_prefix_lengths.append(used_prefix_len)
        record = {
            "request_id": f"{workload_name}-{mode}-{index:06d}",
            "workload_name": workload_name,
            "workload_mode": mode,
            "request_type": spec.request_type,
            "prompt_len": actual_len,
            "output_len": spec.output_len,
            "prompt": prompt,
            "arrival_time": round(float(arrival), 6),
            "cache_group_id": (
                f"{workload_name}-cache-group-000" if intended_reuse else None
            ),
            "intended_prefix_reuse": intended_reuse,
            "unique_marker": marker,
        }
        if tuple(record) != REQUEST_FIELDS:
            raise RuntimeError("Internal JSON record field order/schema mismatch")
        records.append(record)
        encoded_prompts.append(adapter.encode_full(prompt))
    if clipped_prefix_lengths and min(clipped_prefix_lengths) < common_prefix_len:
        LOGGER.warning(
            "%s: common prefix requested=%d tokens, effective minimum=%d because "
            "short prompts must retain request-specific content",
            workload_name,
            common_prefix_len,
            min(clipped_prefix_lengths),
        )
    return records, encoded_prompts


def load_json_records(path: Path) -> list[Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Real dataset does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        records: list[Any] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        return records
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        for key in ("data", "conversations", "records"):
            if isinstance(data.get(key), list):
                return list(data[key])
        raise ValueError("JSON object must contain a list under data/conversations/records")
    if not isinstance(data, list):
        raise ValueError("Real dataset must be a JSON list or JSONL")
    return data


def extract_real_text(record: Any) -> tuple[str, str] | None:
    if not isinstance(record, dict):
        return None
    if isinstance(record.get("prompt"), str):
        response = record.get("output", record.get("response", record.get("completion", "")))
        if isinstance(response, str) and record["prompt"].strip() and response.strip():
            return record["prompt"].strip(), response.strip()
    conversation = record.get("conversations", record.get("conversation"))
    if not isinstance(conversation, list):
        return None
    prompt_parts: list[str] = []
    for message in conversation:
        if not isinstance(message, dict):
            continue
        role = str(message.get("from", message.get("role", ""))).lower()
        value = message.get("value", message.get("content"))
        if not isinstance(value, str) or not value.strip():
            continue
        if role in {"human", "user", "system"}:
            prompt_parts.append(f"{role}: {value.strip()}")
        elif role in {"gpt", "assistant"} and prompt_parts:
            return "\n".join(prompt_parts), value.strip()
    return None


def bucket_real_sample(sample: RealSample) -> str | None:
    p, o = sample.prompt_len, sample.output_len
    if 64 <= p <= 512 and 1 <= o <= 128:
        return "short_short"
    if 1024 <= p < 3072 and 1 <= o <= 128:
        return "prefill_heavy_2k"
    if p >= 3072 and 1 <= o <= 128:
        return "prefill_heavy_4k"
    if p <= 512 and 129 <= o < 384:
        return "decode_heavy_256"
    if p <= 512 and o >= 384:
        return "decode_heavy_512"
    if p >= 1024 and o >= 129:
        return "long_long"
    return None


def prepare_real_pools(path: Path, adapter: TokenAdapter) -> dict[str, list[RealSample]]:
    raw_records = load_json_records(path)
    pools: dict[str, list[RealSample]] = {
        "short_short": [],
        "prefill_heavy_2k": [],
        "prefill_heavy_4k": [],
        "decode_heavy_256": [],
        "decode_heavy_512": [],
        "long_long": [],
    }
    skipped = 0
    for index, record in enumerate(raw_records):
        texts = extract_real_text(record)
        if texts is None:
            skipped += 1
            continue
        prompt, response = texts
        sample = RealSample(
            source_id=f"real-{index:08d}",
            prompt=prompt,
            prompt_len=len(adapter.encode_full(prompt)),
            output_len=adapter.output_length(response),
        )
        bucket = bucket_real_sample(sample)
        if bucket is None:
            skipped += 1
        else:
            pools[bucket].append(sample)
    LOGGER.info(
        "Parsed %d real records; accepted=%d, filtered=%d",
        len(raw_records),
        sum(len(pool) for pool in pools.values()),
        skipped,
    )
    empty = [name for name, pool in pools.items() if not pool]
    if empty:
        raise ValueError(
            "Real dataset has no eligible samples for: " + ", ".join(empty)
        )
    return pools


def choose_real_samples(
    workload_name: str,
    definition: dict[str, Any],
    pools: dict[str, list[RealSample]],
    count: int,
    rng: np.random.Generator,
) -> list[tuple[RealSample, str]]:
    if definition.get("type") != "mixture":
        pool = pools[workload_name]
        replace = len(pool) < count
        if replace:
            LOGGER.warning(
                "%s has only %d real samples; sampling with replacement to reach %d",
                workload_name,
                len(pool),
                count,
            )
        indices = rng.choice(len(pool), size=count, replace=replace)
        request_type = classify_request(workload_name)
        return [(pool[int(index)], request_type) for index in indices]

    prefill_ratio = sum(
        float(component["probability"])
        for component in definition["components"]
        if classify_request(str(component["workload"])) == "prefill_heavy"
    )
    prefill_count = round(count * prefill_ratio)
    categories = ["prefill_heavy"] * prefill_count + ["decode_heavy"] * (
        count - prefill_count
    )
    prefill_pool = pools["prefill_heavy_2k"] + pools["prefill_heavy_4k"]
    decode_pool = pools["decode_heavy_256"] + pools["decode_heavy_512"]
    selected: list[tuple[RealSample, str]] = []
    for category in categories:
        pool = prefill_pool if category == "prefill_heavy" else decode_pool
        selected.append((pool[int(rng.integers(len(pool)))], category))
    rng.shuffle(selected)
    return selected


def build_real_records(
    workload_name: str,
    selected: list[tuple[RealSample, str]],
    arrival_rate: float,
    adapter: TokenAdapter,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[list[Any]]]:
    arrivals = poisson_arrivals(len(selected), arrival_rate, rng)
    records: list[dict[str, Any]] = []
    encoded: list[list[Any]] = []
    for index, ((sample, request_type), arrival) in enumerate(
        zip(selected, arrivals, strict=True)
    ):
        records.append(
            {
                "request_id": f"{workload_name}-real_or_sharegpt-{index:06d}",
                "workload_name": workload_name,
                "workload_mode": "real_or_sharegpt",
                "request_type": request_type,
                "prompt_len": sample.prompt_len,
                "output_len": sample.output_len,
                "prompt": sample.prompt,
                "arrival_time": round(float(arrival), 6),
                "cache_group_id": None,
                "intended_prefix_reuse": False,
                "unique_marker": sample.source_id,
            }
        )
        encoded.append(adapter.encode_full(sample.prompt))
    return records, encoded


def longest_common_prefix(left: list[Any], right: list[Any]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def sample_lcp_values(
    encoded: list[list[Any]], max_pairs: int, rng: np.random.Generator
) -> list[int]:
    count = len(encoded)
    if count < 2:
        return [0]
    pairs = [(left, right) for left in range(count) for right in range(left + 1, count)]
    if len(pairs) > max_pairs:
        indices = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[int(index)] for index in indices]
    return [longest_common_prefix(encoded[left], encoded[right]) for left, right in pairs]


def overlap_summary(
    workload_name: str,
    mode: str,
    records: list[dict[str, Any]],
    encoded: list[list[Any]],
    max_pairs: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    values = np.asarray(sample_lcp_values(encoded, max_pairs, rng), dtype=float)
    grouped = sum(bool(record["intended_prefix_reuse"]) for record in records)
    return {
        "workload_name": workload_name,
        "workload_mode": mode,
        "num_requests": len(records),
        "avg_prompt_len": float(np.mean([record["prompt_len"] for record in records])),
        "p50_lcp_tokens": float(np.percentile(values, 50)),
        "p95_lcp_tokens": float(np.percentile(values, 95)),
        "max_lcp_tokens": int(np.max(values)),
        "avg_lcp_tokens": float(np.mean(values)),
        "prefix_reuse_ratio_estimated": grouped / len(records),
    }


def workload_summary(
    records: list[dict[str, Any]], overlap: dict[str, Any]
) -> dict[str, Any]:
    count = len(records)
    return {
        "workload_name": records[0]["workload_name"],
        "workload_mode": records[0]["workload_mode"],
        "num_requests": count,
        "avg_prompt_len": sum(record["prompt_len"] for record in records) / count,
        "avg_output_len": sum(record["output_len"] for record in records) / count,
        "prefill_heavy_ratio": sum(
            record["request_type"] == "prefill_heavy" for record in records
        )
        / count,
        "decode_heavy_ratio": sum(
            record["request_type"] == "decode_heavy" for record in records
        )
        / count,
        "avg_lcp_tokens": overlap["avg_lcp_tokens"],
        "p95_lcp_tokens": overlap["p95_lcp_tokens"],
        "max_lcp_tokens": overlap["max_lcp_tokens"],
    }


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_csv(
    path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]
) -> None:
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


def validate_args(args: argparse.Namespace) -> None:
    if args.num_requests <= 0:
        raise ValueError("--num-requests must be positive")
    if not math.isfinite(args.arrival_rate) or args.arrival_rate <= 0:
        raise ValueError("--arrival-rate must be positive and finite")
    if args.common_prefix_len < 0:
        raise ValueError("--common-prefix-len cannot be negative")
    if not 0 <= args.prefix_reuse_ratio <= 1:
        raise ValueError("--prefix-reuse-ratio must be in [0, 1]")
    if args.lcp_sample_pairs <= 0:
        raise ValueError("--lcp-sample-pairs must be positive")
    if args.mode == "synthetic_unique" and (
        args.common_prefix_len or args.prefix_reuse_ratio
    ):
        LOGGER.warning(
            "synthetic_unique ignores common-prefix-len and prefix-reuse-ratio"
        )
    if args.mode == "synthetic_cache_friendly":
        if args.prefix_reuse_ratio > 0 and args.common_prefix_len == 0:
            raise ValueError(
                "cache-friendly reuse requires --common-prefix-len greater than zero"
            )
        if args.prefix_reuse_ratio == 0:
            LOGGER.warning(
                "synthetic_cache_friendly has prefix-reuse-ratio=0; no cache reuse will occur"
            )
    if args.mode == "real_or_sharegpt" and args.real_data is None:
        raise ValueError(
            "real_or_sharegpt requires --real-data; refusing to label synthetic data as real"
        )


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    workloads = validate_workloads(load_yaml(args.config))
    tokenizer = load_tokenizer(args.tokenizer, args.allow_tokenizer_download)
    adapter = TokenAdapter(tokenizer)
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    real_pools = (
        prepare_real_pools(args.real_data, adapter)
        if args.mode == "real_or_sharegpt" and args.real_data is not None
        else None
    )
    overlap_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    LOGGER.info(
        "Building mode=%s, workloads=%d, requests/workload=%d, seed=%d",
        args.mode,
        len(workloads),
        args.num_requests,
        args.seed,
    )

    for workload_name, definition in workloads.items():
        if args.mode == "real_or_sharegpt":
            assert real_pools is not None
            selected = choose_real_samples(
                workload_name, definition, real_pools, args.num_requests, rng
            )
            records, encoded = build_real_records(
                workload_name, selected, args.arrival_rate, adapter, rng
            )
        else:
            specs = build_specs(
                workload_name, definition, workloads, args.num_requests, rng
            )
            records, encoded = build_synthetic_records(
                workload_name,
                specs,
                args.mode,
                adapter,
                args.common_prefix_len if args.mode == "synthetic_cache_friendly" else 0,
                args.prefix_reuse_ratio if args.mode == "synthetic_cache_friendly" else 0.0,
                args.arrival_rate,
                rng,
            )
        if len(records) != args.num_requests:
            raise RuntimeError(f"Generated incorrect request count for {workload_name}")
        if args.mode != "real_or_sharegpt" and len(
            {record["unique_marker"] for record in records}
        ) != len(records):
            raise RuntimeError(f"Duplicate unique_marker detected in {workload_name}")
        if any(
            records[index]["arrival_time"] >= records[index + 1]["arrival_time"]
            for index in range(len(records) - 1)
        ):
            raise RuntimeError(f"Arrival times are not strictly increasing for {workload_name}")

        overlap = overlap_summary(
            workload_name,
            args.mode,
            records,
            encoded,
            args.lcp_sample_pairs,
            rng,
        )
        if args.mode == "synthetic_unique" and overlap["p95_lcp_tokens"] > 32:
            raise RuntimeError(
                f"Refusing to label {workload_name} synthetic_unique: "
                f"p95 LCP={overlap['p95_lcp_tokens']} tokens exceeds limit 32"
            )
        if (
            args.mode == "synthetic_cache_friendly"
            and args.prefix_reuse_ratio >= 0.5
            and args.common_prefix_len >= 16
            and overlap["p95_lcp_tokens"] < min(16, args.common_prefix_len / 2)
        ):
            raise RuntimeError(
                f"Cache-friendly construction failed for {workload_name}: "
                f"p95 LCP={overlap['p95_lcp_tokens']} is unexpectedly low"
            )

        output = args.output_dir / f"workload_{workload_name}_{args.mode}.jsonl"
        atomic_write_jsonl(output, records)
        overlap_rows.append(overlap)
        summary_rows.append(workload_summary(records, overlap))
        LOGGER.info(
            "Wrote %s | avg_len=%.1f, LCP p50=%.1f p95=%.1f max=%d",
            output,
            overlap["avg_prompt_len"],
            overlap["p50_lcp_tokens"],
            overlap["p95_lcp_tokens"],
            overlap["max_lcp_tokens"],
        )

    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    overlap_path = metrics_dir / f"prefix_overlap_summary_{args.mode}.csv"
    summary_path = metrics_dir / f"workload_summary_{args.mode}.csv"
    atomic_write_csv(overlap_path, OVERLAP_FIELDS, overlap_rows)
    atomic_write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    LOGGER.info("Wrote prefix overlap summary: %s", overlap_path)
    LOGGER.info("Wrote workload summary: %s", summary_path)
    LOGGER.info("Workload construction completed successfully")


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
        LOGGER.exception("Workload construction failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
