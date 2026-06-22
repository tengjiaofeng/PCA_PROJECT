#!/usr/bin/env python3
"""Launch and stop reproducible four-GPU vLLM serving topologies."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_MODES = (
    "colocated_4replica",
    "aggregated_tp4",
    "real_pd_1p3d",
    "real_pd_2p2d",
    "real_pd_3p1d",
    "real_pd_nixl_1p1d",
    "real_pd_nixl_1p3d",
    "real_pd_nixl_2p2d",
    "real_pd_nixl_3p1d",
)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/real4gpu.yaml"))
    parser.add_argument("--mode", choices=VALID_MODES, default="colocated_4replica")
    parser.add_argument(
        "--output-log-dir", type=Path, default=Path("outputs/logs/real4gpu")
    )
    parser.add_argument(
        "--dry-run", type=str2bool, default=True,
        help="Print/write the launch plan without starting processes (default: true).",
    )
    parser.add_argument("--stop-existing", action="store_true")
    parser.add_argument(
        "--stop-only", action="store_true", help="Stop processes in the PID manifest and exit."
    )
    parser.add_argument(
        "--check-disagg-support", action="store_true",
        help="Only write the disaggregated-prefill capability report.",
    )
    parser.add_argument(
        "--check-nixl-support", action="store_true",
        help="Only write the read-only NIXL/LMCache capability report.",
    )
    parser.add_argument(
        "--allow-experimental-pd", type=str2bool, default=False,
        help="Required to launch a real_pd_* topology after capability checks pass.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"configuration not found: {path}")
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    required = {"model_name", "gpu_ids", "ports", "num_gpus"}
    missing = sorted(required - set(cfg or {}))
    if missing:
        raise ValueError(f"missing config keys: {missing}")
    if len(cfg["gpu_ids"]) != int(cfg["num_gpus"]):
        raise ValueError("num_gpus must equal len(gpu_ids)")
    if len(set(cfg["gpu_ids"])) != len(cfg["gpu_ids"]):
        raise ValueError("gpu_ids must be unique")
    return cfg


def verified_nixl_1p1d_smoke() -> tuple[bool, str]:
    """Require measured end-to-end evidence before enabling larger NIXL ratios."""

    evidence = (
        PROJECT_ROOT
        / "outputs/metrics/real4gpu/online_smoke_real_pd_nixl_1p1d.csv"
    )
    if not evidence.is_file():
        return False, f"missing evidence CSV: {evidence}"
    try:
        with evidence.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        return False, f"cannot read evidence CSV: {exc}"
    for row in rows:
        success = str(row.get("success", "")).strip().lower() in {"true", "1"}
        try:
            output_tokens = int(float(row.get("output_len_actual", "0")))
            ttft_ms = float(row.get("ttft_ms", "nan"))
        except (TypeError, ValueError):
            continue
        if (
            success
            and row.get("serving_mode") == "real_pd_nixl_1p1d"
            and row.get("result_type") == "real_disaggregated_pd"
            and row.get("connector") == "NixlConnector"
            and output_tokens >= 1
            and ttft_ms >= 0
        ):
            return True, str(evidence)
    return False, f"no qualifying successful row in {evidence}"


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def check_vllm_disagg_support(
    config: dict[str, Any], output_path: Path
) -> dict[str, Any]:
    """Inspect P2P-NCCL/NIXL/LMCache files without initializing CUDA."""

    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        version = "not-installed"

    vllm_spec = importlib.util.find_spec("vllm")
    package_root = (
        Path(vllm_spec.origin).resolve().parent
        if vllm_spec and vllm_spec.origin else None
    )
    configured_checkout = Path("/home/tjfeng/vllm")
    if configured_checkout.is_dir():
        vllm_path = configured_checkout
    elif package_root and (package_root.parent / "examples").is_dir():
        vllm_path = package_root.parent
    else:
        vllm_path = package_root

    searchable_roots = []
    if vllm_path:
        searchable_roots = [vllm_path / "vllm", vllm_path / "examples"]
    p2p_matches: list[str] = []
    allowed_suffixes = {".py", ".md", ".sh", ".yaml", ".yml"}
    for root in searchable_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
                continue
            try:
                if "P2pNcclConnector" in path.read_text(
                    encoding="utf-8", errors="ignore"
                ):
                    p2p_matches.append(str(path.resolve()))
            except OSError:
                continue

    online_root = vllm_path / "examples/online_serving" if vllm_path else None
    p2p_example_matches: list[str] = []
    if online_root and online_root.is_dir():
        for path in online_root.rglob("*"):
            lowered = str(path.relative_to(online_root)).lower()
            if "p2p" in lowered and "nccl" in lowered and (
                "disaggregated" in lowered or "disagg" in lowered or "xpyd" in lowered
            ):
                p2p_example_matches.append(str(path.resolve()))
    p2p_matches = sorted(set(p2p_matches))
    p2p_example_matches = sorted(set(p2p_example_matches))
    p2p_connector_found = bool(p2p_matches)
    p2p_examples_found = bool(p2p_example_matches)

    nixl_available = module_available("nixl")
    lmcache_available = module_available("lmcache")
    proxy_dependencies = {
        name: module_available(name) for name in ("aiohttp", "msgpack", "quart", "zmq")
    }
    p2p_runtime_dependencies_importable = all(proxy_dependencies.values())
    if p2p_connector_found and p2p_examples_found:
        recommended_route = "p2p_nccl_xpyd"
    elif nixl_available:
        recommended_route = "nixl_connector"
    elif lmcache_available and nixl_available:
        recommended_route = "lmcache_nixl"
    else:
        recommended_route = "simulated_pd_fallback"

    launch_candidate = (
        recommended_route == "p2p_nccl_xpyd"
        and p2p_runtime_dependencies_importable
    ) or recommended_route in {"nixl_connector", "lmcache_nixl"}
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "vllm_path": str(vllm_path) if vllm_path else None,
        "vllm_version": version,
        "p2p_nccl_connector_found": p2p_connector_found,
        "p2p_nccl_connector_match_files": p2p_matches,
        "p2p_nccl_examples_found": p2p_examples_found,
        "p2p_nccl_example_paths": p2p_example_matches,
        "p2p_proxy_dependencies": proxy_dependencies,
        "p2p_runtime_dependencies_importable": p2p_runtime_dependencies_importable,
        "nixl_importable": nixl_available,
        "lmcache_importable": lmcache_available,
        "recommended_route": recommended_route,
        "real_pd_launch_candidate": launch_candidate,
        "runtime_validated": False,
        "note": (
            "P2P source/example discovery does not validate NCCL transport, GPU topology, "
            "KV correctness, or end-to-end serving. Only a successful streamed smoke test "
            "may produce real_disaggregated_pd results."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["vLLM disaggregated-prefill support check", "=" * 44]
    for key, value in report.items():
        if isinstance(value, (bool, list, dict)):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def check_nixl_lmcache_support(output_path: Path) -> dict[str, Any]:
    """Discover native NIXL/LMCache support without importing CUDA modules."""

    vllm_path = Path("/home/tjfeng/vllm")
    nixl_connector = vllm_path / (
        "vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py"
    )
    native_examples = [
        vllm_path / "docs/features/nixl_connector_usage.md",
        vllm_path / "tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh",
        vllm_path / "tests/v1/kv_connector/nixl_integration/toy_proxy_server.py",
    ]
    lmcache_root = vllm_path / "examples/others/lmcache"
    nixl_importable = module_available("nixl")
    lmcache_importable = module_available("lmcache")
    connector_found = nixl_connector.is_file()
    nixl_examples_found = all(path.is_file() for path in native_examples)
    lmcache_examples = sorted(
        str(path.resolve()) for path in lmcache_root.rglob("*") if path.is_file()
    ) if lmcache_root.is_dir() else []
    if nixl_importable and connector_found:
        next_step = "try_vllm_nixl_1p1d_smoke"
    elif not nixl_importable and connector_found:
        next_step = "create_isolated_env_and_install_nixl"
    elif lmcache_examples and nixl_importable:
        next_step = "create_isolated_env_and_install_lmcache_nixl"
    else:
        next_step = "skip_real_pd_use_simulator"
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "vllm_path": str(vllm_path),
        "nixl_importable": nixl_importable,
        "lmcache_importable": lmcache_importable,
        "nixl_connector_found_in_vllm": connector_found,
        "nixl_connector_path": str(nixl_connector),
        "nixl_examples_found": nixl_examples_found,
        "nixl_example_paths": [str(path) for path in native_examples if path.is_file()],
        "lmcache_examples_found": bool(lmcache_examples),
        "lmcache_example_paths": lmcache_examples[:100],
        "recommended_next_step": next_step,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["NIXL / LMCache support check", "=" * 31]
    for key, value in report.items():
        rendered = (
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (bool, list, dict)) else str(value)
        )
        lines.append(f"{key}: {rendered}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def common_server_args(
    config: dict[str, Any], port: int, tp: int, *, host: str = "127.0.0.1",
    gpu_memory_utilization: float | None = None,
) -> list[str]:
    args = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(config["model_name"]),
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        str(config.get("dtype", "float16")),
        "--max-model-len",
        str(config.get("max_model_len", 8192)),
        "--gpu-memory-utilization",
        str(
            config.get("gpu_memory_utilization", 0.85)
            if gpu_memory_utilization is None else gpu_memory_utilization
        ),
        "--tensor-parallel-size",
        str(tp),
    ]
    if config.get("enable_prefix_caching", False):
        args.append("--enable-prefix-caching")
    else:
        args.append("--no-enable-prefix-caching")
    return args


def pd_ratio(mode: str) -> tuple[int, int]:
    match = re.search(r"_(\d+)p(\d+)d$", mode)
    if not match:
        raise ValueError(f"cannot parse P:D ratio from mode: {mode}")
    return int(match.group(1)), int(match.group(2))


def build_p2p_nccl_plan(
    config: dict[str, Any], mode: str, p_count: int, d_count: int,
    p_ports: list[int], d_ports: list[int], gpu_ids: list[int],
) -> list[dict[str, Any]]:
    """Adapt the current vLLM P2P-NCCL XpYd example to configured GPUs."""

    real_pd = config.get("real_pd", {})
    ports = config["ports"]
    example_dir = Path(real_pd["p2p_nccl_example_dir"])
    proxy_script = Path(real_pd["p2p_nccl_proxy_script"])
    shell_example = Path(real_pd["p2p_nccl_shell_example"])
    if not all(path.exists() for path in (example_dir, proxy_script, shell_example)):
        raise RuntimeError(
            "P2P NCCL example files changed or are missing. Do not guess the API; "
            "inspect the current checkout and update real4gpu.yaml."
        )
    proxy_http_port = int(ports["pd_proxy"])
    proxy_discovery_port = int(ports["pd_discovery_proxy"])
    proxy_source = proxy_script.read_text(encoding="utf-8", errors="ignore")
    if (
        f'port={proxy_http_port}' not in proxy_source
        or str(proxy_discovery_port) not in proxy_source
    ):
        proposal = (
            "The upstream P2P proxy currently hard-codes its HTTP/discovery ports. "
            f"Expected HTTP={proxy_http_port}, discovery={proxy_discovery_port}. "
            "Patch proposal: add argparse options to the upstream proxy or change "
            "configs/real4gpu.yaml to match its current literals. No automatic patch applied."
        )
        raise RuntimeError(proposal)
    p_kv_ports = [int(x) for x in real_pd["prefill_kv_ports"][:p_count]]
    d_kv_ports = [int(x) for x in real_pd["decode_kv_ports"][:d_count]]
    if len(p_kv_ports) != p_count or len(d_kv_ports) != d_count:
        raise ValueError("insufficient P2P NCCL KV ports in real4gpu.yaml")

    plan: list[dict[str, Any]] = [{
        "name": "pd_proxy", "gpu_ids": [], "port": proxy_http_port,
        "health_path": None, "command": [sys.executable, str(proxy_script)],
        "env": {"PYTHONUNBUFFERED": "1"}, "cwd": str(example_dir),
        "source_example": str(shell_example), "proxy_discovery_port": proxy_discovery_port,
    }]

    def server_component(
        role: str, index: int, gpu: int, http_port: int, kv_port: int
    ) -> dict[str, Any]:
        producer = role == "prefill"
        kv_config = {
            "kv_connector": "P2pNcclConnector",
            "kv_role": "kv_producer" if producer else "kv_consumer",
            "kv_buffer_size": str(
                real_pd["prefill_kv_buffer_size"]
                if producer else real_pd["decode_kv_buffer_size"]
            ),
            "kv_port": str(kv_port),
            "kv_connector_extra_config": {
                "proxy_ip": "127.0.0.1",
                "proxy_port": str(proxy_discovery_port),
                "http_port": str(http_port),
                "send_type": str(real_pd.get("p2p_send_type", "PUT_ASYNC")),
                "nccl_num_channels": str(real_pd.get("nccl_num_channels", 16)),
                "mem_pool_size_gb": str(real_pd.get("host_mem_pool_size_gb", 4)),
            },
        }
        memory_util = float(
            real_pd["prefill_gpu_memory_utilization"]
            if producer else real_pd["decode_gpu_memory_utilization"]
        )
        command = common_server_args(
            config, http_port, 1, host="0.0.0.0",
            gpu_memory_utilization=memory_util,
        )
        command.extend([
            "--enforce-eager",
            "--seed", str(config.get("seed", 42)),
            "--max-num-batched-tokens", str(real_pd.get("max_num_batched_tokens", 8192)),
            "--max-num-seqs", str(real_pd.get("max_num_seqs", 256)),
            "--trust-remote-code",
            "--kv-transfer-config", json.dumps(kv_config, separators=(",", ":")),
        ])
        component_env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
        component_env.update({
            str(key): str(value)
            for key, value in (real_pd.get("nccl_env") or {}).items()
        })
        return {
            "name": f"{role}_{index}", "gpu_ids": [gpu], "port": http_port,
            "kv_port": kv_port, "health_path": "/v1/models", "command": command,
            "env": component_env,
            "source_example": str(shell_example),
        }

    for index in range(p_count):
        plan.append(server_component(
            "prefill", index, gpu_ids[index], p_ports[index], p_kv_ports[index]
        ))
    for index in range(d_count):
        plan.append(server_component(
            "decode", index, gpu_ids[p_count + index], d_ports[index], d_kv_ports[index]
        ))
    return plan


def build_nixl_plan(
    config: dict[str, Any], p_count: int, d_count: int,
    p_ports: list[int], d_ports: list[int], gpu_ids: list[int],
) -> list[dict[str, Any]]:
    """Secondary NIXL route; never selected while P2P NCCL is recommended."""

    nixl = config.get("nixl", {})
    side_base = int(nixl.get("side_channel_port_base", 5600))
    kv_cfg = json.dumps({
        "kv_connector": "NixlConnector", "kv_role": "kv_both",
        "kv_buffer_device": nixl.get("kv_buffer_device", "cuda"),
        "kv_load_failure_policy": nixl.get("kv_load_failure_policy", "fail"),
    }, separators=(",", ":"))
    plan: list[dict[str, Any]] = []
    common_env = {str(k): str(v) for k, v in (nixl.get("env") or {}).items()}
    for index in range(p_count):
        cmd = common_server_args(
            config, p_ports[index], 1,
            gpu_memory_utilization=float(nixl.get("gpu_memory_utilization", 0.85)),
        )
        cmd.extend([
            "--enforce-eager", "--block-size", str(nixl.get("block_size", 128)),
            "--kv-transfer-config", kv_cfg,
        ])
        env = dict(common_env)
        env.update({
            "CUDA_VISIBLE_DEVICES": str(gpu_ids[index]),
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(side_base + index),
        })
        plan.append({
            "name": f"prefill_{index}", "gpu_ids": [gpu_ids[index]],
            "port": p_ports[index], "health_path": "/v1/models", "command": cmd,
            "env": env, "source_example": nixl.get("usage_guide"),
        })
    for index in range(d_count):
        gpu = gpu_ids[p_count + index]
        cmd = common_server_args(
            config, d_ports[index], 1,
            gpu_memory_utilization=float(nixl.get("gpu_memory_utilization", 0.85)),
        )
        cmd.extend([
            "--enforce-eager", "--block-size", str(nixl.get("block_size", 128)),
            "--kv-transfer-config", kv_cfg,
        ])
        env = dict(common_env)
        env.update({
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(side_base + p_count + index),
        })
        plan.append({
            "name": f"decode_{index}", "gpu_ids": [gpu],
            "port": d_ports[index], "health_path": "/v1/models", "command": cmd,
            "env": env, "source_example": nixl.get("usage_guide"),
        })
    proxy = Path(nixl.get("proxy_script", ""))
    if proxy is None or "toy_proxy_server.py" not in proxy.name:
        raise RuntimeError("No compatible current-vLLM NIXL XpYd proxy was found")
    if not proxy.is_file():
        raise RuntimeError(f"NIXL proxy script is missing: {proxy}")
    proxy_port = int(config["ports"].get("nixl_proxy", 8700))
    proxy_cmd = [
        sys.executable, str(proxy), "--host", "127.0.0.1", "--port", str(proxy_port),
        "--prefiller-hosts", *("127.0.0.1" for _ in p_ports),
        "--prefiller-ports", *map(str, p_ports),
        "--decoder-hosts", *("127.0.0.1" for _ in d_ports),
        "--decoder-ports", *map(str, d_ports),
    ]
    plan.append({
        "name": "pd_proxy", "gpu_ids": [], "port": proxy_port,
        "health_path": "/healthcheck", "command": proxy_cmd, "env": {},
        "start_after_servers": True, "source_example": nixl.get("usage_guide"),
    })
    return plan


def build_launch_plan(
    config: dict[str, Any], mode: str, recommended_route: str
) -> list[dict[str, Any]]:
    gpu_ids = [int(x) for x in config["gpu_ids"]]
    ports = config["ports"]
    plan: list[dict[str, Any]] = []
    if mode == "colocated_4replica":
        if len(ports["colocated"]) != len(gpu_ids):
            raise ValueError("colocated ports must match gpu_ids")
        for index, (gpu, port) in enumerate(zip(gpu_ids, ports["colocated"])):
            plan.append({
                "name": f"colocated_{index}",
                "gpu_ids": [gpu],
                "port": int(port),
                "health_path": "/v1/models",
                "command": common_server_args(config, int(port), 1),
                "env": {"CUDA_VISIBLE_DEVICES": str(gpu)},
            })
        return plan
    if mode == "aggregated_tp4":
        port = int(ports["aggregated_tp4"])
        plan.append({
            "name": "aggregated_tp4",
            "gpu_ids": gpu_ids,
            "port": port,
            "health_path": "/v1/models",
            "command": common_server_args(config, port, len(gpu_ids)),
            "env": {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids))},
        })
        return plan

    p_count, d_count = pd_ratio(mode)
    if mode.startswith("real_pd_nixl_") and p_count + d_count > len(gpu_ids):
        raise ValueError(f"{mode} requires more than {len(gpu_ids)} GPUs")
    if not mode.startswith("real_pd_nixl_") and p_count + d_count != len(gpu_ids):
        raise ValueError(f"{mode} requires {p_count + d_count} GPUs")
    if mode.startswith("real_pd_nixl_"):
        p_ports = [int(x) for x in ports["nixl_prefill"][:p_count]]
        d_ports = [int(x) for x in ports["nixl_decode"][:d_count]]
    else:
        p_ports = [int(x) for x in ports["pd_prefill"][:p_count]]
        d_ports = [int(x) for x in ports["pd_decode"][:d_count]]
    if len(p_ports) != p_count or len(d_ports) != d_count:
        raise ValueError("insufficient PD ports in configuration")
    if recommended_route == "p2p_nccl_xpyd":
        return build_p2p_nccl_plan(
            config, mode, p_count, d_count, p_ports, d_ports, gpu_ids
        )
    if recommended_route in {"nixl_connector", "lmcache_nixl"}:
        return build_nixl_plan(config, p_count, d_count, p_ports, d_ports, gpu_ids)
    raise RuntimeError(
        "No real-PD launch route is structurally available. Use simulated_pd_fallback; "
        "do not label it real_disaggregated_pd."
    )


def pid_manifest_path(log_dir: Path) -> Path:
    return log_dir / "server_processes.json"


def stop_manifest_processes(log_dir: Path) -> None:
    path = pid_manifest_path(log_dir)
    if not path.is_file():
        print(f"No PID manifest found at {path}; nothing stopped.")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    signaled: list[tuple[int, str]] = []
    for item in reversed(data.get("processes", [])):
        pid = int(item["pid"])
        try:
            cmdline_path = Path(f"/proc/{pid}/cmdline")
            cmdline = cmdline_path.read_bytes().replace(b"\x00", b" ").decode(
                "utf-8", errors="replace"
            )
            if item["name"] == "pd_proxy":
                command = item.get("command", [])
                expected = Path(command[1]).name if len(command) > 1 else "proxy"
            else:
                expected = "vllm.entrypoints.openai.api_server"
            if expected not in cmdline:
                print(
                    f"WARNING: skipped stale/reused pid={pid}; command does not match {expected}",
                    file=sys.stderr,
                )
                continue
            os.killpg(pid, signal.SIGTERM)
            signaled.append((pid, item["name"]))
            print(f"Sent SIGTERM to process group {item['name']} (pgid={pid})")
        except (ProcessLookupError, FileNotFoundError):
            print(f"Process already exited: {item['name']} (pid={pid})")
        except PermissionError as exc:
            print(f"WARNING: cannot stop pid={pid}: {exc}", file=sys.stderr)
    if signaled:
        time.sleep(3)
    for pid, name in signaled:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        try:
            os.killpg(pid, signal.SIGKILL)
            print(
                f"Process group {name} did not exit after SIGTERM; sent SIGKILL "
                f"(pgid={pid})"
            )
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            print(f"WARNING: cannot force-stop pid={pid}: {exc}", file=sys.stderr)
    stopped_path = log_dir / (
        f"server_processes.stopped.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    )
    path.rename(stopped_path)


def health_check(port: int, path: str | None, timeout_s: float = 2.0) -> bool:
    if path is None:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=timeout_s):
                return True
        except (OSError, TimeoutError, ConnectionError):
            return False
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}{path}", timeout=timeout_s
        ) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


def launch(plan: list[dict[str, Any]], config: dict[str, Any], mode: str, log_dir: Path) -> None:
    processes: list[dict[str, Any]] = []
    running: list[tuple[dict[str, Any], subprocess.Popen[str]]] = []
    handles: list[Any] = []
    startup_timeout = float(config.get("server_startup_timeout_s", 1200))
    interval = float(config.get("health_check_interval_s", 2))
    try:
        for item in plan:
            if item.get("start_after_servers"):
                continue
            log_path = log_dir / f"{mode}_{item['name']}.log"
            handle = log_path.open("w", encoding="utf-8")
            handles.append(handle)
            env = os.environ.copy()
            env.update(item["env"])
            proc = subprocess.Popen(
                item["command"], stdout=handle, stderr=subprocess.STDOUT,
                env=env, cwd=item.get("cwd"), start_new_session=True, text=True,
            )
            processes.append({
                "name": item["name"], "pid": proc.pid, "log": str(log_path),
                "command": item["command"],
            })
            running.append((item, proc))
            pid_manifest_path(log_dir).write_text(
                json.dumps({"mode": mode, "state": "starting", "processes": processes}, indent=2) + "\n",
                encoding="utf-8",
            )
        deadline = time.monotonic() + startup_timeout
        server_items = [item for item in plan if not item.get("start_after_servers")]
        while time.monotonic() < deadline:
            exited = [
                (item["name"], proc.returncode)
                for item, proc in running if proc.poll() is not None
            ]
            if exited:
                raise RuntimeError(f"server process exited before health check: {exited}")
            if all(health_check(x["port"], x["health_path"]) for x in server_items):
                break
            time.sleep(interval)
        else:
            raise TimeoutError("server health check timed out; inspect component logs")

        for item in plan:
            if not item.get("start_after_servers"):
                continue
            log_path = log_dir / f"{mode}_{item['name']}.log"
            handle = log_path.open("w", encoding="utf-8")
            handles.append(handle)
            env = os.environ.copy()
            env.update(item["env"])
            proc = subprocess.Popen(
                item["command"], stdout=handle, stderr=subprocess.STDOUT,
                env=env, cwd=item.get("cwd"), start_new_session=True, text=True,
            )
            processes.append({
                "name": item["name"], "pid": proc.pid, "log": str(log_path),
                "command": item["command"],
            })
            running.append((item, proc))
            pid_manifest_path(log_dir).write_text(
                json.dumps({"mode": mode, "state": "starting_proxy", "processes": processes}, indent=2) + "\n",
                encoding="utf-8",
            )
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and not health_check(item["port"], item["health_path"]):
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"PD proxy exited before health check with return code {proc.returncode}"
                    )
                time.sleep(1)
            if not health_check(item["port"], item["health_path"]):
                raise TimeoutError("PD proxy health check timed out")

        manifest = {
            "mode": mode, "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "result_type": "real_disaggregated_pd" if mode.startswith("real_pd_") else (
                "real_aggregated_tp" if mode == "aggregated_tp4" else "real_colocated"
            ),
            "processes": processes,
        }
        pid_manifest_path(log_dir).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"All endpoints healthy. PID manifest: {pid_manifest_path(log_dir)}")
    except Exception as exc:
        for item in reversed(processes):
            try:
                os.killpg(int(item["pid"]), signal.SIGTERM)
            except ProcessLookupError:
                pass
        for _, proc in running:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
                except ProcessLookupError:
                    pass
        for item, (_, proc) in zip(processes, running):
            item["return_code"] = proc.returncode
        failure = log_dir / f"{mode}_launch_failure.log"
        failure.write_text(
            f"{datetime.now(timezone.utc).isoformat()} launch failed\n"
            f"{type(exc).__name__}: {exc}\n"
            f"processes={json.dumps(processes)}\n"
            f"See component logs in {log_dir}\n", encoding="utf-8"
        )
        raise
    finally:
        for handle in handles:
            handle.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    args.output_log_dir.mkdir(parents=True, exist_ok=True)
    if args.stop_existing or args.stop_only:
        stop_manifest_processes(args.output_log_dir)
    if args.stop_only:
        return
    support = check_vllm_disagg_support(
        config, args.output_log_dir / "disagg_support_check.txt"
    )
    nixl_support = check_nixl_lmcache_support(
        args.output_log_dir / "nixl_support_check.txt"
    )
    if args.check_disagg_support:
        print(json.dumps(support, indent=2))
        return
    if args.check_nixl_support:
        print(json.dumps(nixl_support, indent=2))
        return
    is_nixl_mode = args.mode.startswith("real_pd_nixl_")
    if is_nixl_mode:
        route = "nixl_connector"
        if not args.dry_run and not nixl_support["nixl_importable"]:
            raise RuntimeError(
                "NIXL is not installed. No package was installed automatically; "
                "create/approve the isolated vllm-nixl environment first."
            )
        if not args.dry_run and args.mode != "real_pd_nixl_1p1d":
            smoke_verified, smoke_evidence = verified_nixl_1p1d_smoke()
            if not smoke_verified:
                raise RuntimeError(
                    "Larger NIXL ratios require a verified 1P1D end-to-end smoke: "
                    f"{smoke_evidence}"
                )
            print(f"VERIFIED_NIXL_1P1D_EVIDENCE={smoke_evidence}")
        if not args.allow_experimental_pd and not args.dry_run:
            raise RuntimeError("Pass --allow-experimental-pd true for NIXL smoke.")
    elif args.mode.startswith("real_pd_"):
        route = support["recommended_route"]
        if not args.dry_run:
            raise RuntimeError(
                "P2pNcclConnector is retired as a formal real-PD route after a "
                "measured KV data-plane failure. Use a real_pd_nixl_* mode."
            )
        if support["recommended_route"] == "simulated_pd_fallback" and not args.dry_run:
            raise RuntimeError(
                "No structural real-PD route was found. See disagg_support_check.txt; "
                "use the measured-colocated + simulated-PD fallback."
            )
        if not support["real_pd_launch_candidate"] and not args.dry_run:
            raise RuntimeError(
                "The recommended real-PD route is present but its user-space runtime "
                "dependencies are incomplete. No package was installed; inspect "
                "disagg_support_check.txt."
            )
        if not args.allow_experimental_pd and not args.dry_run:
            raise RuntimeError(
                "Real PD is structurally available but not runtime validated. Re-run with "
                "--allow-experimental-pd true after reviewing disagg_support_check.txt."
            )
    else:
        route = support["recommended_route"]
    plan = build_launch_plan(config, args.mode, route)
    serializable_plan = {
        "mode": args.mode,
        "dry_run": args.dry_run,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "components": plan,
    }
    plan_path = args.output_log_dir / f"{args.mode}_launch_plan.json"
    plan_path.write_text(json.dumps(serializable_plan, indent=2) + "\n", encoding="utf-8")
    print(f"MODE={args.mode}")
    print(f"MODEL={config['model_name']}")
    print(f"PREFIX_CACHING_ENABLED={bool(config.get('enable_prefix_caching', False))}")
    if args.mode.startswith("real_pd_"):
        p_count, d_count = pd_ratio(args.mode)
        gpu_ids = [int(x) for x in config["gpu_ids"]]
        print(f"RECOMMENDED_ROUTE={route}")
        print(f"PREFILL_GPUS={gpu_ids[:p_count]}")
        print(f"DECODE_GPUS={gpu_ids[p_count:p_count + d_count]}")
        proxy_key = "nixl_proxy" if is_nixl_mode else "pd_proxy"
        print(f"PROXY_HTTP_PORT={config['ports'][proxy_key]}")
        if route == "p2p_nccl_xpyd":
            print(f"PROXY_DISCOVERY_PORT={config['ports']['pd_discovery_proxy']}")
    for item in plan:
        env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in item["env"].items())
        print(f"[{item['name']}] {env_prefix} {shlex.join(item['command'])}".strip())
    print(f"Launch plan written to {plan_path}")
    if args.dry_run:
        print("Dry run: no server process was started.")
        return
    launch(plan, config, args.mode, args.output_log_dir)


if __name__ == "__main__":
    main()
