# Findings

本文件只记录由 measured 或明确标注 simulated 的结果，并保留对应证据路径与适用范围。

<!-- BEGIN AUTO COLOCATED INTERFERENCE -->
## Colocated Prefill–Decode interference

Data source: measured vLLM offline-proxy request metrics. Mixed-workload statistics below include only decode-heavy requests; the decode-only comparator is reweighted to the same 256/512-token output-length mix.

| Workload | Prefill ratio | Decode samples P95 TPOT | vs decode-only | Decode samples P95 TTFT | vs decode-only |
|---|---:|---:|---:|---:|---:|
| mixed_30p70d | 30% | 175.57 ms | 2.22× | 35565.95 ms | 5.67× |
| mixed_50p50d | 50% | 228.07 ms | 2.88× | 51604.99 ms | 8.23× |
| mixed_70p30d | 70% | 280.47 ms | 3.54× | 65408.24 ms | 10.43× |

- **Decode tail interference:** P95 TPOT worsens monotonically as the prefill-heavy ratio increases from 30% to 70%.
- **Queueing effect:** decode-request P95 TTFT also rises monotonically, showing that long prefills delay admission/scheduling before decoding.
- **Why PD separation matters:** colocated prefill kernels consume scheduling and GPU execution capacity needed by decode token steps. Separate pools can isolate decode tail latency and let the scheduler provision prefill and decode resources independently.
- **Scope:** these measurements use a 200-request offline burst and first-output proxies. They provide controlled evidence of interference, not a claim that the same multipliers hold for every online arrival process.

<!-- END AUTO COLOCATED INTERFERENCE -->

<!-- BEGIN AUTO REAL4GPU -->
## Real 4GPU online-serving validation

- **Capability check (not a performance experiment):** editable vLLM
  `0.18.1rc1.dev264+ge31915063.cu124` contains `P2pNcclConnector`, its connector
  engine/factory integration, and the complete
  `disaggregated_serving_p2p_nccl_xpyd` shell/proxy example.
- **P2P NCCL outcome:** the proxy and P/D engines started and NCCL communicator
  initialization succeeded, but the first KV tensor transfer timed out both
  with direct P2P and with P2P disabled plus SHM enabled. This is retained as
  `measured_attempt_failed`; the P2P route is retired from the formal experiment.
- **NIXL 1P1D smoke:** the isolated `vllm-nixl` environment retained the local
  editable vLLM and successfully completed one 128-token prompt / 64-token
  streaming request. The request is recorded as `real_disaggregated_pd` with
  TTFT 1953.58 ms, TPOT 23.06 ms and total latency 3406.18 ms.
- **KV data-plane evidence:** the Decode log reports a passed NIXL compatibility
  check and one successful 16 MiB transfer in 41.70 ms (383.71 MB/s). This
  validates the single-node NIXL/UCX Prefill-to-Decode path for 1P1D.
- **Scope:** one cold 1P1D smoke validates functionality, not performance or a
  fixed-4GPU ratio comparison. It was therefore followed by dedicated
  1P3D/2P2D/3P1D topology smoke tests before loaded experiments.

The subsequent fixed-4GPU smoke tests also completed successfully:

| NIXL mode | TTFT (ms) | TPOT (ms) | Total latency (ms) | KV transfer (ms) | KV throughput (MB/s) |
|---|---:|---:|---:|---:|---:|
| 1P3D | 469.74 | 23.01 | 1919.63 | 49.25 | 324.85 |
| 2P2D | 467.25 | 22.99 | 1915.93 | 44.88 | 356.54 |
| 3P1D | 511.83 | 22.93 | 1956.46 | 48.93 | 327.02 |

All three rows are successful `real_disaggregated_pd` streaming requests with
one logged 16 MiB NIXL transfer. These single-request values validate each
topology but are not used to rank P:D ratios; pilot and loaded runs are needed.
- **Colocated/TP4 status:** launch commands passed dry-run validation only. No
  GPU server was started in this stage, so neither baseline is claimed to have
  run successfully yet.
- **Fallback:** run real `colocated_4replica`, use those measured results to
  validate/calibrate the stage model, and retain `simulated_pd` for P:D ratio
  results. Do not relabel the trace-driven results as real PD.

Evidence: `outputs/logs/real4gpu/p2p_nccl_failure_summary.md`,
`outputs/logs/real4gpu/nixl_support_check.txt`, and
`outputs/logs/real4gpu/nixl_installation_plan.md`.
<!-- END AUTO REAL4GPU -->

## Topology sensitivity / interconnect limitation

- `nvidia-smi topo -m` reports `SYS` for every GPU0–GPU3 pair. These A6000s
  have no NVLink/NVSwitch path; traffic traverses PCIe plus the CPU/SMP
  interconnect even though all four GPUs have NUMA affinity 0.
- `nvidia-smi topo -p2p r` and `-p2p w` report `OK` for all GPU0–GPU3 pairs.
  Therefore peer read/write is supported by the driver; `SYS` is an
  interconnect-quality limitation, not proof that P2P is disabled.
- In the first real 1P3D smoke attempt, Prefill completed and both endpoints
  logged `ncclCommInitRank Success`, but the first KV tensor transfer/decode
  response did not complete within 120 seconds. The request is retained as a
  failed `measured_attempt`, not a successful `real_disaggregated_pd` result.
- A second measured attempt set `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=0`,
  `NCCL_DEBUG=INFO`, and one NCCL channel. The same KV data-plane stall remained,
  so continued P2P tuning is outside the formal route. Any future NIXL result
  remains PCIe/SYS-topology-specific and must not be absolutely aligned with
  paper results from NVLink/NVSwitch systems.

Evidence: `outputs/logs/real4gpu/p2p_capability_check.txt`, component logs, and
`outputs/metrics/real4gpu/online_smoke_real_pd_1p3d.csv`.
