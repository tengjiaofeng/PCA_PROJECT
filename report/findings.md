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
- **Recommended route:** `p2p_nccl_xpyd`. Missing NIXL/LMCache packages do not
  rule out this single-node route. The example proxy currently lacks the
  `quart` Python dependency, so real PD is structurally available but not yet
  ready to launch in the present environment.
- **Current real-PD status:** not runtime validated; no
  `real_disaggregated_pd` request has been measured. NCCL transport, P/D
  registration and KV correctness remain untested.
- **Colocated/TP4 status:** launch commands passed dry-run validation only. No
  GPU server was started in this stage, so neither baseline is claimed to have
  run successfully yet.
- **Fallback:** run real `colocated_4replica`, use those measured results to
  validate/calibrate the stage model, and retain `simulated_pd` for P:D ratio
  results. Do not relabel the trace-driven results as real PD.

Evidence: `outputs/logs/real4gpu/disagg_support_check.txt` and
`outputs/logs/real4gpu/fallback_experiment_plan.md`.
<!-- END AUTO REAL4GPU -->
