# Real-PD fallback plan

Use this plan only after the preferred P2P NCCL route and any explicitly approved
secondary KV-transfer route are unavailable or unstable in an end-to-end smoke test.

1. Run the real `colocated_4replica` online matrix and retain `result_type=real_colocated`.
2. Use its request latency and throughput to check/calibrate the measured stage model.
3. Analyze P:D ratios using `pd_ratio_sim_results*.csv`, retaining
   `result_type=simulated_pd` (or `data_source=simulated` in the older files).
4. Preserve component failure logs under `outputs/logs/real4gpu/`.
5. State explicitly in `report/findings.md` that no real KV-transfer PD result was measured.

This fallback is not an emulation of a working KV connector and must not be renamed
`real_disaggregated_pd`.
