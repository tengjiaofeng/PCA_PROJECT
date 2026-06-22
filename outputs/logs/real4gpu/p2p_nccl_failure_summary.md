# P2P NCCL real-PD failure summary

- result_type: `measured_attempt_failed`
- connector: `P2pNcclConnector`
- mode: `real_pd_1p3d`
- failure_stage: `KV data-plane send/recv`

## Observed behavior

The proxy, one prefill server, and three decode servers started successfully.
Prefill returned HTTP 200, the decode request was admitted, and both NCCL ranks
reported successful communicator initialization. The first real KV tensor
send/recv did not complete, no streaming token reached the client, and the
request timed out after 120 seconds. The same failure occurred with direct P2P
and with `NCCL_P2P_DISABLE=1`, SHM enabled, and one NCCL channel.

## Excluded causes

1. Model loading failure.
2. HTTP/proxy control-plane failure.
3. Unsupported GPU P2P read/write capability.
4. Insufficient host-memory pool.
5. GPU or host OOM.
6. Client streaming parser failure.

## Retained conclusion

The P2P NCCL control plane can start, but real KV tensor transfer is unusable
with the current editable vLLM, NCCL/CUDA stack, and RTX A6000 PCIe-SYS
topology. It cannot produce valid `real_disaggregated_pd` performance data.
All generated request rows remain failed measured attempts and must not be
reported as successful real-PD results.
