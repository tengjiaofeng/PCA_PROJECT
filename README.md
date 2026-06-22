# 基于 Prefill–Decode 分离的 LLM Serving 并行体系结构与资源调度研究

## 项目目标

本项目研究固定 GPU 资源预算下 Prefill GPU pool 与 Decode GPU pool 的合理配比。项目通过
真实 LLM inference profiling 获得阶段服务时间和资源指标，再用 trace-driven simulation 分析
不同负载、到达率、P:D ratio 与 KV transfer 开销对 TTFT、TPOT、throughput、goodput、排队时间
和 GPU utilization 的影响，最终实现 workload-aware PD ratio selection（WA-PDR）。

本项目是课程设计性质的测量驱动研究，不以完整复现工业级 DistServe 为目标。

## 实验模块

1. **Workload construction**：构造短请求、prefill-heavy、decode-heavy、long-long 和三种混合负载。
2. **Real benchmark**：使用 vLLM（必要时使用 HuggingFace `generate` fallback）测量 TTFT、TPOT、
   total latency、throughput、GPU utilization 与 memory usage。
3. **Stage profiling**：把不同 `prompt_len/output_len` 下的测量结果整理成 prefill/decode profile。
4. **PD simulation**：模拟请求经过 prefill queue、KV transfer、decode queue 的过程。
5. **Ratio sweep**：比较 G=4 与 G=8 下的全部合法 P:D ratio。
6. **WA-PDR**：根据 workload 特征自动选择资源配比。
7. **Sensitivity and plots**：分析 KV transfer 开销并生成课程报告/PPT 可用图表与表格。
8. **Prompt 3.5 / real 4GPU validation**：在四张 GPU 上测量 colocated replicas、
   aggregated TP4，并在当前 vLLM/NIXL 运行时通过验证后测量真实 PD。

## 目录结构

```text
PCA_PROJECT/
├── README.md
├── requirements.txt
├── configs/
│   ├── model.yaml
│   ├── workloads.yaml
│   ├── slo.yaml
│   └── simulation.yaml
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
├── outputs/
│   ├── logs/
│   ├── metrics/
│   ├── figures/
│   └── tables/
└── report/
```

## 环境安装

项目使用已有的 Conda 环境 `vllm` 和 `/home/tjfeng/vllm` 下的 editable vLLM 源码。

```bash
conda activate vllm
cd /home/tjfeng/PCA_PROJECT
python -m pip install -r requirements.txt
```

`requirements.txt` 按课程要求列出 `torch` 和 `vllm`。如果现有 CUDA 匹配的 PyTorch 与 editable
vLLM 已能正常导入，建议不要强制升级它们，以免替换本地编译版本。

默认模型为 `meta-llama/Llama-3.1-8B-Instruct`，可能需要 Hugging Face 访问权限。若模型不可用，
可将 `configs/model.yaml` 中的 `model_name` 和 `tokenizer_name` 替换为本地可访问模型；报告中必须
记录实际模型路径、revision 和模型配置。

## 运行命令

以下命令是后续脚本的统一接口。当前阶段仅建立项目骨架，占位脚本不会生成实验数据。

### 构造 workload trace

```bash
python scripts/01_build_workloads.py \
  --config configs/workloads.yaml \
  --output-dir data/processed \
  --num-requests 200 \
  --seed 42 \
  --mode synthetic_unique \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --arrival-rate 1.0 \
  --common-prefix-len 0 \
  --prefix-reuse-ratio 0.0
```

### 运行真实 benchmark

```bash
python scripts/02_run_vllm_benchmark.py \
  --model-config configs/model.yaml \
  --workload data/processed/workload_mixed_50p50d_synthetic_unique.jsonl \
  --output outputs/metrics/benchmark_mixed_50p50d_synthetic_unique.csv \
  --backend vllm \
  --max-requests 200 \
  --warmup 5 \
  --seed 42
```

```bash
python scripts/03_profile_stage_time.py \
  --model-config configs/model.yaml \
  --output outputs/metrics/stage_profile.csv \
  --backend vllm \
  --seed 42 \
  --repeat 5
```

Stage profiling 对每个矩阵点运行隔离单请求，并在每次请求前清空 prefix cache。CSV 中的
`ttft_ms` 是 vLLM offline first-output proxy；`decode_total_ms = total_latency_ms - ttft_ms`，
`tpot_ms = decode_total_ms / (output_len - 1)`。后二者是 measured timestamps 的派生估计，
不代表在真实 PD 部署中直接观测到的独立 stage wall time。

### 提取统一 clean metrics

```bash
python scripts/04_extract_metrics.py \
  --metrics-dir outputs/metrics \
  --output-dir outputs/metrics \
  --workload-mode synthetic_unique
```

- `clean_benchmark_metrics.csv`：去重后的逐请求指标，供分布图、SLO/goodput 计算使用。
- `workload_level_metrics.csv`：workload 聚合延迟、吞吐、GPU 利用率和峰值显存。
- `clean_stage_profile.csv`：stage profiling 的 repeat mean/std，供 simulator lookup/interpolation。
- `outputs/logs/metrics_warnings.txt`：保留异常检查、重复来源和未删除 outlier 的说明。

### 运行 PD ratio simulation

```bash
python scripts/05_simulate_pd_ratio.py \
  --workload data/processed/workload_mixed_50p50d_synthetic_unique.jsonl \
  --stage-profile outputs/metrics/clean_stage_profile.csv \
  --config configs/simulation.yaml \
  --slo configs/slo.yaml \
  --output outputs/metrics/pd_ratio_sim_results.csv \
  --request-output outputs/metrics/pd_ratio_sim_requests.csv
```

Simulator 使用两个 FCFS earliest-available server pools；prefill 完成事件按实际完成时间进入
decode queue。事件时间戳列（`arrival_time`、`*_start/end`）单位为秒，service/queue/latency
指标单位为毫秒。输入 trace 的 interarrival 形状保持不变，并缩放到配置的目标 req/s。
KV transfer 当前是可重叠、无链路争用的固定延迟。TPOT 来自 isolated decode service proxy，
因此不会随 queue wait 改变；continuous batching 和 decode 并发干扰属于当前模型的有效性边界。

### 生成图表

```bash
python scripts/07_plot_results.py \
  --input outputs/metrics/pd_simulation.csv \
  --output-dir outputs/figures
```

所有正式图必须同时输出 `.png` 和 `.pdf`。

## Prompt 3.5：真实 4GPU online serving 验证

本模块默认使用 `configs/real4gpu.yaml` 中的物理 GPU ID `0,1,2,3`。启动脚本默认
`--dry-run true`，因此先生成并检查命令，不会意外占用 GPU。正式运行前应确认这些 GPU 空闲；
本项目约定执行 `nvidia-smi` 等状态探测命令前先取得用户批准。

### 能力检查与 dry run

能力检查只查看当前 Python 环境的软件包和 editable vLLM 源码，不初始化 CUDA：

```bash
conda activate vllm
cd /home/tjfeng/PCA_PROJECT
python scripts/03b_launch_4gpu_servers.py \
  --config configs/real4gpu.yaml \
  --check-disagg-support
```

P2P NCCL 路线已经完成真实 1P3D 尝试，但 KV data-plane 在 direct P2P 和 SHM fallback
两种配置下均未完成首个传输，因此该路线停止继续调试，结果标记为
`measured_attempt_failed`。证据见
`outputs/logs/real4gpu/p2p_nccl_failure_summary.md`，不能标记为
`real_disaggregated_pd`。

当前正式候选路线为 editable vLLM 自带的 `NixlConnector`。以下检查只读取 Python 包和
源码，不初始化 CUDA：

```bash
python scripts/03b_launch_4gpu_servers.py --check-nixl-support
```

结果写入 `outputs/logs/real4gpu/nixl_support_check.txt`。当前环境能找到 connector、官方文档
和测试，但 `nixl` 与 `lmcache` 均不可导入；安装建议见
`outputs/logs/real4gpu/nixl_installation_plan.md`，其中命令尚未执行。

```bash
python scripts/03b_launch_4gpu_servers.py --mode colocated_4replica --dry-run true
python scripts/03b_launch_4gpu_servers.py --mode aggregated_tp4 --dry-run true
python scripts/03b_launch_4gpu_servers.py --mode real_pd_nixl_1p1d --dry-run true
```

### 启动 4Replica 主 baseline

该模式启动四个 TP=1 的 OpenAI-compatible vLLM server，端口为 8100–8103，客户端在
四个 endpoint 之间执行 round-robin、random 或 least-outstanding routing。

```bash
python scripts/03b_launch_4gpu_servers.py \
  --config configs/real4gpu.yaml \
  --mode colocated_4replica \
  --output-log-dir outputs/logs/real4gpu \
  --stop-existing \
  --dry-run false
```

### 启动 Aggregated TP4 补充 baseline

```bash
python scripts/03b_launch_4gpu_servers.py \
  --config configs/real4gpu.yaml \
  --mode aggregated_tp4 \
  --output-log-dir outputs/logs/real4gpu \
  --stop-existing \
  --dry-run false
```

TP4 使用单个 8200 端口。Llama 8B 能装入单张 A6000，因此 TP4 只是参考，不作为默认
吞吐 baseline。

### NIXL 真实 PD 候选路线

先在隔离环境中安装并验证 NIXL，避免破坏现有 editable vLLM 环境（以下仅为建议，项目脚本
不会自动执行）：

```bash
conda create -n vllm-nixl --clone vllm
conda activate vllm-nixl
python -m pip install nixl
python -c 'import vllm,nixl; print(vllm.__file__); assert vllm.__file__.startswith("/home/tjfeng/vllm/")'
```

安装经用户批准并验证后，只先运行 1P1D smoke（GPU 0 prefill、GPU 1 decode）。1P3D、2P2D、
3P1D 当前只允许 dry-run：

```bash
python scripts/03b_launch_4gpu_servers.py \
  --config configs/real4gpu.yaml \
  --mode real_pd_nixl_1p1d \
  --allow-experimental-pd true \
  --stop-existing \
  --dry-run false
```

NIXL 1P1D 使用 prefill 端口 8500、decode 端口 8600、proxy 端口 8700，以及独立 side-channel
端口。只有 server、proxy health check、P/D 注册和实际 streaming 请求均成功，
结果才能标为 `real_disaggregated_pd`。任何失败都应保留 component log 和
`*_launch_failure.log`。

### 运行 online workload

单次 colocated 示例：

```bash
python scripts/03c_run_online_workload_client.py \
  --config configs/real4gpu.yaml \
  --workload data/processed/workload_mixed_50p50d_synthetic_unique.jsonl \
  --mode colocated_4replica \
  --output outputs/metrics/real4gpu/online_mixed_50p50d_colocated_4replica_r2_c8.csv \
  --routing least_outstanding \
  --arrival-rate 2.0 \
  --concurrency 8 \
  --max-requests 200 \
  --stream true \
  --seed 42
```

客户端调用 `/v1/completions`，设置 `ignore_eos=true`，并使用第一个 non-empty SSE token/chunk
测量 TTFT。最终 usage 中的 completion token 数用于计算 TPOT。warmup 失败时不会开始正式测量；
正式阶段的单请求失败则写入 CSV 而不中断其他请求。

### 自动编排多模式实验

`03f_run_real4gpu_experiments.py` 自动完成“启动并等待健康检查、运行客户端、保存状态、停止
服务、切换模式”。默认只输出计划，不启动服务：

```bash
python scripts/03f_run_real4gpu_experiments.py --preset pilot --execute false
```

确认计划后，运行三个 NIXL ratio 的 20-request mixed-50 pilot：

```bash
python scripts/03f_run_real4gpu_experiments.py --preset pilot --execute true
```

两个 baseline 使用相同 pilot 参数并由独立预设自动切换：

```bash
python scripts/03f_run_real4gpu_experiments.py --preset baseline --execute false
python scripts/03f_run_real4gpu_experiments.py --preset baseline --execute true
```

`formal` 预设包含 4Replica、TP4、NIXL 1P3D/2P2D/3P1D，三个 mixed workload、五个
arrival rate 和三次重复。必须先检查其 225-run 计划，再显式执行：

```bash
python scripts/03f_run_real4gpu_experiments.py --preset formal --execute false
python scripts/03f_run_real4gpu_experiments.py --preset formal --execute true
```

编排器默认跳过已有成功 CSV，失败时停止当前服务并终止后续实验，且在异常或 Ctrl-C 时执行
最终清理。GPU telemetry 默认关闭；只有取得 GPU 监控批准后才同时传入
`--collect-gpu-metrics true --gpu-monitoring-approved true`。运行计划保存在
`outputs/logs/real4gpu/experiment_plan_<preset>.json`，逐项状态保存在
`outputs/metrics/real4gpu/experiment_status_<preset>_<timestamp>.csv`。

对于 `aggregated_tp4` 或真实 PD，替换 `--mode` 和输出文件名：

```bash
python scripts/03c_run_online_workload_client.py \
  --workload data/processed/workload_mixed_50p50d_synthetic_unique.jsonl \
  --mode aggregated_tp4 \
  --output outputs/metrics/real4gpu/online_mixed_50p50d_aggregated_tp4_r2_c8.csv \
  --arrival-rate 2.0 --concurrency 8 --stream true

python scripts/03c_run_online_workload_client.py \
  --workload data/processed/workload_short_short_synthetic_unique.jsonl \
  --mode real_pd_nixl_1p1d \
  --output outputs/metrics/real4gpu/online_smoke_real_pd_nixl_1p1d.csv \
  --arrival-rate 0.5 --concurrency 1 --max-requests 1 --warmup 0 \
  --request-timeout-s 120 --stream true
```

### 并行采集 GPU telemetry

在另一个终端启动采集器，并令 `--run-id` 与客户端显式传入的 run ID 相同，聚合器即可精确
关联。采集器优先用 pynvml；只有 pynvml 不可用时才调用 `nvidia-smi` fallback。

```bash
python scripts/03d_collect_4gpu_metrics.py \
  --gpu-ids 0,1,2,3 \
  --interval-ms 500 \
  --output outputs/metrics/real4gpu/gpu_trace_colocated_mixed50_r2_c8.csv \
  --run-id colocated-mixed50-r2-c8 \
  --mode colocated_4replica \
  --workload-name mixed_50p50d \
  --arrival-rate 2.0 \
  --concurrency 8
```

客户端相应增加 `--run-id colocated-mixed50-r2-c8`。benchmark 结束后用 `Ctrl-C` 停止采样；
采样进程独立运行，失败不会终止 serving benchmark。

### 停止 server、聚合与绘图

launcher 只停止其 PID manifest 中记录的进程，不使用宽泛的 `pkill`：

```bash
python scripts/03b_launch_4gpu_servers.py --stop-only
python scripts/03e_analyze_4gpu_results.py \
  --metrics-dir outputs/metrics/real4gpu
python scripts/07_plot_results.py \
  --real4gpu-summary outputs/metrics/real4gpu/real4gpu_summary.csv \
  --output-dir outputs/figures/real4gpu
```

聚合结果为 `real4gpu_summary.csv` 和 `real4gpu_request_metrics.csv`，报告表为
`outputs/tables/table_real4gpu_summary.md`。图表只在对应 measured 数据充分时生成；不会用空表
或 simulated 数据制作“真实实验”图。

### Real-PD fallback 与 `result_type`

若 P2P NCCL、proxy、NIXL 或其他 KV connector 路线均不可用，执行
`outputs/logs/real4gpu/fallback_experiment_plan.md`：使用真实 4Replica 结果校准 stage/simulator，
然后读取 `pd_ratio_sim_results*.csv` 分析比例。不得把 fallback 改名为真实 PD。

- `real_colocated`：真实四个独立 vLLM replica 的 online measured 请求。
- `real_aggregated_tp`：真实单实例 TP=4 measured 请求。
- `real_disaggregated_pd`：真实 KV transfer 的 P/D 分离请求；必须通过端到端验证。
- `emulated_pd`：明确实现的 P/D emulation，不代表真实 KV-transfer serving。
- `simulated_pd`：trace-driven queueing simulation。
- `measured_attempt_failed`：启动或真实请求已执行、但端到端 PD 未成功的失败尝试；不得纳入
  `real_disaggregated_pd` 性能统计。

## 数据真实性与注意事项

- **真实测量（`measured`）**：仅指成功执行 vLLM/HuggingFace 推理后采集到的请求延迟、吞吐、
  GPU utilization 和显存数据。
- **Trace-driven simulation（`simulated`）**：指基于请求 trace、真实 stage profile、队列模型和
  KV transfer 参数计算的模拟结果，不等同于真实多 GPU PD 部署测量。
- **Fallback（`fallback`）**：真实 benchmark 因模型、驱动或硬件问题无法运行时，用于检查代码
  流程的参数化数据；不得写成 measured，也不得作为主要实验结论。
- 所有输出 CSV 都必须包含 `data_source` 字段，并记录 seed、配置与 run ID。
- benchmark 失败时保留完整错误日志，不填补或伪造缺失测量值。
- 所有随机过程固定 seed；当前统一为 `42`。
- vLLM 可以开启 Automatic Prefix Caching；主实验用 `synthetic_unique` 在 prompt 起始处注入
  request-specific marker，并用 token-level LCP 报告确认 prefix-diverse。补充实验使用
  `synthetic_cache_friendly` 显式控制共享前缀长度、复用比例和 cache group。
- benchmark 的 warmup 完成后会调用 vLLM `reset_prefix_cache()`；这只清除 warmup 请求留下的
  prefix KV，不清除已经预热的 kernel/CUDA graph。正式 measured batch 从空 prefix cache 开始。
- 逐请求 benchmark CSV 保留 `workload_mode`、`cache_group_id`、`intended_prefix_reuse`、
  `unique_marker` 和 vLLM 实际返回的 `num_cached_tokens`。summary 同时报告 cache-hit 请求比例与
  cached prompt-token 比例，便于区分“设计为复用”和“实际发生复用”。
- 任何 GPU 状态探测命令（包括 `nvidia-smi`）执行前需获得用户批准。
