# RTX 3090 补采执行记录（plan/followup-3090）

在自有 **NVIDIA GeForce RTX 3090** 上完成 [plan/followup-3090](../plan/followup-3090/README.md) 的三个工作单：
20 个配置（16 新 + 4 同期对照）的三策略编译、配对采集、冻结预测、诊断、复跑与严格审计。
本轮**未出现拆分获益，也未出现默认错选**：direct 20/20 明确为 `fuse`，replay 13 `fuse` /
3 `equivalent` / 4 `uncertain`；冻结的 3090 源模型在所有明确标签上全部选对。

## 1. 主机与环境

与 phase-1 同一台机器、同一软件栈，未更换设备：

| 项 | 值 |
|---|---|
| GPU | `NVIDIA GeForce RTX 3090`（sm_86、82 SM、6 MiB L2、24 GB） |
| UUID | `GPU-86a3c91f-c06f-1539-983d-ed541bafc759` |
| 驱动 | 595.112.07 |
| 栈 | `python 3.12.15 / torch 2.6.0+cu124 / triton 3.2.0` |

环境锁 `runs/followup-env-3090-20261126T082237Z`（`device_check.pass=true`，
name/UUID 与 `nvidia-smi` 一致）。NixOS 主机按 [3090-runbook.md §0](3090-runbook.md#0-一次性环境)
设置 `LD_LIBRARY_PATH=/run/opengl-driver/lib`、`TRITON_LIBCUDA_PATH`、`CC`/`CXX`；
`FB_EXPECT_DEVICE="RTX 3090"`。

校准（本轮，与 phase-1 快照对照）：

| 量 | 本轮 `followup-calib-3090-20261126T082237Z` | phase-1 `full-meas-3090-20261126T041621Z` |
|---|---|---|
| L direct | 6.484 µs/kernel | 6.500 µs/kernel |
| L replay | 0.916 µs/kernel | 0.913 µs/kernel |
| BW main | 836.0 GB/s（512/256/128 MiB） | 836.9 GB/s（同尺寸） |

正式采集内另有一次校准（`full-meas…/calibration_fit.json`）：6.539 / 0.912 µs，BW 836.2 GB/s。

## 2. 运行清单

| run | 层 | 关键产物 |
|---|---|---|
| `runs/followup-env-3090-20261126T082237Z` | 环境 | `environment/*.json` |
| `runs/followup-calib-3090-20261126T082237Z` | 校准 | `calibration.jsonl` + `calibration_fit.json` |
| `runs/followup-pilot-3090-20261126T081808Z` | 编译试点 | 3 试点 × 3 策略 |
| `runs/followup-cand-3090-20261126T081857Z` | 编译全量 | 20 × 3 = 60 变体 |
| `runs/followup-pilot-meas-3090-20261126T082252Z` | 测量试点 | 30 对（两阶段暂停+恢复） |
| `runs/followup-transfer-prediction-20261126T082320Z` | 冻结预测 | `predictions.json`、`split_timings.jsonl` |
| `runs/followup-full-meas-3090-20261126T082429Z` | 正式采集 | **1200 对 / 2400 timing 行** |
| `runs/followup-rerun-3090-20261126T082700Z` | 复跑（seed 2029） | 4 配置 × 2 模式 × 30 对 |
| `analysis/out/followup-full-meas-3090-20261126T082429Z` | 审计 | 40 结果、0 error/0 warning、7 图 |
| `analysis/out/followup-rerun-3090-20261126T082700Z` | 复跑审计 | 8 结果、0 error/0 warning |

## 3. 语料接入（01-candidates）

`plan/followup-3090/cases.csv` 是唯一配置来源，导出 20 行 `cases.jsonl`
（组、形状、输入 stride、`control_id`、`pilot`、sha256；与 CSV 逐行一致）。

- **新模板 N1**：`t = mean(x.square(), dim=1)` → `y = rsqrt(t + 1e-5)`，fp32 累加，
  `t`/输出形状 `(M,)`，逻辑节省 `8*M` 字节，容差 `(2e-4, 1e-4)`；零输入参考 `1/sqrt(1e-5)=316.2278`。
- **布局 `transpose`**：逻辑形状 `(M,K)`、物理 stride `(1,M)`（`(K,M)` 连续缓冲的视图），
  沿逻辑轴 1 归约。同一 seed 生成逻辑值再复制进该存储；`iter_inputs` 对每个输入断言
  shape/stride/dtype。
- 切边识别：N1 的消费者是 `rsqrt`，其输入先经 `add(t, eps)`；`strategies._find_edge_fx_nodes`
  改为在消费者的传递输入里找**最近**匹配 producer_ops 的节点，实测 realize 命中
  `aten.mean.dim`（`fx node mean -> buffer buf1`），即边 `t` 本身。
- 接入面：编译入口 / 独立 worker / `CandidateStore` / `FusionBoundaryProvider` 同一份显式清单；
  `build_corpus()`、`--all` 与既有 72 case ID 行为不变（`manifest/phase1-72-cases.jsonl` 逐字节未变）。

编译结果（`followup-cand-3090-20261126T081857Z`）：**60/60 valid，240/240 校验，`check-run problems=[]`**。
split 全部物化 `t`（fp32，×(M,)）；`unrelated_partition_change=false` ×60；
`default_maps_to=fused` ×20。kernel 数：
N1 与 M=256 的长归约 1/2/1（default/split/fused），其余长归约与转置组 2/3/2 ——
后者的多阶段归约有两策略共有的前置 kernel（`shared_kernel_count=1`）。

## 4. 采集（02-collection）

```bash
export PYTHONPATH=$PWD/src FB_RUN_ROOT=$PWD/runs
export FB_EXPECT_DEVICE="RTX 3090" FB_EXPECT_UUID=GPU-86a3c91f-c06f-1539-983d-ed541bafc759

# 0) 编译（写 followup 清单 + 20×3 变体；PILOT=1 只编 3 个试点）
bash scripts/run_followup.sh

# 1) 环境 + 校准
"$PY" -m fb env       --run-root "$FB_RUN_ROOT" --run-id env-... \
  --expect-device "RTX 3090" --expect-uuid "$FB_EXPECT_UUID"
"$PY" -m fb calibrate --run-root "$FB_RUN_ROOT" --run-id calib-... \
  --expect-device "RTX 3090" --expect-uuid "$FB_EXPECT_UUID"

# 2) 测量试点：3 个 pilot=yes，两阶段（先暂停再恢复）
CASES="R1-m16-k16384-f32-contig-t,R2-m16-k8192-f32-transpose-t,N1-m16-k8192-f32-contig-t"
"$PY" -m fb pilot --run-root "$FB_RUN_ROOT" --run-id "$PILOT" --provider "$CAND" \
  --expect-device "RTX 3090" --expect-uuid "$FB_EXPECT_UUID" --cases "$CASES" \
  --blocks 5 --calibrate --stop-after-blocks 2
"$PY" -m fb pilot ... (同一 RUN_ID，去掉 --stop-after-blocks)

# 3) 冻结预测（必须在正式配对前，只测 split）
"$PY" scripts/validate_transfer.py freeze \
  --models analysis/out/full-meas-3090-20261126T041621Z/models.json \
  --calibration "$FB_RUN_ROOT/$CALIB" --candidates "$FB_RUN_ROOT/$CAND" \
  --uuid "$FB_EXPECT_UUID" --output "$FB_RUN_ROOT/transfer-prediction"

# 4) 正式采集：显式传 20 个 ID，30 block，direct+replay
ids=$(python -c "import json;print(','.join(json.loads(l)['case_id'] for l in open('$CAND/cases.jsonl') if l.strip()))")
"$PY" -m fb collect --purpose full --run-root "$FB_RUN_ROOT" --run-id "$FULL" \
  --provider "$CAND" --expect-device "RTX 3090" --expect-uuid "$FB_EXPECT_UUID" \
  --cases "$ids" --blocks 30 --modes direct,replay --calibrate

# 5) 诊断（计时全部完成后，≤4 个配置）与复跑（新 run_id + 不同 seed）
"$PY" -m fb profile --run-dir "$FB_RUN_ROOT/$FULL" --cases "$DIAG" --iters 10
"$PY" -m fb collect --purpose full --run-root "$FB_RUN_ROOT" --run-id "$RERUN" \
  --provider "$CAND" --expect-device "RTX 3090" --expect-uuid "$FB_EXPECT_UUID" \
  --cases "$DIAG" --blocks 30 --modes direct,replay --seed 2029 --calibrate

# 6) 封存 + 分析 + 评分
"$PY" -m fb seal   --run-dir "$FB_RUN_ROOT/$FULL"
"$PY" -m fb verify --run-dir "$FB_RUN_ROOT/$FULL"
"$PY" -m fusionaudit build --run "$FB_RUN_ROOT/$FULL" --out "analysis/out/$FULL" --strict
"$PY" scripts/validate_transfer.py score-followup \
  --predictions "$PRED/predictions.json" \
  --target-results "analysis/out/$FULL/results.jsonl" --target-run "$FB_RUN_ROOT/$FULL" \
  --history-results analysis/out/full-meas-3090-20261126T041621Z/results.jsonl \
  --output "$PRED/transfer-comparison.json"
```

| 阶段 | 结果 |
|---|---|
| 测量试点 | **30/30 对**，0 error/污染，两模式；两阶段恢复不重复计入 |
| 正式 | **1200/1200 对**（20 case × 2 mode × 30 block），2400 timing 行、200 checks（全通过）、600 校准、130 遥测、200 profile 行，integrity PASS |
| 复跑（seed 2029，4 配置） | **240/240 对**，integrity PASS |
| 审计 | 40 结果，**0 error / 0 warning**；复跑 8 结果同样 0/0 |
| 排除/失败 | 无（`cases.jsonl` 20 行全部 paired，`verification.md` 无 failed/error/contaminated 行） |

## 5. 收益与标签（03-analysis）

`run.json` 记账：`expected_case_count=20`、`planned_template_counts={"R1":8,"R2":8,"N1":4}`、
`expected_blocks=30`。`B = T_split − T_fused`（µs），整对 bootstrap 10,000 次（seed 2026），
标签口径 ±2%。

| 组 | direct | replay |
|---|---|---|
| anchor (R1/R2, K=8192, contig) | 4 `fuse` | 2 `fuse` / 2 `equivalent` |
| long (K=16384/32768) | 8 `fuse` | 5 `fuse` / 3 `uncertain` |
| layout (transpose) | 4 `fuse` | 4 `fuse` |
| rmsnorm (N1) | 4 `fuse` | 2 `fuse` / 1 `equivalent` / 1 `uncertain` |

| case_id | group | direct rel% | label | replay rel% | label | k(def/split/fused) |
|---|---|---|---|---|---|---|
| `R1-m16-k8192-f32-contig-t` | R1 | 37.34 | fuse | 0.59 | equivalent | 1/2/1 |
| `R1-m256-k8192-f32-contig-t` | R1 | 11.93 | fuse | 7.72 | fuse | 1/2/1 |
| `R2-m16-k8192-f32-contig-t` | R2 | 37.64 | fuse | 0.56 | equivalent | 1/2/1 |
| `R2-m256-k8192-f32-contig-t` | R2 | 11.52 | fuse | 7.28 | fuse | 1/2/1 |
| `R1-m16-k8192-f32-transpose-t` | layout | 24.74 | fuse | 2.74 | fuse | 2/3/2 |
| `R1-m256-k8192-f32-transpose-t` | layout | 10.89 | fuse | 5.79 | fuse | 2/3/2 |
| `R2-m16-k8192-f32-transpose-t` | layout | 25.25 | fuse | 2.79 | fuse | 2/3/2 |
| `R2-m256-k8192-f32-transpose-t` | layout | 11.05 | fuse | 5.94 | fuse | 2/3/2 |
| `R1-m16-k16384-f32-contig-t` | long | 24.82 | fuse | 2.41 | uncertain | 2/3/2 |
| `R1-m16-k32768-f32-contig-t` | long | 24.18 | fuse | 1.77 | uncertain | 2/3/2 |
| `R1-m256-k16384-f32-contig-t` | long | 7.04 | fuse | 5.07 | fuse | 1/2/1 |
| `R1-m256-k32768-f32-contig-t` | long | 3.86 | fuse | 3.05 | fuse | 1/2/1 |
| `R2-m16-k16384-f32-contig-t` | long | 24.46 | fuse | 2.23 | uncertain | 2/3/2 |
| `R2-m16-k32768-f32-contig-t` | long | 24.48 | fuse | 4.30 | fuse | 2/3/2 |
| `R2-m256-k16384-f32-contig-t` | long | 6.52 | fuse | 4.45 | fuse | 1/2/1 |
| `R2-m256-k32768-f32-contig-t` | long | 3.66 | fuse | 2.74 | fuse | 1/2/1 |
| `N1-m16-k4096-f32-contig-t` | rmsnorm | 37.61 | fuse | 0.22 | equivalent | 1/2/1 |
| `N1-m16-k8192-f32-contig-t` | rmsnorm | 37.01 | fuse | 1.25 | uncertain | 1/2/1 |
| `N1-m256-k4096-f32-contig-t` | rmsnorm | 19.91 | fuse | 12.04 | fuse | 1/2/1 |
| `N1-m256-k8192-f32-contig-t` | rmsnorm | 12.15 | fuse | 7.73 | fuse | 1/2/1 |

要点：direct 下融合恒优，收益随 K 增大而下降（M=256 的 `R1/R2-k32768` 仅 3.7–3.9%）；
转置布局与同形连续对照同向（K=8192 时 24.7–25.3% vs 37.3–37.6%，转置组收益更低）；
N1 的新边（row-mean → inverse-RMS）与归约边同层，direct 收益 12–38%。
replay 下启动开销被消除，近零收益集中在 K=4096/8192 的 M=16 行与部分长归约行。

## 6. 冻结预测与评分

`runs/followup-transfer-prediction-20261126T082320Z/predictions.json`
（`frozen_utc=2026-11-26T08:23:54Z`，早于正式采集 `started_utc=08:24:29Z`）。
使用归档的 3090 源模型（`analysis/out/full-meas-3090-20261126T041621Z/models.json` 的
`full_fits`：两项 `a=0.386/0.983`，三项 +`c=0.150`，字节阈值 `τ=63`）与本轮校准，
在 20 个配置上做独立 split-only 采样，20×2 全部可用，无 timing error。

评分（`score-followup`，分母为本轮有效配置）：

| mode | method | 覆盖率 | 明确标签 | 选对 | 符号准确率 | mean/max regret µs |
|---|---|---|---|---|---|---|
| direct | model_2term | 1.00 | 20 | 20 | 1.00 | 0 / 0 |
| direct | model_2term_common / 3term | 0.60 (12/20) | 12 | 12 | 1.00 | 0 / 0 |
| direct | byte_threshold (τ=63) | 1.00 | 20 | 20 | 1.00 | 0 / 0 |
| direct | default_inductor | 1.00 | 20 | 20 | 1.00 | 0 / 0 |
| direct | always_fused | 1.00 | 20 | 20 | 1.00 | 0 / 0 |
| direct | always_split | 1.00 | 20 | 0 | 0.00 | 2.34 / 3.18 |
| replay | model_2term | 1.00 | 13 | 13 | 1.00 | 0 / 0 |
| replay | model_2term_common / 3term | 0.60 (12/20) | 8 | 8 | 1.00 | 0 / 0 |
| replay | byte_threshold / default / always_fused | 1.00 | 13 | 13 | 1.00 | 0 / 0 |
| replay | always_split | 1.00 | 13 | 0 | 0.00 | 0.68 / 1.37 |

三项/两项-common 的 0.60 覆盖率来自 `x3` 只对资源齐全子集可用（18 个变体有 split/fused
各 1 或 2 kernel 的资源元数据；8 个 case 为单 kernel 归约，占位缺失 → 保留 null 与原因）。

## 7. 对照同 ID 历史快照（phase-1）

`anchor` 4 行与本轮 `control_id` 同 ID，同机同栈：

| case_id | mode | phase-1 rel% / label | 本轮 rel% / label |
|---|---|---|---|
| `R1-m16-k8192-f32-contig-t` | direct | 36.81 / fuse | 37.34 / fuse |
| `R1-m256-k8192-f32-contig-t` | direct | 11.98 / fuse | 11.93 / fuse |
| `R2-m16-k8192-f32-contig-t` | direct | 37.15 / fuse | 37.64 / fuse |
| `R2-m256-k8192-f32-contig-t` | direct | 12.16 / fuse | 11.52 / fuse |
| `R1-m16-k8192-f32-contig-t` | replay | 0.77 / equivalent | 0.59 / equivalent |
| `R1-m256-k8192-f32-contig-t` | replay | 8.11 / fuse | 7.72 / fuse |
| `R2-m16-k8192-f32-contig-t` | replay | 1.46 / uncertain | 0.56 / equivalent |
| `R2-m256-k8192-f32-contig-t` | replay | 7.88 / fuse | 7.28 / fuse |

replay 绝对值低 ~0.2–0.9 pp；`R2-m16-k8192` replay 从 `uncertain` 落到 `equivalent`（都在 ±2% 带内）。

## 8. 复跑（seed 2029）

4 个诊断配置各 direct/replay 30 对：sign 与本次一致（8/8），rel 差 ≤0.8 pp，
仅 `R2-m16-k8192` replay 由 `equivalent`(0.56%) 变为 `uncertain`(1.05%)——同一 ±2% 边界的抖动，
两份结果都保留（`analysis/out/followup-rerun-3090-20261126T082700Z`）。

## 9. 诊断（profiler）

按“优先 direct/replay 拆分占优、默认错选；否则取各组近零 CI 配置”的顺序，本轮无可用的
前两类，故选每组 replay 相对收益最接近 0 的一条：

`R2-m16-k8192-f32-contig-t`、`R1-m16-k32768-f32-contig-t`、
`R1-m16-k8192-f32-transpose-t`、`N1-m16-k4096-f32-contig-t`。

`fb profile --iters 10` 逐 kernel 记录：`torch.profiler` 给 kernel 时间与次数，
编译元数据给寄存器/shared/grid/block；**硬件计数器（DRAM/L2、achieved occupancy）本机无
Nsight Compute 权限**，按协议记 `null` + 原因，并生成
`artifacts/profiles/ncu-commands.sh` 供有权限时补测。

## 10. 归档与复核

原始数据与派生分析（排除可重建的编译缓存）打包为
`data/fb-3090-followup-20261126.tar.zst`，含外部 sha256 与内部 `SHA256SUMS`。
解包后按 [data/README.md](../data/README.md) 复核：

```bash
sha256sum -c data/fb-3090-followup-20261126.tar.zst.sha256
mkdir -p /tmp/fb-followup && zstd -dc data/fb-3090-followup-20261126.tar.zst | tar -xf - -C /tmp/fb-followup
(cd /tmp/fb-followup/runs/followup-full-meas-3090-20261126T082429Z && sha256sum -c SHA256SUMS)
python -m fusionaudit build --run /tmp/fb-followup/runs/followup-full-meas-3090-20261126T082429Z \
  --out /tmp/fb-followup/audit --strict
python -m fusionaudit trace --run /tmp/fb-followup/runs/followup-full-meas-3090-20261126T082429Z \
  --case-id N1-m256-k8192-f32-contig-t --mode direct
```

可追溯实例：`N1-m256-k8192-f32-contig-t / direct` —— 变体证据
`runs/followup-cand-3090-20261126T081857Z/artifacts/N1-m256-k8192-f32-contig-t/{default,split,fused}/`，
split 的 `edge_evidence.json` 记录 `t` 由 `triton_red_fused_*` 写出、由 `triton_poi_fused_add_rsqrt_*`
读回，`default_maps_to=fused`；`timings.jsonl` 中 block 0–29 两策略各一行，`us_per_call` 可由
`elapsed_ms*1000/n` 复算。

## 11. 结论与边界

- 本轮**扩展了网格而未改变结论**：长归约（K≤32768）、转置输入与 RMSNorm 逆 RMS 边在 3090 上
  仍是融合占优；未出现可报告的拆分获益点，因此没有可标注的符号翻转区间。
- 默认策略与冻结的两项模型在所有 20 个 direct 明确标签上选对，round 内
  `always_split` 的代价上界为 3.18 µs/次调用。
- 未覆盖：硬件计数器（无权限）；`x3` 只在 12/20 配置可用；同网格的相邻 K 之间未出现反向标签，
  故“翻转边界”仍待更长 K 或不同 dtype/布局。
