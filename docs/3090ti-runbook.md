# RTX 3090 Ti 操作手册（编译与取证）

在 3090 Ti 主机上用**同一版本 Inductor、同一单图入口**完成 `default` / `split` / `fused`
三份产物，并交付两个试点的切边证据。计时由测量负责人执行（接口见
[interface-for-b.md](interface-for-b.md)）。

## 主机执行顺序

启动装有 3090 Ti 的主机，从其 `~/Workspace` 打开仓库。将 `FB_RUN_ROOT` 设为主机持久盘上的目录。
用 `scripts/run_pilot.sh` 生成候选，按输出的 run ID 设置
`FB_CANDIDATE_RUN="$FB_RUN_ROOT/对应的候选运行ID"`，再执行 `scripts/measure_pilot.sh`。
确认 P1 和 R1 的 direct、replay 均取齐配对后，才执行 `scripts/run_corpus.sh`。
将 `FB_CANDIDATE_RUN` 更新到全量候选运行目录，再执行 `scripts/measure_full.sh`。
备份完整测量目录并验证 SHA256SUMS。可运行 `fusionaudit build` 重建分析。
命令和配对验收详见[测量手册](measurement-runbook.md)与[分析手册](analysis-usage.md)。

## 0. 一次性环境

```bash
cd ~/Workspace/tensor-compiler-fusion-boundary
git rev-parse --short HEAD
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
export FB_RUN_ROOT="$PWD/runs"
export PYTHONPATH="$PWD/src"
```

先确认 `nvidia-smi` 列出 `NVIDIA GeForce RTX 3090 Ti`。`run.json` 会记录 GPU UUID。更换设备后使用新的 run ID，并重新编译和校准。多卡主机先选定设备，再确保 `fb env` 记录的 `nvidia-smi` GPU UUID 与 PyTorch 的设备 UUID 一致。

NixOS 上，如果 `gcc`/`g++` 不在 `PATH`，先运行 `nix-shell -p gcc`，并在该 shell 中继续。Triton 找不到驱动库时，设置以下变量：

```bash
export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
export CC="$(command -v gcc)" CXX="$(command -v g++)"
```

在主机上创建环境并核验实际版本（需要 `uv`）：

```bash
VENV=.venv bash scripts/setup_env.sh
.venv/bin/python -m fb env --run-root "$FB_RUN_ROOT" \
    --run-id "env-3090ti-$(date -u +%Y%m%dT%H%M%SZ)" --expect-device "RTX 3090 Ti"
```

预期为 Python 3.12、PyTorch 2.6.0（CUDA 构建 12.4）、Triton 3.2.0 且 `cuda available True`。3090 Ti 的 CUDA 计算能力是 8.6。 [NVIDIA 设备表](https://developer.nvidia.com/cuda/gpus)

## 1. 冒烟（约 1 分钟）

```bash
SMOKE_RUN="$FB_RUN_ROOT/smoke-$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python -m fusionboundary manifest --run "$SMOKE_RUN"
.venv/bin/python -m fusionboundary compile --run "$SMOKE_RUN" \
    --case P1-n1024-f32-contig-t --strategy default,split,fused
.venv/bin/python -m fusionboundary status --run "$SMOKE_RUN"
```

期望：三行 `valid`，split 两个 kernel、default/fused 一个 kernel。

## 2. 试点（两个试点 × 三策略 + 数值校验 + 证据）

```bash
PILOT_ID="pilot-3090ti-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="$PILOT_ID" bash scripts/run_pilot.sh
```

产物：

```
runs/<run_id>/cases.jsonl                      72 行清单
runs/<run_id>/variants.jsonl                   6 行（2 试点 × 3 策略）
runs/<run_id>/checks.jsonl                     24 行（2 × 3 × 4 组输入）
runs/<run_id>/evidence/pilot-evidence.md       汇总证据（含代码行号）
runs/<run_id>/artifacts/<case>/<strategy>/trace/output_code.py
runs/<run_id>/artifacts/<case>/<strategy>/evidence.md
runs/<run_id>/artifacts/<case>/partition_map.json
```

判定：

```bash
cat "$FB_RUN_ROOT/$PILOT_ID/evidence/pilot-evidence.md"
.venv/bin/python -m fusionboundary verify --run "$FB_RUN_ROOT/$PILOT_ID" --pilot
```

通过条件（缺一不可）：

1. 三个策略 `status=valid`；split 的 `t_materialized=true`，`t_alloc_dtype=float32`
   （R1）或 `float32`（P1）；
2. fused 的 `fused_kernels_crossing_edge` 段在 `evidence.md` 中列出同一个 kernel 同时
   含生产者与消费者算子，且没有 t 形状的中间 buffer 写→读；
3. `checks_failed=0`（4 组输入，fp32 逐元素 atol 1e-6；fp32 求和 atol 2e-4）；
4. `graph_break_count=0` 且 `graph_count=1`（split 没有引入额外图/Python 调用）；
5. `unrelated_partition_change=false`。

编译试点通过后启动测量试点：

```bash
FB_CANDIDATE_RUN="$FB_RUN_ROOT/$PILOT_ID" bash scripts/measure_pilot.sh
```

按[测量手册的试点验收条件](measurement-runbook.md#4-试点p1--r1)核对校准和两种模式的配对结果，再进入全量运行。

## 3. 全量语料（72 case）

```bash
tmux new -s fusion
bash scripts/run_corpus.sh                 # 结果写 runs/corpus-<UTC>/
# 断开后：tmux attach -t fusion
```

单 case 约 6–20 s（3 次编译 + 4 组校验），72 case 预计 10–25 分钟。
中途失败不阻塞其它 case；结束后：

```bash
.venv/bin/python -m fusionboundary status --run runs/corpus-<UTC>
```

表里每个 case 都有 `default/split/fused` 状态和 `maps_to`。交付前跑一次跨表自检：

```bash
.venv/bin/python -m fusionboundary check-run --run runs/corpus-<UTC>
```

它核对：72 个 case、每个 case 三个策略齐全、`(case_id, strategy)` 不重复、各策略产物文件
存在且 `output_code.py` 哈希与 `variants.jsonl` 一致、每个有效变体有 4 条 `mode=direct`
校验行且全部通过。`problems` 非空即不可交付。

## 4. 诊断控制（单独归档，不进主配对）

```bash
bash scripts/run_diag.sh
```

对两个试点分别跑：

| 策略 | 手段 | 预期记录 |
|---|---|---|
| `diag_graph_break` | `torch._dynamo.graph_break()` | `graph_count>1`：额外 Python 调用 |
| `diag_clone` | 消费者前 `.clone()` | 通常仍被融合 → `partition_mismatch`，说明显式拷贝不是切边手段 |
| `diag_no_fusion` | `max_fusion_size=0` | `unrelated_partition_change` 可能为真：全局禁融合会改动无关分区 |

这三项的目的就是**量化控制手段的代价**，它们的失败/副作用本身就是工作单要求的记录。

## 5. 待 RTX 3090 Ti 实测清单

报告引用以下 3090 Ti 主机上的实际产物与测量：

- [ ] 两个试点的 `runs/<run_id>/` （default/split/fused 各一份 `output_code.py`）
- [ ] 试点的 4 组输入数值校验。P1 fp32 atol 1e-6、R1 fp32 atol 2e-4
- [ ] split 证据中 t 的写出/读回行号、实际 `num_warps` 与 grid
- [ ] 72 case 的状态表与 `partition_map.json`
- [ ] 默认策略的实际选择（`default_maps_to`）分布：`fused` 多少、`split` 多少、`distinct` 多少
- [ ] 诊断控制的分区/图断裂副作用（`diag_*` 三行）
- [ ] B 的候选来源、编译产物哈希与实际发射记录一致，校准和配对原始行可复算

## 6. 开发机复现

可用其他 GPU 验证流程与取证。将 `--purpose local-smoke-<gpu>` 写入 `run.json`。

```bash
.venv/bin/python -m fusionboundary pilot --run runs/pilot-local --purpose local-smoke
```

RTX 4060 Laptop 已跑通 72 个配置的编译、两个试点的三种策略、数值校验与代码级取证。正式产物在 3090 Ti 主机上重新生成。

## 7. 失败处理

| 现象 | 判定 | 动作 |
|---|---|---|
| `status=unsupported` + "guard never matched" | 该版本调度器接口变了 | 读 `logs/<case>-split.log`，在 `strategies._resolve_gate` 的候选函数里补；记录版本与位置 |
| `status=partition_mismatch` + "expected a single fused kernel" | 默认策略已经物化 t | 把该 case 记为"默认不可融合"，按工作单从默认二选一评价移出 |
| `status=incorrect` | 数值不过 | 帧存 `checks.jsonl` 的 `argmax_abs_err`/种子，用 `verify` 复现 |
| `compile_error` | 编译失败 | 保留 `compile_result.json` + log；不要静默重试 |
| 无 `output_code.py` | 该版本 trace 目录布局不同 | 检查 `config.trace.*` 实际值（`compile_key.json`），修 `forensics.collect_trace_artifacts` |

所有失败都保留行与原因，不要删除。
