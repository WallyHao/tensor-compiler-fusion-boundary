# 4090 操作手册（编译/取证侧）

目标：在租用实例上用**同一版本 Inductor、同一单图入口**完成 default / split / fused
三份产物，并交付两个试点的切边证据。计时由测量负责人执行（接口见
[interface-for-b.md](interface-for-b.md)）。

## 云端执行顺序

先在实例挂载持久卷，设置 `FB_RUN_ROOT` 指向卷内的 runs 目录。
用 `scripts/run_pilot.sh` 生成候选，按输出的 run ID 设置
`FB_CANDIDATE_RUN="$FB_RUN_ROOT/对应的候选运行ID"`，再执行 `scripts/measure_pilot.sh`。
确认 P1 和 R1 的 direct、replay 均取齐配对后，才执行 `scripts/run_corpus.sh`。
将 `FB_CANDIDATE_RUN` 更新到全量候选运行目录，再执行 `scripts/measure_full.sh`。
导出完整测量目录并验证 SHA256SUMS，本地可运行 `fusionaudit build` 重建分析。
命令和配对验收详见[测量手册](measurement-runbook.md)与[分析手册](analysis-usage.md)。

## 0. 一次性环境

```bash
git clone <repo> && cd tensor-compiler-fusion-boundary
git checkout <commit>                     # 记录到 run.json 的就是这个 commit
VENV=.venv bash scripts/setup_env.sh      # 生成 pip-freeze.txt；需要 uv
# 若实例访问 PyPI 很慢：INDEX=https://mirrors.aliyun.com/pypi/simple/ bash scripts/setup_env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

`setup_env.sh` 打印的 torch/triton/CUDA 版本必须与 `requirements.lock` 一致。
GPU 必须显示 `NVIDIA GeForce RTX 4090`；否则停止，换实例（`run.json` 的 GPU 快照
会记录 UUID，实例变化必须开新 run）。

## 1. 冒烟（约 1 分钟）

```bash
export PYTHONPATH=src
.venv/bin/python -m fusionboundary manifest --run runs/smoke
.venv/bin/python -m fusionboundary compile --run runs/smoke \
    --case P1-n1024-f32-contig-t --strategy default,split,fused
.venv/bin/python -m fusionboundary status --run runs/smoke
```

期望：三行 `valid`，split 两个 kernel、default/fused 一个 kernel。

## 2. 试点（两个试点 × 三策略 + 数值校验 + 证据）

```bash
bash scripts/run_pilot.sh
# 或显式 run id：
RUN_ID=pilot-4090-$(date -u +%Y%m%dT%H%M%SZ) bash scripts/run_pilot.sh
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
cat runs/<run_id>/evidence/pilot-evidence.md
.venv/bin/python -m fusionboundary verify --run runs/<run_id> --pilot   # 走 B 的接口重载
```

通过条件（缺一不可）：

1. 三个策略 `status=valid`；split 的 `t_materialized=true`，`t_alloc_dtype=float32`
   （R1）或 `float32`（P1）；
2. fused 的 `fused_kernels_crossing_edge` 段在 `evidence.md` 中列出同一个 kernel 同时
   含生产者与消费者算子，且没有 t 形状的中间 buffer 写→读；
3. `checks_failed=0`（4 组输入，fp32 逐元素 atol 1e-6；fp32 求和 atol 2e-4）；
4. `graph_break_count=0` 且 `graph_count=1`（split 没有引入额外图/Python 调用）；
5. `unrelated_partition_change=false`。

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

## 5. 待 4090 实测量清单

本仓库在本地只能完成到"编译、取证、数值"这一层；下列项**必须**在 4090 上执行后才能
写入报告：

- [ ] 两个试点的 `runs/<run_id>/` （default/split/fused 各一份 `output_code.py`）
- [ ] 试点的 4 组输入数值校验：P1 fp32 atol 1e-6、R1 fp32 atol 2e-4（本机 4060 的
      结果只能作为流程验证，不作为实验数据）
- [ ] split 证据中 t 的写出/读回行号（4090 上的 `output_code.py` 与 sm_89 无关，但缓存
      键、`num_warps`/grid 可能随 SM 数变化，需以该机产物为准）
- [ ] 72 case 的状态表与 `partition_map.json`
- [ ] 默认策略的实际选择（`default_maps_to`）分布：`fused` 多少、`split` 多少、`distinct` 多少
- [ ] 诊断控制的分区/图断裂副作用（`diag_*` 三行）
- [ ] 与 B 的配对计时对齐：同一 `run_id` 下的 `variants.jsonl` 被 B 的 `CandidateStore` 读取

## 6. 本机（无 4090 时）复现

同一套命令在本地 GPU 上也能跑，用来验证流程与取证，只是结果不得当作实验数据
（`--purpose local-smoke-<gpu>` 写入 `run.json`）。非 Debian 系环境需要：

```bash
# NixOS 等没有 /sbin/ldconfig、默认 PATH 里没有 C 编译器的机器
export LD_LIBRARY_PATH=/run/opengl-driver/lib
export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib   # Triton 用它定位 libcuda.so.1
export CC=$(command -v gcc) CXX=$(command -v g++)
export PYTHONPATH=src
.venv/bin/python -m fusionboundary pilot --run runs/pilot-local --purpose local-smoke
```

本仓库已在 RTX 4060 Laptop（sm_89，torch 2.6.0+cu124 / triton 3.2.0）上跑通：72 case 全量
编译、两个试点三种策略、数值校验与代码级取证。4090 上只需重跑同一命令。

## 7. 失败处理

| 现象 | 判定 | 动作 |
|---|---|---|
| `status=unsupported` + "guard never matched" | 该版本调度器接口变了 | 读 `logs/<case>-split.log`，在 `strategies._resolve_gate` 的候选函数里补；记录版本与位置 |
| `status=partition_mismatch` + "expected a single fused kernel" | 默认策略已经物化 t | 把该 case 记为"默认不可融合"，按工作单从默认二选一评价移出 |
| `status=incorrect` | 数值不过 | 帧存 `checks.jsonl` 的 `argmax_abs_err`/种子，用 `verify` 复现 |
| `compile_error` | 编译失败 | 保留 `compile_result.json` + log；不要静默重试 |
| 无 `output_code.py` | 该版本 trace 目录布局不同 | 检查 `config.trace.*` 实际值（`compile_key.json`），修 `forensics.collect_trace_artifacts` |

所有失败都保留行与原因，不要删除。
