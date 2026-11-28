# Fusion Boundaries

RTX 3090 的源实验以及 RTX 4090、H800 PCIe 的迁移验证已完成。目前的 RTX 3090 补采安排见[执行计划](plan/README.md)，既有数据见[实验归档](data/README.md)。

在固定版本的 TorchInductor 上，对**同一条合法切边** `t`（单消费者生产者→消费者边）生成
三种编译产物，并交付可复核的代码级证据与数值校验：

| 策略 | 含义 | 主证据 |
|---|---|---|
| `default` | 不干预，记录 Inductor 自己的选择 | 分区签名 + `default_maps_to` |
| `split` | 同一单图入口与配置，只在调度器否决跨该边的垂直融合 | `t` 的 `empty_strided` + kernel A `tl.store` → kernel B `tl.load` |
| `fused` | 允许跨边合并（与 default 同路径，独立缓存） | 单个 kernel 同时含两侧算子，`t` 无物化 |

语料 72 case（P1/P2 各 12，R1/R2 各 24）；试点 `P1-n65536-f32-contig-t` 与
`R1-m256-k512-f32-contig-t`。[完整项目计划](docs/research-plan.md) · [第一阶段工作单](plan/phase1-3090/README.md)

## 快速开始

在 3090 主机上先核对 GPU 与工具链，再跑编译和测量试点。[主机操作手册](docs/3090-runbook.md#0-一次性环境)

```bash
VENV=.venv bash scripts/setup_env.sh     # torch==2.6.0 (cu124), triton 3.2.0
export PYTHONPATH=src

# 清单：72 行
.venv/bin/python -m fusionboundary manifest --run runs/pilot

# 两个试点：编译 + 数值校验 + 证据 + 重新加载校验
bash scripts/run_pilot.sh

# 状态表
.venv/bin/python -m fusionboundary status --run runs/pilot-<UTC>

# 全量 72 case（3090，tmux 内）
bash scripts/run_corpus.sh
```

3090 上的完整步骤、判定条件与待实测清单：[docs/3090-runbook.md](docs/3090-runbook.md)。
给测量负责人 B 的接口：[docs/interface-for-b.md](docs/interface-for-b.md)。
产物与取证口径：[docs/artifacts.md](docs/artifacts.md)。
实现选择与理由（为什么是 lowering realize + 调度器否决、为什么独立缓存、诊断控制）：
[docs/design-decisions.md](docs/design-decisions.md)。
当前交付状态、证据摘要与待 3090 实测清单：[docs/status-phase1-compile.md](docs/status-phase1-compile.md)。
一条边的两种编译结果（含切边图与源码片段）：[docs/edge-walkthrough.md](docs/edge-walkthrough.md)。

## 目录

```
src/fusionboundary/
  corpus.py      72 case 清单（case_id、t shape/dtype、逻辑字节、容差、种子）
  templates.py   模板表达式 + 单图入口 + 边规格
  strategies.py  default/split/fused 编译；split 的调度器否决；诊断控制
  forensics.py   解析 output_code.py / IR dump：kernel、buffer、t 的写读、分区比较
  validate.py    固定种子输入、eager/fp64 参考、容差表
  runtime.py     CandidateStore / Candidate（B 的加载与调用接口）
  records.py     run 目录、JSONL、哈希、环境快照
  cli.py         manifest / compile / worker / verify / pilot / status
fb/             B 的测量程序（环境、校准、配对计时、遥测、重试恢复、校验和）
fusionaudit/    C 的原始数据审计、分析与报告生成
tests/           CPU 测试（含合成 fixture，标注 synthetic）
scripts/         环境、试点、全量、诊断
docs/            接口、产物口径、3090 手册
```

## 当前状态

| 项 | 状态 |
|---|---|
| 72 case 清单 | 完成，路径为 `manifest/phase1-72-cases.jsonl`，`tests/test_manifest_sync.py` 防漂移 |
| 参考实现 / 容差 / 数值校验 | 完成，864 条校验行全通过（本机冒烟） |
| default / split / fused | 三个策略独立编译、独立缓存；split = lowering `realize()` + 调度器按物化 buffer 否决 |
| 代码级切边证据（写/读行号、复用别名、分区比较） | 完成，`edge_evidence.json` + `evidence.md` |
| 默认策略取证（分区 + 决策 pass） | 完成，`default_maps_to`、`fusion_decision_stage` |
| 诊断控制（clone / graph break / 全局禁融合） | 完成并归档副作用 |
| B 的加载/调用接口 | 校验并加载归档 wrapper，普通和 replay 模式分别核验发射 |
| 全量 72 case（本机 RTX 4060 冒烟） | **216 变体全部 valid，0 失败校验**，`runs/corpus-4060-local/` |
| 3090 编译（试点 + 全量 + 诊断） | **完成**：试点 6/6 valid；全量 **216/216 valid、864/864 校验**，`check-run` problems=[] |
| 3090 测量（试点 + 全量） | **完成**：试点 **20/20 配对**；全量 **4320/4320 配对**（8640 timing 行），0 error / 0 污染 |
| 3090 分析 | **完成**：`fusionaudit build --strict` → 144 结果，0 error / 0 warning |
| 3090 补采（followup） | **完成**：20 配置（16 新 + 4 对照）编译 60/60 valid、正式 **1200/1200 配对**、冻结预测 + 复跑 + 严格审计 0 error / 0 warning；见 [docs/followup-3090-runbook.md](docs/followup-3090-runbook.md) |

RTX 4060 Laptop 上的冒烟仅用于验证流程。正式 run 的 GPU 型号、驱动和设备 UUID 写入 `run.json`。

## 审阅发现：72-case 语料无法产生负对照；边界在 stress family 中被观测到

本节记录 2026-11-26 的审阅结论。它不改变 `runs/` 下的原始数据，但直接影响论文能主张的结论。

### 现象

- direct 标签在 176 个 case-mode（3090 72 + 4090 72 + H800 12 + 3090 补采 20）中为 **0 个 split**：3090 68 fuse / 3 equivalent / 1 uncertain，4090 68 / 4 / 0，H800 12 / 0 / 0，补采 20 / 0 / 0。来源：`analysis/out/paper-sources/3090/analysis/out/full-meas-3090-20261126T041621Z/report.md`、`analysis/out/4090-transfer-20261126/analysis-4090/report.md`、`analysis/out/paper-sources/scoring/3090-followup.json`。
- `default_maps_to=fused` 72/72。68 个“明确”标签全为 fuse，导致 `always_fused`、`default_inductor`、两项与三项模型的 mean regret 都是 **0 µs**，`byte_threshold` 也只有 0.37 µs。标签退化，模型比较没有区分力。
- 4090 编译语料的 144 个 split kernel 中 `n_spills=0`、`n_regs` 8–28、occupancy 只有 1.0 和 0.5 两档。语料从未接近寄存器/占用率悬崖，融合没有可损失的东西。

### 原因

`split` 的实现是同一张图 + lowering `realize(t)` + 调度器否决跨边合并（`src/fusionboundary/strategies.py` 的 `split_context`）。Producer kernel 的 config 不变。以本地归档的 `R1-m4096-k8192-f32-contig-t` 为例（`runs/corpus-4060-local/artifacts/.../kernels.json`）：

- split 的 reduction kernel：`XBLOCK=1, RBLOCK=2048, num_warps=16, n_regs=28, n_spills=0`
- fused 的 reduction kernel：**完全相同**，只是名字多了 `relu`
- split 只多一个 `triton_poi_fused_relu_1` 和 `t` 的写读

因此 `split` = `fused` 的同一个 producer kernel + 中间物化 + 一次额外 launch。只要融合不让这个 kernel 溢出或掉 occupancy，split 就严格更慢。0/176 不是采样不足，而是候选被定义成了严格劣于 fused 的版本。

### 实测：设想的负对照不成立

在 4060 Laptop（torch 2.6.0+cu124、triton 3.2.0）上直接用 `split_context` 编译、按 200 次稳态调用计时（µs，`B = split − fused`）：

| case | fused / default | guarded split | B |
|---|---:|---:|---:|
| R1 4096×8192 f32 | 616.9 | 618.2 | +1.3 |
| R1 256×512 f32 | 12.83 | 15.82 | +2.99 |
| R1 16×8192 f32 | 12.84 | 15.89 | +3.05 |
| expand 16×8192 f32 | 13.22 | 17.38 | +4.15 |
| expand 256×8192 f32 | 16.90 | 24.90 | +8.00 |
| expand 256×32768 f16 | 64.43 | 77.35 | +12.92 |
| expand 4096×8192 f32 | 1224.0 | 1688.0 | +464 |

其中 expand 指 `t = x.sum(dim=1); y = t[:, None] * x`，即“消费者并行度高于生产者”的候选；四组形状全部是融合更快。R1 4096×8192 的 +1.3 µs 与 3090/4090 的 ~1.3–1.4 µs 一致。

作为对照：把 producer/consumer **分开编译**（各自选 schedule）时，R1 4096×8192 在 4060 上是 split 540.6 µs、fused 616.9 µs，split 反而快 12%。但这不是当前 `split` 的定义，不能直接记为 split 标签；它说明真正值得测的替代是“独立调度”，并且必须在 3090 上复测。2026-11-28 在 4090 上用 pinned config 复测：单独编译 producer 的调度与 fused producer 逐 case 相同（见下节），该 12% 不能复现。

### 复核（2026-11-28，RTX 4090）：第 1 条候选不成立

在租用 RTX 4090（`torch 2.6.0+cu124`、`triton 3.2.0`、GPU UUID
`417fcd0a-3ce5-cc0a-72c6-8820aa90f5cd`）上，用与归档相同的 pinned Inductor config
复核了 `split_independent`：

- **调度完全相同（72/72）**：单独编译 `t = producer(x)` 得到的
  `block/num_warps/num_stages` 与归档 fused producer 逐 case 相同；仅 `n_regs` 差
  −4…+2，occupancy 全同。3090 归档的 split/fused producer 同样 72/72 相同。
- **两图组合严格更慢**：`independent`（producer/consumer 各自编译、两次 Python 调用）
  相对 fused 的 $B$ 为 +1.55 µs（`R1-m4096-k8192-f16`，95% CI [1.47, 1.64]）、
  +3.59 µs（f32，[1.10, 8.03]）、+34.08 µs（`R1-m256-k8192-f32`；fused 只有 34.8 µs，
  第二个 Python 入口主导）。
- 结论：同图 split 已经是 Inductor 对这条边能给出的独立调度；`split_independent` 与它
  逐 kernel 相同，只多一次 Python 入口。4060 上的 12% 在 pinned config 下不能复现。
- 产物：`runs/probe-standalone-producer-72-20261128T074500Z/`（72 case 审计与
  `summary.json`）、`runs/probe-independent-20261128T073500Z/`（3 case × 5 role、
  30 block 配对与 `summary.json`）；复现脚本 `scripts/run_probe_all.sh`、
  `scripts/run_probe.sh`、`scripts/probe_independent.py`。

### 第 2 条（资源悬崖）：已在 4090 上测到 split 占优区

1. ~~新增 `split_independent` 候选~~：已否定，见上。
2. **保留同图 split，构造资源悬崖**：已用新的 stress family 实现，详见
   [docs/cliff-probe-4090.md](docs/cliff-probe-4090.md)。

家族：`t = sum(x, 1)`；`y = t[:, None] * (x + Σ_{s=1..B} roll(x, s, 1))`，`B` 个独立
load 构成 stencil epilogue。同图 split 仍是 `realize(t)` + 调度器否决。

结果（24 个配置，30/60 配对 block，正反顺序交替，bootstrap 95% CI）：
**10 个 split、13 个 fuse、1 个 uncertain**。占优区为 **M ∈ {512, 1024}、K = 8192、
fp32、B ≥ 3**，最深处 `M=1024, B=8`：split 38.42 µs vs fused 66.42 µs，**−28.0 µs
（−72.9%）**。区间外融合仍然领先，`M=4096, B=5` 时 +119.6 µs（fused 快 28.7%）。

机制：fused 的 epilogue 继承归约的 launch config（`XBLOCK=1`、16 warps、grid=M），
而 split 的消费者是独立的点式 kernel（`XBLOCK=1024`、4 warps、grid=M·K/1024）；
逐 kernel 计时显示 epilogue 在 fused 里每个 load 约 5 µs，在 split 里约 0.9 µs。
B ≥ 8 时 `num_load + num_reduction ≥ 10` 再触发 `_reduction_configs` 的
`register_intensive`，把 `RBLOCK` 从 2048 降到 1024/512、warps 上限从 16 降到 8，
而 split 的 producer 始终保留 2048/16。

边界是**非单调**的：M=256 由 launch 成本主导回到 fuse，M ≥ 2048 由行局部性/网格
规模回到 fuse；K=4096 未判定，K=2048 与 fp16 同一 (M,K) 均为 fuse。因此论文不能主张
“融合普遍有害”，只能主张“在 epilogue 并行度被归约 tile 绑定的区域，融合会亏”，并
把该区域的四条边界（M、K、dtype、B）写清楚。

产物：`runs/cliff-20261128T090000Z/`（`summary.json` + 逐配置 json、编译证据、逐 kernel
profiler 时间）；复现：`scripts/run_cliff.sh`、`scripts/probe_cliff.py`、
`scripts/probe_cliff_paired.py`、`scripts/cliff_summary.py`。

## RTX 3090 实测结果（2026-11-26）

主机 GPU 为 **NVIDIA GeForce RTX 3090**（sm_86、82 SM、6 MiB L2、24 GB，UUID
`GPU-86a3c91f-c06f-1539-983d-ed541bafc759`，驱动 595.112.07）；栈为
`python 3.12.15 / torch 2.6.0+cu124 / triton 3.2.0`。全部 run 在 `runs/` 下，环境锁与 GPU
UUID 写入各自 `run.json`。NixOS 主机需按 [3090-runbook.md §0](docs/3090-runbook.md#0-一次性环境)
设置 `LD_LIBRARY_PATH=/run/opengl-driver/lib`、`TRITON_LIBCUDA_PATH` 与 `CC`/`CXX`；
`FB_EXPECT_DEVICE` 用 `"RTX 3090"`。

| run | 产物 | 结果 |
|---|---|---|
| `env-3090-20261126T040400Z` | 环境锁 | `device_check.pass=true`（nvidia-smi 与 torch 的 name/UUID 一致） |
| `pilot-3090-20261126T040436Z` | 两个试点 × 三策略 | 6/6 valid，30/30 校验；split 物化 `t`（fp32），fused 单 kernel |
| `corpus-3090-20261126T040513Z` | 72 case × 3 策略 | **216/216 valid，864/864 校验**；`default_maps_to=fused`×72；split 恒 2 kernel |
| `diag-3090-20261126T041458Z` | 诊断控制 | clone→仍融合；graph_break→2 图/`partition_mismatch`；no_fusion→P1 1k / R1 2k |
| `pilot-meas-20261126T041540Z` | 测量试点 | **20/20 配对**，0 error/污染，两模式，断点恢复 |
| `full-meas-3090-20261126T041621Z` | 正式测量 | **4320/4320 配对**，8640 timing 行全 `quality=ok`，integrity PASS |

标定：`L_direct = 6.50 µs/kernel`、`L_replay = 0.913 µs/kernel`、`BW = 836.9 GB/s`。
分析（`analysis/out/full-meas-3090-20261126T041621Z/`）direct 标签 68 `fuse` / 3
`equivalent` / 1 `uncertain` / 0 `split`；`always_fused`、默认 Inductor 与两/三项模型的
regret 均为 0 µs，`always_split` 平均 10.97 µs；replay 模式多数落回 `equivalent`。

## 测量（B）：配对采集与数据验收

测量程序在 `fb/`，消费 A 的 `runs/<candidate_run>/variants.jsonl` 和归档 wrapper，加载与预热均在计时外。
协议、单位、重试/恢复规则和交付清单见
[docs/measurement-runbook.md](docs/measurement-runbook.md)，与 C 的表契约见
[docs/measurement-interface.md](docs/measurement-interface.md)。

```bash
export PYTHONPATH=$PWD/src
export FB_RUN_ROOT="$PWD/runs"                     # 3090 主机上的持久盘
export FB_CANDIDATE_RUN="$FB_RUN_ROOT/你的候选运行ID"

# 试点：P1+R1，两阶段（先暂停再恢复，验证不重复计入）
FB_CANDIDATE_RUN=$FB_CANDIDATE_RUN bash scripts/measure_pilot.sh

# 正式：脱离终端，全量 72 case，每 case/mode 30 个配对 block
mkdir -p "$FB_RUN_ROOT" && \
FB_CANDIDATE_RUN=$FB_CANDIDATE_RUN setsid nohup bash scripts/measure_full.sh \
  > "$FB_RUN_ROOT/full-launch.log" 2>&1 < /dev/null &

# 只读校验 / 重新封存
.venv/bin/python -m fb verify --run-dir "$FB_RUN_ROOT/<run_id>"
.venv/bin/python -m fb seal   --run-dir "$FB_RUN_ROOT/<run_id>"
```

C 的最小 synthetic 配对样例：`fixtures/measurement-synthetic/`（所有行
`synthetic: true`，仅用于读表与校验，不是测量值）；由
`python -m fb synth-export --out fixtures/measurement-synthetic --force` 重建。
本地夹具：`python -m unittest discover -s tests -p 'test_*collection.py'` 与
`tests/test_telemetry_window.py`。


## 分析（C）

```bash
.venv/bin/python -m fusionaudit selftest
.venv/bin/python -m fusionaudit build --run /path/to/measurement-run \
  --out analysis/out/measurement-run --strict
```

测量目录包含可随原始数据导出的编译证据，复制整目录后可在本地重建报告。
正式结果以 3090 的两个试点通过后的数据为准。[分析手册](docs/analysis-usage.md)
