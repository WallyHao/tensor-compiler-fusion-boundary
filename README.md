# Fusion Boundaries · phase 1 (RTX 3090)

在固定版本的 TorchInductor 上，对**同一条合法切边** `t`（单消费者生产者→消费者边）生成
三种编译产物，并交付可复核的代码级证据与数值校验：

| 策略 | 含义 | 主证据 |
|---|---|---|
| `default` | 不干预，记录 Inductor 自己的选择 | 分区签名 + `default_maps_to` |
| `split` | 同一单图入口与配置，只在调度器否决跨该边的垂直融合 | `t` 的 `empty_strided` + kernel A `tl.store` → kernel B `tl.load` |
| `fused` | 允许跨边合并（与 default 同路径，独立缓存） | 单个 kernel 同时含两侧算子，`t` 无物化 |

语料 72 case（P1/P2 各 12，R1/R2 各 24）；试点 `P1-n65536-f32-contig-t` 与
`R1-m256-k512-f32-contig-t`。研究设计与工作单见 `docs/research-plan.md`、
`plan/phase1-3090/`。

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

RTX 4060 Laptop 上的冒烟仅用于验证流程。正式 run 的 GPU 型号、驱动和设备 UUID 写入 `run.json`。

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
