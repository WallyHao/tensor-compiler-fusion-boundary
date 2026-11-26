# Fusion Boundaries · phase 1 (RTX 4090)

在固定版本的 TorchInductor 上，对**同一条合法切边** `t`（单消费者生产者→消费者边）生成
三种编译产物，并交付可复核的代码级证据与数值校验：

| 策略 | 含义 | 主证据 |
|---|---|---|
| `default` | 不干预，记录 Inductor 自己的选择 | 分区签名 + `default_maps_to` |
| `split` | 同一单图入口与配置，只在调度器否决跨该边的垂直融合 | `t` 的 `empty_strided` + kernel A `tl.store` → kernel B `tl.load` |
| `fused` | 允许跨边合并（与 default 同路径，独立缓存） | 单个 kernel 同时含两侧算子，`t` 无物化 |

语料 72 case（P1/P2 各 12，R1/R2 各 24）；试点 `P1-n65536-f32-contig-t` 与
`R1-m256-k512-f32-contig-t`。研究设计与工作单见 `docs/research-plan.md`、
`plan/phase1-4090/`。

## 快速开始

```bash
VENV=.venv bash scripts/setup_env.sh     # torch==2.6.0 (cu124), triton 3.2.0
export PYTHONPATH=src

# 清单：72 行
.venv/bin/python -m fusionboundary manifest --run runs/pilot

# 两个试点：编译 + 数值校验 + 证据 + 重新加载校验
bash scripts/run_pilot.sh

# 状态表
.venv/bin/python -m fusionboundary status --run runs/pilot-<UTC>

# 全量 72 case（4090，tmux 内）
bash scripts/run_corpus.sh
```

4090 上的完整步骤、判定条件与待实测清单：[docs/4090-runbook.md](docs/4090-runbook.md)。
给测量负责人 B 的接口：[docs/interface-for-b.md](docs/interface-for-b.md)。
产物与取证口径：[docs/artifacts.md](docs/artifacts.md)。
实现选择与理由（为什么是 lowering realize + 调度器否决、为什么独立缓存、诊断控制）：
[docs/design-decisions.md](docs/design-decisions.md)。
当前交付状态、证据摘要与待 4090 实测清单：[docs/status-phase1-compile.md](docs/status-phase1-compile.md)。
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
docs/            接口、产物口径、4090 手册
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
| 4090 试点/全量 | **待实测**，见 runbook 第 5 节 |

本机结果不进入报告：GPU 型号、驱动与实例 UUID 都写进 `run.json`，分析侧按 run 区分。

## 测量（B）：配对采集与数据验收

测量程序在 `fb/`，消费 A 的 `runs/<candidate_run>/variants.jsonl` 和归档 wrapper，加载与预热均在计时外。
协议、单位、重试/恢复规则和交付清单见
[docs/measurement-runbook.md](docs/measurement-runbook.md)，与 C 的表契约见
[docs/measurement-interface.md](docs/measurement-interface.md)。

```bash
export PYTHONPATH=$PWD/src
export FB_RUN_ROOT=/data/fusion-boundaries/runs    # 持久卷（云实例）
export FB_CANDIDATE_RUN="$FB_RUN_ROOT/你的候选运行ID"

# 试点：P1+R1，两阶段（先暂停再恢复，验证不重复计入）
FB_CANDIDATE_RUN=$FB_CANDIDATE_RUN bash scripts/measure_pilot.sh

# 正式：脱离终端，全量 72 case，每 case/mode 30 个配对 block
cd ~/tensor-compiler-fusion-boundary && mkdir -p "$FB_RUN_ROOT" && \
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
正式结果以 4090 的两个试点通过后的数据为准。[分析手册](docs/analysis-usage.md)
