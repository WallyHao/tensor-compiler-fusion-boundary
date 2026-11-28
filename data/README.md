# GPU 实验原始数据归档

## RTX 3090 第一阶段

本目录存放 2026‑11‑26 在 **NVIDIA GeForce RTX 3090** 主机上跑完 phase‑1 全部流程后的
原始数据快照。仓库根目录的 `runs/` 与 `analysis/out/` 被 `.gitignore` 排除（体积大、且
每个 `(case, strategy)` 的 `TORCHINDUCTOR_CACHE_DIR` 可重生成），因此把除编译缓存外的
全部数据打包入库，作为可复核的原始记录。

```
fb-3090-phase1-runs-20261126.tar.zst        2.1 MiB  （未压缩 101 MiB，15670 个条目）
fb-3090-phase1-runs-20261126.tar.zst.sha256
```

## 校验与解包

```bash
mkdir -p /tmp/fb-evidence/3090
sha256sum -c data/fb-3090-phase1-runs-20261126.tar.zst.sha256
tar -I zstd -xf data/fb-3090-phase1-runs-20261126.tar.zst -C /tmp/fb-evidence/3090
```

包含 `runs/`（`env-3090-…`、`pilot-3090-…`、`corpus-3090-…`、`diag-3090-…`、
`pilot-meas-…`、`full-meas-3090-…`）与 `analysis/out/`（正式分析产物）。**排除**
`*/cache/`（Inductor 编译缓存，可重生成）；`*.log` 因 `.gitignore` 未纳入。

## 内容与结论摘要

主机：`RTX 3090`（sm_86、82 SM、6 MiB L2、24 GB，UUID
`GPU-86a3c91f-c06f-1539-983d-ed541bafc759`，驱动 595.112.07）；栈
`python 3.12.15 / torch 2.6.0+cu124 / triton 3.2.0`。

| run | 关键文件 | 结果 |
|---|---|---|
| `runs/env-3090-20261126T040400Z` | `environment/*.json` | `device_check.pass=true` |
| `runs/pilot-3090-20261126T040436Z` | `variants.jsonl`、`evidence/pilot-evidence.md` | 6/6 valid，30/30 校验 |
| `runs/corpus-3090-20261126T040513Z` | `variants.jsonl`、`checks.jsonl`、`artifacts/` | **216/216 valid，864/864 校验** |
| `runs/diag-3090-20261126T041458Z` | `variants.jsonl` | clone/graph_break/no_fusion 副作用表 |
| `runs/pilot-meas-20261126T041540Z` | `timings.jsonl` | **20/20 配对**，0 error/污染 |
| `runs/full-meas-3090-20261126T041621Z` | `timings.jsonl`、`calibration*.json*`、`blocks/`、`freeze/`、`SHA256SUMS` | **4320/4320 配对**，8640 timing 行，integrity PASS |
| `analysis/out/full-meas-3090-20261126T041621Z` | `report.md`、`results.jsonl`、`models.json`、`audit.json`、`figures/*.svg` | 144 结果，0 error/0 warning；direct 标签 68 `fuse`/3 `equivalent`/1 `uncertain`/0 `split` |

标定（`full-meas-…/calibration_fit.json`）：`L_direct = 6.50 µs/kernel`、
`L_replay = 0.913 µs/kernel`、`BW = 836.9 GB/s`（512/256/128 MiB）。

判定字段（corpus 汇总）：`default_maps_to=fused`×72；split 物化 `t`（fp32，恒 2 kernel）
×72；default/fused 各 1 kernel；`graph_break_count=0`、`unrelated_partition_change=false`
×216；`fusion_decision_stage` = `lowering`×24（P1/P2） / `scheduler`×48（R1/R2）。

流程与判定口径见 `docs/status-phase1-compile.md`、`docs/measurement-status.md`、
`docs/measurement-runbook.md`。

## RTX 3090 补采（followup-3090）

`plan/followup-3090` 的 20 配置（16 新 + 4 同期对照）在同一台 RTX 3090 上的完整原始数据与派生分析，
排除可重建的编译缓存。

```
fb-3090-followup-20261126.tar.zst            953 KiB（6777 个条目）
fb-3090-followup-20261126.tar.zst.sha256
```

含：`runs/followup-env-3090-*`、`followup-calib-3090-*`、`followup-pilot-3090-*`（编译试点）、
`followup-cand-3090-*`（20×3 变体）、`followup-pilot-meas-3090-*`、`followup-transfer-prediction-*`
（冻结预测 + `transfer-comparison.json`）、`followup-full-meas-3090-*`（正式 1200 对）、
`followup-rerun-3090-*`（seed 2029 复跑），以及 `analysis/out/followup-full-meas-3090-*` 与
`analysis/out/followup-rerun-3090-*` 的审计结果与图表。排除 `*/cache/`。

执行记录、命令与结果表见 [docs/followup-3090-runbook.md](../docs/followup-3090-runbook.md)。
复核：

```bash
sha256sum -c data/fb-3090-followup-20261126.tar.zst.sha256
mkdir -p /tmp/fb-followup && zstd -dc data/fb-3090-followup-20261126.tar.zst | tar -xf - -C /tmp/fb-followup
(cd /tmp/fb-followup/runs/followup-full-meas-3090-20261126T082429Z && sha256sum -c SHA256SUMS)
python -m fusionaudit build --run /tmp/fb-followup/runs/followup-full-meas-3090-20261126T082429Z \
  --out /tmp/fb-followup/audit --strict
```

## 跨设备验证归档

| 设备 | 归档 | 正式 run | 覆盖 |
|---|---|---|---|
| RTX 4090 | `fb-rtx4090-transfer-20261126.tar.gz` | `runs/full-meas-4090` | 原 72 配置，4,320 个有效配对 block |
| H800 PCIe | `fb-h800-transfer-20261126.tar.gz` | `runs/measure-h800` | 预选 12 配置（包内 `cases.txt`），720 个有效配对 block |

两包包含环境锁、候选与正确性证据、校准、split-only 原始计时和冻结预测、配对原始数据、严格审计结果及实际使用的 `validate_transfer.py`。编译缓存可根据记录重建。两个正式 run 的严格审计均为 0 error / 0 warning。运行设备与软件栈差异见[完整项目计划](../docs/research-plan.md#阶段责任与交付)。

从仓库根目录开始，并先按上面的步骤解包 3090 源数据。

```bash
mkdir -p /tmp/fb-evidence/{4090,h800}
sha256sum -c data/fb-rtx4090-transfer-20261126.tar.gz.sha256 \
  data/fb-h800-transfer-20261126.tar.gz.sha256
tar -xzf data/fb-rtx4090-transfer-20261126.tar.gz -C /tmp/fb-evidence/4090
tar -xzf data/fb-h800-transfer-20261126.tar.gz -C /tmp/fb-evidence/h800
(cd /tmp/fb-evidence/4090/runs/full-meas-4090 && sha256sum -c SHA256SUMS)
(cd /tmp/fb-evidence/h800/runs/measure-h800 && sha256sum -c SHA256SUMS)
```

可用 Python 3.12 重算源模型在两个目标设备上的覆盖率、符号准确率、regret 和标签迁移。命令读取冻结预测和配对结果，无需 GPU。

```bash
evidence=/tmp/fb-evidence
source_results="$evidence/3090/analysis/out/full-meas-3090-20261126T041621Z/results.jsonl"
export PYTHONPATH="$PWD:$PWD/src"
python scripts/validate_transfer.py compare \
  --predictions "$evidence/4090/transfer-prediction/predictions.json" \
  --source-results "$source_results" \
  --target-results "$evidence/4090/analysis-4090/results.jsonl" \
  --target-run "$evidence/4090/runs/full-meas-4090" \
  --output "$evidence/4090/transfer-comparison.json"
python scripts/validate_transfer.py compare \
  --predictions "$evidence/h800/transfer-prediction/predictions.json" \
  --source-results "$source_results" \
  --target-results "$evidence/h800/analysis-h800/results.jsonl" \
  --target-run "$evidence/h800/runs/measure-h800" \
  --output "$evidence/h800/transfer-comparison.json"
```

两次比较分别包含 144 和 24 个共同 case/mode。H800 输出的 `source_only` 列出其余 120 个仅在源设备覆盖的 case/mode。两个目标设备的预测冻结 UTC 均早于对应的正式配对起始 UTC。预期 direct 明确标签数分别为 68 和 12，两项、三项与默认策略均全部选对。replay 中，RTX 4090 的三项模型选对 25/27，两项与默认策略各选对 27/27。H800 子集上三种方法均选对 7/7。

## RTX 4090 独立调度复核（2026-11-28）

```
fb-probe-4090-20261128.tar.zst              42 KiB（1087 个条目）
fb-probe-4090-20261128.tar.zst.sha256
```

含 `runs/probe-independent-20261128T073500Z`（3 个 case × `producer`/`consumer`/`fused`/
`independent`/`paired` 五个 role，paired 为 30 block、正反顺序交替的配对）与
`runs/probe-standalone-producer-72-20261128T074500Z`（72 个 case 的 standalone producer
编译证据、逐 case 对照与 `summary.json`）。排除 `*/cache/`。

设备：`NVIDIA GeForce RTX 4090`，`torch 2.6.0+cu124`、`triton 3.2.0`、Python 3.12.3，
GPU UUID `417fcd0a-3ce5-cc0a-72c6-8820aa90f5cd`。

结论：单独编译的 producer 与归档 fused producer 在 `block/num_warps/num_stages` 上
**72/72 相同**（`n_regs` 差 −4…+2，occupancy 全同）；两图组合比 fused 慢
+1.55 µs（`R1-m4096-k8192-f16`，95% CI [1.47, 1.64]）、+3.59 µs（f32，[1.10, 8.03]）、
+34.08 µs（`R1-m256-k8192-f32`）。`split_independent` 与现有同图 split 逐 kernel 相同，
只多一次 Python 入口；4060 上的 −12% 在 pinned config 下不能复现。详见仓库根 README 的
「复核」小节。脚本：`scripts/probe_independent.py`、`scripts/run_probe.sh`、
`scripts/run_probe_all.sh`。

复核：

```bash
sha256sum -c data/fb-probe-4090-20261128.tar.zst.sha256
mkdir -p /tmp/fb-probe && zstd -dc data/fb-probe-4090-20261128.tar.zst | tar -xf - -C /tmp/fb-probe
python3 -m json.tool \
  /tmp/fb-probe/runs/probe-standalone-producer-72-20261128T074500Z/summary.json | head -20
```

## 本机 RTX 4060 冒烟、适配器与审阅运行

```
fb-local-4060-20261128.tar.zst              602 KiB（7512 个条目）
fb-local-4060-20261128.tar.zst.sha256
```

含 27 个此前未入库的本机 run：`env-local-*`、`pilot-4060-local`、`corpus-4060-local`
（72 case 冒烟）、`diag-4060-local`、`dryrun-local-*`、`review-*`（审阅期间的 4060 测量）、
`cand-pilot-*`、`pilot-adapter-*`、`ext-gpu-test*`、`kernel-guard-test`、`manifest-20261125`、
`synthetic-c-sample`，以及 `analysis/out/synthetic-fixture-20261125`。排除 `*/cache/` 与
`*.log`。这些 run 只用于流程验证与审阅对照，不进入正式统计。

## 融合资源悬崖（RTX 4090，2026-11-28）

```
fb-cliff-4090-20261128.tar.zst              67 KiB（707 个条目）
fb-cliff-4090-20261128.tar.zst.sha256
```

含 `runs/cliff-20261128T090000Z`：24 个配置的 fused/split 编译证据（`output_code.py`、
逐 kernel 资源、`num_load`/`num_reduction`）、30/60 个配对 block 的 A/B JSON、逐 kernel
`torch.profiler` 时间与 `summary.json`。排除 `*/cache/`。

设备：`NVIDIA GeForce RTX 4090`（128 SM、72 MiB L2），`torch 2.6.0+cu124`、
`triton 3.2.0`。家族、协议、24 行结果表与机制见
[docs/cliff-probe-4090.md](../docs/cliff-probe-4090.md)；复现：
`PYTHON=/path/to/venv/bin/python bash scripts/run_cliff.sh`。

同目录的 `cliff-4090-results.csv` 是 24 行扁平表（两侧时间、$B$、相对收益与 CI、标签、
两侧 kernel 的 block/warps/regs/occupancy、`num_load`、guard/realize 状态），便于不
解包直接复核。

结论：10 个 `split`、13 个 `fuse`、1 个 `uncertain`。占优区为 M ∈ {512, 1024}、
K = 8192、fp32、B ≥ 3；最深处 `M=1024, B=8` 的 split 为 38.42 µs 对 fused 66.42 µs
（−28.0 µs，−72.9%）。区间外融合仍领先（`M=4096, B=5`：+119.6 µs）。
