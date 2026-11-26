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
