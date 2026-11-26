# RTX 3090 phase‑1 原始数据归档

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
sha256sum -c fb-3090-phase1-runs-20261126.tar.zst.sha256
zstd -dc fb-3090-phase1-runs-20261126.tar.zst | tar -xf - -C <target-dir>
# 解包后：<target-dir>/runs/... 与 <target-dir>/analysis/out/...
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
