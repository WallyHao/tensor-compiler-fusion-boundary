# 结果复核与图表重建

可直接取用[中文结果章节](results.md)、[图 1 PDF](figures/3090-followup.pdf)和[图 2 PDF](figures/layout-contrasts.pdf)。原始包和校验和见[数据归档](../data/README.md)。

## 主张与证据

| 论文主张 | 证据及复核方法 | 适用范围 |
|---|---|---|
| 普通提交 20/20 融合占优，replay 13/20，绝对收益中位数 2.36 和 0.68 µs | [逐配置结果](tables/case-results.csv)选 `3090-followup`，配合[标签计数](tables/label-counts.csv)。补采正式 run 的 1,200 对均通过审计 | 含 16 个新配置和 4 个同期对照。30 个区块是单配置内的重复测量 |
| 融合在补采中每次少一个 kernel，普通提交与 replay 的启动校准斜率分别为 6.54 和 0.91 µs/kernel | 补采正式目录中的 `variants.jsonl` 按 case 匹配 split/fused 的 `static_launch_count`，用 `timings.jsonl` 的 `observed_kernel_names` 复核。校准见同目录 `calibration_fit.json` | 启动校准采用独立的最小 kernel。实际收益同时涉及其他机制 |
| 转置输入的相对降幅下降，绝对节省的变化随模式与行数而异 | [布局对照表](tables/layout-contrasts.csv)按目标和 `control_case_id` 配对。每个布局内部重采样完整计时对，同形配置之间独立抽样 10,000 次计算差值区间 | 4 组 FP32 形状对照，固定 $K=8192$。区间按对照单独计算 |
| R1/R2 长轴降幅减小，N1 边有明确降幅 | [逐配置结果](tables/case-results.csv)选 `long` 与 `rmsnorm`，并用同期 `anchor` 的对应 ID 复核长轴端点 | N1 输出为逆 RMS 因子。论文结论覆盖该切边及片段延迟 |
| 独立复跑仅有一个标签变化 | [逐配置结果](tables/case-results.csv)以 `case_id,mode` 连接 `3090-rerun` 与 `3090-followup`，确认 8/8 点估计同号、7/8 标签一致。复跑 `run.json` 中顺序种子为 2029，240 对均有效 | 4 个诊断配置在原采集后按 replay 点估计接近零选出 |
| 两项模型在测得的验证集上与默认策略打平；但所有 clear label 均为融合且默认始终融合，该比较不能区分策略在拆分占优边上的能力 | [决策指标](tables/decision-metrics.csv)中的 `scope=all`。4090 的两例为 `R2-m16-k512-f16-contig-t` 和 `R2-m16-k2048-f16-contig-t`，regret 分别为 0.622 和 0.324 µs | 明确标签准确率的分母仅含融合或拆分占优样本。平均 regret 使用该方法有预测的全部实测配置 |
| 原网格普通提交高度一致，replay 的标签在不同设备间更易变化 | [逐配置结果](tables/case-results.csv)按 `case_id,mode` 连接原网格。[标签计数](tables/label-counts.csv)中的 `three-GPU subset` 是三卡确切交集 | H800 的 12 个配置在其采集前选定，其中包括 RTX 4090 已知的两个模型错选点 |

在 `followup-full-meas-3090-20261126T082429Z` 正式 run 中，`N1-m256-k8192-f32-contig-t` 的普通提交第 0 个区块对应 `timings.jsonl` 第 662 行 split、第 661 行 fused，均为 `n=472`。拆分代码中的 `triton_red_fused_mean_pow_0` 写出行均方，`triton_poi_fused_add_rsqrt_1` 读回。融合代码中的 `triton_red_fused_add_mean_pow_rsqrt_0` 完成两侧运算。复核入口为 `artifacts/candidates/artifacts/N1-m256-k8192-f32-contig-t/{split,fused}/edge_evidence.json`、对应的 `output_code.py` 和重审后的 `block_audit.csv`。

冻结预测的 UTC 早于目标配对主实验。RTX 4090 对应 `06:11:50.111` 与 `06:12:07.467`，H800 对应 `07:13:55.800` 与 `07:13:56.733`，RTX 3090 补采对应 `08:23:54.274` 与 `08:24:29.790`，日期均为 2026-11-26。重建脚本核对源模型和 split-only 原始计时的 SHA-256。

## 从归档重建

从仓库根目录运行以下 Bash 命令。Python 3.12 用于复核与出表。向该 Python 环境安装 Matplotlib 3.10.8 后可生成 PDF 和 SVG。配对审计、统计及绘图可在 CPU 上完成。

```bash
set -euo pipefail
base="$(mktemp -d /tmp/fb-paper.XXXXXXXX)"
mkdir -p "$base"/{3090,4090,h800,followup}
sha256sum -c data/*.sha256
tar -I zstd -xf data/fb-3090-phase1-runs-20261126.tar.zst -C "$base/3090"
tar -I zstd -xf data/fb-3090-followup-20261126.tar.zst -C "$base/followup"
tar -xzf data/fb-rtx4090-transfer-20261126.tar.gz -C "$base/4090"
tar -xzf data/fb-h800-transfer-20261126.tar.gz -C "$base/h800"

export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
for run in "$base"/3090/runs/full-meas-* "$base"/4090/runs/full-meas-* \
  "$base"/h800/runs/measure-* "$base"/followup/runs/followup-full-meas-* \
  "$base"/followup/runs/followup-rerun-*; do
    [ -d "$run" ] || continue
    (cd "$run" && sha256sum --quiet -c SHA256SUMS)
    python -m fb verify --run-dir "$run"
    python -m fusionaudit build --run "$run" --out "$base/rebuilt/$(basename "$run")" --strict
done
python paper/build_results.py --sources "$base" --out paper
```

五个 run 的严格审计均为 0 error、0 warning，分别生成 144、144、24、40 和 8 个配置与模式组合。模型细项留在临时重建目录的 `scoring/`，论文表格由[重建脚本](build_results.py)输出。

## 英文论文图件

[英文论文](report.pdf)的五张插图采用下列源文件。

| 图 | 主题 | 数据或绘图源 |
|---|---|---|
| 1 | TorchInductor 切边评估流程 | [FigGenie 结构规格](figures/overview/spec.json)、[可编辑 SVG](figures/overview/overview.svg) |
| 2 | RTX 3090 补采中两种提交模式的绝对融合收益 | [逐配置结果](tables/case-results.csv)的 `3090-followup` 记录 |
| 3 | 同形状下转置与连续输入的收益差 | [布局对照](tables/layout-contrasts.csv) |
| 4 | RTX 3090 与 RTX 4090 的标签联合分布 | [逐配置结果](tables/case-results.csv)的 `3090-base` 和 `4090`，按模式及配置 ID 对齐 |
| 5 | RTX 3090 上两条 FP32 逐元素链的长度扫描 | [逐配置结果](tables/case-results.csv)的 `3090-base` 中 P1/P2 的连续 FP32 记录 |

定量图从已审计的 CSV 直接生成。在仓库根目录使用安装了 Matplotlib 3.10.8 的 Python 运行以下命令。脚本为每张定量图输出 SVG 和 PDF。Typst 用 SVG 排版，独立 PDF 用于查看或其他稿件。

```bash
python paper/build_figures.py
typst compile --font-path paper/template/fonts paper/report.typ paper/report.pdf
```
