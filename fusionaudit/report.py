"""Markdown report renderer with an explicit raw-record trace appendix."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from . import config
from .analysis import case_trace


def render_report(
    audit: dict[str, Any],
    analysis: dict[str, Any],
    figures: Sequence[str],
    out_dir: Path,
) -> str:
    run = audit["run"]
    lines: list[str] = []
    add = lines.append
    title = f"# Fusion Boundaries 单卡分析报告 · {run.get('run_id')}"
    add(title)
    add("")
    if audit["synthetic"]:
        add("> **SYNTHETIC FIXTURE — 本文件中的数值是夹具数据，只用于验证读表、配对、分类、留出和追溯流程，不能作为实验结论引用。**")
        add("")
    add(f"- 用途：{run.get('purpose')}")
    add(f"- 运行状态：`{run.get('status')}`；合成数据：`{str(audit['synthetic']).lower()}`")
    add(f"- 分析版本：`{config.ANALYSIS_VERSION}`；bootstrap：{config.BOOTSTRAP_ITERATIONS} 次整对有放回抽样，seed={config.BOOTSTRAP_SEED}")
    add(f"- 本报告由 `python -m fusionaudit build --run <run_dir> --out <out_dir>` 重建；不含当前时间戳，相同输入应得到相同表图。")
    add("")

    _section_run(audit, add)
    _section_coverage(audit, add)
    _section_audit(audit, add)
    _section_results(analysis, add)
    _section_evaluation(analysis, add)
    _section_features(audit, analysis, add)
    _section_diagnostics(analysis, add)
    _section_figures(figures, add)
    _section_provenance(audit, analysis, add)
    _section_freeze(audit, add)
    add("")
    return "\n".join(lines) + "\n"


def _section_run(audit: dict[str, Any], add) -> None:
    run = audit["run"]
    add("## 1. 运行与数据完整性")
    add("")
    add("| 字段 | 值 |")
    add("|---|---|")
    for key in (
        "run_id",
        "purpose",
        "parent_run",
        "protocol_hash",
        "corpus_hash",
        "code_hash",
        "cloud",
        "instance",
        "image",
        "persistent_volume",
        "started_at",
        "ended_at",
        "seed",
        "owner",
        "status",
    ):
        add(f"| `{key}` | {_md(run.get(key))} |")
    add("")
    add("| 输入文件 | SHA-256 |")
    add("|---|---|")
    for name, digest in audit["files"].items():
        add(f"| `{name}` | `{digest or 'missing'}` |")
    add("")
    cal = audit["calibration"].get("modes", {})
    add("| mode | L (µs/kernel) | BW (bytes/µs) | 来源 |")
    add("|---|---|---|---|")
    for mode, row in cal.items():
        add(f"| {mode} | {_num(row.get('launch_us_per_kernel'))} | {_num(row.get('bandwidth_bytes_per_us'))} | {row.get('launch_source')} / {row.get('bandwidth_source')} |")
    add("")
    add(f"- 期望 block 数：{audit['expected_blocks']}；唯一 case 数：{audit['funnel']['cases_total']}。")
    add("")


def _section_coverage(audit: dict[str, Any], add) -> None:
    funnel = audit["funnel"]
    add("## 2. 覆盖率流程表")
    add("")
    add("| 阶段 | direct | replay |")
    add("|---|---:|---:|")
    add(f"| 期望每 case {funnel['expected_blocks']} 个完整块 | — | — |")
    add(f"| 完整合格 case | {funnel['complete_cases'].get('direct', 0)} | {funnel['complete_cases'].get('replay', 0)} |")
    add(f"| 有配对但不完整 case | {funnel['incomplete_cases'].get('direct', 0)} | {funnel['incomplete_cases'].get('replay', 0)} |")
    add(f"| 合格 block 对合计 | {funnel['qualified_pairs_by_mode'].get('direct', 0)} | {funnel['qualified_pairs_by_mode'].get('replay', 0)} |")
    add("")
    add("| 模板 | 计划 case 数 | 完整 direct | 完整 replay | case 状态 |")
    add("|---|---:|---:|---:|---|")
    templates = sorted(set(funnel["template_counts"]) | {str(case.get("template")) for case in audit["cases"].values()})
    for template in templates:
        planned = funnel["template_counts"].get(template, 0)
        cases = [case for case in audit["cases"].values() if str(case.get("template")) == template]
        complete_direct = sum(1 for case in cases if audit["case_modes"].get((case["case_id"], "direct"), {}).get("complete"))
        complete_replay = sum(1 for case in cases if audit["case_modes"].get((case["case_id"], "replay"), {}).get("complete"))
        statuses = ", ".join(f"{key}:{value}" for key, value in sorted(Counter(str(case.get("planned_status")) for case in cases).items()))
        add(f"| {template} | {planned} | {complete_direct} | {complete_replay} | {statuses} |")
    add("")
    add(f"- timing 行：{funnel['timing_rows']}；通过单行校验：{funnel['timing_rows_valid']}；telemetry：{funnel['telemetry_rows']}；profiles：{funnel['profile_rows']}。")
    add(f"- timing 中记录的 GPU UUID：{', '.join(funnel.get('gpu_uuids') or ['—'])}。")
    inventory = audit.get("case_inventory") or []
    if inventory:
        no_timing = sum(1 for row in inventory if row.get("inventory_status") == "no_timing")
        excluded = sum(1 for row in inventory if row.get("inventory_status") == "excluded")
        add(f"- 清单中无 timing 的 case：{no_timing}；主动排除：{excluded}。逐 case 状态见 `case_inventory.csv`。")
    add("")


def _section_audit(audit: dict[str, Any], add) -> None:
    add("## 3. 审计结果")
    add("")
    summary = audit["summary"]
    add(f"- 错误：{summary['error_count']}；警告：{summary['warning_count']}；提示：{summary['info_count']}。")
    add("")
    if summary["code_counts"]:
        add("| code | count |")
        add("|---|---:|")
        for code, count in sorted(audit["summary"]["code_counts"].items(), key=lambda item: (-item[1], item[0])):
            add(f"| `{code}` | {count} |")
        add("")
    rejected = [row for row in audit["block_audit"] if row["status"] != "selected"]
    if rejected:
        add("### 被拒或未被采用的 attempt（保留原始行号）")
        add("")
        add("| case_id | mode | block | attempt | status | split line | fused line | reason |")
        add("|---|---|---:|---:|---|---:|---:|---|")
        for row in rejected[:24]:
            add(
                f"| {row['case_id']} | {row['mode']} | {row['block']} | {row['attempt'] if row['attempt'] is not None else '—'} | "
                f"{row['status']} | {row['split_line'] or '—'} | {row['fused_line'] or '—'} | {_md(row['reason'])} |"
            )
        if len(rejected) > 24:
            add(f"| … | | | | | | | 另有 {len(rejected) - 24} 行，见 `block_audit.csv` |")
        add("")
    add("- 完整逐行问题清单：`audit_issues.csv`；逐 block 选择记录：`block_audit.csv`；逐 case 的清单左连接（variants/checks/timings/complete）：`case_inventory.csv`；机器可读汇总：`audit.json`。")
    add("- 输入 `results.jsonl` 中已有的 fold 指定不会被采用；分析始终按 case 的 `template` 重建 leave-one-template-out 折。" + (
        f" 本 run 的输入 fold 表含 {len(audit['input_fold_leak']['leaks'])} 个模板泄漏，已拒绝。" if audit["input_fold_leak"]["leaks"] else ""
    ))
    add("")


def _section_results(analysis: dict[str, Any], add) -> None:
    add("## 4. 配对收益与四类标签")
    add("")
    add("口径：$T_{split}$、$T_{fused}$ 为所选 block 的算术平均；$B=\\mathrm{mean}(s_i-f_i)$；相对收益 $B/T_{split}$；区间为整对 block 的 95% bootstrap 区间。标签阈值 ±2%，敏感性表另给 ±1% 与 ±5%。")
    add("")
    direct = [row for row in analysis["results"] if row["mode"] == "direct"]
    replay = [row for row in analysis["results"] if row["mode"] == "replay"]
    _result_table(direct, "direct 模式", add)
    if replay:
        _result_table(replay, "replay 模式（机制对照，不混入主模型）", add)
    labels = Counter(row["label"] for row in direct)
    add("| direct 标签 | count |")
    add("|---|---:|")
    for label in config.LABEL_ORDER:
        add(f"| {label} | {labels.get(label, 0)} |")
    add("")
    by_template: dict[str, Counter] = {}
    for row in direct:
        by_template.setdefault(str(row["template"]), Counter())[row["label"]] += 1
    if by_template:
        add("| template | fuse | split | equivalent | uncertain | total |")
        add("|---|---:|---:|---:|---:|---:|")
        for template, counts in sorted(by_template.items()):
            add(
                f"| {template} | {counts.get('fuse', 0)} | {counts.get('split', 0)} | {counts.get('equivalent', 0)} | "
                f"{counts.get('uncertain', 0)} | {sum(counts.values())} |"
            )
        add("")
    if analysis["sign_flips"]:
        add("### 已测相邻尺寸间的可靠符号翻转")
        add("")
        add("| mode | template | dtype | axis | left case | left label | right case | right label |")
        add("|---|---|---|---|---|---|---|---|")
        for flip in analysis["sign_flips"]:
            add(f"| {flip['mode']} | {flip['template']} | {flip['dtype']} | {flip.get('axis', '')} | {flip['left_case_id']} | {flip['left_label']} | {flip['right_case_id']} | {flip['right_label']} |")
        add("")
    add("结果表另有 `results.jsonl`（逐 case/mode）与 `results.csv`。未完整 case 不进入结果表，但其原始行与排除原因保留在审计表。")
    add("")


def _result_table(rows: list[dict[str, Any]], title: str, add) -> None:
    add(f"### {title}")
    add("")
    if not rows:
        add("无完整结果。")
        add("")
        return
    add("| case_id | template | shape | dtype | pairs | T_split µs | T_fused µs | B µs | rel | rel 95% CI | label(±2%) | ±1 / ±5 | attempts |")
    add("|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|")
    for row in rows:
        shape = row.get("shape") or {}
        shape_text = "N=" + str(shape.get("N")) if "N" in shape else f"M={shape.get('M')},K={shape.get('K')}"
        rel = row.get("relative_benefit")
        ci = row.get("relative_ci95") or [None, None]
        labels = row.get("labels_by_threshold") or {}
        attempts = ",".join(f"{block}:{attempt}" for block, attempt in sorted((int(k), v) for k, v in row.get("selected_attempts", {}).items()))
        add(
            f"| {row['case_id']} | {row['template']} | {shape_text} | {row['dtype']} | {row['n_pairs']} | "
            f"{_num(row.get('T_split_us'))} | {_num(row.get('T_fused_us'))} | {_num(row.get('B_us'))} | "
            f"{_pct(rel)} | [{_pct(ci[0])}, {_pct(ci[1])}] | {row['label']} | "
            f"{labels.get('0.01')}/{labels.get('0.05')} | {attempts} |"
        )
    add("")


def _section_evaluation(analysis: dict[str, Any], add) -> None:
    add("## 5. 留模板决策评估")
    add("")
    add("分组：每次留出一个完整 `template`，其余模板训练；所有 shape、dtype 跟随模板。主要指标为 $\\mathrm{regret}=T_{chosen}-\\min(T_{split},T_{fused})$，单位 µs/call。覆盖率分母为该折全部完整 case；符号准确率只统计 fuse/split 明确点（表中 `n clear`），equivalent/uncertain 仍进入 regret 但与符号准确率无关。")
    add("")
    for mode, evaluation in analysis["evaluations"].items():
        add(f"### {mode} 模式")
        add("")
        if not evaluation["folds"]:
            add(f"不可评估：{evaluation.get('unavailable_reason')}")
            add("")
            continue
        add("| fold(test template) | method | coverage | n pred | n clear | sign acc | mean regret µs | max regret µs |")
        add("|---|---|---:|---:|---:|---:|---:|---:|")
        for fold in evaluation["folds"]:
            for method, metrics in fold["methods"].items():
                add(
                    f"| {fold['test_template']} | {method} | {_pct(metrics.get('coverage'))} | {metrics.get('n_predicted')} | "
                    f"{metrics.get('n_clear')} | {_pct(metrics.get('sign_accuracy'))} | {_num(metrics.get('mean_regret_us'))} | {_num(metrics.get('max_regret_us'))} |"
                )
        add("")
        add("**汇总（全部测试折）**")
        add("")
        add("| method | coverage | n pred | n clear | clear share | mean regret µs | max regret µs | template macro µs | sign acc |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for method, metrics in evaluation["methods"].items():
            clear_share = (metrics.get("n_clear") / metrics["n_predicted"]) if metrics.get("n_predicted") else None
            add(
                f"| {method} | {_pct(metrics.get('coverage'))} | {metrics.get('n_predicted')} | {metrics.get('n_clear')} | {_pct(clear_share)} | "
                f"{_num(metrics.get('mean_regret_us'))} | {_num(metrics.get('max_regret_us'))} | "
                f"{_num(metrics.get('template_macro_regret_us'))} | {_pct(metrics.get('sign_accuracy'))} |"
            )
        add("")
        add("**默认策略可映射同一组 ID 上的对照**")
        add("")
        add("| method | coverage | n pred | n clear | mean regret µs | max regret µs | sign acc |")
        add("|---|---:|---:|---:|---:|---:|---:|")
        for method, metrics in evaluation["matched_methods"].items():
            add(
                f"| {method} | {_pct(metrics.get('coverage'))} | {metrics.get('n_predicted')} | {metrics.get('n_clear')} | "
                f"{_num(metrics.get('mean_regret_us'))} | {_num(metrics.get('max_regret_us'))} | {_pct(metrics.get('sign_accuracy'))} |"
            )
        add("")
        add("### 每折阈值与模型")
        add("")
        add("| fold | byte τ | byte rule train loss µs | model 2-term rank/cond | model 3-term rank/cond |")
        add("|---|---:|---:|---|---|")
        for fold in evaluation["folds"]:
            byte = fold["byte_threshold"]
            m2 = fold["models"]["model_2term"]
            m3 = fold["models"]["model_3term"]
            add(
                f"| {fold['test_template']} | {_num(byte.get('tau'))} | {_num(byte.get('loss_us'))} | "
                f"{m2.get('rank')}/{_num(m2.get('condition_number'))} | {m3.get('rank')}/{_num(m3.get('condition_number'))} |"
            )
        add("")
        add("决策逐行明细：`decisions.jsonl` / `decisions.csv`。`model_2term_common` 与 `model_3term` 在 x3 齐全的同一子集上训练和评估。")
        add("")
        reps = analysis.get("representatives") or {}
        if mode == "direct" and reps.get("methods"):
            add("### 事后展示的代表点（每方法最大 regret）")
            add("")
            add("选择规则：" + str(reps.get("selection_rule")) + "。")
            add("")
            add("| method | fold | case_id | predicted | actual | T_split µs | T_fused µs | regret µs |")
            add("|---|---|---|---|---|---:|---:|---:|")
            for method, row in reps["methods"].items():
                add(
                    f"| {method} | {row.get('fold')} | {row['case_id']} | {row['predicted']} | {row['actual_label']} | "
                    f"{_num(row.get('T_split_us'))} | {_num(row.get('T_fused_us'))} | {_num(row.get('regret_us'))} |"
                )
            add("")


def _section_features(audit: dict[str, Any], analysis: dict[str, Any], add) -> None:
    add("## 6. 特征白名单与模型")
    add("")
    add("| 特征 | 定义 |")
    add("|---|---|")
    for name in config.FEATURE_WHITELIST:
        add(f"| `{name}` | {config.FEATURE_UNITS[name]} |")
    add("")
    add("禁止进入模型的字段（只作标签/诊断）：" + ", ".join(f"`{name}`" for name in config.FORBIDDEN_MODEL_FIELDS) + "。")
    add("")
    rows = analysis["feature_rows"]
    add(f"- 完整 case 的特征行：{len(rows)}；x1 非空：{sum(1 for row in rows if row.get('x1_us') is not None)}；x2 非空：{sum(1 for row in rows if row.get('x2_us') is not None)}；x3 非空：{sum(1 for row in rows if row.get('x3_us') is not None)}。")
    add("")
    for mode, evaluation in analysis["evaluations"].items():
        full = analysis.get("full_fits", {}).get(mode)
        if not full:
            continue
        add(f"### {mode} 全数据重拟合（仅供冻结包；留出指标见上一节）")
        add("")
        add("| model | features | n | rank | condition | coefficients | c (penalty on x3) |")
        add("|---|---|---:|---:|---:|---|---:|")
        for name in ("model_2term", "model_2term_common", "model_3term"):
            model = full.get(name) or {}
            coeffs = ", ".join(f"{key}={_num(value)}" for key, value in (model.get("coefficients") or {}).items())
            add(
                f"| {name} | {', '.join(model.get('features') or [])} | {model.get('n')} | {model.get('rank')} | "
                f"{_num(model.get('condition_number'))} | {coeffs} | {_num(model.get('c_penalty_us'))} |"
            )
        byte = full.get("byte_threshold") or {}
        add("")
        add(f"- 字节阈值：τ={_num(byte.get('tau'))} bytes；规则 `{byte.get('rule')}`；训练平均 regret={_num(byte.get('loss_us'))} µs/call。")
        add("")


def _section_diagnostics(analysis: dict[str, Any], add) -> None:
    diagnostics = analysis.get("diagnostics") or {}
    add("## 7. 机制诊断（只作诊断，不入模）")
    add("")
    add("遥测与 profiler 计数器不进入特征表；它们只用于解释机制、污染判定和失败案例。")
    add("")
    telemetry = diagnostics.get("telemetry") or {}
    add("| telemetry | 值 |")
    add("|---|---|")
    add(f"| rows | {telemetry.get('rows')} |")
    for key in ("temperature_c", "power_w", "clock_sm_mhz"):
        value = telemetry.get(key)
        add(f"| {key} | {_minmax(value)} |")
    add(f"| throttle states | {_md(telemetry.get('throttle_states'))} |")
    add(f"| external activity rows | {telemetry.get('external_activity_rows')} |")
    add("")
    profiles = diagnostics.get("profiles") or []
    if profiles:
        add("| case_id | strategy | metric | n | mean | min | max | unit |")
        add("|---|---|---|---:|---:|---:|---:|---|")
        for row in profiles[:40]:
            add(
                f"| {row['case_id']} | {row['strategy']} | {row['metric']} | {row['n']} | "
                f"{_num(row['mean'])} | {_num(row['min'])} | {_num(row['max'])} | {_md(row.get('unit'))} |"
            )
        if len(profiles) > 40:
            add(f"| … | | | | | | | 另有 {len(profiles) - 40} 行，见 `diagnostics.json` |")
    else:
        add("本 run 没有可解析的 `profiles.jsonl`；机制诊断保留为缺失，不阻塞主结果。")
    add("")


def _minmax(value: Any) -> str:
    if not isinstance(value, dict):
        return "—"
    return f"[{_num(value.get('min'))}, {_num(value.get('max'))}]"


def _section_figures(figures: Sequence[str], add) -> None:
    add("## 8. 图")
    add("")
    for figure in figures:
        title = Path(figure).stem.replace("_", " ")
        add(f"### {title}")
        add("")
        add(f"![{title}]({figure})")
        add("")


def _section_provenance(audit: dict[str, Any], analysis: dict[str, Any], add) -> None:
    add("## 9. 从报告数字追溯到原始 block、case_id 与编译产物")
    add("")
    direct = [row for row in analysis["results"] if row["mode"] == "direct"]
    if not direct:
        add("无完整 direct 结果可追溯。")
        add("")
        return
    # Prefer a case with a clear fuse label; otherwise the largest absolute gain.
    preferred = [row for row in direct if row["label"] == "fuse"]
    example = max(preferred or direct, key=lambda row: abs(row.get("B_us") or 0.0))
    trace = case_trace(audit, example["case_id"], "direct")
    add(f"选择 `{example['case_id']}`（direct）。结果表中的 $T_{{split}}$={_num(example['T_split_us'])} µs、$T_{{fused}}$={_num(example['T_fused_us'])} µs、$B$={_num(example['B_us'])} µs 可按下式复核：")
    add("")
    add("```text")
    add("us_per_call = elapsed_ms * 1000 / n")
    add("T_split_us = mean(split us_per_call over selected blocks)")
    add("T_fused_us = mean(fused us_per_call over selected blocks)")
    add("B_us       = mean(split us_per_call - fused us_per_call)")
    add("```")
    add("")
    add("| block | attempt | split line | split elapsed_ms | split n | split µs | fused line | fused elapsed_ms | fused n | fused µs |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for pair in trace.get("pairs", []):
        split = pair["split"]
        fused = pair["fused"]
        add(
            f"| {pair['block']} | {pair['attempt']} | {split['line']} | {_num(split['elapsed_ms'])} | {split['n']} | {_num(split['us_per_call'])} | "
            f"{fused['line']} | {_num(fused['elapsed_ms'])} | {fused['n']} | {_num(fused['us_per_call'])} |"
        )
    add("")
    add("| strategy | artifact path | SHA-256 | trace | verdict |")
    add("|---|---|---|---|---|")
    for strategy, artifact in trace.get("artifacts", {}).items():
        add(f"| {strategy} | `{artifact.get('path')}` | `{artifact.get('hash')}` | `{artifact.get('trace_path')}` | {artifact.get('trace_verdict')} |")
    add("")
    add("复算命令：")
    add("")
    add("```bash")
    add(f"python -m fusionaudit trace --run {audit['run_dir']} --case-id {example['case_id']} --mode direct")
    add("```")
    add("")
    add("上述 artifact 路径相对 run 根目录；`freeze/manifest.json` 保存输入文件和分析代码哈希。")
    add("")


def _section_freeze(audit: dict[str, Any], add) -> None:
    add("## 10. 冻结包与限制")
    add("")
    add("- `freeze/manifest.json`：输入哈希、输出哈希、分析配置、分组规则。")
    add("- `freeze/features.json`：预测字段白名单、禁用字段、单位与来源。")
    add("- `freeze/models.json`：每折模型、全数据重拟合系数、字节阈值与适用条件。")
    add("- `freeze/sha256sums.txt`：本报告、表、图与冻结文件的校验和。")
    add("- `freeze/README.md`：复跑命令与跨设备验证时仍待检验的假设。")
    add("")
    if audit["synthetic"]:
        add("**结论状态：** 正式 RTX 4090 数据尚未接入；本 run 只证明分析程序能发现无效记录、选择首个完整合格 attempt、给出四类标签并阻断模板泄漏。真实数据到达后使用同一入口重跑，替换 run 目录即可。")
    else:
        add("**结论范围：** 本报告只覆盖该 run 的单一设备与所列模板；跨设备迁移需要另一型号 GPU 的冻结包验证，当前未执行。")
    add("")


def _md(value: Any) -> str:
    if value is None:
        return "—"
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text


def _num(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _md(value)
    if not (abs(number) < float("inf")):
        return str(number)
    if number == 0:
        return "0"
    if abs(number) >= 1000:
        return f"{number:.6g}"
    return f"{number:.6g}"


def _pct(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.3f}%"
    except (TypeError, ValueError):
        return "—"
