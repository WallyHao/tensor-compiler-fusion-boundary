"""Build paper tables and vector plots from the strictly audited experiment runs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import random
from types import SimpleNamespace

from fb.util import read_json, read_jsonl, sha256_file
from fusionaudit.stats import mean, percentile
from scripts.validate_transfer import compare, score_followup


RUNS = {
    "3090-base": ("3090", "full-meas-3090-20261126T041621Z"),
    "4090": ("4090", "full-meas-4090"),
    "H800": ("h800", "measure-h800"),
    "3090-followup": ("followup", "followup-full-meas-3090-20261126T082429Z"),
    "3090-rerun": ("followup", "followup-rerun-3090-20261126T082700Z"),
}
MODES = ("direct", "replay")
LABELS = ("fuse", "equivalent", "uncertain", "split")
GROUPS = {"anchor": "Controls", "long": "Long reduction", "layout": "Transposed input",
          "rmsnorm": "Inverse RMS"}
COLORS = {"direct": "#176997", "replay": "#ac4f18"}


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def label_counts(results):
    common = results["3090-base"].keys() & results["4090"].keys() & results["H800"].keys()
    assert len(common) == 24
    for cohort, cases in results.items():
        scopes = {"all": cases}
        if cohort in ("3090-base", "4090", "H800"):
            scopes["three-GPU subset"] = {k: cases[k] for k in sorted(common)}
        if cohort == "3090-followup":
            scopes.update({group: {key: row for key, row in cases.items() if row["group"] == group}
                           for group in GROUPS})
        for scope, subset in scopes.items():
            for mode in MODES:
                rows = [row for row in subset.values() if row["mode"] == mode]
                for threshold in ("0.01", "0.02", "0.05"):
                    counts = Counter(row["thresholds"][threshold] for row in rows)
                    yield {"cohort": cohort, "scope": scope, "mode": mode,
                           "threshold": threshold, "n": len(rows),
                           **{label: counts[label] for label in LABELS}}


def split_fused_pairs(run, results):
    attempts = defaultdict(dict)
    for row in read_jsonl(run / "timings.jsonl"):
        key = row["case_id"], row["mode"]
        if (row["quality"] == "ok"
                and row["attempt"] == results[key]["selected_attempts"][str(row["block"])]):
            attempts[key, row["block"]][row["strategy"]] = row["elapsed_ms"] * 1000 / row["n"]
    pairs = defaultdict(list)
    for (key, _), entry in sorted(attempts.items()):
        pairs[key].append((entry["split"], entry["fused"]))
    assert all(len(pairs[key]) == result["n_pairs"] for key, result in results.items())
    return pairs


def bootstrap_layout_difference(transposed, control, rng):
    """Resample intact timing blocks separately in each layout."""
    def point(pairs):
        split, fused = map(mean, zip(*pairs))
        return split - fused, 100 * (split - fused) / split

    adjusted = []
    for _ in range(10000):
        transposed_draw = rng.choices(transposed, k=len(transposed))
        control_draw = rng.choices(control, k=len(control))
        (t_us, t_pct), (c_us, c_pct) = point(transposed_draw), point(control_draw)
        adjusted.append((t_us - c_us, t_pct - c_pct))
    return [[percentile(values, q) for q in (2.5, 97.5)] for values in zip(*adjusted)]


def layout_contrasts(run, results):
    pairs = split_fused_pairs(run, results)
    rng = random.Random(2718)
    catalog = [r for r in read_jsonl(run / "cases.jsonl") if r["group"] == "layout"]
    for case in sorted(catalog, key=lambda r: (r["m"], r["template"])):
        for mode in MODES:
            key, control_id = (case["case_id"], mode), (case["control_id"], mode)
            transposed, control = results[key], results[control_id]
            ci_us, ci_pp = bootstrap_layout_difference(pairs[key], pairs[control_id], rng)
            yield {
                "template": case["template"], "M": case["m"], "K": case["k"], "mode": mode,
                "transposed_case_id": key[0], "control_case_id": control_id[0],
                "control_split_us": control["T_split_us"], "transposed_split_us": transposed["T_split_us"],
                "control_benefit_us": control["B_us"], "transposed_benefit_us": transposed["B_us"],
                "control_benefit_pct": 100 * control["relative_benefit"],
                "transposed_benefit_pct": 100 * transposed["relative_benefit"],
                "difference_us": transposed["B_us"] - control["B_us"],
                "difference_us_lo95": ci_us[0], "difference_us_hi95": ci_us[1],
                "difference_pp": 100 * (transposed["relative_benefit"] - control["relative_benefit"]),
                "difference_pp_lo95": ci_pp[0], "difference_pp_hi95": ci_pp[1],
            }


def score_models(base, directories):
    source = directories["3090-base"] / "results.jsonl"
    for cohort, prefix, prediction_dir in (
        ("4090", "4090", "transfer-prediction"),
        ("H800", "h800", "transfer-prediction"),
        ("3090-followup", "followup", "runs/followup-transfer-prediction-20261126T082320Z"),
    ):
        predictions = base / prefix / prediction_dir / "predictions.json"
        frozen = read_json(predictions)
        assert frozen["split_timings_sha256"] == sha256_file(predictions.with_name("split_timings.jsonl"))
        archived_models = base / "3090/analysis/out" / RUNS["3090-base"][1] / "models.json"
        assert frozen["source_models_sha256"] == sha256_file(archived_models)
        target_dir, run_id = RUNS[cohort]
        target = dict(predictions=predictions, target_results=directories[cohort] / "results.jsonl",
                      target_run=base / target_dir / "runs" / run_id,
                      output=base / "scoring" / f"{cohort}.json")
        target["output"].parent.mkdir(exist_ok=True)
        if cohort == "3090-followup":
            score_followup(SimpleNamespace(history_results=source, **target))
        else:
            compare(SimpleNamespace(source_results=source, **target))
        for row in read_json(target["output"])["metrics"]:
            scope = row.get("template") or row.get("group")
            yield {"cohort": cohort, "scope": scope,
                   **{key: value for key, value in row.items() if key not in ("template", "group")}}


def plot_results(out, followup, layouts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.hashsalt": "fusion-paper"})

    def save(fig, label):
        for extension in ("pdf", "svg"):
            metadata = {"CreationDate": None} if extension == "pdf" else {"Date": None}
            fig.savefig(out / f"{label}.{extension}", metadata=metadata)
        plt.close(fig)

    ordered = []
    for group in GROUPS:
        cases = [row for row in followup.values() if row["group"] == group and row["mode"] == "direct"]
        cases.sort(key=lambda r: (r["shape"]["M"], r["template"], r["shape"]["K"]))
        ordered.append((group, cases))
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(7.2, 7.9))
    fig.subplots_adjust(left=0.23, right=0.96, top=0.93, bottom=0.10, wspace=0.20)
    for panel, (axis, mode, title, limits) in enumerate(zip(
        axes, MODES, ("Ordinary submission", "CUDA Graph replay"), ((-3, 42), (-3, 15)),
    )):
        axis.set_title(f"({'ab'[panel]}) {title}", loc="left", fontweight="bold", fontsize=10)
        axis.axvspan(-2, 2, color="#e9ebee", zorder=0)
        axis.axvline(0, color="#7c8793", linewidth=0.8, zorder=1)
        axis.grid(axis="x", color="#e1e5e9", linewidth=0.6)
        ticks, labels = [], []
        y = 0
        for group, cases in ordered:
            if panel == 0:
                axis.text(-0.04, y, GROUPS[group], transform=axis.get_yaxis_transform(),
                          ha="right", va="center", fontsize=9, fontweight="bold", clip_on=False)
            y += 1
            for case in cases:
                row = followup[case["case_id"], mode]
                mid = 100 * row["relative_benefit"]
                lo, hi = (100 * value for value in row["relative_ci95"])
                filled = row["label"] == "fuse"
                axis.hlines(y, lo, hi, color=COLORS[mode], linewidth=1.25)
                axis.plot(mid, y, marker="o", markersize=4.2, markeredgecolor=COLORS[mode],
                          markerfacecolor=COLORS[mode] if filled else "white")
                ticks.append(y)
                m, k = case["shape"]["M"], case["shape"]["K"]
                labels.append(f"{case['template']}  {m} × {k:,}")
                y += 1
            y += 0.45
        axis.set(ylim=(y - 0.4, -0.65), xlim=limits, xlabel="Latency reduction (%)")
        axis.tick_params(axis="y", length=0)
        axis.spines["left"].set_visible(False)
        if panel == 0:
            axis.set_yticks(ticks, labels, fontsize=8)
    save(fig, "3090-followup")

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(7.2, 2.95))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.68, bottom=0.24, wspace=0.22)
    for panel, (axis, units, title, xlabel) in enumerate(zip(
        axes, ("pp", "us"), ("Relative benefit", "Absolute saving"),
        ("Transposed minus contiguous (pp)", "Transposed minus contiguous (µs)"),
    )):
        axis.set_title(f"({'ab'[panel]}) {title}", loc="left", fontweight="bold", fontsize=10)
        axis.axvline(0, color="#7c8793", linewidth=0.8)
        axis.grid(axis="x", color="#e1e5e9", linewidth=0.6)
        for i, row in enumerate(layouts):
            mode, y = row["mode"], i // 2 + (0.15 if row["mode"] == "replay" else -0.15)
            key = f"difference_{units}"
            axis.hlines(y, row[key + "_lo95"], row[key + "_hi95"], color=COLORS[mode])
            axis.plot(row[key], y, marker="o" if mode == "direct" else "D", color=COLORS[mode], markersize=4)
        axis.set(ylim=(3.5, -0.5), xlabel=xlabel)
        axis.spines["left"].set_visible(False)
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(4), [f"{r['template']}, M={r['M']}" for r in layouts[::2]])
    handles = [Line2D([], [], color=COLORS[m], marker=marker, label=label)
               for m, marker, label in zip(MODES, ("o", "D"), ("Ordinary submission", "CUDA Graph replay"))]
    fig.legend(handles=handles, loc="upper center", frameon=False, ncol=2)
    save(fig, "layout-contrasts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True, help="extracted and audited archive directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tables, figures = args.out / "tables", args.out / "figures"
    for directory in (tables, figures):
        directory.mkdir(parents=True, exist_ok=True)
    directories = {cohort: args.sources / "rebuilt" / run_id
                   for cohort, (_, run_id) in RUNS.items()}
    results = {}
    case_rows = []
    for cohort, directory in directories.items():
        rows = read_jsonl(directory / "results.jsonl")
        assert all(not row["synthetic"] and row["n_pairs"] == 30 for row in rows)
        results[cohort] = {(row["case_id"], row["mode"]): row for row in rows}
        for row in rows:
            case_rows.append({
                "cohort": cohort, "run_id": row["run_id"], "group": row["group"],
                "case_id": row["case_id"], "mode": row["mode"], "n_pairs": row["n_pairs"],
                "T_split_us": row["T_split_us"], "T_fused_us": row["T_fused_us"], "B_us": row["B_us"],
                "B_lo95_us": row["B_ci95"][0], "B_hi95_us": row["B_ci95"][1],
                "benefit_pct": 100 * row["relative_benefit"],
                "benefit_lo95_pct": 100 * row["relative_ci95"][0],
                "benefit_hi95_pct": 100 * row["relative_ci95"][1],
                "label_2pct": row["label"], "default": row["default_decision"],
            })
    write_csv(tables / "case-results.csv", case_rows)
    write_csv(tables / "label-counts.csv", list(label_counts(results)))
    write_csv(tables / "decision-metrics.csv", list(score_models(args.sources, directories)))
    root, run_id = RUNS["3090-followup"]
    layouts = list(layout_contrasts(args.sources / root / "runs" / run_id, results["3090-followup"]))
    write_csv(tables / "layout-contrasts.csv", layouts)
    plot_results(figures, results["3090-followup"], layouts)


if __name__ == "__main__":
    main()
