"""Render the manuscript's quantitative figures from the audited CSV tables.

The visual system is intentionally small and shared across every panel:

* 394 pt canvas (the ACM text width), Linux Biolinum for figure text;
* Okabe-Ito colours, with separate palettes for submission mode and decision;
* 0.7 pt rules, 3.8 pt markers, capped 95% intervals;
* no decorative fills, no heatmaps for small count tables unless the matrix
  itself is the comparison being made.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from collections import Counter
from math import log2
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator


PAPER = Path(__file__).resolve().parent
WIDTH = 394 / 72
MODES = ("direct", "replay")
MODE_NAMES = {"direct": "Direct submission", "replay": "CUDA Graph replay"}
MODE_SHORT = {"direct": "Direct", "replay": "Replay"}

# Okabe-Ito, with darker text-grade variants.
COLORS = {"direct": "#0072B2", "replay": "#D55E00"}
MARKERS = {"direct": "o", "replay": "D"}
DECISION_COLORS = {
    "fuse": "#009E73",
    "equivalent": "#9AA5AD",
    "uncertain": "#D9DEE2",
    "split": "#CC79A7",
}
FAMILY_COLORS = {
    "P1": "#0072B2",
    "P2": "#56B4E9",
    "R1": "#009E73",
    "R2": "#E69F00",
    "N1": "#CC79A7",
}
FAMILY_TEXT = {
    "P1": "white",
    "P2": "#1F2933",
    "R1": "white",
    "R2": "#1F2933",
    "N1": "white",
}
GROUPS = {
    "anchor": "Contiguous controls",
    "long": "Long reductions",
    "layout": "Transposed inputs",
    "rmsnorm": "Inverse RMS",
}
INK = "#1F2933"
MUTED = "#6B7780"
GRID = "#E8EDF0"
BAND = "#F3F6F8"
CONNECTOR = "#B8C2C8"
CMAP = LinearSegmentedColormap.from_list(
    "paper_blues", ["#F7FAFC", "#DCE8F0", "#A9C9D8", "#5B93AE", "#174A66"]
)

FONT_DIR = PAPER / "template" / "fonts"
for font_path in sorted(FONT_DIR.glob("LinBiolinum_*.otf")):
    fm.fontManager.addfont(str(font_path))


def read_table(name):
    with (PAPER / "tables" / f"{name}.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    path = PAPER / "figures" / name
    fig.savefig(path.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None})
    plt.close(fig)


def draw_ci_h(ax, y, row, columns, *, color, marker, lw=0.8, ms=3.8):
    mid, lo, hi = (float(row[column]) for column in columns)
    ax.hlines(y, lo, hi, color=color, linewidth=lw, zorder=3)
    ax.vlines((lo, hi), y - 0.055, y + 0.055, color=color, linewidth=lw, zorder=3)
    ax.plot(mid, y, marker=marker, color=color, markersize=ms, mec="white",
            mew=0.5, linestyle="none", zorder=4)


def draw_ci_v(ax, x, row, columns, *, color, marker, lw=0.8, ms=3.8):
    mid, lo, hi = (float(row[column]) for column in columns)
    ax.vlines(x, lo, hi, color=color, linewidth=lw, zorder=3)
    ax.hlines((lo, hi), x - 0.04, x + 0.04, color=color, linewidth=lw, zorder=3)
    ax.plot(x, mid, marker=marker, color=color, markersize=ms, mec="white",
            mew=0.5, linestyle="none", zorder=4)


def mode_legend(fig, anchor=(0.5, 1.0), *, lines=False):
    handles = []
    for mode in MODES:
        handles.append(
            Line2D(
                [], [],
                marker=MARKERS[mode],
                color=COLORS[mode],
                linewidth=0.9 if lines else 0,
                linestyle=("-" if mode == "direct" else (0, (4, 2))) if lines else "none",
                markeredgecolor="white",
                markeredgewidth=0.4,
                markersize=4.2,
                label=MODE_NAMES[mode],
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=anchor,
        ncol=2,
        frameon=False,
        handletextpad=0.5,
        handlelength=1.6 if lines else 0.8,
        columnspacing=1.4,
        borderaxespad=0,
    )


def panel_title(ax, text):
    ax.set_title(text, loc="left", fontsize=8.8, fontweight="bold", pad=4)


def followup_saving(rows):
    paired = {
        (r["case_id"], r["mode"]): r
        for r in rows
        if r["cohort"] == "3090-followup"
    }
    fig, ax = plt.subplots(figsize=(WIDTH, 4.25))
    fig.subplots_adjust(left=0.305, right=0.965, top=0.885, bottom=0.095)
    mode_legend(fig, (0.5, 0.99))

    ticks, labels, y = [], [], 0.0
    columns = ("B_us", "B_lo95_us", "B_hi95_us")
    for index, (group, title) in enumerate(GROUPS.items()):
        cases = [
            case
            for (case, mode), row in paired.items()
            if mode == "direct" and row["group"] == group
        ]
        cases.sort(
            key=lambda case: (
                re.match(r"^(\w+)-m(\d+)-k(\d+)-", case).group(1),
                int(re.match(r"^(\w+)-m(\d+)-k(\d+)-", case).group(2)),
                int(re.match(r"^(\w+)-m(\d+)-k(\d+)-", case).group(3)),
            )
        )
        step = 0.88
        if index % 2 == 0:
            ax.axhspan(y - 0.45, y + len(cases) * step - 0.15, color=BAND, zorder=0)
        ax.text(
            -0.012, y - 0.25, title.upper(),
            transform=ax.get_yaxis_transform(),
            ha="right", va="bottom",
            fontsize=7.0, fontweight="bold", color="#52606A",
        )
        y += 0.10
        for case in cases:
            direct, replay = (paired[case, mode] for mode in MODES)
            direct_mid = float(direct["B_us"])
            replay_mid = float(replay["B_us"])
            ax.plot(
                (replay_mid, direct_mid), (y, y),
                color=CONNECTOR, linewidth=0.7, zorder=1,
                solid_capstyle="round",
            )
            draw_ci_h(ax, y, direct, columns, color=COLORS["direct"],
                      marker=MARKERS["direct"])
            draw_ci_h(ax, y, replay, columns, color=COLORS["replay"],
                      marker=MARKERS["replay"])
            template, m, k = re.match(r"^(\w+)-m(\d+)-k(\d+)-", case).groups()
            ticks.append(y)
            labels.append(f"{template}   {m} × {int(k):,}")
            y += step
        y += 0.75

    ax.set(
        xlabel="Fusion saving per call (µs)",
        xlim=(-0.25, 3.55),
        ylim=(y - 0.5, -0.7),
    )
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.grid(True, color=GRID, linewidth=0.55)
    ax.axvline(0, color=MUTED, linewidth=0.75, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.text(
        0.995, 0.012, "median saving 2.36 → 0.68 µs",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.0, color=MUTED,
    )
    save(fig, "followup-saving")


def layout_effects(rows):
    order = sorted({(int(r["M"]), r["template"]) for r in rows})
    lookup = {(int(r["M"]), r["template"], r["mode"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(WIDTH, 2.25))
    fig.subplots_adjust(left=0.18, right=0.965, top=0.785, bottom=0.215, wspace=0.15)
    for ax, unit, title, label in zip(
        axes,
        ("pp", "us"),
        ("(a) Relative saving", "(b) Absolute saving"),
        ("Change in relative saving (pp)", "Change in absolute saving (µs/call)"),
    ):
        key = f"difference_{unit}"
        for y, (m, template) in enumerate(order):
            for offset, mode in zip((-0.11, 0.11), MODES):
                row = lookup[m, template, mode]
                draw_ci_h(
                    ax, y + offset, row,
                    (key, f"{key}_lo95", f"{key}_hi95"),
                    color=COLORS[mode], marker=MARKERS[mode],
                )
        ax.set_title(title, loc="left", fontsize=8.8, fontweight="bold", pad=4)
        ax.set(xlabel=label, ylim=(len(order) - 0.5, -0.5))
        ax.set_yticks(range(len(order)), [f"{t} · M={m}" for m, t in order])
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color=GRID, linewidth=0.55)
        ax.yaxis.grid(False)
        ax.axvline(0, color=MUTED, linewidth=0.75, zorder=0)
        ax.xaxis.set_major_locator(MultipleLocator(5 if unit == "pp" else 0.1))
        for boundary in range(1, len(order)):
            ax.axhline(boundary - 0.5, color=GRID, linewidth=0.5, zorder=0)
        ax.margins(x=0.06)
    mode_legend(fig, (0.5, 0.99))
    save(fig, "layout-effects")


def device_labels(rows):
    labels = ("fuse", "equivalent", "uncertain")
    display = ("Fuse", "Equivalent", "Inconclusive")
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(WIDTH, 2.40))
    fig.subplots_adjust(left=0.15, right=0.975, bottom=0.295, top=0.755, wspace=0.12)
    for i, (ax, mode) in enumerate(zip(axes, MODES)):
        by_gpu = [
            {
                r["case_id"]: r["label_2pct"]
                for r in rows
                if r["cohort"] == gpu and r["mode"] == mode
            }
            for gpu in ("3090-base", "4090")
        ]
        source, target = by_gpu
        if source.keys() != target.keys() or len(source) != 72:
            raise ValueError(
                "RTX label matrices require the same 72 configurations on both devices"
            )
        if (set(source.values()) | set(target.values())) - set(labels):
            raise ValueError("Add a matrix category for split-favored results")
        counts = Counter((source[key], target[key]) for key in source)
        matrix = np.array([[counts[s, t] for t in labels] for s in labels])

        ax.imshow(matrix, cmap=CMAP, vmin=0, vmax=72, aspect="auto")
        ax.set_xticks(range(3), display, fontsize=7.0)
        ax.set_yticks(range(3), display if i == 0 else [], fontsize=7.2)
        if i == 0:
            ax.set_ylabel("RTX 3090 label")
            ax.tick_params(axis="y", length=0, pad=3)
        else:
            ax.tick_params(axis="y", length=0, labelleft=False)
        ax.set_xlabel("RTX 4090 label", labelpad=4)
        ax.tick_params(axis="x", length=0, pad=3)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for y in range(3):
            for x in range(3):
                value = int(matrix[y, x])
                ax.add_patch(
                    Rectangle(
                        (x - 0.5, y - 0.5), 1, 1,
                        fill=False, edgecolor="white", linewidth=1.2, zorder=5,
                    )
                )
                color = "white" if value >= 30 else INK
                if value == 0:
                    color = "#AAB3B9"
                ax.text(
                    x, y, str(value),
                    ha="center", va="center",
                    fontsize=9.2,
                    fontweight="bold" if value else "normal",
                    color=color,
                )
        for diagonal in range(3):
            ax.add_patch(
                Rectangle(
                    (diagonal - 0.5, diagonal - 0.5), 1, 1,
                    fill=False, edgecolor=DECISION_COLORS["fuse"], linewidth=1.15,
                    zorder=6,
                )
            )
        ax.set_title(
            f"({'ab'[i]}) {MODE_NAMES[mode]}",
            loc="left", fontsize=8.8, fontweight="bold", pad=4,
        )
    fig.text(
        0.5, 0.018,
        "Green outline: same label.  Diagonal 71/72 (direct) and 27/72 (replay).  "
        "No split-favored cases.",
        ha="center", va="bottom", fontsize=7.0, color=MUTED,
    )
    save(fig, "device-labels")


def pointwise_scaling(rows):
    chosen = [
        r
        for r in rows
        if r["cohort"] == "3090-base"
        and r["group"] in ("P1", "P2")
        and "-f32-" in r["case_id"]
    ]
    sizes = sorted({int(r["case_id"].split("-")[1][1:]) for r in chosen})
    positions = [log2(n) for n in sizes]
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(WIDTH, 2.25))
    fig.subplots_adjust(left=0.115, right=0.975, top=0.765, bottom=0.205, wspace=0.10)
    columns = ("benefit_pct", "benefit_lo95_pct", "benefit_hi95_pct")
    for i, (ax, template) in enumerate(zip(axes, ("P1", "P2"))):
        ax.axhspan(-2, 2, color=BAND, zorder=0)
        for boundary in (-2, 2):
            ax.axhline(boundary, color="#C9D3D9", linewidth=0.55,
                       linestyle=(0, (3, 2)), zorder=1)
        ax.axhline(0, color=MUTED, linewidth=0.6, zorder=1)
        for mode in MODES:
            series = sorted(
                (
                    (int(r["case_id"].split("-")[1][1:]), r)
                    for r in chosen
                    if r["group"] == template and r["mode"] == mode
                ),
                key=lambda item: item[0],
            )
            xs = [log2(n) for n, _ in series]
            ys = [float(r["benefit_pct"]) for _, r in series]
            ax.plot(
                xs, ys,
                color=COLORS[mode], linewidth=0.9,
                linestyle="-" if mode == "direct" else (0, (4, 2)),
                zorder=2,
            )
            for x, (_, row) in zip(xs, series):
                draw_ci_v(
                    ax, x, row, columns,
                    color=COLORS[mode], marker=MARKERS[mode],
                )
        ax.set_xticks(
            positions,
            [f"{n // 1024}K" if n < 2**20 else f"{n // 2**20}M" for n in sizes],
        )
        ax.set(
            xlim=(positions[0] - 0.45, positions[-1] + 0.45),
            ylim=(-4, 55),
            xlabel="Input length (elements, log scale)",
        )
        ax.yaxis.grid(True, color=GRID, linewidth=0.55)
        title = "(a) Affine–ReLU (P1)" if template == "P1" else "(b) Tanh–affine (P2)"
        ax.set_title(title, loc="left", fontsize=8.8, fontweight="bold", pad=4)
        ax.text(
            0.02, 0.965, "FP32 · contiguous",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=6.9, color=MUTED,
        )
        ax.text(
            0.985, 0.105, "±2% equivalence",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.9, color=MUTED,
        )
        largest = positions[-1]
        ax.annotate(
            "≈160 µs/call",
            xy=(largest - 0.08, 49.8),
            xytext=(largest - 1.55, 53.2),
            ha="right", va="center",
            fontsize=6.9, color=INK,
            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.55},
        )
    axes[0].set_ylabel("Fusion saving (% of split latency)")
    mode_legend(fig, (0.5, 0.99), lines=True)
    save(fig, "pointwise-scaling")


def archived_text(archive, member):
    return subprocess.check_output(
        ["tar", "-I", "zstd", "-xOf", PAPER.parent / "data" / archive, member],
        text=True,
    )


def workload_coverage(rows):
    cohorts = ("3090-base", "4090", "H800", "3090-followup")
    families = ("P1", "P2", "R1", "R2", "N1")
    counts = Counter(
        (r["cohort"], r["case_id"].split("-")[0])
        for r in rows
        if r["mode"] == "direct"
    )
    values = {family: [counts[c, family] for c in cohorts] for family in families}
    totals = [sum(values[family][i] for family in families) for i in range(len(cohorts))]
    y = np.arange(len(cohorts))
    fig, ax = plt.subplots(figsize=(WIDTH, 1.98))
    fig.subplots_adjust(left=0.235, right=0.965, top=0.735, bottom=0.205)

    left = np.zeros(len(cohorts))
    for family in families:
        vals = np.array(values[family])
        ax.barh(
            y, vals, left=left, height=0.56,
            color=FAMILY_COLORS[family], edgecolor="white", linewidth=1.0,
            label=family, zorder=2,
        )
        for yi, (value, start) in enumerate(zip(vals, left)):
            small_h800 = cohorts[yi] == "H800" and value < 5
            if value and not small_h800:
                ax.text(
                    start + value / 2, yi, str(value),
                    ha="center", va="center",
                    fontsize=6.6 if value < 5 else 7.2,
                    color=FAMILY_TEXT[family],
                    fontweight="bold" if value >= 8 else "normal",
                    zorder=3,
                )
        left += vals

    for yi, total in enumerate(totals):
        suffix = "  (3 each)" if cohorts[yi] == "H800" else ""
        ax.text(total + 1.2, yi, f"{total}{suffix}", ha="left", va="center",
                fontsize=8.0, fontweight="bold", color=INK)
    ax.set_yticks(
        y,
        ["RTX 3090 original", "RTX 4090 original",
         "H800 PCIe subset", "RTX 3090 follow-up"],
    )
    ax.invert_yaxis()
    ax.set(
        xlim=(0, 80),
        xlabel="Configurations",
        xticks=[0, 20, 40, 60, 80],
    )
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.55)
    ax.spines["left"].set_visible(False)
    family_names = {
        "P1": "P1 affine–ReLU",
        "P2": "P2 tanh–affine",
        "R1": "R1 sum–ReLU",
        "R2": "R2 max–affine",
        "N1": "N1 inverse RMS",
    }
    handles = [
        Patch(
            facecolor=FAMILY_COLORS[family],
            edgecolor="white",
            label=family_names[family],
        )
        for family in families
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        ncol=3, frameon=False, handlelength=0.9, handleheight=0.9,
        handletextpad=0.35, columnspacing=1.0, borderaxespad=0,
    )
    save(fig, "workload-coverage")


def calibration_curve():
    fit = json.loads(
        archived_text(
            "fb-3090-phase1-runs-20261126.tar.zst",
            "runs/full-meas-3090-20261126T041621Z/calibration_fit.json",
        )
    )
    fig, ax = plt.subplots(figsize=(WIDTH, 2.10))
    fig.subplots_adjust(left=0.145, right=0.965, top=0.775, bottom=0.225)
    for mode in MODES:
        entry = fit["launch_" + mode]
        points = entry["point_dispersion"]
        lengths = sorted(map(int, points))
        grid = np.geomspace(min(lengths), max(lengths), 220)
        ax.plot(
            grid, entry["intercept"] + grid * entry["slope"],
            color=COLORS[mode], linewidth=0.95,
            linestyle="-" if mode == "direct" else (0, (4, 2)),
            label=f'{MODE_SHORT[mode]} · {entry["slope"]:.2f} µs/kernel',
        )
        for k in lengths:
            point = points[str(k)]
            ax.vlines(
                k, point["min_segment_us"], point["max_segment_us"],
                color=COLORS[mode], linewidth=0.9, alpha=0.7,
            )
            ax.plot(
                k, point["median_segment_us"], MARKERS[mode],
                color=COLORS[mode], markersize=4.0, mec="white", mew=0.5,
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 32, 128], ["1", "8", "32", "128"])
    ax.minorticks_off()
    ax.set(
        xlabel="Minimal kernels per sequence (log scale)",
        ylabel="Sequence time (µs)",
    )
    ax.yaxis.grid(True, color=GRID, linewidth=0.55)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.6)
    save(fig, "launch-calibration")


def policy_regret():
    rows = {
        r["method"]: r
        for r in read_table("decision-metrics")
        if (r["cohort"], r["scope"], r["mode"]) == ("4090", "all", "replay")
    }
    methods = ("default_inductor", "model_2term", "model_3term")
    labels = ("Inductor default", "Two-term model", "Three-term model")
    fig, ax = plt.subplots(figsize=(WIDTH, 1.80))
    fig.subplots_adjust(left=0.235, right=0.965, top=0.805, bottom=0.225)
    for i, method in enumerate(methods):
        row = rows[method]
        value = float(row["mean_regret_us"]) * 1000
        color = COLORS["direct"] if i < 2 else COLORS["replay"]
        ax.hlines(i, 0, value, color=color, linewidth=1.6, zorder=2)
        ax.plot(value, i, "o", color=color, markersize=5.0,
                mec="white", mew=0.6, zorder=3)
        ax.text(value + 0.8, i, f"{value:.1f} ns", va="center",
                fontsize=7.6, color=INK)
        ax.text(31.5, i, f'{row["n_correct"]}/{row["n_clear"]} correct',
                ha="right", va="center", fontsize=7.0, color=MUTED)
    ax.set_yticks(range(len(methods)), labels)
    ax.invert_yaxis()
    ax.set(
        xlim=(0, 34),
        ylim=(2.55, -0.65),
        xlabel="Mean regret per invocation (ns)",
        xticks=[0, 5, 10, 15, 20, 25, 30],
    )
    ax.tick_params(axis="y", length=0, pad=4)
    ax.xaxis.grid(True, color=GRID, linewidth=0.55)
    ax.set_title("RTX 4090 · CUDA Graph replay", loc="left",
                 fontsize=8.8, fontweight="bold", pad=4)
    save(fig, "policy-regret")


def sensitivity():
    rows = [
        r
        for r in read_table("label-counts")
        if (r["scope"], r["cohort"]) == ("all", "3090-followup")
    ]
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(WIDTH, 2.02))
    fig.subplots_adjust(left=0.12, right=0.965, top=0.715, bottom=0.225, wspace=0.12)
    handles = [
        Patch(facecolor=DECISION_COLORS[label], edgecolor="white",
              label=name)
        for label, name in (
            ("fuse", "Fuse"),
            ("equivalent", "Equivalent"),
            ("uncertain", "Inconclusive"),
        )
    ]
    for ax, mode in zip(axes, MODES):
        data = sorted(
            (r for r in rows if r["mode"] == mode),
            key=lambda r: float(r["threshold"]),
        )
        for i, row in enumerate(data):
            left = 0
            for label in ("fuse", "equivalent", "uncertain"):
                count = int(row[label])
                if count:
                    ax.barh(
                        i, count, left=left,
                        color=DECISION_COLORS[label],
                        height=0.62, edgecolor="white", linewidth=0.8,
                    )
                    ax.text(
                        left + count / 2, i, str(count),
                        ha="center", va="center", fontsize=7.0,
                        color="white" if label == "fuse" else INK,
                    )
                left += count
            if left != int(row["n"]) or left != 20:
                raise ValueError("label counts must sum to n=20")
        ax.set_title(MODE_NAMES[mode], loc="left",
                     fontsize=8.8, fontweight="bold", pad=4)
        ax.set(xlim=(0, 20), xlabel="Configurations (n=20)",
               xticks=[0, 5, 10, 15, 20])
        ax.tick_params(axis="x", length=0)
        ax.xaxis.grid(True, color=GRID, linewidth=0.55)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(range(3), ["±1%", "±2%", "±5%"])
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", length=0, pad=4)
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        ncol=3, frameon=False, handlelength=1.0, handletextpad=0.4,
        columnspacing=1.5, borderaxespad=0,
    )
    save(fig, "threshold-sensitivity")


def paired_trace(rows):
    case = "N1-m256-k8192-f32-contig-t"
    raw = archived_text(
        "fb-3090-followup-20261126.tar.zst",
        "runs/followup-full-meas-3090-20261126T082429Z/timings.jsonl",
    )
    pairs = {}
    for line in raw.splitlines():
        record = json.loads(line)
        if record["case_id"] == case and record["quality"] == "ok":
            key = (record["mode"], record["block"], record["strategy"])
            assert key not in pairs and not record["synthetic"]
            pairs[key] = 1000 * record["elapsed_ms"] / record["n"]
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(WIDTH, 2.55))
    fig.subplots_adjust(left=0.135, right=0.965, top=0.885, bottom=0.205, hspace=0.38)
    for ax, mode in zip(axes, MODES):
        values = [pairs[mode, i, "split"] - pairs[mode, i, "fused"] for i in range(30)]
        summary = next(
            r for r in rows
            if r["cohort"] == "3090-followup"
            and r["case_id"] == case
            and r["mode"] == mode
        )
        avg = float(summary["B_us"])
        assert abs(avg - sum(values) / 30) < 1e-10
        ax.plot(
            range(30), values,
            color=COLORS[mode], linewidth=0.65,
            marker=MARKERS[mode], markersize=3.0,
            mec="white", mew=0.3,
        )
        ax.axhline(avg, color=INK, linestyle=(0, (4, 2)), linewidth=0.75)
        ax.set_title(
            f"{MODE_NAMES[mode]}  ·  mean {avg:.3f} µs",
            loc="left", fontsize=8.2, fontweight="bold", pad=3,
        )
        ax.set_ylabel("Saving (µs)")
        ax.yaxis.grid(True, color=GRID, linewidth=0.55)
        ax.margins(y=0.12)
    axes[1].set(xlabel="Paired block in collection order",
                xticks=[0, 5, 10, 15, 20, 25, 29])
    save(fig, "paired-trace")


CLIFF_COLORS = {256: "#B8C2C8", 512: "#56B4E9", 1024: "#0072B2",
                2048: "#E69F00", 4096: "#D55E00"}


def cliff_saving(rows):
    data = {}
    for row in rows:
        if row["dtype"] != "float32" or int(row["K"]) != 8192:
            continue
        data.setdefault(int(row["M"]), []).append(row)

    fig, axes = plt.subplots(
        1, 2, figsize=(WIDTH, 2.55), sharex=True,
        gridspec_kw={"wspace": 0.30},
    )
    axes[0].axhspan(-2, 2, color=BAND, zorder=0)
    for ax in axes:
        ax.axhline(0, color=MUTED, linewidth=0.7, zorder=1)
        ax.yaxis.grid(True, color=GRID, linewidth=0.55)
        ax.set_xlabel("Shifted reads in the epilogue $B$")

    for m, series in sorted(data.items()):
        series.sort(key=lambda r: int(r["B"]))
        xs = [int(r["B"]) for r in series]
        color = CLIFF_COLORS[m]
        axes[0].plot(xs, [float(r["relative"]) * 100 for r in series],
                     color=color, linewidth=0.7, zorder=2)
        axes[1].plot(xs, [float(r["B_us"]) for r in series],
                     color=color, linewidth=0.7, zorder=2)
        for r, x in zip(series, xs):
            split = r["label"] == "split"
            lo, hi = float(r["rel_ci95_lo"]) * 100, float(r["rel_ci95_hi"]) * 100
            axes[0].vlines(x, lo, hi, color=color, linewidth=0.8, zorder=3)
            axes[0].plot(x, float(r["relative"]) * 100, marker="o", markersize=3.8,
                         color=color, mfc=color if split else "white",
                         mec=color, mew=0.8, linestyle="none", zorder=4)
            lo, hi = float(r["B_ci95_lo"]), float(r["B_ci95_hi"])
            axes[1].vlines(x, lo, hi, color=color, linewidth=0.8, zorder=3)
            axes[1].plot(x, float(r["B_us"]), marker="o", markersize=3.8,
                         color=color, mfc=color if split else "white",
                         mec=color, mew=0.8, linestyle="none", zorder=4)

    axes[0].set_ylabel("Relative saving (% of split)")
    axes[1].set_ylabel("Absolute saving (µs/call)")
    panel_title(axes[0], "Relative saving")
    panel_title(axes[1], "Absolute saving")
    axes[0].set_xticks(range(0, 15, 2))

    handles = [
        Line2D([], [], color=CLIFF_COLORS[m], marker="o", markersize=3.4,
               markeredgecolor="white", markeredgewidth=0.4, linewidth=0.7,
               label=f"$M={m}$")
        for m in sorted(data)
    ]
    handles += [
        Line2D([], [], color=MUTED, marker="o", markersize=3.4, linestyle="none",
               mfc=INK, mec=INK, label="Split favored"),
        Line2D([], [], color=MUTED, marker="o", markersize=3.4, linestyle="none",
               mfc="white", mec=MUTED, label="Fusion favored / unresolved"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=7, frameon=False, fontsize=6.6, handletextpad=0.35,
               handlelength=1.2, columnspacing=0.7)
    fig.subplots_adjust(top=0.80)
    save(fig, "cliff-saving")


def set_style():
    plt.rcParams.update({
        "font.family": ["Linux Biolinum O", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 8.8,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.axisbelow": True,
        "lines.linewidth": 0.9,
        "lines.markersize": 3.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "path",
        "svg.hashsalt": "fusion-paper",
    })


def main():
    set_style()
    rows = read_table("case-results")
    followup_saving(rows)
    layout_effects(read_table("layout-contrasts"))
    device_labels(rows)
    pointwise_scaling(rows)
    workload_coverage(rows)
    calibration_curve()
    policy_regret()
    sensitivity()
    paired_trace(rows)


if __name__ == "__main__":
    if "--cliff-only" in sys.argv:
        set_style()
        cliff_saving(read_table("cliff-results"))
    else:
        main()
