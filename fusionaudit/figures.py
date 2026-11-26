"""Dependency-free SVG figures for the audit report."""

from __future__ import annotations

import math
from html import escape
from pathlib import Path
from typing import Any, Sequence


PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]


def render_all(out_dir: Path, analysis: dict[str, Any], audit: dict[str, Any]) -> list[str]:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    results = [row for row in analysis["results"] if row["mode"] == "direct"]
    by_template: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        by_template.setdefault(str(row["template"]), []).append(row)
    for template, rows in sorted(by_template.items()):
        reduction = any("M" in (row.get("shape") or {}) for row in rows)
        if reduction:
            dtypes = sorted({str(row.get("dtype")) for row in rows})
            for dtype in dtypes:
                subset = [row for row in rows if str(row.get("dtype")) == dtype]
                path = figures_dir / f"benefit_{_slug(template)}_{_slug(dtype)}.svg"
                _reduction_heatmap(path, template, dtype, subset)
                created.append(str(path.relative_to(out_dir)))
        else:
            path = figures_dir / f"benefit_{_slug(template)}.svg"
            _template_benefit_chart(path, template, rows)
            created.append(str(path.relative_to(out_dir)))

    coverage_items = [
        ("cases", audit["funnel"]["cases_total"]),
        ("qualified pairs (all modes)", audit["funnel"]["qualified_pairs"]),
        ("complete direct cases", audit["funnel"]["complete_cases"].get("direct", 0)),
        ("complete replay cases", audit["funnel"]["complete_cases"].get("replay", 0)),
    ]
    path = figures_dir / "coverage_funnel.svg"
    _bar_chart(path, "Coverage funnel", "records", coverage_items)
    created.append(str(path.relative_to(out_dir)))

    if "direct" in analysis["evaluations"]:
        methods = analysis["evaluations"]["direct"]["methods"]
        items = [
            (method, metrics.get("mean_regret_us"))
            for method, metrics in methods.items()
            if metrics.get("mean_regret_us") is not None
        ]
        path = figures_dir / "decision_regret.svg"
        _bar_chart(path, "Mean decision regret by method (direct, us/call)", "us/call", items)
        created.append(str(path.relative_to(out_dir)))
        coverage = [
            (method, (metrics.get("coverage") or 0.0) * 100.0)
            for method, metrics in methods.items()
        ]
        path = figures_dir / "decision_coverage.svg"
        _bar_chart(path, "Decision coverage by method (direct, %)", "%", coverage)
        created.append(str(path.relative_to(out_dir)))

    direct = {row["case_id"]: row for row in analysis["results"] if row["mode"] == "direct"}
    replay = {row["case_id"]: row for row in analysis["results"] if row["mode"] == "replay"}
    common = sorted(set(direct) & set(replay))
    if common:
        points = [
            (direct[case_id]["B_us"], replay[case_id]["B_us"], case_id)
            for case_id in common
        ]
        path = figures_dir / "replay_vs_direct.svg"
        _scatter_chart(path, "Paired benefit: direct vs replay", "direct B (us/call)", "replay B (us/call)", points)
        created.append(str(path.relative_to(out_dir)))
    return created


def _reduction_heatmap(path: Path, template: str, dtype: str, rows: list[dict[str, Any]]) -> None:
    points = []
    for row in rows:
        shape = row.get("shape") or {}
        try:
            k_value = float(shape.get("K"))
            m_value = float(shape.get("M"))
        except (TypeError, ValueError):
            continue
        if row.get("relative_benefit") is None:
            continue
        points.append(
            {
                "k": k_value,
                "m": m_value,
                "value": float(row["relative_benefit"]) * 100.0,
                "label": row["case_id"],
            }
        )
    if not points:
        return
    ks = sorted({point["k"] for point in points})
    ms = sorted({point["m"] for point in points})
    cell_w = 96
    cell_h = 38
    left, right, top, bottom = 96, 230, 84, 64
    legend_steps = 9
    legend_step_h = 14
    width = max(700, left + cell_w * len(ks) + right)
    height = max(top + cell_h * len(ms) + bottom, top + legend_steps * legend_step_h + 54)
    values = [point["value"] for point in points]
    extreme = max(2.0, max(abs(value) for value in values))
    by_cell = {(point["k"], point["m"]): point for point in points}

    def color(value: float) -> str:
        ratio = max(-1.0, min(1.0, value / extreme))
        if ratio >= 0:
            target = (44, 160, 44)
            amount = ratio
        else:
            target = (214, 39, 40)
            amount = -ratio
        base = (255, 255, 255)
        rgb = tuple(int(round(base[i] * (1 - amount) + target[i] * amount)) for i in range(3))
        return "#%02x%02x%02x" % rgb

    def text_color(value: float) -> str:
        return "#ffffff" if abs(value) / extreme > 0.55 else "#111111"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{escape(template)} reduction benefit">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-family="sans-serif" font-size="19" font-weight="600" fill="#111">{escape(template)} · {escape(dtype)}: B/T_split (%)</text>',
        f'<text x="{left}" y="56" font-family="sans-serif" font-size="13" fill="#555">x = K (reduction length), y = M (output rows), color = relative paired benefit</text>',
    ]
    for k_index, k in enumerate(ks):
        x = left + k_index * cell_w
        parts.append(f'<text x="{x + cell_w / 2}" y="{top - 10}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#222">K={_fmt_size(k)}</text>')
    for m_index, m in enumerate(ms):
        y = top + m_index * cell_h
        parts.append(f'<text x="{left - 10}" y="{y + cell_h / 2 + 5}" text-anchor="end" font-family="sans-serif" font-size="13" fill="#222">M={_fmt_size(m)}</text>')
        for k_index, k in enumerate(ks):
            x = left + k_index * cell_w
            point = by_cell.get((k, m))
            if point is None:
                parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="#f2f2f2" stroke="#ffffff"/>')
                parts.append(f'<text x="{x + cell_w / 2}" y="{y + cell_h / 2 + 5}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#999">n/a</text>')
            else:
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color(point["value"])}" stroke="#ffffff">'
                    f'<title>{escape(point["label"])}: {point["value"]:.4g}%</title></rect>'
                )
                parts.append(
                    f'<text x="{x + cell_w / 2}" y="{y + cell_h / 2 + 5}" text-anchor="middle" font-family="monospace" font-size="12" fill="{text_color(point["value"])}">{point["value"]:.2f}</text>'
                )
    legend_x = left + cell_w * len(ks) + 24
    legend_y = top
    for index in range(legend_steps):
        value = extreme - (2 * extreme) * index / (legend_steps - 1)
        parts.append(f'<rect x="{legend_x}" y="{legend_y + index * legend_step_h}" width="24" height="{legend_step_h}" fill="{color(value)}"/>')
        if index in (0, (legend_steps - 1) // 2, legend_steps - 1):
            parts.append(f'<text x="{legend_x + 30}" y="{legend_y + index * legend_step_h + 11}" font-family="sans-serif" font-size="11" fill="#333">{value:.3g}%</text>')
    parts.append(f'<text x="{legend_x}" y="{legend_y - 10}" font-family="sans-serif" font-size="12" fill="#333">relative benefit</text>')
    parts.append(f'<text x="{left}" y="{height - 14}" font-family="sans-serif" font-size="12" fill="#555">Only measured (K,M) cells are shown; exact case IDs and 95% intervals are in results.csv.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _template_benefit_chart(path: Path, template: str, rows: list[dict[str, Any]]) -> None:
    elementwise = all("N" in (row.get("shape") or {}) for row in rows)
    series: dict[str, list[dict[str, Any]]] = {}
    if elementwise:
        for row in rows:
            name = str(row.get("dtype"))
            series.setdefault(name, []).append(row)
        x_label = "N (elements)"
    else:
        for row in rows:
            shape = row.get("shape") or {}
            name = f"{row.get('dtype')} · M={_fmt_size(shape.get('M'))}"
            series.setdefault(name, []).append(row)
        x_label = "K (reduction length)"

    rendered: list[tuple[str, list[dict[str, Any]]]] = []
    for name, items in sorted(series.items()):
        points = []
        for row in sorted(items, key=lambda item: _x_value(item)):
            relative = row.get("relative_benefit")
            if relative is None:
                continue
            ci = row.get("relative_ci95") or [None, None]
            points.append(
                {
                    "x": _x_value(row),
                    "y": relative * 100.0,
                    "ylo": (ci[0] * 100.0) if ci[0] is not None else None,
                    "yhi": (ci[1] * 100.0) if ci[1] is not None else None,
                    "label": row["case_id"],
                }
            )
        if points:
            rendered.append((name, points))
    if not rendered:
        return
    _line_chart(
        path,
        f"{template}: paired benefit B/T_split with 95% block bootstrap interval",
        x_label,
        "relative benefit (%)",
        rendered,
        band=(-2.0, 2.0),
    )


def _x_value(row: dict[str, Any]) -> float:
    shape = row.get("shape") or {}
    if "N" in shape:
        return float(shape["N"])
    return float(shape.get("K", 0))


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in text).strip("-") or "figure"


def _fmt_size(value: Any) -> str:
    if value is None:
        return "?"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number >= 1_000_000:
        return f"{number / 1_000_000:g}M"
    if number >= 1_000:
        return f"{number / 1_000:g}k"
    return f"{number:g}"


def _nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    span = hi - lo
    raw = span / max(1, count)
    magnitude = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    for factor in (1, 2, 2.5, 5, 10):
        step = magnitude * factor
        if step >= raw:
            break
    start = math.ceil(lo / step) * step
    ticks = []
    value = start
    while value <= hi + step * 0.5:
        ticks.append(value)
        value += step
    return ticks


def _line_chart(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[tuple[str, list[dict[str, Any]]]],
    *,
    band: tuple[float, float] | None = None,
) -> None:
    width, height = 960, 480
    left, right, top, bottom = 82, 190, 64, 74
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [point["x"] for _, points in series for point in points]
    ys = [point["y"] for _, points in series for point in points]
    ylos = [point["ylo"] for _, points in series for point in points if point.get("ylo") is not None]
    yhis = [point["yhi"] for _, points in series for point in points if point.get("yhi") is not None]
    if band:
        ys.extend(band)
    if not xs or not ys:
        return
    x_min, x_max = min(xs), max(xs)
    log_x = x_min > 0 and x_max / x_min > 25
    if log_x:
        x_min_t, x_max_t = math.log10(x_min), math.log10(x_max)
    else:
        x_min_t, x_max_t = x_min, x_max
    if x_max_t == x_min_t:
        x_min_t -= 1.0
        x_max_t += 1.0
    y_min = min(ys + (ylos or []))
    y_max = max(ys + (yhis or []))
    if y_max == y_min:
        y_max = y_min + 1.0
        y_min -= 1.0
    pad = (y_max - y_min) * 0.12
    y_min -= pad
    y_max += pad

    def sx(value: float) -> float:
        t = math.log10(value) if log_x else value
        return left + (t - x_min_t) / (x_max_t - x_min_t) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{escape(title)}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-family="sans-serif" font-size="19" font-weight="600" fill="#111">{escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#c9c9c9"/>',
    ]
    if band:
        y1, y2 = sy(band[1]), sy(band[0])
        parts.append(f'<rect x="{left}" y="{y1}" width="{plot_w}" height="{y2 - y1}" fill="#e8f5e9" opacity="0.8"/>')
        parts.append(f'<text x="{left + 8}" y="{y1 + 16}" font-family="sans-serif" font-size="12" fill="#2e7d32">±2% practical band</text>')
    for tick in _nice_ticks(y_min, y_max, 5):
        y = sy(tick)
        if not (top <= y <= top + plot_h):
            continue
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#eeeeee"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#444">{tick:.4g}</text>')
    zero_y = sy(0.0)
    parts.append(f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + plot_w}" y2="{zero_y:.2f}" stroke="#666" stroke-width="1"/>')

    tick_values = sorted({point["x"] for _, points in series for point in points})
    if len(tick_values) > 9:
        step = max(1, len(tick_values) // 8)
        tick_values = tick_values[::step] + [tick_values[-1]]
    for value in tick_values:
        x = sx(value)
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 5}" stroke="#666"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#444">{_fmt_size(value)}</text>')

    for index, (name, points) in enumerate(series):
        color = PALETTE[index % len(PALETTE)]
        ordered = sorted(points, key=lambda point: point["x"])
        commands = " ".join(f"{sx(point['x']):.2f},{sy(point['y']):.2f}" for point in ordered)
        parts.append(f'<polyline points="{commands}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        for point in ordered:
            x, y = sx(point["x"]), sy(point["y"])
            if point.get("ylo") is not None and point.get("yhi") is not None:
                y1, y2 = sy(point["yhi"]), sy(point["ylo"])
                parts.append(f'<line x1="{x:.2f}" y1="{y1:.2f}" x2="{x:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="1.2" opacity="0.8"/>')
                parts.append(f'<line x1="{x - 4:.2f}" y1="{y1:.2f}" x2="{x + 4:.2f}" y2="{y1:.2f}" stroke="{color}"/>')
                parts.append(f'<line x1="{x - 4:.2f}" y1="{y2:.2f}" x2="{x + 4:.2f}" y2="{y2:.2f}" stroke="{color}"/>')
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"><title>{escape(point["label"])}</title></circle>')
        legend_x = left + plot_w + 16
        legend_y = top + 8 + index * 24
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 32}" y="{legend_y + 4}" font-family="sans-serif" font-size="13" fill="#222">{escape(name)}</text>')

    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 22}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#222">{escape(x_label)}</text>')
    parts.append(f'<text x="20" y="{top + plot_h / 2}" transform="rotate(-90 20 {top + plot_h / 2})" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#222">{escape(y_label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _bar_chart(path: Path, title: str, value_label: str, items: Sequence[tuple[str, float | None]]) -> None:
    items = [(label, value) for label, value in items if value is not None]
    if not items:
        return
    width = 900
    row_h = 34
    height = 90 + row_h * len(items)
    left, right, top = 230, 90, 58
    plot_w = width - left - right
    maximum = max(float(value) for _, value in items) or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{escape(title)}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="30" y="34" font-family="sans-serif" font-size="19" font-weight="600" fill="#111">{escape(title)}</text>',
    ]
    for index, (label, value) in enumerate(items):
        y = top + index * row_h
        bar = max(1.0, plot_w * float(value) / maximum)
        color = PALETTE[index % len(PALETTE)]
        parts.append(f'<text x="{left - 10}" y="{y + 16}" text-anchor="end" font-family="sans-serif" font-size="13" fill="#222">{escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y + 4}" width="{bar:.2f}" height="20" fill="{color}" opacity="0.85"/>')
        parts.append(f'<text x="{left + bar + 8:.2f}" y="{y + 19}" font-family="monospace" font-size="12" fill="#222">{float(value):.4g}</text>')
    parts.append(f'<text x="{left}" y="{height - 14}" font-family="sans-serif" font-size="13" fill="#444">{escape(value_label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _scatter_chart(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    points: Sequence[tuple[float, float, str]],
) -> None:
    if not points:
        return
    width, height = 720, 520
    left, right, top, bottom = 82, 36, 64, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [value for x, y, _ in points for value in (x, y)]
    lo, hi = min(values), max(values)
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    pad = (hi - lo) * 0.08
    lo -= pad
    hi += pad

    def sx(value: float) -> float:
        return left + (value - lo) / (hi - lo) * plot_w

    def sy(value: float) -> float:
        return top + (hi - value) / (hi - lo) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{escape(title)}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-family="sans-serif" font-size="19" font-weight="600" fill="#111">{escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#c9c9c9"/>',
    ]
    for tick in _nice_ticks(lo, hi, 5):
        x, y = sx(tick), sy(tick)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#f0f0f0"/>')
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#f0f0f0"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 20}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#444">{tick:.3g}</text>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#444">{tick:.3g}</text>')
    parts.append(f'<line x1="{sx(lo):.2f}" y1="{sy(lo):.2f}" x2="{sx(hi):.2f}" y2="{sy(hi):.2f}" stroke="#888" stroke-dasharray="5 5"/>')
    for x_value, y_value, label in points:
        parts.append(f'<circle cx="{sx(x_value):.2f}" cy="{sy(y_value):.2f}" r="4" fill="#1f77b4"><title>{escape(label)}</title></circle>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#222">{escape(x_label)}</text>')
    parts.append(f'<text x="20" y="{top + plot_h / 2}" transform="rotate(-90 20 {top + plot_h / 2})" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#222">{escape(y_label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
