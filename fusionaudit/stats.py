"""Small, dependency-free statistics used by the frozen analysis.

The routines are intentionally explicit: percentile interpolation, paired block
bootstrap, OLS with rank/condition diagnostics, and the byte-threshold rule.
"""

from __future__ import annotations

import math
import random
from typing import Any, Iterable, Sequence

DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 2026
DEFAULT_PRACTICAL_BAND = 0.02


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def median(values: Sequence[float]) -> float:
    return percentile(values, 50.0)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (q / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def iqr(values: Sequence[float]) -> tuple[float, float]:
    return percentile(values, 25.0), percentile(values, 75.0)


def paired_bootstrap(
    losses: Sequence[float],
    *,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap a mean of paired block gains.

    Each input row is an intact pair (one split and one fused measurement from
    the same block); resampling rows preserves the pairing.
    """
    if not losses:
        return {"n": 0, "iterations": iterations, "seed": seed, "mean": None, "ci95": [None, None]}
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    rng = random.Random(seed)
    n = len(losses)
    samples: list[float] = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += losses[rng.randrange(n)]
        samples.append(total / n)
    return {
        "n": n,
        "iterations": iterations,
        "seed": seed,
        "mean": mean(losses),
        "median": median(losses),
        "ci95": [percentile(samples, 2.5), percentile(samples, 97.5)],
    }


def classify_interval(lo: float | None, hi: float | None, threshold: float) -> str:
    """Map a relative-benefit interval to the pre-registered four-way label."""
    if lo is None or hi is None:
        return "uncertain"
    if lo > threshold:
        return "fuse"
    if hi < -threshold:
        return "split"
    if lo >= -threshold and hi <= threshold:
        return "equivalent"
    return "uncertain"


def threshold_labels(relative_ci: Sequence[float] | None, thresholds: Iterable[float]) -> dict[str, str]:
    if not relative_ci or relative_ci[0] is None or relative_ci[1] is None:
        return {str(thr): "uncertain" for thr in thresholds}
    return {str(thr): classify_interval(relative_ci[0], relative_ci[1], thr) for thr in thresholds}


def _jacobi_eigenvalues(matrix: list[list[float]], tolerance: float = 1e-12) -> list[float]:
    """Eigenvalues of a small symmetric matrix (p <= 3 in this project)."""
    p = len(matrix)
    a = [row[:] for row in matrix]
    for _ in range(100):
        off = 0.0
        for i in range(p):
            for j in range(i + 1, p):
                off += abs(a[i][j])
        if off <= tolerance:
            break
        for i in range(p):
            for j in range(i + 1, p):
                if abs(a[i][j]) < tolerance:
                    continue
                theta = (a[j][j] - a[i][i]) / (2.0 * a[i][j])
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(p):
                    aik, ajk = a[i][k], a[j][k]
                    a[i][k] = c * aik - s * ajk
                    a[j][k] = s * aik + c * ajk
                for k in range(p):
                    aki, akj = a[k][i], a[k][j]
                    a[k][i] = c * aki - s * akj
                    a[k][j] = s * aki + c * akj
    return sorted(a[i][i] for i in range(p))


def fit_linear(
    rows: Sequence[dict[str, float]],
    features: Sequence[str],
    target: str,
) -> dict[str, Any]:
    """No-intercept OLS with rank and condition diagnostics.

    Rank-deficient designs return a basic least-squares solution with free
    coefficients set to zero.  Consumers must consult ``rank`` before treating
    the coefficients as identified.
    """
    if not rows:
        return {
            "features": list(features),
            "n": 0,
            "coefficients": {name: None for name in features},
            "rank": 0,
            "condition_number": None,
            "identifiable": False,
            "reason": "no training rows",
        }
    p = len(features)
    xtx = [[0.0] * p for _ in range(p)]
    xty = [0.0] * p
    for row in rows:
        x = [float(row[name]) for name in features]
        y = float(row[target])
        for i in range(p):
            xty[i] += x[i] * y
            for j in range(p):
                xtx[i][j] += x[i] * x[j]

    augmented = [xtx[i][:] + [xty[i]] for i in range(p)]
    scale = max((abs(xtx[i][i]) for i in range(p)), default=0.0)
    tolerance = max(scale, 1.0) * 1e-10
    pivots: list[int] = []
    row = 0
    for col in range(p):
        best = max(range(row, p), key=lambda r: abs(augmented[r][col]), default=None)
        if best is None or abs(augmented[best][col]) <= tolerance:
            continue
        augmented[row], augmented[best] = augmented[best], augmented[row]
        pivot = augmented[row][col]
        augmented[row] = [value / pivot for value in augmented[row]]
        for r in range(p):
            if r == row:
                continue
            factor = augmented[r][col]
            if abs(factor) <= tolerance:
                continue
            augmented[r] = [a - factor * b for a, b in zip(augmented[r], augmented[row])]
        pivots.append(col)
        row += 1
        if row == p:
            break

    coefficients = [0.0] * p
    for i, col in enumerate(pivots):
        coefficients[col] = augmented[i][p]

    rank = len(pivots)
    eigenvalues = [value for value in _jacobi_eigenvalues(xtx) if value > 0.0]
    condition = None
    if rank == p and len(eigenvalues) == p:
        condition = math.sqrt(max(eigenvalues) / min(eigenvalues))

    return {
        "features": list(features),
        "n": len(rows),
        "coefficients": {name: coefficients[i] for i, name in enumerate(features)},
        "rank": rank,
        "condition_number": condition,
        "identifiable": rank == p,
        "reason": None if rank == p else "rank deficient; free coefficients set to zero",
    }


def predict_linear(model: dict[str, Any], row: dict[str, float]) -> float | None:
    if not model.get("identifiable"):
        # A basic solution is still defined and is reported; it is not used as
        # an identified model unless the fold curve is full rank.
        pass
    coefficients = model.get("coefficients") or {}
    total = 0.0
    for name in model.get("features", []):
        value = row.get(name)
        coefficient = coefficients.get(name)
        if value is None or coefficient is None:
            return None
        total += coefficient * float(value)
    return total


def regret_us(t_split: float, t_fused: float, chosen: str) -> float:
    return max(0.0, t_fused - t_split) if chosen == "fused" else max(0.0, t_split - t_fused)


def fit_byte_threshold(train_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Enumerate adjacent midpoints and the two constant endpoints.

    Decision rule: ``logical_saved_bytes > tau`` means fuse.  Loss is the mean
    absolute decision loss (regret) on the training fold; ties choose the larger
    tau, i.e. the more conservative (split-preferring) threshold.
    """
    usable = [row for row in train_rows if row.get("logical_saved_bytes") is not None]
    if not usable:
        return {"tau": None, "loss_us": None, "candidates": 0, "reason": "no training rows with byte counts"}
    sizes = sorted({float(row["logical_saved_bytes"]) for row in usable})
    candidates: list[float] = [sizes[0] - 1.0]
    for left, right in zip(sizes, sizes[1:]):
        candidates.append((left + right) / 2.0)
    candidates.append(sizes[-1] + 1.0)

    best: tuple[float, float] | None = None
    for tau in candidates:
        losses = []
        for row in usable:
            chosen = "fused" if float(row["logical_saved_bytes"]) > tau else "split"
            losses.append(regret_us(float(row["T_split_us"]), float(row["T_fused_us"]), chosen))
        loss = mean(losses)
        # min loss, then larger tau
        if best is None or loss < best[0] - 1e-15 or (abs(loss - best[0]) <= 1e-15 and tau > best[1]):
            best = (loss, tau)
    assert best is not None
    return {
        "tau": best[1],
        "loss_us": best[0],
        "candidates": len(candidates),
        "reason": None,
        "rule": "logical_saved_bytes > tau => fused",
    }


def aggregate_metrics(decisions: Sequence[dict[str, Any]], denominator: int) -> dict[str, Any]:
    predicted = [row for row in decisions if row.get("predicted") in {"split", "fused"}]
    regrets = [float(row["regret_us"]) for row in predicted]
    relative = [float(row["regret_relative"]) for row in predicted if row.get("regret_relative") is not None]
    clear = [row for row in predicted if row.get("actual_label") in {"split", "fuse"}]
    correct = sum(1 for row in clear if _label_strategy(row["actual_label"]) == row["predicted"])
    by_template: dict[str, list[float]] = {}
    for row in predicted:
        by_template.setdefault(str(row.get("template")), []).append(float(row["regret_us"]))
    return {
        "denominator": denominator,
        "n_predicted": len(predicted),
        "coverage": (len(predicted) / denominator) if denominator else None,
        "n_clear": len(clear),
        "n_correct": correct,
        "sign_accuracy": (correct / len(clear)) if clear else None,
        "mean_regret_us": mean(regrets) if regrets else None,
        "max_regret_us": max(regrets) if regrets else None,
        "mean_regret_relative": mean(relative) if relative else None,
        "template_macro_regret_us": mean([mean(values) for values in by_template.values()]) if by_template else None,
        "template_max_regret_us": max((max(values) for values in by_template.values()), default=None),
    }


def _label_strategy(label: str) -> str:
    return "fused" if label == "fuse" else "split"
