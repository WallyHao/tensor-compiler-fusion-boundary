"""Pre-registered analysis configuration and feature whitelist."""

from __future__ import annotations

ANALYSIS_VERSION = "1.0.0"

DEFAULT_EXPECTED_BLOCKS = 30
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 2026
PRACTICAL_BAND = 0.02
SENSITIVITY_BANDS = (0.01, 0.02, 0.05)

# Only these three quantities may enter a predictive model.  Everything else
# (especially fused timing, its difference from split, measured counters or
# post-run occupancy) is a label or a diagnostic.
FEATURE_WHITELIST = ("x1_us", "x2_us", "x3_us")
FORBIDDEN_MODEL_FIELDS = (
    "T_fused_us",
    "fused_us",
    "B_us",
    "relative_benefit",
    "T_chosen_us",
    "regret_us",
    "achieved_occupancy",
    "dram_bytes_measured",
    "l2_bytes_measured",
    "best_measured_kernel",
)

FEATURE_UNITS = {
    "x1_us": "us/call: (split static launches - fused static launches) * launch calibration L",
    "x2_us": "us/call: logical saved bytes / calibrated bandwidth BW",
    "x3_us": "us/call: T_split * (o_split - o_fused), o from compile-time occupancy proxy",
}

VALID_QUALITY = {"valid", "ok", "good", "pass", "passed"}
VALID_CASE_STATUS = {"valid", "planned"}
VALID_MODES = {"direct", "replay"}
VALID_STRATEGIES = {"default", "split", "fused"}

LABEL_ORDER = ("fuse", "split", "equivalent", "uncertain")
