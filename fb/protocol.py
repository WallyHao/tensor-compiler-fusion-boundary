"""Frozen measurement protocol. Every field here is hashed into run.json.

Changing any default requires a new run (resume refuses a mismatching hash).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

# Keep the serialized identifier for compatibility with existing runs. GPU
# identity is recorded and checked separately from the protocol version.
PROTOCOL_VERSION = "fb-4090-p1-v2"


@dataclass(frozen=True)
class Protocol:
    version: str = PROTOCOL_VERSION
    seed: int = 17
    input_seeds: tuple[int, ...] = (17, 29, 43, 0)
    blocks: int = 30
    strategies: tuple[str, ...] = ("split", "fused")
    modes: tuple[str, ...] = ("direct", "replay")
    pattern_blocks: tuple[int, int] = (15, 15)  # SF, FS
    target_batch_ms: float = 5.0
    n_margin: float = 1.10
    stabilize_ms: float = 500.0
    max_batch_us: float = 500_000.0
    n_cap: int = 500_000
    trial_iters: int = 32
    warmup_calls: int = 50
    untimed_calls: int = 5
    n_policy: str = "same_n_per_case_mode_from_faster_strategy_v1"
    max_attempts: int = 3  # first attempt + 2 retries
    telemetry_interval_s: float = 1.0
    telemetry_gpu_fields: tuple[str, ...] = (
        "timestamp",
        "name",
        "uuid",
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "clocks.sm",
        "clocks.mem",
        "clocks_throttle_reasons.active",
        "utilization.gpu",
        "memory.used",
    )
    # Calibration
    launch_k: tuple[int, ...] = (1, 8, 32, 128)
    calibration_blocks: int = 20
    calibration_target_ms: float = 2.0
    calibration_assumed_launch_us: float = 5.0
    bw_sizes_bytes: tuple[int, ...] = tuple(1 << b for b in range(8, 30))
    # 256 B .. 512 MiB, doubling; the tail above 64 MiB exists only to get
    # three L2-exceeding points on cards with a large L2 (RTX 4090: 72 MiB).
    bw_blocks: int = 20
    bw_target_ms: float = 2.0
    bw_main_rule: str = "median_of_largest_three_sizes_above_l2; at_least_three_required"
    quality_rules: str = (
        "contaminate_on:external_cuda_process|throttle_state_change;"
        "missing_telemetry_marks_reason_but_keeps_row_v1"
    )
    order_policy: str = "case_shuffle_seed;block_pattern_SF15_FS15_shuffled_v1"

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    def choose_n(self, trial_split_us: float, trial_fused_us: float) -> int:
        fastest = min(trial_split_us, trial_fused_us)
        if not math.isfinite(fastest) or fastest <= 0:
            raise ValueError(f"invalid per-call trial estimate: {fastest}")
        n = math.ceil(self.target_batch_ms * 1000.0 * self.n_margin / fastest)
        n = max(1, min(n, self.n_cap))
        slowest = max(trial_split_us, trial_fused_us)
        while n > 1 and slowest * n > self.max_batch_us:
            n = max(1, n // 2)
        return int(n)

    def validate(self) -> None:
        if self.blocks != sum(self.pattern_blocks):
            raise ValueError("pattern_blocks must sum to blocks")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if len(set(self.strategies)) != 2:
            raise ValueError("exactly two strategies are paired")
        if self.blocks != 30:
            # Documented protocol is 30; smaller values are only for pilot/dry runs
            # and must be recorded in run.json (which happens via the hash).
            pass
