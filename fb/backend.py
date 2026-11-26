"""Backends: real CUDA events, CUDA Graph replay, and a deterministic fake.

The runner only depends on this protocol, so pairing/retry/resume logic is
tested without a GPU.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable


class CandidateError(RuntimeError):
    pass


@dataclass
class PreparedVariant:
    case_id: str
    strategy: str
    meta: dict[str, Any]
    _call: Callable[[], None]
    _check: Callable[[str], dict] | None = None
    _close: Callable[[], None] | None = None
    _call_n: Callable[[int], None] | None = None
    _replay: Callable[[], tuple["PreparedVariant", float]] | None = None
    calls: int = 0

    def call(self) -> None:
        self.calls += 1
        self._call()

    def check(self, input_id: str) -> dict:
        if self._check is None:
            return {"pass": None, "reason": "provider_has_no_check"}
        return self._check(input_id)

    def close(self) -> None:
        if self._close is not None:
            self._close()


class Backend:
    name = "base"
    device = None
    gpu_uuid: str | None = None

    def sync(self) -> None:
        raise NotImplementedError

    def call_n(self, variant: PreparedVariant, n: int) -> None:
        if variant._call_n is not None:
            variant._call_n(n)
            variant.calls += n
        else:
            for _ in range(n):
                variant.call()

    def warmup(self, variant: PreparedVariant, calls: int) -> None:
        self.call_n(variant, calls)
        self.sync()

    def estimate_us(self, variant: PreparedVariant, iters: int) -> float:
        raise NotImplementedError

    def time_batch(self, variant: PreparedVariant, n: int, phase: str = "timed") -> float:
        raise NotImplementedError

    def make_replay(self, variant: PreparedVariant) -> tuple[PreparedVariant, float]:
        raise NotImplementedError

    def device_info(self) -> dict:
        return {"backend": self.name, "device": str(self.device)}


class FakeBackend(Backend):
    """Deterministic virtual clock.

    Each variant contributes ``cost_us`` per call plus a fixed pseudo-noise
    derived from (case, strategy, block, attempt, call index). Failures are
    injected by variant meta ``fail_after``/``fail_times``.
    """

    name = "fake"
    gpu_uuid = "GPU-SYNTHETIC-0000"

    def __init__(self, costs: dict[tuple[str, str], float] | None = None,
                 jitter_us: float = 0.0, replay_launch_saving_us: float = 0.0,
                 seed: int = 0):
        self.costs = costs or {}
        self.jitter_us = jitter_us
        self.replay_launch_saving_us = replay_launch_saving_us
        self.seed = seed
        self._call_counts: dict[tuple[str, str], int] = {}
        self._batch_counts: dict[tuple[str, str], int] = {}
        self.replay_flags: dict[tuple[str, str], bool] = {}
        self.failures: list[dict] = []

    def _key(self, v: PreparedVariant) -> tuple[str, str]:
        return (v.case_id, (v.meta.get("base_strategy") or v.strategy))

    def _cost(self, v: PreparedVariant) -> float:
        key = self._key(v)
        base = self.costs.get(key)
        if base is None:
            base = float(v.meta.get("cost_us", 5.0))
        if self.replay_flags.get(key):
            base -= self.replay_launch_saving_us
        return max(0.1, base)

    def _noise(self, key: tuple[str, str]) -> float:
        if self.jitter_us <= 0:
            return 0.0
        import hashlib

        h = int.from_bytes(
            hashlib.sha256(repr((self.seed, key)).encode()).digest()[:4], "big"
        )
        return ((h % 2001) - 1000) / 1000.0 * self.jitter_us

    def _maybe_fail(self, v: PreparedVariant, batch_idx: int) -> None:
        key = self._key(v)
        fail_on = v.meta.get("fail_batch_indices") or ()
        if batch_idx in set(fail_on):
            self.failures.append(
                {"case_id": v.case_id, "strategy": v.strategy, "batch_idx": batch_idx}
            )
            raise CandidateError(f"injected failure for {v.case_id}/{v.strategy}")

    def sync(self) -> None:
        pass

    def estimate_us(self, v: PreparedVariant, iters: int) -> float:
        key = self._key(v)
        return self._cost(v) + self._noise(key)

    def time_batch(self, v: PreparedVariant, n: int, phase: str = "timed") -> float:
        key = self._key(v)
        idx = self._batch_counts.get(key, 0)
        if phase == "timed":
            self._batch_counts[key] = idx + 1
            self._maybe_fail(v, idx)
        per_call = self._cost(v) + self._noise(key) + 0.01 * (idx % 7)
        return per_call * n / 1000.0

    def make_replay(self, v: PreparedVariant) -> tuple[PreparedVariant, float]:
        key = self._key(v)
        self.replay_flags[key] = True
        meta = dict(v.meta)
        meta["base_strategy"] = key[1]
        meta["submit_mode"] = "replay"
        return (
            PreparedVariant(
                case_id=v.case_id,
                strategy=v.strategy,
                meta=meta,
                _call=v._call,
                _check=v._check,
                _close=v._close,
            ),
            0.0,
        )

    def device_info(self) -> dict:
        return {
            "backend": self.name,
            "device": "fake",
            "gpu_uuid": self.gpu_uuid,
            "synthetic": True,
        }


class CudaBackend(Backend):
    """torch CUDA events around a full batch; optional manual graph replay."""

    name = "cuda"
    ready = False

    def __init__(self, device: str = "cuda:0"):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.current_stream(self.device)
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self.gpu_uuid = None
        try:
            self.gpu_uuid = torch.cuda.get_device_properties(self.device).uuid
            self.gpu_uuid = f"GPU-{self.gpu_uuid}" if self.gpu_uuid else None
        except Exception:
            pass
        self.ready = True

    def sync(self) -> None:
        self.torch.cuda.synchronize(self.device)

    def estimate_us(self, v: PreparedVariant, iters: int) -> float:
        self.sync()
        self._start.record(self.stream)
        self.call_n(v, iters)
        self._end.record(self.stream)
        self._end.synchronize()
        ms = self._start.elapsed_time(self._end)
        if ms <= 0:
            raise CandidateError(f"non-positive elapsed for estimate: {ms}")
        return ms * 1000.0 / iters

    def time_batch(self, v: PreparedVariant, n: int, phase: str = "timed") -> float:
        self.sync()
        self._start.record(self.stream)
        self.call_n(v, n)
        self._end.record(self.stream)
        self._end.synchronize()
        return float(self._start.elapsed_time(self._end))

    def make_replay(self, v: PreparedVariant) -> tuple[PreparedVariant, float]:
        if v._replay is not None:
            return v._replay()
        torch = self.torch
        self.sync()
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(self.stream)
        t0 = time.perf_counter()
        with torch.cuda.stream(side):
            for _ in range(3):
                v.call()
        self.stream.wait_stream(side)
        self.sync()
        graph = torch.cuda.CUDAGraph()
        # Capture on a non-default stream is what torch.cuda.graph does
        # internally; the allocations made by the call are pooled by the graph.
        with torch.cuda.graph(graph):
            v.call()
        self.sync()
        capture_ms = (time.perf_counter() - t0) * 1000.0

        def replay() -> None:
            graph.replay()

        meta = dict(v.meta)
        meta["submit_mode"] = "replay"
        meta["base_strategy"] = v.strategy
        return (
            PreparedVariant(
                case_id=v.case_id,
                strategy=v.strategy,
                meta=meta,
                _call=replay,
                _check=None,
                _close=None,
            ),
            capture_ms,
        )

    def device_info(self) -> dict:
        props = self.torch.cuda.get_device_properties(self.device)
        return {
            "backend": self.name,
            "device": str(self.device),
            "gpu_name": props.name,
            "gpu_uuid": self.gpu_uuid,
            "sm_count": props.multi_processor_count,
            "total_memory_bytes": props.total_memory,
            "l2_cache_bytes": getattr(props, "L2_cache_size", None),
            "capability": list(props.major_minor) if hasattr(props, "major_minor") else [props.major, props.minor],
            "torch_version": self.torch.__version__,
        }
