"""Telemetry window assessment without a GPU."""
from __future__ import annotations

import unittest

from fb.telemetry import TelemetrySampler, our_process_ids


def gpu(ts, mask="0x0000000000000000", temp=45, power=100):
    return {
        "kind": "gpu", "ts_unix": ts, "ts_utc": "x", "source": "test",
        "temperature.gpu": str(temp), "power.draw": str(power),
        "clocks_throttle_reasons.active": mask,
    }


def apps(ts, pids):
    return {
        "kind": "apps", "ts_unix": ts, "ts_utc": "x", "source": "test",
        "processes": [{"pid": p, "process_name": f"p{p}", "used_memory": "1"} for p in pids],
    }


class TelemetryAssessmentTest(unittest.TestCase):
    def sampler(self, rows, available=True):
        s = TelemetrySampler(None, 1.0, 0)
        s.available = available
        s.rows = rows
        return s

    def test_idle_bit_flip_is_not_contamination(self):
        s = self.sampler([gpu(0.0, "0x1"), gpu(0.5, "0x0")])
        r = s.assess(0.2, 0.4, {100})
        self.assertEqual(r["quality"], "ok", r)

    def test_power_cap_transition_inside_window_contaminates(self):
        s = self.sampler([gpu(0.0, "0x0"), gpu(0.3, "0x4")])
        r = s.assess(0.2, 0.4, {100})
        self.assertEqual(r["quality"], "contaminated")
        self.assertIn("throttle_state_change", r["reasons"])

    def test_external_process_contaminates(self):
        s = self.sampler([gpu(0.0), gpu(0.3), apps(0.0, [100]), apps(0.3, [100, 999])])
        r = s.assess(0.1, 0.2, {100})
        self.assertEqual(r["quality"], "contaminated")
        self.assertIn("external_cuda_process", r["reasons"])

    def test_own_process_is_not_external(self):
        s = self.sampler([gpu(0.0), gpu(0.3), apps(0.0, [100])])
        r = s.assess(0.1, 0.2, {100})
        self.assertEqual(r["quality"], "ok", r)

    def test_stale_bracket_does_not_flag_transition(self):
        # sample 5 s before the block is not evidence about a 10 ms window
        s = self.sampler([gpu(3.0, "0x0"), gpu(8.0, "0x4")])
        r = s.assess(5.01, 5.02, {100})
        self.assertEqual(r["quality"], "ok", r)
        self.assertIn("telemetry_gap", r["reasons"])

    def test_missing_telemetry_does_not_block(self):
        s = self.sampler([], available=False)
        r = s.assess(0, 1, set())
        self.assertEqual(r["quality"], "ok")
        self.assertIn("telemetry_unavailable", r["reasons"])

    def test_process_id_helper(self):
        self.assertIn(__import__("os").getpid(), our_process_ids())


if __name__ == "__main__":
    unittest.main()
