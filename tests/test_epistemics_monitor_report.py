from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from agent.epistemics.monitor import EpistemicMonitor
from agent.epistemics.report import build_epistemics_report
from agent.epistemics.types import GateDecision
from memory.db import MemoryDB


class _StubDB:
    def __init__(self) -> None:
        self.audit_calls = []
        self.activity_calls = []
        self.anomaly_calls = []

    def audit_log_create(self, **kwargs):  # type: ignore[no-untyped-def]
        self.audit_calls.append(kwargs)
        return 1

    def log_activity(self, event_type, payload):  # type: ignore[no-untyped-def]
        self.activity_calls.append((event_type, payload))

    def activity_log_list_recent(self, event_type, limit=50):  # type: ignore[no-untyped-def]
        return []

    def get_anomalies(self, user_id, start_date, end_date, limit=100):  # type: ignore[no-untyped-def]
        return []

    def insert_anomaly_events(self, user_id, observed_at, events):  # type: ignore[no-untyped-def]
        self.anomaly_calls.append((user_id, observed_at, events))
        return 1


class TestEpistemicsMonitorReport(unittest.TestCase):
    def test_audit_payload_contains_required_fields(self) -> None:
        db = _StubDB()
        monitor = EpistemicMonitor(db, window_size=50, spike_threshold=0.35, spike_min_samples=10)
        decision = GateDecision(
            intercepted=True,
            user_text="I’m not sure.\n\nWhich one?",
            reasons=("Z_REASON", "A_REASON"),
            hard_reasons=("HARD_B", "HARD_A"),
            score=0.88,
            question="Which one?",
            contract_errors=("ERR_A",),
            candidate_kind="clarify",
            claims_summary=(("memory", 1), ("none", 0), ("tool", 1), ("user", 2)),
            unsupported_claims_count=0,
            claim_provenance_refs=("memory:1", "tool:audit-1", "user:thread-1:u:2"),
        )
        monitor.record("user-1", decision, active_thread_id="thread-7")
        self.assertEqual(1, len(db.audit_calls))
        details = db.audit_calls[0]["details"]
        self.assertEqual("thread-7", details["active_thread_id"])
        self.assertEqual(0.88, details["uncertainty_score"])
        self.assertEqual(["A_REASON", "Z_REASON"], details["reasons"])
        self.assertEqual(["HARD_A", "HARD_B"], details["hard_reasons"])
        self.assertEqual("clarify", details["candidate_kind"])
        self.assertEqual(
            [
                {"support": "memory", "count": 1},
                {"support": "none", "count": 0},
                {"support": "tool", "count": 1},
                {"support": "user", "count": 2},
            ],
            details["claims_summary"],
        )
        self.assertEqual(0, details["unsupported_claims_count"])
        self.assertEqual(
            ["memory:1", "tool:audit-1", "user:thread-1:u:2"],
            details["claim_provenance_refs"],
        )

    def test_report_output_stable_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = MemoryDB(db_path)
            schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql"))
            db.init_schema(schema_path)
            db.log_activity("epistemic_gate", {"intercepted": True, "reasons": ["B", "A"]})
            db.log_activity("epistemic_gate", {"intercepted": True, "reasons": ["A"]})
            db.log_activity("epistemic_gate", {"intercepted": False, "reasons": []})
            db.log_activity("epistemic_gate", {"intercepted": False, "reasons": ["C"]})
            report = build_epistemics_report(db, window_size=50, spike_threshold=0.35)
            expected = "\n".join(
                [
                    "Epistemics report",
                    "window_size: 50",
                    "passes: 2",
                    "intercepts: 2",
                    "rolling_uncertain_rate: 0.500",
                    "spike_flag: false",
                    "top_reasons:",
                    "- A: 2",
                    "- B: 1",
                    "- C: 1",
                ]
            )
            self.assertEqual(expected, report)
            db.close()

    def test_monitor_env_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {"ROLLING_WINDOW_SIZE": "77", "SPIKE_THRESHOLD": "0.21"},
            clear=False,
        ):
            db = _StubDB()
            monitor = EpistemicMonitor(db)
            self.assertEqual(77, monitor.window_size)
            self.assertAlmostEqual(0.21, monitor.spike_threshold, places=6)


if __name__ == "__main__":
    unittest.main()
