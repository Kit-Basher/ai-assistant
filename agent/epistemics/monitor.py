from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agent.epistemics.types import GateDecision


class EpistemicMonitor:
    def __init__(
        self,
        db: Any,
        window_size: int = 50,
        spike_threshold: float = 0.40,
        spike_min_samples: int = 10,
    ) -> None:
        self.db = db
        self.window_size = int(window_size)
        self.spike_threshold = float(spike_threshold)
        self.spike_min_samples = int(spike_min_samples)

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def record(self, user_id: str, decision: GateDecision) -> None:
        now_iso = self._now_iso()
        status = "intercepted" if decision.intercepted else "passed"
        details = {
            "event_type": "epistemic_gate",
            "intercepted": bool(decision.intercepted),
            "score": float(decision.score),
            "reasons": list(decision.reasons),
            "hard_reasons": list(decision.hard_reasons),
            "contract_errors": list(decision.contract_errors),
        }
        try:
            self.db.audit_log_create(
                user_id=user_id,
                action_type="epistemic_gate",
                action_id=f"gate:{now_iso}",
                status=status,
                details=details,
                created_at=now_iso,
            )
        except Exception:
            pass

        try:
            self.db.log_activity(
                "epistemic_gate",
                {
                    "user_id": user_id,
                    "intercepted": bool(decision.intercepted),
                    "score": float(decision.score),
                    "reasons": list(decision.reasons),
                    "hard_reasons": list(decision.hard_reasons),
                },
            )
        except Exception:
            pass

        try:
            recent = self.db.activity_log_list_recent("epistemic_gate", limit=self.window_size)
        except Exception:
            recent = []

        if not recent:
            return

        intercept_flags = []
        for row in recent:
            payload = row.get("payload") or {}
            intercept_flags.append(bool(payload.get("intercepted")))
        if not intercept_flags:
            return

        uncertainty_rate = sum(1 for val in intercept_flags if val) / float(len(intercept_flags))
        if len(intercept_flags) < self.spike_min_samples or uncertainty_rate < self.spike_threshold:
            return

        local_date = now_iso[:10]
        try:
            existing = self.db.get_anomalies(user_id, local_date, local_date, limit=100)
        except Exception:
            existing = []
        if any(row.get("anomaly_key") == "epistemic_uncertainty_rate_spike" for row in existing):
            return

        event = {
            "snapshot_id": None,
            "source": "epistemics",
            "anomaly_key": "epistemic_uncertainty_rate_spike",
            "severity": "warn",
            "message": f"Epistemic uncertainty rate spike: {uncertainty_rate:.2f} over {len(intercept_flags)} replies",
            "metric_name": "uncertainty_rate",
            "metric_value": float(f"{uncertainty_rate:.6f}"),
            "metric_unit": "ratio",
            "context": {
                "window_size": len(intercept_flags),
                "spike_threshold": self.spike_threshold,
            },
        }
        try:
            self.db.insert_anomaly_events(user_id=user_id, observed_at=now_iso, events=[event])
        except Exception:
            pass

