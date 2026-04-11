from __future__ import annotations

import os
import tempfile
import unittest

from agent.api_server import AgentRuntime
from agent.failure_ux import build_failure_recovery
from agent.packs.state_truth import build_pack_state_snapshot
from agent.runtime_truth_service import RuntimeTruthService


class _FakePackStore:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def list_external_packs(self) -> list[dict[str, object]]:
        return list(self._rows)


class _FakeDiscovery:
    def __init__(self, sources: list[dict[str, object]], packs: dict[str, list[dict[str, object]]]) -> None:
        self._sources = sources
        self._packs = packs

    def list_sources(self) -> list[dict[str, object]]:
        return list(self._sources)

    def list_packs(self, source_id: str) -> dict[str, object]:
        return {
            "packs": list(self._packs.get(source_id, [])),
            "from_cache": False,
            "stale": False,
        }


class TestFailureRecoveryUx(unittest.TestCase):
    def test_canonical_recovery_templates_are_actionable(self) -> None:
        runtime = build_failure_recovery("runtime_initializing")
        self.assertEqual("runtime_initializing", runtime["kind"])
        self.assertEqual("Initializing", runtime["state_label"])
        self.assertIn("Startup", runtime["reason"])
        self.assertIn("Wait for startup", runtime["next_step"])

        preview = build_failure_recovery("pack_available_previewable")
        self.assertEqual("pack_available_previewable", preview["kind"])
        self.assertEqual("Available", preview["state_label"])
        self.assertIn("preview", preview["summary"].lower())
        self.assertIn("install", preview["next_step"].lower())

        unconfirmed = build_failure_recovery("pack_task_unconfirmed")
        self.assertEqual("pack_task_unconfirmed", unconfirmed["kind"])
        self.assertEqual("Installed · Healthy", unconfirmed["state_label"])
        self.assertIn("task usability is not confirmed", unconfirmed["summary"].lower())
        self.assertIn("preview", unconfirmed["next_step"].lower())

        approval = build_failure_recovery("confirm_token_expired")
        self.assertEqual("confirm_token_expired", approval["kind"])
        self.assertIn("expired", approval["summary"].lower())
        self.assertIn("preview", approval["next_step"].lower())

        discovery = build_failure_recovery("discovery_unavailable")
        self.assertEqual("discovery_unavailable", discovery["kind"])
        self.assertIn("discovery is unavailable", discovery["summary"].lower())
        self.assertIn("retry later", discovery["next_step"].lower())

    def test_pack_state_snapshot_uses_consistent_recovery_kinds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            enabled_path = os.path.join(tmpdir, "enabled")
            blocked_path = os.path.join(tmpdir, "blocked")
            open(enabled_path, "w", encoding="utf-8").close()
            open(blocked_path, "w", encoding="utf-8").close()

            pack_store = _FakePackStore(
                [
                    {
                        "pack_id": "pack.voice.local_fast",
                        "name": "Local Voice",
                        "status": "normalized",
                        "enabled": False,
                        "normalized_path": enabled_path,
                        "canonical_pack": {
                            "pack_identity": {"canonical_id": "pack.voice.local_fast"},
                            "display_name": "Local Voice",
                            "source": {"name": "Local"},
                            "capabilities": {"declared": ["voice_output"], "summary": "Local speech output."},
                        },
                        "review_envelope": {"pack_name": "Local Voice"},
                    },
                    {
                        "pack_id": "pack.voice.local_healthy",
                        "name": "Studio Voice",
                        "status": "normalized",
                        "enabled": True,
                        "normalized_path": enabled_path,
                        "canonical_pack": {
                            "pack_identity": {"canonical_id": "pack.voice.local_healthy"},
                            "display_name": "Studio Voice",
                            "source": {"name": "Local"},
                            "capabilities": {"declared": ["voice_output"], "summary": "Studio speech output."},
                        },
                        "review_envelope": {"pack_name": "Studio Voice"},
                    },
                    {
                        "pack_id": "pack.voice.blocked",
                        "name": "Blocked Voice",
                        "status": "blocked",
                        "enabled": True,
                        "normalized_path": blocked_path,
                        "canonical_pack": {
                            "pack_identity": {"canonical_id": "pack.voice.blocked"},
                            "display_name": "Blocked Voice",
                            "source": {"name": "Local"},
                            "capabilities": {"declared": ["voice_output"], "summary": "Blocked speech output."},
                        },
                        "review_envelope": {"pack_name": "Blocked Voice", "why_risk": ["missing GPU acceleration"]},
                        "risk_report": {"blocked_reason": "missing GPU acceleration"},
                    },
                ]
            )
            discovery = _FakeDiscovery(
                sources=[{"id": "local", "name": "Local Catalog", "kind": "local_catalog", "enabled": True}],
                packs={
                    "local": [
                        {
                            "remote_id": "voice-local-preview",
                            "name": "Voice Preview",
                            "summary": "Lightweight voice support.",
                            "artifact_type_hint": "portable_text_skill",
                            "installable_by_current_policy": True,
                            "preview_available": True,
                        },
                        {
                            "remote_id": "voice-blocked-preview",
                            "name": "Voice Blocked",
                            "summary": "Voice support requiring more resources.",
                            "artifact_type_hint": "native_code_pack",
                            "installable_by_current_policy": False,
                            "blocked_reason": "missing GPU acceleration",
                        },
                    ]
                },
            )

            snapshot = build_pack_state_snapshot(pack_store=pack_store, discovery=discovery)
            self.assertTrue(snapshot["ok"])
            installed = {row["name"]: row for row in snapshot["packs"]}
            available = {row["name"]: row for row in snapshot["available_packs"]}

            self.assertEqual("pack_disabled", installed["Local Voice"]["recovery"]["kind"])
            self.assertIn("Enable it", installed["Local Voice"]["recovery"]["next_step"])

            self.assertEqual("pack_task_unconfirmed", installed["Studio Voice"]["recovery"]["kind"])
            self.assertIn("preview", installed["Studio Voice"]["recovery"]["next_step"].lower())

            self.assertEqual("pack_available_previewable", available["Voice Preview"]["recovery"]["kind"])
            self.assertIn("Open the preview", available["Voice Preview"]["recovery"]["next_step"])

            self.assertEqual("pack_blocked", available["Voice Blocked"]["recovery"]["kind"])
            self.assertIn("blocked", available["Voice Blocked"]["recovery"]["reason"].lower())

            self.assertEqual(5, snapshot["summary"]["total"])
            self.assertEqual(3, snapshot["summary"]["installed"])
            self.assertEqual(1, snapshot["summary"]["available"])

    def test_api_confirm_required_payload_includes_canonical_recovery(self) -> None:
        payload = AgentRuntime._confirm_required_payload(
            what_happened="I need a preview before I can continue.",
            why="There is no pending plan to confirm.",
            next_action="Request a new preview.",
        )
        self.assertFalse(payload["ok"])
        self.assertEqual("confirm_required", payload["error_kind"])
        self.assertEqual("confirm_plan_missing", payload["recovery"]["kind"])
        self.assertIn("preview", payload["recovery"]["next_step"].lower())

    def test_api_failure_recovery_mapping_is_stable_for_approval_states(self) -> None:
        cases = {
            "confirm_token_stale": "confirm_token_stale",
            "confirm_token_consumed": "confirm_token_consumed",
            "confirm_token_expired": "confirm_token_expired",
            "confirm_plan_missing": "confirm_plan_missing",
            "confirm_token_mismatch": "confirm_token_mismatch",
            "confirm_downstream_failed": "confirm_downstream_failed",
        }
        for error, expected_kind in cases.items():
            with self.subTest(error=error):
                recovery = AgentRuntime._failure_recovery_for_error(
                    error=error,
                    error_kind="needs_clarification",
                    message="message",
                    next_action="Request a new preview.",
                    why="why",
                )
                self.assertEqual(expected_kind, recovery["kind"])
                self.assertTrue(str(recovery["next_step"] or "").strip())

    def test_runtime_failure_recovery_helper_is_specific(self) -> None:
        initializing = RuntimeTruthService._runtime_failure_recovery(
            ready={"failure_code": None, "runtime_status": {"runtime_mode": "degraded"}},
            llm_status={},
            normalized_status={"runtime_mode": "degraded"},
            runtime_mode="degraded",
            startup_phase="starting",
        )
        self.assertEqual("runtime_initializing", initializing["kind"])
        self.assertIn("startup", initializing["reason"].lower())

        blocked = RuntimeTruthService._runtime_failure_recovery(
            ready={"failure_code": "config_load_failed", "runtime_status": {"runtime_mode": "blocked"}},
            llm_status={},
            normalized_status={"runtime_mode": "blocked", "failure_code": "config_load_failed"},
            runtime_mode="blocked",
            startup_phase="ready",
        )
        self.assertEqual("runtime_blocked", blocked["kind"])
        self.assertIn("fix the blocker", blocked["next_step"].lower())

        dependency = RuntimeTruthService._runtime_failure_recovery(
            ready={"failure_code": None, "runtime_status": {"runtime_mode": "degraded"}},
            llm_status={"active_provider_health": {"status": "down"}},
            normalized_status={"runtime_mode": "degraded"},
            runtime_mode="degraded",
            startup_phase="ready",
        )
        self.assertEqual("dependency_unavailable", dependency["kind"])
        self.assertIn("retry later", dependency["next_step"].lower())


if __name__ == "__main__":
    unittest.main()
