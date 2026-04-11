from __future__ import annotations

import os
import tempfile
import sqlite3
import unittest

from agent.packs.state_truth import build_pack_state_snapshot
from agent.runtime_contract import normalize_user_facing_status


class _BusyDiscovery:
    def list_sources(self) -> list[dict[str, object]]:
        raise sqlite3.OperationalError("database is locked")

    def list_packs(self, source_id: str) -> dict[str, object]:  # pragma: no cover - never reached
        return {"packs": [], "from_cache": False, "stale": False}


class _StaticDiscovery:
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


class _StaticPackStore:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def list_external_packs(self) -> list[dict[str, object]]:
        return list(self._rows)


class TestObservabilitySelfExplanation(unittest.TestCase):
    def test_runtime_failure_explanation_includes_reason_and_next_step(self) -> None:
        bootstrap = normalize_user_facing_status(
            ready=False,
            bootstrap_required=True,
            failure_code="no_chat_model",
            provider=None,
            model=None,
            local_providers={"ollama"},
        )
        self.assertEqual("Initializing", bootstrap["state_label"])
        self.assertEqual("runtime_initializing", bootstrap["recovery"]["kind"])
        self.assertIn("Startup is still in progress.", bootstrap["summary"])
        self.assertIn("Wait for startup", bootstrap["summary"])
        self.assertIn("startup", str(bootstrap["reason"]).lower())
        self.assertIn("wait", str(bootstrap["next_step"]).lower())

        blocked = normalize_user_facing_status(
            ready=False,
            bootstrap_required=False,
            failure_code="config_load_failed",
            provider=None,
            model=None,
            local_providers={"ollama"},
        )
        self.assertEqual("Blocked", blocked["state_label"])
        self.assertEqual("runtime_blocked", blocked["recovery"]["kind"])
        self.assertIn("System is blocked.", blocked["summary"])
        self.assertIn("required dependency or configuration is missing", blocked["summary"].lower())
        self.assertIn("Fix the blocker", blocked["summary"])
        self.assertIn("configuration", str(blocked["reason"]).lower())
        self.assertIn("fix", str(blocked["next_step"]).lower())

    def test_pack_state_snapshot_explains_missing_files_and_invalid_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            existing = os.path.join(tmpdir, "existing")
            open(existing, "w", encoding="utf-8").close()
            missing = os.path.join(tmpdir, "missing")
            pack_store = _StaticPackStore(
                [
                    {
                        "pack_id": "pack.voice.missing_files",
                        "name": "Missing Files Voice",
                        "status": "normalized",
                        "enabled": True,
                        "normalized_path": missing,
                        "canonical_pack": {
                            "pack_identity": {"canonical_id": "pack.voice.missing_files"},
                            "display_name": "Missing Files Voice",
                            "source": {"name": "Local"},
                            "capabilities": {"declared": ["voice_output"]},
                        },
                        "review_envelope": {"pack_name": "Missing Files Voice"},
                    },
                    {
                        "pack_id": "pack.voice.invalid_metadata",
                        "name": "Invalid Metadata Voice",
                        "status": "normalized",
                        "enabled": True,
                        "normalized_path": existing,
                        "canonical_pack": {},
                        "review_envelope": {"pack_name": "Invalid Metadata Voice"},
                    },
                ]
            )
            discovery = _StaticDiscovery(
                sources=[{"id": "local", "name": "Local Catalog", "kind": "local_catalog", "enabled": True}],
                packs={"local": []},
            )

            snapshot = build_pack_state_snapshot(pack_store=pack_store, discovery=discovery)
            self.assertTrue(snapshot["ok"])

            cards = {row["name"]: row for row in snapshot["packs"]}
            self.assertEqual("pack_missing_files", cards["Missing Files Voice"]["recovery"]["kind"])
            self.assertIn("missing", cards["Missing Files Voice"]["recovery"]["reason"].lower())
            self.assertIn("Reinstall", cards["Missing Files Voice"]["recovery"]["next_step"])
            self.assertEqual("pack_invalid_metadata", cards["Invalid Metadata Voice"]["recovery"]["kind"])
            self.assertIn("incomplete", cards["Invalid Metadata Voice"]["recovery"]["reason"].lower())
            self.assertIn("Rebuild the manifest", cards["Invalid Metadata Voice"]["recovery"]["next_step"])

            self.assertEqual("Ready", snapshot["state_label"])
            self.assertIsNone(snapshot["recovery"])
            self.assertEqual(2, snapshot["summary"]["installed"])
            self.assertEqual(2, snapshot["summary"]["total"])

    def test_pack_state_snapshot_reports_degraded_sources_clearly(self) -> None:
        pack_store = _StaticPackStore([])
        snapshot = build_pack_state_snapshot(pack_store=pack_store, discovery=_BusyDiscovery())
        self.assertTrue(snapshot["ok"])
        self.assertEqual("Persistence busy", snapshot["state_label"])
        self.assertEqual("db_busy", snapshot["recovery"]["kind"])
        self.assertIn("busy", snapshot["recovery"]["summary"].lower())
        self.assertIn("retry", snapshot["recovery"]["next_step"].lower())
        self.assertTrue(snapshot["source_warnings"])


if __name__ == "__main__":
    unittest.main()
