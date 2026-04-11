from __future__ import annotations

from pathlib import Path
import unittest

from agent.packs.state_truth import normalize_available_pack_truth, normalize_installed_pack_truth


class TestPackStateTruth(unittest.TestCase):
    def test_installed_normalized_pack_without_explicit_enablement_is_healthy_but_task_unconfirmed(self) -> None:
        truth = normalize_installed_pack_truth(
            {
                "status": "normalized",
                "normalized_path": str(Path(__file__).resolve()),
                "canonical_pack": {
                    "display_name": "Local Voice",
                    "source": {"name": "Local"},
                    "pack_identity": {"canonical_id": "pack.voice.local_fast"},
                    "capabilities": {"declared": ["voice_output"]},
                },
            }
        )
        self.assertEqual("installed", truth["install_state"])
        self.assertEqual("unknown", truth["activation_state"])
        self.assertEqual("healthy", truth["health_state"])
        self.assertEqual("compatible", truth["compatibility_state"])
        self.assertEqual("task_unconfirmed", truth["usability_state"])
        self.assertTrue(truth["installed"])
        self.assertIsNone(truth["enabled"])
        self.assertTrue(truth["healthy"])
        self.assertTrue(truth["machine_usable"])
        self.assertFalse(truth["task_usable"])
        self.assertEqual("Installed · Healthy", truth["state_label"])
        self.assertIn("task usability is not confirmed", truth["status_note"])

    def test_installed_disabled_pack_stays_healthy_but_not_machine_usable(self) -> None:
        truth = normalize_installed_pack_truth(
            {
                "status": "normalized",
                "enabled": False,
                "normalized_path": str(Path(__file__).resolve()),
                "canonical_pack": {
                    "display_name": "Local Voice",
                    "source": {"name": "Local"},
                    "pack_identity": {"canonical_id": "pack.voice.local_fast"},
                    "capabilities": {"declared": ["voice_output"]},
                },
            }
        )
        self.assertEqual("disabled", truth["activation_state"])
        self.assertEqual("healthy", truth["health_state"])
        self.assertEqual("compatible", truth["compatibility_state"])
        self.assertEqual("unusable", truth["usability_state"])
        self.assertFalse(truth["enabled"])
        self.assertTrue(truth["healthy"])
        self.assertFalse(truth["machine_usable"])
        self.assertFalse(truth["task_usable"])
        self.assertEqual("Installed · Disabled", truth["state_label"])
        self.assertEqual("Installed, but disabled.", truth["status_note"])

    def test_installed_partial_and_blocked_packs_stay_distinct(self) -> None:
        limited = normalize_installed_pack_truth(
            {
                "status": "partial_safe_import",
                "enabled": False,
                "normalized_path": str(Path(__file__).resolve()),
                "canonical_pack": {
                    "display_name": "Local Voice",
                    "source": {"name": "Local"},
                    "pack_identity": {"canonical_id": "pack.voice.local_fast"},
                    "capabilities": {"declared": ["voice_output"]},
                },
            }
        )
        blocked = normalize_installed_pack_truth(
            {
                "status": "blocked",
                "enabled": False,
                "normalized_path": str(Path(__file__).resolve()),
                "canonical_pack": {
                    "display_name": "Blocked Voice",
                    "source": {"name": "Local"},
                    "pack_identity": {"canonical_id": "pack.voice.blocked"},
                    "capabilities": {"declared": ["voice_output"]},
                },
                "risk_report": {"blocked_reason": "missing GPU acceleration"},
            }
        )
        self.assertEqual("degraded", limited["health_state"])
        self.assertEqual("unconfirmed", limited["compatibility_state"])
        self.assertEqual("unusable", limited["usability_state"])
        self.assertEqual("Installed · Limited", limited["state_label"])
        self.assertEqual("failing", blocked["health_state"])
        self.assertEqual("blocked", blocked["compatibility_state"])
        self.assertEqual("unusable", blocked["usability_state"])
        self.assertEqual("Installed · Blocked", blocked["state_label"])
        self.assertEqual("missing GPU acceleration", blocked["blocker"])

    def test_discovered_available_and_blocked_listings_keep_install_and_compatibility_separate(self) -> None:
        available = normalize_available_pack_truth(
            {"id": "local", "name": "Local Catalog", "kind": "local_catalog"},
            {
                "remote_id": "voice-pack",
                "name": "Local Voice",
                "artifact_type_hint": "portable_text_skill",
                "installable_by_current_policy": True,
                "preview_available": True,
                "tags": ["voice_output"],
            },
        )
        blocked = normalize_available_pack_truth(
            {"id": "remote", "name": "Remote Catalog", "kind": "github_repo"},
            {
                "remote_id": "camera-pack",
                "name": "Robot Camera",
                "artifact_type_hint": "native_code_pack",
                "installable_by_current_policy": False,
                "install_block_reason_if_known": "missing GPU acceleration",
                "preview_available": False,
                "tags": ["camera_feed"],
            },
        )
        self.assertEqual("previewable", available["discovery_state"])
        self.assertEqual("installable", available["install_state"])
        self.assertEqual("unknown", available["health_state"])
        self.assertEqual("unconfirmed", available["compatibility_state"])
        self.assertEqual("unknown", available["usability_state"])
        self.assertFalse(available["machine_usable"])
        self.assertFalse(available["task_usable"])
        self.assertEqual("Available", available["state_label"])
        self.assertEqual("Blocked", blocked["state_label"])
        self.assertEqual("failing", blocked["health_state"])
        self.assertEqual("blocked", blocked["compatibility_state"])
        self.assertEqual("unusable", blocked["usability_state"])
        self.assertEqual("missing GPU acceleration", blocked["blocker"])


if __name__ == "__main__":
    unittest.main()
