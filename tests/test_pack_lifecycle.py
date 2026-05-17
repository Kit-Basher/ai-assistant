from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from agent.packs.lifecycle import PackLifecycleService, PackLifecycleState, render_lifecycle_response


def _pack_row(
    *,
    status: str = "normalized",
    approved: bool = False,
    enabled: bool | None = None,
    adapters: list[dict[str, object]] | None = None,
    configuration: dict[str, object] | None = None,
    normalized_path: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "pack_id": "pack.youtube-history-search",
        "name": "YouTube History Search",
        "status": status,
        "enabled": enabled,
        "approved": approved,
        "normalized_path": normalized_path,
        "canonical_pack": {
            "display_name": "YouTube History Search",
            "pack_identity": {"canonical_id": "pack.youtube-history-search", "content_hash": "hash-1"},
            "source": {"origin": "generated_scaffold"},
            "capabilities": {"declared": ["youtube_history_search"]},
        },
    }
    if adapters is not None:
        row["canonical_pack"]["managed_adapters"] = adapters  # type: ignore[index]
    if configuration is not None:
        row["configuration"] = configuration
    return row


class TestPackLifecycleService(unittest.TestCase):
    def setUp(self) -> None:
        self.service = PackLifecycleService()

    def test_missing_without_candidate_has_discovery_next_step(self) -> None:
        result = self.service.evaluate(capability="robot_arm_control")

        self.assertEqual(PackLifecycleState.MISSING, result.state)
        self.assertFalse(result.usable)
        self.assertEqual("discovery", result.missing_gate)
        self.assertEqual("discover_or_scaffold", result.next_step.action)

    def test_catalog_candidate_is_discovered_not_usable(self) -> None:
        result = self.service.evaluate(
            capability="voice_output",
            catalog_pack={"remote_id": "voice-pack", "name": "Voice Pack"},
        )

        self.assertEqual(PackLifecycleState.DISCOVERED, result.state)
        self.assertFalse(result.usable)
        self.assertEqual("preview", result.missing_gate)
        self.assertTrue(result.required_confirmation)

    def test_catalog_preview_needs_quarantine_import(self) -> None:
        result = self.service.evaluate(
            catalog_preview={"listing": {"remote_id": "voice-pack", "name": "Voice Pack"}},
        )

        self.assertEqual(PackLifecycleState.PREVIEWED, result.state)
        self.assertEqual("quarantine", result.missing_gate)
        self.assertEqual("import_for_review", result.next_step.action)

    def test_scaffold_offered_then_previewed(self) -> None:
        preview = {"scaffold_id": "youtube-history-search", "capability": "youtube_history_search", "title": "YouTube History Search"}

        offered = self.service.evaluate(capability="youtube_history_search", scaffold_preview=preview)
        shown = self.service.evaluate(capability="youtube_history_search", scaffold_preview=preview, scaffold_preview_shown=True)

        self.assertEqual(PackLifecycleState.MISSING, offered.state)
        self.assertEqual("scaffold_preview", offered.next_step.action)
        self.assertEqual(PackLifecycleState.SCAFFOLD_PREVIEWED, shown.state)
        self.assertEqual("create_review_candidate", shown.next_step.action)

    def test_generated_quarantined_candidate_needs_inspection(self) -> None:
        result = self.service.evaluate(
            generated_candidate={"pack_id": "generated-youtube", "name": "YouTube History Search"},
        )

        self.assertEqual(PackLifecycleState.GENERATED_QUARANTINED, result.state)
        self.assertFalse(result.usable)
        self.assertEqual("inspection", result.missing_gate)

    def test_imported_for_review_is_not_usable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.service.evaluate(imported_pack=_pack_row(normalized_path=tmpdir))

        self.assertEqual(PackLifecycleState.IMPORTED_FOR_REVIEW, result.state)
        self.assertFalse(result.usable)
        self.assertEqual("approval", result.missing_gate)
        self.assertIn("review only", result.user_message_summary)

    def test_approved_but_disabled_asks_to_enable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.service.evaluate(imported_pack=_pack_row(approved=True, enabled=False, normalized_path=tmpdir))

        self.assertEqual(PackLifecycleState.APPROVED, result.state)
        self.assertFalse(result.usable)
        self.assertEqual("enablement", result.missing_gate)
        self.assertEqual("enable", result.next_step.action)

    def test_enabled_with_missing_configuration_asks_for_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.service.evaluate(
                imported_pack=_pack_row(approved=True, enabled=True, normalized_path=tmpdir),
                required_configuration=["takeout_profile"],
            )

        self.assertEqual(PackLifecycleState.NEEDS_CONFIGURATION, result.state)
        self.assertEqual(("takeout_profile",), result.required_configuration)
        self.assertEqual("configure", result.next_step.action)

    def test_enabled_with_missing_adapter_grant_asks_for_permission(self) -> None:
        adapter = {"kind": "local_file_import", "purpose": "Import a selected file."}
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.service.evaluate(
                imported_pack=_pack_row(approved=True, enabled=True, adapters=[adapter], normalized_path=tmpdir),
                permission_grants=[],
            )

        self.assertEqual(PackLifecycleState.NEEDS_PERMISSION, result.state)
        self.assertEqual(("local_file_import",), result.required_permissions)
        self.assertEqual("request_permission", result.next_step.action)

    def test_usable_only_after_all_gates_pass(self) -> None:
        adapter = {"kind": "local_file_import", "purpose": "Import a selected file."}
        grant = {"pack_id": "pack.youtube-history-search", "adapter_kind": "local_file_import", "state": "granted"}
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.service.evaluate(
                imported_pack=_pack_row(
                    approved=True,
                    enabled=True,
                    adapters=[adapter],
                    configuration={"takeout_profile": "default"},
                    normalized_path=tmpdir,
                ),
                permission_grants=[grant],
                required_configuration=["takeout_profile"],
            )

        self.assertEqual(PackLifecycleState.USABLE, result.state)
        self.assertTrue(result.usable)
        self.assertEqual("use", result.next_step.action)

    def test_blocked_candidate_cannot_be_used(self) -> None:
        result = self.service.evaluate(imported_pack=_pack_row(status="blocked", approved=True, enabled=True))

        self.assertEqual(PackLifecycleState.BLOCKED, result.state)
        self.assertTrue(result.blocked)
        self.assertFalse(result.usable)

    def test_disabled_and_removed_are_not_usable(self) -> None:
        disabled = self.service.evaluate(imported_pack={**_pack_row(approved=True, enabled=True), "disabled": True})
        removed = self.service.evaluate(removed_pack={"pack_id": "pack.old", "name": "Old Pack"})

        self.assertEqual(PackLifecycleState.DISABLED, disabled.state)
        self.assertFalse(disabled.usable)
        self.assertEqual(PackLifecycleState.REMOVED, removed.state)
        self.assertFalse(removed.usable)

    def test_render_lifecycle_response_names_next_safe_step(self) -> None:
        result = self.service.evaluate(capability="voice_output")

        rendered = render_lifecycle_response(result)

        self.assertIn("No installed, approved, enabled pack is usable", rendered)
        self.assertIn("Next safe step:", rendered)


if __name__ == "__main__":
    unittest.main()
