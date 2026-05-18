from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from agent.packs.lifecycle import PackLifecycleService, PackLifecycleState
from agent.packs.lifecycle_actions import PackLifecycleActionController
from agent.packs.managed_adapters import (
    ADAPTER_LOCAL_FILE_IMPORT,
    ManagedAdapterSpec,
    build_permission_request,
    create_metadata_only_grant,
)


def _row(
    *,
    approved: bool = False,
    enabled: bool | None = None,
    adapters: list[dict[str, object]] | None = None,
    normalized_path: str | None = None,
    status: str = "normalized",
) -> dict[str, object]:
    return {
        "pack_id": "pack.generic.local-import",
        "name": "Generic Local Import",
        "status": status,
        "approved": approved,
        "enabled": enabled,
        "normalized_path": normalized_path,
        "canonical_pack": {
            "display_name": "Generic Local Import",
            "pack_identity": {"canonical_id": "pack.generic.local-import", "content_hash": "hash-1"},
            "source": {"origin": "generated_scaffold"},
            "managed_adapters": list(adapters or []),
        },
    }


class TestPackLifecycleActionController(unittest.TestCase):
    def setUp(self) -> None:
        self.service = PackLifecycleService()

    def test_wrong_action_state_is_refused(self) -> None:
        lifecycle = self.service.evaluate(catalog_pack={"remote_id": "voice", "name": "Voice Pack"})
        controller = PackLifecycleActionController(handlers={"import_for_review": lambda ctx: {"ok": True}})

        result = controller.dispatch(lifecycle, action="import_for_review")

        self.assertFalse(result.ok)
        self.assertTrue(result.refused)
        self.assertIn("cannot run import_for_review", result.text)

    def test_discovered_to_preview(self) -> None:
        lifecycle = self.service.evaluate(catalog_pack={"remote_id": "voice", "name": "Voice Pack"})
        controller = PackLifecycleActionController(
            handlers={
                "preview": lambda ctx: {
                    "ok": True,
                    "text": "Preview shown.",
                    "preview": {"listing": {"remote_id": "voice", "name": "Voice Pack"}},
                }
            }
        )

        result = controller.dispatch(lifecycle)

        self.assertTrue(result.ok)
        self.assertEqual("preview", result.action)
        self.assertEqual(PackLifecycleState.PREVIEWED, result.lifecycle_after["state"])

    def test_previewed_to_import_for_review(self) -> None:
        lifecycle = self.service.evaluate(catalog_preview={"listing": {"remote_id": "voice", "name": "Voice Pack"}})
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = PackLifecycleActionController(
                handlers={
                    "import_for_review": lambda ctx: {
                        "ok": True,
                        "pack": _row(normalized_path=tmpdir),
                    }
                }
            )
            result = controller.dispatch(lifecycle)

        self.assertTrue(result.ok)
        self.assertEqual(PackLifecycleState.IMPORTED_FOR_REVIEW, result.lifecycle_after["state"])
        self.assertFalse(result.lifecycle_after["usable"])

    def test_scaffold_previewed_to_create_review_candidate(self) -> None:
        preview = {"scaffold_id": "generic-local-import", "capability": "generic_local_import", "title": "Generic Local Import"}
        lifecycle = self.service.evaluate(scaffold_preview=preview, scaffold_preview_shown=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = PackLifecycleActionController(
                handlers={
                    "create_review_candidate": lambda ctx: {
                        "ok": True,
                        "pack": _row(normalized_path=tmpdir),
                    }
                }
            )
            result = controller.dispatch(lifecycle)

        self.assertTrue(result.ok)
        self.assertEqual(PackLifecycleState.IMPORTED_FOR_REVIEW, result.lifecycle_after["state"])
        self.assertFalse(result.lifecycle_after["usable"])

    def test_imported_for_review_does_not_become_usable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lifecycle = self.service.evaluate(imported_pack=_row(normalized_path=tmpdir))
            controller = PackLifecycleActionController(
                handlers={
                    "review_approve": lambda ctx: {
                        "ok": True,
                        "pack": _row(approved=True, enabled=False, normalized_path=tmpdir),
                    }
                }
            )
            result = controller.dispatch(lifecycle, action="review_approve")

        self.assertTrue(result.ok)
        self.assertEqual(PackLifecycleState.APPROVED, result.lifecycle_after["state"])
        self.assertFalse(result.lifecycle_after["usable"])

    def test_approved_to_enable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lifecycle = self.service.evaluate(imported_pack=_row(approved=True, enabled=False, normalized_path=tmpdir))
            controller = PackLifecycleActionController(
                handlers={
                    "enable": lambda ctx: {
                        "ok": True,
                        "pack": _row(approved=True, enabled=True, normalized_path=tmpdir),
                    }
                }
            )
            result = controller.dispatch(lifecycle)

        self.assertTrue(result.ok)
        self.assertEqual(PackLifecycleState.USABLE, result.lifecycle_after["state"])

    def test_enabled_missing_permission_asks_permission(self) -> None:
        adapter = {"kind": ADAPTER_LOCAL_FILE_IMPORT, "purpose": "Import a selected file."}
        with tempfile.TemporaryDirectory() as tmpdir:
            lifecycle = self.service.evaluate(imported_pack=_row(approved=True, enabled=True, adapters=[adapter], normalized_path=tmpdir))
            controller = PackLifecycleActionController(
                handlers={"request_permission": lambda ctx: {"ok": True, "text": "Give me a local path first."}}
            )
            result = controller.dispatch(lifecycle)

        self.assertTrue(result.ok)
        self.assertEqual("request_permission", result.action)
        self.assertIsNone(result.lifecycle_after)
        self.assertIn("local path", result.text)

    def test_permission_preview_yes_records_grant_only(self) -> None:
        adapter = ManagedAdapterSpec(
            kind=ADAPTER_LOCAL_FILE_IMPORT,
            purpose="Import a selected file.",
            allowed_extensions=(".json",),
            max_file_size_mb=50,
            path_policy="user_selected_file_only",
            network_allowed=False,
        )
        request = build_permission_request(
            pack_id="pack.generic.local-import",
            pack_name="Generic Local Import",
            adapter=adapter,
            requested_path="/tmp/watch-history.json",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = _row(approved=True, enabled=True, adapters=[adapter.to_dict()], normalized_path=tmpdir)
            lifecycle = self.service.evaluate(imported_pack=pack, permission_grants=[])
            grant = create_metadata_only_grant(request=request, path_metadata={"exists": True})
            controller = PackLifecycleActionController(
                handlers={
                    "record_permission_grant": lambda ctx: {
                        "ok": True,
                        "text": "Recorded grant metadata only.",
                        "pack": pack,
                        "permission_grants": [grant.to_dict()],
                    }
                }
            )
            result = controller.dispatch(lifecycle, action="record_permission_grant")

        self.assertTrue(result.ok)
        self.assertEqual(PackLifecycleState.USABLE, result.lifecycle_after["state"])
        self.assertEqual([], result.payload["permission_grants"][0]["permissions_granted"])
        self.assertFalse(result.payload["permission_grants"][0]["executes_code"])

    def test_blocked_candidate_cannot_continue(self) -> None:
        lifecycle = self.service.evaluate(imported_pack=_row(status="blocked", approved=True, enabled=True))
        controller = PackLifecycleActionController(handlers={"use_if_usable": lambda ctx: {"ok": True}})

        result = controller.dispatch(lifecycle, action="use_if_usable")

        self.assertFalse(result.ok)
        self.assertTrue(result.refused)
        self.assertIn("blocked", result.text)

    def test_generic_non_youtube_fixture_uses_same_flow(self) -> None:
        lifecycle = self.service.evaluate(
            capability="generic_local_import",
            catalog_pack={"remote_id": "generic-local-import", "name": "Generic Local Import"},
        )
        controller = PackLifecycleActionController(
            handlers={
                "preview": lambda ctx: {
                    "ok": True,
                    "preview": {"listing": {"remote_id": "generic-local-import", "name": "Generic Local Import"}},
                }
            }
        )

        result = controller.dispatch(lifecycle)

        self.assertTrue(result.ok)
        self.assertEqual("generic_local_import", result.lifecycle_after["capability"])
        self.assertEqual(PackLifecycleState.PREVIEWED, result.lifecycle_after["state"])


if __name__ == "__main__":
    unittest.main()
