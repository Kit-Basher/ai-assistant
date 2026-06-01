from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from agent.packs.managed_adapters import (
    ADAPTER_LOCAL_FILE_IMPORT,
    ManagedAdapterSpec,
    build_permission_request,
    create_metadata_only_grant,
    list_adapter_grants,
    record_adapter_grant,
    render_permission_preview,
    validate_local_file_path_metadata,
    validate_managed_adapter_declarations,
    validate_managed_adapter_spec,
)
from agent.packs.scaffolding import build_scaffold_preview


def _valid_spec() -> dict[str, object]:
    return {
        "kind": ADAPTER_LOCAL_FILE_IMPORT,
        "purpose": "Import a user-selected Google Takeout YouTube watch-history file.",
        "allowed_extensions": [".json", ".html"],
        "max_file_size_mb": 50,
        "path_policy": "user_selected_file_only",
        "stores_local_index": True,
        "network_allowed": False,
    }


class TestManagedAdapters(unittest.TestCase):
    def test_scaffold_declares_local_file_import_adapter(self) -> None:
        preview = build_scaffold_preview("youtube_history_search")
        assert preview is not None
        manifest = preview["proposed_manifest"]
        adapters = manifest["managed_adapters"]
        ok, errors, normalized = validate_managed_adapter_declarations(adapters)

        self.assertTrue(ok)
        self.assertEqual([], errors)
        self.assertEqual("local_file_import", normalized[0]["kind"])
        self.assertEqual([".json", ".html"], normalized[0]["allowed_extensions"])
        self.assertEqual("user_selected_file_only", normalized[0]["path_policy"])
        self.assertFalse(normalized[0]["network_allowed"])

    def test_rejects_unknown_disabled_network_wildcard_executable_and_path_policy(self) -> None:
        cases = [
            ({"kind": "not_real"}, "adapter_kind_unknown"),
            ({"kind": "network_fetch"}, "adapter_kind_disabled"),
            ({"network_allowed": True}, "local_file_import_network_not_allowed"),
            ({"allowed_extensions": ["*"]}, "wildcard_extensions_not_allowed"),
            ({"allowed_extensions": [".py"]}, "executable_extensions_not_allowed"),
            ({"path_policy": "scan_directory"}, "path_policy_must_be_user_selected_file_only"),
        ]
        for override, expected in cases:
            spec = {**_valid_spec(), **override}
            ok, errors, _adapter = validate_managed_adapter_spec(spec)
            self.assertFalse(ok)
            self.assertIn(expected, errors)

    def test_permission_preview_is_honest_and_does_not_read_path(self) -> None:
        spec = ManagedAdapterSpec.from_mapping(_valid_spec())
        request = build_permission_request(
            pack_id="pack-youtube",
            pack_name="YouTube History Search",
            adapter=spec,
            requested_path="/home/c/Takeout/YouTube/history/watch-history.json",
        )
        with mock.patch("pathlib.Path.read_text", side_effect=AssertionError("should not read file")):
            rendered = render_permission_preview(request)

        self.assertIn("one user-selected local file only", rendered)
        self.assertIn(".json, .html", rendered)
        self.assertIn("<redacted-local-history-path>/watch-history.json", rendered)
        self.assertIn("raw file contents are not logged", rendered)
        self.assertIn("browser profile scraping", rendered)
        self.assertIn("network fetches", rendered)
        self.assertIn("I will not read or parse the file", rendered)

    def test_grant_records_metadata_only_after_path_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "watch-history.json"
            history_path.write_text('{"private": "history contents"}\n', encoding="utf-8")
            spec = ManagedAdapterSpec.from_mapping(_valid_spec())
            ok, errors, metadata = validate_local_file_path_metadata(str(history_path), spec)
            self.assertTrue(ok)
            self.assertEqual([], errors)
            self.assertEqual(".json", metadata["extension"])

            request = build_permission_request(
                pack_id="pack-youtube",
                pack_name="YouTube History Search",
                adapter=spec,
                requested_path=str(history_path),
            )
            grant = create_metadata_only_grant(request=request, path_metadata=metadata)
            record = record_adapter_grant(tmpdir, grant)
            rows = list_adapter_grants(tmpdir)

            self.assertEqual(1, len(rows))
            self.assertEqual(record["grant_id"], rows[0]["grant_id"])
            self.assertEqual("granted", record["state"])
            self.assertEqual([], record["permissions_granted"])
            self.assertFalse(record["executes_code"])
            self.assertNotIn("history contents", str(record))
            self.assertIn("<redacted-local-history-path>/watch-history.json", record["granted_path_redacted"])
            journal = record.get("managed_action_journal") if isinstance(record.get("managed_action_journal"), dict) else {}
            self.assertEqual("managed_adapter_permission_grant", journal.get("action_type"))
            self.assertTrue(journal.get("verification_result", {}).get("ok"))
            self.assertFalse(journal.get("rollback_result", {}).get("attempted"))


if __name__ == "__main__":
    unittest.main()
