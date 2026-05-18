from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from agent.packs.lifecycle import PackLifecycleService
from agent.packs.managed_adapter_invocation import (
    MANAGED_ADAPTER_OPERATION_REGISTRY,
    OP_DESCRIBE_CAPABILITY,
    OP_DRY_RUN,
    OP_VALIDATE_GRANT,
    ManagedAdapterInvocationRequest,
    ManagedAdapterInvoker,
)
from agent.packs.managed_adapters import (
    ADAPTER_LOCAL_FILE_IMPORT,
    ManagedAdapterSpec,
    build_permission_request,
    create_metadata_only_grant,
)


def _adapter(max_mb: int = 50, extensions: tuple[str, ...] = (".json", ".html")) -> ManagedAdapterSpec:
    return ManagedAdapterSpec(
        kind=ADAPTER_LOCAL_FILE_IMPORT,
        purpose="Import one selected local file.",
        allowed_extensions=extensions,
        max_file_size_mb=max_mb,
        path_policy="user_selected_file_only",
        stores_local_index=False,
        network_allowed=False,
    )


def _pack(adapter: ManagedAdapterSpec, *, approved: bool = True, enabled: bool = True) -> dict[str, object]:
    return {
        "pack_id": "pack.generic.local-import",
        "name": "Generic Local Import",
        "status": "normalized",
        "approved": approved,
        "enabled": enabled,
        "canonical_pack": {
            "display_name": "Generic Local Import",
            "pack_identity": {"canonical_id": "pack.generic.local-import", "content_hash": "hash-1"},
            "managed_adapters": [adapter.to_dict()],
        },
    }


def _request(operation: str = OP_VALIDATE_GRANT, adapter_kind: str = ADAPTER_LOCAL_FILE_IMPORT) -> ManagedAdapterInvocationRequest:
    return ManagedAdapterInvocationRequest(
        pack_id="pack.generic.local-import",
        canonical_id="pack.generic.local-import",
        pack_name="Generic Local Import",
        adapter_kind=adapter_kind,
        operation=operation,
        user_id="user1",
        thread_id="thread1",
        dry_run=True,
    )


def _grant(path: str, adapter: ManagedAdapterSpec) -> dict[str, object]:
    metadata = {
        "path_redacted": "<redacted-local-history-path>/" + Path(path).name,
        "extension": Path(path).suffix.lower(),
        "exists": True,
        "is_file": True,
        "size_bytes": Path(path).stat().st_size,
        "max_file_size_mb": adapter.max_file_size_mb,
    }
    request = build_permission_request(
        pack_id="pack.generic.local-import",
        pack_name="Generic Local Import",
        adapter=adapter,
        requested_path=path,
    )
    return create_metadata_only_grant(request=request, path_metadata=metadata).to_dict()


class TestManagedAdapterInvocation(unittest.TestCase):
    def setUp(self) -> None:
        self.invoker = ManagedAdapterInvoker()
        self.lifecycle_service = PackLifecycleService()

    def test_unknown_adapter_rejected(self) -> None:
        result = self.invoker.invoke(_request(adapter_kind="not_real"), lifecycle={"usable": True})

        self.assertFalse(result.ok)
        self.assertEqual("adapter_kind_unknown", result.errors[0].code)

    def test_generic_operation_registry_exposes_no_content_operations(self) -> None:
        operations = MANAGED_ADAPTER_OPERATION_REGISTRY[ADAPTER_LOCAL_FILE_IMPORT]

        self.assertEqual({OP_VALIDATE_GRANT, OP_DESCRIBE_CAPABILITY, OP_DRY_RUN}, set(operations))
        for operation in operations.values():
            self.assertFalse(operation.reads_content)
            self.assertFalse(operation.writes_content)
            self.assertFalse(operation.uses_network)
            self.assertFalse(operation.executes_code)

    def test_unsupported_operation_rejected(self) -> None:
        adapter = _adapter()
        result = self.invoker.invoke(
            _request(operation="parse_contents"),
            lifecycle={"usable": True},
            adapter_declarations=[adapter.to_dict()],
            permission_grants=[],
        )

        self.assertFalse(result.ok)
        self.assertEqual("operation_unsupported", result.errors[0].code)

    def test_invocation_refused_if_lifecycle_not_usable(self) -> None:
        adapter = _adapter()
        lifecycle = self.lifecycle_service.evaluate(imported_pack=_pack(adapter, approved=False, enabled=False))

        result = self.invoker.invoke(
            _request(),
            lifecycle=lifecycle,
            adapter_declarations=[adapter.to_dict()],
            permission_grants=[],
        )

        self.assertFalse(result.ok)
        self.assertEqual("lifecycle_not_usable", result.errors[0].code)
        self.assertIn("approval", str(result.to_dict()))

    def test_local_file_import_validate_grant_succeeds_with_matching_metadata(self) -> None:
        adapter = _adapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "watch-history.json"
            path.write_text('{"private": true}\n', encoding="utf-8")
            grant = _grant(str(path), adapter)
            lifecycle = self.lifecycle_service.evaluate(
                imported_pack=_pack(adapter),
                permission_grants=[grant],
            )
            result = self.invoker.invoke(
                _request(OP_VALIDATE_GRANT),
                lifecycle=lifecycle,
                adapter_declarations=[adapter.to_dict()],
                permission_grants=[grant],
            )

        self.assertTrue(result.ok)
        self.assertFalse(result.did_work)
        self.assertIn("No file contents were read", result.summary)

    def test_validate_grant_fails_without_matching_grant(self) -> None:
        adapter = _adapter()
        result = self.invoker.invoke(
            _request(OP_VALIDATE_GRANT),
            lifecycle={"usable": True},
            adapter_declarations=[adapter.to_dict()],
            permission_grants=[],
        )

        self.assertFalse(result.ok)
        self.assertEqual("permission_grant_missing", result.errors[0].code)

    def test_validate_grant_fails_if_extension_or_size_policy_mismatches(self) -> None:
        adapter = _adapter(max_mb=1, extensions=(".html",))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "watch-history.json"
            path.write_text("{}\n", encoding="utf-8")
            grant = _grant(str(path), _adapter())
            result = self.invoker.invoke(
                _request(OP_VALIDATE_GRANT),
                lifecycle={"usable": True},
                adapter_declarations=[adapter.to_dict()],
                permission_grants=[grant],
            )

        self.assertFalse(result.ok)
        self.assertEqual("permission_grant_invalid", result.errors[0].code)
        self.assertIn("extension_not_allowed", str(result.to_dict()))

    def test_describe_capability_redacts_raw_path(self) -> None:
        adapter = _adapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "watch-history.json"
            path.write_text("{}\n", encoding="utf-8")
            grant = _grant(str(path), adapter)
            result = self.invoker.invoke(
                _request(OP_DESCRIBE_CAPABILITY),
                lifecycle={"usable": True},
                adapter_declarations=[adapter.to_dict()],
                permission_grants=[grant],
            )

        payload = result.to_dict()
        self.assertTrue(result.ok)
        self.assertIn("<redacted-local-history-path>/watch-history.json", str(payload))
        self.assertNotIn(str(path), str(payload))

    def test_dry_run_checks_file_without_reading_contents(self) -> None:
        adapter = _adapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "watch-history.json"
            path.write_text('{"private": "history contents"}\n', encoding="utf-8")
            grant = _grant(str(path), adapter)
            with mock.patch("pathlib.Path.read_text", side_effect=AssertionError("must not read file")):
                result = self.invoker.invoke(
                    _request(OP_DRY_RUN),
                    lifecycle={"usable": True},
                    adapter_declarations=[adapter.to_dict()],
                    permission_grants=[grant],
                )

        self.assertTrue(result.ok)
        self.assertTrue(result.did_work)
        self.assertFalse(result.data["read_contents"])
        self.assertNotIn("history contents", str(result.to_dict()))

    def test_no_network_subprocess_or_code_execution_path_exists(self) -> None:
        adapter = _adapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "watch-history.json"
            path.write_text("{}\n", encoding="utf-8")
            grant = _grant(str(path), adapter)
            with mock.patch("subprocess.run", side_effect=AssertionError("no subprocess")):
                result = self.invoker.invoke(
                    _request(OP_DRY_RUN),
                    lifecycle={"usable": True},
                    adapter_declarations=[adapter.to_dict()],
                    permission_grants=[grant],
                )

        self.assertTrue(result.ok)
        self.assertIn("No network", " ".join(result.privacy_notes))

    def test_non_youtube_fixture_pack_uses_same_invocation_path(self) -> None:
        adapter = _adapter(extensions=(".csv",))
        pack = {
            "pack_id": "pack.recipes.csv-import",
            "name": "Recipe CSV Import",
            "status": "normalized",
            "approved": True,
            "enabled": True,
            "canonical_pack": {
                "display_name": "Recipe CSV Import",
                "pack_identity": {"canonical_id": "pack.recipes.csv-import", "content_hash": "hash-recipes"},
                "managed_adapters": [adapter.to_dict()],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "recipes.csv"
            path.write_text("title\nSoup\n", encoding="utf-8")
            metadata = {
                "path_redacted": "<redacted-local-history-path>/recipes.csv",
                "extension": ".csv",
                "exists": True,
                "is_file": True,
                "size_bytes": path.stat().st_size,
                "max_file_size_mb": adapter.max_file_size_mb,
            }
            permission_request = build_permission_request(
                pack_id="pack.recipes.csv-import",
                pack_name="Recipe CSV Import",
                adapter=adapter,
                requested_path=str(path),
            )
            grant = create_metadata_only_grant(request=permission_request, path_metadata=metadata).to_dict()
            lifecycle = self.lifecycle_service.evaluate(imported_pack=pack, permission_grants=[grant])
            request = ManagedAdapterInvocationRequest(
                pack_id="pack.recipes.csv-import",
                canonical_id="pack.recipes.csv-import",
                pack_name="Recipe CSV Import",
                adapter_kind=ADAPTER_LOCAL_FILE_IMPORT,
                operation=OP_DRY_RUN,
                dry_run=True,
            )
            result = self.invoker.invoke(
                request,
                lifecycle=lifecycle,
                adapter_declarations=[adapter.to_dict()],
                permission_grants=[grant],
            )

        self.assertTrue(result.ok)
        self.assertEqual(OP_DRY_RUN, result.operation)
        self.assertTrue(result.did_work)
        self.assertNotIn("Soup", str(result.to_dict()))


if __name__ == "__main__":
    unittest.main()
