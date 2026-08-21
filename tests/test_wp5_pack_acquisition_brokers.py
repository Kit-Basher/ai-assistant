from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import struct
import zlib

import pytest

import agent.api_server as api_server_module
from agent.packs.brokers import BrokerError, CoreBrokerRuntime, PackPrivateStore, ScopedHttpsBroker, SelectedLocalDataBroker, file_metadata
from agent.packs.capability_contracts import PackCapabilityContractError, load_manifest
from agent.packs.draft_builder import PackDraftBuilder
from agent.packs.secure_transport import SecureFetchResult, _address_allowed
from agent.packs.wp5_contracts import ACQUISITION_SCHEMA, BROKER_SCHEMA, AcquisitionSourceV1, BrokerDeclarationV1, WP5ContractError
from agent.mutation_plan import build_mutation_confirmation
from test_pack_search_authorization import _runtime
from test_unified_conversation_routing import _chat, _understanding
from test_wp4_pack_runtime import _make_usable, _write_pack
from test_remote_pack_fetch import _FakeOpener, _FakeResponse, _zip_bytes
from agent.packs.remote_fetch import RemotePackFetcher, RemotePackSource
from agent.task_loop import TaskState, build_deterministic_plan


def _local_broker() -> dict[str, object]:
    return {"schema_version": BROKER_SCHEMA, "kind": "selected_local_data", "mode": "read_only", "scopes": ["one_exact_user_selected_file", "derived_index_search"], "data_flow": "local_private", "limits": {"input_bytes": 8 * 1024 * 1024, "output_bytes": 65536, "items": 10000, "wall_ms": 2000, "requests": 1}, "config": {"extensions": [".json", ".csv", ".html", ".htm", ".txt"], "raw_content_retained": False}}


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "100.64.0.1", "::1", "fc00::1", "fe80::1", "::ffff:127.0.0.1", "0.0.0.0", "224.0.0.1"])
def test_ssrf_address_classes_are_denied(address: str) -> None:
    assert _address_allowed(address) is False


def test_acquisition_contract_rejects_credentials_queries_private_and_unknown_authority() -> None:
    base = {"schema_version": ACQUISITION_SCHEMA, "source_kind": "generic_archive_url", "requested_url": "https://example.com/pack.zip", "requested_ref": None, "resolved_commit": None, "catalog_source_id": None}
    assert AcquisitionSourceV1.parse(base).requested_url == "https://example.com/pack.zip"
    for url in ("http://example.com/a.zip", "https://user:pass@example.com/a.zip", "https://127.0.0.1/a.zip", "https://example.com/a.zip?token=x"):
        with pytest.raises(WP5ContractError):
            AcquisitionSourceV1.parse({**base, "requested_url": url})
    with pytest.raises(WP5ContractError, match="unknown_fields"):
        AcquisitionSourceV1.parse({**base, "approved": True})


def test_broker_contract_rejects_dynamic_network_authority_and_private_exfiltration_shape() -> None:
    with pytest.raises(WP5ContractError, match="authority_field"):
        BrokerDeclarationV1.parse({**_local_broker(), "config": {"extensions": [".json"], "raw_content_retained": False, "path": "/home"}})
    network = {"schema_version": BROKER_SCHEMA, "kind": "scoped_https", "mode": "read_only", "scopes": ["public_weather"], "data_flow": "public_network", "limits": {"input_bytes": 1024, "output_bytes": 4096, "items": 10, "wall_ms": 2000, "requests": 1}, "config": {"origins": ["https://example.com"], "path_templates": ["/weather"], "parameter_names": ["city"], "methods": ["GET"]}}
    assert BrokerDeclarationV1.parse(network).config["methods"] == ["GET"]
    with pytest.raises(WP5ContractError):
        BrokerDeclarationV1.parse({**network, "config": {**network["config"], "origins": ["https://127.0.0.1"]}})


def test_exact_local_file_index_search_persistence_revocation_and_no_raw_retention() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        source = root / "export.json"
        source.write_text(json.dumps([{"title": "Blue Album", "year": 1994}, {"title": "Green Book", "year": 2018}]), encoding="utf-8")
        metadata = file_metadata(source, allowed_roots=(root,))
        grant = {"state": "granted", "pack_id": "local-search", "granted_path": str(source), "path_metadata": metadata}
        store = PackPrivateStore(str(root / "agent.db"))
        broker = SelectedLocalDataBroker(store=store, allowed_roots=(root,))
        indexed = broker.import_index(pack_id="local-search", version="1", actor_id="alice", path=str(source), grant=grant)
        assert indexed["indexed_records"] == 2 and indexed["raw_content_retained"] is False
        match = broker.search(pack_id="local-search", version="1", actor_id="alice", query="blue")
        assert match["match_count"] == 1 and match["matches"][0]["fields"]["title"] == "Blue Album"
        assert source.read_text(encoding="utf-8").encode("utf-8") not in root.joinpath("agent.db").read_bytes()
        restarted = SelectedLocalDataBroker(store=PackPrivateStore(str(root / "agent.db")), allowed_roots=(root,))
        assert restarted.search(pack_id="local-search", version="1", actor_id="alice", query="green")["match_count"] == 1
        assert store.remove_pack(pack_id="local-search", version="1", actor_id="alice") == 2
        assert restarted.search(pack_id="local-search", version="1", actor_id="alice", query="green")["match_count"] == 0


def test_local_file_swap_and_cross_pack_private_store_access_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw); source = root / "export.csv"
        source.write_text("name\nfirst\n", encoding="utf-8")
        metadata = file_metadata(source, allowed_roots=(root,))
        grant = {"state": "granted", "pack_id": "one", "granted_path": str(source), "path_metadata": metadata}
        source.write_text("name\nchanged\n", encoding="utf-8")
        broker = SelectedLocalDataBroker(store=PackPrivateStore(str(root / "agent.db")), allowed_roots=(root,))
        with pytest.raises(BrokerError, match="changed_after_grant"):
            broker.import_index(pack_id="one", version="1", actor_id="alice", path=str(source), grant=grant)
        assert broker.search(pack_id="two", version="1", actor_id="alice", query="first")["match_count"] == 0


def test_assistant_created_local_search_draft_is_quarantine_only_and_valid_pack() -> None:
    with tempfile.TemporaryDirectory() as raw:
        builder = PackDraftBuilder(raw)
        preview = builder.preview({"template": "local_data_search", "name": "Watch History Search"})
        assert preview["requires_confirmation"] and not preview["created"]
        created = builder.create_quarantine(preview)
        assert created["quarantine_only"] and not created["approved"] and not created["enabled"]
        manifest = load_manifest(created["path"])
        assert manifest["capabilities"][0]["capability_id"] == "pack.watch-history-search.search"
        assert manifest["capabilities"][0]["invocation"]["kind"] == "core_broker"
        assert manifest["capabilities"][0]["permissions"] == ["broker:selected_local_data"]


def test_assistant_created_native_declarative_draft_uses_supported_result_contract() -> None:
    with tempfile.TemporaryDirectory() as raw:
        builder = PackDraftBuilder(raw)
        preview = builder.preview({"template": "declarative_native", "name": "System Report", "capability_id": "system.status"})
        created = builder.create_quarantine(preview)
        manifest = load_manifest(created["path"])
        capability = manifest["capabilities"][0]
        assert capability["invocation"] == {
            "kind": "registered_capability",
            "capability_id": "system.status",
            "inputs": {},
            "result_field": "text",
        }
        assert created["quarantine_only"] is True


def test_generated_pack_cannot_request_raw_code_or_unknown_broker() -> None:
    with tempfile.TemporaryDirectory() as raw:
        builder = PackDraftBuilder(raw)
        with pytest.raises(Exception):
            builder.preview({"template": "python", "name": "unsafe"})
        with pytest.raises(WP5ContractError):
            BrokerDeclarationV1.parse({**_local_broker(), "kind": "shell"})


def test_pack_manifest_broker_permission_is_exact_and_cannot_downgrade() -> None:
    with tempfile.TemporaryDirectory() as raw:
        builder = PackDraftBuilder(raw)
        created = builder.create_quarantine(builder.preview({"template": "local_data_search", "name": "Bound Search"}))
        path = Path(created["path"]) / "personal-agent-pack.json"
        manifest = json.loads(path.read_text())
        manifest["capabilities"][0]["permissions"] = []
        path.chmod(0o600); path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(PackCapabilityContractError, match="broker_declaration_or_permission_missing"):
            load_manifest(created["path"])


def test_core_runtime_denies_private_to_network_composition_before_transport() -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = CoreBrokerRuntime(db_path=str(Path(raw) / "agent.db"), storage_root=raw, allowed_roots=(raw,))
        declaration = {"schema_version": BROKER_SCHEMA, "kind": "scoped_https", "mode": "read_only", "scopes": ["public"], "data_flow": "public_network", "limits": {"input_bytes": 1024, "output_bytes": 4096, "items": 10, "wall_ms": 1000, "requests": 1}, "config": {"origins": ["https://example.com"], "path_templates": ["/data"], "parameter_names": [], "methods": ["GET"]}}
        with pytest.raises(BrokerError, match="private_to_network"):
            runtime.invoke(pack_id="p", version="1", actor_id="a", declaration=declaration, operation="get", inputs={"path": "/data", "taint_sources": ["selected_local_data"]})


def test_scoped_https_broker_returns_bounded_tainted_public_data_from_exact_scope() -> None:
    class Transport:
        def fetch_bytes(self, url, **kwargs):  # type: ignore[no-untyped-def]
            assert url == "https://api.example.com/weather?city=Regina"
            assert kwargs["method"] == "GET" and kwargs["max_bytes"] == 4096
            body = b'{"temperature_c":21}'
            return body, SecureFetchResult("https://api.example.com/weather", "https://api.example.com/weather", 200, "application/json", len(body), __import__("hashlib").sha256(body).hexdigest(), 0, 12)

    declaration = BrokerDeclarationV1.parse({"schema_version": BROKER_SCHEMA, "kind": "scoped_https", "mode": "read_only", "scopes": ["public_weather"], "data_flow": "public_network", "limits": {"input_bytes": 1024, "output_bytes": 4096, "items": 10, "wall_ms": 2000, "requests": 1}, "config": {"origins": ["https://api.example.com"], "path_templates": ["/weather"], "parameter_names": ["city"], "methods": ["GET"]}})
    result = ScopedHttpsBroker(transport=Transport()).request(declaration, method="GET", path="/weather", parameters={"city": "Regina"})
    assert result["ok"] and result["tainted"]
    assert result["body"] == '{"temperature_c":21}' and result["verification"]["within_scope"]
    for bad in ({"path": "/admin", "method": "GET"}, {"path": "/weather", "method": "POST"}):
        with pytest.raises(BrokerError):
            ScopedHttpsBroker(transport=Transport()).request(declaration, parameters={}, **bad)


def test_unpinned_github_ref_is_resolved_to_immutable_commit_before_archive_review(tmp_path: Path) -> None:
    commit = "a" * 40
    archive = _zip_bytes({"repo-main/SKILL.md": b"# Reviewed guidance\n"})

    class Transport:
        def fetch_bytes(self, url, **_kwargs):  # type: ignore[no-untyped-def]
            assert url == "https://api.github.com/repos/example/repo/commits/main"
            body = json.dumps({"sha": commit}).encode()
            return body, SecureFetchResult(url, url, 200, "application/json", len(body), __import__("hashlib").sha256(body).hexdigest(), 0, 5)

        def fetch_to_temp(self, url, *, parent, **_kwargs):  # type: ignore[no-untyped-def]
            assert url == f"https://github.com/example/repo/archive/{commit}.zip"
            path = Path(parent) / "resolved.partial"
            path.write_bytes(archive)
            return path, SecureFetchResult(url, url, 200, "application/zip", len(archive), __import__("hashlib").sha256(archive).hexdigest(), 0, 8)

    result = RemotePackFetcher(str(tmp_path / "external"), secure_transport=Transport()).fetch(
        RemotePackSource(kind="github_repo", url="https://github.com/example/repo", ref="main")
    )
    assert result.source.ref == "main"
    assert result.source.commit_hash_resolved == commit
    assert result.source.resolved_url == f"https://github.com/example/repo/archive/{commit}.zip"
    assert any("immutable commit" in note for note in result.source.provenance_notes)


def test_reviewed_scoped_https_pack_is_selected_and_invoked_through_production_chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Transport:
        def fetch_bytes(self, url, **kwargs):  # type: ignore[no-untyped-def]
            assert url == "https://api.example.com/weather"
            body = b'{"temperature_c":21,"city":"Regina"}'
            return body, SecureFetchResult(url, url, 200, "application/json", len(body), __import__("hashlib").sha256(body).hexdigest(), 0, 9)

    monkeypatch.setenv("AGENT_EXTERNAL_PACKS_DIR", str(tmp_path / "external"))
    runtime = _runtime(tmp_path, perception_roots=(str(tmp_path),))
    pack = tmp_path / "weather-pack"
    pack.mkdir()
    broker = {
        "schema_version": BROKER_SCHEMA,
        "kind": "scoped_https",
        "mode": "read_only",
        "scopes": ["public_weather"],
        "data_flow": "public_network",
        "limits": {"input_bytes": 1024, "output_bytes": 4096, "items": 10, "wall_ms": 2000, "requests": 1},
        "config": {"origins": ["https://api.example.com"], "path_templates": ["/weather"], "parameter_names": [], "methods": ["GET"]},
    }
    manifest = {
        "schema_version": "personal-agent.pack.v1", "id": "public-weather", "version": "1.0.0",
        "pack_class": "declarative", "display_name": "Public Weather", "description": "Read public current weather from one reviewed endpoint",
        "brokers": [broker],
        "capabilities": [{
            "schema_version": "personal-agent.pack-capability.v1", "name": "current", "display_name": "Current Public Weather",
            "description": "Read the current public weather from the reviewed public weather endpoint",
            "examples": ["check current public weather", "show weather from the public feed"],
            "input_schema": {"type": "object", "properties": {}, "required": []},
            "output_schema": {"type": "object", "properties": {"result": {"type": "string", "maxLength": 4096}}, "required": ["result"]},
            "mode": "read_only", "task_composable": True, "permissions": ["broker:scoped_https"],
            "invocation": {"kind": "core_broker", "broker_kind": "scoped_https", "operation": "get", "inputs": {"path": "/weather"}, "result_field": "body"},
            "verifier": {"kind": "nonempty", "field": "result"}, "limits": {"wall_ms": 2000, "output_bytes": 4096}, "self_test_input": {},
        }],
    }
    (pack / "personal-agent-pack.json").write_text(json.dumps(manifest), encoding="utf-8")
    rid = _apply_capability_gate(runtime, "import", {"source_dir": str(pack)})["record"]["record_id"]
    _apply_capability_gate(runtime, "gate", {"record_id": rid, "gate": "review_approved", "value": True})
    _apply_capability_gate(runtime, "gate", {"record_id": rid, "gate": "grants", "value": ["broker:scoped_https"]})
    runtime.orchestrator()._pack_broker_runtime.https = ScopedHttpsBroker(transport=Transport())  # noqa: SLF001
    _apply_capability_gate(runtime, "gate", {"record_id": rid, "gate": "enabled", "value": True})
    response = _chat(runtime, "could you check the current public weather feed", user="alice", thread="alice:weather")
    assert _understanding(response)["selected_capability_id"] == "pack.public-weather.current", response
    assert "temperature_c" in str(response["setup"].get("result"))


def _apply_capability_gate(runtime, action: str, payload: dict[str, object]) -> dict[str, object]:
    orch = runtime.orchestrator()
    plan = orch.pack_capability_mutation_preview(action, payload, actor_id="alice", session_id="s", thread_id="t")
    return orch.pack_capability_mutation_apply(plan["plan_id"], plan["binding_digest"], actor_id="alice", session_id="s", thread_id="t")


def _apply_central(runtime, operation: str, request: dict[str, object]) -> dict[str, object]:
    ok, preview = runtime.route_pack_search_mutation(operation, request)
    assert ok and preview["requires_confirmation"]
    plan = preview["plan"]
    ok, result = runtime.route_pack_search_mutation(operation, {**request, "mutation_plan": plan, "confirmation": build_mutation_confirmation(plan, confirmation_id=f"wp5-{plan['plan_id']}")})
    assert ok, result
    return result


def _apply_capability_api(runtime, action: str, payload: dict[str, object]) -> dict[str, object]:
    scoped = {**payload, "actor_id": "alice", "session_id": "s", "thread_id": "t"}
    ok, preview = runtime.pack_capability_mutation_preview(action, scoped)
    assert ok, preview
    plan = preview["plan"]
    ok, result = runtime.pack_capability_mutation_apply({"plan_id": plan["plan_id"], "binding_digest": plan["binding_digest"], "confirmed": True, "actor_id": "alice", "session_id": "s", "thread_id": "t"})
    assert ok, result
    return result["result"]


def test_useful_local_data_pack_registers_searches_through_chat_and_revocation_removes_authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_EXTERNAL_PACKS_DIR", str(tmp_path / "external"))
    runtime = _runtime(tmp_path, perception_roots=(str(tmp_path),))
    created = _apply_central(runtime, "external_pack.draft", {"template": "local_data_search", "name": "Library Export Search"})
    assert created["quarantine_only"] and created["imported_for_review"]
    assert created["approved"] is False and created["enabled"] is False and created["usable"] is False
    imported = created["record"]
    rid = imported["record_id"]
    _apply_capability_gate(runtime, "gate", {"record_id": rid, "gate": "review_approved", "value": True})
    _apply_capability_gate(runtime, "gate", {"record_id": rid, "gate": "grants", "value": ["broker:selected_local_data"]})
    _apply_capability_gate(runtime, "gate", {"record_id": rid, "gate": "enabled", "value": True})
    source = tmp_path / "library.json"
    source.write_text(json.dumps([{"title": "Dune", "author": "Frank Herbert"}, {"title": "Hyperion", "author": "Dan Simmons"}]), encoding="utf-8")
    adapter = {"kind": "local_file_import", "purpose": "build a private bounded search index", "allowed_extensions": [".json"], "max_file_size_mb": 8, "path_policy": "user_selected_file_only", "stores_local_index": True, "network_allowed": False}
    _apply_central(runtime, "external_pack.grant", {"pack_id": "library-export-search", "adapter": adapter, "requested_path": str(source)})
    _apply_central(runtime, "external_pack.index", {"pack_id": "library-export-search", "record_id": rid})
    response = _chat(runtime, "Please find Dune in my library export", user="alice", thread="alice:t")
    assert _understanding(response).get("selected_capability_id") == "pack.library-export-search.search", response
    assert response["setup"].get("result"), response
    assert json.loads(response["setup"]["result"])["match_count"] == 1
    revoke_preview = _chat(runtime, "revoke this skill's file access", user="alice", thread="alice:t")
    assert _understanding(revoke_preview).get("selected_capability_id") == "packs.manage", revoke_preview
    assert revoke_preview["setup"].get("type") == "action_confirmation_required"
    assert revoke_preview["setup"].get("action_type") == "external_pack.revoke"
    revoked = _chat(runtime, "yes", user="alice", thread="alice:t")
    assert revoked["setup"].get("mutated") is True, revoked
    assert runtime.orchestrator()._capability_registry.get("pack.library-export-search.search") is None


def test_update_activation_is_atomic_and_exact_rollback_restores_reviewed_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_EXTERNAL_PACKS_DIR", str(tmp_path / "external"))
    runtime = _runtime(tmp_path, perception_roots=(str(tmp_path),))
    old_rid, capability_id = _make_usable(runtime, _write_pack(tmp_path / "v1"))
    v2 = _write_pack(tmp_path / "v2", version="2.0.0", wat='(module (func (export "invoke") (param i32) (result i32) local.get 0 i32.const 3 i32.mul))')
    imported = _apply_capability_gate(runtime, "import", {"source_dir": str(v2)})["record"]
    new_rid = imported["record_id"]
    assert runtime.orchestrator().pack_capability_status(old_rid)["active"]
    _apply_capability_gate(runtime, "gate", {"record_id": new_rid, "gate": "review_approved", "value": True})
    _apply_capability_gate(runtime, "gate", {"record_id": new_rid, "gate": "grants", "value": ["pure_compute"]})
    staged = _apply_capability_gate(runtime, "gate", {"record_id": new_rid, "gate": "enabled", "value": True})["record"]
    assert staged["lifecycle"]["missing_gate"] == "activation"
    activated = _apply_capability_api(runtime, "activate", {"record_id": new_rid})["record"]
    assert activated["active"] and not runtime.orchestrator().pack_capability_status(old_rid)["active"]
    assert runtime.orchestrator()._capability_registry.invoke(capability_id, {"value": 3, "user_id": "a", "text": "triple 3"}).data["runtime_payload"]["result"] == 9
    comparison = runtime.pack_capability_compare(old_rid, new_rid)
    assert comparison[0] and comparison[1]["result"]["authority_changed"]
    rolled = _apply_capability_api(runtime, "rollback", {"record_id": old_rid})["record"]
    assert rolled["active"] and not runtime.orchestrator().pack_capability_status(new_rid)["active"]


def test_pack_lifecycle_controls_are_reachable_through_natural_chat_and_exact_confirmation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_EXTERNAL_PACKS_DIR", str(tmp_path / "external"))
    runtime = _runtime(tmp_path, perception_roots=(str(tmp_path),))
    rid, capability_id = _make_usable(runtime, _write_pack(tmp_path / "chat-pack", pack_id="chat-doubler"))

    preview = _chat(runtime, "could u disable the chat doubler skill please", user="alice", thread="alice:pack-admin")
    assert _understanding(preview).get("selected_capability_id") == "packs.manage", preview
    assert preview["setup"]["type"] == "pack_capability_mutation_preview"
    assert runtime.orchestrator()._capability_registry.get(capability_id) is not None

    unrelated = _chat(runtime, "what model are you using now", user="alice", thread="alice:pack-admin")
    assert _understanding(unrelated).get("selected_capability_id") == "models.inventory", unrelated
    assert runtime.orchestrator()._capability_registry.get(capability_id) is not None

    applied = _chat(runtime, "yes", user="alice", thread="alice:pack-admin")
    assert applied["setup"]["type"] == "pack_capability_mutation_applied", applied
    assert runtime.orchestrator().pack_capability_status(rid)["enabled"] is False
    assert runtime.orchestrator()._capability_registry.get(capability_id) is None


def test_pack_update_question_with_zero_external_packs_is_grounded_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_EXTERNAL_PACKS_DIR", str(tmp_path / "external"))
    runtime = _runtime(tmp_path, perception_roots=(str(tmp_path),))
    response = _chat(runtime, "is there an update for an installed skill pack?", user="alice", thread="alice:none")
    assert _understanding(response).get("selected_capability_id") == "packs.manage", response
    assert response["setup"].get("count") == 0
    assert response["setup"].get("mutated") is False
    assert "no external capability-pack versions" in str(response.get("message") or "").lower()


def test_pack_update_diff_and_rollback_are_truthful_chat_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_EXTERNAL_PACKS_DIR", str(tmp_path / "external"))
    runtime = _runtime(tmp_path, perception_roots=(str(tmp_path),))
    old_rid, capability_id = _make_usable(runtime, _write_pack(tmp_path / "chat-v1", pack_id="chat-update"))
    newer = _write_pack(tmp_path / "chat-v2", pack_id="chat-update", version="2.0.0", wat='(module (func (export "invoke") (param i32) (result i32) local.get 0 i32.const 3 i32.mul))')
    new_rid = _apply_capability_gate(runtime, "import", {"source_dir": str(newer)})["record"]["record_id"]
    _apply_capability_gate(runtime, "gate", {"record_id": new_rid, "gate": "review_approved", "value": True})
    _apply_capability_gate(runtime, "gate", {"record_id": new_rid, "gate": "grants", "value": ["pure_compute"]})
    _apply_capability_gate(runtime, "gate", {"record_id": new_rid, "gate": "enabled", "value": True})
    _apply_capability_api(runtime, "activate", {"record_id": new_rid})

    comparison = _chat(runtime, "what changed in the chat update skill?", user="alice", thread="alice:update")
    assert _understanding(comparison).get("selected_capability_id") == "packs.manage", comparison
    assert comparison["setup"].get("comparison", {}).get("authority_changed") is True

    short_followup = _chat(runtime, "is there an update?", user="alice", thread="alice:update")
    assert _understanding(short_followup).get("selected_capability_id") == "packs.manage", short_followup
    assert short_followup["setup"].get("comparison", {}).get("authority_changed") is True

    preview = _chat(runtime, "roll back that skill to the prior reviewed version", user="alice", thread="alice:update")
    assert preview["setup"]["type"] == "pack_capability_mutation_preview", preview
    assert runtime.orchestrator().pack_capability_status(new_rid)["active"] is True
    applied = _chat(runtime, "yes", user="alice", thread="alice:update")
    assert applied["setup"]["type"] == "pack_capability_mutation_applied", applied
    assert runtime.orchestrator().pack_capability_status(old_rid)["active"] is True
    assert runtime.orchestrator()._capability_registry.get(capability_id) is not None


def _png_header(width: int, height: int) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    raster = (
        b"\x00"
        if width * height > 1_000_000
        else b"".join(b"\x00" + (b"\x00\x00\x00\xff" * width) for _ in range(height))
    )
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raster)) + chunk(b"IEND", b"")


def test_visualizer_pack_is_core_rendered_reduced_motion_safe_and_disable_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_EXTERNAL_PACKS_DIR", str(tmp_path / "external"))
    runtime = _runtime(tmp_path, perception_roots=(str(tmp_path),))
    sprite = tmp_path / "sprite.png"; sprite.write_bytes(_png_header(2, 1))
    created = _apply_central(runtime, "external_pack.draft", {"template": "presence_visualizer", "name": "Calm Sprite", "asset_path": str(sprite), "frame_width": 1, "frame_height": 1, "animations": {"idle": {"frames": [0], "frame_duration_ms": 300, "loop": True}, "thinking": {"frames": [0, 1], "frame_duration_ms": 120, "loop": True}}})
    imported = created["record"]
    rid = imported["record_id"]
    _apply_capability_gate(runtime, "gate", {"record_id": rid, "gate": "review_approved", "value": True})
    enabled = _apply_capability_gate(runtime, "gate", {"record_id": rid, "gate": "enabled", "value": True})["record"]
    assert enabled["active"]
    status = runtime.pack_visualizer_status()
    assert status["available"] and status["reduced_motion_supported"] and not status["scripts_allowed"] and not status["remote_requests"]
    assert runtime.pack_visualizer_asset("calm-sprite", "1.0.0")[0] == sprite.read_bytes()
    _apply_capability_gate(runtime, "gate", {"record_id": rid, "gate": "enabled", "value": False})
    assert runtime.pack_visualizer_status()["fallback"] == "core_default" and not runtime.pack_visualizer_status()["available"]


def test_visualizer_rejects_svg_corrupt_and_pixel_bomb(tmp_path: Path) -> None:
    builder = PackDraftBuilder(tmp_path / "packs", allowed_asset_roots=(tmp_path,))
    svg = tmp_path / "bad.svg"; svg.write_text("<svg><script>alert(1)</script></svg>")
    corrupt = tmp_path / "bad.png"; corrupt.write_bytes(b"not a png")
    bomb = tmp_path / "bomb.png"; bomb.write_bytes(_png_header(100000, 100000))
    for path in (svg, corrupt, bomb):
        with pytest.raises(Exception):
            builder.preview({"template": "presence_visualizer", "name": "bad", "asset_path": str(path)})


def test_remote_capability_archive_uses_central_confirmation_and_stops_at_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_EXTERNAL_PACKS_DIR", str(tmp_path / "external"))
    runtime = _runtime(tmp_path, perception_roots=(str(tmp_path),))
    url = "https://example.com/health-pack.zip"
    manifest = {"schema_version": "personal-agent.pack.v1", "id": "remote-health", "version": "1.0.0", "pack_class": "declarative", "display_name": "Remote Health", "description": "Reviewed health report wrapper", "brokers": [], "capabilities": [{"schema_version": "personal-agent.pack-capability.v1", "name": "report", "display_name": "Remote Health Report", "description": "Report machine status through the native status capability", "examples": ["use remote health report", "show the reviewed health wrapper"], "input_schema": {"type": "object", "properties": {}, "required": []}, "output_schema": {"type": "object", "properties": {"result": {"type": "string", "maxLength": 65536}}, "required": ["result"]}, "mode": "read_only", "task_composable": True, "permissions": [], "invocation": {"kind": "registered_capability", "capability_id": "system.status", "inputs": {}, "result_field": "text"}, "verifier": {"kind": "nonempty", "field": "result"}, "limits": {"wall_ms": 1000, "output_bytes": 8192}, "self_test_input": {}}]}
    archive = _zip_bytes({"remote-health/personal-agent-pack.json": json.dumps(manifest).encode()})
    fetcher = RemotePackFetcher(str(tmp_path / "external"), opener=_FakeOpener({url: _FakeResponse(archive, url=url, content_length=len(archive))}))
    monkeypatch.setattr(api_server_module, "RemotePackFetcher", lambda *_args, **_kwargs: fetcher)
    result = _apply_central(runtime, "external_pack.fetch", {"source": {"url": url, "kind": "generic_archive_url"}})
    assert result["fetched_to_quarantine"] and result["imported_for_review"]
    assert result["record"]["record_id"] and result["record"]["pack_id"] == "remote-health"
    row = runtime.orchestrator().pack_capability_status()
    assert len(row["packs"]) == 1
    recorded = row["packs"][0]
    assert recorded["pack_id"] == "remote-health" and not recorded["review_approved"] and not recorded["enabled"] and not recorded["lifecycle"]["usable"]
    assert recorded["provenance"]["archive_sha256"] == __import__("hashlib").sha256(archive).hexdigest()
    assert recorded["provenance"]["source_kind"] == "generic_archive_url"
    assert runtime.orchestrator()._capability_registry.get("pack.remote-health.report") is None
    rid = recorded["record_id"]
    _apply_capability_gate(runtime, "gate", {"record_id": rid, "gate": "review_approved", "value": True})
    _apply_capability_gate(runtime, "gate", {"record_id": rid, "gate": "grants", "value": ["system:read"]})
    _apply_capability_gate(runtime, "gate", {"record_id": rid, "gate": "enabled", "value": True})
    response = _chat(runtime, "could you use the reviewed health wrapper", user="alice", thread="alice:remote")
    assert _understanding(response)["selected_capability_id"] == "pack.remote-health.report"
    assert response["setup"].get("result")
    coordinator = runtime.orchestrator()._task_coordinator  # noqa: SLF001 - production coordinator/registry boundary
    proposal = build_deterministic_plan(
        goal="inspect native status and the reviewed pack report",
        capability_requests=[
            ("system.status", {"user_id": "alice", "text": "inspect native status"}),
            ("pack.remote-health.report", {"user_id": "alice", "text": "use the reviewed pack report"}),
        ],
        actor_id="alice",
    )
    task = coordinator.create(proposal, actor_id="alice", session_id="s", thread_id="alice:task")
    finished = coordinator.run(task["task_id"], actor_id="alice", thread_id="alice:task")
    assert finished["state"] == TaskState.SUCCEEDED.value
    assert [step["capability_id"] for step in finished["steps"]] == ["system.status", "pack.remote-health.report"]
    assert all(step["verifier_status"] == "pass" for step in finished["steps"])
    assert finished["outcome"]["verified"] is True
    _apply_capability_gate(runtime, "gate", {"record_id": rid, "gate": "enabled", "value": False})
    assert runtime.orchestrator()._capability_registry.get("pack.remote-health.report") is None
    removed = _apply_capability_api(runtime, "remove", {"record_id": rid, "private_data": "delete"})
    assert removed["removed"] and removed["private_data"] == "delete"
    assert runtime.orchestrator().pack_capability_status(rid) is None
