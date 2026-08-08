from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from agent.api_server import APIServerHandler, AgentRuntime
from agent.capability_registry import ApprovalPolicy, CapabilityMode
from scripts import native_capability_proof
from test_api_server import _config


class _Handler(APIServerHandler):
    def __init__(self, runtime: AgentRuntime, path: str, payload: dict[str, object] | None = None) -> None:
        self.runtime = runtime
        self.path = path
        self.headers = {"Content-Length": "0"}
        self._payload = payload or {}
        self.status = 0
        self.body: dict[str, object] = {}

    def _read_json(self) -> dict[str, object]:  # type: ignore[override]
        return dict(self._payload)

    def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
        self.status = status
        self.body = json.loads(json.dumps(payload))


def _chat(runtime: AgentRuntime, text: str, *, user: str = "wp2", thread: str = "wp2:t") -> dict[str, object]:
    handler = _Handler(runtime, "/chat", {
        "messages": [{"role": "user", "content": text}],
        "user_id": user,
        "thread_id": thread,
        "session_id": thread,
        "source_surface": "webui",
    })
    handler.do_POST()
    assert handler.status in {200, 400}, handler.body
    return handler.body


def _capability_id(body: dict[str, object]) -> str | None:
    def visit(value):  # type: ignore[no-untyped-def]
        if isinstance(value, dict):
            if "selected_capability_id" in value:
                return str(value.get("selected_capability_id") or "") or None
            for nested in value.values():
                found = visit(nested)
                if found:
                    return found
        elif isinstance(value, list):
            for nested in value:
                found = visit(nested)
                if found:
                    return found
        return None

    return visit(body)


def _with_registry_mutation(mutator):  # type: ignore[no-untyped-def]
    original = native_capability_proof._build_registry

    def changed():  # type: ignore[no-untyped-def]
        registry, cleanup = original()
        mutator(registry)
        return registry, cleanup

    return patch.object(native_capability_proof, "_build_registry", changed)


def test_protected_inventory_reconciles_registry_and_public_surfaces() -> None:
    report = native_capability_proof.run_proof()
    assert report["ok"] is True, report["failures"]
    assert report["totals"]["expected"] == 22
    assert report["totals"]["registered"] == 22
    assert report["totals"]["api_surfaces"] >= 120
    assert report["surface_totals"]["native_skills"] == 15
    assert report["redaction"]["passed"] is True


def test_gate_sensitivity_missing_registration_and_unproved_extra_fail() -> None:
    with _with_registry_mutation(lambda registry: registry._items.pop("filesystem.read")):  # noqa: SLF001
        report = native_capability_proof.run_proof()
    assert report["ok"] is False
    assert any(row["reason"] == "protected_capability_missing" for row in report["failures"])

    def add_extra(registry):  # type: ignore[no-untyped-def]
        source = registry.require("assistant.presence")
        registry._items["invented.unproved"] = replace(source, capability_id="invented.unproved")  # noqa: SLF001

    with _with_registry_mutation(add_extra):
        report = native_capability_proof.run_proof()
    assert report["ok"] is False
    assert any(row["reason"] == "unmanifested_registry_entry" for row in report["failures"])


def test_gate_sensitivity_health_selftest_and_policy_breakage_fail() -> None:
    def break_hooks(registry):  # type: ignore[no-untyped-def]
        source = registry.require("system.status")
        registry._items["system.status"] = replace(source, self_test_hook=lambda: {"ok": False, "reason": "fixture_broken"})  # noqa: SLF001

    with _with_registry_mutation(break_hooks):
        report = native_capability_proof.run_proof()
    assert any(row["reason"] == "fixture_broken" for row in report["failures"])

    def break_health(registry):  # type: ignore[no-untyped-def]
        source = registry.require("assistant.presence")
        registry._items["assistant.presence"] = replace(source, health_hook=lambda: (_ for _ in ()).throw(RuntimeError("broken")))  # noqa: SLF001

    with _with_registry_mutation(break_health):
        report = native_capability_proof.run_proof()
    assert any(str(row["reason"]).startswith("health_check_failed:") for row in report["failures"])

    def break_policy(registry):  # type: ignore[no-untyped-def]
        source = registry.require("models.switch")
        registry._items["models.switch"] = replace(source, mode=CapabilityMode.READ_ONLY, approval_policy=ApprovalPolicy.NEVER)  # noqa: SLF001

    with _with_registry_mutation(break_policy):
        report = native_capability_proof.run_proof()
    assert any("mode_mismatch" in row["reason"] for row in report["failures"])


def test_gate_sensitivity_new_unmapped_api_and_stale_identity_fail_closed() -> None:
    original_api_literals = native_capability_proof._api_literals
    with patch.object(native_capability_proof, "_api_literals", lambda: [*original_api_literals(), "/new-public-mutation"]):
        report = native_capability_proof.run_proof()
    assert report["ok"] is False
    assert any(row["item"] == "/new-public-mutation" for row in report["failures"])

    original_ui_ids = native_capability_proof._ui_section_ids
    with patch.object(native_capability_proof, "_ui_section_ids", lambda: {*original_ui_ids(), "new_public_control"}):
        report = native_capability_proof.run_proof()
    assert report["ok"] is False
    assert any(row["item"] == "ui:new_public_control" for row in report["failures"])

    report = native_capability_proof.run_proof()
    expected_hash = hashlib.sha256(native_capability_proof.INVENTORY_PATH.read_bytes()).hexdigest()
    assert report["inventory_sha256"] == expected_hash
    assert len(report["candidate"]["candidate_fingerprint"]) == 64


def test_capability_status_api_has_plain_and_advanced_views() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db"), perception_roots=(raw,)))
        normal = _Handler(runtime, "/capabilities")
        normal.do_GET()
        assert normal.status == 200
        assert normal.body["source"] == "live_capability_registry"
        assert normal.body["counts"]["total"] == 22
        assert "id" not in normal.body["capabilities"][0]
        advanced = _Handler(runtime, "/capabilities?advanced=1")
        advanced.do_GET()
        assert advanced.status == 200
        assert advanced.body["advanced"] is True
        assert advanced.body["capabilities"][0]["id"]
        serialized = json.dumps(advanced.body).lower()
        assert "api_key" not in serialized
        assert "token" not in serialized
        search = next(row for row in advanced.body["capabilities"] if row["id"] == "search.web")
        assert search["available"] is False
        assert search["status"] == "unavailable"
        assert search["reason"] == "search_disabled"
        assert "searxng" in str(search["next_step"]).lower()


def test_new_native_capabilities_are_reachable_through_production_chat() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        allowed = root / "allowed"
        allowed.mkdir()
        pack_dir = allowed / "sample-pack"
        pack_dir.mkdir()
        (pack_dir / "SKILL.md").write_text("# Sample\n\nText-only fixture.\n", encoding="utf-8")
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db"), perception_roots=(raw,)))
        cases = [
            ("could you inspect the Python version installed here", "system.shell.inspect"),
            (f"please create a new folder at {allowed / 'new-folder'}", "filesystem.create_directory"),
            ("preview installing the ripgrep package with apt", "system.package.install"),
            ("what is the health of saved continuity memory", "memory.status"),
            ("please forget my saved long term memory", "memory.manage"),
            (f"preview importing the local text pack at {pack_dir}", "packs.manage"),
            ("is the Telegram transport connected and healthy", "telegram.status"),
            ("please turn off the Telegram adapter", "telegram.manage"),
            ("show the list of assistant backups", "operator.status"),
            ("prepare a preview for backing up assistant state", "operator.lifecycle"),
            ("search the web for current local AI news", "search.web"),
        ]
        for index, (text, expected) in enumerate(cases):
            body = _chat(runtime, text, user=f"wp2-{index}", thread=f"wp2:{index}")
            assert _capability_id(body) == expected, (text, expected, _capability_id(body), body)
            assistant = body.get("assistant") if isinstance(body.get("assistant"), dict) else {}
            assert str(assistant.get("content") or body.get("response") or body.get("message") or "").strip()


def test_read_only_capabilities_never_request_mutation_approval() -> None:
    report = native_capability_proof.run_proof()
    for row in report["capabilities"]:
        if row["mode"] == "read_only":
            assert row["approval"] == "never", row


def test_natural_capability_health_questions_use_live_runtime_and_policy() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db"), perception_roots=(raw,)))
        file_status = _chat(runtime, "is file search working?", user="wp2-status-file", thread="wp2:status-file")
        assert _capability_id(file_status) == "filesystem.search"
        file_text = str(file_status.get("message") or file_status.get("response") or "").lower()
        assert "read-only file searches" in file_text
        assert "what filename or text" in file_text

        model_status = _chat(runtime, "can you install a model?", user="wp2-status-model", thread="wp2:status-model")
        assert _capability_id(model_status) == "models.inventory"
        model_text = str(model_status.get("message") or model_status.get("response") or "").lower()
        assert "bounded model controller" in model_text
        assert "explicit preview and confirmation" in model_text
        assert "silently" in model_text

        memory_status = _chat(runtime, "is saved continuity memory healthy?", user="wp2-status-memory", thread="wp2:status-memory")
        assert _capability_id(memory_status) == "memory.status"

        directory_preview = _chat(
            runtime,
            f"preview creating a folder called inflected-preview in {raw}",
            user="wp2-inflected-directory",
            thread="wp2:inflected-directory",
        )
        assert _capability_id(directory_preview) == "filesystem.create_directory"


def test_container_path_and_hardware_memory_language_route_by_semantic_domain() -> None:
    """Live-found unseen forms stay governed by concepts, not sentence aliases."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db"), perception_roots=(raw,)))

        listing = _chat(
            runtime,
            f"show what is in {raw}",
            user="wp2-container-language",
            thread="wp2:container-language",
        )
        assert _capability_id(listing) == "filesystem.list"

        resources = _chat(
            runtime,
            "how are this computer's CPU and memory doing?",
            user="wp2-resource-language",
            thread="wp2:resource-language",
        )
        assert _capability_id(resources) == "system.status"


def test_systematic_transformations_cover_every_wp2_capability_family() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        allowed = root / "allowed"
        allowed.mkdir()
        pack_dir = allowed / "transform-pack"
        pack_dir.mkdir()
        (pack_dir / "SKILL.md").write_text("# Transform fixture\n", encoding="utf-8")
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db"), perception_roots=(raw,)))
        bases = {
            "system.shell.inspect": ("inspect the python version", "tell me which Python interpreter release is present"),
            "filesystem.create_directory": (f"create a folder at {allowed / 'made'}", f"add an empty directory beneath {allowed / 'made'}"),
            "system.package.install": ("preview install package ripgrep with apt", "prepare adding the ripgrep utility through Debian packages"),
            "memory.status": ("check continuity memory health", "explain whether remembered context is available"),
            "memory.manage": ("forget saved memory", "erase the retained long term context after confirmation"),
            "packs.manage": (f"import text pack at {pack_dir}", "prepare approval to enable the reviewed guidance bundle"),
            "telegram.status": ("check telegram transport health", "is the optional bot connection operational"),
            "telegram.manage": ("disable telegram adapter", "turn off the optional messaging transport"),
            "operator.status": ("show backup list status", "enumerate Personal Agent recovery archives"),
            "operator.lifecycle": ("preview create assistant backup", "prepare a protected copy of assistant state"),
            "search.web": ("search web for current AI news", "look online for recent artificial intelligence reports"),
        }
        misspellings = {
            "python": "pyhton", "folder": "fodler", "package": "pakcage", "memory": "memroy",
            "pack": "pakc", "telegram": "telgeram", "backup": "bakcup", "web": "wbe",
        }
        total = 0
        for capability_id, (base, unseen) in bases.items():
            misspelled = base
            for source, target in misspellings.items():
                if source in misspelled:
                    misspelled = misspelled.replace(source, target, 1)
                    break
            words = base.split()
            reordered = " ".join([*words[1:], words[0]]) if len(words) > 1 else base
            variants = {
                "casing": base.upper(),
                "punctuation": f"{base}?!?!!",
                "spacing": "   ".join(words),
                "common_misspelling": misspelled,
                "adjacent_transposition": misspelled,
                "sms_shorthand": f"pls {base}",
                "dropped_function_words": " ".join(word for word in words if word not in {"a", "the", "at", "with", "for"}),
                "reordered_wording": reordered,
                "polite_filler": f"could you please {base} for me",
                "unseen_paraphrase": unseen,
            }
            for category, text in variants.items():
                total += 1
                response = _chat(runtime, text, user=f"wp2-transform-{total}", thread=f"wp2-transform:{total}")
                assert _capability_id(response) == capability_id, (capability_id, category, text, _capability_id(response), response)
        assert total == 110
