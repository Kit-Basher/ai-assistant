from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import load_config
from agent.packs.capability_recommendation import recommend_packs_for_capability, render_pack_capability_response
from agent.packs.state_truth import build_pack_state_snapshot


def _config(registry_path: str, db_path: str, **overrides: object):
    cfg = load_config(require_telegram_token=False)
    base = replace(
        cfg,
        db_path=db_path,
        log_path=os.path.join(os.path.dirname(db_path), "agent.log"),
        llm_registry_path=registry_path,
        llm_automation_enabled=False,
        telegram_enabled=False,
    )
    return replace(base, **overrides)


class _HandlerForTest(APIServerHandler):
    def __init__(self, runtime_obj: AgentRuntime, path: str) -> None:
        self.runtime = runtime_obj
        self.path = path
        self.headers = {}
        self.status_code = 0
        self.content_type = ""
        self.body = b""

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        self.status_code = status
        self.content_type = "application/json"
        self.body = json.dumps(payload, ensure_ascii=True).encode("utf-8")

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        *,
        content_type: str,
        cache_control: str | None = None,
    ) -> None:
        _ = cache_control
        self.status_code = status
        self.content_type = content_type
        self.body = body


class _FakePackStore:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def list_external_packs(self) -> list[dict[str, object]]:
        return list(self._rows)


class _FakeDiscovery:
    def __init__(self, sources: list[dict[str, object]], pack_map: dict[tuple[str, str], list[dict[str, object]]]) -> None:
        self._sources = sources
        self._pack_map = pack_map

    def list_sources(self) -> list[dict[str, object]]:
        return list(self._sources)

    def list_packs(self, source_id: str) -> dict[str, object]:
        source = next((row for row in self._sources if str(row.get("id") or "") == source_id), {})
        packs = self._pack_map.get((source_id, "*"), self._pack_map.get((source_id, ""), []))
        return {
            "source": dict(source) if isinstance(source, dict) else {},
            "packs": list(packs),
            "from_cache": False,
            "stale": False,
        }

    def search(self, source_id: str, query: str) -> dict[str, object]:
        source = next((row for row in self._sources if str(row.get("id") or "") == source_id), {})
        packs = self._pack_map.get((source_id, query), self._pack_map.get((source_id, "*"), self._pack_map.get((source_id, ""), [])))
        return {
            "source": dict(source) if isinstance(source, dict) else {},
            "search": {"results": list(packs)},
            "from_cache": False,
            "stale": False,
        }


class _StubTruthService:
    def __init__(self, *, provider: str, model: str) -> None:
        self._provider = provider
        self._model = model

    def current_chat_target_status(self) -> dict[str, object]:
        return {
            "effective_provider": self._provider,
            "effective_model": self._model,
            "effective_provider_health_status": "ok",
            "effective_model_health_status": "ok",
            "provider_health_status": "ok",
            "health_status": "ok",
        }

    def ui_state(self, *, ready_payload: dict[str, object] | None = None) -> dict[str, object]:
        ready = dict(ready_payload or {})
        runtime_status = ready.get("runtime_status") if isinstance(ready.get("runtime_status"), dict) else {}
        summary = str(ready.get("message") or runtime_status.get("summary") or "").strip()
        next_action = str(runtime_status.get("next_action") or "").strip() or None
        runtime_mode = str(runtime_status.get("runtime_mode") or ready.get("runtime_mode") or "READY").strip().lower() or "ready"
        return {
            "ok": True,
            "updated_at": "2026-04-09T00:00:00+00:00",
            "runtime": {
                "status": runtime_mode,
                "summary": summary,
                "next_action": next_action,
            },
            "model": {
                "provider": self._provider,
                "model": self._model,
                "path": f"{self._provider} / {self._model}",
                "routing_mode": "auto",
                "health": "up",
            },
            "conversation": {
                "topic": None,
                "recent_request": None,
                "open_loop": None,
            },
            "action": {
                "pending_approval": False,
                "blocked_reason": None,
                "last_action": None,
            },
            "signals": {
                "response_style": "concise",
                "confidence_visible": False,
            },
            "source": "stub",
        }


class TestStateTruthUnification(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def _runtime(self) -> AgentRuntime:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.get_defaults = lambda: {  # type: ignore[method-assign]
            "default_provider": "ollama",
            "resolved_default_model": "ollama:qwen2.5:3b-instruct",
            "routing_mode": "auto",
            "allow_remote_fallback": True,
        }
        runtime.llm_control_mode_status = lambda: {  # type: ignore[method-assign]
            "mode": "safe",
            "mode_label": "SAFE MODE",
            "mode_source": "config_default",
            "allow_remote_switch": True,
            "allow_install_pull": True,
            "forbidden_actions": [],
            "approval_required_actions": [],
        }
        return runtime

    def test_ready_and_state_adapters_share_the_same_runtime_truth(self) -> None:
        runtime = self._runtime()
        ready_payload = {
            "ok": True,
            "ready": True,
            "runtime_mode": "READY",
            "runtime_status": {
                "runtime_mode": "READY",
                "summary": "Ready. Using ollama / ollama:qwen2.5:3b-instruct.",
                "next_action": "No action needed.",
                "failure_code": None,
            },
            "message": "Ready. Using ollama / ollama:qwen2.5:3b-instruct.",
            "safe_mode_target": {"enabled": False, "configured_valid": True},
            "blocked": {"blocked": False, "kind": None, "reason": None, "message": None},
        }
        runtime.ready_status = lambda: dict(ready_payload)  # type: ignore[method-assign]
        runtime.runtime_truth_service = lambda: _StubTruthService(  # type: ignore[method-assign]
            provider="ollama",
            model="ollama:qwen2.5:3b-instruct",
        )

        ready_handler = _HandlerForTest(runtime, "/ready")
        ready_handler.do_GET()
        state_handler = _HandlerForTest(runtime, "/state")
        state_handler.do_GET()

        ready_payload_json = json.loads(ready_handler.body.decode("utf-8"))
        state_payload_json = json.loads(state_handler.body.decode("utf-8"))
        self.assertEqual(200, ready_handler.status_code)
        self.assertEqual(200, state_handler.status_code)
        self.assertEqual(ready_payload_json["runtime_status"]["summary"], state_payload_json["runtime"]["summary"])
        self.assertEqual(ready_payload_json["runtime_status"]["next_action"], state_payload_json["runtime"]["next_action"])
        self.assertEqual(ready_payload_json["runtime_status"]["runtime_mode"], "READY")
        self.assertEqual(state_payload_json["model"]["path"], "ollama / ollama:qwen2.5:3b-instruct")
        self.assertNotIn("Agent is", ready_payload_json["runtime_status"]["summary"])
        self.assertNotIn("OpenAI", state_payload_json["runtime"]["summary"])

    def test_pack_state_snapshot_and_endpoint_agree_on_disabled_pack_truth(self) -> None:
        installed_row = {
            "pack_id": "pack.voice.local_fast",
            "name": "Local Voice",
            "status": "normalized",
            "enabled": False,
            "normalized_path": str(Path(__file__).resolve()),
            "canonical_pack": {
                "display_name": "Local Voice",
                "source": {"name": "Local", "kind": "local_catalog"},
                "pack_identity": {"canonical_id": "pack.voice.local_fast"},
                "capabilities": {
                    "summary": "Local speech output for this machine.",
                    "declared": ["voice_output"],
                },
            },
            "review_envelope": {"pack_name": "Local Voice"},
        }
        discovery = _FakeDiscovery(
            sources=[
                {"id": "local", "name": "Local Catalog", "kind": "local_catalog", "enabled": True, "allowed_by_policy": True},
            ],
            pack_map={
                ("local", "*"): [
                    {
                        "remote_id": "voice-pack",
                        "name": "Local Voice Pack",
                        "summary": "Local speech output for this machine.",
                        "artifact_type_hint": "portable_text_skill",
                        "installable_by_current_policy": True,
                        "preview_available": True,
                        "tags": ["voice_output", "lightweight"],
                    }
                ]
            },
        )
        runtime = self._runtime()
        runtime.pack_store = _FakePackStore([installed_row])  # type: ignore[assignment]
        runtime._pack_registry_discovery = lambda: discovery  # type: ignore[method-assign]

        helper_snapshot = build_pack_state_snapshot(pack_store=runtime.pack_store, discovery=discovery)
        endpoint_snapshot = runtime.packs_state()

        self.assertEqual(helper_snapshot["summary"], endpoint_snapshot["summary"])
        self.assertEqual(helper_snapshot["packs"][0]["state_label"], "Installed · Disabled")
        self.assertEqual(endpoint_snapshot["packs"][0]["state_label"], "Installed · Disabled")
        self.assertEqual(helper_snapshot["available_packs"][0]["state_label"], "Available")
        self.assertEqual(endpoint_snapshot["available_packs"][0]["state_label"], "Available")
        self.assertEqual(helper_snapshot["packs"][0]["normalized_state"]["status_note"], "Installed, but disabled.")
        self.assertEqual(endpoint_snapshot["packs"][0]["normalized_state"]["status_note"], "Installed, but disabled.")

    def test_pack_recommendation_and_pack_state_use_the_same_disabled_truth(self) -> None:
        installed_row = {
            "pack_id": "pack.voice.local_fast",
            "name": "Local Voice",
            "status": "normalized",
            "enabled": False,
            "normalized_path": str(Path(__file__).resolve()),
            "canonical_pack": {
                "display_name": "Local Voice",
                "source": {"name": "Local", "kind": "local_catalog"},
                "pack_identity": {"canonical_id": "pack.voice.local_fast"},
                "capabilities": {
                    "summary": "Local speech output for this machine.",
                    "declared": ["voice_output"],
                },
            },
            "review_envelope": {"pack_name": "Local Voice"},
        }
        discovery = _FakeDiscovery(
            sources=[
                {"id": "local", "name": "Local Catalog", "kind": "local_catalog", "enabled": True, "allowed_by_policy": True},
            ],
            pack_map={
                ("local", "*"): [
                    {
                        "remote_id": "local-voice",
                        "name": "Local Voice",
                        "summary": "Lightweight local speech output for this machine.",
                        "artifact_type_hint": "portable_text_skill",
                        "installable_by_current_policy": True,
                        "preview_available": True,
                        "tags": ["voice_output", "lightweight"],
                    }
                ],
                ("local", "voice"): [
                    {
                        "remote_id": "local-voice",
                        "name": "Local Voice",
                        "summary": "Lightweight local speech output for this machine.",
                        "artifact_type_hint": "portable_text_skill",
                        "installable_by_current_policy": True,
                        "preview_available": True,
                        "tags": ["voice_output", "lightweight"],
                    }
                ],
            },
        )
        store = _FakePackStore([installed_row])
        snapshot = build_pack_state_snapshot(pack_store=store, discovery=discovery)

        recommendation = recommend_packs_for_capability(
            "Talk to me out loud",
            pack_store=store,
            pack_registry_discovery=discovery,
        )
        self.assertIsNotNone(recommendation)
        assert recommendation is not None
        self.assertIsNotNone(recommendation["installed_pack"])
        self.assertEqual(
            snapshot["packs"][0]["state_label"],
            recommendation["installed_pack"]["normalized_state"]["state_label"],
        )
        rendered = render_pack_capability_response(recommendation)
        self.assertIn("Voice output is installed, but it is disabled.", rendered)
        self.assertTrue(
            "Say yes and I'll show the install preview." in rendered
            or "I can keep responding in text." in rendered
        )

    def test_recommendation_degrades_to_text_only_when_discovery_is_unavailable(self) -> None:
        class _BrokenDiscovery:
            def list_sources(self) -> list[dict[str, object]]:
                raise RuntimeError("temporary discovery failure")

        store = _FakePackStore([])
        recommendation = recommend_packs_for_capability(
            "Talk to me out loud",
            pack_store=store,
            pack_registry_discovery=_BrokenDiscovery(),
        )
        self.assertIsNotNone(recommendation)
        assert recommendation is not None
        self.assertEqual("text_only", recommendation["fallback"])
        rendered = render_pack_capability_response(recommendation)
        self.assertIn("I can keep responding in text.", rendered)
        self.assertNotIn("install preview", rendered.lower())


if __name__ == "__main__":
    unittest.main()
