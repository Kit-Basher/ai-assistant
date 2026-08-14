from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import tempfile

from agent.api_server import AgentRuntime
from agent.llm.model_runtime_truth import ModelRuntimeTruth, canonical_model_ref, canonical_native_name
from agent.runtime_truth_service import RuntimeTruthService
from test_api_server import _config
from test_unified_conversation_routing import _chat


class _TruthRows:
    def __init__(self, rows):
        self.rows = rows

    def _runtime_inventory_rows(self):
        return list(self.rows)


class _Runtime:
    git_commit = "candidate"
    startup_phase = "ready"
    _startup_warmup_started = True
    _repo_root = "/nonexistent"
    config = SimpleNamespace(ollama_base_url="http://127.0.0.1:11434", ollama_host=None)

    def __init__(self, rows=()):
        self._truth = _TruthRows(rows)

    def get_defaults(self):
        return {
            "default_provider": "ollama",
            "default_model": "ollama:Gemma:latest",
            "resolved_default_model": "ollama:Gemma:latest",
            "allow_remote_fallback": False,
        }

    def runtime_truth_service(self):
        return self._truth


def _tag(name: str, digest: str, size: int = 1000):
    return {
        "name": name,
        "digest": digest,
        "size": size,
        "details": {"family": "test", "parameter_size": "4B", "quantization_level": "Q4"},
    }


def test_canonical_identity_normalizes_provider_case_and_latest_alias():
    assert canonical_native_name("OLLAMA:Gemma") == "gemma:latest"
    assert canonical_model_ref("OLLAMA", "Gemma") == "ollama:gemma:latest"
    assert canonical_model_ref("ollama", "ollama:Gemma:latest") == "ollama:gemma:latest"


def test_physical_inventory_collapses_same_digest_and_separates_remote_history():
    runtime = _Runtime(
        rows=[
            {"id": "ollama:gemma", "provider": "ollama", "model_name": "gemma", "available": True, "routable": True},
            {"id": "openrouter:remote/model", "provider": "openrouter", "model_name": "remote/model", "available": False},
        ]
    )
    service = ModelRuntimeTruth(runtime)
    with patch.object(service, "_ollama_tags", return_value=([_tag("Gemma:latest", "same"), _tag("gemma", "same")], None)):
        payload = service.refresh()
    assert payload["counts"]["physically_installed"] == 1
    assert len(payload["installed"][0]["aliases"]) >= 2
    assert payload["installed"][0]["effective"] is True
    assert len(payload["remote_registered"]) == 1


def test_provider_timeout_preserves_last_observation_as_unknown_not_absent():
    service = ModelRuntimeTruth(_Runtime())
    with patch.object(service, "_ollama_tags", return_value=([_tag("llama3:latest", "digest")], None)):
        first = service.refresh()
    with patch.object(service, "_ollama_tags", return_value=([], "provider_unavailable")):
        second = service.refresh()
    assert first["counts"]["physically_installed"] == 1
    assert second["counts"]["physically_installed"] == 1
    assert second["observation"]["status"] == "unknown"
    assert second["observation"]["stale"] is True


def test_frontdoor_selection_never_calls_readiness_or_provider_probe():
    fake = SimpleNamespace(
        get_defaults=lambda: {"default_provider": "ollama", "default_model": "ollama:Gemma:latest"},
        registry_document={"models": {}},
    )
    assert AgentRuntime.assistant_frontdoor_active(fake) is True


def test_observed_ready_snapshot_never_builds_live_readiness_when_cache_is_warm():
    runtime = _Runtime()
    runtime.safe_mode_target_status = lambda: {}
    service = RuntimeTruthService(runtime)
    service._snapshot_cache()["ready_status"] = {
        "created_at": 1.0,
        "value": {"ready": True, "message": "ready", "runtime_status": {"runtime_mode": "READY"}},
    }
    with patch.object(service, "_ready_status_uncached", side_effect=AssertionError("live probe forbidden")):
        payload = service.ready_status_observed()
    assert payload["ready"] is True
    assert "observation_stale" in payload


def test_runtime_status_uses_nonprobing_observation_path():
    runtime = _Runtime()
    runtime.safe_mode_target_status = lambda: {}
    service = RuntimeTruthService(runtime)
    with patch.object(service, "ready_status", side_effect=AssertionError("live probe forbidden")), patch.object(
        service,
        "ready_status_observed",
        return_value={"ready": True, "runtime_mode": "READY", "message": "Core chat is ready.", "runtime_status": {}},
    ), patch.object(
        service,
        "_configured_chat_target_status",
        return_value={"provider": "ollama", "model": "ollama:Gemma:latest"},
    ):
        payload = service.runtime_status("runtime_status")
    assert payload["ready"] is True
    assert payload["model"] == "ollama:Gemma:latest"


def test_production_chat_recommendation_uses_host_evidence_and_never_switches():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db")))
        truth = runtime.runtime_truth_service()
        added, add_result = runtime.add_provider_model(
            "ollama", {"model": "Gemma:latest", "capabilities": ["chat"], "available": True}
        )
        assert added, add_result
        ok, configured = runtime.update_defaults(
            {"default_provider": "ollama", "chat_model": "ollama:Gemma:latest"}
        )
        assert ok, configured
        before = runtime.get_defaults().get("default_model")
        fixture = {
            "ok": True,
            "selection": {"provider": "ollama", "default_model": "ollama:Gemma:latest", "effective_model": "ollama:Gemma:latest"},
            "evaluation": {
                "status": "current",
                "observed_at": "2026-08-13T00:00:00+00:00",
                "evaluated_models": [
                    {"model": "qwen2.5:3b-instruct", "score": {"passed": 8, "total": 9}, "latency": {"median_ms": 480}},
                    {"model": "Gemma:latest", "score": {"passed": 7, "total": 9}, "latency": {"median_ms": 2000}},
                ],
            },
            "recommendation": {"default_general_assistant": "qwen2.5:3b-instruct"},
            "installed": [],
        }
        with patch.object(truth, "model_runtime_truth", return_value=fixture):
            response = _chat(runtime, "which installed model is best for this assistant and why?", user="wp45", thread="wp45:t")
        assert response.get("ok") is True
        assert str((response.get("meta") or {}).get("route")) == "action_tool"
        assert "ollama:qwen2.5:3b-instruct" in str(response.get("message") or "")
        assert "no model was switched" in str(response.get("message") or "").lower()
        assert runtime.get_defaults().get("default_model") == before
