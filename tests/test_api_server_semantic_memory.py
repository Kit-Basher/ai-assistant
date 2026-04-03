from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import Config
from agent.llm.providers.base import Provider
from agent.llm.registry import DefaultsConfig, ModelConfig, ProviderConfig, Registry
from agent.llm.types import EmbeddingResponse, Request, Response, Usage
from agent.semantic_memory.service import SemanticMemoryService
from agent.semantic_memory.types import SemanticSourceKind


class _FakeEmbeddingProvider(Provider):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def available(self) -> bool:
        return True

    def chat(self, request: Request, *, model: str, timeout_seconds: float) -> Response:
        _ = request
        _ = model
        _ = timeout_seconds
        return Response(text="ok", provider=self._name, model=model, usage=Usage())

    def embed_texts(self, texts: tuple[str, ...], *, model: str, timeout_seconds: float) -> EmbeddingResponse:
        _ = model
        _ = timeout_seconds
        vectors: list[tuple[float, ...]] = []
        for text in texts:
            lowered = str(text or "").lower()
            if "cats" in lowered:
                vectors.append((1.0, 0.0, 0.0))
            elif "dogs" in lowered:
                vectors.append((0.0, 1.0, 0.0))
            else:
                vectors.append((0.0, 0.0, 1.0))
        return EmbeddingResponse(provider=self._name, model=model, vectors=tuple(vectors), usage=Usage())


def _config(db_path: str, **overrides: object) -> Config:
    base = Config(
        telegram_bot_token="token",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        openai_model_worker=None,
        agent_timezone="UTC",
        db_path=db_path,
        log_path=os.path.join(os.path.dirname(db_path), "agent.log"),
        skills_path="/tmp/skills",
        ollama_host="http://127.0.0.1:11434",
        ollama_model="llama3",
        ollama_model_sentinel=None,
        ollama_model_worker=None,
        allow_cloud=True,
        prefer_local=True,
        llm_timeout_seconds=15,
        llm_provider="none",
        enable_llm_presentation=False,
        openai_base_url=None,
        ollama_base_url="http://127.0.0.1:11434",
        anthropic_api_key=None,
        llm_selector="single",
        llm_broker_policy_path=None,
        llm_allow_remote=True,
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_model="openai/gpt-4o-mini",
        openrouter_site_url=None,
        openrouter_app_name=None,
        llm_registry_path=None,
        llm_routing_mode="auto",
        llm_retry_attempts=1,
        llm_retry_base_delay_ms=0,
        llm_circuit_breaker_failures=2,
        llm_circuit_breaker_window_seconds=60,
        llm_circuit_breaker_cooldown_seconds=30,
        llm_usage_stats_path=os.path.join(os.path.dirname(db_path), "usage_stats.json"),
        llm_health_state_path=os.path.join(os.path.dirname(db_path), "llm_health_state.json"),
        llm_automation_enabled=False,
        model_scout_state_path=os.path.join(os.path.dirname(db_path), "model_scout_state.json"),
        autopilot_notify_store_path=os.path.join(os.path.dirname(db_path), "llm_notifications.json"),
        semantic_memory_enabled=True,
    )
    return base.__class__(**{**base.__dict__, **overrides})


def _registry(embed_model: str = "fake:embed") -> Registry:
    providers = {
        "fake": ProviderConfig(
            id="fake",
            provider_type="openai_compat",
            base_url="http://fake.local/v1",
            chat_path="/v1/chat/completions",
            api_key_source=None,
            default_headers={},
            default_query_params={},
            enabled=True,
            local=True,
        )
    }
    models = {
        embed_model: ModelConfig(
            id=embed_model,
            provider="fake",
            model="fake-embed",
            capabilities=frozenset({"embedding"}),
            task_types=("embedding",),
            quality_rank=1,
            cost_rank=1,
            default_for=("embedding",),
            enabled=True,
            available=True,
            input_cost_per_million_tokens=None,
            output_cost_per_million_tokens=None,
            max_context_tokens=None,
        )
    }
    defaults = DefaultsConfig(
        routing_mode="auto",
        default_provider=None,
        default_model=None,
        allow_remote_fallback=True,
        chat_model=None,
        embed_model=embed_model,
        last_chat_model=None,
    )
    return Registry(schema_version=2, path=None, providers=providers, models=models, defaults=defaults, fallback_chain=())


class _HandlerForTest(APIServerHandler):
    def __init__(self, runtime_obj: AgentRuntime, path: str, payload: dict[str, object] | None = None) -> None:
        self.runtime = runtime_obj
        self.path = path
        self.headers = {"Content-Type": "application/json"}
        self.status_code = 0
        self.content_type = ""
        self.body = b""
        self._payload = payload or {}

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

    def _read_json(self) -> dict[str, object]:
        return self._payload

    def _request_is_loopback(self) -> bool:
        return True


class TestAPIServerSemanticMemory(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")
        self.runtime = AgentRuntime(_config(self.db_path))
        provider = _FakeEmbeddingProvider("fake")
        self.semantic_service = SemanticMemoryService(
            config=_config(self.db_path),
            registry=_registry(),
            provider_resolver=lambda provider_id: provider if provider_id == "fake" else None,
            db_path=os.path.join(self.tmpdir.name, "semantic.db"),
        )
        self.runtime._semantic_memory_service = self.semantic_service  # noqa: SLF001

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def _delete_vectors_for_source(self, source_id: str) -> None:
        conn = sqlite3.connect(self.semantic_service.store.db_path)
        try:
            conn.execute(
                "DELETE FROM semantic_vectors WHERE chunk_id IN (SELECT id FROM semantic_chunks WHERE source_id = ?)",
                (source_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def test_semantic_status_surface_reports_recovery_state(self) -> None:
        handler = _HandlerForTest(self.runtime, "/semantic/status")
        handler.do_GET()
        self.assertEqual(200, handler.status_code)
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertFalse(payload["healthy"])
        self.assertEqual("index_missing", payload["recovery"]["state"])
        self.assertIn("recommended_action", payload["recovery"])
        self.assertIn("summary", payload)
        self.assertIn("enabled", payload["summary"] or "")

    def test_document_ingest_endpoint_accepts_file_path(self) -> None:
        doc_path = Path(self.tmpdir.name) / "guide.md"
        doc_path.write_text("cats document ingestion path", encoding="utf-8")
        handler = _HandlerForTest(
            self.runtime,
            "/semantic/documents/ingest",
            {"path": str(doc_path), "title": "Guide"},
        )
        handler.do_POST()
        self.assertEqual(200, handler.status_code)
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual(SemanticSourceKind.DOCUMENT.value, payload["source_kind"])
        self.assertTrue(self.runtime._semantic_memory_service.status().healthy)  # noqa: SLF001

    def test_rebuild_endpoint_repairs_partial_index(self) -> None:
        doc_path = Path(self.tmpdir.name) / "repair.md"
        doc_path.write_text("cats repair flow and provenance", encoding="utf-8")
        ingest_handler = _HandlerForTest(
            self.runtime,
            "/semantic/documents/ingest",
            {"path": str(doc_path), "title": "Repair"},
        )
        ingest_handler.do_POST()
        ingest_payload = json.loads(ingest_handler.body.decode("utf-8"))
        source_id = str(ingest_payload["source_id"])

        self._delete_vectors_for_source(source_id)
        stale_status = self.runtime.semantic_memory_status()
        self.assertIn(stale_status["recovery"]["state"], {"index_state_mismatch", "partial_index"})

        rebuild_handler = _HandlerForTest(
            self.runtime,
            "/semantic/rebuild",
            {"scope": f"document:{doc_path}"},
        )
        rebuild_handler.do_POST()
        self.assertEqual(200, rebuild_handler.status_code)
        rebuild_payload = json.loads(rebuild_handler.body.decode("utf-8"))
        self.assertTrue(rebuild_payload["ok"])
        self.assertEqual("ready", rebuild_payload["status"])
        self.assertEqual(1, len(rebuild_payload["rebuilt_sources"]))
        self.assertIn("summary", rebuild_payload)
        self.assertIn("rebuilt=1", rebuild_payload["summary"])
        self.assertTrue(self.runtime.semantic_memory_status()["healthy"])


if __name__ == "__main__":
    unittest.main()
