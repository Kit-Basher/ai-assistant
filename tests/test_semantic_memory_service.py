from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from agent.config import Config
from agent.llm.providers.base import Provider
from agent.llm.registry import DefaultsConfig, ModelConfig, ProviderConfig, Registry
from agent.llm.types import EmbeddingResponse, Request, Response, Usage
from agent.semantic_memory.service import SemanticMemoryService
from agent.semantic_memory.types import SemanticSourceKind


class _FakeEmbeddingProvider(Provider):
    def __init__(self, name: str, *, available: bool = True, vector_mode: str = "narrow") -> None:
        self._name = name
        self._available = available
        self.vector_mode = vector_mode

    @property
    def name(self) -> str:
        return self._name

    def available(self) -> bool:
        return self._available

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
            if self.vector_mode == "wide":
                if "cats" in lowered:
                    vectors.append((1.0, 0.0, 0.0, 0.0))
                elif "dogs" in lowered:
                    vectors.append((0.0, 1.0, 0.0, 0.0))
                else:
                    vectors.append((0.0, 0.0, 1.0, 0.0))
            else:
                if "cats" in lowered:
                    vectors.append((1.0, 0.0, 0.0))
                elif "dogs" in lowered:
                    vectors.append((0.0, 1.0, 0.0))
                else:
                    vectors.append((0.0, 0.0, 1.0))
        return EmbeddingResponse(provider=self._name, model=model, vectors=tuple(vectors), usage=Usage())


def _config(tmpdir: str, **overrides: object) -> Config:
    base = Config(
        telegram_bot_token="token",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        openai_model_worker=None,
        agent_timezone="UTC",
        db_path=f"{tmpdir}/agent.db",
        log_path=f"{tmpdir}/agent.log",
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
        llm_usage_stats_path=f"{tmpdir}/usage.json",
        semantic_memory_enabled=True,
        semantic_memory_conversations_enabled=True,
        semantic_memory_notes_enabled=True,
        semantic_memory_documents_enabled=True,
        semantic_memory_max_candidates=4,
        semantic_memory_max_prompt_chars=400,
        semantic_memory_chunk_size=120,
        semantic_memory_query_limit=50,
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
            task_types=(),
            quality_rank=1,
            cost_rank=1,
            default_for=("embedding",),
            enabled=True,
            available=True,
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


class TestSemanticMemoryService(unittest.TestCase):
    @staticmethod
    def _delete_vectors_for_source(service: SemanticMemoryService, source_id: str) -> None:
        conn = sqlite3.connect(service.store.db_path)
        try:
            conn.execute(
                "DELETE FROM semantic_vectors WHERE chunk_id IN (SELECT id FROM semantic_chunks WHERE source_id = ?)",
                (source_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def test_enabled_retrieves_and_ranks_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = _FakeEmbeddingProvider("fake")
            service = SemanticMemoryService(
                config=_config(tmpdir),
                registry=_registry(),
                provider_resolver=lambda provider_id: provider if provider_id == "fake" else None,
                db_path=f"{tmpdir}/semantic.db",
            )
            service.ingest_text(
                source_kind=SemanticSourceKind.NOTE,
                source_ref="note:1",
                text="cats alpha favorite",
                scope="project:alpha",
                title="Pinned alpha note",
                project_id="alpha",
                pinned=True,
            )
            service.ingest_text(
                source_kind=SemanticSourceKind.NOTE,
                source_ref="note:2",
                text="cats beta backup",
                scope="global",
                title="Global note",
                project_id="beta",
                pinned=False,
            )
            selection = service.build_context_for_payload(
                {
                    "messages": [{"role": "user", "content": "cats please"}],
                    "project": "alpha",
                    "thread_id": "thread-1",
                    "user_id": "user-1",
                },
                intent="chat",
            )

            self.assertEqual(["SC-"], [candidate.id[:3] for candidate in selection.candidates[:1]])
            self.assertGreaterEqual(len(selection.candidates), 1)
            self.assertEqual("note:1", selection.candidates[0].source_ref)
            self.assertIn("SEMANTIC_MEMORY[", selection.merged_context_text)
            self.assertLessEqual(len(selection.merged_context_text), 400)
            self.assertTrue(selection.debug.get("selected_ids"))

    def test_disabled_falls_back_to_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = _FakeEmbeddingProvider("fake")
            service = SemanticMemoryService(
                config=_config(tmpdir, semantic_memory_enabled=False),
                registry=_registry(),
                provider_resolver=lambda provider_id: provider if provider_id == "fake" else None,
                db_path=f"{tmpdir}/semantic.db",
            )
            selection = service.build_context_for_payload(
                {"messages": [{"role": "user", "content": "cats"}], "project": "alpha"},
                intent="chat",
            )
            self.assertEqual([], selection.candidates)
            self.assertEqual("semantic_memory_disabled", selection.debug.get("reason"))

    def test_source_kind_flags_disable_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = _FakeEmbeddingProvider("fake")
            service = SemanticMemoryService(
                config=_config(tmpdir, semantic_memory_enabled=True, semantic_memory_notes_enabled=False),
                registry=_registry(),
                provider_resolver=lambda provider_id: provider if provider_id == "fake" else None,
                db_path=f"{tmpdir}/semantic.db",
            )
            ingest_result = service.ingest_text(
                source_kind=SemanticSourceKind.NOTE,
                source_ref="note:1",
                text="cats alpha favorite",
                scope="project:alpha",
                pinned=True,
            )
            self.assertEqual("note_semantic_disabled", ingest_result["reason"])
            selection = service.build_context_for_payload(
                {"messages": [{"role": "user", "content": "cats"}], "project": "alpha"},
                intent="chat",
            )
            self.assertEqual([], selection.candidates)
            self.assertIn(selection.debug.get("reason"), {"index_missing", "semantic_source_kinds_disabled"})

    def test_missing_index_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = _FakeEmbeddingProvider("fake")
            service = SemanticMemoryService(
                config=_config(tmpdir),
                registry=_registry(),
                provider_resolver=lambda provider_id: provider if provider_id == "fake" else None,
                db_path=f"{tmpdir}/semantic.db",
            )
            selection = service.build_context_for_payload(
                {"messages": [{"role": "user", "content": "cats"}]},
                intent="chat",
            )
            self.assertEqual([], selection.candidates)
            self.assertEqual("index_missing", selection.debug.get("reason"))

    def test_provider_unavailable_skips_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = _FakeEmbeddingProvider("fake")
            service = SemanticMemoryService(
                config=_config(tmpdir),
                registry=_registry(),
                provider_resolver=lambda provider_id: provider if provider_id == "fake" else None,
                db_path=f"{tmpdir}/semantic.db",
            )
            service.ingest_text(
                source_kind=SemanticSourceKind.NOTE,
                source_ref="note:1",
                text="cats alpha favorite",
                scope="project:alpha",
                pinned=True,
            )
            unavailable_service = SemanticMemoryService(
                config=_config(tmpdir),
                registry=_registry(),
                provider_resolver=lambda _provider_id: None,
                db_path=f"{tmpdir}/semantic.db",
            )
            selection = unavailable_service.build_context_for_payload(
                {"messages": [{"role": "user", "content": "cats"}], "project": "alpha"},
                intent="chat",
            )
            self.assertEqual([], selection.candidates)
            self.assertEqual("embedding_provider_unavailable", selection.debug.get("reason"))

    def test_stale_index_is_skipped_when_embed_model_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = _FakeEmbeddingProvider("fake")
            registry = _registry()
            service = SemanticMemoryService(
                config=_config(tmpdir),
                registry=registry,
                provider_resolver=lambda provider_id: provider if provider_id == "fake" else None,
                db_path=f"{tmpdir}/semantic.db",
            )
            service.ingest_text(
                source_kind=SemanticSourceKind.NOTE,
                source_ref="note:1",
                text="cats alpha favorite",
                scope="project:alpha",
                pinned=True,
            )
            stale_registry = Registry(
                schema_version=registry.schema_version,
                path=registry.path,
                providers=registry.providers,
                models={
                    **registry.models,
                    "fake:other-embed": ModelConfig(
                        id="fake:other-embed",
                        provider="fake",
                        model="fake-embed-v2",
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
                    ),
                },
                defaults=DefaultsConfig(
                    routing_mode="auto",
                    default_provider=None,
                    default_model=None,
                    allow_remote_fallback=True,
                    chat_model=None,
                    embed_model="fake:other-embed",
                    last_chat_model=None,
                ),
                fallback_chain=registry.fallback_chain,
            )
            service.registry = stale_registry
            selection = service.build_context_for_payload(
                {"messages": [{"role": "user", "content": "cats"}], "project": "alpha"},
                intent="chat",
            )
            self.assertEqual([], selection.candidates)
            self.assertEqual("stale_embed_model", selection.debug.get("reason"))

    def test_dimension_mismatch_skips_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = _FakeEmbeddingProvider("fake", vector_mode="narrow")
            service = SemanticMemoryService(
                config=_config(tmpdir),
                registry=_registry(),
                provider_resolver=lambda provider_id: provider if provider_id == "fake" else None,
                db_path=f"{tmpdir}/semantic.db",
            )
            service.ingest_text(
                source_kind=SemanticSourceKind.NOTE,
                source_ref="note:1",
                text="cats alpha favorite",
                scope="project:alpha",
                pinned=True,
            )
            provider.vector_mode = "wide"
            selection = service.build_context_for_payload(
                {"messages": [{"role": "user", "content": "cats"}], "project": "alpha"},
                intent="chat",
            )
            self.assertEqual([], selection.candidates)
            skipped_reasons = {str(row.get("reason") or "") for row in selection.debug.get("skipped", []) if isinstance(row, dict)}
            self.assertIn("dimension_mismatch", skipped_reasons)

    def test_prompt_limits_and_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = _FakeEmbeddingProvider("fake")
            service = SemanticMemoryService(
                config=_config(tmpdir, semantic_memory_max_prompt_chars=220, semantic_memory_max_candidates=2),
                registry=_registry(),
                provider_resolver=lambda provider_id: provider if provider_id == "fake" else None,
                db_path=f"{tmpdir}/semantic.db",
            )
            service.ingest_text(
                source_kind=SemanticSourceKind.DOCUMENT,
                source_ref="docs/spec.md",
                text="cats " + ("lorem ipsum dolor sit amet " * 20),
                scope="document:docs/spec.md",
                title="Spec",
                pinned=True,
            )
            selection = service.build_context_for_payload(
                {"messages": [{"role": "user", "content": "cats"}], "document_ref": "docs/spec.md"},
                intent="chat",
            )
            self.assertLessEqual(len(selection.merged_context_text), 220)
            self.assertIn('"', selection.merged_context_text)
            self.assertTrue(selection.merged_context_text.startswith("SEMANTIC_MEMORY["))

    def test_prompt_budget_holds_for_multiple_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = _FakeEmbeddingProvider("fake")
            service = SemanticMemoryService(
                config=_config(tmpdir, semantic_memory_max_prompt_chars=500, semantic_memory_max_candidates=3),
                registry=_registry(),
                provider_resolver=lambda provider_id: provider if provider_id == "fake" else None,
                db_path=f"{tmpdir}/semantic.db",
            )
            for index in range(3):
                service.ingest_text(
                    source_kind=SemanticSourceKind.DOCUMENT,
                    source_ref=f"docs/spec-{index}.md",
                    text=f"cats candidate {index} " + ("lorem ipsum dolor sit amet " * 8),
                    scope=f"document:docs/spec-{index}.md",
                    title=f"Spec {index}",
                    pinned=True,
                )
            selection = service.build_context_for_payload(
                {"messages": [{"role": "user", "content": "cats"}], "document_ref": "docs/spec-1.md"},
                intent="chat",
            )
            self.assertLessEqual(len(selection.merged_context_text), 500)
            self.assertLessEqual(selection.merged_context_text.count("SEMANTIC_MEMORY["), 3)

    def test_document_file_ingestion_and_repair_scope_rebuilds_missing_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = _FakeEmbeddingProvider("fake")
            service = SemanticMemoryService(
                config=_config(tmpdir),
                registry=_registry(),
                provider_resolver=lambda provider_id: provider if provider_id == "fake" else None,
                db_path=f"{tmpdir}/semantic.db",
            )
            doc_path = Path(tmpdir) / "guide.md"
            doc_path.write_text("cats document provenance and repair path", encoding="utf-8")

            ingest = service.ingest_document_file(path=str(doc_path), title="Guide")
            self.assertTrue(ingest["ok"])
            source_id = str(ingest["source_id"])
            source = service.store.get_source(source_id)
            self.assertIsNotNone(source)
            self.assertEqual(SemanticSourceKind.DOCUMENT, source.source_kind if source else None)

            self._delete_vectors_for_source(service, source_id)

            stale_status = service.status()
            self.assertFalse(stale_status.healthy)
            self.assertIn(stale_status.reason, {"index_state_mismatch", "partial_index"})

            repair = service.repair_scope(scope=f"document:{doc_path}")
            self.assertTrue(repair["ok"])
            self.assertEqual(1, len(repair["rebuilt_sources"]))
            self.assertEqual([], repair["failed_sources"])
            self.assertEqual([], repair["skipped_sources"])
            self.assertTrue(service.status().healthy)
            selection = service.build_context_for_payload(
                {"messages": [{"role": "user", "content": "cats"}], "document_ref": str(doc_path)},
                intent="chat",
            )
            self.assertTrue(selection.candidates)
            self.assertEqual(SemanticSourceKind.DOCUMENT, selection.candidates[0].source_kind)

    def test_stale_embed_model_repair_updates_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = _FakeEmbeddingProvider("fake")
            registry = _registry("fake:embed-v1")
            service = SemanticMemoryService(
                config=_config(tmpdir),
                registry=registry,
                provider_resolver=lambda provider_id: provider if provider_id == "fake" else None,
                db_path=f"{tmpdir}/semantic.db",
            )
            service.ingest_text(
                source_kind=SemanticSourceKind.NOTE,
                source_ref="note:1",
                text="cats alpha favorite",
                scope="project:alpha",
                pinned=True,
            )
            updated_registry = _registry("fake:embed-v2")
            updated_registry = Registry(
                schema_version=updated_registry.schema_version,
                path=updated_registry.path,
                providers=updated_registry.providers,
                models={
                    **updated_registry.models,
                    "fake:embed-v2": ModelConfig(
                        id="fake:embed-v2",
                        provider="fake",
                        model="fake-embed-v2",
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
                    ),
                },
                defaults=DefaultsConfig(
                    routing_mode="auto",
                    default_provider=None,
                    default_model=None,
                    allow_remote_fallback=True,
                    chat_model=None,
                    embed_model="fake:embed-v2",
                    last_chat_model=None,
                ),
                fallback_chain=(),
            )
            service.registry = updated_registry
            stale_status = service.status()
            self.assertFalse(stale_status.healthy)
            self.assertEqual("stale_embed_model", stale_status.reason)

            repair = service.repair_scope(scope="project:alpha")
            self.assertTrue(repair["ok"])
            self.assertEqual("ready", repair["status"])
            self.assertTrue(service.status().healthy)
            index_state = service.store.get_index_state()
            self.assertIsNotNone(index_state)
            self.assertEqual("fake:embed-v2", index_state.embed_model if index_state else None)

    def test_missing_index_status_surface_reports_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = _FakeEmbeddingProvider("fake")
            service = SemanticMemoryService(
                config=_config(tmpdir),
                registry=_registry(),
                provider_resolver=lambda provider_id: provider if provider_id == "fake" else None,
                db_path=f"{tmpdir}/semantic.db",
            )
            report = service.report()
            self.assertFalse(report["healthy"])
            self.assertEqual("index_missing", report["recovery"]["state"])
            self.assertEqual("run a semantic rebuild", report["recovery"]["recommended_action"])

    def test_embedding_dimension_validation_is_enforced_at_storage_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = _FakeEmbeddingProvider("fake")
            service = SemanticMemoryService(
                config=_config(tmpdir),
                registry=_registry(),
                provider_resolver=lambda provider_id: provider if provider_id == "fake" else None,
                db_path=f"{tmpdir}/semantic.db",
            )
            service.store.upsert_source(
                source_id="SS-test",
                source_kind=SemanticSourceKind.DOCUMENT,
                source_ref="doc:1",
                scope="document:1",
                content_hash="hash",
                metadata={},
            )
            service.store.replace_chunks(
                source_id="SS-test",
                chunks=[
                    {
                        "id": "SC-test",
                        "chunk_index": 0,
                        "text": "cats",
                        "chunk_hash": "chunk-hash",
                        "char_start": 0,
                        "char_end": 4,
                        "created_at": 1,
                        "updated_at": 1,
                        "metadata": {},
                    }
                ],
            )
            with self.assertRaises(ValueError):
                service.store.upsert_vector(
                    chunk_id="SC-test",
                    embed_provider="fake",
                    embed_model="fake:embed",
                    embedding_dim=3,
                    vector=(1.0, 0.0),
                )


if __name__ == "__main__":
    unittest.main()
