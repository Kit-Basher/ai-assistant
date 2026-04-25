from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.llm.model_discovery_manager import ModelDiscoveryManager, _normalize_result_row
from agent.modelops.discovery import list_models_openrouter
from agent.orchestrator import Orchestrator
from agent.runtime_truth_service import RuntimeTruthService
from memory.db import MemoryDB


class _FakeHTTPResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _ = exc_type
        _ = exc
        _ = tb
        return False


class _FakeRuntime:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            llm_registry_path="",
            openrouter_api_key="test-openrouter-secret",
        )
        self.registry_document = {
            "providers": {
                "openrouter": {
                    "enabled": True,
                }
            }
        }
        self.secret_store = SimpleNamespace(
            get_secret=lambda _name: "test-openrouter-secret",
        )


class _RegistrylessRuntime(_FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.registry_document = None


class TestModelDiscoveryManagerIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "test.db")
        self.log_path = os.path.join(self.tmpdir.name, "events.log")
        self.skills_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills"))

        self.db = MemoryDB(self.db_path)
        schema_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql")
        )
        self.db.init_schema(schema_path)

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def _manager(self) -> ModelDiscoveryManager:
        return ModelDiscoveryManager(runtime=_FakeRuntime())

    def _orchestrator(self) -> Orchestrator:
        return Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
            runtime_truth_service=RuntimeTruthService(_FakeRuntime()),
        )

    def test_orchestrator_discovery_intents_route_through_model_discovery_manager(self) -> None:
        orchestrator = self._orchestrator()
        payload = {
            "ok": True,
            "query": "find models",
            "message": "Found 1 model(s).",
            "models": [
                {
                    "id": "ollama:qwen3.5:4b",
                    "provider": "ollama",
                    "source": "ollama",
                    "capabilities": ["chat"],
                    "local": True,
                    "installable": True,
                    "confidence": 0.9,
                }
            ],
            "sources": [
                {
                    "source": "ollama",
                    "enabled": True,
                    "queried": True,
                    "ok": True,
                    "count": 1,
                    "error_kind": None,
                    "error": None,
                }
            ],
            "debug": {"source_registry": ["huggingface", "openrouter", "ollama", "external_snapshots"]},
        }

        with patch("agent.llm.model_discovery_manager.ModelDiscoveryManager.query", return_value=payload) as query_mock:
            discovery_response = orchestrator._model_scout_discovery_response("find models")
            ollama_response = orchestrator._find_ollama_models_response("installed ollama")

        self.assertEqual(2, query_mock.call_count)
        self.assertEqual("find models", query_mock.call_args_list[0].args[0])
        self.assertEqual({}, query_mock.call_args_list[0].args[1])
        self.assertEqual("installed ollama", query_mock.call_args_list[1].args[0])
        self.assertEqual({"sources": ["ollama"]}, query_mock.call_args_list[1].args[1])
        self.assertEqual(["model_discovery_manager"], discovery_response.data["used_tools"])
        self.assertEqual(["model_discovery_manager"], ollama_response.data["used_tools"])

    def test_missing_runtime_registry_document_bootstraps_default_registry(self) -> None:
        manager = ModelDiscoveryManager(runtime=_RegistrylessRuntime())

        with patch.object(
            manager,
            "_query_openrouter",
            return_value=(
                [],
                {
                    "source": "openrouter",
                    "enabled": True,
                    "queried": True,
                    "ok": True,
                    "count": 0,
                    "error_kind": None,
                    "error": None,
                },
            ),
        ) as openrouter_mock:
            result = manager.query(None, {"sources": ["openrouter"]})

        self.assertTrue(result["ok"])
        registry_document = openrouter_mock.call_args.kwargs["registry_document"]
        self.assertEqual("prefer_local_lowest_cost_capable", registry_document["defaults"]["routing_mode"])
        self.assertIn("openrouter", registry_document["providers"])

    def test_partial_source_failure_keeps_other_discovery_sources(self) -> None:
        manager = self._manager()

        hf_status = {
            "source": "huggingface",
            "enabled": True,
            "queried": True,
            "ok": False,
            "count": 0,
            "error_kind": "fetch_failed",
            "error": "hf timeout",
        }

        with patch.object(
            manager,
            "_query_huggingface",
            return_value=([], hf_status),
        ), patch.object(
            manager,
            "_query_openrouter",
            return_value=(
                [
                    {
                        "id": "openrouter:vendor/coder-pro",
                        "provider": "openrouter",
                        "source": "openrouter",
                        "capabilities": ["chat"],
                        "local": False,
                        "installable": True,
                        "confidence": 0.75,
                    }
                ],
                {
                    "source": "openrouter",
                    "enabled": True,
                    "queried": True,
                    "ok": True,
                    "count": 1,
                    "error_kind": None,
                    "error": None,
                },
            ),
        ), patch.object(
            manager,
            "_query_ollama",
            return_value=(
                [
                    {
                        "id": "ollama:qwen3.5:4b",
                        "provider": "ollama",
                        "source": "ollama",
                        "capabilities": ["chat"],
                        "local": True,
                        "installable": True,
                        "confidence": 0.9,
                    }
                ],
                {
                    "source": "ollama",
                    "enabled": True,
                    "queried": True,
                    "ok": True,
                    "count": 1,
                    "error_kind": None,
                    "error": None,
                },
            ),
        ), patch.object(
            manager,
            "_query_external_snapshots",
            return_value=(
                [],
                {
                    "source": "external_snapshots",
                    "enabled": True,
                    "queried": True,
                    "ok": True,
                    "count": 0,
                    "error_kind": None,
                    "error": None,
                },
            ),
        ):
            result = manager.query(None, {})

        self.assertTrue(result["ok"])
        self.assertEqual(2, len(result["models"]))
        self.assertIn("some sources failed", result["message"])
        self.assertIn("huggingface: hf timeout", result["message"])
        self.assertIn("source_errors", result["debug"])
        self.assertEqual("fetch_failed", result["debug"]["source_errors"]["huggingface"]["error_kind"])
        self.assertEqual("hf timeout", result["debug"]["source_errors"]["huggingface"]["error"])

    def test_partial_fuzzy_gemma_prompt_broadens_and_ranks_family_match_first(self) -> None:
        manager = self._manager()
        hf_queries: list[str | None] = []

        def _hf_side_effect(query: str | None):
            hf_queries.append(query)
            lowered = str(query or "").strip().lower()
            if "gemma" in lowered:
                return (
                    [
                        {
                            "id": "huggingface:google/gemma-2-2b-it",
                            "provider": "huggingface",
                            "source": "huggingface",
                            "model_name": "google/gemma-2-2b-it",
                            "capabilities": ["chat"],
                            "local": False,
                            "installable": False,
                            "confidence": 0.78,
                        }
                    ],
                    {
                        "source": "huggingface",
                        "enabled": True,
                        "queried": True,
                        "ok": True,
                        "count": 1,
                        "error_kind": None,
                        "error": None,
                    },
                )
            return (
                [],
                {
                    "source": "huggingface",
                    "enabled": True,
                    "queried": True,
                    "ok": True,
                    "count": 0,
                    "error_kind": None,
                    "error": None,
                },
            )

        with patch.object(manager, "_query_huggingface", side_effect=_hf_side_effect), patch.object(
            manager,
            "_query_openrouter",
            return_value=(
                [
                    {
                        "id": "openrouter:google/gemma-2-9b",
                        "provider": "openrouter",
                        "source": "openrouter",
                        "model_name": "google/gemma-2-9b",
                        "capabilities": ["chat"],
                        "local": False,
                        "installable": False,
                        "confidence": 0.95,
                    }
                ],
                {
                    "source": "openrouter",
                    "enabled": True,
                    "queried": True,
                    "ok": True,
                    "count": 1,
                    "error_kind": None,
                    "error": None,
                },
            ),
        ), patch.object(
            manager,
            "_query_ollama",
            return_value=(
                [
                    {
                        "id": "ollama:qwen2.5:7b-instruct",
                        "provider": "ollama",
                        "source": "ollama",
                        "model_name": "qwen2.5:7b-instruct",
                        "capabilities": ["chat"],
                        "local": True,
                        "installable": True,
                        "confidence": 0.9,
                    }
                ],
                {
                    "source": "ollama",
                    "enabled": True,
                    "queried": True,
                    "ok": True,
                    "count": 1,
                    "error_kind": None,
                    "error": None,
                },
            ),
        ), patch.object(
            manager,
            "_query_external_snapshots",
            return_value=(
                [],
                {
                    "source": "external_snapshots",
                    "enabled": True,
                    "queried": True,
                    "ok": True,
                    "count": 0,
                    "error_kind": None,
                    "error": None,
                },
            ),
        ):
            result = manager.query("tiny Gemma 4", {})

        self.assertTrue(result["ok"])
        self.assertGreater(len(hf_queries), 1)
        self.assertEqual("tiny Gemma 4", hf_queries[0])
        self.assertTrue(any("gemma" in str(query or "").lower() for query in hf_queries[1:]))
        self.assertEqual("huggingface:google/gemma-2-2b-it", result["models"][0]["id"])
        self.assertGreaterEqual(len(result["models"]), 2)
        self.assertGreater(float(result["models"][0]["match_score"]), float(result["models"][1]["match_score"]))
        self.assertIn("broadened the search", result["message"].lower())
        self.assertIn("likely model(s)", result["message"].lower())

    def test_small_local_coding_prompt_prefers_local_candidate_before_generic_alternative(self) -> None:
        manager = self._manager()

        with patch.object(
            manager,
            "_query_huggingface",
            return_value=(
                [],
                {
                    "source": "huggingface",
                    "enabled": True,
                    "queried": True,
                    "ok": True,
                    "count": 0,
                    "error_kind": None,
                    "error": None,
                },
            ),
        ), patch.object(
            manager,
            "_query_openrouter",
            return_value=(
                [
                    {
                        "id": "openrouter:vendor/assistant-lite",
                        "provider": "openrouter",
                        "source": "openrouter",
                        "model_name": "vendor/assistant-lite",
                        "capabilities": ["chat"],
                        "local": False,
                        "installable": False,
                        "confidence": 0.96,
                    }
                ],
                {
                    "source": "openrouter",
                    "enabled": True,
                    "queried": True,
                    "ok": True,
                    "count": 1,
                    "error_kind": None,
                    "error": None,
                },
            ),
        ), patch.object(
            manager,
            "_query_ollama",
            return_value=(
                [
                    {
                        "id": "ollama:qwen2.5-coder:7b",
                        "provider": "ollama",
                        "source": "ollama",
                        "model_name": "qwen2.5-coder:7b",
                        "capabilities": ["chat"],
                        "local": True,
                        "installable": True,
                        "confidence": 0.88,
                    }
                ],
                {
                    "source": "ollama",
                    "enabled": True,
                    "queried": True,
                    "ok": True,
                    "count": 1,
                    "error_kind": None,
                    "error": None,
                },
            ),
        ), patch.object(
            manager,
            "_query_external_snapshots",
            return_value=(
                [],
                {
                    "source": "external_snapshots",
                    "enabled": True,
                    "queried": True,
                    "ok": True,
                    "count": 0,
                    "error_kind": None,
                    "error": None,
                },
            ),
        ):
            result = manager.query("small local coding model", {})

        self.assertTrue(result["ok"])
        self.assertGreaterEqual(len(result["models"]), 1)
        self.assertEqual("ollama:qwen2.5-coder:7b", result["models"][0]["id"])
        self.assertTrue(bool(result["models"][0]["local"]))
        self.assertIn("likely model(s)", result["message"].lower())

    def test_empty_results_include_explanation_and_actionable_next_steps(self) -> None:
        manager = self._manager()

        ok_status = {
            "source": "huggingface",
            "enabled": True,
            "queried": True,
            "ok": True,
            "count": 0,
            "error_kind": None,
            "error": None,
        }

        with patch.object(manager, "_query_huggingface", return_value=([], ok_status)), patch.object(
            manager,
            "_query_openrouter",
            return_value=([], dict(ok_status, source="openrouter")),
        ), patch.object(
            manager,
            "_query_ollama",
            return_value=([], dict(ok_status, source="ollama")),
        ), patch.object(
            manager,
            "_query_external_snapshots",
            return_value=([], dict(ok_status, source="external_snapshots")),
        ):
            result = manager.query("alpha beta", {})

        self.assertTrue(result["ok"])
        self.assertEqual([], result["models"])
        self.assertEqual(0, result["debug"]["matched_count"])
        self.assertIn("I searched for 'alpha beta'.", result["message"])
        self.assertIn("I couldn't find a close fit.", result["message"])
        self.assertIn("Try a model family, a size hint, or check whether the queried sources are enabled.", result["message"])

    def test_broadened_search_after_no_exact_match_uses_variant_queries(self) -> None:
        manager = self._manager()
        hf_queries: list[str | None] = []

        def _hf_side_effect(query: str | None):
            hf_queries.append(query)
            lowered = str(query or "").strip().lower()
            if lowered == "new qwen coder":
                return (
                    [],
                    {
                        "source": "huggingface",
                        "enabled": True,
                        "queried": True,
                        "ok": True,
                        "count": 0,
                        "error_kind": None,
                        "error": None,
                    },
                )
            if "qwen coder" in lowered:
                return (
                    [
                        {
                            "id": "huggingface:qwen2.5-coder-7b",
                            "provider": "huggingface",
                            "source": "huggingface",
                            "model_name": "qwen2.5-coder-7b",
                            "capabilities": ["chat"],
                            "local": False,
                            "installable": False,
                            "confidence": 0.72,
                        }
                    ],
                    {
                        "source": "huggingface",
                        "enabled": True,
                        "queried": True,
                        "ok": True,
                        "count": 1,
                        "error_kind": None,
                        "error": None,
                    },
                )
            return (
                [],
                {
                    "source": "huggingface",
                    "enabled": True,
                    "queried": True,
                    "ok": True,
                    "count": 0,
                    "error_kind": None,
                    "error": None,
                },
            )

        with patch.object(manager, "_query_huggingface", side_effect=_hf_side_effect), patch.object(
            manager,
            "_query_openrouter",
            return_value=(
                [],
                {
                    "source": "openrouter",
                    "enabled": True,
                    "queried": True,
                    "ok": True,
                    "count": 0,
                    "error_kind": None,
                    "error": None,
                },
            ),
        ), patch.object(
            manager,
            "_query_ollama",
            return_value=(
                [],
                {
                    "source": "ollama",
                    "enabled": True,
                    "queried": True,
                    "ok": True,
                    "count": 0,
                    "error_kind": None,
                    "error": None,
                },
            ),
        ), patch.object(
            manager,
            "_query_external_snapshots",
            return_value=(
                [],
                {
                    "source": "external_snapshots",
                    "enabled": True,
                    "queried": True,
                    "ok": True,
                    "count": 0,
                    "error_kind": None,
                    "error": None,
                },
            ),
        ):
            result = manager.query("new qwen coder", {})

        self.assertTrue(result["ok"])
        self.assertGreater(len(hf_queries), 1)
        self.assertEqual("new qwen coder", hf_queries[0])
        self.assertIn("qwen coder", " ".join(str(query or "") for query in hf_queries[1:]).lower())
        self.assertEqual("huggingface:qwen2.5-coder-7b", result["models"][0]["id"])
        self.assertIn("broadened to", result["message"].lower())
        self.assertIn("qwen coder", result["message"].lower())

    def test_normalized_rows_expose_required_fields_and_preserve_richer_metadata(self) -> None:
        row = _normalize_result_row(
            source_id="openrouter",
            row={
                "id": "vendor/fast-chat",
                "provider_id": "openrouter",
                "provider": "openrouter",
                "model": "vendor/fast-chat",
                "model_name": "vendor/fast-chat",
                "capabilities": ["chat"],
                "local": False,
                "available": True,
                "pricing": {"prompt_per_million": 0.2},
                "custom_field": "preserved",
            },
            query=None,
        )

        self.assertEqual("openrouter:vendor/fast-chat", row["id"])
        self.assertEqual("openrouter", row["provider"])
        self.assertEqual("openrouter", row["source"])
        self.assertIsInstance(row["capabilities"], list)
        self.assertFalse(row["local"])
        self.assertTrue(row["installable"])
        self.assertIsInstance(row["confidence"], float)
        self.assertIn("pricing", row)
        self.assertEqual({"prompt_per_million": 0.2}, row["pricing"])
        self.assertEqual("preserved", row["custom_field"])

    def test_legacy_openrouter_discovery_path_returns_live_rows(self) -> None:
        payload = json.dumps(
            {
                "data": [
                    {
                        "id": "openrouter:vendor/coder-pro",
                        "pricing": {
                            "prompt_per_million": 1.25,
                            "completion_per_million": 2.5,
                        },
                    }
                ]
            }
        ).encode("utf-8")

        with patch("agent.modelops.discovery.urllib.request.urlopen", return_value=_FakeHTTPResponse(payload)):
            rows = list_models_openrouter("test-openrouter-secret")

        self.assertEqual(1, len(rows))
        self.assertEqual("openrouter", rows[0].provider)
        self.assertEqual("openrouter:vendor/coder-pro", rows[0].model_id)
        self.assertEqual("openrouter_models", rows[0].metadata["source"])
        self.assertEqual(1.25, rows[0].metadata["pricing"]["prompt_per_million"])


if __name__ == "__main__":
    unittest.main()
