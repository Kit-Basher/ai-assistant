from __future__ import annotations

import tempfile
import unittest

from agent.llm.catalog import CatalogStore, fetch_provider_catalog, normalize_catalog_entry


class TestLLMCatalog(unittest.TestCase):
    def test_normalize_catalog_entry_is_deterministic(self) -> None:
        row = normalize_catalog_entry(
            "OpenRouter",
            "openai/gpt-4o-mini",
            capabilities=["tools", "chat", "chat"],
            max_context_tokens="128000",
            input_cost_per_million_tokens=0.1500000004,
            output_cost_per_million_tokens="0.6000000009",
            source="openrouter_models",
        )
        self.assertEqual("openrouter:openai/gpt-4o-mini", row["id"])
        self.assertEqual(["chat", "tools"], row["capabilities"])
        self.assertEqual(128000, row["max_context_tokens"])
        self.assertEqual(0.15, row["input_cost_per_million_tokens"])
        self.assertEqual(0.6, row["output_cost_per_million_tokens"])

    def test_fetch_openrouter_catalog_parses_pricing_and_context(self) -> None:
        def _fake_http(
            url: str,
            *,
            headers: dict[str, str],
            timeout_seconds: float,
            allowed_hosts: set[str],
        ) -> dict[str, object]:
            _ = timeout_seconds
            self.assertTrue(url.endswith("/models"))
            self.assertIn("openrouter.ai", allowed_hosts)
            self.assertTrue(str(headers.get("Authorization") or "").startswith("Bearer "))
            return {
                "data": [
                    {
                        "id": "openai/gpt-4o-mini",
                        "context_length": 128000,
                        "pricing": {
                            "prompt": "0.0000001504",
                            "completion": "0.0000006009",
                        },
                    }
                ]
            }

        result = fetch_provider_catalog(
            "openrouter",
            {
                "base_url": "https://openrouter.ai/api/v1",
                "local": False,
                "resolved_headers": {"Authorization": "Bearer sk-test"},
            },
            _fake_http,
        )
        self.assertTrue(result["ok"])
        self.assertEqual("openrouter_models", result["source"])
        self.assertEqual(1, len(result["models"]))
        row = result["models"][0]
        self.assertEqual("openrouter:openai/gpt-4o-mini", row["id"])
        self.assertEqual(128000, row["max_context_tokens"])
        self.assertEqual(0.1504, row["input_cost_per_million_tokens"])
        self.assertEqual(0.6009, row["output_cost_per_million_tokens"])

    def test_fetch_openrouter_catalog_applies_known_task_types_for_exact_models(self) -> None:
        def _fake_http(
            url: str,
            *,
            headers: dict[str, str],
            timeout_seconds: float,
            allowed_hosts: set[str],
        ) -> dict[str, object]:
            _ = timeout_seconds
            self.assertTrue(url.endswith("/models"))
            self.assertIn("openrouter.ai", allowed_hosts)
            self.assertTrue(str(headers.get("Authorization") or "").startswith("Bearer "))
            return {
                "data": [
                    {
                        "id": "openai/gpt-4.1",
                        "context_length": 1047576,
                        "pricing": {
                            "prompt": "0.000002",
                            "completion": "0.000008",
                        },
                    }
                ]
            }

        result = fetch_provider_catalog(
            "openrouter",
            {
                "base_url": "https://openrouter.ai/api/v1",
                "local": False,
                "resolved_headers": {"Authorization": "Bearer sk-test"},
            },
            _fake_http,
        )
        self.assertTrue(result["ok"])
        row = result["models"][0]
        self.assertEqual(
            ["coding", "general_chat", "reasoning"],
            row["task_types"],
        )
        self.assertEqual(1047576, row["max_context_tokens"])

    def test_fetch_openrouter_catalog_applies_curated_exact_metadata_for_high_value_models(self) -> None:
        def _fake_http(
            url: str,
            *,
            headers: dict[str, str],
            timeout_seconds: float,
            allowed_hosts: set[str],
        ) -> dict[str, object]:
            _ = timeout_seconds
            self.assertTrue(url.endswith("/models"))
            self.assertIn("openrouter.ai", allowed_hosts)
            self.assertTrue(str(headers.get("Authorization") or "").startswith("Bearer "))
            return {
                "data": [
                    {
                        "id": "openai/gpt-4o",
                        "context_length": 128000,
                        "pricing": {
                            "prompt": "0.0000025",
                            "completion": "0.0000100",
                        },
                    },
                    {
                        "id": "anthropic/claude-3.5-sonnet",
                        "context_length": 200000,
                        "pricing": {
                            "prompt": "0.0000010",
                            "completion": "0.0000040",
                        },
                    },
                    {
                        "id": "google/gemini-pro-1.5",
                        "context_length": 2097152,
                        "pricing": {
                            "prompt": "0.0000010",
                            "completion": "0.0000040",
                        },
                    },
                ]
            }

        result = fetch_provider_catalog(
            "openrouter",
            {
                "base_url": "https://openrouter.ai/api/v1",
                "local": False,
                "resolved_headers": {"Authorization": "Bearer sk-test"},
            },
            _fake_http,
        )
        self.assertTrue(result["ok"])
        rows = {
            str(row.get("id") or ""): row
            for row in result["models"]
            if isinstance(row, dict)
        }
        self.assertEqual(
            ["coding", "general_chat", "reasoning"],
            rows["openrouter:openai/gpt-4o"]["task_types"],
        )
        self.assertEqual(128000, rows["openrouter:openai/gpt-4o"]["max_context_tokens"])
        self.assertEqual(
            ["coding", "general_chat", "reasoning"],
            rows["openrouter:anthropic/claude-3.5-sonnet"]["task_types"],
        )
        self.assertEqual(200000, rows["openrouter:anthropic/claude-3.5-sonnet"]["max_context_tokens"])
        self.assertEqual(
            ["general_chat", "reasoning"],
            rows["openrouter:google/gemini-pro-1.5"]["task_types"],
        )
        self.assertEqual(2097152, rows["openrouter:google/gemini-pro-1.5"]["max_context_tokens"])

    def test_fetch_openrouter_catalog_keeps_explicit_per_million_pricing(self) -> None:
        def _fake_http(
            url: str,
            *,
            headers: dict[str, str],
            timeout_seconds: float,
            allowed_hosts: set[str],
        ) -> dict[str, object]:
            _ = timeout_seconds
            self.assertTrue(url.endswith("/models"))
            self.assertIn("openrouter.ai", allowed_hosts)
            self.assertTrue(str(headers.get("Authorization") or "").startswith("Bearer "))
            return {
                "data": [
                    {
                        "id": "anthropic/claude-opus-4",
                        "context_length": 200000,
                        "pricing": {
                            "input_per_million_tokens": 15.0,
                            "output_per_million_tokens": 75.0,
                        },
                    }
                ]
            }

        result = fetch_provider_catalog(
            "openrouter",
            {
                "base_url": "https://openrouter.ai/api/v1",
                "local": False,
                "resolved_headers": {"Authorization": "Bearer sk-test"},
            },
            _fake_http,
        )
        self.assertTrue(result["ok"])
        row = result["models"][0]
        self.assertEqual(15.0, row["input_cost_per_million_tokens"])
        self.assertEqual(75.0, row["output_cost_per_million_tokens"])

    def test_fetch_openrouter_catalog_preserves_structured_modality_fields(self) -> None:
        def _fake_http(
            url: str,
            *,
            headers: dict[str, str],
            timeout_seconds: float,
            allowed_hosts: set[str],
        ) -> dict[str, object]:
            _ = timeout_seconds
            self.assertTrue(url.endswith("/models"))
            self.assertIn("openrouter.ai", allowed_hosts)
            self.assertTrue(str(headers.get("Authorization") or "").startswith("Bearer "))
            return {
                "data": [
                    {
                        "id": "vendor/vision-pro",
                        "context_length": 65536,
                        "architecture": {
                            "modality": "text+image->text",
                            "input_modalities": ["text", "image"],
                            "output_modalities": ["text"],
                        },
                        "pricing": {
                            "prompt": "0.0000010",
                            "completion": "0.0000040",
                        },
                    }
                ]
            }

        result = fetch_provider_catalog(
            "openrouter",
            {
                "base_url": "https://openrouter.ai/api/v1",
                "local": False,
                "resolved_headers": {"Authorization": "Bearer sk-test"},
            },
            _fake_http,
        )
        self.assertTrue(result["ok"])
        row = result["models"][0]
        self.assertEqual("text+image->text", row["architecture_modality"])
        self.assertEqual(["image", "text"], row["input_modalities"])
        self.assertEqual(["text"], row["output_modalities"])
        self.assertEqual(["chat", "image", "vision"], row["capabilities"])
        self.assertEqual([], row["task_types"])
        self.assertEqual(65536, row["max_context_tokens"])
        self.assertEqual(1.0, row["input_cost_per_million_tokens"])
        self.assertEqual(4.0, row["output_cost_per_million_tokens"])

    def test_fetch_openrouter_catalog_leaves_unknown_models_untyped_when_source_lacks_metadata(self) -> None:
        def _fake_http(
            url: str,
            *,
            headers: dict[str, str],
            timeout_seconds: float,
            allowed_hosts: set[str],
        ) -> dict[str, object]:
            _ = timeout_seconds
            self.assertTrue(url.endswith("/models"))
            self.assertIn("openrouter.ai", allowed_hosts)
            self.assertTrue(str(headers.get("Authorization") or "").startswith("Bearer "))
            return {
                "data": [
                    {
                        "id": "vendor/plain-chat",
                        "pricing": {
                            "prompt": "0.0000005000",
                            "completion": "0.0000015000",
                        },
                    }
                ]
            }

        result = fetch_provider_catalog(
            "openrouter",
            {
                "base_url": "https://openrouter.ai/api/v1",
                "local": False,
                "resolved_headers": {"Authorization": "Bearer sk-test"},
            },
            _fake_http,
        )
        self.assertTrue(result["ok"])
        row = result["models"][0]
        self.assertEqual(["chat"], row["capabilities"])
        self.assertEqual([], row["task_types"])
        self.assertIsNone(row["architecture_modality"])
        self.assertEqual([], row["input_modalities"])
        self.assertEqual([], row["output_modalities"])

    def test_catalog_store_keeps_last_good_snapshot_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/llm_catalog.json"
            store = CatalogStore(path=path)
            ok_result = {
                "ok": True,
                "provider_id": "openrouter",
                "source": "openrouter_models",
                "models": [
                    normalize_catalog_entry(
                        "openrouter",
                        "openai/gpt-4o-mini",
                        capabilities=["chat"],
                        max_context_tokens=128000,
                        input_cost_per_million_tokens=0.15,
                        output_cost_per_million_tokens=0.6,
                        source="openrouter_models",
                    )
                ],
                "error_kind": None,
            }
            store.update_provider_result("openrouter", ok_result, now_epoch=100)
            store.update_provider_result(
                "openrouter",
                {
                    "ok": False,
                    "provider_id": "openrouter",
                    "source": "openrouter_models",
                    "models": [],
                    "error_kind": "http_500",
                },
                now_epoch=200,
            )
            status = store.status()
            provider_status = status["providers"][0]
            self.assertEqual("http_500", provider_status["last_error_kind"])
            self.assertEqual(100, provider_status["last_refresh_at"])
            rows = store.provider_models("openrouter")
            self.assertEqual(1, len(rows))
            self.assertEqual("openrouter:openai/gpt-4o-mini", rows[0]["id"])

            reloaded = CatalogStore(path=path)
            self.assertEqual(rows, reloaded.provider_models("openrouter"))


if __name__ == "__main__":
    unittest.main()
