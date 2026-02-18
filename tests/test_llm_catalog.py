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
                            "prompt": "0.1500000004",
                            "completion": "0.6000000009",
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
        self.assertEqual(0.15, row["input_cost_per_million_tokens"])
        self.assertEqual(0.6, row["output_cost_per_million_tokens"])

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
