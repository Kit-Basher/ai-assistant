from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest

from agent.llm.registry_txn import RegistrySnapshotStore, apply_with_rollback


def _registry_doc() -> dict[str, object]:
    return {
        "schema_version": 2,
        "providers": {
            "ollama": {
                "provider_type": "openai_compat",
                "base_url": "http://127.0.0.1:11434",
                "chat_path": "/v1/chat/completions",
                "api_key_source": None,
                "default_headers": {},
                "default_query_params": {},
                "enabled": True,
                "local": True,
            },
            "openrouter": {
                "provider_type": "openai_compat",
                "base_url": "https://openrouter.ai/api/v1",
                "chat_path": "/chat/completions",
                "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
                "default_headers": {},
                "default_query_params": {},
                "enabled": True,
                "local": False,
            },
        },
        "models": {
            "ollama:qwen2.5:3b-instruct": {
                "provider": "ollama",
                "model": "qwen2.5:3b-instruct",
                "capabilities": ["chat"],
                "enabled": True,
                "available": True,
                "quality_rank": 2,
                "cost_rank": 0,
                "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                "max_context_tokens": 8192,
            }
        },
        "defaults": {
            "routing_mode": "prefer_local_lowest_cost_capable",
            "default_provider": "ollama",
            "default_model": "ollama:qwen2.5:3b-instruct",
            "allow_remote_fallback": False,
        },
    }


class TestRegistryTxn(unittest.TestCase):
    def test_apply_with_rollback_restores_on_verify_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = os.path.join(tmpdir, "registry.json")
            snapshots_dir = os.path.join(tmpdir, "snapshots")
            with open(registry_path, "w", encoding="utf-8") as handle:
                json.dump(_registry_doc(), handle)
            store = RegistrySnapshotStore(path=snapshots_dir, max_items=10)

            ok_result = apply_with_rollback(
                registry_path=registry_path,
                snapshot_store=store,
                plan_apply_fn=lambda current: {
                    **current,
                    "defaults": {
                        **(current.get("defaults") or {}),
                        "allow_remote_fallback": True,
                    },
                },
            )
            self.assertTrue(ok_result["ok"])
            with open(registry_path, "r", encoding="utf-8") as handle:
                after_ok = json.load(handle)
            self.assertTrue(bool((after_ok.get("defaults") or {}).get("allow_remote_fallback")))

            before_bad = copy.deepcopy(after_ok)
            bad_result = apply_with_rollback(
                registry_path=registry_path,
                snapshot_store=store,
                plan_apply_fn=lambda current: {
                    **current,
                    "defaults": {
                        **(current.get("defaults") or {}),
                        "default_provider": "unknown-provider",
                    },
                },
            )
            self.assertFalse(bad_result["ok"])
            self.assertEqual("verify_failed", bad_result["error_kind"])
            with open(registry_path, "r", encoding="utf-8") as handle:
                after_bad = json.load(handle)
            self.assertEqual(before_bad, after_bad)

    def test_snapshot_retention_prunes_oldest_first_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = os.path.join(tmpdir, "registry.json")
            snapshots_dir = os.path.join(tmpdir, "snapshots")
            base = _registry_doc()
            with open(registry_path, "w", encoding="utf-8") as handle:
                json.dump(base, handle)
            store = RegistrySnapshotStore(path=snapshots_dir, max_items=2)

            for idx in range(4):
                payload = copy.deepcopy(base)
                payload["defaults"] = {
                    **(payload.get("defaults") or {}),
                    "allow_remote_fallback": bool(idx % 2),
                    "routing_mode": f"mode-{idx}",
                }
                store.create_snapshot(registry_path, payload)

            rows = store.list_snapshots(limit=10)
            self.assertEqual(2, len(rows))
            # Newest first and deterministic sequence IDs.
            self.assertTrue(rows[0]["snapshot_id"].startswith("s00000004-"))
            self.assertTrue(rows[1]["snapshot_id"].startswith("s00000003-"))


if __name__ == "__main__":
    unittest.main()
