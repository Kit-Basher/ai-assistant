from __future__ import annotations

import tempfile
import unittest

from agent.bootstrap.snapshot import BootstrapSnapshot
from agent.memory_v2.ingest import ingest_bootstrap_snapshot
from agent.memory_v2.storage import SQLiteMemoryStore
from agent.memory_v2.types import MemoryLevel


def _snapshot(created_at: int, enabled_ids: list[str]) -> BootstrapSnapshot:
    return BootstrapSnapshot(
        created_at_ts=created_at,
        os={
            "name": "Ubuntu",
            "version": "24.04",
            "pretty_name": "Ubuntu 24.04 LTS",
            "kernel": "6.8.0",
            "arch": "x86_64",
            "hostname": "host-a",
            "os_release": {"NAME": "Ubuntu"},
        },
        hardware={
            "cpu_count_logical": 8,
            "cpu_freq_mhz": 2300.0,
            "cpu_load_1m": 0.21,
            "mem_total_bytes": 16_000,
            "swap_total_bytes": 1024,
            "gpu": {"available": True, "memory_total_mb": 6144, "usage_pct": 0.0, "error": None},
        },
        interfaces={
            "api": {"listening": "http://127.0.0.1:8765"},
            "memory_v2_enabled": True,
            "model_watch_enabled": True,
            "llm_automation_enabled": True,
            "telegram_configured": False,
            "webui_dev_proxy": False,
        },
        providers={
            "enabled_ids": enabled_ids,
            "rows": [
                {
                    "id": provider_id,
                    "enabled": True,
                    "local": provider_id == "ollama",
                    "health": {"status": "ok", "last_error_kind": None, "status_code": None},
                }
                for provider_id in enabled_ids
            ],
            "defaults": {
                "default_provider": "ollama",
                "default_model": "ollama:llama3",
                "routing_mode": "auto",
            },
        },
        capsules={"installed": ["llm", "ops"]},
        routes={
            "methods": {"GET": ["/health"], "POST": ["/chat"], "PUT": [], "DELETE": []},
            "counts": {"GET": 1, "POST": 1, "PUT": 0, "DELETE": 0},
            "total": 2,
        },
        notes=[],
    )


class TestSemanticVersioning(unittest.TestCase):
    def test_bootstrap_semantic_versions_only_change_on_value_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteMemoryStore(f"{tmpdir}/memory.db")

            result_a = ingest_bootstrap_snapshot(
                store=store,
                snapshot=_snapshot(1_700_500_001, ["ollama"]),
                source_ref="bootstrap_a",
            )
            current_after_a = store.list_memory_items(level=MemoryLevel.SEMANTIC)
            self.assertEqual(5, len(current_after_a))

            result_b = ingest_bootstrap_snapshot(
                store=store,
                snapshot=_snapshot(1_700_500_100, ["ollama", "openrouter"]),
                source_ref="bootstrap_b",
            )
            updates_b = result_b.get("semantic_updates") if isinstance(result_b.get("semantic_updates"), dict) else {}
            inserted_b = updates_b.get("inserted") if isinstance(updates_b.get("inserted"), list) else []
            unchanged_b = updates_b.get("unchanged") if isinstance(updates_b.get("unchanged"), list) else []
            superseded_b = updates_b.get("superseded") if isinstance(updates_b.get("superseded"), list) else []
            self.assertEqual(1, len(inserted_b))
            self.assertEqual("providers.enabled_ids", inserted_b[0].get("fact_key"))
            self.assertEqual(4, len(unchanged_b))
            self.assertEqual(1, len(superseded_b))
            self.assertEqual("providers.enabled_ids", superseded_b[0].get("fact_key"))

            current_after_b = store.list_memory_items(level=MemoryLevel.SEMANTIC)
            self.assertEqual(5, len(current_after_b))
            history_after_b = store.list_memory_items(level=MemoryLevel.SEMANTIC, include_history=True)
            self.assertEqual(6, len(history_after_b))

            provider_rows = [item for item in history_after_b if item.fact_key == "providers.enabled_ids"]
            self.assertEqual(2, len(provider_rows))
            current_provider = store.get_current_semantic(fact_group="bootstrap", fact_key="providers.enabled_ids")
            self.assertIsNotNone(current_provider)
            assert current_provider is not None
            self.assertTrue(current_provider.is_current)
            self.assertIn("openrouter", current_provider.text)

            superseded_provider_rows = [row for row in provider_rows if not row.is_current]
            self.assertEqual(1, len(superseded_provider_rows))
            self.assertEqual(1_700_500_100, superseded_provider_rows[0].superseded_at)
            self.assertNotEqual(superseded_provider_rows[0].source_ref, current_provider.source_ref)

            self.assertIn(result_a["section_to_episodic_id"]["providers"], superseded_provider_rows[0].source_ref)
            self.assertIn(result_b["section_to_episodic_id"]["providers"], current_provider.source_ref)

            result_c = ingest_bootstrap_snapshot(
                store=store,
                snapshot=_snapshot(1_700_500_200, ["ollama", "openrouter"]),
                source_ref="bootstrap_c",
            )
            updates_c = result_c.get("semantic_updates") if isinstance(result_c.get("semantic_updates"), dict) else {}
            inserted_c = updates_c.get("inserted") if isinstance(updates_c.get("inserted"), list) else []
            unchanged_c = updates_c.get("unchanged") if isinstance(updates_c.get("unchanged"), list) else []
            superseded_c = updates_c.get("superseded") if isinstance(updates_c.get("superseded"), list) else []
            self.assertEqual([], inserted_c)
            self.assertEqual(5, len(unchanged_c))
            self.assertEqual([], superseded_c)
            history_after_c = store.list_memory_items(level=MemoryLevel.SEMANTIC, include_history=True)
            self.assertEqual(6, len(history_after_c))


if __name__ == "__main__":
    unittest.main()
