from __future__ import annotations

import json
import tempfile
import unittest

from agent.bootstrap.snapshot import BootstrapSnapshot
from agent.memory_v2.ingest import ingest_bootstrap_snapshot
from agent.memory_v2.storage import SQLiteMemoryStore
from agent.memory_v2.types import MemoryLevel


class TestBootstrapIngest(unittest.TestCase):
    def test_ingest_creates_episodic_and_allowlist_semantic_with_provenance(self) -> None:
        snapshot = BootstrapSnapshot(
            created_at_ts=1_700_111_000,
            os={
                "name": "Ubuntu 24.04",
                "version": "24.04",
                "pretty_name": "Ubuntu 24.04 LTS",
                "kernel": "6.8.0",
                "arch": "x86_64",
                "hostname": "host-a",
                "os_release": {"NAME": "Ubuntu 24.04"},
            },
            hardware={
                "cpu_count_logical": 8,
                "cpu_freq_mhz": 2400.0,
                "cpu_load_1m": 0.23,
                "mem_total_bytes": 17179869184,
                "swap_total_bytes": 2147483648,
                "gpu": {
                    "available": True,
                    "memory_total_mb": 6144,
                    "usage_pct": 0.0,
                    "error": None,
                },
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
                "enabled_ids": ["ollama", "openrouter"],
                "rows": [
                    {"id": "ollama", "enabled": True, "local": True, "health": {"status": "ok", "last_error_kind": None, "status_code": None}},
                    {"id": "openrouter", "enabled": True, "local": False, "health": {"status": "ok", "last_error_kind": None, "status_code": None}},
                ],
                "defaults": {"default_provider": "ollama", "default_model": "ollama:llama3", "routing_mode": "auto"},
            },
            capsules={"installed": ["llm", "ops"]},
            routes={
                "methods": {"GET": ["/health"], "POST": ["/chat"], "PUT": [], "DELETE": []},
                "counts": {"GET": 1, "POST": 1, "PUT": 0, "DELETE": 0},
                "total": 2,
            },
            notes=["telegram_not_configured"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteMemoryStore(f"{tmpdir}/memory.db")
            result = ingest_bootstrap_snapshot(
                store=store,
                snapshot=snapshot,
                source_ref="test_bootstrap",
            )

            self.assertEqual(7, len(result["episodic_ids"]))
            sections = result.get("section_to_episodic_id") if isinstance(result.get("section_to_episodic_id"), dict) else {}
            self.assertEqual({"capsules", "hardware", "interfaces", "notes", "os", "providers", "routes"}, set(sections.keys()))

            episodic_items = store.list_episodic_events(limit=50)
            by_id = {item.id: item for item in episodic_items}
            for section, event_id in sections.items():
                self.assertIn(event_id, by_id)
                self.assertEqual("bootstrap", by_id[event_id].source_kind)
                self.assertEqual("test_bootstrap", by_id[event_id].source_ref)
                self.assertEqual("bootstrap_snapshot", by_id[event_id].tags.get("kind"))
                self.assertEqual(section, by_id[event_id].tags.get("section"))

            expected_os_text = json.dumps(snapshot.os, ensure_ascii=True, sort_keys=True, indent=2)
            os_event = by_id[sections["os"]]
            self.assertEqual(expected_os_text, os_event.text)

            semantic_items = store.list_memory_items(level=MemoryLevel.SEMANTIC)
            semantic_ids = sorted(item.id for item in semantic_items)
            self.assertEqual(
                [
                    "S-bootstrap-capsules-installed",
                    "S-bootstrap-hardware-gpu-available",
                    "S-bootstrap-interfaces-available",
                    "S-bootstrap-os-name",
                    "S-bootstrap-providers-enabled-ids",
                ],
                semantic_ids,
            )

            semantic_map = {item.id: item for item in semantic_items}
            self.assertIn('"name": "Ubuntu 24.04"', semantic_map["S-bootstrap-os-name"].text)
            self.assertIn('"available": true', semantic_map["S-bootstrap-hardware-gpu-available"].text)

            for item in semantic_items:
                self.assertEqual("bootstrap", item.source_kind)
                self.assertTrue(bool(item.source_ref.strip()))
                event_refs = [part for part in item.source_ref.split(",") if part]
                self.assertTrue(event_refs)
                for event_id in event_refs:
                    self.assertIn(event_id, by_id)


if __name__ == "__main__":
    unittest.main()
