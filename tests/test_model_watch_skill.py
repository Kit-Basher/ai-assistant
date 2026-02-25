from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from agent.model_watch_catalog import build_openrouter_snapshot, write_snapshot_atomic
from agent.model_watch_skill import run_model_watch_check


def _write_registry(path: Path) -> None:
    document = {
        "schema_version": 2,
        "providers": {
            "ollama": {
                "provider_type": "openai_compat",
                "base_url": "http://127.0.0.1:11434",
                "chat_path": "/v1/chat/completions",
                "enabled": True,
                "local": True,
            },
            "openrouter": {
                "provider_type": "openai_compat",
                "base_url": "https://openrouter.ai/api/v1",
                "chat_path": "/chat/completions",
                "enabled": True,
                "local": False,
                "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
            },
        },
        "models": {},
        "defaults": {
            "routing_mode": "auto",
            "default_provider": "ollama",
            "default_model": None,
            "allow_remote_fallback": True,
            "fallback_chain": [],
        },
    }
    path.write_text(json.dumps(document, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _config(tmpdir: str) -> SimpleNamespace:
    return SimpleNamespace(
        llm_registry_path=str(Path(tmpdir) / "registry.json"),
        model_watch_state_path=str(Path(tmpdir) / "model_watch_state.json"),
        model_watch_config_path=str(Path(tmpdir) / "model_watch_config.json"),
        model_watch_interval_seconds=86400,
        openrouter_api_key="sk-test",
    )


def _fetch_stub(url: str) -> object:
    if url.endswith("/api/tags"):
        return {"models": []}
    if "huggingface.co/api/models" in url:
        return []
    raise RuntimeError(f"unexpected_url:{url}")


class TestModelWatchSkill(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.config = _config(self.tmpdir.name)
        _write_registry(Path(self.config.llm_registry_path))
        Path(self.config.model_watch_config_path).write_text(
            json.dumps(
                {
                    "huggingface_watch_authors": [],
                    "max_size_gb": 8,
                    "require_license": ["apache-2.0", "mit"],
                },
                ensure_ascii=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_run_uses_snapshot_pool_count_before_filtering(self) -> None:
        snapshot_path = Path(self.tmpdir.name) / "catalog.json"
        os.environ["AGENT_MODEL_WATCH_CATALOG_PATH"] = str(snapshot_path)
        try:
            snapshot = build_openrouter_snapshot(
                raw_payload={
                    "data": [
                        {"id": "acme/model-a", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
                        {"id": "acme/model-b"},
                        {"id": "acme/model-c"},
                        {"id": "acme/model-d"},
                        {"id": "acme/model-e"},
                    ]
                },
                fetched_at=1700000000,
            )
            write_snapshot_atomic(snapshot_path, snapshot)

            result = run_model_watch_check(self.config, fetch_json=_fetch_stub, now_epoch=1700000100)
        finally:
            os.environ.pop("AGENT_MODEL_WATCH_CATALOG_PATH", None)

        self.assertTrue(result["ok"])
        self.assertEqual(5, int(result["catalog_snapshot_model_count"]))
        self.assertEqual(5, int(result["catalog_models_considered"]))
        self.assertEqual(5, int(result["fetched_candidates"]))

    def test_batch_id_changes_when_snapshot_hash_changes(self) -> None:
        snapshot_path = Path(self.tmpdir.name) / "catalog.json"
        os.environ["AGENT_MODEL_WATCH_CATALOG_PATH"] = str(snapshot_path)
        try:
            snapshot_a = build_openrouter_snapshot(
                raw_payload={
                    "data": [
                        {
                            "id": "acme/model-a",
                            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                        }
                    ]
                },
                fetched_at=1700000200,
            )
            write_snapshot_atomic(snapshot_path, snapshot_a)
            result_a = run_model_watch_check(self.config, fetch_json=_fetch_stub, now_epoch=1700000300)
            batch_a = str(result_a.get("batch_id") or "")
            self.assertTrue(batch_a)

            snapshot_b = build_openrouter_snapshot(
                raw_payload={
                    "data": [
                        {
                            "id": "acme/model-a",
                            "pricing": {"prompt": "0.000002", "completion": "0.000004"},
                        }
                    ]
                },
                fetched_at=1700000400,
            )
            write_snapshot_atomic(snapshot_path, snapshot_b)
            result_b = run_model_watch_check(self.config, fetch_json=_fetch_stub, now_epoch=1700000500)
            batch_b = str(result_b.get("batch_id") or "")
            self.assertTrue(batch_b)
        finally:
            os.environ.pop("AGENT_MODEL_WATCH_CATALOG_PATH", None)

        self.assertNotEqual(batch_a, batch_b)
        self.assertTrue(bool(result_b.get("new_batch_created")))

    def test_latest_batch_created_after_snapshot_fetch(self) -> None:
        snapshot_path = Path(self.tmpdir.name) / "catalog.json"
        os.environ["AGENT_MODEL_WATCH_CATALOG_PATH"] = str(snapshot_path)
        try:
            snapshot = build_openrouter_snapshot(
                raw_payload={"data": [{"id": "acme/model-a"}]},
                fetched_at=1700000600,
            )
            write_snapshot_atomic(snapshot_path, snapshot)
            result = run_model_watch_check(self.config, fetch_json=_fetch_stub, now_epoch=1700000700)
        finally:
            os.environ.pop("AGENT_MODEL_WATCH_CATALOG_PATH", None)

        latest = result.get("latest_batch") if isinstance(result.get("latest_batch"), dict) else {}
        self.assertTrue(bool(result.get("new_batch_created")))
        self.assertGreaterEqual(int(latest.get("created_at") or 0), 1700000600)
        self.assertEqual(str(result.get("batch_id") or ""), str(latest.get("batch_id") or ""))

    def test_missing_snapshot_returns_clear_reason(self) -> None:
        snapshot_path = Path(self.tmpdir.name) / "catalog-missing.json"
        os.environ["AGENT_MODEL_WATCH_CATALOG_PATH"] = str(snapshot_path)
        try:
            result = run_model_watch_check(self.config, fetch_json=_fetch_stub, now_epoch=1700000800)
        finally:
            os.environ.pop("AGENT_MODEL_WATCH_CATALOG_PATH", None)
        self.assertTrue(result["ok"])
        self.assertFalse(result["found"])
        self.assertIn("run refresh", str(result.get("reason") or "").lower())

    def test_missing_feature_tradeoffs_are_exposed(self) -> None:
        snapshot_path = Path(self.tmpdir.name) / "catalog.json"
        os.environ["AGENT_MODEL_WATCH_CATALOG_PATH"] = str(snapshot_path)
        try:
            snapshot = build_openrouter_snapshot(
                raw_payload={"data": [{"id": "acme/no-meta-model"}]},
                fetched_at=1700000900,
            )
            write_snapshot_atomic(snapshot_path, snapshot)
            result = run_model_watch_check(self.config, fetch_json=_fetch_stub, now_epoch=1700001000)
        finally:
            os.environ.pop("AGENT_MODEL_WATCH_CATALOG_PATH", None)

        latest = result.get("latest_batch") if isinstance(result.get("latest_batch"), dict) else {}
        top = latest.get("top_pick") if isinstance(latest.get("top_pick"), dict) else {}
        self.assertIsNotNone(top.get("score"))
        self.assertTrue(bool(top.get("reason")))
        tradeoffs = [str(item).strip() for item in (top.get("tradeoffs") or []) if str(item).strip()]
        self.assertIn("missing:context_length", tradeoffs)
        self.assertIn("missing:pricing", tradeoffs)
        self.assertIn("missing:params_b", tradeoffs)
        subscores = top.get("subscores") if isinstance(top.get("subscores"), dict) else {}
        self.assertEqual(
            {"task_fit", "local_feasibility", "cost_efficiency", "quality_proxy", "switch_gain"},
            set(subscores.keys()),
        )


if __name__ == "__main__":
    unittest.main()
