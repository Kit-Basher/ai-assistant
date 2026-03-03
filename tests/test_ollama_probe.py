from __future__ import annotations

import json
import os
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from agent.api_server import AgentRuntime
from agent.config import Config
from agent.llm.ollama_endpoints import normalize_ollama_base_urls
from agent.llm.probes import probe_provider


def _config(registry_path: str, db_path: str, *, ollama_base_url: str) -> Config:
    base = Config(
        telegram_bot_token="token",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        openai_model_worker=None,
        agent_timezone="UTC",
        db_path=db_path,
        log_path=os.path.join(os.path.dirname(db_path), "agent.log"),
        skills_path="/tmp/skills",
        ollama_host=ollama_base_url,
        ollama_model="llama3",
        ollama_model_sentinel=None,
        ollama_model_worker=None,
        allow_cloud=True,
        prefer_local=True,
        llm_timeout_seconds=15,
        llm_provider="none",
        enable_llm_presentation=False,
        openai_base_url=None,
        ollama_base_url=ollama_base_url,
        anthropic_api_key=None,
        llm_selector="single",
        llm_broker_policy_path=None,
        llm_allow_remote=True,
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_model="openai/gpt-4o-mini",
        openrouter_site_url=None,
        openrouter_app_name=None,
        llm_registry_path=registry_path,
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
    )
    return base.__class__(**{**base.__dict__})


class TestOllamaEndpointNormalization(unittest.TestCase):
    def test_normalize_without_v1(self) -> None:
        payload = normalize_ollama_base_urls("http://127.0.0.1:11434")
        self.assertEqual("http://127.0.0.1:11434", payload["configured_base_url"])
        self.assertEqual("http://127.0.0.1:11434", payload["native_base"])
        self.assertEqual("http://127.0.0.1:11434/v1", payload["openai_base"])

    def test_normalize_with_v1_suffix(self) -> None:
        payload = normalize_ollama_base_urls("http://127.0.0.1:11434/v1")
        self.assertEqual("http://127.0.0.1:11434/v1", payload["configured_base_url"])
        self.assertEqual("http://127.0.0.1:11434", payload["native_base"])
        self.assertEqual("http://127.0.0.1:11434/v1", payload["openai_base"])

    def test_normalize_with_extra_slashes(self) -> None:
        payload = normalize_ollama_base_urls("http://127.0.0.1:11434//v1//")
        self.assertEqual("http://127.0.0.1:11434", payload["native_base"])
        self.assertEqual("http://127.0.0.1:11434/v1", payload["openai_base"])


class TestOllamaProviderProbe(unittest.TestCase):
    def _cfg(self, base_url: str) -> dict[str, object]:
        return {
            "id": "ollama",
            "provider_type": "openai_compat",
            "base_url": base_url,
            "chat_path": "/v1/chat/completions",
            "enabled": True,
            "local": True,
            "allow_remote_fallback": True,
            "api_key_source": None,
            "headers": {},
            "_resolved_api_key_present": False,
            "available": True,
        }

    def test_native_ok_openai_404_is_still_up(self) -> None:
        def _getter(url: str, *, timeout_seconds: float, headers: dict[str, str]) -> dict[str, object]:
            _ = (timeout_seconds, headers)
            if url.endswith("/api/tags"):
                return {"models": []}
            if url.endswith("/v1/models"):
                raise urllib.error.HTTPError(url, 404, "not found", None, None)
            raise AssertionError(url)

        result = probe_provider(self._cfg("http://127.0.0.1:11434"), http_get_json=_getter)
        self.assertEqual("ok", result["status"])
        self.assertTrue(bool(result.get("native_ok")))
        self.assertFalse(bool(result.get("openai_compat_ok")))

    def test_native_connection_refused_is_down(self) -> None:
        def _getter(url: str, *, timeout_seconds: float, headers: dict[str, str]) -> dict[str, object]:
            _ = (timeout_seconds, headers, url)
            raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))

        result = probe_provider(self._cfg("http://127.0.0.1:11434"), http_get_json=_getter)
        self.assertEqual("down", result["status"])
        self.assertEqual("connection_refused", result["error_kind"])

    def test_native_500_is_down_bad_status_code(self) -> None:
        def _getter(url: str, *, timeout_seconds: float, headers: dict[str, str]) -> dict[str, object]:
            _ = (timeout_seconds, headers)
            raise urllib.error.HTTPError(url, 500, "server error", None, None)

        result = probe_provider(self._cfg("http://127.0.0.1:11434"), http_get_json=_getter)
        self.assertEqual("down", result["status"])
        self.assertEqual("bad_status_code", result["error_kind"])
        self.assertEqual(500, result["status_code"])


class TestOllamaHealthRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def _runtime(self, base_url: str) -> AgentRuntime:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, ollama_base_url=base_url))
        runtime._health_monitor._candidate_models = lambda _doc: [("ollama", "ollama:llama3")]  # type: ignore[attr-defined]
        return runtime

    def test_health_marks_ollama_up_when_native_ok_openai_404(self) -> None:
        runtime = self._runtime("http://127.0.0.1:11434")

        def _get(url: str, timeout_seconds: float = 0.0, headers: dict[str, str] | None = None) -> dict[str, object]:
            _ = (timeout_seconds, headers)
            if url.endswith("/api/tags"):
                return {"models": [{"name": "llama3"}]}
            if url.endswith("/v1/models"):
                raise urllib.error.HTTPError(url, 404, "not found", None, None)
            raise AssertionError(url)

        with patch.object(runtime, "_http_get_json", side_effect=_get), patch.object(runtime, "_http_post_json") as post_mock:
            ok_run, _payload = runtime.run_llm_health(trigger="manual")
            status = runtime.model_status()

        self.assertTrue(ok_run)
        providers = status["llm_availability"]["providers"]
        self.assertIn("ollama", providers["local_up"])
        self.assertNotIn("ollama", providers["local_down"])
        ollama = status["llm_availability"]["ollama"]
        self.assertTrue(bool(ollama["native_ok"]))
        self.assertFalse(bool(ollama["openai_compat_ok"]))
        post_mock.assert_not_called()

    def test_health_marks_ollama_up_with_v1_configured_base(self) -> None:
        runtime = self._runtime("http://127.0.0.1:11434/v1")

        def _get(url: str, timeout_seconds: float = 0.0, headers: dict[str, str] | None = None) -> dict[str, object]:
            _ = (timeout_seconds, headers)
            if url.endswith("/api/tags"):
                return {"models": [{"name": "llama3"}]}
            if url.endswith("/v1/models"):
                raise urllib.error.HTTPError(url, 404, "not found", None, None)
            raise AssertionError(url)

        with patch.object(runtime, "_http_get_json", side_effect=_get):
            ok_run, _payload = runtime.run_llm_health(trigger="manual")
            status = runtime.model_status()

        self.assertTrue(ok_run)
        providers = status["llm_availability"]["providers"]
        self.assertIn("ollama", providers["local_up"])
        ollama = status["llm_availability"]["ollama"]
        self.assertEqual("http://127.0.0.1:11434", ollama["native_base"])
        self.assertEqual("http://127.0.0.1:11434/v1", ollama["openai_base"])

    def test_health_marks_ollama_down_on_connection_refused(self) -> None:
        runtime = self._runtime("http://127.0.0.1:11434")

        def _get(url: str, timeout_seconds: float = 0.0, headers: dict[str, str] | None = None) -> dict[str, object]:
            _ = (timeout_seconds, headers, url)
            raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))

        with patch.object(runtime, "_http_get_json", side_effect=_get):
            ok_run, _payload = runtime.run_llm_health(trigger="manual")
            status = runtime.model_status()

        self.assertTrue(ok_run)
        providers = status["llm_availability"]["providers"]
        self.assertIn("ollama", providers["local_down"])
        ollama = status["llm_availability"]["ollama"]
        self.assertEqual("connection_refused", ollama["last_error_kind"])

    def test_health_marks_ollama_down_on_native_500(self) -> None:
        runtime = self._runtime("http://127.0.0.1:11434")

        def _get(url: str, timeout_seconds: float = 0.0, headers: dict[str, str] | None = None) -> dict[str, object]:
            _ = (timeout_seconds, headers)
            raise urllib.error.HTTPError(url, 500, "server error", None, None)

        with patch.object(runtime, "_http_get_json", side_effect=_get):
            ok_run, _payload = runtime.run_llm_health(trigger="manual")
            status = runtime.model_status()

        self.assertTrue(ok_run)
        providers = status["llm_availability"]["providers"]
        self.assertIn("ollama", providers["local_down"])
        ollama = status["llm_availability"]["ollama"]
        self.assertEqual("bad_status_code", ollama["last_error_kind"])
        self.assertEqual(500, ollama["last_status_code"])
        serialized = json.dumps(status, ensure_ascii=True)
        self.assertNotIn("OPENROUTER_API_KEY", serialized)
        self.assertNotIn("sk-", serialized)


if __name__ == "__main__":
    unittest.main()
