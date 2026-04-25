from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TestProviderMatrixSmoke(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_module(REPO_ROOT / "scripts" / "provider_matrix_smoke.py", "provider_matrix_smoke_script")

    def test_provider_gate_prefers_usable_local_model_and_skips_unusable_cloud(self) -> None:
        module = self.module
        local_row = {
            "id": "ollama",
            "local": True,
            "enabled": True,
            "configured": True,
            "health": {"status": "ok"},
            "connection_state": "configured_and_usable",
            "selection_state": "configured_and_usable",
            "models": [
                {"model_id": "ollama:llama3", "usable_now": False},
                {"model_id": "ollama:qwen2.5:7b-instruct", "usable_now": True},
            ],
        }
        status, target, reason = module._build_provider_target(local_row, required=True)
        self.assertEqual("ok", status)
        self.assertIsNotNone(target)
        self.assertEqual("ollama", target.provider_id)
        self.assertEqual("ollama:qwen2.5:7b-instruct", target.model_id)
        self.assertTrue(target.required)
        self.assertEqual("", reason)

        cloud_row = {
            "id": "openrouter",
            "local": False,
            "enabled": True,
            "configured": True,
            "auth_required": True,
            "secret_present": False,
            "health": {"status": "ok"},
            "connection_state": "configured_and_usable",
            "selection_state": "configured_and_usable",
            "models": [{"model_id": "openrouter:openai/gpt-4o-mini", "usable_now": True}],
        }
        status, target, reason = module._build_provider_target(cloud_row, required=False)
        self.assertEqual("skip", status)
        self.assertIsNone(target)
        self.assertIn("missing credentials", reason)

    def test_run_target_uses_explicit_target_without_mutating_defaults(self) -> None:
        module = self.module
        target = module.ProviderTarget(
            provider_id="ollama",
            model_id="ollama:qwen2.5:7b-instruct",
            source="provider_row",
            health_status="ok",
            configured=True,
            active=False,
            required=True,
        )
        original_defaults = {
            "default_provider": "ollama",
            "default_model": "ollama:llama3",
            "routing_mode": "auto",
            "allow_remote_fallback": True,
        }
        calls: list[tuple[str, str, dict[str, object] | None]] = []

        def _fake_request_payload(base_url, method, path, payload=None, *, timeout):  # noqa: ANN001
            _ = base_url
            _ = timeout
            calls.append((method, path, payload if isinstance(payload, dict) else None))
            if method == "GET" and path == "/llm/status":
                return {
                    "ok": True,
                    "payload": {
                        "runtime_mode": "READY",
                        "active_provider": target.provider_id,
                        "active_model_id": target.model_id,
                        "active_provider_health": {"status": "ok"},
                        "active_model_health": {"status": "ok"},
                        "visible_counts": {"total": 2},
                        "compat_only": True,
                        "non_canonical_for_assistant": True,
                    },
                }
            raise AssertionError(f"unexpected request: {method} {path}")

        def _fake_chat_probe(base_url, prompt, *, user_id, thread_id, timeout, provider=None, model=None):  # noqa: ANN001
            _ = base_url
            _ = prompt
            _ = user_id
            _ = thread_id
            _ = timeout
            self.assertEqual(target.provider_id, provider)
            self.assertEqual(target.model_id, model)
            return {
                "ok": True,
                "status": 200,
                "text": "Alive.",
                "first_line": "Alive.",
                "route": "generic_chat",
                "payload": {
                    "assistant": {"content": "Alive."},
                    "meta": {
                        "used_llm": True,
                        "provider": target.provider_id,
                        "model": target.model_id,
                        "generic_fallback_used": False,
                        "chat_timing_ms": {"llm_request_ms": 23},
                        "route": "generic_chat",
                    },
                },
            }

        with patch.object(module, "_request_payload", side_effect=_fake_request_payload), patch.object(
            module,
            "_chat_probe",
            side_effect=_fake_chat_probe,
        ):
            ok, detail = module._run_target("http://127.0.0.1:8765", target, original_defaults, timeout=45.0)

        self.assertTrue(ok)
        self.assertIn("ollama", detail)
        self.assertIn("qwen2.5:7b-instruct", detail)
        self.assertEqual([("GET", "/llm/status", None)], calls)

    def test_main_runs_local_target_and_skips_unconfigured_cloud(self) -> None:
        module = self.module
        local_row = {
            "id": "ollama",
            "local": True,
            "enabled": True,
            "configured": True,
            "health": {"status": "ok"},
            "connection_state": "configured_and_usable",
            "selection_state": "configured_and_usable",
            "models": [{"model_id": "ollama:qwen2.5:7b-instruct", "usable_now": True}],
        }
        providers_payload = {"providers": [local_row]}
        defaults_payload = {
            "default_provider": "ollama",
            "default_model": "ollama:llama3",
            "routing_mode": "auto",
            "allow_remote_fallback": True,
        }
        status_payload = {
            "runtime_mode": "READY",
            "active_provider": "ollama",
            "active_model_id": "ollama:qwen2.5:7b-instruct",
        }
        calls: list[tuple[str, str]] = []

        def _fake_request_payload(base_url, method, path, payload=None, *, timeout):  # noqa: ANN001
            _ = base_url
            _ = payload
            _ = timeout
            calls.append((method, path))
            if method == "GET" and path == "/providers":
                return {"ok": True, "payload": providers_payload}
            if method == "GET" and path == "/defaults":
                return {"ok": True, "payload": defaults_payload}
            if method == "GET" and path == "/llm/status":
                return {"ok": True, "payload": status_payload}
            if method == "PUT" and path == "/defaults":
                return {"ok": True, "payload": {"ok": True}}
            raise AssertionError(f"unexpected request: {method} {path}")

        def _fake_run_target(base_url, target, defaults, *, timeout):  # noqa: ANN001
            _ = base_url
            _ = defaults
            _ = timeout
            self.assertEqual("ollama", target.provider_id)
            self.assertEqual("ollama:qwen2.5:7b-instruct", target.model_id)
            return True, "provider=ollama model=ollama:qwen2.5:7b-instruct chat=Alive."

        with patch.object(module, "_request_payload", side_effect=_fake_request_payload), patch.object(
            module,
            "_run_target",
            side_effect=_fake_run_target,
        ):
            exit_code = module.main(["--base-url", "http://127.0.0.1:8765"])

        self.assertEqual(0, exit_code)
        self.assertIn(("GET", "/providers"), calls)
        self.assertIn(("GET", "/defaults"), calls)
        self.assertIn(("GET", "/llm/status"), calls)

    def test_chat_contract_check_allows_optional_provider_override_drift(self) -> None:
        module = self.module
        target = module.ProviderTarget(
            provider_id="openrouter",
            model_id="openrouter:openai/gpt-4o-mini",
            source="provider_row",
            health_status="ok",
            configured=True,
            active=False,
            required=False,
        )
        ok, detail = module._chat_contract_check(
            {
                "ok": True,
                "status": 200,
                "text": "A bicycle has two wheels.",
                "first_line": "A bicycle has two wheels.",
                "payload": {
                    "assistant": {"content": "A bicycle has two wheels."},
                    "meta": {
                        "used_llm": True,
                        "provider": "ollama",
                        "model": "Gemma:latest",
                        "chat_timing_ms": {"llm_request_ms": 12},
                        "route": "generic_chat",
                    },
                },
            },
            target,
        )
        self.assertTrue(ok)
        self.assertIn("override not selected", detail)

    def test_chat_contract_check_allows_required_provider_model_drift(self) -> None:
        module = self.module
        target = module.ProviderTarget(
            provider_id="ollama",
            model_id="ollama:qwen2.5:7b-instruct",
            source="provider_row",
            health_status="ok",
            configured=True,
            active=True,
            required=True,
        )
        ok, detail = module._chat_contract_check(
            {
                "ok": True,
                "status": 200,
                "text": "A bicycle has two wheels.",
                "first_line": "A bicycle has two wheels.",
                "payload": {
                    "assistant": {"content": "A bicycle has two wheels."},
                    "meta": {
                        "used_llm": True,
                        "provider": "ollama",
                        "model": "Gemma:latest",
                        "chat_timing_ms": {"llm_request_ms": 12},
                        "route": "generic_chat",
                    },
                },
            },
            target,
        )
        self.assertTrue(ok)
        self.assertIn("A bicycle has two wheels.", detail)


if __name__ == "__main__":
    unittest.main()
