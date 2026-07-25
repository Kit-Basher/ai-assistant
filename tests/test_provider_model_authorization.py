from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor

from agent.api_server import AgentRuntime
from agent.mutation_plan import build_mutation_confirmation
from test_api_server import _config


def _runtime(tmp: Path, **overrides: object) -> AgentRuntime:
    os.environ["AGENT_SECRET_STORE_PATH"] = str(tmp / "secrets.enc.json")
    os.environ["AGENT_PERMISSIONS_PATH"] = str(tmp / "permissions.json")
    os.environ["AGENT_AUDIT_LOG_PATH"] = str(tmp / "audit.jsonl")
    return AgentRuntime(
        _config(
            str(tmp / "registry.json"),
            str(tmp / "agent.db"),
            **{"llm_autopilot_safe_mode": False, **overrides},
        )
    )


def _confirmation(plan: dict[str, object]) -> dict[str, object]:
    return build_mutation_confirmation(plan, confirmation_id="explicit-user-confirmation")


def test_provider_alias_requires_scoped_plan_not_boolean_confirmation(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw))
        ok, body = runtime.route_provider_model_mutation(
            "provider.add",
            {
                "id": "fixture_provider",
                "base_url": "https://example.invalid/v1",
                "local": False,
                "confirm": True,
            },
        )
        assert not ok
        assert body["error"] == "boolean_confirmation_not_authorization"
        assert "fixture_provider" not in runtime.registry_document.get("providers", {})


def test_provider_plan_apply_uses_executor_and_is_single_use(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw))
        request = {
            "id": "fixture_provider",
            "base_url": "https://example.invalid/v1",
            "local": False,
            "actor_id": "actor-a",
            "thread_id": "thread-a",
            "session_id": "session-a",
        }
        ok, preview = runtime.route_provider_model_mutation("provider.add", request)
        assert ok and preview["requires_confirmation"]
        plan = preview["plan"]
        apply = {**request, "mutation_plan": plan, "confirmation": _confirmation(plan)}
        ok, receipt = runtime.route_provider_model_mutation("provider.add", apply)
        assert ok, (receipt.get("error_code"), receipt.get("details"), receipt)
        assert receipt["capability_id"] == "provider.configure"
        assert receipt["executor_id"] == "provider.add.v1"
        assert "fixture_provider" in runtime.registry_document.get("providers", {})
        replay_ok, replay = runtime.route_provider_model_mutation("provider.add", apply)
        assert not replay_ok
        assert replay["error"] in {"mutation_plan_target_changed", "mutation_confirmation_consumed"}


def test_secret_plan_and_receipt_never_contain_plaintext(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw))
        secret = "super-private-provider-key-value"
        request = {
            "provider_id": "openrouter",
            "api_key": secret,
            "verify_provider": False,
            "actor_id": "actor-secret",
            "thread_id": "thread-secret",
            "session_id": "session-secret",
        }
        ok, preview = runtime.route_provider_model_mutation("provider.secret.set", request)
        assert ok
        serialized_plan = json.dumps(preview, sort_keys=True)
        assert secret not in serialized_plan
        assert "opaque-v1:" in serialized_plan
        plan = preview["plan"]
        ok, receipt = runtime.route_provider_model_mutation(
            "provider.secret.set",
            {**request, "mutation_plan": plan, "confirmation": _confirmation(plan)},
        )
        assert ok
        assert secret not in json.dumps(receipt, sort_keys=True)
        assert runtime.secret_store.get_provider_api_key("openrouter") == secret


def test_hostile_provider_urls_fail_before_plan(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw))
        for url in (
            "http://user:password@example.invalid/v1",
            "file:///tmp/provider",
            "http://169.254.169.254/latest/meta-data",
            "http://127.0.0.1:9999/v1",
        ):
            ok, body = runtime.route_provider_model_mutation(
                "provider.add", {"id": "hostile", "base_url": url, "local": False}
            )
            assert not ok
            assert str(body["error"]).startswith("provider_")


def test_safe_mode_blocks_provider_change_but_allows_control_mode_plan(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw), safe_mode_enabled=True)
        ok, blocked = runtime.route_provider_model_mutation(
            "provider.add", {"id": "blocked", "base_url": "https://example.invalid/v1"}
        )
        assert not ok and blocked["error"] == "safe_mode_mutation_blocked"
        ok, preview = runtime.route_provider_model_mutation(
            "runtime.control_mode", {"mode": "controlled", "actor_id": "operator"}
        )
        assert ok and preview["requires_confirmation"]


def test_safe_mode_allows_confirmed_installed_usable_ollama_switch_once(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw), safe_mode_enabled=True)
        target = "ollama:qwen2.5:3b-instruct"
        eligibility_calls: list[tuple[str, str | None, bool, bool]] = []
        dispatched: list[dict[str, object]] = []

        def _eligible(
            model_id: str,
            *,
            provider_id: str | None = None,
            require_available: bool = True,
            require_usable: bool = False,
        ) -> tuple[bool, dict[str, object]]:
            eligibility_calls.append((model_id, provider_id, require_available, require_usable))
            return True, {"ok": True, "provider": "ollama", "model_id": target, "local": True}

        def _switch(payload: dict[str, object]) -> tuple[bool, dict[str, object]]:
            dispatched.append(dict(payload))
            return True, {
                "ok": True,
                "provider": "ollama",
                "model_id": target,
                "message": f"Now using {target} for chat.",
            }

        monkeypatch.setattr(runtime, "_resolve_switch_target_with_policy_guard", _eligible)
        monkeypatch.setattr(runtime, "llm_models_switch", _switch)
        request = {
            "provider": "ollama",
            "model_id": target,
            "actor_id": "actor-local",
            "thread_id": "thread-local",
            "session_id": "session-local",
        }

        ok, preview = runtime.route_provider_model_mutation("model.switch", request)
        assert ok, preview
        assert preview["requires_confirmation"]
        plan = preview["plan"]
        apply = {**request, "mutation_plan": plan, "confirmation": _confirmation(plan)}
        ok, receipt = runtime.route_provider_model_mutation("model.switch", apply)
        assert ok, receipt
        replay_ok, replay = runtime.route_provider_model_mutation("model.switch", apply)

        assert not replay_ok
        assert (replay.get("error") or replay.get("error_code")) in {
            "mutation_confirmation_consumed",
            "mutation_confirmation_replayed",
            "confirmation_consumed",
        }
        assert eligibility_calls == [(target, "ollama", True, True)]
        assert dispatched == [
            {
                "provider": "ollama",
                "model_id": "qwen2.5:3b-instruct",
                "model": "qwen2.5:3b-instruct",
                "confirm": True,
            }
        ]


def test_safe_mode_rejects_uninstalled_local_switch_before_confirmation(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw), safe_mode_enabled=True)

        def _unavailable(*_args: object, **_kwargs: object) -> tuple[bool, dict[str, object]]:
            return False, {
                "ok": False,
                "error": "switch_target_unavailable",
                "error_kind": "switch_target_unavailable",
                "message": "That local model is not installed and usable.",
            }

        monkeypatch.setattr(runtime, "_resolve_switch_target_with_policy_guard", _unavailable)
        ok, body = runtime.route_provider_model_mutation(
            "model.switch",
            {"provider": "ollama", "model_id": "ollama:not-installed"},
        )

        assert not ok
        assert body["error"] == "switch_target_unavailable"
        assert body.get("requires_confirmation") is not True
        assert "not present as an installed" in body["message"]
        assert "local inventory" in body["next_action"]


def test_safe_mode_still_blocks_remote_switch_and_model_download() -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw), safe_mode_enabled=True)

        remote_ok, remote = runtime.route_provider_model_mutation(
            "model.switch",
            {"provider": "openrouter", "model_id": "openrouter:openai/gpt-4o-mini"},
        )
        download_ok, download = runtime.route_provider_model_mutation(
            "model.acquire",
            {"provider": "ollama", "model_id": "ollama:qwen2.5:3b-instruct"},
        )

        assert not remote_ok
        assert remote["error"] == "safe_mode_remote_switch_blocked"
        assert not download_ok
        assert download["error"] == "safe_mode_model_acquisition_blocked"
        assert "does not allow model downloads" in download["why"]


def test_confirmation_scope_and_changed_secret_are_rejected(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw))
        request = {
            "provider_id": "openrouter",
            "api_key": "first-secret-value",
            "verify_provider": False,
            "actor_id": "actor-a",
            "thread_id": "thread-a",
            "session_id": "session-a",
        }
        ok, preview = runtime.route_provider_model_mutation("provider.secret.set", request)
        assert ok
        plan = preview["plan"]
        forged = build_mutation_confirmation(plan, confirmation_id="forged", actor_id="actor-b")
        ok, body = runtime.route_provider_model_mutation(
            "provider.secret.set", {**request, "mutation_plan": plan, "confirmation": forged}
        )
        assert not ok
        assert body["error_code"] == "mutation_confirmation_actor_id_mismatch"
        ok, changed = runtime.route_provider_model_mutation(
            "provider.secret.set",
            {
                **request,
                "api_key": "changed-after-preview",
                "mutation_plan": plan,
                "confirmation": _confirmation(plan),
            },
        )
        assert not ok
        assert changed["error"] == "mutation_plan_target_changed"


def test_concurrent_confirmation_has_one_executor_winner(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw))
        request = {
            "id": "race_provider",
            "base_url": "https://example.invalid/v1",
            "local": False,
            "actor_id": "actor-race",
            "thread_id": "thread-race",
            "session_id": "session-race",
        }
        ok, preview = runtime.route_provider_model_mutation("provider.add", request)
        assert ok
        plan = preview["plan"]
        apply = {**request, "mutation_plan": plan, "confirmation": _confirmation(plan)}
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _idx: runtime.route_provider_model_mutation("provider.add", apply), range(2)))
        assert sum(1 for ok_result, _body in results if ok_result) == 1
        assert sum(1 for ok_result, _body in results if not ok_result) == 1


def test_direct_domain_executor_call_fails_closed(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw))
        service = runtime._provider_model_authorization
        spec = service.registry.lookup("provider.add")
        assert spec is not None and spec.run is not None
        result = spec.run(
            {"plan_id": "direct", "plan_fingerprint": "x", "target_fingerprint": "y"},
            {"parameters": {"id": "bypass", "base_url": "https://example.invalid/v1"}},
        )
        assert result["ok"] is False
        assert result["mutated"] is False
        assert result["error_code"] == "generic_bypass_blocked"


def test_canonical_model_id_is_normalized_once_for_runtime_dispatch(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw))
        dispatched: list[dict[str, object]] = []

        def _switch(payload: dict[str, object]) -> tuple[bool, dict[str, object]]:
            dispatched.append(dict(payload))
            return True, {
                "ok": True,
                "provider": "ollama",
                "model_id": "ollama:deepseek-r1:7b",
                "message": "Now using ollama:deepseek-r1:7b for chat.",
            }

        monkeypatch.setattr(runtime, "llm_models_switch", _switch)
        request = {
            "provider": "ollama",
            "model_id": "ollama:deepseek-r1:7b",
            "actor_id": "actor-model",
            "thread_id": "thread-model",
            "session_id": "session-model",
        }
        ok, preview = runtime.route_provider_model_mutation("model.switch", request)
        assert ok
        plan = preview["plan"]
        ok, receipt = runtime.route_provider_model_mutation(
            "model.switch",
            {**request, "mutation_plan": plan, "confirmation": _confirmation(plan)},
        )
        assert ok, receipt
        assert dispatched == [
            {
                "provider": "ollama",
                "model_id": "deepseek-r1:7b",
                "model": "deepseek-r1:7b",
                "confirm": True,
            }
        ]
