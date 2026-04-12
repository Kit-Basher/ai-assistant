from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agent.config import Config, canonical_config_dir, canonical_state_dir, load_config
from agent.golden_path import next_step_for_failure
from agent.secret_store import SecretStore
from agent.telegram_runtime_state import read_telegram_enablement


@dataclass(frozen=True)
class StartupCheck:
    check_id: str
    status: str
    failure_code: str | None
    message: str
    next_action: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "status": self.status,
            "failure_code": self.failure_code,
            "message": self.message,
            "next_action": self.next_action,
        }


def _trace_id(service: str) -> str:
    now_epoch = int(time.time())
    seed = f"{service}:{now_epoch}:{os.getpid()}".encode("utf-8")
    return f"startup-{service}-{hashlib.sha256(seed).hexdigest()[:10]}"


def _secret_store_path() -> Path:
    configured = str(os.getenv("AGENT_SECRET_STORE_PATH", "")).strip()
    if configured:
        return Path(configured).expanduser()
    return canonical_state_dir() / "secrets.enc.json"


def _required_dirs() -> list[Path]:
    return [
        canonical_state_dir(),
        canonical_config_dir(),
        Path.home() / ".config" / "systemd" / "user",
    ]


def _telegram_lock_path() -> Path:
    env_dir = str(os.getenv("AGENT_TELEGRAM_POLL_LOCK_DIR", "")).strip()
    root = Path(env_dir).expanduser() if env_dir else canonical_state_dir()
    return root / "telegram_poll.default.lock"


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _telegram_enabled(config: Config | None) -> bool:
    if config is not None and isinstance(getattr(config, "telegram_enabled", None), bool):
        return bool(getattr(config, "telegram_enabled"))
    try:
        return bool(read_telegram_enablement().get("enabled", False))
    except Exception:
        pass
    return _truthy(os.getenv("TELEGRAM_ENABLED", "0"))


def _check_required_dirs() -> StartupCheck:
    missing = [str(path) for path in _required_dirs() if not path.is_dir()]
    if not missing:
        return StartupCheck("dirs.required", "PASS", None, "required directories present", None)
    return StartupCheck(
        "dirs.required",
        "WARN",
        "required_dirs_missing",
        "missing directories: " + ", ".join(missing),
        "Run: python -m agent doctor --fix",
    )


def _check_secret_store() -> StartupCheck:
    path = _secret_store_path()
    if not path.is_file():
        return StartupCheck(
            "secret_store.readable",
            "WARN",
            "secret_store_missing",
            f"secret store missing: {path}",
            next_step_for_failure("config_invalid"),
        )
    if not os.access(path, os.R_OK):
        return StartupCheck(
            "secret_store.readable",
            "FAIL",
            "secret_store_unreadable",
            f"secret store unreadable: {path}",
            next_step_for_failure("config_invalid"),
        )
    try:
        store = SecretStore(path=str(path))
        store.validate()
    except Exception:
        return StartupCheck(
            "secret_store.readable",
            "FAIL",
            "secret_store_decrypt_failed",
            f"secret store decrypt failed: {path}",
            next_step_for_failure("config_invalid"),
        )
    return StartupCheck("secret_store.readable", "PASS", None, f"secret store readable: {path}", None)


def _check_lock_path() -> StartupCheck:
    primary_path = _telegram_lock_path()
    candidates: list[Path] = [primary_path]
    if not str(os.getenv("AGENT_TELEGRAM_POLL_LOCK_DIR", "")).strip():
        candidates.append(Path("/tmp") / "personal-agent" / primary_path.name)

    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix="startup-lock-", dir=str(candidate.parent), delete=True):
                pass
            return StartupCheck("lock.path", "PASS", None, f"lock path available: {candidate.parent}", None)
        except PermissionError:
            continue
        except Exception:
            continue

    return StartupCheck(
        "lock.path",
        "FAIL",
        "lock_path_unavailable",
        f"lock path unavailable: {primary_path.parent}",
        next_step_for_failure("lock_path_unavailable"),
    )


def _check_registry(config: Config) -> StartupCheck:
    registry_path = str(config.llm_registry_path or "").strip()
    if not registry_path:
        return StartupCheck(
            "registry.readable",
            "WARN",
            "registry_missing",
            "LLM registry path is not configured",
            next_step_for_failure("registry_unreadable"),
        )
    path = Path(registry_path).expanduser()
    if not path.is_file():
        return StartupCheck(
            "registry.readable",
            "WARN",
            "registry_missing",
            f"LLM registry missing: {path}",
            next_step_for_failure("registry_unreadable"),
        )
    if not os.access(path, os.R_OK):
        return StartupCheck(
            "registry.readable",
            "FAIL",
            "registry_unreadable",
            f"LLM registry unreadable: {path}",
            next_step_for_failure("registry_unreadable"),
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            parsed = json.load(handle)
        if not isinstance(parsed, dict):
            raise ValueError("registry_not_object")
    except Exception:
        return StartupCheck(
            "registry.readable",
            "FAIL",
            "registry_invalid_json",
            f"LLM registry invalid JSON: {path}",
            next_step_for_failure("registry_invalid_json"),
        )
    return StartupCheck("registry.readable", "PASS", None, f"LLM registry readable: {path}", None)


def _check_router_config(config: Config) -> StartupCheck:
    provider = str(getattr(config, "llm_provider", "") or "").strip().lower()
    if not provider:
        return StartupCheck(
            "api.router_config",
            "FAIL",
            "router_unavailable",
            "LLM provider is not configured",
            next_step_for_failure("router_unavailable"),
        )
    return StartupCheck("api.router_config", "PASS", None, f"LLM provider configured: {provider}", None)


def _check_local_provider_endpoint(config: Config) -> StartupCheck:
    provider = str(getattr(config, "llm_provider", "") or "").strip().lower()
    if provider != "ollama":
        return StartupCheck("provider.endpoint_sanity", "PASS", None, "local provider sanity skipped", None)
    base_url = str(getattr(config, "ollama_base_url", "") or getattr(config, "ollama_host", "") or "").strip()
    if not base_url:
        return StartupCheck(
            "provider.endpoint_sanity",
            "WARN",
            "provider_endpoint_missing",
            "Ollama endpoint is not configured",
            next_step_for_failure("config_invalid"),
        )
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return StartupCheck(
            "provider.endpoint_sanity",
            "FAIL",
            "provider_endpoint_invalid",
            f"Ollama endpoint is invalid: {base_url}",
            next_step_for_failure("config_invalid"),
        )
    return StartupCheck("provider.endpoint_sanity", "PASS", None, f"Ollama endpoint looks valid: {base_url}", None)


def _check_telegram_token(token: str | None) -> StartupCheck:
    value = str(token or "").strip()
    if value:
        return StartupCheck("telegram.token_present", "PASS", None, "telegram token present", None)
    return StartupCheck(
        "telegram.token_present",
        "FAIL",
        "telegram_token_missing",
        "telegram token is missing",
        next_step_for_failure("telegram_token_missing"),
    )


def _check_telegram_enabled(enabled: bool) -> StartupCheck:
    if bool(enabled):
        return StartupCheck(
            "telegram.enabled",
            "PASS",
            None,
            "telegram adapter enabled",
            None,
        )
    return StartupCheck(
        "telegram.enabled",
        "PASS",
        None,
        "telegram adapter disabled (optional)",
        None,
    )


def run_startup_checks(
    *,
    service: str,
    config: Config | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    normalized_service = str(service or "").strip().lower() or "unknown"
    trace_id = _trace_id(normalized_service)

    checks: list[StartupCheck] = [
        _check_required_dirs(),
        _check_secret_store(),
        _check_lock_path(),
    ]

    loaded_config = config
    if loaded_config is None:
        try:
            loaded_config = load_config(require_telegram_token=False)
        except Exception as exc:
            config_failure_status = "WARN" if normalized_service == "telegram" else "FAIL"
            checks.append(
                StartupCheck(
                    "config.load",
                    config_failure_status,
                    "config_load_failed",
                    f"config load failed: {exc.__class__.__name__}",
                    next_step_for_failure("config_load_failed"),
                )
            )
            if config_failure_status == "FAIL":
                return {
                    "trace_id": trace_id,
                    "component": f"{normalized_service}.startup",
                    "status": "FAIL",
                    "checks": [item.to_dict() for item in checks],
                    "failure_code": "config_load_failed",
                    "next_action": next_step_for_failure("config_load_failed"),
                }

    if loaded_config is not None:
        checks.append(StartupCheck("config.load", "PASS", None, "config loaded", None))
        checks.append(_check_registry(loaded_config))
        if normalized_service == "api":
            checks.append(_check_router_config(loaded_config))
        checks.append(_check_local_provider_endpoint(loaded_config))
    else:
        checks.append(
            StartupCheck(
                "registry.readable",
                "WARN",
                "registry_check_skipped",
                "registry check skipped because config did not load",
                next_step_for_failure("config_load_failed"),
            )
        )
        checks.append(
            StartupCheck(
                "provider.endpoint_sanity",
                "WARN",
                "provider_endpoint_check_skipped",
                "provider endpoint check skipped because config did not load",
                next_step_for_failure("config_load_failed"),
            )
        )
    if normalized_service == "telegram":
        telegram_enabled = _telegram_enabled(loaded_config)
        checks.append(_check_telegram_enabled(telegram_enabled))
        if telegram_enabled:
            checks.append(_check_telegram_token(token))

    status = "PASS"
    failure_code: str | None = None
    next_action: str | None = None
    for item in checks:
        if item.status == "FAIL":
            status = "FAIL"
            failure_code = item.failure_code or "startup_check_failed"
            next_action = item.next_action or next_step_for_failure(failure_code)
            break
        if item.status == "WARN" and status != "FAIL":
            status = "WARN"
            if next_action is None:
                next_action = item.next_action or next_step_for_failure(item.failure_code)

    return {
        "trace_id": trace_id,
        "component": f"{normalized_service}.startup",
        "status": status,
        "checks": [item.to_dict() for item in checks],
        "failure_code": failure_code,
        "next_action": next_action,
    }


__all__ = ["run_startup_checks"]
