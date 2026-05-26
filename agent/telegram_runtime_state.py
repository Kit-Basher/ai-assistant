from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

from agent.actions.managed_action_recovery import ManagedActionJournal

from agent.secret_store import SecretStore


TELEGRAM_SERVICE_NAME = "personal-agent-telegram.service"


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_int(value: Any) -> int | None:
    try:
        return int(str(value or "").strip())
    except (TypeError, ValueError):
        return None


def telegram_control_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    active = dict(env or os.environ)
    active.pop("TELEGRAM_ENABLED", None)
    return active


def telegram_dropin_dir(*, home: Path | None = None) -> Path:
    root = home or Path.home()
    return root / ".config" / "systemd" / "user" / f"{TELEGRAM_SERVICE_NAME}.d"


def telegram_dropin_path(*, home: Path | None = None) -> Path:
    return telegram_dropin_dir(home=home) / "override.conf"


def is_personal_agent_telegram_dropin_path(path: str | Path, *, home: Path | None = None) -> bool:
    try:
        return Path(path).expanduser().resolve(strict=False) == telegram_dropin_path(home=home).expanduser().resolve(strict=False)
    except Exception:
        return False


def _default_secret_store_path(*, home: Path | None = None) -> Path:
    root = home or Path.home()
    return root / ".local" / "share" / "personal-agent" / "secrets.enc.json"


def _parse_environment_overrides(content: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in str(content or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("Environment="):
            continue
        assignment = line.split("=", 1)[1]
        if "=" not in assignment:
            continue
        key, value = assignment.split("=", 1)
        values[str(key).strip()] = str(value).strip().strip('"')
    return values


def read_telegram_enablement(*, env: Mapping[str, str] | None = None, home: Path | None = None) -> dict[str, Any]:
    active_env = dict(env or os.environ)
    raw_env = str(active_env.get("TELEGRAM_ENABLED", "") or "").strip()
    if raw_env:
        return {
            "enabled": _truthy(raw_env),
            "config_source": "env",
            "raw_value": raw_env,
            "source_path": None,
        }
    dropin = telegram_dropin_path(home=home)
    if dropin.is_file():
        try:
            content = dropin.read_text(encoding="utf-8")
        except Exception:
            content = ""
        overrides = _parse_environment_overrides(content)
        raw_dropin = str(overrides.get("TELEGRAM_ENABLED", "") or "").strip()
        if raw_dropin:
            return {
                "enabled": _truthy(raw_dropin),
                "config_source": "config",
                "raw_value": raw_dropin,
                "source_path": str(dropin),
            }
    return {
        "enabled": False,
        "config_source": "default",
        "raw_value": "0",
        "source_path": None,
    }


def resolve_telegram_token_with_source(
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    secret_store_path: str | None = None,
) -> tuple[str | None, str]:
    active_env = dict(env or os.environ)
    configured_secret_store = str(secret_store_path or active_env.get("AGENT_SECRET_STORE_PATH", "") or "").strip()
    secret_path = Path(configured_secret_store).expanduser() if configured_secret_store else _default_secret_store_path(home=home)
    try:
        store = SecretStore(path=str(secret_path))
        secret_token = str(store.get_secret("telegram:bot_token") or "").strip()
    except Exception:
        secret_token = ""
    if secret_token:
        return secret_token, "secret_store"
    env_token = str(active_env.get("TELEGRAM_BOT_TOKEN", "") or "").strip()
    if env_token:
        return env_token, "env"
    return None, "missing"


def _token_hash(token: str | None) -> str:
    import hashlib

    value = str(token or "").strip()
    if not value:
        return "default"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def telegram_lock_paths(
    token: str | None,
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> list[Path]:
    active_env = dict(env or os.environ)
    env_dir = str(active_env.get("AGENT_TELEGRAM_POLL_LOCK_DIR", "") or "").strip()
    root = Path(env_dir).expanduser() if env_dir else ((home or Path.home()) / ".local" / "share" / "personal-agent")
    token_name = f"telegram_poll.{_token_hash(token)}.lock"
    primary = root / token_name
    candidates = [primary]
    if not env_dir:
        candidates.append(Path("/tmp") / "personal-agent" / token_name)
    candidates.extend(
        [
            root / "telegram_poll.lock",
            root / "telegram_poll.default.lock",
        ]
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        normalized = str(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(path)
    return unique


def _is_pid_running(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def inspect_telegram_lock(
    token: str | None,
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> dict[str, Any]:
    for path in telegram_lock_paths(token, env=env, home=home):
        if not path.exists():
            continue
        try:
            raw_lines = path.read_text(encoding="utf-8").strip().splitlines()
        except Exception:
            raw_lines = []
        pid = _safe_int(raw_lines[0]) if raw_lines else None
        pid_running = _is_pid_running(pid) if pid is not None else False
        stale = bool(pid is not None and not pid_running)
        live = bool(pid is not None and pid_running)
        return {
            "present": True,
            "path": str(path),
            "pid": pid,
            "stale": stale,
            "live": live,
        }
    return {
        "present": False,
        "path": None,
        "pid": None,
        "stale": False,
        "live": False,
    }


def clear_stale_telegram_locks(
    token: str | None,
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> list[str]:
    removed: list[str] = []
    for path in telegram_lock_paths(token, env=env, home=home):
        if not path.exists():
            continue
        try:
            raw_lines = path.read_text(encoding="utf-8").strip().splitlines()
        except Exception:
            raw_lines = []
        pid = _safe_int(raw_lines[0]) if raw_lines else None
        if pid is None or _is_pid_running(pid):
            continue
        try:
            path.unlink()
            removed.append(str(path))
        except OSError:
            continue
    return removed


def _systemctl_user(
    args: list[str],
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    timeout_seconds: float = 1.0,
) -> subprocess.CompletedProcess[str]:
    runner = run or subprocess.run
    return runner(
        ["systemctl", "--user", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=float(timeout_seconds),
    )


def _service_installed(*, run: Callable[..., subprocess.CompletedProcess[str]] | None = None) -> bool:
    try:
        proc = _systemctl_user(["cat", TELEGRAM_SERVICE_NAME], run=run)
    except Exception:
        return False
    return proc.returncode == 0


def _service_active(*, run: Callable[..., subprocess.CompletedProcess[str]] | None = None) -> bool:
    try:
        proc = _systemctl_user(["is-active", TELEGRAM_SERVICE_NAME], run=run)
    except Exception:
        return False
    return (proc.stdout or "").strip() == "active"


def _service_enabled(*, run: Callable[..., subprocess.CompletedProcess[str]] | None = None) -> bool:
    try:
        proc = _systemctl_user(["is-enabled", TELEGRAM_SERVICE_NAME], run=run)
    except Exception:
        return False
    return (proc.stdout or "").strip() == "enabled"


def get_telegram_runtime_state(
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    secret_store_path: str | None = None,
) -> dict[str, Any]:
    enablement = read_telegram_enablement(env=env, home=home)
    token, token_source = resolve_telegram_token_with_source(
        env=env,
        home=home,
        secret_store_path=secret_store_path,
    )
    lock_info = inspect_telegram_lock(token, env=env, home=home)
    service_installed = _service_installed(run=run)
    service_active = _service_active(run=run) if service_installed else False
    service_enabled = _service_enabled(run=run) if service_installed else False
    enabled = bool(enablement.get("enabled", False))
    token_configured = bool(token)

    if not enabled:
        effective_state = "disabled_optional"
        next_action = "Run: python -m agent telegram_enable"
        ready_state = "disabled_optional"
    elif not service_installed:
        effective_state = "enabled_misconfigured"
        next_action = f"Install or restore {TELEGRAM_SERVICE_NAME}"
        ready_state = "stopped"
    elif not token_configured:
        effective_state = "enabled_misconfigured"
        next_action = "Run: python -m agent.secrets set telegram:bot_token"
        ready_state = "disabled_missing_token"
    elif service_active:
        effective_state = "enabled_running"
        next_action = "No action needed."
        ready_state = "running"
    elif bool(lock_info.get("present")):
        effective_state = "enabled_blocked_by_lock"
        next_action = "Run: python -m agent telegram_enable"
        ready_state = "stopped"
    else:
        effective_state = "enabled_stopped"
        next_action = "Run: python -m agent telegram_enable"
        ready_state = "stopped"

    return {
        "enabled": enabled,
        "config_source": str(enablement.get("config_source") or "default"),
        "config_source_path": enablement.get("source_path"),
        "service_installed": bool(service_installed),
        "service_active": bool(service_active),
        "service_enabled": bool(service_enabled),
        "token_configured": token_configured,
        "token_source": token_source,
        "lock_present": bool(lock_info.get("present", False)),
        "lock_live": bool(lock_info.get("live", False)),
        "lock_stale": bool(lock_info.get("stale", False)),
        "lock_path": lock_info.get("path"),
        "lock_pid": lock_info.get("pid"),
        "effective_state": effective_state,
        "ready_state": ready_state,
        "next_action": next_action,
    }


def write_telegram_enablement(
    enabled: bool,
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    secret_store_path: str | None = None,
) -> str:
    active_env = dict(env or os.environ)
    path = telegram_dropin_path(home=home)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = ""
    if path.is_file():
        try:
            existing = path.read_text(encoding="utf-8")
        except Exception:
            existing = ""
    overrides = _parse_environment_overrides(existing)
    overrides["TELEGRAM_ENABLED"] = "1" if enabled else "0"

    configured_secret_path = str(secret_store_path or active_env.get("AGENT_SECRET_STORE_PATH", "") or "").strip()
    if configured_secret_path:
        overrides["AGENT_SECRET_STORE_PATH"] = configured_secret_path
    else:
        default_secret_path = _default_secret_store_path(home=home)
        if default_secret_path.exists():
            overrides.setdefault("AGENT_SECRET_STORE_PATH", str(default_secret_path))

    content_lines = ["[Service]"]
    for key in sorted(overrides.keys()):
        content_lines.append(f"Environment={key}={overrides[key]}")
    new_content = "\n".join(content_lines) + "\n"
    path.write_text(new_content, encoding="utf-8")
    return str(path)


def write_telegram_enablement_managed(
    enabled: bool,
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    secret_store_path: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    path = telegram_dropin_path(home=home)
    journal = ManagedActionJournal(action_type="telegram_enablement_config", target=str(path))
    journal.plan_step("preflight_telegram_dropin_path", resource=str(path))
    journal.plan_step("capture_previous_telegram_dropin", resource=str(path))
    journal.plan_step("write_telegram_enablement", resource=str(path))
    journal.plan_step("verify_telegram_enablement_write", resource=str(path))

    if not is_personal_agent_telegram_dropin_path(path, home=home):
        journal.record_step("preflight_telegram_dropin_path", ok=False, resource=str(path), reason="unexpected_dropin_path")
        return False, {
            "ok": False,
            "error": "telegram_dropin_path_invalid",
            "error_kind": "telegram_dropin_path_invalid",
            "message": "Telegram setup did not work: the service drop-in target was not the known Personal Agent path.",
            "managed_action_journal": journal.to_dict(),
        }
    journal.record_step("preflight_telegram_dropin_path", ok=True, resource=str(path))

    previous_exists = path.is_file()
    previous_content = ""
    if previous_exists:
        try:
            previous_content = path.read_text(encoding="utf-8")
        except Exception:
            previous_content = ""
    journal.record_step(
        "capture_previous_telegram_dropin",
        ok=True,
        resource=str(path),
        previous_exists=previous_exists,
        previous_length=len(previous_content),
    )

    try:
        written_path = write_telegram_enablement(
            enabled,
            env=env,
            home=home,
            secret_store_path=secret_store_path,
        )
    except Exception as exc:
        journal.record_step("write_telegram_enablement", ok=False, resource=str(path), error=exc.__class__.__name__)
        journal.mark_verification(ok=False, dropin_write_verified=False)
        return False, {
            "ok": False,
            "error": "telegram_dropin_write_failed",
            "error_kind": "telegram_dropin_write_failed",
            "message": "Telegram setup did not work: I could not write the Personal Agent Telegram service config.",
            "managed_action_journal": journal.to_dict(),
        }

    journal.record_step("write_telegram_enablement", ok=True, resource=written_path, enabled=bool(enabled))
    journal.record_changed_resource(
        "telegram_service_dropin",
        str(path),
        rollback_supported=True,
        previous_exists=previous_exists,
    )

    try:
        current = path.read_text(encoding="utf-8")
        expected = f"Environment=TELEGRAM_ENABLED={'1' if enabled else '0'}"
        verified = expected in current
        journal.record_step(
            "verify_telegram_enablement_write",
            ok=verified,
            resource=str(path),
            expected_enabled=bool(enabled),
            persisted_length=len(current),
        )
    except Exception as exc:
        verified = False
        journal.record_step(
            "verify_telegram_enablement_write",
            ok=False,
            resource=str(path),
            error=exc.__class__.__name__,
        )

    if not verified:
        rollback_ok = _rollback_telegram_dropin(path, previous_exists, previous_content, journal)
        summary = (
            "restored previous Telegram service config"
            if previous_exists and rollback_ok
            else "removed failed new Telegram service config"
            if rollback_ok
            else "could not restore Telegram service config automatically"
        )
        journal.mark_verification(ok=False, dropin_write_verified=False)
        journal.mark_rollback(ok=rollback_ok, attempted=True, summary=summary)
        return False, {
            "ok": False,
            "error": "telegram_dropin_write_verification_failed",
            "error_kind": "telegram_dropin_write_verification_failed",
            "message": f"Telegram setup did not work: I could not verify the service config. {summary}.",
            "rollback_ok": rollback_ok,
            "rollback_summary": summary,
            "managed_action_journal": journal.to_dict(),
        }

    journal.mark_verification(ok=True, dropin_write_verified=True)
    journal.mark_rollback(ok=True, attempted=False, summary="No rollback needed.")
    return True, {
        "ok": True,
        "path": str(path),
        "enabled": bool(enabled),
        "managed_action_journal": journal.to_dict(),
    }


def _rollback_telegram_dropin(
    path: Path,
    previous_exists: bool,
    previous_content: str,
    journal: ManagedActionJournal,
) -> bool:
    try:
        if previous_exists:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(previous_content, encoding="utf-8")
        elif path.exists():
            path.unlink()
        journal.record_rollback_step(
            "restore_telegram_dropin",
            ok=True,
            resource=str(path),
            previous_exists=previous_exists,
        )
        return True
    except Exception as exc:
        journal.record_rollback_step(
            "restore_telegram_dropin",
            ok=False,
            resource=str(path),
            error=exc.__class__.__name__,
        )
        return False


__all__ = [
    "TELEGRAM_SERVICE_NAME",
    "clear_stale_telegram_locks",
    "get_telegram_runtime_state",
    "inspect_telegram_lock",
    "read_telegram_enablement",
    "resolve_telegram_token_with_source",
    "is_personal_agent_telegram_dropin_path",
    "telegram_control_env",
    "telegram_dropin_path",
    "telegram_lock_paths",
    "write_telegram_enablement",
    "write_telegram_enablement_managed",
]
