from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

from agent.actions.managed_action_recovery import ManagedActionJournal
from agent.actions.persistent_journal import PersistentManagedActionJournalStore

from agent.secret_store import SecretStore


TELEGRAM_SERVICE_NAME = "personal-agent-telegram.service"
_APPROVED_SYSTEMCTL_USER_ACTIONS = {
    ("--version",),
    ("cat", TELEGRAM_SERVICE_NAME),
    ("is-active", TELEGRAM_SERVICE_NAME),
    ("is-enabled", TELEGRAM_SERVICE_NAME),
    ("daemon-reload",),
    ("restart", TELEGRAM_SERVICE_NAME),
    ("stop", TELEGRAM_SERVICE_NAME),
}
_TELEGRAM_TOKEN_RE = re.compile(r"\b\d{5,}:[A-Za-z0-9_-]{10,}\b")
_SECRET_ARG_RE = re.compile(r"(?i)(token|password|api[_-]?key|secret)=\S+")


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_int(value: Any) -> int | None:
    try:
        return int(str(value or "").strip())
    except (TypeError, ValueError):
        return None


def _redact_process_evidence(value: str) -> str:
    text = _TELEGRAM_TOKEN_RE.sub("[REDACTED_TELEGRAM_TOKEN]", str(value or ""))
    return _SECRET_ARG_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)


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


def inspect_secret_store_status(
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    secret_store_path: str | None = None,
) -> dict[str, Any]:
    active_env = dict(env or os.environ)
    configured_secret_store = str(secret_store_path or active_env.get("AGENT_SECRET_STORE_PATH", "") or "").strip()
    secret_path = Path(configured_secret_store).expanduser() if configured_secret_store else _default_secret_store_path(home=home)
    try:
        store = SecretStore(path=str(secret_path))
        status = store.status()
    except Exception as exc:
        status = {
            "backend": "encrypted_file",
            "path": str(secret_path),
            "exists": None,
            "readable": False,
            "valid": False,
            "error_kind": exc.__class__.__name__,
            "state": "error",
        }
    return {
        "backend": str(status.get("backend") or "encrypted_file"),
        "exists": status.get("exists"),
        "readable": status.get("readable"),
        "valid": bool(status.get("valid", False)),
        "error_kind": status.get("error_kind"),
        "state": str(status.get("state") or "unknown"),
    }


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


def inspect_telegram_pollers(
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    timeout_seconds: float = 0.6,
) -> dict[str, Any]:
    runner = run or subprocess.run
    try:
        proc = runner(
            ["ps", "-eo", "pid,args"],
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
        )
    except Exception as exc:
        return {
            "available": False,
            "count": None,
            "duplicate": False,
            "evidence": [],
            "error": exc.__class__.__name__,
        }
    lines: list[str] = []
    for row in str(proc.stdout or "").splitlines():
        low = row.lower()
        if "telegram_adapter" in low and "python" in low:
            lines.append(_redact_process_evidence(" ".join(row.strip().split()))[:220])
    return {
        "available": proc.returncode == 0,
        "count": len(lines),
        "duplicate": len(lines) > 1,
        "evidence": lines[:4],
        "error": None if proc.returncode == 0 else f"returncode:{proc.returncode}",
    }


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


def is_approved_telegram_systemctl_user_action(args: list[str]) -> bool:
    return tuple(str(item) for item in list(args or [])) in _APPROVED_SYSTEMCTL_USER_ACTIONS


def _safe_status_snapshot(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(state.get("enabled", False)),
        "config_source": str(state.get("config_source") or "default"),
        "service_installed": bool(state.get("service_installed", False)),
        "service_active": bool(state.get("service_active", False)),
        "service_enabled": bool(state.get("service_enabled", False)),
        "token_configured": bool(state.get("token_configured", False)),
        "lock_present": bool(state.get("lock_present", False)),
        "lock_stale": bool(state.get("lock_stale", False)),
        "duplicate_pollers": bool(state.get("duplicate_pollers", False)),
        "effective_state": str(state.get("effective_state") or "unknown"),
        "ready_state": str(state.get("ready_state") or "unknown"),
    }


def _run_approved_systemctl_user(
    args: list[str],
    *,
    journal: ManagedActionJournal,
    step_name: str,
    run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    timeout_seconds: float = 10.0,
) -> subprocess.CompletedProcess[str] | None:
    resource = " ".join(["systemctl", "--user", *args])
    if not is_approved_telegram_systemctl_user_action(args):
        journal.record_step(step_name, ok=False, resource=resource, reason="systemctl_action_not_approved")
        return None
    try:
        proc = _systemctl_user(args, run=run, timeout_seconds=timeout_seconds)
    except Exception as exc:
        journal.record_step(step_name, ok=False, resource=resource, error=exc.__class__.__name__)
        return None
    stdout = str(proc.stdout or "").strip()
    stderr = str(proc.stderr or "").strip()
    journal.record_step(
        step_name,
        ok=proc.returncode == 0,
        resource=resource,
        returncode=int(proc.returncode),
        stdout=stdout[:160],
        stderr=stderr[:160],
    )
    return proc


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
    secret_store_status = inspect_secret_store_status(env=env, home=home, secret_store_path=secret_store_path)
    lock_info = inspect_telegram_lock(token, env=env, home=home)
    poller_info = inspect_telegram_pollers(run=run)
    service_installed = _service_installed(run=run)
    service_active = _service_active(run=run) if service_installed else False
    service_enabled = _service_enabled(run=run) if service_installed else False
    enabled = bool(enablement.get("enabled", False))
    token_configured = bool(token)

    duplicate_pollers = bool(poller_info.get("duplicate", False))

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
        if duplicate_pollers:
            effective_state = "enabled_duplicate_pollers"
            next_action = "Stop duplicate Telegram pollers and keep only one running instance."
        else:
            effective_state = "enabled_running"
            next_action = "No action needed."
        ready_state = "running"
    elif duplicate_pollers:
        effective_state = "enabled_duplicate_pollers"
        next_action = "Stop duplicate Telegram pollers and keep only one running instance."
        ready_state = "stopped"
    elif bool(lock_info.get("present")):
        effective_state = "enabled_blocked_by_lock"
        next_action = "Run: python -m agent telegram_enable"
        ready_state = "stopped"
    else:
        effective_state = "enabled_stopped"
        next_action = "Run: python -m agent telegram_enable"
        ready_state = "stopped"

    polling_active = bool(service_active)
    handler_registered = bool(service_installed)
    duplicate_consumer_suspected = bool(duplicate_pollers)
    return {
        "enabled": enabled,
        "configured": token_configured,
        "token_present": token_configured,
        "token_validated": token_configured,
        "transport_mode": "polling",
        "polling_active": polling_active,
        "webhook_active": False,
        "handler_registered": handler_registered,
        "last_update_received_at": None,
        "last_update_processed_at": None,
        "last_reply_attempt_at": None,
        "last_reply_success_at": None,
        "last_error_code": "poll_conflict" if duplicate_consumer_suspected else None,
        "last_error_summary": "Duplicate Telegram pollers detected." if duplicate_consumer_suspected else None,
        "duplicate_consumer_suspected": duplicate_consumer_suspected,
        "runtime_reachable": None,
        "telegram_transport_healthy": bool(token_configured and polling_active and handler_registered and not duplicate_consumer_suspected),
        "config_source": str(enablement.get("config_source") or "default"),
        "config_source_path": enablement.get("source_path"),
        "service_installed": bool(service_installed),
        "service_active": bool(service_active),
        "service_enabled": bool(service_enabled),
        "token_configured": token_configured,
        "token_source": token_source,
        "secret_store_state": secret_store_status.get("state"),
        "secret_store_valid": bool(secret_store_status.get("valid", False)),
        "secret_store_error_kind": secret_store_status.get("error_kind"),
        "lock_present": bool(lock_info.get("present", False)),
        "lock_live": bool(lock_info.get("live", False)),
        "lock_stale": bool(lock_info.get("stale", False)),
        "lock_path": lock_info.get("path"),
        "lock_pid": lock_info.get("pid"),
        "poller_inspection_available": bool(poller_info.get("available", False)),
        "poller_count": poller_info.get("count"),
        "duplicate_pollers": duplicate_pollers,
        "poller_evidence": list(poller_info.get("evidence") or []),
        "poller_inspection_error": poller_info.get("error"),
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
    managed_action_journal_store: PersistentManagedActionJournalStore | str | Path | None = None,
) -> tuple[bool, dict[str, Any]]:
    path = telegram_dropin_path(home=home)
    journal = ManagedActionJournal(action_type="telegram_enablement_config", target="telegram_service_dropin")
    journal.plan_step("preflight_telegram_dropin_path", resource=str(path))
    journal.plan_step("capture_previous_telegram_dropin", resource=str(path))
    journal.plan_step("write_telegram_enablement", resource=str(path))
    journal.plan_step("verify_telegram_enablement_write", resource=str(path))

    if not is_personal_agent_telegram_dropin_path(path, home=home):
        journal.record_step("preflight_telegram_dropin_path", ok=False, resource=str(path), reason="unexpected_dropin_path")
        journal.mark_verification(ok=False, dropin_write_verified=False)
        _persist_telegram_managed_action_journal(
            journal,
            managed_action_journal_store,
            status="failed",
            recovery_hint="Retry only against the known Personal Agent Telegram service drop-in path.",
        )
        return False, {
            "ok": False,
            "error": "telegram_dropin_path_invalid",
            "error_kind": "telegram_dropin_path_invalid",
            "message": "Telegram setup did not work: the service drop-in target was not the known Personal Agent path.",
            "managed_action_journal": journal.to_dict(),
        }
    journal.record_step("preflight_telegram_dropin_path", ok=True, resource=str(path))
    _persist_telegram_managed_action_journal(journal, managed_action_journal_store, status="planned")

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
    _persist_telegram_managed_action_journal(journal, managed_action_journal_store, status="running")

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
        _persist_telegram_managed_action_journal(
            journal,
            managed_action_journal_store,
            status="failed",
            recovery_hint="Check write access to the Telegram service drop-in, then retry setup.",
        )
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
    _persist_telegram_managed_action_journal(journal, managed_action_journal_store, status="running")

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
        _persist_telegram_managed_action_journal(
            journal,
            managed_action_journal_store,
            status="rolled_back" if rollback_ok else "recovery_needed",
            recovery_hint="Confirm the Telegram drop-in contents before retrying service setup.",
        )
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
    _persist_telegram_managed_action_journal(journal, managed_action_journal_store, status="verified")
    return True, {
        "ok": True,
        "path": str(path),
        "enabled": bool(enabled),
        "managed_action_journal": journal.to_dict(),
    }


def manage_telegram_service_state(
    enabled: bool,
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    secret_store_path: str | None = None,
    run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    managed_action_journal_store: PersistentManagedActionJournalStore | str | Path | None = None,
) -> tuple[bool, dict[str, Any]]:
    action_name = "telegram_service_enable" if enabled else "telegram_service_disable"
    journal = ManagedActionJournal(action_type=action_name, target=TELEGRAM_SERVICE_NAME)
    path = telegram_dropin_path(home=home)
    journal.plan_step("preflight_telegram_dropin_path", resource=str(path))
    journal.plan_step("preflight_systemd_user_available", resource="systemctl --user --version")
    journal.plan_step("capture_previous_telegram_state", resource=TELEGRAM_SERVICE_NAME)
    journal.plan_step("write_telegram_enablement", resource=str(path))
    journal.plan_step("systemctl_daemon_reload", resource="systemctl --user daemon-reload")
    journal.plan_step("systemctl_service_action", resource=TELEGRAM_SERVICE_NAME)
    journal.plan_step("verify_telegram_runtime_state", resource=TELEGRAM_SERVICE_NAME)

    if not is_personal_agent_telegram_dropin_path(path, home=home):
        journal.record_step("preflight_telegram_dropin_path", ok=False, resource=str(path), reason="unexpected_dropin_path")
        journal.mark_verification(ok=False, dropin_path_valid=False)
        _persist_telegram_managed_action_journal(
            journal,
            managed_action_journal_store,
            status="failed",
            recovery_hint="Retry only against the known Personal Agent Telegram service drop-in path.",
        )
        return False, {
            "ok": False,
            "error": "telegram_dropin_path_invalid",
            "error_kind": "telegram_dropin_path_invalid",
            "message": "Telegram setup/start did not finish: the service config target was not the known Personal Agent path.",
            "managed_action_journal": journal.to_dict(),
        }
    journal.record_step("preflight_telegram_dropin_path", ok=True, resource=str(path))
    _persist_telegram_managed_action_journal(journal, managed_action_journal_store, status="planned")

    version_proc = _run_approved_systemctl_user(
        ["--version"],
        journal=journal,
        step_name="preflight_systemd_user_available",
        run=run,
        timeout_seconds=2.0,
    )
    if version_proc is None or version_proc.returncode != 0:
        journal.mark_verification(ok=False, systemd_user_available=False)
        _persist_telegram_managed_action_journal(
            journal,
            managed_action_journal_store,
            status="failed",
            recovery_hint="Check the user systemd manager, then retry Telegram service setup.",
        )
        return False, {
            "ok": False,
            "error": "systemd_user_unavailable",
            "error_kind": "systemd_user_unavailable",
            "message": "Telegram setup/start did not finish: systemd user services are not available. Check the user service manager, then try again.",
            "managed_action_journal": journal.to_dict(),
        }

    previous_exists = path.is_file()
    previous_content = ""
    if previous_exists:
        try:
            previous_content = path.read_text(encoding="utf-8")
        except Exception:
            previous_content = ""
    state_before = get_telegram_runtime_state(env=env, home=home, run=run, secret_store_path=secret_store_path)
    journal.record_step(
        "capture_previous_telegram_state",
        ok=True,
        resource=TELEGRAM_SERVICE_NAME,
        previous_dropin_exists=previous_exists,
        previous_state=_safe_status_snapshot(state_before),
    )
    _persist_telegram_managed_action_journal(journal, managed_action_journal_store, status="running")

    write_ok, write_body = write_telegram_enablement_managed(
        enabled,
        env=env,
        home=home,
        secret_store_path=secret_store_path,
        managed_action_journal_store=managed_action_journal_store,
    )
    journal.record_step(
        "write_telegram_enablement",
        ok=write_ok,
        resource=str(path),
        nested_action_id=str((write_body.get("managed_action_journal") or {}).get("action_id") or ""),
    )
    if not write_ok:
        journal.mark_verification(ok=False, dropin_write_verified=False)
        _persist_telegram_managed_action_journal(
            journal,
            managed_action_journal_store,
            status="failed",
            recovery_hint="Resolve the Telegram drop-in write failure, then retry service setup.",
        )
        body = dict(write_body)
        body["managed_action_journal"] = journal.to_dict()
        return False, body
    journal.record_changed_resource(
        "telegram_service_dropin",
        str(path),
        rollback_supported=True,
        previous_exists=previous_exists,
    )

    reload_proc = _run_approved_systemctl_user(
        ["daemon-reload"],
        journal=journal,
        step_name="systemctl_daemon_reload",
        run=run,
    )
    if reload_proc is None or reload_proc.returncode != 0:
        rollback_ok = _rollback_telegram_dropin(path, previous_exists, previous_content, journal)
        journal.mark_verification(ok=False, daemon_reload=False)
        journal.mark_rollback(
            ok=rollback_ok,
            attempted=True,
            summary=_telegram_dropin_rollback_summary(previous_exists, rollback_ok),
        )
        _persist_telegram_managed_action_journal(
            journal,
            managed_action_journal_store,
            status="rolled_back" if rollback_ok else "recovery_needed",
            recovery_hint="Confirm Telegram service config before retrying setup.",
        )
        return _telegram_service_failure_response(
            journal=journal,
            error_kind="telegram_daemon_reload_failed",
            detail="I could not reload systemd after changing the Telegram service config.",
            rollback_ok=rollback_ok,
            previous_exists=previous_exists,
        )

    state_after_write = get_telegram_runtime_state(env=env, home=home, run=run, secret_store_path=secret_store_path)
    service_installed = bool(state_after_write.get("service_installed", False))
    token_configured = bool(state_after_write.get("token_configured", False))
    service_action_attempted = False
    if enabled and service_installed and token_configured:
        service_action_attempted = True
        try:
            token, _token_source = resolve_telegram_token_with_source(
                env=env,
                home=home,
                secret_store_path=secret_store_path,
            )
        except Exception:
            token = None
        removed_locks = clear_stale_telegram_locks(token, env=env, home=home)
        journal.record_step("clear_stale_telegram_locks", ok=True, resource=TELEGRAM_SERVICE_NAME, removed_count=len(removed_locks))
        proc = _run_approved_systemctl_user(
            ["restart", TELEGRAM_SERVICE_NAME],
            journal=journal,
            step_name="systemctl_service_action",
            run=run,
        )
        if proc is None or proc.returncode != 0:
            return _recover_failed_telegram_service_action(
                journal=journal,
                path=path,
                previous_exists=previous_exists,
                previous_content=previous_content,
                state_before=state_before,
                run=run,
                managed_action_journal_store=managed_action_journal_store,
                error_kind="telegram_service_restart_failed",
                detail="I could not restart the Personal Agent Telegram service.",
            )
    elif not enabled and service_installed:
        service_action_attempted = True
        proc = _run_approved_systemctl_user(
            ["stop", TELEGRAM_SERVICE_NAME],
            journal=journal,
            step_name="systemctl_service_action",
            run=run,
        )
        if proc is None or proc.returncode != 0:
            return _recover_failed_telegram_service_action(
                journal=journal,
                path=path,
                previous_exists=previous_exists,
                previous_content=previous_content,
                state_before=state_before,
                run=run,
                managed_action_journal_store=managed_action_journal_store,
                error_kind="telegram_service_stop_failed",
                detail="I could not stop the Personal Agent Telegram service.",
            )
    else:
        journal.record_step(
            "systemctl_service_action",
            ok=True,
            resource=TELEGRAM_SERVICE_NAME,
            skipped=True,
            reason="service_not_installed" if not service_installed else "token_not_configured",
        )

    state_after = get_telegram_runtime_state(env=env, home=home, run=run, secret_store_path=secret_store_path)
    verification_ok = _telegram_service_verification_ok(
        enabled=enabled,
        state=state_after,
        service_action_attempted=service_action_attempted,
    )
    journal.record_step(
        "verify_telegram_runtime_state",
        ok=verification_ok,
        resource=TELEGRAM_SERVICE_NAME,
        current_state=_safe_status_snapshot(state_after),
        service_action_attempted=service_action_attempted,
    )
    if not verification_ok:
        return _recover_failed_telegram_service_action(
            journal=journal,
            path=path,
            previous_exists=previous_exists,
            previous_content=previous_content,
            state_before=state_before,
            run=run,
            managed_action_journal_store=managed_action_journal_store,
            error_kind="telegram_service_verification_failed",
            detail="I could not verify the Telegram service state after the change.",
        )

    journal.mark_verification(ok=True, current_state=_safe_status_snapshot(state_after), online_getme_required=False)
    journal.mark_rollback(ok=True, attempted=False, summary="No rollback needed.")
    _persist_telegram_managed_action_journal(journal, managed_action_journal_store, status="verified")
    return True, {
        "ok": True,
        "enabled": bool(enabled),
        "state": state_after,
        "message": "Telegram service setup finished." if enabled else "Telegram service disable finished.",
        "managed_action_journal": journal.to_dict(),
    }


def _telegram_service_verification_ok(
    *,
    enabled: bool,
    state: Mapping[str, Any],
    service_action_attempted: bool,
) -> bool:
    if enabled:
        if not bool(state.get("enabled", False)):
            return False
        if service_action_attempted:
            return bool(state.get("service_active", False)) and str(state.get("ready_state") or "") == "running"
        return True
    if bool(state.get("enabled", True)):
        return False
    if service_action_attempted and bool(state.get("service_active", False)):
        return False
    return True


def _recover_failed_telegram_service_action(
    *,
    journal: ManagedActionJournal,
    path: Path,
    previous_exists: bool,
    previous_content: str,
    state_before: Mapping[str, Any],
    run: Callable[..., subprocess.CompletedProcess[str]] | None,
    managed_action_journal_store: PersistentManagedActionJournalStore | str | Path | None = None,
    error_kind: str,
    detail: str,
) -> tuple[bool, dict[str, Any]]:
    rollback_ok = _rollback_telegram_dropin(path, previous_exists, previous_content, journal)
    _run_approved_systemctl_user(
        ["daemon-reload"],
        journal=journal,
        step_name="rollback_systemctl_daemon_reload",
        run=run,
    )
    previous_active = bool(state_before.get("service_active", False))
    if not previous_active:
        _run_approved_systemctl_user(
            ["stop", TELEGRAM_SERVICE_NAME],
            journal=journal,
            step_name="rollback_stop_service_if_not_previously_active",
            run=run,
        )
    summary = _telegram_dropin_rollback_summary(previous_exists, rollback_ok)
    if not rollback_ok:
        summary = f"{summary}; the Telegram service config may still need manual attention"
    journal.mark_verification(ok=False, error_kind=error_kind, previous_state=_safe_status_snapshot(state_before))
    journal.mark_rollback(ok=rollback_ok, attempted=True, summary=summary)
    _persist_telegram_managed_action_journal(
        journal,
        managed_action_journal_store,
        status="rolled_back" if rollback_ok else "recovery_needed",
        recovery_hint="Confirm Telegram service status and drop-in contents before retrying setup.",
    )
    return _telegram_service_failure_response(
        journal=journal,
        error_kind=error_kind,
        detail=detail,
        rollback_ok=rollback_ok,
        previous_exists=previous_exists,
    )


def _telegram_dropin_rollback_summary(previous_exists: bool, rollback_ok: bool) -> str:
    if previous_exists and rollback_ok:
        return "restored previous Telegram service config"
    if rollback_ok:
        return "removed failed new Telegram service config"
    return "could not restore Telegram service config automatically"


def _telegram_service_failure_response(
    *,
    journal: ManagedActionJournal,
    error_kind: str,
    detail: str,
    rollback_ok: bool,
    previous_exists: bool,
) -> tuple[bool, dict[str, Any]]:
    rollback_summary = _telegram_dropin_rollback_summary(previous_exists, rollback_ok)
    next_step = "Check the Telegram service status, then run python -m agent telegram_status."
    return False, {
        "ok": False,
        "error": error_kind,
        "error_kind": error_kind,
        "message": (
            f"Telegram setup/start did not finish: {detail} "
            f"{rollback_summary}. {next_step}"
        ),
        "rollback_ok": rollback_ok,
        "rollback_summary": rollback_summary,
        "next_step": next_step,
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


def _persist_telegram_managed_action_journal(
    journal: ManagedActionJournal,
    managed_action_journal_store: PersistentManagedActionJournalStore | str | Path | None,
    *,
    status: str,
    recovery_hint: str | None = None,
) -> None:
    try:
        store = (
            managed_action_journal_store
            if isinstance(managed_action_journal_store, PersistentManagedActionJournalStore)
            else PersistentManagedActionJournalStore(managed_action_journal_store)
        )
        store.upsert(journal, status=status, recovery_hint=recovery_hint)
    except Exception:
        pass


__all__ = [
    "TELEGRAM_SERVICE_NAME",
    "clear_stale_telegram_locks",
    "get_telegram_runtime_state",
    "inspect_telegram_pollers",
    "inspect_secret_store_status",
    "inspect_telegram_lock",
    "is_approved_telegram_systemctl_user_action",
    "read_telegram_enablement",
    "resolve_telegram_token_with_source",
    "is_personal_agent_telegram_dropin_path",
    "manage_telegram_service_state",
    "telegram_control_env",
    "telegram_dropin_path",
    "telegram_lock_paths",
    "write_telegram_enablement",
    "write_telegram_enablement_managed",
]
