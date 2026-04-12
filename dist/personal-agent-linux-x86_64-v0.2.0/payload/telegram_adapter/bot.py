from __future__ import annotations

"""Telegram transport adapter.

This module should stay transport-focused; core product/runtime decisions belong to the shared runtime contracts.
"""

import argparse
import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timezone, time
from zoneinfo import ZoneInfo
import http.client
import inspect
import json
import hashlib
import logging
import os
from pathlib import Path
import re
import socket
import sys
import time as pytime
import traceback
import uuid
from typing import Any, Callable
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlsplit

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
except ModuleNotFoundError:  # pragma: no cover - testing without telegram installed
    Update = object  # type: ignore
    Application = object  # type: ignore
    CommandHandler = object  # type: ignore
    ContextTypes = object  # type: ignore
    MessageHandler = object  # type: ignore
    filters = object()  # type: ignore

from agent.config import Config, load_config
from agent.doctor import run_doctor_report
from agent.error_response_ux import deterministic_error_message
from agent.fallback_ladder import run_with_fallback
from agent.llm_router import LLMRouter
from agent.logging_utils import log_event
from agent.scheduled_snapshots import (
    safe_run_scheduled_snapshot,
    safe_run_storage_snapshot,
    safe_run_resource_snapshot,
    safe_run_network_snapshot,
)
from agent.debug_protocol import DebugProtocol
from agent.orchestrator import Orchestrator
from agent.cards import render_cards_markdown, validate_cards_payload
from agent.daily_brief import should_send_daily_brief
from agent.model_scout import build_model_scout
from agent.audit_log import AuditLog
from agent.identity import get_public_identity
from agent.logging_bootstrap import configure_logging_if_needed
from agent.setup_wizard import render_telegram_setup_text
from agent.secret_store import SecretStore
from agent.setup_chat_flow import classify_runtime_chat_route
from agent.startup_checks import run_startup_checks
from agent.telegram_runtime_state import clear_stale_telegram_locks, get_telegram_runtime_state
from agent.telegram_bridge import (
    build_telegram_chat_api_payload,
    build_telegram_chat_payload_result,
    build_telegram_chat_proxy_error_result,
    build_telegram_help,
    build_telegram_setup,
    build_telegram_status,
    classify_telegram_text_command,
    handle_telegram_command,
)
from agent.permissions import PermissionStore
from agent.ux.llm_fixit_wizard import (
    LLMFixitWizardStore,
    OperatorRecoveryStore,
    confirm_token_for_plan_rows,
)
from memory.db import MemoryDB

_LOGGER = logging.getLogger(__name__)

_TELEGRAM_BOT_TOKEN_SECRET_KEY = "telegram:bot_token"
_TELEGRAM_FALLBACK_TEXT = "I hit an internal error, but I’m still running. Try one of these:"
_TELEGRAM_HELP_TEXT = (
    "Available commands:\n\n"
    "doctor – diagnostics\n"
    "setup – setup/recovery guidance\n"
    "status – runtime status\n"
    "health – health snapshot\n"
    "brief – system summary\n"
    "memory – what are we doing / resume"
)
_LOCAL_API_READY_TIMEOUT_SECONDS = 1.0
_LOCAL_API_CHAT_TIMEOUT_SECONDS = 10.0
_LOCAL_API_SETUP_CHAT_TIMEOUT_SECONDS = 15.0
_LOCAL_API_SETUP_EXECUTION_CHAT_TIMEOUT_SECONDS = 30.0
_NO_ACTIVE_CHOICE_TEXT = "No active choice right now. Say status, setup, or check models."
_DEFAULT_API_BASE_URL = "http://127.0.0.1:8765"


@dataclass(frozen=True)
class TelegramPollLock:
    fd: int
    path: str
    token_hash: str


def resolve_telegram_bot_token_with_source() -> tuple[str | None, str]:
    secret_store = SecretStore(path=os.getenv("AGENT_SECRET_STORE_PATH", "").strip() or None)
    secret_token = (secret_store.get_secret(_TELEGRAM_BOT_TOKEN_SECRET_KEY) or "").strip()
    if secret_token:
        return secret_token, "secret_store"
    env_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if env_token:
        return env_token, "env"
    return None, "missing"


def _resolve_telegram_bot_token() -> str | None:
    token, _source = resolve_telegram_bot_token_with_source()
    return token


def _safe_reply_text(text: str | None) -> str:
    value = str(text or "").strip()
    return value if value else "I’m still here. What should I do next?"


def _token_hash(token: str | None) -> str:
    value = str(token or "").strip()
    if not value:
        return "default"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def telegram_poll_lock_path(token: str | None, *, base_dir: Path | None = None) -> Path:
    env_dir = os.getenv("AGENT_TELEGRAM_POLL_LOCK_DIR", "").strip()
    root = base_dir or (Path(env_dir).expanduser() if env_dir else (Path.home() / ".local" / "share" / "personal-agent"))
    return root / f"telegram_poll.{_token_hash(token)}.lock"


def acquire_telegram_poll_lock(token: str | None, *, base_dir: Path | None = None) -> TelegramPollLock | None:
    primary_path = telegram_poll_lock_path(token, base_dir=base_dir)
    candidates = [primary_path]
    if base_dir is None:
        candidates.append(Path("/tmp") / "personal-agent" / primary_path.name)
    flags = os.O_RDWR | os.O_CREAT
    for lock_path in candidates:
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(lock_path), flags, 0o600)
        except PermissionError:
            continue
        except OSError:
            return None
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            return None
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
        except OSError:
            pass
        return TelegramPollLock(fd=fd, path=str(lock_path), token_hash=_token_hash(token))
    return None


def release_telegram_poll_lock(lock: TelegramPollLock | None) -> None:
    if lock is None:
        return
    try:
        import fcntl

        fcntl.flock(lock.fd, fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        os.close(lock.fd)
    except OSError:
        pass


def is_telegram_conflict_error(error: BaseException) -> bool:
    if error.__class__.__name__ == "Conflict":
        return True
    lowered = str(error or "").lower()
    if "terminated by other getupdates request" in lowered:
        return True
    return "getupdates" in lowered and "conflict" in lowered


def telegram_conflict_backoff_seconds(attempt: int) -> float:
    index = max(1, int(attempt))
    return float(2 + ((index - 1) % 5))


def _short_trace_token(trace_id: str | None) -> str:
    compact = "".join(ch for ch in str(trace_id or "") if ch.isalnum())
    if not compact:
        return "unknown"
    return compact[-6:]


def _text_prefix(text: str | None, *, limit: int = 80) -> str:
    normalized = " ".join(str(text or "").split()).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip()


def _normalize_user_text(text: str | None) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _contains_any(normalized_text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in normalized_text for phrase in phrases)


def _truncate_telegram_text(text: str, *, max_len: int = 3900) -> tuple[str, bool]:
    value = _safe_reply_text(text)
    if len(value) <= int(max_len):
        return value, False
    suffix = "… (truncated)"
    keep = max(0, int(max_len) - len(suffix))
    trimmed = value[:keep].rstrip()
    return f"{trimmed}{suffix}", True


def _plain_text_retry_variant(text: str) -> str:
    value = str(text or "")
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    chars: list[str] = []
    for ch in value:
        code = ord(ch)
        if ch in {"\n", "\t"} or code >= 32:
            chars.append(ch)
        else:
            chars.append(" ")
    plain = "".join(chars).strip()
    return plain or "Message unavailable."


def _is_bad_request_error(exc: Exception) -> bool:
    class_name = exc.__class__.__name__.lower()
    if class_name == "badrequest" or class_name.endswith("badrequest"):
        return True
    lowered = str(exc or "").lower()
    if "bad request" in lowered:
        return True
    return "can't parse entities" in lowered


async def _send_reply(
    *,
    message: Any,
    log_path: str | None,
    chat_id: str,
    route: str,
    text: str,
    trace_id: str,
    parse_mode: str | None = None,
    **kwargs: Any,
) -> str:
    primary_text, primary_truncated = _truncate_telegram_text(text)
    primary_kwargs = dict(kwargs)
    if parse_mode is not None:
        primary_kwargs["parse_mode"] = parse_mode

    delivered_text = primary_text
    send_fallback = False

    try:
        await message.reply_text(primary_text, **primary_kwargs)
    except Exception as exc:
        if _is_bad_request_error(exc):
            _LOGGER.error(
                "telegram.reply.bad_request %s",
                json.dumps(
                    {
                        "trace_id": trace_id,
                        "chat_id": chat_id,
                        "msg_len": len(primary_text),
                        "parse_mode": str(parse_mode or "none"),
                        "error": str(exc),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
            )
            lowered_error = str(exc or "").lower()
            plain_retry = bool(parse_mode is not None) or "parse entities" in lowered_error
            retry_base = _plain_text_retry_variant(primary_text) if plain_retry else primary_text
            retry_text, retry_truncated = _truncate_telegram_text(retry_base)
            retry_kwargs = dict(kwargs)
            retry_kwargs["parse_mode"] = None
            retry_kwargs["disable_web_page_preview"] = True
            try:
                await message.reply_text(retry_text, **retry_kwargs)
            except TypeError:
                retry_kwargs.pop("disable_web_page_preview", None)
                await message.reply_text(retry_text, **retry_kwargs)
            except Exception as retry_exc:
                _LOGGER.error(
                    "telegram.reply.error %s",
                    json.dumps(
                        {
                            "trace_id": trace_id,
                            "chat_id": chat_id,
                            "msg_len": len(retry_text),
                            "parse_mode": "none",
                            "error": str(retry_exc),
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
                )
                _LOGGER.error("%s", traceback.format_exc())
                raise
            delivered_text = retry_text
            send_fallback = True
            primary_truncated = primary_truncated or retry_truncated
        else:
            _LOGGER.error(
                "telegram.reply.error %s",
                json.dumps(
                    {
                        "trace_id": trace_id,
                        "chat_id": chat_id,
                        "msg_len": len(primary_text),
                        "parse_mode": str(parse_mode or "none"),
                        "error": str(exc),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
            )
            _LOGGER.error("%s", traceback.format_exc())
            raise

    log_event(
        log_path,
        "telegram.out",
        {
            "user_id": chat_id,
            "chat_id": chat_id,
            "route": route,
            "reply_prefix": _text_prefix(delivered_text),
            "trace_id": trace_id,
            "msg_len": len(delivered_text),
            "send_fallback": bool(send_fallback),
            "truncated": bool(primary_truncated),
        },
    )
    _LOGGER.info(
        "telegram.out %s",
        json.dumps(
            {
                "trace_id": trace_id,
                "user_id": chat_id,
                "chat_id": chat_id,
                "route": route,
                "reply_prefix": _text_prefix(delivered_text),
                "msg_len": len(delivered_text),
                "send_fallback": bool(send_fallback),
                "truncated": bool(primary_truncated),
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
    )
    log_event(
        log_path,
        "response_sent",
        {
            "user_id": chat_id,
            "chat_id": chat_id,
            "route": route,
            "reply_prefix": _text_prefix(delivered_text),
            "trace_id": trace_id,
        },
    )
    return delivered_text


async def _send_placeholder_message(
    message: Any,
    text: str,
    *,
    trace_id: str | None = None,
    chat_id: str | None = None,
) -> Any | None:
    primary_text, _ = _truncate_telegram_text(text)
    try:
        return await message.reply_text(primary_text)
    except Exception as exc:
        if not _is_bad_request_error(exc):
            raise
        _LOGGER.error(
            "telegram.reply.bad_request %s",
            json.dumps(
                {
                    "trace_id": str(trace_id or "").strip() or None,
                    "chat_id": str(chat_id or "").strip() or None,
                    "msg_len": len(primary_text),
                    "parse_mode": "none",
                    "error": str(exc),
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
        )
        retry_text, _ = _truncate_telegram_text(_plain_text_retry_variant(primary_text))
        try:
            return await message.reply_text(retry_text, parse_mode=None, disable_web_page_preview=True)
        except TypeError:
            return await message.reply_text(retry_text, parse_mode=None)


async def _edit_reply(
    *,
    message: Any,
    log_path: str | None,
    chat_id: str,
    route: str,
    text: str,
    trace_id: str,
    parse_mode: str | None = None,
    **kwargs: Any,
) -> str:
    primary_text, primary_truncated = _truncate_telegram_text(text)
    edit_kwargs = dict(kwargs)
    if parse_mode is not None:
        edit_kwargs["parse_mode"] = parse_mode
    delivered_text = primary_text
    edit_fallback = False
    try:
        await message.edit_text(primary_text, **edit_kwargs)
    except Exception as exc:
        if _is_bad_request_error(exc):
            retry_text, retry_truncated = _truncate_telegram_text(_plain_text_retry_variant(primary_text))
            retry_kwargs = dict(kwargs)
            retry_kwargs["parse_mode"] = None
            retry_kwargs["disable_web_page_preview"] = True
            try:
                await message.edit_text(retry_text, **retry_kwargs)
            except TypeError:
                retry_kwargs.pop("disable_web_page_preview", None)
                await message.edit_text(retry_text, **retry_kwargs)
            delivered_text = retry_text
            edit_fallback = True
            primary_truncated = primary_truncated or retry_truncated
        else:
            raise
    log_event(
        log_path,
        "telegram.out",
        {
            "user_id": chat_id,
            "chat_id": chat_id,
            "route": route,
            "reply_prefix": _text_prefix(delivered_text),
            "trace_id": trace_id,
            "msg_len": len(delivered_text),
            "send_fallback": bool(edit_fallback),
            "truncated": bool(primary_truncated),
            "delivery_mode": "edit",
        },
    )
    log_event(
        log_path,
        "response_sent",
        {
            "user_id": chat_id,
            "chat_id": chat_id,
            "route": route,
            "reply_prefix": _text_prefix(delivered_text),
            "trace_id": trace_id,
            "delivery_mode": "edit",
        },
    )
    _LOGGER.info(
        "telegram.out %s",
        json.dumps(
            {
                "trace_id": trace_id,
                "user_id": chat_id,
                "chat_id": chat_id,
                "route": route,
                "reply_prefix": _text_prefix(delivered_text),
                "msg_len": len(delivered_text),
                "send_fallback": bool(edit_fallback),
                "truncated": bool(primary_truncated),
                "delivery_mode": "edit",
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
    )
    return delivered_text


def _telegram_background_tasks(bot_data: dict[str, Any]) -> set[asyncio.Task[Any]]:
    tasks = bot_data.get("_background_tasks")
    if isinstance(tasks, set):
        return tasks
    tasks = set()
    bot_data["_background_tasks"] = tasks
    return tasks


def _emit_telegram_runtime_event(
    *,
    bot_data: dict[str, Any],
    log_path: str | None,
    event_name: str,
    **fields: Any,
) -> None:
    runtime = bot_data.get("runtime")
    record_runtime_event = getattr(runtime, "record_runtime_event", None)
    if callable(record_runtime_event):
        try:
            record_runtime_event(event_name, **fields)
        except Exception:
            pass
    log_event(log_path, event_name, fields)


async def _deliver_telegram_text_result(
    *,
    placeholder_message: Any | None,
    original_message: Any,
    log_path: str | None,
    chat_id: str,
    route: str,
    text: str,
    trace_id: str,
) -> None:
    if placeholder_message is not None and callable(getattr(placeholder_message, "edit_text", None)):
        await _edit_reply(
            message=placeholder_message,
            log_path=log_path,
            chat_id=chat_id,
            route=route,
            text=text,
            trace_id=trace_id,
        )
        return
    await _send_reply(
        message=original_message,
        log_path=log_path,
        chat_id=chat_id,
        route=route,
        text=text,
        trace_id=trace_id,
    )


def _format_commit_short(value: str | None) -> str:
    commit = str(value or "").strip()
    if not commit:
        return "unknown"
    return commit[:12]


def _api_base_url() -> str:
    configured = str(os.getenv("AGENT_API_BASE_URL") or os.getenv("PERSONAL_AGENT_API_BASE_URL") or "").strip()
    if configured:
        return configured.rstrip("/")
    return _DEFAULT_API_BASE_URL


def _fetch_local_api_json(path: str, *, timeout_seconds: float = 0.6) -> dict[str, Any]:
    endpoint = str(path or "").strip()
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"
    url = f"{_api_base_url()}{endpoint}"
    request = urllib_request.Request(url=url, method="GET")
    try:
        with urllib_request.urlopen(request, timeout=float(timeout_seconds)) as response:
            body = response.read()
    except (urllib_error.URLError, TimeoutError, OSError):
        return {}
    except Exception:
        return {}
    try:
        decoded = body.decode("utf-8", errors="replace")
        payload = json.loads(decoded)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _local_api_health_snapshot(*, timeout_seconds: float = _LOCAL_API_READY_TIMEOUT_SECONDS) -> dict[str, Any]:
    ready = _fetch_local_api_json("/ready", timeout_seconds=float(timeout_seconds))
    phase = str(ready.get("phase") or "").strip() or None if isinstance(ready, dict) else None
    return {
        "reachable": bool(ready),
        "ok": bool(ready.get("ok", False)) if isinstance(ready, dict) else False,
        "ready": bool(ready.get("ready", False)) if isinstance(ready, dict) else False,
        "phase": phase,
    }


def _chat_proxy_error_payload(
    *,
    kind: str,
    detail: str | None = None,
    status_code: int | None = None,
    elapsed_ms: int | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    health = _local_api_health_snapshot(timeout_seconds=_LOCAL_API_READY_TIMEOUT_SECONDS)
    return {
        "_proxy_error": {
            "kind": str(kind or "proxy_error").strip().lower() or "proxy_error",
            "detail": str(detail or "").strip() or None,
            "status_code": int(status_code) if status_code is not None else None,
            "elapsed_ms": int(elapsed_ms) if elapsed_ms is not None else None,
            "timeout_seconds": float(timeout_seconds) if timeout_seconds is not None else None,
            "backend_reachable": bool(health.get("reachable", False)),
            "backend_ok": bool(health.get("ok", False)),
            "backend_ready": bool(health.get("ready", False)),
            "backend_phase": str(health.get("phase") or "").strip() or None,
        }
    }


def _classify_chat_proxy_failure(exc: BaseException) -> tuple[str, str | None]:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return "timeout", str(exc or "").strip() or "request timed out"
    if isinstance(
        exc,
        (
            BrokenPipeError,
            ConnectionAbortedError,
            ConnectionResetError,
            asyncio.IncompleteReadError,
            http.client.RemoteDisconnected,
            http.client.IncompleteRead,
        ),
    ):
        return "disconnect", str(exc or "").strip() or exc.__class__.__name__
    if isinstance(exc, urllib_error.URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return "timeout", str(reason or "").strip() or "request timed out"
        if isinstance(
            reason,
            (
                BrokenPipeError,
                ConnectionAbortedError,
                ConnectionResetError,
                asyncio.IncompleteReadError,
                http.client.RemoteDisconnected,
                http.client.IncompleteRead,
            ),
        ):
            return "disconnect", str(reason or "").strip() or reason.__class__.__name__
        lowered = str(reason or exc).strip().lower()
        if "timed out" in lowered:
            return "timeout", lowered
        if any(token in lowered for token in ("broken pipe", "remote end closed", "connection aborted", "connection reset")):
            return "disconnect", lowered
        return "unreachable", str(reason or exc or "").strip() or "connection failed"
    if isinstance(exc, OSError):
        lowered = str(exc or "").strip().lower()
        if any(token in lowered for token in ("timed out", "timeout")):
            return "timeout", lowered
        if any(token in lowered for token in ("broken pipe", "connection aborted", "connection reset", "remote end closed")):
            return "disconnect", lowered
        return "unreachable", str(exc or "").strip() or "connection failed"
    return "proxy_error", str(exc or "").strip() or exc.__class__.__name__


def _chat_proxy_timeout_seconds(payload: dict[str, Any]) -> float:
    setup_state_hint = payload.get("setup_state_hint") if isinstance(payload.get("setup_state_hint"), dict) else {}
    hint_step = str(setup_state_hint.get("step") or "").strip().lower()
    awaiting_secret = bool(setup_state_hint.get("awaiting_secret")) or hint_step == "awaiting_openrouter_key"
    awaiting_confirmation = bool(setup_state_hint.get("awaiting_confirmation")) or hint_step in {
        "awaiting_switch_confirm",
        "awaiting_openrouter_reuse_confirm",
    }
    if awaiting_confirmation:
        return _LOCAL_API_SETUP_EXECUTION_CHAT_TIMEOUT_SECONDS
    if awaiting_secret:
        return _LOCAL_API_SETUP_CHAT_TIMEOUT_SECONDS
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    user_text = ""
    for row in reversed(messages):
        if not isinstance(row, dict):
            continue
        if str(row.get("role") or "").strip().lower() != "user":
            continue
        user_text = str(row.get("content") or "").strip()
        if user_text:
            break
    route = classify_runtime_chat_route(user_text)
    route_name = str(route.get("route") or "generic_chat").strip().lower() or "generic_chat"
    if route_name == "setup_flow":
        return _LOCAL_API_SETUP_CHAT_TIMEOUT_SECONDS
    return _LOCAL_API_CHAT_TIMEOUT_SECONDS


def _decode_chat_proxy_response_body(
    body: bytes,
    *,
    status_code: int | None,
    elapsed_ms: int,
    timeout_seconds: float | None,
    execution_mode: str | None = None,
) -> dict[str, Any]:
    try:
        decoded = body.decode("utf-8", errors="replace")
        result = json.loads(decoded)
    except Exception:
        return _chat_proxy_error_payload(
            kind="invalid_response",
            detail="invalid_json_response",
            status_code=status_code,
            elapsed_ms=elapsed_ms,
            timeout_seconds=timeout_seconds,
        )
    if isinstance(result, dict):
        proxy_meta: dict[str, Any] = {
            "elapsed_ms": elapsed_ms,
            "timeout_seconds": timeout_seconds,
        }
        if execution_mode:
            proxy_meta["execution_mode"] = execution_mode
        if status_code is not None:
            proxy_meta["status_code"] = status_code
        result["_proxy_meta"] = proxy_meta
        return result
    return _chat_proxy_error_payload(
        kind="invalid_response",
        detail="non_object_json_response",
        status_code=status_code,
        elapsed_ms=elapsed_ms,
        timeout_seconds=timeout_seconds,
    )


def _post_local_api_chat_json(payload: dict[str, Any], *, timeout_seconds: float | None = None) -> dict[str, Any]:
    url = f"{_api_base_url()}/chat"
    body_bytes = json.dumps(payload if isinstance(payload, dict) else {}, ensure_ascii=True).encode("utf-8")
    request = urllib_request.Request(
        url=url,
        data=body_bytes,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    effective_timeout = float(timeout_seconds) if timeout_seconds is not None else _chat_proxy_timeout_seconds(payload)
    started_at = pytime.monotonic()
    try:
        with urllib_request.urlopen(request, timeout=effective_timeout) as response:
            status_code = int(getattr(response, "status", 200) or 200)
            body = response.read()
    except urllib_error.HTTPError as exc:
        elapsed_ms = int(max(0.0, pytime.monotonic() - started_at) * 1000)
        try:
            error_body = exc.read()
        except Exception:
            error_body = b""
        if error_body:
            return _decode_chat_proxy_response_body(
                error_body,
                status_code=int(exc.code),
                elapsed_ms=elapsed_ms,
                timeout_seconds=effective_timeout,
            )
        return _chat_proxy_error_payload(
            kind="http_error",
            detail=str(exc.reason or exc).strip() or "http error",
            status_code=int(exc.code),
            elapsed_ms=elapsed_ms,
            timeout_seconds=effective_timeout,
        )
    except Exception as exc:
        elapsed_ms = int(max(0.0, pytime.monotonic() - started_at) * 1000)
        kind, detail = _classify_chat_proxy_failure(exc)
        return _chat_proxy_error_payload(
            kind=kind,
            detail=detail,
            elapsed_ms=elapsed_ms,
            timeout_seconds=effective_timeout,
        )
    elapsed_ms = int(max(0.0, pytime.monotonic() - started_at) * 1000)
    return _decode_chat_proxy_response_body(
        body,
        status_code=status_code,
        elapsed_ms=elapsed_ms,
        timeout_seconds=effective_timeout,
    )


def _local_api_request_parts(endpoint: str) -> tuple[str, int, str, str]:
    path = str(endpoint or "").strip()
    if not path.startswith("/"):
        path = f"/{path}"
    parsed = urlsplit(f"{_api_base_url()}{path}")
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or (443 if parsed.scheme == "https" else 80))
    request_path = parsed.path or "/"
    if parsed.query:
        request_path = f"{request_path}?{parsed.query}"
    host_header = parsed.netloc or host
    return host, port, request_path, host_header


async def _await_with_optional_timeout(awaitable: Any, timeout_seconds: float | None) -> Any:
    if timeout_seconds is None:
        return await awaitable
    return await asyncio.wait_for(awaitable, timeout=timeout_seconds)


async def _read_http_response_async(
    *,
    reader: asyncio.StreamReader,
    timeout_seconds: float | None,
) -> tuple[int, dict[str, str], bytes]:
    status_line = await _await_with_optional_timeout(reader.readline(), timeout_seconds)
    if not status_line:
        raise ConnectionResetError("empty http response")
    try:
        decoded_status = status_line.decode("iso-8859-1", errors="replace").strip()
        _version, status_code, _reason = decoded_status.split(" ", 2)
        code = int(status_code)
    except Exception as exc:  # pragma: no cover - defensive parsing
        raise ConnectionResetError("invalid http status line") from exc

    headers: dict[str, str] = {}
    while True:
        line = await _await_with_optional_timeout(reader.readline(), timeout_seconds)
        if line in {b"", b"\r\n", b"\n"}:
            break
        decoded_line = line.decode("iso-8859-1", errors="replace")
        name, separator, value = decoded_line.partition(":")
        if separator:
            headers[name.strip().lower()] = value.strip()

    if "content-length" in headers:
        length = max(0, int(headers.get("content-length") or 0))
        body = await _await_with_optional_timeout(reader.readexactly(length), timeout_seconds)
        return code, headers, body

    if "chunked" in str(headers.get("transfer-encoding") or "").lower():
        chunks: list[bytes] = []
        while True:
            size_line = await _await_with_optional_timeout(reader.readline(), timeout_seconds)
            if not size_line:
                raise ConnectionResetError("incomplete chunked response")
            size_text = size_line.decode("iso-8859-1", errors="replace").split(";", 1)[0].strip()
            chunk_size = int(size_text, 16)
            if chunk_size == 0:
                await _await_with_optional_timeout(reader.readline(), timeout_seconds)
                break
            chunk = await _await_with_optional_timeout(reader.readexactly(chunk_size), timeout_seconds)
            chunks.append(chunk)
            await _await_with_optional_timeout(reader.readexactly(2), timeout_seconds)
        return code, headers, b"".join(chunks)

    body = await _await_with_optional_timeout(reader.read(), timeout_seconds)
    return code, headers, body


async def _fetch_local_api_json_async(path: str, *, timeout_seconds: float = 0.6) -> dict[str, Any]:
    host, port, request_path, host_header = _local_api_request_parts(path)
    writer: asyncio.StreamWriter | None = None
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout_seconds)
        request_bytes = (
            f"GET {request_path} HTTP/1.1\r\n"
            f"Host: {host_header}\r\n"
            "Accept: application/json\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode("ascii")
        writer.write(request_bytes)
        await asyncio.wait_for(writer.drain(), timeout=timeout_seconds)
        status_code, _headers, body = await _read_http_response_async(reader=reader, timeout_seconds=timeout_seconds)
        if status_code >= 400:
            return {}
        decoded = body.decode("utf-8", errors="replace")
        payload = json.loads(decoded)
    except Exception:
        return {}
    finally:
        if writer is not None:
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), timeout=0.2)
            except Exception:
                pass
    return payload if isinstance(payload, dict) else {}


async def _local_api_health_snapshot_async(
    *,
    timeout_seconds: float = _LOCAL_API_READY_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    ready = await _fetch_local_api_json_async("/ready", timeout_seconds=float(timeout_seconds))
    phase = str(ready.get("phase") or "").strip() or None if isinstance(ready, dict) else None
    return {
        "reachable": bool(ready),
        "ok": bool(ready.get("ok", False)) if isinstance(ready, dict) else False,
        "ready": bool(ready.get("ready", False)) if isinstance(ready, dict) else False,
        "phase": phase,
    }


async def _chat_proxy_error_payload_async(
    *,
    kind: str,
    detail: str | None = None,
    status_code: int | None = None,
    elapsed_ms: int | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    health = await _local_api_health_snapshot_async(timeout_seconds=_LOCAL_API_READY_TIMEOUT_SECONDS)
    return {
        "_proxy_error": {
            "kind": str(kind or "proxy_error").strip().lower() or "proxy_error",
            "detail": str(detail or "").strip() or None,
            "status_code": int(status_code) if status_code is not None else None,
            "elapsed_ms": int(elapsed_ms) if elapsed_ms is not None else None,
            "timeout_seconds": float(timeout_seconds) if timeout_seconds is not None else None,
            "backend_reachable": bool(health.get("reachable", False)),
            "backend_ok": bool(health.get("ok", False)),
            "backend_ready": bool(health.get("ready", False)),
            "backend_phase": str(health.get("phase") or "").strip() or None,
        }
    }


async def _post_local_api_chat_json_async(
    payload: dict[str, Any],
    *,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    host, port, request_path, host_header = _local_api_request_parts("/chat")
    body_bytes = json.dumps(payload if isinstance(payload, dict) else {}, ensure_ascii=True).encode("utf-8")
    effective_timeout = float(timeout_seconds) if timeout_seconds is not None else None
    started_at = pytime.monotonic()
    writer: asyncio.StreamWriter | None = None
    try:
        reader, writer = await asyncio.open_connection(host, port)
        request_bytes = (
            f"POST {request_path} HTTP/1.1\r\n"
            f"Host: {host_header}\r\n"
            "Content-Type: application/json\r\n"
            "Accept: application/json\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode("ascii") + body_bytes
        writer.write(request_bytes)
        await writer.drain()
        status_code, _headers, body = await _read_http_response_async(reader=reader, timeout_seconds=effective_timeout)
    except Exception as exc:
        elapsed_ms = int(max(0.0, pytime.monotonic() - started_at) * 1000)
        kind, detail = _classify_chat_proxy_failure(exc)
        return await _chat_proxy_error_payload_async(
            kind=kind,
            detail=detail,
            elapsed_ms=elapsed_ms,
            timeout_seconds=effective_timeout,
        )
    finally:
        if writer is not None:
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), timeout=0.2)
            except Exception:
                pass

    elapsed_ms = int(max(0.0, pytime.monotonic() - started_at) * 1000)
    result = _decode_chat_proxy_response_body(
        body,
        status_code=status_code,
        elapsed_ms=elapsed_ms,
        timeout_seconds=effective_timeout,
        execution_mode="async_http",
    )
    if status_code >= 400 and not (isinstance(result, dict) and result and "_proxy_error" not in result):
        return await _chat_proxy_error_payload_async(
            kind="http_error",
            detail=f"http {status_code}",
            status_code=status_code,
            elapsed_ms=elapsed_ms,
            timeout_seconds=effective_timeout,
        )
    return result


def _telegram_chat_proxy_state(bot_data: dict[str, Any]) -> dict[str, Any]:
    state = bot_data.get("_telegram_chat_proxy_state")
    if not isinstance(state, dict):
        state = {}
        bot_data["_telegram_chat_proxy_state"] = state
    in_flight_by_chat = state.get("in_flight_by_chat")
    if not isinstance(in_flight_by_chat, dict):
        in_flight_by_chat = {}
        state["in_flight_by_chat"] = in_flight_by_chat
    setup_hint_by_chat = state.get("setup_hint_by_chat")
    if not isinstance(setup_hint_by_chat, dict):
        setup_hint_by_chat = {}
        state["setup_hint_by_chat"] = setup_hint_by_chat
    if not isinstance(state.get("in_flight_total"), int):
        state["in_flight_total"] = 0
    return state


def _normalize_setup_state_hint(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    step = str(value.get("step") or "").strip().lower()
    awaiting_secret = bool(value.get("awaiting_secret")) or step == "awaiting_openrouter_key"
    awaiting_confirmation = bool(value.get("awaiting_confirmation")) or step in {
        "awaiting_switch_confirm",
        "awaiting_openrouter_reuse_confirm",
    }
    if not awaiting_secret and not awaiting_confirmation:
        return None
    return {
        "route": "setup_flow",
        "step": step or None,
        "awaiting_secret": awaiting_secret,
        "awaiting_confirmation": awaiting_confirmation,
    }


def _setup_state_hint_from_chat_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    setup = payload.get("setup") if isinstance(payload.get("setup"), dict) else {}
    route = str(((payload.get("meta") if isinstance(payload.get("meta"), dict) else {}) or {}).get("route") or "").strip().lower()
    setup_type = str(setup.get("type") or "").strip().lower()
    if route != "setup_flow":
        return None
    if setup_type == "request_secret":
        return {
            "route": "setup_flow",
            "step": "awaiting_openrouter_key",
            "awaiting_secret": True,
            "awaiting_confirmation": False,
        }
    if setup_type == "confirm_switch_model":
        return {
            "route": "setup_flow",
            "step": "awaiting_switch_confirm",
            "awaiting_secret": False,
            "awaiting_confirmation": True,
        }
    if setup_type == "confirm_reuse_secret":
        return {
            "route": "setup_flow",
            "step": "awaiting_openrouter_reuse_confirm",
            "awaiting_secret": False,
            "awaiting_confirmation": True,
        }
    return None


async def _maybe_await_result(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _handle_telegram_text_via_local_api(
    *,
    text: str,
    chat_id: str,
    trace_id: str,
    bot_data: dict[str, Any],
    log_path: str | None,
    runtime: Any | None,
    orchestrator: Any | None,
    runtime_version: str | None,
    runtime_git_commit: str | None,
    runtime_started_ts: float | int | None,
) -> dict[str, Any]:
    _ = runtime_version
    _ = runtime_git_commit
    _ = runtime_started_ts
    command = classify_telegram_text_command(text)
    if command is not None:
        return handle_telegram_command(
            command=command,
            chat_id=chat_id,
            trace_id=trace_id,
            runtime=runtime,
            orchestrator=orchestrator,
            runtime_version=str(bot_data.get("runtime_version") or "").strip() or None,
            runtime_git_commit=str(bot_data.get("runtime_git_commit") or "").strip() or None,
            runtime_started_ts=bot_data.get("runtime_started_ts"),
            fetch_local_api_json=_fetch_local_api_json,
        )
    chat_proxy_fetch = (
        bot_data.get("fetch_local_api_chat_json")
        if callable(bot_data.get("fetch_local_api_chat_json"))
        else _post_local_api_chat_json_async
    )
    state = _telegram_chat_proxy_state(bot_data)
    chat_key = str(chat_id or "").strip() or "unknown"
    request_id = uuid.uuid4().hex
    in_flight_by_chat = (
        state.get("in_flight_by_chat") if isinstance(state.get("in_flight_by_chat"), dict) else {}
    )
    setup_hint_by_chat = (
        state.get("setup_hint_by_chat") if isinstance(state.get("setup_hint_by_chat"), dict) else {}
    )
    setup_state_hint = _normalize_setup_state_hint(setup_hint_by_chat.get(chat_key))
    if int(in_flight_by_chat.get(chat_key) or 0) > 0:
        message = "I'm still working on your last request. Please wait for that reply or try again in a moment."
        log_event(
            log_path,
            "telegram.local_api_chat.rejected",
            {
                "trace_id": trace_id,
                "request_id": request_id,
                "chat_id_redacted": _redact_chat_id(chat_id),
                "selected_route_hint": "chat_busy",
                "execution_mode": "async_http",
                "overlap_policy": "reject_same_chat_in_flight",
                "overlap_rejected": True,
                "failure_kind": "same_chat_in_flight",
                "in_flight_count": int(in_flight_by_chat.get(chat_key) or 0),
                "in_flight_total": int(state.get("in_flight_total") or 0),
                "elapsed_ms": 0,
                "timeout_used": None,
            },
        )
        return {
            "ok": False,
            "handled": True,
            "text": message,
            "route": "chat_busy",
            "trace_id": trace_id,
            "request_id": request_id,
            "next_action": None,
            "selected_route": "chat_busy",
            "handler_name": "api_chat_proxy_busy",
            "used_llm": False,
            "used_memory": False,
            "used_runtime_state": False,
            "used_tools": [],
            "legacy_compatibility": False,
            "generic_fallback_used": False,
            "generic_fallback_reason": "same_chat_in_flight",
            "chat_meta": {
                "proxy_execution_mode": "async_http",
                "proxy_failure_kind": "same_chat_in_flight",
                "proxy_elapsed_ms": 0,
                "proxy_in_flight_count": int(in_flight_by_chat.get(chat_key) or 0),
                "proxy_in_flight_total": int(state.get("in_flight_total") or 0),
                "proxy_overlap_rejected": True,
                "proxy_timeout_used": None,
                "runtime_state_failure_reason": "same_chat_in_flight",
            },
        }

    state["in_flight_total"] = int(state.get("in_flight_total") or 0) + 1
    in_flight_by_chat[chat_key] = int(in_flight_by_chat.get(chat_key) or 0) + 1
    in_flight_total = int(state.get("in_flight_total") or 0)
    route_hint = classify_runtime_chat_route(
        text,
        awaiting_secret=bool((setup_state_hint or {}).get("awaiting_secret")),
        awaiting_confirmation=bool((setup_state_hint or {}).get("awaiting_confirmation")),
    )
    selected_route_hint = str(route_hint.get("route") or "generic_chat").strip().lower() or "generic_chat"
    call_started = pytime.monotonic()
    log_event(
        log_path,
        "telegram.local_api_chat.start",
            {
                "trace_id": trace_id,
                "request_id": request_id,
                "chat_id_redacted": _redact_chat_id(chat_id),
                "selected_route_hint": selected_route_hint,
                "execution_mode": "async_http",
                "overlap_policy": "reject_same_chat_in_flight",
                "overlap_rejected": False,
                "in_flight_total": in_flight_total,
                "in_flight_chat": int(in_flight_by_chat.get(chat_key) or 0),
                "setup_state_hint": dict(setup_state_hint) if isinstance(setup_state_hint, dict) else None,
            },
        )
    result: dict[str, Any] | None = None
    try:
        payload = build_telegram_chat_api_payload(
            text=text,
            chat_id=chat_id,
            trace_id=trace_id,
            request_id=request_id,
            setup_state_hint=setup_state_hint,
        )
        proxy_payload = await _maybe_await_result(chat_proxy_fetch(payload))
        proxy_error = (
            proxy_payload.get("_proxy_error")
            if isinstance(proxy_payload, dict) and isinstance(proxy_payload.get("_proxy_error"), dict)
            else None
        )
        proxy_meta = (
            proxy_payload.get("_proxy_meta")
            if isinstance(proxy_payload, dict) and isinstance(proxy_payload.get("_proxy_meta"), dict)
            else {}
        )
        if isinstance(proxy_error, dict):
            result = build_telegram_chat_proxy_error_result(
                proxy_error,
                trace_id=trace_id,
                proxy_meta=proxy_meta,
            )
        elif isinstance(proxy_payload, dict) and proxy_payload:
            result = build_telegram_chat_payload_result(
                proxy_payload,
                trace_id=trace_id,
                ok=bool(proxy_payload.get("ok", True)),
                handler_name="api_chat_proxy",
                legacy_compatibility=False,
            )
            next_setup_hint = _setup_state_hint_from_chat_payload(proxy_payload)
            if next_setup_hint is not None:
                setup_hint_by_chat[chat_key] = next_setup_hint
            elif bool(proxy_payload.get("ok", True)):
                setup_hint_by_chat.pop(chat_key, None)
        else:
            result = build_telegram_chat_proxy_error_result(
                {"kind": "invalid_response"},
                trace_id=trace_id,
            )
        if isinstance(result, dict):
            chat_meta = result.get("chat_meta") if isinstance(result.get("chat_meta"), dict) else {}
            chat_meta = {
                **chat_meta,
                "proxy_execution_mode": "async_http",
                "proxy_in_flight_count": int(in_flight_by_chat.get(chat_key) or 0),
                "proxy_in_flight_total": in_flight_total,
                "proxy_overlap_rejected": False,
                "proxy_timeout_used": (
                    float(chat_meta.get("proxy_timeout_seconds"))
                    if isinstance(chat_meta.get("proxy_timeout_seconds"), (int, float))
                    else None
                ),
                "proxy_total_elapsed_ms": int(max(0.0, pytime.monotonic() - call_started) * 1000),
                "proxy_overlap_policy": "reject_same_chat_in_flight",
            }
            result["chat_meta"] = chat_meta
        return result if isinstance(result, dict) else {}
    finally:
        elapsed_ms = int(max(0.0, pytime.monotonic() - call_started) * 1000)
        state["in_flight_total"] = max(0, int(state.get("in_flight_total") or 0) - 1)
        current_chat_in_flight = max(0, int(in_flight_by_chat.get(chat_key) or 0) - 1)
        if current_chat_in_flight:
            in_flight_by_chat[chat_key] = current_chat_in_flight
        else:
            in_flight_by_chat.pop(chat_key, None)
        log_event(
            log_path,
            "telegram.local_api_chat.finish",
                {
                    "trace_id": trace_id,
                    "request_id": request_id,
                    "chat_id_redacted": _redact_chat_id(chat_id),
                    "selected_route": (
                        str((result or {}).get("selected_route") or (result or {}).get("route") or selected_route_hint)
                        .strip()
                        .lower()
                        or selected_route_hint
                    ),
                    "execution_mode": "async_http",
                    "elapsed_ms": elapsed_ms,
                    "overlap_policy": "reject_same_chat_in_flight",
                    "overlap_rejected": False,
                    "failure_kind": (
                        str(((result or {}).get("chat_meta") or {}).get("proxy_failure_kind") or "").strip() or None
                    ),
                    "timeout_used": (
                        float(((result or {}).get("chat_meta") or {}).get("proxy_timeout_used"))
                        if isinstance(((result or {}).get("chat_meta") or {}).get("proxy_timeout_used"), (int, float))
                        else None
                    ),
                    "in_flight_count": int(current_chat_in_flight),
                    "in_flight_total": int(state.get("in_flight_total") or 0),
                    "ok": bool((result or {}).get("ok", False)),
                    "setup_state_hint": dict(setup_state_hint) if isinstance(setup_state_hint, dict) else None,
                },
        )


async def _run_async_telegram_chat(
    *,
    update: Update,
    chat_id: str,
    text: str,
    trace_id: str,
    bot_data: dict[str, Any],
    log_path: str,
    orchestrator: Orchestrator,
    audit_log: AuditLog | None,
    placeholder_message: Any | None,
) -> None:
    result: dict[str, Any] | None = None
    started = pytime.monotonic()
    _emit_telegram_runtime_event(
        bot_data=bot_data,
        log_path=log_path,
        event_name="telegram_async_start",
        trace_id=trace_id,
        chat_id_redacted=_redact_chat_id(chat_id),
        source="telegram",
    )
    try:
        result = await _handle_telegram_text_via_local_api(
            text=text,
            chat_id=chat_id,
            trace_id=trace_id,
            bot_data=bot_data,
            log_path=log_path,
            runtime=bot_data.get("runtime"),
            orchestrator=orchestrator,
            runtime_version=str(bot_data.get("runtime_version") or "").strip() or None,
            runtime_git_commit=str(bot_data.get("runtime_git_commit") or "").strip() or None,
            runtime_started_ts=bot_data.get("runtime_started_ts"),
        )
        route = str(result.get("selected_route") or result.get("route") or "chat").strip().lower() or "chat"
        diagnosis = result.get("diagnosis") if isinstance(result.get("diagnosis"), dict) else {}
        chat_meta = result.get("chat_meta") if isinstance(result.get("chat_meta"), dict) else {}
        runtime_state_failure_reason = (
            str(chat_meta.get("runtime_state_failure_reason") or "").strip()
            or (
                str((chat_meta.get("runtime_payload") or {}).get("reason") or "").strip()
                if isinstance(chat_meta.get("runtime_payload"), dict)
                else ""
            )
            or None
        )
        proxy_failure_kind = str(chat_meta.get("proxy_failure_kind") or "").strip() or None
        proxy_failure_detail = str(chat_meta.get("proxy_failure_detail") or "").strip() or None
        proxy_failure_status_code = (
            int(chat_meta.get("proxy_failure_status_code"))
            if isinstance(chat_meta.get("proxy_failure_status_code"), int)
            else None
        )
        proxy_elapsed_ms = (
            int(chat_meta.get("proxy_elapsed_ms"))
            if isinstance(chat_meta.get("proxy_elapsed_ms"), int)
            else None
        )
        proxy_timeout_seconds = (
            float(chat_meta.get("proxy_timeout_seconds"))
            if isinstance(chat_meta.get("proxy_timeout_seconds"), (int, float))
            else None
        )
        proxy_timeout_used = (
            float(chat_meta.get("proxy_timeout_used"))
            if isinstance(chat_meta.get("proxy_timeout_used"), (int, float))
            else None
        )
        proxy_execution_mode = str(chat_meta.get("proxy_execution_mode") or "").strip() or None
        proxy_chat_lock_wait_ms = (
            int(chat_meta.get("proxy_chat_lock_wait_ms"))
            if isinstance(chat_meta.get("proxy_chat_lock_wait_ms"), int)
            else None
        )
        proxy_chat_lock_contended = (
            bool(chat_meta.get("proxy_chat_lock_contended"))
            if isinstance(chat_meta.get("proxy_chat_lock_contended"), bool)
            else None
        )
        proxy_in_flight_count = (
            int(chat_meta.get("proxy_in_flight_count"))
            if isinstance(chat_meta.get("proxy_in_flight_count"), int)
            else None
        )
        proxy_in_flight_total = (
            int(chat_meta.get("proxy_in_flight_total"))
            if isinstance(chat_meta.get("proxy_in_flight_total"), int)
            else None
        )
        proxy_overlap_rejected = (
            bool(chat_meta.get("proxy_overlap_rejected"))
            if isinstance(chat_meta.get("proxy_overlap_rejected"), bool)
            else None
        )
        proxy_total_elapsed_ms = (
            int(chat_meta.get("proxy_total_elapsed_ms"))
            if isinstance(chat_meta.get("proxy_total_elapsed_ms"), int)
            else None
        )
        proxy_backend_reachable = (
            bool(chat_meta.get("proxy_backend_reachable"))
            if isinstance(chat_meta.get("proxy_backend_reachable"), bool)
            else None
        )
        proxy_backend_ready = (
            bool(chat_meta.get("proxy_backend_ready"))
            if isinstance(chat_meta.get("proxy_backend_ready"), bool)
            else None
        )
        generic_fallback_used = bool(result.get("generic_fallback_used", False))
        generic_fallback_reason = str(result.get("generic_fallback_reason") or "").strip() or None
        await _deliver_telegram_text_result(
            placeholder_message=placeholder_message,
            original_message=update.effective_message,
            log_path=log_path,
            chat_id=chat_id,
            route=route,
            text=str(result.get("text") or ""),
            trace_id=trace_id,
        )
        _safe_append_telegram_message_audit(
            audit_log=audit_log,
            action="telegram.message.handled",
            chat_id=chat_id,
            message_kind="text",
            route=route,
            outcome="handled",
            generic_fallback_used=generic_fallback_used,
            generic_fallback_reason=generic_fallback_reason,
        )
        _log_telegram_text_handled(
            log_path=log_path,
            chat_id=chat_id,
            trace_id=trace_id,
            route=str(result.get("route") or route),
            selected_route=str(result.get("selected_route") or route),
            handler_name=str(result.get("handler_name") or "canonical_router"),
            used_llm=bool(result.get("used_llm", False)),
            used_memory=bool(result.get("used_memory", False)),
            used_runtime_state=bool(result.get("used_runtime_state", False)),
            used_tools=list(result.get("used_tools") or []),
            legacy_compatibility=bool(result.get("legacy_compatibility", False)),
            generic_fallback_used=generic_fallback_used,
            generic_fallback_reason=generic_fallback_reason,
            runtime_state_failure_reason=runtime_state_failure_reason,
            proxy_failure_kind=proxy_failure_kind,
            proxy_failure_detail=proxy_failure_detail,
            proxy_failure_status_code=proxy_failure_status_code,
            proxy_backend_reachable=proxy_backend_reachable,
            proxy_backend_ready=proxy_backend_ready,
            proxy_elapsed_ms=proxy_elapsed_ms,
            proxy_timeout_seconds=proxy_timeout_seconds,
            proxy_timeout_used=proxy_timeout_used,
            proxy_execution_mode=proxy_execution_mode,
            proxy_chat_lock_wait_ms=proxy_chat_lock_wait_ms,
            proxy_chat_lock_contended=proxy_chat_lock_contended,
            proxy_in_flight_count=proxy_in_flight_count,
            proxy_in_flight_total=proxy_in_flight_total,
            proxy_overlap_rejected=proxy_overlap_rejected,
            proxy_total_elapsed_ms=proxy_total_elapsed_ms,
            diagnosis=diagnosis,
        )
        _emit_telegram_runtime_event(
            bot_data=bot_data,
            log_path=log_path,
            event_name="telegram_async_complete",
            trace_id=trace_id,
            chat_id_redacted=_redact_chat_id(chat_id),
            route=route,
            duration_ms=int(max(0.0, pytime.monotonic() - started) * 1000),
            ok=bool(result.get("ok", False)),
        )
    except Exception as exc:
        _safe_append_telegram_message_audit(
            audit_log=audit_log,
            action="telegram.message.handled",
            chat_id=chat_id,
            message_kind="text",
            route="chat",
            outcome="failed",
            error_kind=exc.__class__.__name__,
        )
        _LOGGER.error(
            "telegram.message.error %s",
            json.dumps(
                {
                    "trace_id": trace_id,
                    "chat_id_redacted": _redact_chat_id(chat_id),
                    "text_prefix": _text_prefix(text),
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
        )
        _LOGGER.error("%s", traceback.format_exc())
        _emit_telegram_runtime_event(
            bot_data=bot_data,
            log_path=log_path,
            event_name="telegram_async_error",
            trace_id=trace_id,
            chat_id_redacted=_redact_chat_id(chat_id),
            duration_ms=int(max(0.0, pytime.monotonic() - started) * 1000),
            error_type=exc.__class__.__name__,
            error=str(exc),
        )
        try:
            await _deliver_telegram_text_result(
                placeholder_message=placeholder_message,
                original_message=update.effective_message,
                log_path=log_path,
                chat_id=chat_id,
                route="chat",
                text="Sorry — the agent encountered an error.",
                trace_id=trace_id,
            )
        except Exception:
            return


def _runtime_status_text(bot_data: dict[str, Any]) -> str:
    runtime = bot_data.get("runtime")
    result = build_telegram_status(
        runtime=runtime,
        trace_id=f"tg-status-{int(pytime.time())}-{os.getpid()}",
        runtime_version=str(bot_data.get("runtime_version") or "").strip() or None,
        runtime_git_commit=str(bot_data.get("runtime_git_commit") or "").strip() or None,
        runtime_started_ts=bot_data.get("runtime_started_ts"),
        fetch_local_api_json=(None if runtime is not None else (lambda path: _fetch_local_api_json(path))),
    )
    return _safe_reply_text(str(result.get("text") or ""))


def _log_telegram_route_decision(
    *,
    log_path: str | None,
    trace_id: str,
    route: str,
    generic_fallback_used: bool,
    generic_fallback_reason: str | None,
) -> None:
    log_event(
        log_path,
        "telegram.route.selected",
        {
            "trace_id": trace_id,
            "route": str(route or "").strip().lower() or "chat",
            "generic_fallback_used": bool(generic_fallback_used),
            "generic_fallback_reason": str(generic_fallback_reason or "").strip() or None,
        },
    )


def _log_telegram_text_handled(
    *,
    log_path: str | None,
    chat_id: str,
    trace_id: str,
    route: str,
    selected_route: str | None,
    handler_name: str,
    used_llm: bool,
    used_memory: bool,
    used_runtime_state: bool,
    used_tools: list[str],
    legacy_compatibility: bool,
    generic_fallback_used: bool,
    generic_fallback_reason: str | None,
    runtime_state_failure_reason: str | None = None,
    proxy_failure_kind: str | None = None,
    proxy_failure_detail: str | None = None,
    proxy_failure_status_code: int | None = None,
    proxy_backend_reachable: bool | None = None,
    proxy_backend_ready: bool | None = None,
    proxy_elapsed_ms: int | None = None,
    proxy_timeout_seconds: float | None = None,
    proxy_timeout_used: float | None = None,
    proxy_execution_mode: str | None = None,
    proxy_chat_lock_wait_ms: int | None = None,
    proxy_chat_lock_contended: bool | None = None,
    proxy_in_flight_count: int | None = None,
    proxy_in_flight_total: int | None = None,
    proxy_overlap_rejected: bool | None = None,
    proxy_total_elapsed_ms: int | None = None,
    diagnosis: dict[str, Any] | None = None,
) -> None:
    normalized_route = str(route or "").strip().lower() or "chat"
    normalized_selected_route = str(selected_route or normalized_route).strip().lower() or normalized_route
    diagnosis = diagnosis if isinstance(diagnosis, dict) else {}
    _log_telegram_route_decision(
        log_path=log_path,
        trace_id=trace_id,
        route=normalized_selected_route,
        generic_fallback_used=generic_fallback_used,
        generic_fallback_reason=generic_fallback_reason,
    )
    log_event(
        log_path,
        "telegram_message",
        {
            "chat_id_redacted": _redact_chat_id(chat_id),
            "route": normalized_route,
            "selected_route": normalized_selected_route,
            "trace_id": trace_id,
            "handler_name": str(handler_name or "telegram_text"),
            "used_llm": bool(used_llm),
            "used_memory": bool(used_memory),
            "used_runtime_state": bool(used_runtime_state),
            "used_tools": [str(item).strip() for item in used_tools if str(item).strip()],
            "legacy_compatibility": bool(legacy_compatibility),
            "generic_fallback_used": bool(generic_fallback_used),
            "generic_fallback_reason": str(generic_fallback_reason or "").strip() or None,
            "runtime_state_failure_reason": str(runtime_state_failure_reason or "").strip() or None,
            "proxy_failure_kind": str(proxy_failure_kind or "").strip() or None,
            "proxy_failure_detail": str(proxy_failure_detail or "").strip() or None,
            "proxy_failure_status_code": int(proxy_failure_status_code) if proxy_failure_status_code is not None else None,
            "proxy_backend_reachable": proxy_backend_reachable if isinstance(proxy_backend_reachable, bool) else None,
            "proxy_backend_ready": proxy_backend_ready if isinstance(proxy_backend_ready, bool) else None,
            "proxy_elapsed_ms": int(proxy_elapsed_ms) if proxy_elapsed_ms is not None else None,
            "proxy_timeout_seconds": float(proxy_timeout_seconds) if proxy_timeout_seconds is not None else None,
            "proxy_timeout_used": float(proxy_timeout_used) if proxy_timeout_used is not None else None,
            "proxy_execution_mode": str(proxy_execution_mode or "").strip() or None,
            "proxy_chat_lock_wait_ms": int(proxy_chat_lock_wait_ms) if proxy_chat_lock_wait_ms is not None else None,
            "proxy_chat_lock_contended": proxy_chat_lock_contended if isinstance(proxy_chat_lock_contended, bool) else None,
            "proxy_in_flight_count": int(proxy_in_flight_count) if proxy_in_flight_count is not None else None,
            "proxy_in_flight_total": int(proxy_in_flight_total) if proxy_in_flight_total is not None else None,
            "proxy_overlap_rejected": proxy_overlap_rejected if isinstance(proxy_overlap_rejected, bool) else None,
            "proxy_total_elapsed_ms": int(proxy_total_elapsed_ms) if proxy_total_elapsed_ms is not None else None,
            "diagnosis_source": str(diagnosis.get("source") or "").strip() or None,
            "diagnosis_confidence": str(diagnosis.get("confidence") or "").strip() or None,
            "diagnosis_mapped_state": str(diagnosis.get("mapped_state") or "").strip() or None,
        },
    )


def _model_status_report(*, runtime: Any | None) -> str:
    if runtime is not None and hasattr(runtime, "public_llm_identity_string"):
        try:
            identity_line = str(runtime.public_llm_identity_string()).strip()
        except Exception:
            identity_line = ""
    else:
        identity_line = ""
    if runtime is not None and hasattr(runtime, "model_status"):
        try:
            payload = runtime.model_status()  # type: ignore[attr-defined]
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            current = payload.get("current") if isinstance(payload.get("current"), dict) else {}
            availability = (
                payload.get("llm_availability")
                if isinstance(payload.get("llm_availability"), dict)
                else {}
            )
            providers = availability.get("providers") if isinstance(availability.get("providers"), dict) else {}
            watch = payload.get("model_watch") if isinstance(payload.get("model_watch"), dict) else {}
            default_provider = str(current.get("provider") or "unknown").strip() or "unknown"
            resolved_model = str(current.get("model_id") or "unknown").strip() or "unknown"
            provider_ids = providers.get("configured") if isinstance(providers.get("configured"), list) else []
            provider_line = ", ".join(
                sorted(
                    str(item).strip().lower()
                    for item in provider_ids
                    if str(item).strip()
                )
            ) or "none"
            model_watch_line = str(watch.get("summary_line") or "Model Watch: no summary available.").strip()
            identity = str(payload.get("identity") or "").strip() or identity_line
            if not identity:
                identity_payload = get_public_identity(
                    provider=default_provider if default_provider != "unknown" else None,
                    model=resolved_model if resolved_model != "unknown" else None,
                    local_providers={"ollama"},
                )
                identity = str(identity_payload.get("summary") or "").strip()
            current_line = identity or f"Current provider/model: {default_provider} / {resolved_model}"
            return "\n".join(
                [
                    current_line,
                    f"Configured providers: {provider_line}",
                    model_watch_line,
                ]
            )

    status: dict[str, Any] = {}
    if runtime is not None and hasattr(runtime, "llm_status"):
        try:
            row = runtime.llm_status()  # type: ignore[attr-defined]
            if isinstance(row, dict):
                status = row
        except Exception:
            status = {}
    default_provider = str(status.get("default_provider") or "unknown").strip() or "unknown"
    resolved_model = (
        str(status.get("resolved_default_model") or "").strip()
        or str(status.get("default_model") or "").strip()
        or "unknown"
    )
    providers_rows = status.get("providers") if isinstance(status.get("providers"), list) else []
    provider_ids = sorted(
        str(row.get("id") or "").strip().lower()
        for row in providers_rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    )
    provider_line = ", ".join(provider_ids) if provider_ids else "none"

    model_watch_line = "Model Watch: no summary available."
    if runtime is not None and hasattr(runtime, "model_watch_latest"):
        try:
            latest = runtime.model_watch_latest()  # type: ignore[attr-defined]
        except Exception:
            latest = {}
        if isinstance(latest, dict):
            if bool(latest.get("found")) and isinstance(latest.get("batch"), dict):
                batch = latest.get("batch") if isinstance(latest.get("batch"), dict) else {}
                top = batch.get("top_pick") if isinstance(batch.get("top_pick"), dict) else {}
                top_model = str(top.get("model") or top.get("id") or "unknown").strip()
                top_score = float(top.get("score") or 0.0)
                model_watch_line = f"Model Watch: top candidate {top_model} (score {top_score:.2f})."
            elif str(latest.get("reason") or "").strip():
                model_watch_line = f"Model Watch: {str(latest.get('reason') or '').strip()}."

    if identity_line:
        current_line = identity_line
    else:
        identity_payload = get_public_identity(
            provider=default_provider if default_provider != "unknown" else None,
            model=resolved_model if resolved_model != "unknown" else None,
            local_providers={"ollama"},
        )
        current_line = str(identity_payload.get("summary") or f"Current provider/model: {default_provider} / {resolved_model}")
    return "\n".join(
        [
            current_line,
            f"Configured providers: {provider_line}",
            model_watch_line,
        ]
    )


def _model_watch_run_summary(*, runtime: Any | None) -> str:
    if runtime is None or not hasattr(runtime, "run_model_watch_once"):
        return "Model watch is unavailable in this runtime."
    try:
        ok, body = runtime.run_model_watch_once(trigger="manual")  # type: ignore[attr-defined]
    except Exception:
        return "Model watch run failed."
    payload = body if isinstance(body, dict) else {}
    if not ok:
        message = str(payload.get("message") or payload.get("error") or "Model watch run failed.").strip()
        return message or "Model watch run failed."
    proposal = payload.get("proposal") if isinstance(payload.get("proposal"), dict) else None
    if not isinstance(proposal, dict):
        return "No new better models in configured providers."
    proposal_type = str(payload.get("proposal_type") or proposal.get("proposal_type") or "").strip().lower()
    if proposal_type == "local_download":
        repo_id = str(proposal.get("repo_id") or "unknown").strip()
        revision = str(proposal.get("revision") or "unknown").strip()
        return (
            f"Model watch found a local download candidate: {repo_id} @ {revision}.\n"
            "Use the current fix-it prompt to approve or snooze."
        )
    to_model = str(proposal.get("to_model") or "unknown").strip()
    score_delta = float(proposal.get("score_delta") or 0.0)
    return (
        "Model watch found a better default candidate.\n"
        f"Top candidate: {to_model} (+{score_delta:.2f}).\n"
        "Use the current fix-it prompt to apply or snooze."
    )
def _trace_id_from_update(update: Update, *, fallback_prefix: str = "tg") -> str:
    chat_id = (
        str(getattr(getattr(update, "effective_chat", None), "id", "") or "").strip()
        if update is not None
        else ""
    )
    message_id = (
        str(getattr(getattr(update, "effective_message", None), "message_id", "") or "").strip()
        if update is not None
        else ""
    )
    if chat_id and message_id:
        return f"{fallback_prefix}-{chat_id}-{message_id}"
    if chat_id:
        return f"{fallback_prefix}-{chat_id}"
    return f"{fallback_prefix}-unknown"


def _envelope_from_exception(
    *,
    exc: Exception,
    intent: str,
    trace_id: str,
    log_path: str | None,
) -> dict[str, object]:
    def _raiser() -> dict[str, object]:
        raise exc

    return run_with_fallback(
        fn=_raiser,
        context={
            "intent": intent,
            "trace_id": trace_id,
            "log_path": log_path,
            "actions": [],
        },
    )


def _redact_chat_id(chat_id: str) -> str:
    value = str(chat_id or "").strip()
    if not value:
        return "unknown"
    if len(value) <= 4:
        return "***"
    return f"***{value[-4:]}"


def _safe_append_telegram_message_audit(
    *,
    audit_log: AuditLog | None,
    action: str,
    chat_id: str,
    message_kind: str,
    route: str,
    outcome: str,
    error_kind: str | None = None,
    generic_fallback_used: bool | None = None,
    generic_fallback_reason: str | None = None,
) -> None:
    if audit_log is None:
        return
    params = {
        "chat_id_redacted": _redact_chat_id(chat_id),
        "message_kind": str(message_kind or "text"),
        "route": str(route or "ignored"),
    }
    if generic_fallback_used is not None:
        params["generic_fallback_used"] = bool(generic_fallback_used)
    if generic_fallback_reason:
        params["generic_fallback_reason"] = str(generic_fallback_reason).strip()
    try:
        audit_log.append(
            actor="telegram",
            action=action,
            params=params,
            decision="allow",
            reason=f"{route}:{outcome}",
            dry_run=False,
            outcome=str(outcome or "handled"),
            error_kind=str(error_kind or "") or None,
            duration_ms=0,
        )
    except Exception:
        pass


def _safe_audit_fixit_prompt_shown(
    *,
    audit_log: AuditLog | None,
    chat_id: str,
    status: str,
    issue_code: str,
    step: str,
) -> None:
    if audit_log is None:
        return
    try:
        audit_log.append(
            actor="telegram",
            action="telegram.fixit.prompt_shown",
            params={
                "chat_id_redacted": _redact_chat_id(chat_id),
                "status": str(status or "").strip(),
                "issue_code": str(issue_code or "").strip(),
                "step": str(step or "").strip(),
            },
            decision="allow",
            reason="prompt_shown",
            dry_run=False,
            outcome="success",
            error_kind=None,
            duration_ms=0,
        )
    except Exception:
        pass


def _wizard_status_from_state(state: dict[str, Any]) -> str:
    if not isinstance(state, dict):
        return "idle"
    if not bool(state.get("active", False)):
        return "idle"
    step = str(state.get("step") or "").strip().lower()
    if step == "awaiting_choice":
        return "needs_user_choice"
    if step == "awaiting_confirm":
        return "needs_confirmation"
    return "idle"


def _map_fixit_reply_to_payload(state: dict[str, Any], text: str) -> tuple[dict[str, Any] | None, str | None]:
    status = _wizard_status_from_state(state)
    if status == "idle":
        return None, None
    normalized = " ".join(str(text or "").strip().lower().split())
    if not normalized:
        if status == "needs_user_choice":
            return None, "Reply 1, 2, or 3."
        return None, "Reply YES (1) or NO (2)."

    if status == "needs_user_choice":
        raw_choices = state.get("choices") if isinstance(state.get("choices"), list) else []
        choices: list[dict[str, str]] = []
        for row in raw_choices:
            if not isinstance(row, dict):
                continue
            choice_id = str(row.get("id") or "").strip()
            label = str(row.get("label") or "").strip()
            if not choice_id:
                continue
            choices.append({"id": choice_id, "label": label})
        if not choices:
            return None, "No pending choices right now. Ask me to run fix-it again."
        if normalized.isdigit():
            idx = int(normalized)
            if 1 <= idx <= len(choices):
                return {"answer": choices[idx - 1]["id"]}, None
        for choice in choices:
            choice_id = str(choice.get("id") or "").strip().lower()
            label = str(choice.get("label") or "").strip().lower()
            if normalized == choice_id or (label and normalized == label):
                return {"answer": choice.get("id")}, None
        return None, "Reply 1, 2, or 3."

    if status == "needs_confirmation":
        if normalized in {"1", "yes", "y", "apply", "ok"}:
            return {"confirm": True}, None
        if normalized in {"2", "no", "n", "cancel"}:
            return {"confirm": False}, None
        return None, "Reply YES (1) or NO (2)."

    return None, None


def _normalize_fixit_choices(choices_raw: Any) -> list[dict[str, Any]]:
    if not isinstance(choices_raw, list):
        return []
    choices: list[dict[str, Any]] = []
    for row in choices_raw:
        if not isinstance(row, dict):
            continue
        choice_id = str(row.get("id") or "").strip()
        label = str(row.get("label") or "").strip()
        if not choice_id or not label:
            continue
        choices.append(
            {
                "id": choice_id,
                "label": label,
                "recommended": bool(row.get("recommended", False)),
            }
        )
    return choices


def _persist_fixit_prompt_state_from_response(
    *,
    wizard_store: LLMFixitWizardStore | None,
    body: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if wizard_store is None or not isinstance(body, dict):
        return None
    status = str(body.get("status") or "").strip().lower()
    issue_code = str(body.get("issue_code") or "").strip()
    message = str(body.get("message") or "").strip()
    next_question = str(body.get("next_question") or "").strip()
    now_epoch = int(pytime.time())
    existing_state = (
        wizard_store.state if isinstance(wizard_store.state, dict) else wizard_store.empty_state()
    )
    openrouter_last_test = (
        existing_state.get("openrouter_last_test")
        if isinstance(existing_state.get("openrouter_last_test"), dict)
        else None
    )

    if status == "needs_user_choice":
        choices = _normalize_fixit_choices(body.get("choices"))
        if not choices:
            return None
        return wizard_store.save(
            {
                "active": True,
                "issue_hash": str(existing_state.get("issue_hash") or "").strip() or None,
                "issue_code": issue_code or str(existing_state.get("issue_code") or "").strip() or None,
                "step": "awaiting_choice",
                "question": next_question or message,
                "choices": choices,
                "pending_plan": [],
                "pending_confirm_token": None,
                "pending_created_ts": None,
                "pending_expires_ts": None,
                "pending_issue_code": None,
                "last_prompt_ts": now_epoch,
                "openrouter_last_test": openrouter_last_test,
                "proposal_type": str(
                    body.get("proposal_type") or existing_state.get("proposal_type") or ""
                ).strip()
                or None,
                "proposal_details": str(
                    body.get("proposal_details") or existing_state.get("proposal_details") or ""
                ).strip()
                or None,
            }
        )

    if status == "needs_confirmation":
        plan_rows_raw = body.get("plan")
        plan_rows = [row for row in plan_rows_raw if isinstance(row, dict)] if isinstance(plan_rows_raw, list) else []
        if not plan_rows:
            return None
        confirm_token = confirm_token_for_plan_rows(plan_rows)
        choices = _normalize_fixit_choices(body.get("choices"))
        return wizard_store.save(
            {
                "active": True,
                "issue_hash": str(existing_state.get("issue_hash") or "").strip() or None,
                "issue_code": issue_code or str(existing_state.get("issue_code") or "").strip() or None,
                "step": "awaiting_confirm",
                "question": next_question or message,
                "choices": choices,
                "pending_plan": plan_rows,
                "pending_confirm_token": confirm_token,
                "pending_created_ts": now_epoch,
                "pending_expires_ts": now_epoch + 300,
                "pending_issue_code": issue_code or str(existing_state.get("issue_code") or "").strip() or None,
                "last_prompt_ts": now_epoch,
                "openrouter_last_test": openrouter_last_test,
                "proposal_type": str(
                    body.get("proposal_type") or existing_state.get("proposal_type") or ""
                ).strip()
                or None,
                "proposal_details": str(
                    body.get("proposal_details") or existing_state.get("proposal_details") or ""
                ).strip()
                or None,
            }
        )
    return None


def _maybe_handle_operator_recovery_reply_with_route(
    *,
    operator_recovery_fn: Callable[[dict[str, Any]], tuple[bool, dict[str, Any]]] | None,
    recovery_store: OperatorRecoveryStore | None,
    audit_log: AuditLog | None,
    chat_id: str,
    text: str,
    log_path: str | None,
) -> tuple[str | None, str | None]:
    if operator_recovery_fn is None or recovery_store is None:
        return None, None
    try:
        state = recovery_store.load()
        recovery_store.state = state
    except Exception:
        state = recovery_store.empty_state()
    status = _wizard_status_from_state(state)
    payload, hint_message = _map_fixit_reply_to_payload(state, text)
    if payload is None:
        if hint_message is None:
            return None, None
        if audit_log is not None:
            try:
                audit_log.append(
                    actor="telegram",
                    action="llm.fixit.telegram",
                    params={
                        "chat_id_redacted": _redact_chat_id(chat_id),
                        "status": status,
                        "mapped": False,
                    },
                    decision="allow",
                    reason="reply_not_mapped",
                    dry_run=False,
                    outcome="handled",
                    error_kind="needs_clarification",
                    duration_ms=0,
                )
            except Exception:
                pass
        return hint_message, "fixit_invalid_choice"

    payload = dict(payload)
    payload["actor"] = "telegram"
    ok, body = operator_recovery_fn(payload)
    persisted_state = _persist_fixit_prompt_state_from_response(
        wizard_store=recovery_store,
        body=body if isinstance(body, dict) else None,
    )
    if isinstance(persisted_state, dict):
        prompt_step = str(persisted_state.get("step") or "").strip()
        prompt_issue = str(persisted_state.get("issue_code") or "").strip()
        prompt_status = "needs_confirmation" if prompt_step == "awaiting_confirm" else "needs_user_choice"
        _safe_audit_fixit_prompt_shown(
            audit_log=audit_log if isinstance(audit_log, AuditLog) else None,
            chat_id=chat_id,
            status=prompt_status,
            issue_code=prompt_issue,
            step=prompt_step,
        )
    message = str(body.get("message") or body.get("next_question") or "").strip()
    if not message:
        message = "I processed that fix-it step."
    if audit_log is not None:
        try:
            audit_log.append(
                actor="telegram",
                action="llm.fixit.telegram",
                params={
                    "chat_id_redacted": _redact_chat_id(chat_id),
                    "status": status,
                    "mapped": True,
                    "answer": payload.get("answer"),
                    "confirm": payload.get("confirm"),
                },
                decision="allow",
                reason="fixit_reply",
                dry_run=False,
                outcome="success" if ok else "failed",
                error_kind=str(body.get("error_kind") or "") or None,
                duration_ms=0,
            )
        except Exception:
            pass
    if log_path:
        log_event(
            log_path,
            "llm_fixit_telegram",
            {
                "chat_id_redacted": _redact_chat_id(chat_id),
                "status": status,
                "mapped": True,
                "ok": bool(ok),
                "error_kind": str(body.get("error_kind") or "") or None,
            },
        )
    return message, "fixit"


def _maybe_handle_llm_fixit_reply_with_route(
    *,
    llm_fixit_fn: Callable[[dict[str, Any]], tuple[bool, dict[str, Any]]] | None,
    wizard_store: LLMFixitWizardStore | None,
    audit_log: AuditLog | None,
    chat_id: str,
    text: str,
    log_path: str | None,
) -> tuple[str | None, str | None]:
    return _maybe_handle_operator_recovery_reply_with_route(
        operator_recovery_fn=llm_fixit_fn,
        recovery_store=wizard_store,
        audit_log=audit_log,
        chat_id=chat_id,
        text=text,
        log_path=log_path,
    )


def maybe_handle_operator_recovery_reply(
    *,
    operator_recovery_fn: Callable[[dict[str, Any]], tuple[bool, dict[str, Any]]] | None,
    recovery_store: OperatorRecoveryStore | None,
    audit_log: AuditLog | None,
    chat_id: str,
    text: str,
    log_path: str | None,
) -> str | None:
    message, _route = _maybe_handle_operator_recovery_reply_with_route(
        operator_recovery_fn=operator_recovery_fn,
        recovery_store=recovery_store,
        audit_log=audit_log,
        chat_id=chat_id,
        text=text,
        log_path=log_path,
    )
    return message


def maybe_handle_llm_fixit_reply(
    *,
    llm_fixit_fn: Callable[[dict[str, Any]], tuple[bool, dict[str, Any]]] | None,
    wizard_store: LLMFixitWizardStore | None,
    audit_log: AuditLog | None,
    chat_id: str,
    text: str,
    log_path: str | None,
) -> str | None:
    message, _route = _maybe_handle_operator_recovery_reply_with_route(
        operator_recovery_fn=llm_fixit_fn,
        recovery_store=wizard_store,
        audit_log=audit_log,
        chat_id=chat_id,
        text=text,
        log_path=log_path,
    )
    return message


async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    audit_log_value = context.application.bot_data.get("audit_log")
    audit_log = audit_log_value if isinstance(audit_log_value, AuditLog) else None

    if update.effective_chat is None or update.effective_message is None:
        _safe_append_telegram_message_audit(
            audit_log=audit_log,
            action="telegram.message.ignored",
            chat_id="",
            message_kind="non_text",
            route="ignored",
            outcome="ignored",
        )
        return

    chat_id = str(update.effective_chat.id)
    text = (update.effective_message.text or "").strip()
    if not text:
        _safe_append_telegram_message_audit(
            audit_log=audit_log,
            action="telegram.message.ignored",
            chat_id=chat_id,
            message_kind="text",
            route="ignored",
            outcome="ignored",
        )
        return

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    db: MemoryDB = context.application.bot_data["db"]
    log_path: str = context.application.bot_data["log_path"]
    bot_data = context.application.bot_data
    trace_id = _trace_id_from_update(update)
    operator_recovery_fn = bot_data.get("operator_recovery_fn")
    if not callable(operator_recovery_fn):
        operator_recovery_fn = bot_data.get("llm_fixit_fn")
    recovery_store = bot_data.get("operator_recovery_store")
    if not isinstance(recovery_store, OperatorRecoveryStore):
        recovery_store = bot_data.get("llm_fixit_store")
    _LOGGER.info(
        "telegram.in %s",
        json.dumps(
            {
                "trace_id": trace_id,
                "user_id": chat_id,
                "text_prefix": _text_prefix(text),
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
    )
    log_event(
        log_path,
        "telegram.in",
        {
            "user_id": chat_id,
            "text_prefix": _text_prefix(text),
            "trace_id": trace_id,
        },
    )
    log_event(
        log_path,
        "incoming_message",
        {
            "user_id": chat_id,
            "text_prefix": _text_prefix(text),
            "trace_id": trace_id,
        },
    )
    _safe_append_telegram_message_audit(
        audit_log=audit_log,
        action="telegram.message.received",
        chat_id=chat_id,
        message_kind="text",
        route="chat",
        outcome="received",
    )
    log_event(
        log_path,
        "telegram.route",
        {
            "user_id": chat_id,
            "message_kind": "text",
            "route": "text",
            "trace_id": trace_id,
        },
    )

    try:
        # Remember which chat we're talking to (useful for reminders/jobs).
        db.set_preference("telegram_chat_id", chat_id)

        normalized_text = _normalize_user_text(text)

        fixit_reply, fixit_route = _maybe_handle_operator_recovery_reply_with_route(
            operator_recovery_fn=operator_recovery_fn if callable(operator_recovery_fn) else None,
            recovery_store=recovery_store if isinstance(recovery_store, OperatorRecoveryStore) else None,
            audit_log=audit_log if isinstance(audit_log, AuditLog) else None,
            chat_id=chat_id,
            text=text,
            log_path=log_path,
        )
        if fixit_reply is not None:
            route = str(fixit_route or "fixit")
            generic_fallback_used = False
            generic_fallback_reason = None
            _safe_append_telegram_message_audit(
                audit_log=audit_log,
                action="telegram.message.handled",
                chat_id=chat_id,
                message_kind="text",
                route=route,
                outcome="handled",
                generic_fallback_used=generic_fallback_used,
                generic_fallback_reason=generic_fallback_reason,
            )
            await _send_reply(
                message=update.effective_message,
                log_path=log_path,
                chat_id=chat_id,
                route=route,
                text=str(fixit_reply),
                trace_id=trace_id,
            )
            _log_telegram_text_handled(
                log_path=log_path,
                chat_id=chat_id,
                trace_id=trace_id,
                route=route,
                selected_route=route,
                handler_name="fixit_wizard",
                used_llm=False,
                used_memory=False,
                used_runtime_state=False,
                used_tools=["llm_fixit"],
                legacy_compatibility=False,
                generic_fallback_used=generic_fallback_used,
                generic_fallback_reason=generic_fallback_reason,
            )
            return

        if normalized_text in {"1", "2", "3"}:
            route = "numeric_no_wizard"
            generic_fallback_used = False
            generic_fallback_reason = None
            _safe_append_telegram_message_audit(
                audit_log=audit_log,
                action="telegram.message.handled",
                chat_id=chat_id,
                message_kind="text",
                route=route,
                outcome="handled",
                generic_fallback_used=generic_fallback_used,
                generic_fallback_reason=generic_fallback_reason,
            )
            await _send_reply(
                message=update.effective_message,
                log_path=log_path,
                chat_id=chat_id,
                route=route,
                text=_NO_ACTIVE_CHOICE_TEXT,
                trace_id=trace_id,
            )
            _log_telegram_text_handled(
                log_path=log_path,
                chat_id=chat_id,
                trace_id=trace_id,
                route=route,
                selected_route=route,
                handler_name="numeric_choice_guard",
                used_llm=False,
                used_memory=False,
                used_runtime_state=False,
                used_tools=[],
                legacy_compatibility=False,
                generic_fallback_used=generic_fallback_used,
                generic_fallback_reason=generic_fallback_reason,
            )
            return

        placeholder_message = await _send_placeholder_message(
            update.effective_message,
            "Thinking…",
            trace_id=trace_id,
            chat_id=chat_id,
        )
        background_task = asyncio.create_task(
            _run_async_telegram_chat(
                update=update,
                chat_id=chat_id,
                text=text,
                trace_id=trace_id,
                bot_data=bot_data,
                log_path=log_path,
                orchestrator=orchestrator,
                audit_log=audit_log,
                placeholder_message=placeholder_message,
            )
        )
        tasks = _telegram_background_tasks(bot_data)
        tasks.add(background_task)
        background_task.add_done_callback(lambda task: tasks.discard(task))
        await asyncio.sleep(0)
        return
    except Exception as exc:
        _safe_append_telegram_message_audit(
            audit_log=audit_log,
            action="telegram.message.handled",
            chat_id=chat_id,
            message_kind="text",
            route="chat",
            outcome="failed",
            error_kind=exc.__class__.__name__,
        )
        _LOGGER.error(
            "telegram.message.error %s",
            json.dumps(
                {
                    "trace_id": trace_id,
                    "chat_id_redacted": _redact_chat_id(chat_id),
                    "text_prefix": _text_prefix(text),
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
        )
        _LOGGER.error("%s", traceback.format_exc())
        envelope = _envelope_from_exception(
            exc=exc,
            intent="telegram.message",
            trace_id=trace_id,
            log_path=log_path,
        )
        message = deterministic_error_message(
            title=f"❌ {_safe_reply_text(str(envelope.get('message') or _TELEGRAM_FALLBACK_TEXT))}",
            trace_id=trace_id,
            component="telegram_adapter",
            next_action="run `agent doctor`",
        )
        try:
            await _send_reply(
                message=update.effective_message,
                log_path=log_path,
                chat_id=chat_id,
                route="chat",
                text=message,
                trace_id=trace_id,
            )
        except Exception:
            return


def _command_payload(text: str, command: str) -> str:
    if not text:
        return ""
    parts = text.split(maxsplit=1)
    if not parts:
        return ""
    token = parts[0]
    if token.startswith(command):
        return parts[1] if len(parts) > 1 else ""
    return ""


def _log_command_route(
    *,
    log_path: str | None,
    chat_id: str,
    command: str,
) -> None:
    if not log_path:
        return
    log_event(
        log_path,
        "telegram.route",
        {
            "user_id": chat_id,
            "message_kind": "command",
            "route": "command",
            "command": command,
        },
    )


async def _handle_remind(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    text = update.effective_message.text or ""
    content = _command_payload(text, "/remind")
    prompt = f"remind me {content}".strip()

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    log_path: str = context.application.bot_data["log_path"]
    _log_command_route(log_path=log_path, chat_id=chat_id, command="/remind")

    response = orchestrator.handle_message(prompt, user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))
    log_event(log_path, "telegram_command", {"chat_id": chat_id, "text": text, "forwarded": prompt})


async def _handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    log_path: str = context.application.bot_data["log_path"]
    _log_command_route(log_path=log_path, chat_id=chat_id, command="/status")
    trace_id = _trace_id_from_update(update)
    runtime = context.application.bot_data.get("runtime")
    result = handle_telegram_command(
        command="/status",
        chat_id=chat_id,
        trace_id=trace_id,
        runtime=runtime,
        orchestrator=context.application.bot_data.get("orchestrator"),
        runtime_version=str(context.application.bot_data.get("runtime_version") or "").strip() or None,
        runtime_git_commit=str(context.application.bot_data.get("runtime_git_commit") or "").strip() or None,
        runtime_started_ts=context.application.bot_data.get("runtime_started_ts"),
        fetch_local_api_json=(None if runtime is not None else (lambda path: _fetch_local_api_json(path))),
    )
    await _send_reply(
        message=update.effective_message,
        log_path=log_path,
        chat_id=chat_id,
        route=str(result.get("route") or "status"),
        text=str(result.get("text") or ""),
        trace_id=trace_id,
    )


async def _handle_doctor(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    log_path: str = context.application.bot_data["log_path"]
    _log_command_route(log_path=log_path, chat_id=chat_id, command="/doctor")
    trace_id = _trace_id_from_update(update)
    result = handle_telegram_command(
        command="/doctor",
        chat_id=chat_id,
        trace_id=trace_id,
        runtime=context.application.bot_data.get("runtime"),
        orchestrator=context.application.bot_data.get("orchestrator"),
    )
    await _send_reply(
        message=update.effective_message,
        log_path=log_path,
        chat_id=chat_id,
        route=str(result.get("route") or "doctor"),
        text=str(result.get("text") or "Doctor failed. Run: python -m agent doctor --json"),
        trace_id=trace_id,
    )


async def _handle_disk_grow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    text = update.effective_message.text or ""
    content = _command_payload(text, "/disk_grow")
    prompt = "/disk_grow {}".format(content).strip()

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message(prompt, user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_audit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    audit_log: AuditLog = context.application.bot_data["audit_log"]
    rows = audit_log.recent(limit=5)
    if not rows:
        await update.effective_message.reply_text("No ModelOps audit events yet.")
        return
    lines = ["Recent ModelOps audit events:"]
    for row in rows:
        lines.append(
            f"- {row.get('ts')} {row.get('action')} "
            f"[{row.get('decision')}/{row.get('outcome')}] reason={row.get('reason')}"
        )
    await update.effective_message.reply_text("\n".join(lines))


async def _handle_permissions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    permission_store: PermissionStore = context.application.bot_data["permission_store"]
    permissions = permission_store.load()
    mode = permissions.get("mode") or "manual_confirm"
    actions = permissions.get("actions") if isinstance(permissions.get("actions"), dict) else {}
    constraints = permissions.get("constraints") if isinstance(permissions.get("constraints"), dict) else {}

    lines = [f"ModelOps permissions mode: {mode}", "Actions:"]
    for action_name in sorted(actions.keys()):
        lines.append(f"- {action_name}: {'allow' if bool(actions[action_name]) else 'deny'}")
    lines.append("Constraints:")
    lines.append(f"- max_download_gb: {constraints.get('max_download_gb')}")
    lines.append(f"- allow_install_ollama: {constraints.get('allow_install_ollama')}")
    lines.append(f"- allow_remote_models: {constraints.get('allow_remote_models')}")
    lines.append(f"- allowed_providers: {', '.join(constraints.get('allowed_providers') or [])}")
    await update.effective_message.reply_text("\n".join(lines))


async def _handle_storage_snapshot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]

    response = orchestrator.handle_message("/storage_snapshot", user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_storage_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]

    response = orchestrator.handle_message("/storage_report", user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_resource_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]

    response = orchestrator.handle_message("/resource_report", user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_brief(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    log_path: str = context.application.bot_data["log_path"]
    _log_command_route(log_path=log_path, chat_id=chat_id, command="/brief")
    trace_id = _trace_id_from_update(update)
    result = handle_telegram_command(
        command="/brief",
        chat_id=chat_id,
        trace_id=trace_id,
        runtime=context.application.bot_data.get("runtime"),
        orchestrator=orchestrator,
    )
    await _send_reply(
        message=update.effective_message,
        log_path=log_path,
        chat_id=chat_id,
        route=str(result.get("route") or "brief"),
        text=str(result.get("text") or ""),
        trace_id=trace_id,
    )


async def _handle_brief_alias(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    audit_log_value = context.application.bot_data.get("audit_log")
    audit_log = audit_log_value if isinstance(audit_log_value, AuditLog) else None
    _safe_append_telegram_message_audit(
        audit_log=audit_log,
        action="telegram.message.handled",
        chat_id=chat_id,
        message_kind="command",
        route="alias",
        outcome="handled",
    )
    await _handle_brief(update, context)


async def _handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    audit_log_value = context.application.bot_data.get("audit_log")
    audit_log = audit_log_value if isinstance(audit_log_value, AuditLog) else None
    _safe_append_telegram_message_audit(
        audit_log=audit_log,
        action="telegram.message.handled",
        chat_id=chat_id,
        message_kind="command",
        route="help",
        outcome="handled",
    )
    log_path: str = context.application.bot_data["log_path"]
    trace_id = _trace_id_from_update(update)
    result = handle_telegram_command(
        command="/help",
        chat_id=chat_id,
        trace_id=trace_id,
        runtime=context.application.bot_data.get("runtime"),
        orchestrator=context.application.bot_data.get("orchestrator"),
    )
    await _send_reply(
        message=update.effective_message,
        log_path=log_path,
        chat_id=chat_id,
        route=str(result.get("route") or "help"),
        text=str(result.get("text") or ""),
        trace_id=trace_id,
    )


async def _handle_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    audit_log_value = context.application.bot_data.get("audit_log")
    audit_log = audit_log_value if isinstance(audit_log_value, AuditLog) else None
    _safe_append_telegram_message_audit(
        audit_log=audit_log,
        action="telegram.message.handled",
        chat_id=chat_id,
        message_kind="command",
        route="status",
        outcome="handled",
    )
    runtime = context.application.bot_data.get("runtime")
    await update.effective_message.reply_text(_safe_reply_text(_model_status_report(runtime=runtime)))


async def _handle_network_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]

    response = orchestrator.handle_message("/network_report", user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_sys_metrics_snapshot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message("/sys_metrics_snapshot", user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_sys_health_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message("/sys_health_report", user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_sys_inventory_summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message("/sys_inventory_summary", user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_weekly_reflection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]

    response = orchestrator.handle_message("/weekly_reflection", user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message("/today", user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_task_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    text = update.effective_message.text or ""
    content = _command_payload(text, "/task_add")
    prompt = f"/task_add {content}".strip()
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message(prompt, user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_done(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    text = update.effective_message.text or ""
    content = _command_payload(text, "/done")
    prompt = f"/done {content}".strip()
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message(prompt, user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_open_loops(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    text = update.effective_message.text or ""
    content = _command_payload(text, "/open_loops")
    prompt = f"/open_loops {content}".strip()
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message(prompt, user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_health(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    log_path: str = context.application.bot_data["log_path"]
    _log_command_route(log_path=log_path, chat_id=chat_id, command="/health")
    trace_id = _trace_id_from_update(update)
    result = handle_telegram_command(
        command="/health",
        chat_id=chat_id,
        trace_id=trace_id,
        runtime=context.application.bot_data.get("runtime"),
        orchestrator=orchestrator,
    )
    await _send_reply(
        message=update.effective_message,
        log_path=log_path,
        chat_id=chat_id,
        route=str(result.get("route") or "health"),
        text=str(result.get("text") or ""),
        trace_id=trace_id,
    )


async def _handle_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    log_path = str(context.application.bot_data.get("log_path") or "")
    _log_command_route(log_path=log_path, chat_id=chat_id, command="/memory")
    trace_id = _trace_id_from_update(update)
    result = handle_telegram_command(
        command="/memory",
        chat_id=chat_id,
        trace_id=trace_id,
        runtime=context.application.bot_data.get("runtime"),
        orchestrator=orchestrator,
    )
    await _send_reply(
        message=update.effective_message,
        log_path=log_path,
        chat_id=chat_id,
        route=str(result.get("route") or "memory"),
        text=str(result.get("text") or ""),
        trace_id=trace_id,
    )


async def _handle_daily_brief_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message("/daily_brief_status", user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    text = update.effective_message.text or ""
    content = _command_payload(text, "/ask")
    prompt = "/ask {}".format(content).strip()

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message(prompt, user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_ask_opinion(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    text = update.effective_message.text or ""
    content = _command_payload(text, "/ask_opinion")
    prompt = "/ask_opinion {}".format(content).strip()

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message(prompt, user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_scout(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    model_scout = context.application.bot_data.get("model_scout")
    if model_scout is None:
        await update.effective_message.reply_text("Model Scout is unavailable in this runtime.")
        return

    suggestions = model_scout.list_suggestions(status="new", limit=5)
    message = model_scout.format_scout_details(suggestions, limit=5)
    await update.effective_message.reply_text(message)


async def _handle_scout_dismiss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    model_scout = context.application.bot_data.get("model_scout")
    if model_scout is None:
        await update.effective_message.reply_text("Model Scout is unavailable in this runtime.")
        return

    text = update.effective_message.text or ""
    suggestion_id = _command_payload(text, "/scout_dismiss").strip()
    if not suggestion_id:
        await update.effective_message.reply_text("Usage: /scout_dismiss <suggestion_id>")
        return

    if model_scout.dismiss(suggestion_id):
        await update.effective_message.reply_text(f"Dismissed suggestion {suggestion_id}.")
    else:
        await update.effective_message.reply_text(f"Suggestion not found: {suggestion_id}")


async def _handle_scout_installed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    model_scout = context.application.bot_data.get("model_scout")
    if model_scout is None:
        await update.effective_message.reply_text("Model Scout is unavailable in this runtime.")
        return

    text = update.effective_message.text or ""
    suggestion_id = _command_payload(text, "/scout_installed").strip()
    if not suggestion_id:
        await update.effective_message.reply_text("Usage: /scout_installed <suggestion_id>")
        return

    if model_scout.mark_installed(suggestion_id):
        await update.effective_message.reply_text(f"Marked suggestion as installed: {suggestion_id}.")
    else:
        await update.effective_message.reply_text(f"Suggestion not found: {suggestion_id}")


async def _on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    import logging

    logger = logging.getLogger(__name__)
    error = context.error
    try:
        from telegram.error import Conflict
    except Exception:  # pragma: no cover - defensive import
        Conflict = None
    if Conflict is not None and isinstance(error, Conflict):
        bot_data = getattr(getattr(context, "application", None), "bot_data", {}) or {}
        conflict_count = int(bot_data.get("telegram_conflict_count") or 0) + 1
        bot_data["telegram_conflict_count"] = conflict_count
        backoff_seconds = telegram_conflict_backoff_seconds(conflict_count)
        logger.error(
            "getUpdates conflict — another poller is active for this token. "
            "Rotate token or stop the other instance. backoff=%.1fs",
            backoff_seconds,
            exc_info=error,
        )
        await asyncio.sleep(backoff_seconds)
        return
    logger.exception("Telegram handler error", exc_info=error)
    message = getattr(update, "effective_message", None)
    if message is None:
        return
    trace_id = _trace_id_from_update(update) if isinstance(update, Update) else "tg-error"
    bot_data = getattr(getattr(context, "application", None), "bot_data", {}) or {}
    log_path = str(bot_data.get("log_path") or "").strip() or None
    envelope = _envelope_from_exception(
        exc=error if isinstance(error, Exception) else RuntimeError("TelegramHandlerError"),
        intent="telegram.error_handler",
        trace_id=trace_id,
        log_path=log_path,
    )
    try:
        await message.reply_text(_safe_reply_text(str(envelope.get("message") or _TELEGRAM_FALLBACK_TEXT)))
    except Exception:
        return


async def _check_reminders(context: ContextTypes.DEFAULT_TYPE) -> None:
    db: MemoryDB = context.application.bot_data["db"]
    log_path: str = context.application.bot_data["log_path"]
    debug_protocol: DebugProtocol | None = context.application.bot_data.get("debug_protocol")
    orchestrator: Orchestrator | None = context.application.bot_data.get("orchestrator")

    chat_id = db.get_preference("telegram_chat_id")
    if not chat_id:
        return

    now_ts = datetime.now(timezone.utc).isoformat()
    reminders = db.list_due_reminders(now_ts)
    for reminder in reminders:
        try:
            audit_id = db.audit_log_create(
                user_id=str(chat_id),
                action_type="reminder_send",
                action_id=str(reminder.id),
                status="attempted",
                details={
                    "event_type": "reminder_send",
                    "reminder_id": reminder.id,
                    "claim_attempted": True,
                    "claim_succeeded": False,
                    "send_succeeded": False,
                },
            )
        except Exception:
            return

        claim_ok = db.claim_reminder_sent(reminder.id, now_ts)
        if not claim_ok:
            try:
                db.audit_log_update_status(
                    audit_id,
                    "skipped",
                    details={
                        "event_type": "reminder_send",
                        "reminder_id": reminder.id,
                        "claim_attempted": True,
                        "claim_succeeded": False,
                        "send_succeeded": False,
                        "status_transition": "pending->skipped",
                    },
                )
            except Exception:
                return
            continue

        if debug_protocol and orchestrator:
            if debug_protocol.record_reminder(
                str(chat_id),
                reminder.text,
                datetime.now(timezone.utc),
            ):
                response = orchestrator.handle_message("/status", user_id=str(chat_id))
                await context.bot.send_message(chat_id=chat_id, text=response.text)
                db.mark_reminder_failed(reminder.id, "debug_protocol_triggered")
                try:
                    db.audit_log_update_status(
                        audit_id,
                        "failed",
                        details={
                            "event_type": "reminder_send",
                            "reminder_id": reminder.id,
                            "claim_attempted": True,
                            "claim_succeeded": True,
                            "send_succeeded": False,
                            "status_transition": "sent->failed",
                            "error": "debug_protocol_triggered",
                        },
                    )
                except Exception:
                    return
                continue

        try:
            await context.bot.send_message(chat_id=chat_id, text=f"Reminder: {reminder.text}")
            try:
                db.audit_log_update_status(
                    audit_id,
                    "executed",
                    details={
                        "event_type": "reminder_send",
                        "reminder_id": reminder.id,
                        "claim_attempted": True,
                        "claim_succeeded": True,
                        "send_succeeded": True,
                        "status_transition": "pending->sent",
                    },
                )
            except Exception:
                return
            log_event(log_path, "reminder_sent", {"reminder_id": reminder.id})
        except Exception as exc:
            db.mark_reminder_failed(reminder.id, str(exc))
            try:
                db.audit_log_update_status(
                    audit_id,
                    "failed",
                    details={
                        "event_type": "reminder_send",
                        "reminder_id": reminder.id,
                        "claim_attempted": True,
                        "claim_succeeded": True,
                        "send_succeeded": False,
                        "status_transition": "sent->failed",
                        "error": str(exc),
                    },
                )
            except Exception:
                return
            if debug_protocol and orchestrator:
                if debug_protocol.record_audit_event(
                    "reminder_send",
                    str(reminder.id),
                    "failed",
                    datetime.now(timezone.utc),
                ):
                    response = orchestrator.handle_message("/status", user_id=str(chat_id))
                    await context.bot.send_message(chat_id=chat_id, text=response.text)


async def _scheduled_disk_snapshot(context: ContextTypes.DEFAULT_TYPE) -> None:
    db: MemoryDB = context.application.bot_data["db"]
    home_path: str = context.application.bot_data["home_path"]
    safe_run_scheduled_snapshot(db, home_path, "/")


async def _scheduled_storage_snapshot(context: ContextTypes.DEFAULT_TYPE) -> None:
    db: MemoryDB = context.application.bot_data["db"]
    home_path: str = context.application.bot_data["home_path"]
    timezone: str = context.application.bot_data["timezone"]
    user_id = db.get_preference("telegram_chat_id") or "system"
    safe_run_storage_snapshot(db, timezone, home_path, user_id)


async def _scheduled_resource_snapshot(context: ContextTypes.DEFAULT_TYPE) -> None:
    db: MemoryDB = context.application.bot_data["db"]
    timezone: str = context.application.bot_data["timezone"]
    user_id = db.get_preference("telegram_chat_id") or "system"
    safe_run_resource_snapshot(db, timezone, user_id)


async def _scheduled_network_snapshot(context: ContextTypes.DEFAULT_TYPE) -> None:
    db: MemoryDB = context.application.bot_data["db"]
    timezone: str = context.application.bot_data["timezone"]
    user_id = db.get_preference("telegram_chat_id") or "system"
    safe_run_network_snapshot(db, timezone, user_id)


async def _scheduled_daily_brief(context: ContextTypes.DEFAULT_TYPE) -> None:
    db: MemoryDB = context.application.bot_data["db"]
    timezone_name: str = context.application.bot_data["timezone"]
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    chat_id = db.get_preference("telegram_chat_id")
    if not chat_id:
        return
    enabled = (db.get_preference("daily_brief_enabled") or "off").strip().lower() in {"on", "true", "1", "yes"}
    local_time = (db.get_preference("daily_brief_time") or "09:00").strip()
    quiet_mode = (db.get_preference("daily_brief_quiet_mode") or "off").strip().lower() in {"on", "true", "1", "yes"}
    threshold_pref = db.get_preference("disk_delta_threshold_mb")
    disk_delta_threshold_mb = float(threshold_pref) if (threshold_pref and threshold_pref.isdigit()) else 250.0
    svc_gate = (db.get_preference("only_send_if_service_unhealthy") or "off").strip().lower() in {
        "on",
        "true",
        "1",
        "yes",
    }
    include_due_pref = db.get_preference("include_open_loops_due_within_days")
    include_due_days = int(include_due_pref) if (include_due_pref and include_due_pref.isdigit()) else 2
    last_sent = db.get_preference("daily_brief_last_sent_date")
    payload = orchestrator.build_daily_brief_cards(str(chat_id))
    signals = payload.get("daily_brief_signals") if isinstance(payload, dict) else {}
    disk_delta_mb = signals.get("disk_delta_mb") if isinstance(signals, dict) else None
    service_unhealthy = bool(signals.get("service_unhealthy")) if isinstance(signals, dict) else False
    due_open_loops = int(signals.get("due_open_loops_count") or 0) if isinstance(signals, dict) else 0
    decision = should_send_daily_brief(
        now_utc=datetime.now(timezone.utc),
        timezone_name=timezone_name,
        enabled=enabled,
        local_time_hhmm=local_time,
        last_sent_local_date=last_sent,
        quiet_mode=quiet_mode,
        disk_delta_mb=disk_delta_mb,
        disk_delta_threshold_mb=disk_delta_threshold_mb,
        service_unhealthy=service_unhealthy,
        only_send_if_service_unhealthy=svc_gate,
        has_due_open_loops=due_open_loops > 0 and include_due_days > 0,
    )
    if not decision.should_send:
        return
    text = render_cards_markdown(payload)
    send_ok = False
    for attempt in (1, 2):
        try:
            await context.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
            send_ok = True
            break
        except Exception:
            if attempt == 1:
                await asyncio.sleep(1.0)
                continue
            return
    if send_ok:
        db.set_preference("daily_brief_last_sent_date", decision.local_date)


def register_handlers(app: Application) -> None:
    # Log exceptions to journalctl instead of swallowing them.
    app.add_error_handler(_on_error)

    # Explicit command handlers (commands should NOT go through the generic text handler).
    app.add_handler(CommandHandler("remind", _handle_remind))
    app.add_handler(CommandHandler("start", _handle_status))
    app.add_handler(CommandHandler("status", _handle_status))
    app.add_handler(CommandHandler("doctor", _handle_doctor))
    app.add_handler(CommandHandler("disk_grow", _handle_disk_grow))
    app.add_handler(CommandHandler("audit", _handle_audit))
    app.add_handler(CommandHandler("permissions", _handle_permissions))
    app.add_handler(CommandHandler("storage_snapshot", _handle_storage_snapshot))
    app.add_handler(CommandHandler("storage_report", _handle_storage_report))
    app.add_handler(CommandHandler("resource_report", _handle_resource_report))
    app.add_handler(CommandHandler("brief", _handle_brief))
    app.add_handler(CommandHandler("breif", _handle_brief_alias))
    app.add_handler(CommandHandler("help", _handle_help))
    app.add_handler(CommandHandler("model", _handle_model))
    app.add_handler(CommandHandler("network_report", _handle_network_report))
    app.add_handler(CommandHandler("sys_metrics_snapshot", _handle_sys_metrics_snapshot))
    app.add_handler(CommandHandler("sys_health_report", _handle_sys_health_report))
    app.add_handler(CommandHandler("sys_inventory_summary", _handle_sys_inventory_summary))
    app.add_handler(CommandHandler("weekly_reflection", _handle_weekly_reflection))
    app.add_handler(CommandHandler("today", _handle_today))
    app.add_handler(CommandHandler("task_add", _handle_task_add))
    app.add_handler(CommandHandler("done", _handle_done))
    app.add_handler(CommandHandler("open_loops", _handle_open_loops))
    app.add_handler(CommandHandler("health", _handle_health))
    app.add_handler(CommandHandler("memory", _handle_memory))
    app.add_handler(CommandHandler("daily_brief_status", _handle_daily_brief_status))
    app.add_handler(CommandHandler("ask", _handle_ask))
    app.add_handler(CommandHandler("ask_opinion", _handle_ask_opinion))
    app.add_handler(CommandHandler("scout", _handle_scout))
    app.add_handler(CommandHandler("scout_dismiss", _handle_scout_dismiss))
    app.add_handler(CommandHandler("scout_installed", _handle_scout_installed))

    # Non-command messages only.
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))


def build_app(
    *,
    config: Config | None = None,
    token: str | None = None,
    operator_recovery_fn: Callable[[dict[str, Any]], tuple[bool, dict[str, Any]]] | None = None,
    operator_recovery_store: OperatorRecoveryStore | None = None,
    llm_fixit_fn: Callable[[dict[str, Any]], tuple[bool, dict[str, Any]]] | None = None,
    llm_fixit_store: LLMFixitWizardStore | None = None,
    audit_log: AuditLog | None = None,
    runtime: Any | None = None,
) -> Application:
    loaded = config if isinstance(config, Config) else load_config(require_telegram_token=False)
    resolved_token = str(token or _resolve_telegram_bot_token() or "").strip()
    if not resolved_token:
        print(
            "Missing Telegram bot token. Save it in Web UI (telegram:bot_token) or set TELEGRAM_BOT_TOKEN.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
    loaded = replace(loaded, telegram_bot_token=resolved_token)
    db = MemoryDB(loaded.db_path)

    schema_path = Path(__file__).resolve().parents[1] / "memory" / "schema.sql"
    db.init_schema(str(schema_path))

    llm_client = LLMRouter(loaded, log_path=loaded.log_path)

    model_scout = build_model_scout(loaded)
    permission_store = PermissionStore(path=os.getenv("AGENT_PERMISSIONS_PATH", "").strip() or None)
    effective_audit_log = audit_log if isinstance(audit_log, AuditLog) else AuditLog(
        path=os.getenv("AGENT_AUDIT_LOG_PATH", "").strip() or None
    )

    orchestrator = Orchestrator(
        db=db,
        skills_path=loaded.skills_path,
        log_path=loaded.log_path,
        timezone=loaded.agent_timezone,
        llm_client=llm_client,
        enable_writes=loaded.enable_writes,
        perception_enabled=loaded.perception_enabled,
        perception_roots=loaded.perception_roots,
        perception_interval_seconds=loaded.perception_interval_seconds,
        runtime_truth_service=(
            runtime.runtime_truth_service()
            if runtime is not None and callable(getattr(runtime, "runtime_truth_service", None))
            else None
        ),
        chat_runtime_adapter=runtime,
    )
    debug_protocol = DebugProtocol()

    builder = Application.builder().token(loaded.telegram_bot_token)
    if hasattr(builder, "concurrent_updates"):
        builder = builder.concurrent_updates(True)
    app = builder.build()
    register_handlers(app)

    effective_operator_recovery_fn = (
        operator_recovery_fn if callable(operator_recovery_fn) else llm_fixit_fn
    )
    if not callable(effective_operator_recovery_fn) and runtime is not None:
        runtime_recovery_fn = (
            runtime.operator_recovery
            if callable(getattr(runtime, "operator_recovery", None))
            else getattr(runtime, "llm_fixit", None)
        )
        if callable(runtime_recovery_fn):
            effective_operator_recovery_fn = runtime_recovery_fn
    if not callable(effective_operator_recovery_fn):
        def _operator_recovery_internal(payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
            from agent.api_server import build_runtime

            runtime = build_runtime(config=load_config(require_telegram_token=False))
            try:
                handler = (
                    runtime.operator_recovery
                    if callable(getattr(runtime, "operator_recovery", None))
                    else runtime.llm_fixit
                )
                return handler(payload if isinstance(payload, dict) else {})
            finally:
                runtime.close()

        effective_operator_recovery_fn = _operator_recovery_internal
    runtime_recovery_store = (
        runtime.operator_recovery_store()
        if runtime is not None and callable(getattr(runtime, "operator_recovery_store", None))
        else None
    )
    effective_operator_recovery_store = (
        operator_recovery_store
        if isinstance(operator_recovery_store, OperatorRecoveryStore)
        else (
            llm_fixit_store
            if isinstance(llm_fixit_store, LLMFixitWizardStore)
            else (
                runtime_recovery_store
                if isinstance(runtime_recovery_store, OperatorRecoveryStore)
                else OperatorRecoveryStore()
            )
        )
    )

    app.bot_data["db"] = db
    app.bot_data["orchestrator"] = orchestrator
    app.bot_data["debug_protocol"] = debug_protocol
    app.bot_data["log_path"] = loaded.log_path
    app.bot_data["home_path"] = os.path.expanduser("~")
    app.bot_data["timezone"] = loaded.agent_timezone
    app.bot_data["model_scout"] = model_scout
    app.bot_data["permission_store"] = permission_store
    app.bot_data["audit_log"] = effective_audit_log
    app.bot_data["operator_recovery_store"] = effective_operator_recovery_store
    app.bot_data["operator_recovery_fn"] = effective_operator_recovery_fn
    app.bot_data["runtime"] = runtime
    app.bot_data["runtime_version"] = str(
        getattr(runtime, "version", "") or getattr(loaded, "api_version", "") or "0.1.0"
    )
    app.bot_data["runtime_git_commit"] = str(getattr(runtime, "git_commit", "") or "unknown")
    app.bot_data["runtime_started_ts"] = float(pytime.time())

    app.job_queue.run_repeating(_check_reminders, interval=30, first=5)
    if loaded.enable_scheduled_snapshots:
        run_time = time(9, 0, tzinfo=ZoneInfo(loaded.agent_timezone))
        app.job_queue.run_daily(_scheduled_disk_snapshot, time=run_time, name="disk_snapshot_daily")
        app.job_queue.run_daily(_scheduled_storage_snapshot, time=run_time, name="storage_snapshot_daily")
        app.job_queue.run_daily(_scheduled_resource_snapshot, time=run_time, name="resource_snapshot_daily")
        app.job_queue.run_daily(_scheduled_network_snapshot, time=run_time, name="network_snapshot_daily")
    return app


def run_polling_with_backoff(
    *,
    token: str,
    token_source: str,
    config: Config | None = None,
    app_factory: Callable[..., Application] | None = None,
    sleep_fn: Callable[[float], None] = pytime.sleep,
    max_conflict_retries: int | None = None,
) -> int:
    builder = app_factory or build_app
    conflict_retries = 0
    allowed_updates = getattr(Update, "ALL_TYPES", None)
    while True:
        app = builder(config=config, token=token)
        try:
            polling_kwargs: dict[str, Any] = {"drop_pending_updates": True}
            if allowed_updates is not None:
                polling_kwargs["allowed_updates"] = allowed_updates
            app.run_polling(**polling_kwargs)
            return 0
        except KeyboardInterrupt:
            return 0
        except Exception as exc:
            if not is_telegram_conflict_error(exc):
                raise
            conflict_retries += 1
            backoff_seconds = telegram_conflict_backoff_seconds(conflict_retries)
            payload = {
                "error_type": exc.__class__.__name__,
                "error": str(exc),
                "token_source": token_source,
                "backoff_seconds": backoff_seconds,
                "retry": conflict_retries,
            }
            _LOGGER.error(
                "Telegram getUpdates conflict — another poller is active for this token. "
                "Rotate token or stop the other instance. %s",
                json.dumps(payload, ensure_ascii=True, sort_keys=True),
            )
            if max_conflict_retries is not None and conflict_retries >= int(max_conflict_retries):
                return 0
            sleep_fn(backoff_seconds)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Personal Agent Telegram adapter.")
    parser.parse_args([] if argv is None else argv)
    configure_logging_if_needed()
    loaded = load_config(require_telegram_token=False)
    runtime_state = get_telegram_runtime_state()
    if not bool(runtime_state.get("enabled", False)):
        _LOGGER.info(
            "telegram.disabled %s",
            json.dumps(
                {
                    "mode": "adapter",
                    "reason": "config_disabled",
                    "env": "TELEGRAM_ENABLED",
                    "config_source": str(runtime_state.get("config_source") or "default"),
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
        )
        return
    token, token_source = resolve_telegram_bot_token_with_source()
    startup_report = run_startup_checks(service="telegram", config=loaded, token=token)
    for row in (startup_report.get("checks") if isinstance(startup_report.get("checks"), list) else []):
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "").strip().upper()
        if status not in {"WARN", "FAIL"}:
            continue
        log_fn = _LOGGER.error if status == "FAIL" else _LOGGER.warning
        log_fn(
            "telegram.startup.check %s",
            json.dumps(
                {
                    "trace_id": startup_report.get("trace_id"),
                    "component": startup_report.get("component"),
                    "check_id": row.get("check_id"),
                    "status": status,
                    "failure_code": row.get("failure_code"),
                    "next_action": row.get("next_action"),
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
        )
    if str(startup_report.get("status") or "").strip().upper() == "FAIL":
        _LOGGER.error(
            deterministic_error_message(
                title="❌ Startup checks failed",
                trace_id=str(startup_report.get("trace_id") or "startup-telegram-unknown"),
                component=str(startup_report.get("component") or "telegram.startup"),
                failure_code=str(startup_report.get("failure_code") or "startup_check_failed"),
                next_action=str(startup_report.get("next_action") or "Run: python -m agent doctor"),
            )
        )
        raise SystemExit(1)

    if not token:
        _LOGGER.error("telegram.startup.token_missing")
        raise SystemExit(1)
    stale_removed = clear_stale_telegram_locks(token)
    if stale_removed:
        _LOGGER.info(
            "telegram.lock_cleared %s",
            json.dumps(
                {
                    "removed": stale_removed,
                    "token_source": token_source,
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
        )
    poll_lock = acquire_telegram_poll_lock(token)
    if poll_lock is None:
        lock_state = get_telegram_runtime_state()
        warning_payload = {
            "pid": os.getpid(),
            "lock_path": str(telegram_poll_lock_path(token)),
            "token_source": token_source,
            "lock_stale": bool(lock_state.get("lock_stale", False)),
            "lock_live": bool(lock_state.get("lock_live", False)),
        }
        _LOGGER.warning(
            "Telegram polling already active for this token; exiting. %s",
            json.dumps(warning_payload, ensure_ascii=True, sort_keys=True),
        )
        return
    try:
        run_polling_with_backoff(token=token, token_source=token_source)
    finally:
        release_telegram_poll_lock(poll_lock)


def run() -> None:
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
