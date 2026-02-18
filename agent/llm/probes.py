from __future__ import annotations

import json
import socket
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request

from agent.llm.capabilities import is_embedding_model_name


ProbeResult = dict[str, Any]


def _normalize_status(value: str) -> str:
    status = str(value or "").strip().lower()
    if status in {"ok", "degraded", "down"}:
        return status
    return "degraded"


def _result(
    *,
    status: str,
    error_kind: str | None,
    status_code: int | None,
    detail: str,
    started: float,
) -> ProbeResult:
    return {
        "status": _normalize_status(status),
        "error_kind": str(error_kind or "").strip().lower() or None,
        "status_code": int(status_code) if status_code is not None else None,
        "detail": str(detail or "").strip() or "n/a",
        "duration_ms": max(0, int((time.monotonic() - started) * 1000)),
    }


def _http_get_json(url: str, *, timeout_seconds: float, headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET", headers=headers)
    with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
        raw = response.read().decode("utf-8")
    payload = json.loads(raw or "{}")
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return {"data": payload}
    return {}


def _http_post_json(
    url: str,
    *,
    payload: dict[str, Any],
    timeout_seconds: float,
    headers: dict[str, str],
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    request_headers = {"Content-Type": "application/json", **headers}
    request = urllib.request.Request(url, method="POST", headers=request_headers, data=body)
    with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw or "{}")
    return parsed if isinstance(parsed, dict) else {}


def _http_error_result(exc: Exception, *, started: float) -> ProbeResult:
    if isinstance(exc, urllib.error.HTTPError):
        status_code = int(getattr(exc, "code", 0) or 0) or None
        if status_code in {401, 403}:
            return _result(
                status="down",
                error_kind="missing_auth",
                status_code=status_code,
                detail="auth rejected",
                started=started,
            )
        if status_code == 404:
            return _result(
                status="down",
                error_kind="misconfigured_path",
                status_code=status_code,
                detail="endpoint not found",
                started=started,
            )
        if status_code == 429:
            return _result(
                status="degraded",
                error_kind="rate_limit",
                status_code=status_code,
                detail="rate limited",
                started=started,
            )
        if status_code is not None and 500 <= status_code <= 599:
            return _result(
                status="degraded",
                error_kind="server_error",
                status_code=status_code,
                detail="server error",
                started=started,
            )
        return _result(
            status="degraded",
            error_kind="bad_request",
            status_code=status_code,
            detail="request failed",
            started=started,
        )
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return _result(
            status="degraded",
            error_kind="timeout",
            status_code=None,
            detail="request timeout",
            started=started,
        )
    if isinstance(exc, urllib.error.URLError):
        return _result(
            status="degraded",
            error_kind="unreachable",
            status_code=None,
            detail="provider unreachable",
            started=started,
        )
    return _result(
        status="degraded",
        error_kind="provider_error",
        status_code=None,
        detail="probe failed",
        started=started,
    )


def _provider_identity(cfg: dict[str, Any]) -> str:
    provider_id = str(cfg.get("id") or cfg.get("provider_id") or "").strip().lower()
    if provider_id:
        return provider_id
    if bool(cfg.get("local", False)):
        return "ollama"
    return str(cfg.get("provider_type") or "openai_compat").strip().lower() or "openai_compat"


def _normalized_base_url(cfg: dict[str, Any]) -> str:
    return str(cfg.get("base_url") or "").strip().rstrip("/")


def _normalized_headers(cfg: dict[str, Any]) -> dict[str, str]:
    raw = cfg.get("headers") if isinstance(cfg.get("headers"), dict) else {}
    return {
        str(key).strip(): str(value).strip()
        for key, value in raw.items()
        if str(key).strip() and str(value).strip()
    }


def _auth_required(cfg: dict[str, Any]) -> bool:
    return isinstance(cfg.get("api_key_source"), dict)


def _auth_present(cfg: dict[str, Any], headers: dict[str, str]) -> bool:
    auth_header = str(headers.get("Authorization") or "").strip()
    if auth_header.lower().startswith("bearer ") and len(auth_header) > len("bearer "):
        return True
    return bool(cfg.get("_resolved_api_key_present", False))


def _remote_probe_allowed(cfg: dict[str, Any]) -> bool:
    if bool(cfg.get("local", False)):
        return True
    return bool(cfg.get("allow_remote_fallback", True))


def _chat_capable(model_id: str, model_capabilities: Any) -> bool:
    if is_embedding_model_name(model_id):
        return False
    caps = {
        str(item).strip().lower()
        for item in (model_capabilities if isinstance(model_capabilities, list) else [])
        if str(item).strip()
    }
    if caps:
        return "chat" in caps
    return not is_embedding_model_name(model_id)


def probe_provider(
    provider_cfg: dict[str, Any],
    *,
    timeout_seconds: float = 6.0,
    http_get_json: Callable[..., dict[str, Any]] | None = None,
) -> ProbeResult:
    cfg = provider_cfg if isinstance(provider_cfg, dict) else {}
    started = time.monotonic()

    if not bool(cfg.get("enabled", True)):
        return _result(
            status="down",
            error_kind="provider_disabled",
            status_code=None,
            detail="provider disabled",
            started=started,
        )
    if cfg.get("available") is False:
        return _result(
            status="down",
            error_kind="provider_unavailable",
            status_code=None,
            detail="provider unavailable",
            started=started,
        )
    if not _remote_probe_allowed(cfg):
        return _result(
            status="ok",
            error_kind="not_applicable",
            status_code=None,
            detail="remote probe disabled by policy",
            started=started,
        )

    base_url = _normalized_base_url(cfg)
    parsed = urllib.parse.urlparse(base_url)
    if str(parsed.scheme or "").strip().lower() not in {"http", "https"} or not str(parsed.hostname or "").strip():
        return _result(
            status="down",
            error_kind="bad_base_url",
            status_code=None,
            detail="invalid base_url",
            started=started,
        )

    headers = _normalized_headers(cfg)
    if _auth_required(cfg) and not _auth_present(cfg, headers):
        return _result(
            status="down",
            error_kind="missing_auth",
            status_code=None,
            detail="missing authorization",
            started=started,
        )

    provider_id = _provider_identity(cfg)
    endpoint = "/api/tags" if provider_id == "ollama" or bool(cfg.get("local", False)) else "/v1/models"
    getter = http_get_json or _http_get_json
    try:
        getter(base_url + endpoint, timeout_seconds=float(timeout_seconds), headers=headers)
    except Exception as exc:  # pragma: no cover - mapped below
        return _http_error_result(exc, started=started)
    return _result(
        status="ok",
        error_kind=None,
        status_code=None,
        detail="provider probe ok",
        started=started,
    )


def probe_model(
    provider_cfg: dict[str, Any],
    model_id: str,
    *,
    timeout_seconds: float = 6.0,
    model_capabilities: Any = None,
    http_post_json: Callable[..., dict[str, Any]] | None = None,
) -> ProbeResult:
    cfg = provider_cfg if isinstance(provider_cfg, dict) else {}
    started = time.monotonic()

    if not bool(cfg.get("enabled", True)):
        return _result(
            status="down",
            error_kind="provider_disabled",
            status_code=None,
            detail="provider disabled",
            started=started,
        )
    if cfg.get("available") is False:
        return _result(
            status="down",
            error_kind="provider_unavailable",
            status_code=None,
            detail="provider unavailable",
            started=started,
        )
    if not _remote_probe_allowed(cfg):
        return _result(
            status="ok",
            error_kind="not_applicable",
            status_code=None,
            detail="remote probe disabled by policy",
            started=started,
        )

    provider_id = _provider_identity(cfg)
    model_name = str(model_id or "").strip()
    if ":" in model_name and model_name.split(":", 1)[0].strip().lower() == provider_id:
        model_name = model_name.split(":", 1)[1].strip()
    if not model_name:
        return _result(
            status="degraded",
            error_kind="bad_request",
            status_code=None,
            detail="missing model id",
            started=started,
        )

    if not _chat_capable(model_name, model_capabilities):
        return _result(
            status="ok",
            error_kind="not_applicable",
            status_code=None,
            detail="model not chat capable",
            started=started,
        )

    base_url = _normalized_base_url(cfg)
    parsed = urllib.parse.urlparse(base_url)
    if str(parsed.scheme or "").strip().lower() not in {"http", "https"} or not str(parsed.hostname or "").strip():
        return _result(
            status="down",
            error_kind="bad_base_url",
            status_code=None,
            detail="invalid base_url",
            started=started,
        )

    headers = _normalized_headers(cfg)
    if _auth_required(cfg) and not _auth_present(cfg, headers):
        return _result(
            status="down",
            error_kind="missing_auth",
            status_code=None,
            detail="missing authorization",
            started=started,
        )

    if provider_id == "ollama" or bool(cfg.get("local", False)):
        chat_path = "/v1/chat/completions"
    else:
        chat_path = str(cfg.get("chat_path") or "/v1/chat/completions").strip() or "/v1/chat/completions"
        if not chat_path.startswith("/"):
            chat_path = "/" + chat_path
    poster = http_post_json or _http_post_json
    request_payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "ping"}],
        "temperature": 0,
        "max_tokens": 1,
    }
    try:
        poster(
            base_url + chat_path,
            payload=request_payload,
            timeout_seconds=float(timeout_seconds),
            headers=headers,
        )
    except Exception as exc:  # pragma: no cover - mapped below
        return _http_error_result(exc, started=started)
    return _result(
        status="ok",
        error_kind=None,
        status_code=None,
        detail="model probe ok",
        started=started,
    )


__all__ = [
    "ProbeResult",
    "probe_model",
    "probe_provider",
]
