from __future__ import annotations

import json
import re
import socket
from typing import Any, Callable
import urllib.error
import urllib.parse


def _clean_path(path: str) -> str:
    collapsed = re.sub(r"/{2,}", "/", str(path or ""))
    segments = [segment for segment in collapsed.split("/") if segment]
    if segments and segments[-1].lower() == "v1":
        segments = segments[:-1]
    if not segments:
        return ""
    return "/" + "/".join(segments)


def _build_base(parsed: urllib.parse.ParseResult, path: str) -> str:
    path_value = str(path or "")
    return urllib.parse.urlunparse(
        (
            str(parsed.scheme or "").strip().lower(),
            str(parsed.netloc or "").strip(),
            path_value,
            "",
            "",
            "",
        )
    ).rstrip("/")


def normalize_ollama_base_urls(base_url: str) -> dict[str, str]:
    configured = str(base_url or "").strip().rstrip("/")
    parsed = urllib.parse.urlparse(configured)
    native_path = _clean_path(parsed.path)
    openai_path = f"{native_path}/v1" if native_path else "/v1"
    native_base = _build_base(parsed, native_path)
    openai_base = _build_base(parsed, openai_path)
    return {
        "configured_base_url": configured,
        "native_base": native_base,
        "openai_base": openai_base,
    }


def join_base(base_url: str, path: str) -> str:
    return str(base_url or "").rstrip("/") + "/" + str(path or "").lstrip("/")


def _exception_details(exc: Exception) -> tuple[str, int | None, str]:
    if isinstance(exc, urllib.error.HTTPError):
        status_code = int(getattr(exc, "code", 0) or 0) or None
        return "bad_status_code", status_code, f"http_{status_code or 0}"
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return "timeout", None, "timeout"
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        reason_text = str(reason or exc).strip().lower()
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return "timeout", None, "timeout"
        if isinstance(reason, ConnectionRefusedError) or "connection refused" in reason_text:
            return "connection_refused", None, "connection_refused"
        return "connection_error", None, "connection_error"
    if isinstance(exc, json.JSONDecodeError):
        return "invalid_json", None, "invalid_json"
    if isinstance(exc, ValueError):
        return "invalid_json", None, "invalid_json"
    return "provider_error", None, "provider_error"


def probe_ollama_connectivity(
    *,
    base_url: str,
    timeout_seconds: float,
    headers: dict[str, str],
    http_get_json: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    endpoints = normalize_ollama_base_urls(base_url)
    native_base = str(endpoints.get("native_base") or "").strip()
    openai_base = str(endpoints.get("openai_base") or "").strip()
    native_ok = False
    openai_compat_ok = False
    last_error_kind: str | None = None
    last_status_code: int | None = None
    detail = "native_probe_ok"

    try:
        http_get_json(
            join_base(native_base, "/api/tags"),
            timeout_seconds=float(timeout_seconds),
            headers=headers,
        )
        native_ok = True
    except Exception as exc:  # pragma: no cover - mapped deterministically
        error_kind, status_code, mapped_detail = _exception_details(exc)
        return {
            **endpoints,
            "native_ok": False,
            "openai_compat_ok": False,
            "last_error_kind": error_kind,
            "last_status_code": status_code,
            "detail": mapped_detail,
        }

    try:
        http_get_json(
            join_base(openai_base, "/models"),
            timeout_seconds=float(timeout_seconds),
            headers=headers,
        )
        openai_compat_ok = True
    except Exception as exc:  # pragma: no cover - mapped deterministically
        error_kind, status_code, mapped_detail = _exception_details(exc)
        openai_compat_ok = False
        last_error_kind = error_kind
        last_status_code = status_code
        detail = f"openai_compat_{mapped_detail}"

    return {
        **endpoints,
        "native_ok": bool(native_ok),
        "openai_compat_ok": bool(openai_compat_ok),
        "last_error_kind": last_error_kind,
        "last_status_code": last_status_code,
        "detail": detail,
    }


__all__ = [
    "join_base",
    "normalize_ollama_base_urls",
    "probe_ollama_connectivity",
]
