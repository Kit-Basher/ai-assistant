from __future__ import annotations

from typing import Any
import urllib.parse


def normalize_provider_url(base_url: str, chat_path: str) -> tuple[str, str]:
    normalized_base = str(base_url or "").strip().rstrip("/")
    normalized_path = str(chat_path or "").strip() or "/v1/chat/completions"
    if not normalized_path.startswith("/"):
        normalized_path = "/" + normalized_path
    if normalized_base.endswith("/v1") and normalized_path.startswith("/v1/"):
        normalized_base = normalized_base[:-3].rstrip("/")
    return normalized_base, normalized_path


def validate_provider_call_format(
    provider_id: str,
    provider_payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    provider = str(provider_id or "").strip().lower()
    payload = provider_payload if isinstance(provider_payload, dict) else {}
    base_url_raw = str(payload.get("base_url") or "").strip()
    chat_path_raw = str(payload.get("chat_path") or "").strip() or "/v1/chat/completions"
    normalized_base, normalized_path = normalize_provider_url(base_url_raw, chat_path_raw)

    parsed = urllib.parse.urlparse(normalized_base)
    scheme = str(parsed.scheme or "").strip().lower()
    host = str(parsed.hostname or "").strip()
    if scheme not in {"http", "https"} or not host:
        return {
            "ok": False,
            "error_kind": "bad_base_url",
            "message": "provider base_url must be an absolute http(s) URL",
            "details": {
                "provider_id": provider,
                "base_url": base_url_raw,
                "chat_path": chat_path_raw,
            },
        }

    if not normalized_path.startswith("/"):
        return {
            "ok": False,
            "error_kind": "misconfigured_path",
            "message": "chat_path must start with '/'",
            "details": {
                "provider_id": provider,
                "base_url": normalized_base,
                "chat_path": chat_path_raw,
            },
        }

    if normalized_path.count("/v1/v1/") > 0:
        return {
            "ok": False,
            "error_kind": "misconfigured_path",
            "message": "chat_path includes duplicated /v1 segment",
            "details": {
                "provider_id": provider,
                "base_url": normalized_base,
                "chat_path": normalized_path,
            },
        }

    if bool(payload.get("local", False)):
        local_allowed = {"/v1/chat/completions", "/api/chat"}
        if normalized_path not in local_allowed:
            return {
                "ok": False,
                "error_kind": "misconfigured_path",
                "message": "local provider chat_path should be /v1/chat/completions or /api/chat",
                "details": {
                    "provider_id": provider,
                    "base_url": normalized_base,
                    "chat_path": normalized_path,
                },
            }
    else:
        if not normalized_path.endswith("/chat/completions"):
            return {
                "ok": False,
                "error_kind": "misconfigured_path",
                "message": "openai-compatible chat_path should end with /chat/completions",
                "details": {
                    "provider_id": provider,
                    "base_url": normalized_base,
                    "chat_path": normalized_path,
                },
            }

    source = payload.get("api_key_source") if isinstance(payload.get("api_key_source"), dict) else None
    if isinstance(source, dict):
        key_present = False
        request_headers = headers if isinstance(headers, dict) else {}
        auth_header = str(request_headers.get("Authorization") or "").strip()
        if auth_header.lower().startswith("bearer ") and len(auth_header) > len("bearer "):
            key_present = True
        if bool(payload.get("_resolved_api_key_present", False)):
            key_present = True
        if not key_present:
            return {
                "ok": False,
                "error_kind": "missing_auth",
                "message": "provider requires Authorization header from configured key source",
                "details": {
                    "provider_id": provider,
                    "base_url": normalized_base,
                    "chat_path": normalized_path,
                },
            }

    return {
        "ok": True,
        "error_kind": None,
        "message": "ok",
        "details": {
            "provider_id": provider,
            "base_url": normalized_base,
            "chat_path": normalized_path,
        },
    }


__all__ = [
    "normalize_provider_url",
    "validate_provider_call_format",
]
