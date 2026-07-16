from __future__ import annotations

import logging
import re
from typing import Any, Iterable


_SECRET_KEY_TOKENS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "bot_token",
    "password",
    "passphrase",
    "secret",
    "telegram_bot_token",
    "token",
)
_SAFE_SECRETISH_KEYS = {
    "token_source",
    "secret_store_state",
    "secret_store_valid",
    "secret_store_error_kind",
}

_TELEGRAM_API_URL_RE = re.compile(
    r"\b(?P<scheme>https?)://api\.telegram\.org/bot(?P<token>[^/\s\"'<>]+)(?P<path>/[A-Za-z0-9_./-]*)?",
    re.IGNORECASE,
)
_TELEGRAM_BOT_TOKEN_RE = re.compile(r"\bbot(?P<token>\d{5,}:[A-Za-z0-9_-]{6,})\b", re.IGNORECASE)
_TELEGRAM_TOKEN_RE = re.compile(r"\b\d{5,}:[A-Za-z0-9_-]{6,}\b")
_OPENAI_STYLE_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b")
_AUTH_BEARER_RE = re.compile(r"(?i)\b(authorization\s*:\s*bearer\s+)([A-Za-z0-9._~+/=-]+)")
_PARAM_SECRET_RE = re.compile(
    r"(?i)\b(?P<key>api[_-]?key|access[_-]?token|auth[_-]?token|bot[_-]?token|token|secret|password)="
    r"(?P<value>[^&\s\"'<>]+)"
)


def _looks_secret_key(key: str) -> bool:
    lowered = str(key or "").strip().lower()
    if lowered in _SAFE_SECRETISH_KEYS:
        return False
    return any(token in lowered for token in _SECRET_KEY_TOKENS)


def redact_text(value: str | None, *, known_secrets: Iterable[str] | None = None) -> str:
    """Redact known secret formats from a free-form string.

    The function is intentionally conservative except for explicit known secret
    values supplied by callers that already loaded a secret from a trusted store.
    It never raises; log paths must stay robust even during exception handling.
    """

    try:
        text = "" if value is None else str(value)
        for secret in known_secrets or ():
            secret_text = str(secret or "")
            if secret_text:
                text = text.replace(secret_text, "<redacted-secret>")

        def _telegram_url(match: re.Match[str]) -> str:
            path = match.group("path") or ""
            return f"{match.group('scheme')}://api.telegram.org/bot<redacted>{path}"

        text = _TELEGRAM_API_URL_RE.sub(_telegram_url, text)
        text = _TELEGRAM_BOT_TOKEN_RE.sub("bot<redacted>", text)
        text = _TELEGRAM_TOKEN_RE.sub("<redacted-telegram-token>", text)
        text = _OPENAI_STYLE_KEY_RE.sub("<redacted-api-key>", text)
        text = _AUTH_BEARER_RE.sub(r"\1<redacted>", text)
        text = _PARAM_SECRET_RE.sub(lambda m: f"{m.group('key')}=<redacted>", text)
        return text
    except Exception:
        return "<redaction-error>"


def redact_value(value: Any, *, known_secrets: Iterable[str] | None = None) -> Any:
    """Redact strings inside nested JSON-like structures."""

    try:
        if isinstance(value, dict):
            output: dict[str, Any] = {}
            for key, item in value.items():
                key_text = str(key)
                if _looks_secret_key(key_text):
                    output[key_text] = "***redacted***"
                else:
                    output[key_text] = redact_value(item, known_secrets=known_secrets)
            return output
        if isinstance(value, list):
            return [redact_value(item, known_secrets=known_secrets) for item in value]
        if isinstance(value, tuple):
            return tuple(redact_value(item, known_secrets=known_secrets) for item in value)
        if isinstance(value, str):
            return redact_text(value, known_secrets=known_secrets)
        return value
    except Exception:
        return "<redaction-error>"


class RedactingFormatter(logging.Formatter):
    """Formatter that redacts the final rendered log line before emission."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        rendered = super().format(record)
        return redact_text(rendered)
