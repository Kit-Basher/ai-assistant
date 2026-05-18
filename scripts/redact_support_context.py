#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path
import urllib.parse


_TELEGRAM_BOT_URL_RE = re.compile(r"(?i)(https?://api\.telegram\.org/bot)[^/\s\"'<>]+")
_BOT_PATH_TOKEN_RE = re.compile(r"(?i)(/bot)[A-Za-z0-9:_-]{12,}(?=$|[/?#\s\"'<>])")
_TELEGRAM_TOKEN_RE = re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{20,}\b")
_OPENROUTER_KEY_RE = re.compile(r"\bsk-or(?:-v1)?-[A-Za-z0-9_-]{16,}\b")
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")
_BEARER_RE = re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[A-Za-z0-9._~+/=-]+")
_SECRET_STORE_VALUE_RE = re.compile(r"(?i)(secret[-_ ]store(?: value)?\s*[:=]\s*)[^\s,}]+")
_PRIVATE_HISTORY_PATH_RE = re.compile(r"(?i)(?:~|/).{0,180}(?:watch-history|youtube-history|history).{0,80}\.(?:json|html)")
_LOCAL_PRIVATE_PATH_RE = re.compile(r"(?i)(?:~|/home/[^/\s\"'<>]+|/Users/[^/\s\"'<>]+)/(?:[^,\s\"'<>]){0,240}")
_IMPORTED_PACK_TEXT_FIELD_RE = re.compile(
    r"""(?is)
    (["']?(?:skill_text|skill_md|readme|readme_md|imported_guidance|source_skill_text|raw_catalog_entry|raw_manifest)["']?\s*[:=]\s*)
    (?:
        "(?:\\.|[^"\\])*"
        |'(?:\\.|[^'\\])*'
        |\{.*?\}
        |\[.*?\]
        |[^\n\r]+
    )
    """
)
_IMPORTED_PACK_JSON_STRING_RE = re.compile(
    r"""(?is)(["'](?:skill_text|skill_md|readme|readme_md|imported_guidance|source_skill_text|raw_catalog_entry|raw_manifest)["']\s*:\s*)["'](?:\\.|[^"'\\])*["']"""
)
_URL_RE = re.compile(r"https?://[^\s\"'<>]+")
_SECRET_QUERY_KEYS = {"token", "key", "api_key", "access_token", "auth", "signature", "sig"}
_SECRET_ASSIGNMENT_RE = re.compile(
    r"""(?ix)
    (?P<prefix>
        ["']?
        (?:
            [A-Z0-9_-]*TOKEN[A-Z0-9_-]*
            |[A-Z0-9_-]*API[_-]?KEY[A-Z0-9_-]*
            |[A-Z0-9_-]*SECRET[A-Z0-9_-]*
            |telegram:bot_token
            |openai:api_key
            |openrouter:api_key
        )
        ["']?
        \s*[:=]\s*
        ["']?
    )
    (?P<value>[^"'\s,}]+)
    """
)


def _sanitize_url(match: re.Match[str]) -> str:
    raw = match.group(0)
    try:
        parsed = urllib.parse.urlsplit(raw)
    except ValueError:
        return raw
    if not parsed.scheme or not parsed.netloc:
        return raw
    host = str(parsed.hostname or "")
    if not host:
        return raw
    port = f":{parsed.port}" if parsed.port else ""
    query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    safe_pairs = [
        (key, "<redacted>" if key.lower() in _SECRET_QUERY_KEYS or "token" in key.lower() or "secret" in key.lower() or "key" in key.lower() else value)
        for key, value in query_pairs
    ]
    return urllib.parse.urlunsplit((parsed.scheme, f"{host}{port}", parsed.path, urllib.parse.urlencode(safe_pairs), parsed.fragment))


def redact_text(text: str) -> str:
    redacted = str(text or "")
    redacted = _IMPORTED_PACK_JSON_STRING_RE.sub(lambda match: f"{match.group(1)}\"<redacted-imported-pack-text>\"", redacted)
    redacted = _IMPORTED_PACK_TEXT_FIELD_RE.sub(lambda match: f"{match.group(1)}<redacted-imported-pack-text>", redacted)
    redacted = _TELEGRAM_BOT_URL_RE.sub(r"\1<redacted>", redacted)
    redacted = _BOT_PATH_TOKEN_RE.sub(r"\1<redacted>", redacted)
    redacted = _BEARER_RE.sub(r"\1<redacted>", redacted)
    redacted = _SECRET_STORE_VALUE_RE.sub(r"\1<redacted>", redacted)
    redacted = _SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group('prefix')}<redacted>", redacted)
    redacted = _OPENROUTER_KEY_RE.sub("<redacted>", redacted)
    redacted = _OPENAI_KEY_RE.sub("<redacted>", redacted)
    redacted = _TELEGRAM_TOKEN_RE.sub("<redacted>", redacted)
    redacted = _PRIVATE_HISTORY_PATH_RE.sub("<redacted-local-history-path>", redacted)
    redacted = _URL_RE.sub(_sanitize_url, redacted)
    redacted = _LOCAL_PRIVATE_PATH_RE.sub("<redacted-local-path>", redacted)
    return redacted


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) > 1:
        print("usage: redact_support_context.py [path]", file=sys.stderr)
        return 2
    if args:
        text = Path(args[0]).read_text(encoding="utf-8", errors="replace")
    else:
        text = sys.stdin.read()
    sys.stdout.write(redact_text(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
