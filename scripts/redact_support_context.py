#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


_TELEGRAM_BOT_URL_RE = re.compile(r"(?i)(https?://api\.telegram\.org/bot)[^/\s\"'<>]+")
_BOT_PATH_TOKEN_RE = re.compile(r"(?i)(/bot)[A-Za-z0-9:_-]{12,}(?=$|[/?#\s\"'<>])")
_TELEGRAM_TOKEN_RE = re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{20,}\b")
_OPENROUTER_KEY_RE = re.compile(r"\bsk-or(?:-v1)?-[A-Za-z0-9_-]{16,}\b")
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")
_BEARER_RE = re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[A-Za-z0-9._~+/=-]+")
_SECRET_STORE_VALUE_RE = re.compile(r"(?i)(secret[-_ ]store(?: value)?\s*[:=]\s*)[^\s,}]+")
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


def redact_text(text: str) -> str:
    redacted = str(text or "")
    redacted = _TELEGRAM_BOT_URL_RE.sub(r"\1<redacted>", redacted)
    redacted = _BOT_PATH_TOKEN_RE.sub(r"\1<redacted>", redacted)
    redacted = _BEARER_RE.sub(r"\1<redacted>", redacted)
    redacted = _SECRET_STORE_VALUE_RE.sub(r"\1<redacted>", redacted)
    redacted = _SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group('prefix')}<redacted>", redacted)
    redacted = _OPENROUTER_KEY_RE.sub("<redacted>", redacted)
    redacted = _OPENAI_KEY_RE.sub("<redacted>", redacted)
    redacted = _TELEGRAM_TOKEN_RE.sub("<redacted>", redacted)
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
