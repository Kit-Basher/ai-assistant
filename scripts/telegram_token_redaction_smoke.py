#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import logging
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.logging_utils import log_event
from agent.security.redaction import RedactingFormatter, redact_text, redact_value


FAKE_TOKEN = "123456:TEST_TOKEN_REDACT_ME"
MESSAGE_BODY = "private message body must not be logged"


def _check(name: str, condition: bool, evidence: str) -> tuple[str, bool, str]:
    return (name, bool(condition), evidence)


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    url = f"https://api.telegram.org/bot{FAKE_TOKEN}/getUpdates"
    redacted_url = redact_text(url)
    checks.append(_check("Telegram API URL redacted", FAKE_TOKEN not in redacted_url and redacted_url.endswith("/getUpdates"), redacted_url))
    checks.append(_check("method name remains visible", "getUpdates" in redacted_url, redacted_url))

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(RedactingFormatter("%(levelname)s %(name)s %(message)s"))
    logger = logging.getLogger("telegram_token_redaction_smoke.httpx")
    old_handlers = list(logger.handlers)
    old_propagate = logger.propagate
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    try:
        logger.info("HTTP Request: POST %s \"HTTP/1.1 200 OK\"", url)
        logger.exception("Telegram exception for %s", url, exc_info=RuntimeError(f"failed {url}"))
    finally:
        logger.handlers = old_handlers
        logger.propagate = old_propagate
    captured = stream.getvalue()
    checks.append(_check("HTTP request log redacted", FAKE_TOKEN not in captured and "bot<redacted>" in captured, captured[:200]))
    checks.append(_check("exception text redacted", FAKE_TOKEN not in captured, captured[:200]))

    diagnostic = redact_value({"last_error_summary": f"Conflict at {url}", "message_len": len(MESSAGE_BODY)}, known_secrets=[FAKE_TOKEN])
    rendered_diag = json.dumps(diagnostic, sort_keys=True)
    checks.append(_check("diagnostic output redacted", FAKE_TOKEN not in rendered_diag and MESSAGE_BODY not in rendered_diag and "getUpdates" in rendered_diag, rendered_diag))

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = str(Path(tmpdir) / "events.jsonl")
        log_event(
            log_path,
            "telegram_update_received",
            {
                "url": url,
                "telegram_bot_token": FAKE_TOKEN,
                "message_len": len(MESSAGE_BODY),
                "update_payload_present": False,
            },
        )
        event_text = Path(log_path).read_text(encoding="utf-8")
    checks.append(_check("support/audit-style event redacted", FAKE_TOKEN not in event_text and MESSAGE_BODY not in event_text, event_text))

    arbitrary = redact_text(f"known token value {FAKE_TOKEN}", known_secrets=[FAKE_TOKEN])
    checks.append(_check("known token redacted in arbitrary text", FAKE_TOKEN not in arbitrary, arbitrary))
    malformed = redact_value({"bad": object()})
    checks.append(_check("redaction does not crash on malformed input", isinstance(malformed, dict), str(malformed)))

    leaks = sum(1 for _, ok, evidence in checks if not ok or FAKE_TOKEN in evidence or MESSAGE_BODY in evidence)
    failed = sum(1 for _, ok, _ in checks if not ok)
    for name, ok, evidence in checks:
        print(f"{'PASS' if ok else 'FAIL'}: {name}: {evidence}")
    print(f"PASS={len(checks)-failed} WARN=0 FAIL={failed}")
    print(f"TOKEN_LEAKS={leaks}")
    return 1 if failed or leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
