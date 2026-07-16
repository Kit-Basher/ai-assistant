#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib import request, error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.telegram_runtime_state import get_telegram_runtime_state, telegram_control_env
from agent.security.redaction import redact_value


API_URL = "http://127.0.0.1:8765/telegram/status"
RAW_TOKEN_MARKERS = ("https://api.telegram.org/bot", "telegram_bot_token")


def _api_status() -> dict[str, object]:
    try:
        with request.urlopen(API_URL, timeout=1.0) as response:
            parsed = json.loads(response.read().decode("utf-8", errors="replace"))
            return parsed if isinstance(parsed, dict) else {}
    except (OSError, error.URLError, TimeoutError, json.JSONDecodeError):
        return {}


def _normalized_state(state: dict[str, object]) -> dict[str, object]:
    normalized = dict(state)
    normalized.setdefault("configured", bool(state.get("configured", state.get("token_configured", False))))
    normalized.setdefault("token_present", bool(state.get("token_present", state.get("token_configured", state.get("configured", False)))))
    normalized.setdefault("token_validated", bool(state.get("token_validated", state.get("secret_store_valid", False))))
    normalized.setdefault("transport_mode", str(state.get("transport_mode") or "polling"))
    normalized.setdefault("polling_active", bool(state.get("polling_active", state.get("service_active", False))))
    normalized.setdefault("webhook_active", bool(state.get("webhook_active", False)))
    normalized.setdefault("handler_registered", bool(state.get("handler_registered", state.get("embedded_running", False))))
    normalized.setdefault(
        "duplicate_consumer_suspected",
        bool(state.get("duplicate_consumer_suspected", state.get("duplicate_pollers", False))),
    )
    normalized.setdefault("runtime_reachable", state.get("runtime_reachable"))
    normalized.setdefault("telegram_transport_healthy", bool(state.get("telegram_transport_healthy", False)))
    normalized.setdefault("telegram_health_level", str(state.get("telegram_health_level") or "UNVERIFIED"))
    normalized.setdefault("live_reply_verified", bool(state.get("live_reply_verified", False)))
    return normalized


def _token_redacted(state: dict[str, object]) -> bool:
    rendered = json.dumps(redact_value(state), ensure_ascii=True, sort_keys=True).lower()
    return not any(marker.lower() in rendered for marker in RAW_TOKEN_MARKERS)


def main() -> int:
    live = _api_status()
    state = _normalized_state(live if live else get_telegram_runtime_state(env=telegram_control_env()))
    configured = bool(state.get("configured", state.get("token_configured", False)))
    healthy = bool(state.get("telegram_transport_healthy", False))
    checks = [
        ("status payload available", bool(state), "api" if live else "local"),
        ("token redacted", _token_redacted(state), "no raw token values"),
        ("transport mode known", str(state.get("transport_mode") or "") in {"polling", "webhook", "unknown"}, str(state.get("transport_mode"))),
        ("handler status present", "handler_registered" in state, str(state.get("handler_registered"))),
        ("polling status present", "polling_active" in state, str(state.get("polling_active"))),
        ("duplicate consumer status present", "duplicate_consumer_suspected" in state or "duplicate_pollers" in state, str(state.get("duplicate_consumer_suspected", state.get("duplicate_pollers")))),
        ("runtime reachability classified", "runtime_reachable" in state or bool(live), str(state.get("runtime_reachable"))),
        ("health level classified", str(state.get("telegram_health_level") or "") in {"CONFIGURED", "POLLING", "RECEIVING", "DISPATCHING", "REPLYING", "HEALTHY", "DEGRADED", "UNVERIFIED"}, str(state.get("telegram_health_level"))),
        ("reply trace fields present", "last_reply_success_at" in state and "last_reply_attempt_at" in state, f"last_reply_success_at={state.get('last_reply_success_at')}"),
    ]
    failed = 0
    warn = 0
    print("# Telegram Transport Diagnostic")
    for name, ok, evidence in checks:
        print(f"{'PASS' if ok else 'FAIL'}: {name} - {evidence}")
        failed += 0 if ok else 1
    if configured and not healthy:
        warn += 1
        print(f"WARN: Telegram is configured but transport is not proven healthy. level={state.get('telegram_health_level')}")
        print("Suggested checks: poller active, webhook disabled, no duplicate getUpdates consumer, runtime reachable.")
    print(f"TELEGRAM_HEALTH_LEVEL={state.get('telegram_health_level')}")
    print(f"TELEGRAM_LIVE_REPLY_VERIFIED={'true' if bool(state.get('live_reply_verified')) else 'false'}")
    print(f"PASS={len(checks)-failed} WARN={warn} FAIL={failed}")
    print(f"TELEGRAM_CONFIGURED={'true' if configured else 'false'}")
    print(f"TELEGRAM_TRANSPORT_HEALTHY={'true' if healthy else 'false'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
