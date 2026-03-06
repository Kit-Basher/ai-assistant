from __future__ import annotations

import argparse
import os
import re
import sys

from agent.logging_bootstrap import configure_logging_if_needed
from agent.secret_store import SecretStore

_TELEGRAM_BOT_TOKEN_SECRET_KEY = "telegram:bot_token"
_TELEGRAM_TOKEN_PATTERN = re.compile(r"^\d{6,}:[A-Za-z0-9_-]{20,}$")


def _redact_secret(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return "(not set)"
    if len(text) <= 10:
        return "***"
    return f"{text[:4]}...{text[-4:]}"


def _validate_secret(key: str, value: str) -> tuple[bool, str]:
    if key != _TELEGRAM_BOT_TOKEN_SECRET_KEY:
        return True, ""
    cleaned = str(value or "").strip()
    if not cleaned:
        return False, "telegram bot token cannot be empty"
    if not _TELEGRAM_TOKEN_PATTERN.fullmatch(cleaned):
        return False, "telegram bot token format looks invalid"
    return True, ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agent.secrets")
    sub = parser.add_subparsers(dest="command", required=True)

    set_parser = sub.add_parser("set", help="Set a secret value")
    set_parser.add_argument("key", help="Secret key (for example telegram:bot_token)")
    set_parser.add_argument("--value", default=None, help="Secret value (optional; stdin used when omitted)")

    get_parser = sub.add_parser("get", help="Get a secret value")
    get_parser.add_argument("key", help="Secret key")
    get_parser.add_argument("--redacted", action="store_true", help="Show redacted value")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging_if_needed()
    parser = build_parser()
    args = parser.parse_args(argv)
    store = SecretStore(path=os.getenv("AGENT_SECRET_STORE_PATH", "").strip() or None)

    if args.command == "set":
        key = str(args.key or "").strip()
        value_arg = args.value
        value = str(value_arg).strip() if value_arg is not None else str(sys.stdin.read()).strip()
        ok, reason = _validate_secret(key, value)
        if not ok:
            print(reason, file=sys.stderr, flush=True)
            return 2
        store.set_secret(key, value)
        print("saved", flush=True)
        return 0

    if args.command == "get":
        key = str(args.key or "").strip()
        value = store.get_secret(key)
        if args.redacted:
            print(_redact_secret(value), flush=True)
        else:
            print(str(value or ""), flush=True)
        return 0

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
