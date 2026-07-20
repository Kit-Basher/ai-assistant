from __future__ import annotations

import argparse
import os
import re
import sys
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agent.config import runtime_api_base_url

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
    set_parser.add_argument("--api-base-url", default=runtime_api_base_url())
    set_parser.add_argument("--apply-plan", default=None, help="path to a previously previewed Universal Mutation Plan JSON")
    set_parser.add_argument("--confirmation", default=None, help="path to an already valid scoped confirmation artifact JSON")

    get_parser = sub.add_parser("get", help="Get a secret value")
    get_parser.add_argument("key", help="Secret key")
    get_parser.add_argument("--show", action="store_true", help="legacy flag; plaintext output is denied")
    get_parser.add_argument("--redacted", action="store_true", help="show presence only (default)")
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
        if key == _TELEGRAM_BOT_TOKEN_SECRET_KEY:
            operation = "telegram.secret.set"
            secret_payload = {"bot_token": value}
        elif key.startswith("provider:") and key.endswith(":api_key"):
            provider_id = key[len("provider:") : -len(":api_key")].strip().lower()
            if not provider_id:
                print("provider secret key is invalid", file=sys.stderr, flush=True)
                return 2
            operation = "provider.secret.set"
            secret_payload = {"provider_id": provider_id, "api_key": value}
        else:
            print("secret namespace is not supported by the authorized CLI", file=sys.stderr, flush=True)
            return 2
        payload: dict[str, object] = {"operation": operation, **secret_payload, "actor_id": "local_cli", "thread_id": "cli", "session_id": "cli"}
        if args.apply_plan or args.confirmation:
            if not args.apply_plan or not args.confirmation:
                print("both --apply-plan and --confirmation are required", file=sys.stderr, flush=True)
                return 2
            try:
                plan_raw = json.loads(Path(args.apply_plan).read_text(encoding="utf-8"))
                confirmation_raw = json.loads(Path(args.confirmation).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"authorization artifact could not be read: {exc.__class__.__name__}", file=sys.stderr, flush=True)
                return 2
            payload["mutation_plan"] = plan_raw.get("plan") if isinstance(plan_raw, dict) and isinstance(plan_raw.get("plan"), dict) else plan_raw
            payload["confirmation"] = confirmation_raw
        ok_http, response = _authorized_secret_request(str(args.api_base_url), payload)
        if not ok_http:
            print(str(response.get("error") or "authorized secret request failed"), file=sys.stderr, flush=True)
            return 2
        if bool(response.get("requires_confirmation")):
            # The Plan is safe to persist: plaintext was removed before it was built.
            print(json.dumps(response, indent=2, sort_keys=True, ensure_ascii=True), flush=True)
            print("No secret was changed. Confirm this Plan through a trusted front door, then rerun with --apply-plan and --confirmation.", file=sys.stderr, flush=True)
            return 3
        print("saved", flush=True)
        return 0

    if args.command == "get":
        key = str(args.key or "").strip()
        value = store.get_secret(key)
        if args.show:
            print("plaintext secret output is disabled", file=sys.stderr, flush=True)
            return 2
        print("(set)" if value else "(not set)", flush=True)
        return 0

    parser.print_help(sys.stderr)
    return 2


def _authorized_secret_request(api_base_url: str, payload: dict[str, object]) -> tuple[bool, dict[str, object]]:
    base = str(api_base_url or "").strip().rstrip("/")
    request = Request(
        f"{base}/authorized/provider-model/preview",
        data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=15.0) as response:  # noqa: S310 - fixed loopback default; explicit operator override.
            parsed = json.loads(response.read().decode("utf-8"))
            if not isinstance(parsed, dict):
                return False, {"ok": False, "error": "invalid_response"}
            return bool(parsed.get("ok")), parsed
    except HTTPError as exc:
        try:
            parsed = json.loads(exc.read().decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            parsed = {"ok": False, "error": f"http_{exc.code}"}
        return False, parsed if isinstance(parsed, dict) else {"ok": False, "error": f"http_{exc.code}"}
    except (OSError, URLError, json.JSONDecodeError) as exc:
        return False, {"ok": False, "error": f"authorized_api_unavailable:{exc.__class__.__name__}"}


if __name__ == "__main__":
    raise SystemExit(main())
