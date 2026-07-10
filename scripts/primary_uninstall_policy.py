#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.primary_uninstall_policy import (  # noqa: E402
    PRIMARY_UNINSTALL_DEFAULT_DAYS,
    PRIMARY_UNINSTALL_MAX_DAYS,
    build_policy_context,
    disable_primary_uninstall_marker,
    enable_primary_uninstall_marker,
    validate_primary_uninstall_marker,
)


ACK_FLAG = "--acknowledge-primary-uninstall-capability"


def _print_status(status, *, inspect: bool = False) -> None:
    payload = status.redacted_dict()
    if not inspect:
        payload.pop("fingerprint", None)
        payload["details"] = {k: v for k, v in payload.get("details", {}).items() if k != "filesystem"}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status.enabled:
        print(f"Primary preserve-data uninstall is enabled until {status.expires_at}. Purge is unsupported.")
    else:
        print(f"Primary preserve-data uninstall is disabled: {status.reason}. Purge is unsupported.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Local operator policy for primary preserve-data uninstall.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("inspect")
    enable = sub.add_parser("enable")
    enable.add_argument(ACK_FLAG, action="store_true", dest="ack")
    enable.add_argument("--expires-in-days", type=int, default=PRIMARY_UNINSTALL_DEFAULT_DAYS)
    sub.add_parser("disable")
    args = parser.parse_args()

    if args.command in {"status", "inspect"}:
        _print_status(validate_primary_uninstall_marker(build_policy_context()), inspect=args.command == "inspect")
        return 0
    if args.command == "disable":
        _print_status(disable_primary_uninstall_marker())
        return 0
    if args.command == "enable":
        if not args.ack:
            print(
                f"Refusing to enable primary uninstall capability without {ACK_FLAG}.",
                file=sys.stderr,
            )
            return 2
        if args.expires_in_days < 1 or args.expires_in_days > PRIMARY_UNINSTALL_MAX_DAYS:
            print(f"Expiry must be between 1 and {PRIMARY_UNINSTALL_MAX_DAYS} days.", file=sys.stderr)
            return 2
        ctx = build_policy_context(create_identity=True)
        if ctx.repository_path != ROOT.resolve():
            print(f"Refusing copied/untrusted checkout: {ctx.repository_path} != {ROOT.resolve()}", file=sys.stderr)
            return 2
        print(
            "Enabling primary preserve-data uninstall for this local installation only. "
            "This does not uninstall anything. Purge remains unsupported."
        )
        _print_status(enable_primary_uninstall_marker(expires_in_days=args.expires_in_days), inspect=False)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

