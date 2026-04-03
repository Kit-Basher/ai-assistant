#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def _fetch_ready(url: str, timeout_seconds: float) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=max(0.1, float(timeout_seconds))) as response:
        raw = response.read().decode("utf-8", errors="replace")
    parsed = json.loads(raw or "{}")
    if not isinstance(parsed, dict):
        return {}
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for Personal Agent /ready to report ready=true.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--interval", type=float, default=0.2)
    args = parser.parse_args()

    timeout_seconds = max(0.5, float(args.timeout))
    interval_seconds = max(0.05, float(args.interval))
    ready_url = f"http://{args.host}:{int(args.port)}/ready"

    spinner = "|/-\\"
    spin_index = 0
    started = time.monotonic()
    last_payload: dict[str, Any] | None = None
    last_error: str | None = None

    while True:
        elapsed = time.monotonic() - started
        if elapsed > timeout_seconds:
            break
        try:
            payload = _fetch_ready(ready_url, timeout_seconds=1.0)
            last_payload = payload
            last_error = None
            ready = bool(payload.get("ready", False))
            telegram = payload.get("telegram") if isinstance(payload.get("telegram"), dict) else {}
            tg_state = str(telegram.get("state") or "unknown")
            if ready:
                print(f"\rReady ✅ (telegram: {tg_state})".ljust(80))
                return 0
            frame = spinner[spin_index % len(spinner)]
            spin_index += 1
            print(
                f"\rWaiting for agent... ({elapsed:.1f}s) telegram={tg_state} {frame}".ljust(80),
                end="",
                flush=True,
            )
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            last_error = f"{exc.__class__.__name__}: {exc}"
            frame = spinner[spin_index % len(spinner)]
            spin_index += 1
            print(
                f"\rWaiting for agent... ({elapsed:.1f}s) connection={last_error} {frame}".ljust(80),
                end="",
                flush=True,
            )
        time.sleep(interval_seconds)

    print()
    print(f"Timed out waiting for ready endpoint after {timeout_seconds:.1f}s.")
    if isinstance(last_payload, dict):
        print("Last /ready payload:")
        print(json.dumps(last_payload, ensure_ascii=True, sort_keys=True))
    elif last_error:
        print(f"Last connection error: {last_error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
