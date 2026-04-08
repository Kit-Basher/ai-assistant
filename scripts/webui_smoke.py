#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


DEFAULT_BASE_URL = os.environ.get("AGENT_WEBUI_BASE_URL") or os.environ.get("AGENT_API_BASE_URL") or "http://127.0.0.1:8000"


def _first_line(text: str) -> str:
    stripped = str(text or "").strip()
    return stripped.splitlines()[0] if stripped else ""


def _fetch(base_url: str, path: str, timeout: float = 3.0) -> tuple[int, str]:
    request = urllib.request.Request(f"{base_url.rstrip('/')}{path}", headers={"Accept": "text/html,application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
        return int(getattr(response, "status", 200)), body


def _fetch_json(base_url: str, path: str, timeout: float = 3.0) -> tuple[int, dict[str, Any], str]:
    status, body = _fetch(base_url, path, timeout=timeout)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        payload = {}
    return status, payload if isinstance(payload, dict) else {}, body


def _failure_warnings(root_html: str, ready_payload: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    lowered = root_html.lower()
    if "id=\"root\"" not in lowered and "id='root'" not in lowered:
        warnings.append("missing root mount")
    if "personal-agent-webui" not in lowered:
        warnings.append("missing webui marker")
    if "assets/index-" not in lowered:
        warnings.append("missing built asset reference")
    if any(token in lowered for token in ("desktop", "electron")):
        warnings.append("desktop-app wording present")

    if not bool(ready_payload.get("ready", False)):
        warnings.append("ready endpoint is not ready")
    summary = str(ready_payload.get("summary") or ready_payload.get("onboarding", {}).get("summary") or ready_payload.get("recovery", {}).get("summary") or "").strip()
    if not summary:
        warnings.append("empty readiness summary")
    if any(token in summary.lower() for token in ("need more context", "i can't", "i cannot", "failed")):
        warnings.append("dead-end readiness wording")
    return warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke the live web UI served by the API server.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL of the live API server.")
    args = parser.parse_args(argv)

    try:
        root_status, root_html = _fetch(str(args.base_url), "/")
        ready_status, ready_payload, ready_body = _fetch_json(str(args.base_url), "/ready")
    except urllib.error.URLError as exc:
        print(f"FAIL: could not reach web UI at {args.base_url}: {exc}")
        return 1

    root_first_line = _first_line(root_html)
    ready_summary = str(ready_payload.get("summary") or ready_payload.get("onboarding", {}).get("summary") or ready_payload.get("recovery", {}).get("summary") or _first_line(ready_body)).strip()

    print(f"route: / status={root_status}")
    print(f"first_line: {root_first_line}")
    print(f"route: /ready status={ready_status}")
    print(f"first_line: {ready_summary}")

    warnings = _failure_warnings(root_html, ready_payload)
    print(f"dead_end_warnings: {', '.join(warnings) if warnings else 'none'}")
    return 1 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
