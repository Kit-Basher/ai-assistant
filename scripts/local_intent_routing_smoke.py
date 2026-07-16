#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.setup_chat_flow import classify_runtime_chat_route


CASES = [
    ("look at my PC", "operational_status", "operational_observe"),
    ("what is using my memory", "operational_status", "operational_observe"),
    ("can you take a look at my pc and tell me what is using the most memory?", "operational_status", "operational_observe"),
    ("check this machine", "operational_status", "operational_observe"),
    ("what program is using CPU", "operational_status", "operational_observe"),
    ("how do I check memory on Windows", "generic_chat", None),
    ("search for Linux system monitors", "action_tool", "safe_web_search"),
    ("look up Chrome RAM usage", "action_tool", "safe_web_search"),
]


def main() -> int:
    passed = 0
    failed = 0
    local_errors = 0
    web_errors = 0
    print("# Local Intent Routing Smoke")
    for text, expected_route, expected_kind in CASES:
        result = classify_runtime_chat_route(text)
        route = str(result.get("route") or "")
        kind = str(result.get("kind") or "")
        ok = route == expected_route and (expected_kind is None or kind == expected_kind)
        print(f"{'PASS' if ok else 'FAIL'}: {text!r} route={route} kind={kind}")
        if ok:
            passed += 1
        else:
            failed += 1
            if expected_route == "operational_status":
                local_errors += 1
            if expected_kind == "safe_web_search":
                web_errors += 1
    print(f"PASS={passed} WARN=0 FAIL={failed}")
    print(f"LOCAL_ROUTE_ERRORS={local_errors}")
    print(f"WEB_ROUTE_ERRORS={web_errors}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
