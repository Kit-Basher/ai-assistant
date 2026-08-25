"""Isolated Ollama schema probe for WP6.1; never touches Personal Agent state."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
import sys

from agent.assistant_turn import ASSISTANT_TURN_FLAT_SCHEMA, ASSISTANT_TURN_SCHEMA_VERSION


def _request(*, action: str, user_message: str) -> dict:
    return {
        "model": "qwen2.5:3b-instruct",
        "stream": False,
        "options": {"num_predict": 256},
        "format": ASSISTANT_TURN_FLAT_SCHEMA,
        "messages": [
            {"role": "system", "content": (
                "Return only one object matching format. Select action exactly " + action + ". "
                "All fields are required. Use calls_json='[]' unless invoke; for invoke use a JSON encoded one-element call list. "
                "Do not add fields."
            )},
            {"role": "user", "content": user_message},
        ],
    }


def _redacted(payload: dict) -> dict:
    result = dict(payload)
    result["messages"] = [
        {"role": item["role"], "content_sha256": hashlib.sha256(item["content"].encode()).hexdigest(), "content_chars": len(item["content"])}
        for item in payload["messages"]
    ]
    return result


def _call(base: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + "/api/chat",
        data=json.dumps(payload, ensure_ascii=True).encode(),
        headers={"content-type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:11434")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--actions", default="respond,clarify,invoke,propose_task,control_pending,unsupported")
    args = parser.parse_args()
    cases = {
        "respond": "Hello",
        "clarify": "I need help with a file but have not said which file.",
        "invoke": "Find the Personal Agent backup instructions in my allowed files, read them, and summarize the recovery process.",
        "propose_task": "Prepare a durable plan for a multi-step project.",
        "control_pending": "Cancel the pending change.",
        "unsupported": "Do something that is unavailable.",
    }
    report: dict[str, object] = {
        "contract": ASSISTANT_TURN_SCHEMA_VERSION,
        "ollama_endpoint": args.base.rstrip("/") + "/api/chat",
        "format_kind": "full_json_schema",
        "schema": ASSISTANT_TURN_FLAT_SCHEMA,
        "samples_per_action": args.samples,
        "cases": {},
    }
    failed = 0
    selected_actions = [item.strip() for item in args.actions.split(",") if item.strip()]
    for action in selected_actions:
        user_message = cases[action]
        payload = _request(action=action, user_message=user_message)
        successes = 0
        failures: list[dict] = []
        started = time.monotonic()
        for index in range(args.samples):
            try:
                response = _call(args.base, payload, args.timeout)
                content = str((response.get("message") or {}).get("content") or "")
                parsed = json.loads(content)
                valid = (
                    set(parsed) == set(ASSISTANT_TURN_FLAT_SCHEMA["required"])
                    and parsed.get("schema_version") == ASSISTANT_TURN_SCHEMA_VERSION
                    and parsed.get("action") == action
                    and isinstance(parsed.get("message"), str)
                    and isinstance(parsed.get("calls_json"), str)
                    and isinstance(parsed.get("pending_control"), str)
                    and isinstance(parsed.get("reason"), str)
                )
                if valid:
                    successes += 1
                else:
                    failures.append({"kind": "schema_or_action", "response": {"content": content[:600]}})
            except Exception as exc:
                failures.append({"kind": exc.__class__.__name__})
            if (index + 1) % 10 == 0:
                print(f"progress action={action} samples={index + 1} failures={len(failures)}", file=sys.stderr, flush=True)
        report["cases"][action] = {
            "request_redacted": _redacted(payload),
            "successes": successes,
            "failures": len(failures),
            "failure_examples": failures[:3],
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        failed += len(failures)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
