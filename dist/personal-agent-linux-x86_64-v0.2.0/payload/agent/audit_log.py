from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any


_SECRET_KEY_TOKENS = {
    "api_key",
    "token",
    "secret",
    "authorization",
    "password",
    "passphrase",
}

_TELEGRAM_TOKEN_RE = re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{20,}\b")
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b")


def _looks_secret_key(key: str) -> bool:
    lowered = str(key or "").strip().lower()
    return any(token in lowered for token in _SECRET_KEY_TOKENS)


def _redact_string(value: str) -> str:
    redacted = _TELEGRAM_TOKEN_RE.sub("***redacted***", value)
    redacted = _OPENAI_KEY_RE.sub("***redacted***", redacted)
    return redacted


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if _looks_secret_key(str(key)):
                output[str(key)] = "***redacted***"
            else:
                output[str(key)] = redact(item)
        return output
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _redact_string(value)
    return value


class AuditLog:
    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or self.default_path()).expanduser().resolve()

    @staticmethod
    def default_path() -> str:
        env_value = os.getenv("AGENT_AUDIT_LOG_PATH", "").strip()
        if env_value:
            return env_value
        return str(Path.home() / ".local" / "share" / "personal-agent" / "audit.jsonl")

    def append(
        self,
        *,
        actor: str,
        action: str,
        params: dict[str, Any],
        decision: str,
        reason: str,
        dry_run: bool,
        outcome: str,
        error_kind: str | None,
        duration_ms: int,
    ) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "actor": str(actor or "system"),
            "action": str(action or ""),
            "params_redacted": redact(params if isinstance(params, dict) else {}),
            "decision": str(decision or "deny"),
            "reason": str(reason or ""),
            "dry_run": bool(dry_run),
            "outcome": str(outcome or "unknown"),
            "error_kind": str(error_kind) if error_kind else None,
            "duration_ms": max(0, int(duration_ms or 0)),
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        max_rows = max(1, int(limit))
        if not self.path.is_file():
            return []

        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []

        rows: list[dict[str, Any]] = []
        for raw_line in reversed(lines):
            if not raw_line.strip():
                continue
            try:
                parsed = json.loads(raw_line)
            except Exception:
                continue
            if not isinstance(parsed, dict):
                continue
            parsed["params_redacted"] = redact(parsed.get("params_redacted") if isinstance(parsed.get("params_redacted"), dict) else {})
            rows.append(parsed)
            if len(rows) >= max_rows:
                break
        return rows
