from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from agent.internal_writer_authority import perform_registered_internal_write


def load_seen_model_ids(path: Path) -> set[str]:
    """Compat-only helper for legacy operator recommendation surfaces."""
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return set()
    if not isinstance(parsed, dict):
        return set()
    rows = parsed.get("seen_model_ids") if isinstance(parsed.get("seen_model_ids"), list) else []
    return {str(item).strip() for item in rows if str(item).strip()}


def save_seen_model_ids(path: Path, model_ids: set[str]) -> None:
    """Compat-only helper for legacy operator recommendation surfaces."""
    normalized = sorted({str(item).strip() for item in model_ids if str(item).strip()})
    payload = {
        "schema_version": 1,
        "seen_model_ids": normalized,
    }
    def _write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

    perform_registered_internal_write(
        writer_id="modelops_seen_state",
        operation="replace_seen_ids",
        resource_type="model_seen_cache",
        target_scope="state:modelops_seen",
        arguments={"count": len(normalized), "ids_hash": __import__("hashlib").sha256("\n".join(normalized).encode()).hexdigest()},
        callback=_write,
        journal_path=path.with_name(f"{path.name}.internal-writer.sqlite3"),
    )


__all__ = [
    "load_seen_model_ids",
    "save_seen_model_ids",
]
