from __future__ import annotations

import json

from scripts import generic_mutation_bypass_audit as audit


def _detected_paths() -> set[str]:
    found: set[str] = set()
    for path in audit._iter_python_files():
        rel = str(path.relative_to(audit.ROOT))
        if rel in audit.SCRIPT_EXCLUSIONS:
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if any(pattern.regex.search(line) for pattern in audit.PATTERNS):
                found.add(rel)
                break
    return found


def test_mutation_file_inventory_is_exact_and_field_complete() -> None:
    payload = json.loads(audit.CLASSIFICATION_PATH.read_text(encoding="utf-8"))
    rows = payload["classifications"]
    indexed = {row["path"]: row for row in rows}
    assert len(rows) == len(indexed) == payload["reviewed_count"] == 158
    assert set(indexed) == _detected_paths()
    for path, row in indexed.items():
        assert audit.REQUIRED_CLASSIFICATION_FIELDS == set(row)
        assert all(str(row[field]).strip() for field in audit.REQUIRED_CLASSIFICATION_FIELDS), path


def test_v2f_closure_has_no_pending_migration_disposition() -> None:
    payload = json.loads(audit.CLASSIFICATION_PATH.read_text(encoding="utf-8"))
    pending = {row["path"] for row in payload["classifications"] if row["disposition"] == "supported_pending_migration"}
    assert pending == set()
    denied = {row["path"] for row in payload["classifications"] if "unimplemented_denied" in row["disposition"]}
    assert "agent/packs/remote_fetch.py" in denied
