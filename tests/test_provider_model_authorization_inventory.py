from __future__ import annotations

import json
from pathlib import Path

from agent.provider_model_authorization import SPECS
from scripts.provider_model_authorization_audit import ALIASES, OUTPUT, build


def test_domain_inventory_is_byte_exact_and_all_operations_are_bound() -> None:
    payload = build()
    assert payload["validation"] == {"missing_aliases": [], "stale_aliases": []}
    assert set(SPECS) == set(ALIASES)
    assert OUTPUT.read_text(encoding="utf-8") == json.dumps(payload, indent=2, sort_keys=True) + "\n"


def test_v2e_mixed_writer_leaves_have_explicit_bounded_dispositions() -> None:
    payload = build()
    rows = payload["mixed_writers"]
    blocked = [row for row in rows if row["blocker"]]
    assert blocked == []
    by_writer = {row["writer"]: row for row in rows}
    assert {
        "llm_model_discovery_policy",
        "llm_notifications",
        "model_watch",
        "model_watch_hf",
    } <= set(by_writer)
    assert all(
        by_writer[writer]["disposition"] == "resolved_v2e"
        for writer in {
            "llm_model_discovery_policy",
            "llm_notifications",
            "model_watch",
            "model_watch_hf",
        }
    )
    assert by_writer["model_scout"]["disposition"] == "bounded_by_scheduled_parent"
