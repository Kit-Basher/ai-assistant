from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_promotion_tool_warns_restart_does_not_load_checkout_edits() -> None:
    source = (REPO_ROOT / "scripts" / "promote_local_stable.sh").read_text(encoding="utf-8")

    assert "restarting personal-agent-api.service does not load repo checkout edits" in source
    assert "scripts/promote_local_stable.sh" in source


def test_stable_vs_dev_doc_calls_out_api_telegram_split() -> None:
    doc = (REPO_ROOT / "docs" / "operator" / "USING_STABLE_VS_DEV.md").read_text(encoding="utf-8")

    assert "API service uses stable runtime" in doc
    assert "Telegram service may still use the checkout venv" in doc
    assert "promote_local_stable.sh" in doc
