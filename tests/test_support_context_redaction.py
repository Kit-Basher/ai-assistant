from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_redactor():
    path = REPO_ROOT / "scripts" / "redact_support_context.py"
    spec = importlib.util.spec_from_file_location("redact_support_context", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_support_context_redacts_telegram_urls_and_secret_values() -> None:
    fixture = (REPO_ROOT / "tests" / "fixtures" / "support_context_telegram_journal.txt").read_text(encoding="utf-8")
    redacted = _load_redactor().redact_text(fixture)

    assert "https://api.telegram.org/bot<redacted>/sendMessage" in redacted
    assert "https://api.telegram.org/bot<redacted>/getUpdates" in redacted
    assert "/bot<redacted>/webhook" in redacted
    assert "Authorization: Bearer <redacted>" in redacted
    assert "TELEGRAM_BOT_TOKEN=<redacted>" in redacted
    assert '"openai_api_key": "<redacted>' in redacted
    assert "secret-store value: <redacted>" in redacted

    for raw in [
        "123456789:AASecretToken_abcdefghijklmnopqrstuvwxyz",
        "987654321:BBAnotherSecret_abcdefghijklmnopqrstuvwxyz",
        "555555555:CCPathSecret_abcdefghijklmnopqrstuvwxyz",
        "rawBearerToken.abcdef123456789",
        "222222222:DDEnvSecret_abcdefghijklmnopqrstuvwxyz",
        "sk-testabcdefghijklmnopqrstuvwxyz123456",
        "secretStoreRawValue123456",
    ]:
        assert raw not in redacted
