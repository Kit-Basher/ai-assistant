from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_smoke():
    path = REPO_ROOT / "scripts" / "live_model_switch_smoke.py"
    spec = importlib.util.spec_from_file_location("live_model_switch_smoke", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_find_model_reads_llm_status_catalog_shape() -> None:
    smoke = _load_smoke()
    status = {
        "effective_chat_model": "ollama:qwen2.5:3b-instruct",
        "models": [
            {"id": "ollama:qwen2.5:3b-instruct", "selectable_now": True, "chat_usable": True},
            {"id": smoke.TARGET_MODEL, "selectable_now": True, "chat_usable": True},
        ],
    }

    row = smoke._find_model(status, smoke.TARGET_MODEL)

    assert row is not None
    assert row["id"] == smoke.TARGET_MODEL


def test_switch_prompt_does_not_make_default_change() -> None:
    smoke = _load_smoke()

    prompt = smoke._temporary_switch_prompt(smoke.TARGET_MODEL)

    assert smoke.TARGET_MODEL in prompt
    assert "session only" in prompt
    assert "Do not change my default model" in prompt
