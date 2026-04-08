from __future__ import annotations

from dataclasses import dataclass
import json
import urllib.request
import subprocess
from typing import Any


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    model_id: str
    display_name: str
    context_tokens: int | None
    tags: list[str]
    created_at: str | None
    metadata: dict[str, Any]


def _tags_for_model_name(model_name: str) -> list[str]:
    lowered = str(model_name or "").strip().lower()
    values: set[str] = set()
    if any(token in lowered for token in ("code", "coder", "deepseek", "starcoder")):
        values.add("code")
    if any(token in lowered for token in ("chat", "instruct", "assistant", "llama", "qwen", "mistral", "gemma")):
        values.add("chat")
    if any(token in lowered for token in ("story", "creative", "writer")):
        values.add("story")
    if any(token in lowered for token in ("embed", "embedding")):
        values.add("embed")
    if not values:
        values.add("general")
    return sorted(values)


def list_models_ollama() -> list[ModelInfo]:
    try:
        completed = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    lines = [line.strip() for line in str(completed.stdout or "").splitlines() if str(line or "").strip()]
    output: list[ModelInfo] = []
    for line in lines:
        lowered = line.lower()
        if lowered.startswith("name ") and "size" in lowered:
            continue
        model_name = str(line.split()[0] if line.split() else "").strip()
        if not model_name or model_name.lower() == "name":
            continue
        output.append(
            ModelInfo(
                provider="ollama",
                model_id=model_name,
                display_name=model_name,
                context_tokens=None,
                tags=_tags_for_model_name(model_name),
                created_at=None,
                metadata={"source": "ollama_list"},
            )
        )
    output.sort(key=lambda row: (row.provider, row.model_id))
    return output


def list_models_openrouter(api_key: str) -> list[ModelInfo]:
    key = str(api_key or "").strip()
    if not key:
        return []
    try:
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            method="GET",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {key}",
            },
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read().decode("utf-8", errors="replace")
        parsed = json.loads(raw or "{}")
    except Exception:
        return []

    rows = parsed.get("data") if isinstance(parsed, dict) else []
    output: list[ModelInfo] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("id") or "").strip()
        if not model_id:
            continue
        output.append(
            ModelInfo(
                provider="openrouter",
                model_id=model_id,
                display_name=model_id,
                context_tokens=None,
                tags=_tags_for_model_name(model_id),
                created_at=None,
                metadata={
                    "source": "openrouter_models",
                    "pricing": row.get("pricing") if isinstance(row.get("pricing"), dict) else {},
                    "raw": row,
                },
            )
        )
    output.sort(key=lambda row: (row.provider, row.model_id))
    return output


__all__ = ["ModelInfo", "list_models_ollama", "list_models_openrouter"]
