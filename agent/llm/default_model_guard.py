from __future__ import annotations

from typing import Any

from agent.llm.capabilities import is_embedding_model_name

def _model_capabilities(model_payload: dict[str, Any]) -> set[str]:
    return {
        str(item).strip().lower()
        for item in (
            model_payload.get("capabilities")
            if isinstance(model_payload.get("capabilities"), list)
            else []
        )
        if str(item).strip()
    }


def validate_default_model(
    candidate: str | None,
    models: dict[str, Any],
    *,
    purpose: str = "chat",
) -> tuple[bool, str | None, dict[str, Any] | None]:
    model_id = str(candidate or "").strip() or None
    if model_id is None:
        return True, None, None

    model_payload = models.get(model_id) if isinstance(models, dict) else None
    if not isinstance(model_payload, dict):
        return False, None, {"ok": False, "error": "default_model not found"}

    capabilities_set = _model_capabilities(model_payload)
    capabilities = sorted(capabilities_set)
    purpose_key = str(purpose or "chat").strip().lower() or "chat"
    if purpose_key == "chat":
        if "chat" not in capabilities_set:
            return (
                False,
                None,
                {
                    "ok": False,
                    "error": "chat_model_not_chat_capable",
                    "details": {
                        "chat_model": model_id,
                        "capabilities": capabilities,
                    },
                },
            )
    elif purpose_key in {"embed", "embedding", "embeddings"}:
        embedding_capable = (
            "embedding" in capabilities_set
            or "embeddings" in capabilities_set
            or is_embedding_model_name(str(model_payload.get("model") or model_id))
        )
        if not embedding_capable:
            return (
                False,
                None,
                {
                    "ok": False,
                    "error": "embed_model_not_embedding_capable",
                    "details": {
                        "embed_model": model_id,
                        "capabilities": capabilities,
                    },
                },
            )

    return True, model_id, None
