from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from agent.intent.assessment import IntentCandidate

if TYPE_CHECKING:
    from agent.api_server import AgentRuntime


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _extract_json_object(text: str) -> dict:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def rerank_intents_with_llm(
    *,
    candidates: list[IntentCandidate],
    user_text: str,
    runtime: AgentRuntime,
) -> list[IntentCandidate]:
    if not candidates:
        return []
    if not bool(getattr(runtime.config, "intent_llm_rerank_enabled", False)):
        return list(candidates)
    try:
        safe_mode = runtime._safe_mode_status()  # noqa: SLF001
        if bool(safe_mode.get("safe_mode")):
            return list(candidates)
        health_state = runtime._health_monitor.state  # noqa: SLF001
        providers = health_state.get("providers") if isinstance(health_state.get("providers"), dict) else {}
        if providers:
            if not any(str(row.get("status") or "").strip().lower() == "ok" for row in providers.values() if isinstance(row, dict)):
                return list(candidates)
    except Exception:
        return list(candidates)

    try:
        prompt_payload = {
            "task": "intent_rerank",
            "user_text": str(user_text or ""),
            "candidates": [
                {"intent": row.intent, "score": round(float(row.score), 4), "reason": row.reason}
                for row in sorted(candidates, key=lambda item: item.intent)
            ],
            "instructions": (
                "Return strict JSON only: "
                '{"ranked":[{"intent":"<intent>","score":0.0,"reason":"<short_reason>"}]}'
            ),
        }
        request_payload = {
            "messages": [
                {"role": "system", "content": "You are a deterministic intent re-ranker. Return JSON only."},
                {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=True, sort_keys=True)},
            ],
            "temperature": 0,
            "max_tokens": 256,
        }
        ok, body = runtime.chat(request_payload)
        if not ok or not isinstance(body, dict):
            return list(candidates)
        assistant = body.get("assistant") if isinstance(body.get("assistant"), dict) else {}
        content = str((assistant or {}).get("content") or "").strip()
        parsed = _extract_json_object(content)
        ranked_rows = parsed.get("ranked") if isinstance(parsed.get("ranked"), list) else []
        ranked_map = {
            str(row.get("intent") or "").strip(): row
            for row in ranked_rows
            if isinstance(row, dict) and str(row.get("intent") or "").strip()
        }
        output: list[IntentCandidate] = []
        for row in candidates:
            ranked = ranked_map.get(row.intent)
            if not isinstance(ranked, dict):
                output.append(row)
                continue
            try:
                target_score = float(ranked.get("score"))
            except (TypeError, ValueError):
                output.append(row)
                continue
            delta = max(-0.15, min(0.15, target_score - float(row.score)))
            adjusted = round(_clamp01(float(row.score) + delta), 4)
            reason = str(ranked.get("reason") or "").strip() or row.reason
            output.append(
                IntentCandidate(
                    intent=row.intent,
                    score=adjusted,
                    reason=reason,
                    details=dict(row.details),
                )
            )
        output.sort(key=lambda item: (-float(item.score), item.intent))
        return output
    except Exception:
        return list(candidates)


__all__ = ["rerank_intents_with_llm"]
