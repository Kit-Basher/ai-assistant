from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def _error(message: str) -> dict[str, Any]:
    return {"text": message, "data": {"source": "opinion_on_report", "error": message}}


def _facts_hash(facts: dict[str, Any]) -> str:
    payload = json.dumps(facts, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_prompt(facts: dict[str, Any], context_note: str | None) -> str:
    facts_json = json.dumps(facts, ensure_ascii=True, sort_keys=True)
    note = context_note or "report review"
    return (
        "You are generating opinions and cautions based strictly on provided facts.\n"
        "Do not add new facts or numbers.\n"
        "If information is missing, say so.\n"
        "Use hedged language (may, likely, worth watching).\n"
        "Do not give instructions that imply action without user confirmation.\n"
        "Output a short, structured opinion with sections:\n"
        "- Potential risks\n"
        "- Things to keep an eye on\n"
        "- Unknowns / missing data\n\n"
        f"Context: {note}\n"
        "Facts (JSON):\n"
        f"{facts_json}\n"
    )


def _extract_numbers(text: str) -> set[str]:
    if not text:
        return set()
    return set(re.findall(r"\\b\\d+(?:\\.\\d+)?\\b", text))


def _validate_no_new_numbers(output_text: str, facts: dict[str, Any]) -> bool:
    output_nums = _extract_numbers(output_text)
    facts_json = json.dumps(facts, ensure_ascii=True, sort_keys=True)
    fact_nums = _extract_numbers(facts_json)
    if not output_nums:
        return not any(ch.isdigit() for ch in output_text)
    if not fact_nums:
        return False
    return output_nums.issubset(fact_nums)


def _fallback_text() -> str:
    return "I can’t form a reliable opinion from the provided facts."


def _run_llm(context: dict[str, Any], prompt: str) -> str:
    router = (context or {}).get("llm_router")
    if router is not None and hasattr(router, "chat"):
        try:
            result = router.chat(
                [
                    {"role": "system", "content": "Provide concise opinion text only."},
                    {"role": "user", "content": prompt},
                ],
                purpose="presentation_rewrite",
                compute_tier="mid",
            )
            if result.get("ok"):
                return result.get("text") or ""
        except Exception:
            return ""

    broker = (context or {}).get("llm_broker")
    if broker is not None:
        try:
            from agent.llm.broker import TaskSpec

            client, _decision = broker.select(TaskSpec(task="presentation_rewrite", require_local=False))
            if hasattr(client, "generate"):
                return client.generate(prompt) or ""
        except Exception:
            return ""
    client = (context or {}).get("llm_presentation_client")
    if client and hasattr(client, "generate"):
        return client.generate(prompt) or ""
    return ""


def opinion_on_report(context: dict[str, Any], facts: dict[str, Any], context_note: str | None = None) -> dict[str, Any]:
    if not isinstance(facts, dict) or not facts:
        return _error("Facts are required and must be a non-empty JSON object.")

    prompt = _build_prompt(facts, context_note)
    text = _run_llm(context or {}, prompt)
    if not text:
        text = _fallback_text()
    if not _validate_no_new_numbers(text, facts):
        text = _fallback_text()

    return {
        "text": text,
        "data": {
            "source": "opinion_on_report",
            "facts_hash": _facts_hash(facts),
            "limits": ["opinions only", "no new facts added"],
        },
    }
