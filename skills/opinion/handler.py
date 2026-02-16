from __future__ import annotations

from datetime import datetime
import os
import re
from typing import Any


ADVICE_PATTERNS = (
    "should i",
    "what should i do",
    "recommend",
    "recommendation",
    "fix",
    "optimize",
    "how do i",
    "how to",
    "should we",
    "best way",
    "please advise",
    "suggest",
)

OPINION_LABELS = {
    "stable",
    "consistent",
    "variable",
    "elevated",
    "lower than usual",
    "higher than usual",
    "within your normal range",
    "outside your recent baseline",
    "worth monitoring",
    "no clear signal",
}

FORBIDDEN_WORDS = (
    "good",
    "bad",
    "healthy",
    "unhealthy",
    "optimal",
    "suboptimal",
    "problem",
    "issue",
    "fix",
    "resolve",
    "improve",
    "recommend",
    "suggest",
    "should",
    "concern",
    "concerns",
)
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(re.escape(word) for word in FORBIDDEN_WORDS) + r")\b")


def _blocked(message: str) -> dict[str, Any]:
    return {"status": "blocked", "message": message, "text": message}


def _bytes_to_human(num_bytes: int) -> str:
    if num_bytes < 0:
        return "0B"
    units = ["B", "K", "M", "G", "T", "P", "E"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}B"
            formatted = f"{value:.1f}".rstrip("0").rstrip(".")
            return f"{formatted}{unit}"
        value /= 1024
    return f"{int(value)}B"


def _format_delta_bytes(delta: int) -> str:
    sign = "+" if delta >= 0 else "-"
    return f"{sign}{_bytes_to_human(abs(delta))}"


def _min_avg_max(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    total = sum(values)
    return min(values), total / len(values), max(values)


def _question_has_advice(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in ADVICE_PATTERNS)


def ask_opinion(context: dict[str, Any], question: str, timeframe: dict[str, Any], trigger: str) -> dict[str, Any]:
    db = context.get("db") if context else None
    tz_name = (context or {}).get("timezone") or "UTC"
    if not db:
        return _blocked("Database not available.")

    if _question_has_advice(question):
        refusal = (
            "I can provide bounded opinions about historical data, but not advice or actions. "
            "Please ask for observations or opinions only."
        )
        return {"status": "refused", "text": refusal, "message": refusal}

    start_date = timeframe.get("start_date")
    end_date = timeframe.get("end_date")
    start_ts = timeframe.get("start_ts")
    end_ts = timeframe.get("end_ts")
    label = timeframe.get("label") or ""
    if not start_date or not end_date:
        text = "No snapshots found yet."
        return {"status": "ok", "text": text, "message": text}

    lines: list[str] = []
    lines.append(f"Question Restated: {question.strip()}")
    timeframe_line = f"Timeframe: {label} ({start_date} to {end_date}, {tz_name})"
    lines.append(timeframe_line)

    facts: list[str] = []
    domains = _infer_domains(question)
    if not domains:
        domains = ["storage", "resources", "network"]

    if "storage" in domains:
        facts.extend(_storage_facts(db, start_date, end_date, start_ts, end_ts))
    if "resources" in domains:
        facts.extend(_resource_facts(db, start_date, end_date, start_ts, end_ts))
    if "network" in domains:
        facts.extend(_network_facts(db, start_date, end_date, start_ts, end_ts))

    lines.append("Factual Summary:")
    if facts:
        for line in facts:
            lines.append(f"- {line}")
    else:
        lines.append("- insufficient data")

    lines.append("Opinionated Assessment:")
    labels_used: list[str] = []
    domain_lines: list[str] = []
    for domain in ["storage", "resources", "network", "weekly_reflection"]:
        if domain not in domains:
            continue
        label_out, basis = _opinion_for_domain(db, domain, start_date, end_date, start_ts, end_ts)
        label_out = _ensure_label(label_out)
        labels_used.append(label_out)
        domain_line = f"- {domain}: {label_out} (basis: {basis})"
        domain_lines.append(domain_line)
        lines.append(domain_line)

    lines.append("Confidence & Limits:")
    lines.append("- Opinions are bounded labels derived from your historical data only.")
    lines.append("- No advice, instructions, or actions are being provided.")
    lines.append("- If data is sparse, the label defaults to 'no clear signal'.")

    deterministic_text = "\n".join(lines)
    rewrite = _maybe_rewrite_presentation(
        context,
        deterministic_text,
        timeframe_line=timeframe_line,
        domain_lines=domain_lines,
    )
    final_text = rewrite["text"]

    _log_llm_decision(context, rewrite, validation_passed=rewrite["validation_passed"])

    if _contains_forbidden([final_text]):
        return {"status": "failed", "text": "Opinion vocabulary violation.", "message": "Opinion vocabulary violation."}

    audit_details = {
        "command": "/ask_opinion",
        "trigger": trigger,
        "domains": domains,
        "labels_used": labels_used,
        "llm_presentation_attempted": rewrite["attempted"],
        "llm_presentation_used": rewrite["used"],
        "llm_validation_passed": rewrite["validation_passed"],
        "llm_provider": rewrite["provider"],
        "llm_failure_reason": rewrite.get("failure_reason"),
        "llm_selector_mode": rewrite.get("selector_mode"),
        "llm_decision": _summarize_decision(rewrite.get("decision")),
        "llm_remote_used": _decision_remote_used(rewrite.get("decision")),
    }
    try:
        db.audit_log_create(
            user_id=str(timeframe.get("user_id") or "unknown"),
            action_type="ask_opinion",
            action_id="ask_opinion",
            status="executed",
            details=audit_details,
        )
    except Exception:
        return {
            "status": "failed",
            "message": "Audit logging failed. Operation aborted.",
            "text": "Audit logging failed. Operation aborted.",
        }

    return {"status": "ok", "text": final_text, "payload": {"labels_used": labels_used}}


def _infer_domains(question: str) -> list[str]:
    lowered = question.lower()
    domains: list[str] = []
    if any(word in lowered for word in ("disk", "storage", "ssd", "space", "directory", "folder")):
        domains.append("storage")
    if any(word in lowered for word in ("cpu", "memory", "ram", "load", "swap", "process")):
        domains.append("resources")
    if any(word in lowered for word in ("network", "dns", "gateway", "interface", "rx", "tx")):
        domains.append("network")
    if any(word in lowered for word in ("weekly", "reflection", "rollup")):
        domains.append("weekly_reflection")
    return domains


def _filter_rows_by_ts(rows: list[dict[str, Any]], start_ts: str | None, end_ts: str | None) -> list[dict[str, Any]]:
    if not start_ts or not end_ts:
        return rows
    try:
        start_dt = datetime.fromisoformat(start_ts)
        end_dt = datetime.fromisoformat(end_ts)
    except ValueError:
        return rows
    filtered: list[dict[str, Any]] = []
    for row in rows:
        ts = row.get("taken_at")
        if not ts:
            filtered.append(row)
            continue
        try:
            row_dt = datetime.fromisoformat(str(ts))
        except ValueError:
            filtered.append(row)
            continue
        if start_dt <= row_dt <= end_dt:
            filtered.append(row)
    return filtered


def _storage_facts(db: Any, start: str, end: str, start_ts: str | None, end_ts: str | None) -> list[str]:
    lines: list[str] = []
    mountpoints = ["/", "/data", "/data2"]
    for mount in mountpoints:
        rows = db.list_disk_snapshots_between(mount, start, end)
        rows = _filter_rows_by_ts(rows, start_ts, end_ts)
        if len(rows) < 2:
            continue
        delta = int(rows[-1]["used_bytes"]) - int(rows[0]["used_bytes"])
        lines.append(f"{mount} used change: {delta} ({_format_delta_bytes(delta)})")
    return lines


def _resource_facts(db: Any, start: str, end: str, start_ts: str | None, end_ts: str | None) -> list[str]:
    rows = db.list_resource_snapshots_between(start, end)
    rows = _filter_rows_by_ts(rows, start_ts, end_ts)
    if len(rows) < 2:
        return []
    load_1m = [float(r["load_1m"]) for r in rows]
    l1_min, l1_avg, l1_max = _min_avg_max(load_1m)
    return [f"load_1m min/avg/max: {l1_min:.2f}/{l1_avg:.2f}/{l1_max:.2f}"]


def _network_facts(db: Any, start: str, end: str, start_ts: str | None, end_ts: str | None) -> list[str]:
    rows = db.list_network_snapshots_between(start, end)
    rows = _filter_rows_by_ts(rows, start_ts, end_ts)
    if len(rows) < 2:
        return []
    changes = []
    prev_gateway = None
    for row in rows:
        day = row["snapshot_local_date"]
        gateway = row["default_gateway"]
        if prev_gateway is not None and gateway != prev_gateway:
            changes.append(f"{day}: {prev_gateway} -> {gateway}")
        prev_gateway = gateway
    if not changes:
        return ["default gateway changes: none"]
    return [f"default gateway changes: {', '.join(changes)}"]


def _opinion_for_domain(
    db: Any, domain: str, start: str, end: str, start_ts: str | None, end_ts: str | None
) -> tuple[str, str]:
    if domain == "storage":
        rows = db.list_disk_snapshots_between("/", start, end)
        rows = _filter_rows_by_ts(rows, start_ts, end_ts)
        if len(rows) < 2:
            return "no clear signal", "insufficient data (n<2)"
        delta = int(rows[-1]["used_bytes"]) - int(rows[0]["used_bytes"])
        avg = sum(int(r["used_bytes"]) for r in rows) / len(rows)
        label = _label_from_delta(delta, avg)
        return label, f"/ used change {delta} ({_format_delta_bytes(delta)}), avg {int(avg)}"

    if domain == "resources":
        rows = db.list_resource_snapshots_between(start, end)
        rows = _filter_rows_by_ts(rows, start_ts, end_ts)
        if len(rows) < 2:
            return "no clear signal", "insufficient data (n<2)"
        load_1m = [float(r["load_1m"]) for r in rows]
        l1_min, l1_avg, l1_max = _min_avg_max(load_1m)
        range_val = l1_max - l1_min
        label = _label_from_range(range_val, l1_avg)
        return label, f"load_1m range {range_val:.2f}, avg {l1_avg:.2f}"

    if domain == "network":
        rows = db.list_network_snapshots_between(start, end)
        rows = _filter_rows_by_ts(rows, start_ts, end_ts)
        if len(rows) < 2:
            return "no clear signal", "insufficient data (n<2)"
        changes = 0
        prev_gateway = None
        for row in rows:
            gateway = row["default_gateway"]
            if prev_gateway is not None and gateway != prev_gateway:
                changes += 1
            prev_gateway = gateway
        if changes == 0:
            return "stable", "default gateway changes 0"
        return "variable", f"default gateway changes {changes}"

    return "no clear signal", "insufficient data"


def _label_from_delta(delta: int, avg: float) -> str:
    if avg <= 0:
        return "no clear signal"
    ratio = abs(delta) / avg
    if ratio <= 0.05:
        return "stable"
    if ratio <= 0.20:
        return "within your normal range"
    if delta > 0:
        return "higher than usual"
    return "lower than usual"


def _label_from_range(range_val: float, avg: float) -> str:
    if avg <= 0:
        return "no clear signal"
    ratio = range_val / avg
    if ratio <= 0.20:
        return "consistent"
    if ratio <= 0.50:
        return "variable"
    return "outside your recent baseline"


def _ensure_label(label: str) -> str:
    if label not in OPINION_LABELS:
        return "no clear signal"
    return label


def _contains_forbidden(lines: list[str]) -> bool:
    text = "\n".join(lines).lower()
    return _FORBIDDEN_RE.search(text) is not None


def _summarize_decision(decision: dict[str, Any] | None) -> dict[str, Any] | None:
    if not decision or not isinstance(decision, dict):
        return None
    winner = decision.get("winner") or {}
    return {
        "winner_id": decision.get("winner_id"),
        "provider": winner.get("provider"),
        "model": winner.get("model"),
        "score": winner.get("score"),
        "candidates_count": decision.get("candidates_count"),
    }


def _decision_remote_used(decision: dict[str, Any] | None) -> bool:
    if not decision or not isinstance(decision, dict):
        return False
    winner = decision.get("winner") or {}
    return bool(winner.get("remote"))


def _log_llm_decision(
    context: dict[str, Any] | None,
    rewrite: dict[str, Any],
    *,
    validation_passed: bool,
) -> None:
    log_path = (context or {}).get("log_path")
    if not log_path:
        return
    try:
        from agent.logging_utils import log_event

        decision = rewrite.get("decision") if isinstance(rewrite, dict) else None
        winner = (decision or {}).get("winner") or {}
        candidates = []
        for item in (decision or {}).get("candidates", []) or []:
            candidates.append(
                {
                    "id": item.get("id"),
                    "score": item.get("score"),
                    "excluded_reason": None,
                }
            )
        for rejected in (decision or {}).get("rejected", []) or []:
            candidates.append(
                {
                    "id": rejected.get("id"),
                    "score": None,
                    "excluded_reason": rejected.get("reason"),
                }
            )
        decision_payload = {
            "command": "/ask_opinion",
            "task": "presentation_rewrite",
            "selector_mode": rewrite.get("selector_mode"),
            "winner": {
                "id": winner.get("id"),
                "provider": winner.get("provider"),
                "model": winner.get("model"),
                "remote": winner.get("remote"),
                "score": winner.get("score"),
            },
            "candidates": candidates,
            "validation": {
                "passed": bool(validation_passed),
                "reason": rewrite.get("failure_reason") or "ok",
            },
            "fallback_used": not bool(rewrite.get("used")),
        }
        decisions_path = os.path.join(os.path.dirname(log_path), "llm_decisions.jsonl")
        log_event(decisions_path, "llm_decision", decision_payload)
    except Exception:
        return


def _maybe_rewrite_presentation(
    context: dict[str, Any] | None,
    deterministic_text: str,
    *,
    timeframe_line: str,
    domain_lines: list[str],
) -> dict[str, Any]:
    enabled = os.getenv("ENABLE_LLM_PRESENTATION", "").strip().lower() in {"1", "true", "yes", "y", "on"}
    selector = os.getenv("LLM_SELECTOR", "single").strip().lower() or "single"
    provider = os.getenv("LLM_PROVIDER", "none").strip().lower() or "none"
    if not enabled or (selector == "single" and provider == "none"):
        return {
            "text": deterministic_text,
            "attempted": False,
            "used": False,
            "validation_passed": False,
            "provider": provider,
            "selector_mode": selector,
            "decision": None,
        }

    must_keep_lines = [timeframe_line, *domain_lines]
    prompt = _presentation_prompt(deterministic_text, must_keep_lines)
    router = (context or {}).get("llm_router")
    client = (context or {}).get("llm_presentation_client")
    broker = (context or {}).get("llm_broker")

    llm_text = None
    decision = None
    failure_reason = None
    if router and hasattr(router, "chat"):
        provider_override = provider if selector == "single" and provider != "none" else None
        route_result = router.chat(
            [
                {"role": "system", "content": "Rewrite for presentation only. Preserve facts exactly."},
                {"role": "user", "content": prompt},
            ],
            purpose="presentation_rewrite",
            provider_override=provider_override,
            compute_tier="mid",
        )
        attempts = route_result.get("attempts") or []
        decision = {
            "winner_id": route_result.get("model"),
            "winner": {
                "provider": route_result.get("provider"),
                "model": route_result.get("model"),
                "score": None,
                "remote": route_result.get("provider") not in {"ollama", None},
                "id": route_result.get("model"),
            },
            "candidates": [
                {
                    "id": item.get("model"),
                    "provider": item.get("provider"),
                    "model": item.get("model"),
                    "score": None,
                }
                for item in attempts
                if not item.get("reason")
            ],
            "rejected": [
                {
                    "id": item.get("model"),
                    "reason": item.get("reason"),
                }
                for item in attempts
                if item.get("reason")
            ],
            "candidates_count": len(attempts) + (1 if route_result.get("ok") else 0),
        }
        if route_result.get("ok"):
            llm_text = route_result.get("text")
            provider = route_result.get("provider") or provider
        else:
            failure_reason = route_result.get("error_class") or "router_error"
    elif selector == "broker":
        if broker is None:
            return {
                "text": deterministic_text,
                "attempted": True,
                "used": False,
                "validation_passed": False,
                "provider": "none",
                "selector_mode": selector,
                "decision": None,
                "failure_reason": "broker_unavailable",
            }
        try:
            from agent.llm.broker import TaskSpec

            task_spec = TaskSpec(task="presentation_rewrite", require_local=False)
            client, decision = broker.select(task_spec)
            provider = (decision or {}).get("winner", {}).get("provider", "none")
            if hasattr(client, "generate"):
                llm_text = client.generate(prompt)
        except Exception:
            return {
                "text": deterministic_text,
                "attempted": True,
                "used": False,
                "validation_passed": False,
                "provider": "none",
                "selector_mode": selector,
                "decision": None,
                "failure_reason": "broker_error",
            }
    elif client:
        provider = getattr(client, "provider", provider)
        if hasattr(client, "rewrite"):
            result = client.rewrite(deterministic_text, must_keep_lines)
            if isinstance(result, dict):
                llm_text = result.get("text")
                provider = result.get("provider") or provider
            else:
                llm_text = result
        elif hasattr(client, "generate"):
            llm_text = client.generate(prompt)
    else:
        llm_text = None
        failure_reason = "router_unavailable"

    if not llm_text:
        return {
            "text": deterministic_text,
            "attempted": True,
            "used": False,
            "validation_passed": False,
            "provider": provider,
            "selector_mode": selector,
            "decision": decision,
            "failure_reason": failure_reason or "no_llm_output",
        }

    ok, reason = _validate_llm_output(
        llm_text,
        deterministic_text,
        must_keep_lines,
        domain_lines,
    )
    if not ok:
        return {
            "text": deterministic_text,
            "attempted": True,
            "used": False,
            "validation_passed": False,
            "provider": provider,
            "selector_mode": selector,
            "decision": decision,
            "failure_reason": reason,
        }

    return {
        "text": llm_text,
        "attempted": True,
        "used": True,
        "validation_passed": True,
        "provider": provider,
        "selector_mode": selector,
        "decision": decision,
    }


def _presentation_prompt(deterministic_text: str, must_keep_lines: list[str]) -> str:
    keep_block = "\n".join(must_keep_lines)
    return (
        "Rewrite for presentation only. You may rephrase ONLY non-kept sentences.\n"
        "Do not add/remove facts. Do not add advice. Do not use forbidden words.\n"
        "Output plain text only.\n\n"
        "MUST KEEP EXACT LINES:\n"
        f"{keep_block}\n\n"
        "DETERMINISTIC MESSAGE:\n"
        f"{deterministic_text}\n"
    )


def _validate_llm_output(
    llm_text: str,
    deterministic_text: str,
    must_keep_lines: list[str],
    domain_lines: list[str],
) -> tuple[bool, str]:
    output_lines = [line.rstrip() for line in llm_text.splitlines()]
    deterministic_lines = {line.rstrip() for line in deterministic_text.splitlines()}

    required_headers = [
        "Question Restated:",
        "Timeframe:",
        "Factual Summary:",
        "Opinionated Assessment:",
        "Confidence & Limits:",
    ]
    for header in required_headers:
        if not any(line.startswith(header) for line in output_lines):
            return False, "missing_header"

    for line in must_keep_lines:
        if line not in output_lines:
            return False, "missing_must_keep"

    opinion_idx = _section_index(output_lines, "Opinionated Assessment:")
    if opinion_idx is None:
        return False, "missing_opinion_section"
    next_header_idx = _next_header_index(output_lines, opinion_idx + 1)
    opinion_lines = output_lines[opinion_idx + 1 : next_header_idx]
    for line in opinion_lines:
        if line.strip().startswith("- "):
            if line not in domain_lines:
                return False, "extra_domain_line"

    domain_positions = []
    for line in domain_lines:
        try:
            domain_positions.append(output_lines.index(line))
        except ValueError:
            return False, "missing_domain_line"
    if domain_positions != sorted(domain_positions):
        return False, "domain_order_changed"

    if _contains_forbidden([llm_text]):
        return False, "forbidden_word"

    for line in output_lines:
        if any(ch.isdigit() for ch in line):
            if line not in must_keep_lines and line not in deterministic_lines:
                return False, "extra_digits"

    return True, "ok"


def _section_index(lines: list[str], header: str) -> int | None:
    for idx, line in enumerate(lines):
        if line.startswith(header):
            return idx
    return None


def _next_header_index(lines: list[str], start_idx: int) -> int:
    for idx in range(start_idx, len(lines)):
        if any(lines[idx].startswith(h) for h in ("Question Restated:", "Timeframe:", "Factual Summary:", "Opinionated Assessment:", "Confidence & Limits:")):
            return idx
    return len(lines)
