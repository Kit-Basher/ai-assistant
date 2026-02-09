from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
import os
import json
import uuid
import time
from datetime import datetime, timezone, timedelta

from agent.intent_router import route_message
from agent.disk_diff import diff_disk_reports, time_since
from agent.disk_anomalies import detect_anomalies
from agent.runner import Runner
from agent.llm.router import LLMNarrationRouter
from agent.disk_grow import resolve_allowed_path, build_growth_report, _run_du
from agent.action_gate import handle_action_text, propose_action
from agent.commands import parse_command, split_pipe_args
from agent.confirmations import ConfirmationStore, PendingAction
from agent.logging_utils import log_audit_event, log_event
from agent.knowledge_cache import KnowledgeQueryCache, facts_hash
from agent.conversation_memory import record_event
from agent import memory_ingest
from agent.redaction import redact_for_llm
from agent.compare_mode import compare_now_to_what_if
from agent.report_followups import resource_followup
from agent.policy import evaluate_policy
from agent import opinion_gate
from agent.skills_loader import SkillLoader
from agent.ask_timeframe import parse_timeframe
from agent.open_loops import build_open_loops_report
from agent.conversation_layer import SYSTEM_PROMPT as CONVERSATION_SYSTEM_PROMPT
from agent.conversation_layer import build_conversation_context
from agent.llm.broker import TaskSpec
from agent.llm.providers.ollama_provider import ping_ollama_with_reason
from memory.db import MemoryDB
from agent.report_presenter import (
    present_report,
    ui_mode as _ui_mode,
    wants_raw_details,
)
from agent.proc_cpu import sample_top_process_cpu_pct
from agent.changed_report import build_changed_report_from_system_facts
from agent.system_facts import collect_system_facts
from agent.trend_report import build_trend_report_from_system_facts
from agent.opinion_report import build_system_opinion
from agent.observer_flags import (
    flag_disk_full,
    flag_high_load,
    flag_high_load_sustained,
    flag_ram_hog,
    build_helpful_commands,
    commands_need_warning,
)
from agent.journal import make_journal_line
from agent.provider_status import get_provider_status, format_provider_status_block
from agent.doctor import run_doctor
from agent.permissions import role_for_user


AUDIT_HARD_FAIL_MSG = "Audit logging failed. Operation aborted."
_ASK_ADVICE_PHRASES = (
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

_TOP_CPU_QUESTION = re.compile(
    r"\bwhat('?s| is)\s+using\s+my\s+cpu\b|\btop\s+cpu\b|\bhigh\s+cpu\b|\bcpu\s+hog\b|\b(stuck|hung|frozen|not responding)\b",
    re.IGNORECASE,
)


@dataclass


class OrchestratorResponse:
    text: str
    data: dict[str, Any] | None = None


class Orchestrator:
    def __init__(
        self,
        db: MemoryDB,
        skills_path: str,
        log_path: str,
        timezone: str,
        llm_client: Any,
        enable_writes: bool = False,
        llm_broker: Any | None = None,
        llm_broker_error: str | None = None,
    ) -> None:
        self.db = db
        self.skills = SkillLoader(skills_path).load_all()
        self.log_path = log_path
        self.timezone = timezone
        self.llm_client = llm_client
        self.confirmations = ConfirmationStore()
        self.enable_writes = enable_writes
        self._runner: Runner | None = None
        self._llm_broker = llm_broker
        self._llm_broker_error = llm_broker_error
        self._knowledge_cache = KnowledgeQueryCache()
        self._pending_compare: dict[str, dict[str, str]] = {}
        self._intent_assist_last_call_s: dict[str, float] = {}
        # Ephemeral per-user cache for "show details" followups.
        # We keep a small per-user map of cache_key -> raw_text so followups can select older reports.
        self._details_cache_by_user: dict[str, dict[str, str]] = {}
        self._details_cache_order_by_user: dict[str, list[str]] = {}
        self._last_details_cache_key_by_user: dict[str, str] = {}
        self._selected_details_cache_key_by_user: dict[str, str] = {}
        # Flag state to avoid flappy/repetitive warnings in health checks.
        self._flag_last_emitted_s: dict[str, dict[str, float]] = {}
        self._disk_full_active: dict[str, dict[str, bool]] = {}

    def _facts_owner_user_id(self) -> str:
        """
        Shared machine facts timeline owner.

        Preference order:
        - OWNER_USER_ID
        - DIGEST_OWNER_USER_ID (backwards-compatible)

        Canonicalize telegram ids to "tg:<id>" when no explicit prefix exists.
        """
        owner = (os.getenv("OWNER_USER_ID") or os.getenv("DIGEST_OWNER_USER_ID") or "").strip()
        if owner and ":" not in owner:
            owner = f"tg:{owner}"
        return owner

    def _facts_user_id_for_request(self, request_user_id: str) -> str:
        """
        Model 1 multi-user: reports are per requesting user, but system facts are shared.

        If the requester has their own facts timeline, use it; otherwise fall back to the shared owner timeline.
        """
        req = (request_user_id or "").strip()
        if not req:
            return ""
        try:
            if self.db.get_latest_system_facts_snapshot(req):
                return req
        except Exception:
            pass
        owner = self._facts_owner_user_id()
        if owner:
            try:
                if self.db.get_latest_system_facts_snapshot(owner):
                    return owner
            except Exception:
                pass
        return req

    def _truncate_details_for_cache(self, text: str) -> str:
        max_lines = int(os.getenv("MAX_DETAILS_LINES", "250") or 250)
        max_chars = int(os.getenv("MAX_DETAILS_CHARS", "20000") or 20000)
        lines = (text or "").splitlines()
        if max_lines > 0 and len(lines) > max_lines:
            kept = lines[:max_lines]
            omitted = len(lines) - max_lines
            kept.append(f"... ({omitted} more lines truncated)")
            text = "\n".join(kept)
        if max_chars > 0 and len(text) > max_chars:
            # Reserve room for suffix.
            suffix = "\n... (output truncated)"
            text = text[: max(0, max_chars - len(suffix))].rstrip() + suffix
        return text

    def _provider_status_block(self, user_id: str) -> str:
        try:
            st = get_provider_status(self.db, user_id)
            return format_provider_status_block(st)
        except Exception:
            return "[ProviderStatus]\nerror=not_available"

    def _append_provider_status(self, user_id: str, raw_text: str) -> str:
        text = (raw_text or "").strip()
        if not text:
            return ""
        if "[ProviderStatus]" in text:
            return text
        block = self._provider_status_block(user_id)
        return (text + "\n\n" + block).strip()

    def _cache_last_report(self, user_id: str, *, kind: str, raw_text: str) -> str:
        text = (raw_text or "").strip()
        if not text:
            return ""
        text = self._truncate_details_for_cache(text)
        cache_key = str(uuid.uuid4())
        per_user = self._details_cache_by_user.setdefault(user_id, {})
        per_user[cache_key] = text
        order = self._details_cache_order_by_user.setdefault(user_id, [])
        order.append(cache_key)
        # Trim to keep cache bounded. Prefer matching report-history size.
        try:
            keep = int(os.getenv("MAX_REPORT_HISTORY", "5") or 5)
        except Exception:
            keep = 5
        keep = max(keep, 1)
        if len(order) > keep:
            drop = order[:-keep]
            order[:] = order[-keep:]
            for k in drop:
                per_user.pop(k, None)
        self._last_details_cache_key_by_user[user_id] = cache_key
        # Default: "show details" refers to the most recently produced report unless a followup selects older.
        self._selected_details_cache_key_by_user[user_id] = cache_key
        return cache_key

    def _extract_machine_summary_json(self, details_text: str) -> str | None:
        found = []
        for line in (details_text or "").splitlines():
            if line.startswith("machine_summary_json="):
                found.append(line.split("=", 1)[1].strip())
        if len(found) != 1:
            return None
        return found[0] or None

    def _upsert_last_report_registry(
        self,
        *,
        user_id: str,
        kind: str,
        created_at: str,
        details_cache_key: str,
        machine_summary_json: str,
        facts_snapshot_id: str | None,
    ) -> None:
        try:
            self.db.upsert_last_report(
                user_id,
                kind,
                created_at,
                details_cache_key,
                machine_summary_json,
                facts_snapshot_id,
            )
        except Exception as exc:
            try:
                log_audit_event(
                    self.log_path,
                    event="last_report_upsert_failed",
                    user_id=user_id,
                    snapshot_id=str(facts_snapshot_id or ""),
                    error=str(exc) or "upsert_failed",
                    probe="last_report_registry",
                    target="db",
                    severity="warn",
                )
            except Exception:
                pass

    def _register_last_report(
        self,
        *,
        user_id: str,
        kind_hint: str,
        details_cache_key: str,
        details_text: str,
        facts_snapshot_id_hint: str | None = None,
    ) -> None:
        # Phase 6: write report history rows (TTL + ring buffer). Keep last_report_registry table for
        # backwards compatibility, but do not write to it.
        ms_json = self._extract_machine_summary_json(details_text or "")
        if not ms_json:
            return

        kind = kind_hint
        created_at = datetime.now(timezone.utc).isoformat()
        facts_snapshot_id = facts_snapshot_id_hint
        try:
            ms = json.loads(ms_json)
            if isinstance(ms, dict):
                kind = str(ms.get("kind") or kind_hint)
                taken = ms.get("taken_at")
                if isinstance(taken, list) and taken:
                    last = str(taken[-1] or "").strip()
                    if last:
                        created_at = last
                sids = ms.get("snapshots_used")
                if isinstance(sids, list) and sids:
                    last_sid = str(sids[-1] or "").strip()
                    if last_sid:
                        facts_snapshot_id = last_sid
        except Exception:
            pass

        if not details_cache_key:
            return
        try:
            ttl_s = int(os.getenv("EXPIRES_AFTER_S", "1800") or 1800)
        except Exception:
            ttl_s = 1800
        try:
            dt = datetime.fromisoformat(created_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            dt = datetime.now(timezone.utc)
            created_at = dt.isoformat()
        expires_at = (dt + timedelta(seconds=int(ttl_s))).isoformat()

        try:
            report_id = str(uuid.uuid4())
            self.db.insert_report_history(
                id=report_id,
                user_id=user_id,
                kind=kind,
                created_at=created_at,
                expires_at=expires_at,
                details_cache_key=details_cache_key,
                machine_summary_json=ms_json,
                facts_snapshot_id=facts_snapshot_id,
            )
            try:
                keep = int(os.getenv("MAX_REPORT_HISTORY", "5") or 5)
            except Exception:
                keep = 5
            self.db.delete_reports_older_than_limit(user_id, keep=max(keep, 1))
        except Exception as exc:
            try:
                log_audit_event(
                    self.log_path,
                    event="report_history_insert_failed",
                    user_id=user_id,
                    snapshot_id=str(facts_snapshot_id or ""),
                    error=str(exc) or "insert_failed",
                    probe="report_history",
                    target="db",
                    severity="warn",
                )
            except Exception:
                pass
            return

        # Phase 10B: append-only change journal (best-effort; do not affect report_history).
        try:
            try:
                ms = json.loads(ms_json)
            except Exception:
                ms = {}
            jk = str(kind).strip().lower()
            if jk == "slow_diagnosis":
                jk = "slow"
            severity = "ok"
            if isinstance(ms, dict):
                sev = str(ms.get("severity") or "").strip().lower()
                if sev in {"ok", "watch", "act_soon"}:
                    severity = sev
            line = make_journal_line(jk, ms if isinstance(ms, dict) else {})
            self.db.insert_journal_entry(
                id=str(uuid.uuid4()),
                user_id=user_id,
                kind=jk,
                severity=severity,
                created_at=created_at,
                line=line or f"{jk}: {severity}",
                machine_summary_json=ms_json,
            )
        except Exception as exc2:
            try:
                log_audit_event(
                    self.log_path,
                    event="journal_insert_failed",
                    user_id=user_id,
                    snapshot_id=str(facts_snapshot_id or ""),
                    error=str(exc2) or "insert_failed",
                    probe="change_journal",
                    target="db",
                    severity="warn",
                )
            except Exception:
                pass

    def _cooldown_flags(self, user_id: str, flags: list[Any], cooldown_s: float = 600.0) -> list[Any]:
        if not flags:
            return []
        now_s = time.monotonic()
        per_user = self._flag_last_emitted_s.setdefault(user_id, {})
        out = []
        for f in flags:
            key = getattr(f, "key", None) or "unknown"
            last = float(per_user.get(key) or 0.0)
            if last and (now_s - last) < cooldown_s:
                continue
            per_user[key] = now_s
            out.append(f)
        return out

    def _last_report_raw(self, user_id: str) -> str | None:
        per_user = self._details_cache_by_user.get(user_id) or {}
        cache_key = self._selected_details_cache_key_by_user.get(user_id) or ""
        raw = per_user.get(cache_key) if cache_key else None
        if not (isinstance(raw, str) and raw.strip()):
            cache_key2 = self._last_details_cache_key_by_user.get(user_id) or ""
            raw = per_user.get(cache_key2) if cache_key2 else None
        return raw if isinstance(raw, str) and raw.strip() else None

    def _set_selected_details_cache_key(self, user_id: str, cache_key: str) -> None:
        if not cache_key:
            return
        if cache_key in (self._details_cache_by_user.get(user_id) or {}):
            self._selected_details_cache_key_by_user[user_id] = cache_key

    def _report_kind(self, skill_name: str, function_name: str) -> str | None:
        skill = (skill_name or "").strip().lower()
        fn = (function_name or "").strip().lower()
        if skill == "resource_governor" and fn == "resource_report":
            return "resource_report"
        if skill == "hardware_report" and fn == "hardware_report":
            return "hardware_report"
        if skill == "storage_governor" and fn == "storage_report":
            return "storage_report"
        if skill == "storage_governor" and fn == "storage_live_report":
            return "storage_live_report"
        if skill == "network_governor" and fn == "network_report":
            return "network_report"
        if skill == "runtime_status" and fn == "runtime_status":
            return "runtime_status"
        return None

    def _present_readonly_report(
        self,
        *,
        user_id: str,
        kind: str,
        raw_text: str,
        result_dict: dict[str, Any] | None,
        question: str | None,
    ) -> str:
        # On-demand process CPU% sampling (observer-only). Keep it scoped to CPU "top" questions so we
        # don't slow down normal snapshots/reports.
        if kind == "resource_report" and question and _TOP_CPU_QUESTION.search(question):
            sample_taken_at = datetime.now(timezone.utc).isoformat()
            try:
                interval_ms = int(os.getenv("CPU_TOP_SAMPLE_MS", "350") or 350)
            except Exception:
                interval_ms = 350
            try:
                top_n = int(os.getenv("CPU_TOP_N", "5") or 5)
            except Exception:
                top_n = 5
            try:
                cpu_rows = sample_top_process_cpu_pct(interval_ms=interval_ms, top_n=top_n)
            except Exception:
                cpu_rows = []

            if cpu_rows:
                if not isinstance(result_dict, dict):
                    result_dict = {}
                payload = result_dict.get("payload") if isinstance(result_dict.get("payload"), dict) else {}
                payload["cpu_samples"] = cpu_rows
                result_dict["payload"] = payload

                # Append to raw details so "show details" is reproducible without re-running.
                lines = [
                    raw_text.rstrip(),
                    "",
                    "Top CPU processes (sampled):",
                    f"sample_taken_at={sample_taken_at}",
                    f"interval_ms={interval_ms}",
                    f"top_n={top_n}",
                    "cpu_pct_scale=percent_of_one_core (may exceed 100% on multi-core)",
                ]
                for row in cpu_rows[:5]:
                    try:
                        name = row.get("name") or "unknown"
                        pid = int(row.get("pid") or 0)
                        cpu_pct = float(row.get("cpu_pct") or 0.0)
                        lines.append(f"- pid={pid} {name}: {cpu_pct:.1f}%")
                    except Exception:
                        continue
                raw_text = "\n".join([ln for ln in lines if ln is not None]).strip()

        if os.getenv("ENABLE_LLM_PRESENTATION", "0").strip().lower() in {"1", "true", "yes", "y", "on"}:
            router = self.llm_client
            if router and hasattr(router, "chat"):
                safe_text = redact_for_llm(raw_text or "")
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an IT-pro support assistant. "
                            "Summarize the provided system report for a non-expert. "
                            "Output exactly 3-7 bullet points explaining what matters and what key items mean. "
                            'Then add one blank line and the single sentence: Want the full report? Say "show details". '
                            "Do not include raw logs, command output, or secrets."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Report kind: {kind}\nUser question: {question or ''}\nRedacted report:\n{safe_text}",
                    },
                ]
                try:
                    result = router.chat(messages, purpose="presentation_rewrite", compute_tier="low")
                except Exception:
                    result = None
                if isinstance(result, dict) and result.get("ok") and (result.get("text") or "").strip():
                    text = (result.get("text") or "").strip()
                    raw_cache = (raw_text or "").strip()
                    if kind == "runtime_status":
                        raw_cache = self._append_provider_status(user_id, raw_cache)
                    self._cache_last_report(user_id, kind=kind, raw_text=raw_cache)
                    return text
        payload = None
        if isinstance(result_dict, dict):
            payload = result_dict.get("payload") if isinstance(result_dict.get("payload"), dict) else None
        presented = present_report(kind=kind, raw_text=raw_text, payload=payload, question=question)
        raw_cache = presented.raw_text
        if kind == "runtime_status":
            raw_cache = self._append_provider_status(user_id, raw_cache)
        self._cache_last_report(user_id, kind=kind, raw_text=raw_cache)
        return presented.summary_text

    def _context(self) -> dict[str, Any]:
        ctx = {"db": self.db, "timezone": self.timezone, "log_path": self.log_path}
        if self._runner:
            ctx["runner"] = self._runner
        if self._llm_broker:
            ctx["llm_broker"] = self._llm_broker
        if self._llm_broker_error:
            ctx["llm_broker_error"] = self._llm_broker_error
        if self.llm_client:
            ctx["llm_router"] = self.llm_client
        return ctx

    def _ask_contains_advice(self, text: str) -> bool:
        lowered = (text or "").lower()
        return any(phrase in lowered for phrase in _ASK_ADVICE_PHRASES)

    def _opinion_trigger(self, text: str) -> str | None:
        lowered = (text or "").lower()
        for phrase in (
            "what do you think",
            "is this unusual",
            "does this look stable",
            "does this seem stable",
            "is this normal for me",
            "outside my baseline",
        ):
            if phrase in lowered:
                return phrase
        return None

    def _store_pending_clarification(
        self,
        user_id: str,
        chat_id: str,
        intent_type: str,
        partial_args: dict[str, Any],
        question: str,
        options: list[str],
        minutes: int = 10,
    ) -> None:
        now_dt = datetime.now(timezone.utc)
        pending_id = str(uuid.uuid4())
        payload_json = json.dumps(partial_args, ensure_ascii=True)
        options_json = json.dumps(options, ensure_ascii=True)
        expires_at = (now_dt + timedelta(minutes=minutes)).isoformat()
        self.db.replace_pending_clarification(
            pending_id,
            user_id,
            chat_id,
            intent_type,
            payload_json,
            question,
            options_json,
            expires_at,
            now_dt.isoformat(),
        )

    def _intent_context(self, chat_id: str | None = None) -> dict[str, Any]:
        context = dict(self._context())
        if chat_id:
            context["chat_id"] = chat_id
        context["knowledge_cache"] = self._knowledge_cache
        return context

    def _maybe_add_narration(self, kind: str, payload: dict[str, Any], text: str) -> str:
        router = LLMNarrationRouter()
        # Never send raw tool payloads to any LLM. Only send redacted report text.
        safe_text = redact_for_llm(text)
        result = router.summarize(kind, {"report_text": safe_text})
        if not result or not result.text:
            return text
        provider = result.provider or "unknown"
        scope = "local" if provider == "ollama" else "cloud" if provider == "openai" else provider
        header = f"Narration ({scope})"
        return f"{header}\n{result.text}\n\n{text}"

    def _maybe_add_narration_from_text(self, kind: str, text: str) -> str:
        router = LLMNarrationRouter()
        safe_text = redact_for_llm(text)
        result = router.summarize(kind, {"report_text": safe_text})
        if not result or not result.text:
            return text
        provider = result.provider or "unknown"
        scope = "local" if provider == "ollama" else "cloud" if provider == "openai" else provider
        header = f"Narration ({scope})"
        return f"{header}\n{result.text}\n\n{text}"

    def _extract_opinion_facts(
        self, skill_name: str, function_name: str, result: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, str | None]:
        if not isinstance(result, dict):
            return None, None
        if skill_name == "knowledge_query":
            facts = (result.get("data") or {}).get("facts")
            intent = (result.get("data") or {}).get("intent") or {}
            if isinstance(facts, dict):
                return facts, (intent.get("name") if isinstance(intent, dict) else None)
            return None, None
        if skill_name == "storage_governor" and function_name == "storage_report":
            facts = result.get("payload")
            return (facts if isinstance(facts, dict) else None), "storage_report"
        if skill_name == "reflection" and function_name == "weekly_reflection":
            facts = result.get("payload")
            return (facts if isinstance(facts, dict) else None), "weekly_reflection"
        if skill_name == "diagnostics" and function_name == "diagnostics_qa":
            facts = result.get("payload")
            return (facts if isinstance(facts, dict) else None), "diagnostics_qa"
        return None, None

    def _record_conversation_topic(
        self,
        user_id: str,
        topic: str | None,
        intent_type: str,
    ) -> None:
        if not topic:
            return
        try:
            record_event(self.db, user_id, topic, intent_type)
        except Exception:
            return

    def _topic_tags_for_decision(self, decision: dict[str, Any] | None) -> list[str]:
        if not decision:
            return []
        if decision.get("type") == "skill_call":
            skill = decision.get("skill")
            function = decision.get("function")
            if skill == "diagnostics":
                topic = (decision.get("args") or {}).get("topic")
                return [topic] if topic else ["diagnostics"]
            if skill == "reflection" and function == "weekly_reflection":
                return ["weekly_reflection"]
            if skill == "observe_now":
                return ["observe_now"]
            if skill == "what_if":
                return ["what_if"]
            if skill == "compare_now":
                return ["compare_now"]
            if skill == "storage_governor" and function == "storage_report":
                return ["storage_report"]
            if skill == "knowledge_query":
                return ["knowledge_query"]
            if function:
                return [function]
            if skill:
                return [skill]
        return []

    def _store_pending_compare(self, user_id: str, what_if_text: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        expires = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
        self._pending_compare[user_id] = {"what_if_text": what_if_text, "expires_at": expires}

    def _get_pending_compare(self, user_id: str) -> dict[str, Any] | None:
        row = self._pending_compare.get(user_id)
        if not row:
            return None
        try:
            if row.get("expires_at") and datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc):
                self._pending_compare.pop(user_id, None)
                return None
        except Exception:
            return None
        return row

    def _conversation_response(
        self,
        user_id: str,
        text: str,
        topic_tags: list[str],
    ) -> OrchestratorResponse:
        router = self.llm_client
        if not router or not hasattr(router, "chat"):
            return OrchestratorResponse("LLM unavailable — chat running in fallback mode.")
        conversation_context = build_conversation_context(
            self.db,
            user_id,
            topic_tags,
            self.timezone,
        )
        context_json = json.dumps(conversation_context, ensure_ascii=True, sort_keys=True)
        messages = [
            {"role": "system", "content": CONVERSATION_SYSTEM_PROMPT},
            {"role": "system", "content": f"Conversation Context:\n{context_json}"},
            {"role": "user", "content": text},
        ]
        result = router.chat(
            messages,
            purpose="chat",
            compute_tier="low",
        )
        if not result.get("ok"):
            return OrchestratorResponse("LLM unavailable — chat running in fallback mode.")
        return OrchestratorResponse((result.get("text") or "").strip())

    def _looks_like_system_check(self, text: str) -> bool:
        lowered = (text or "").strip().lower()
        if not lowered:
            return False
        triggers = (
            "check my computer",
            "check my pc",
            "check my machine",
            "check my system",
            "check the system",
            "system check",
            "health check",
            "check it",
            "check this",
            "is everything ok",
            "is everything okay",
            "is anything wrong",
            "anything wrong",
            "how's my computer doing",
            "how is my computer doing",
            "how's my pc doing",
            "how is my pc doing",
            "how's my machine doing",
            "how is my machine doing",
        )
        if any(t in lowered for t in triggers):
            return True
        if "check" in lowered and any(word in lowered for word in ("computer", "system", "pc", "machine")):
            return True
        return False

    def _bytes_to_human(self, num_bytes: int) -> str:
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

    def _brief_storage_deltas(self) -> list[str]:
        """
        Optional: DB-only deltas from stored storage snapshots (no probes).
        Only emit high-signal changes to keep the brief short.
        """
        bullets: list[str] = []
        # Mount usage deltas.
        for mp in ("/", "/data", "/data2"):
            latest = self.db.get_latest_disk_snapshot(mp)
            if not latest:
                continue
            prev = self.db.get_previous_disk_snapshot(mp, str(latest.get("taken_at") or ""))
            if not prev:
                continue
            try:
                delta = int(latest.get("used_bytes") or 0) - int(prev.get("used_bytes") or 0)
            except Exception:
                continue
            # High-signal only: >= 1GiB shift.
            if abs(delta) >= 1024**3:
                sign = "+" if delta >= 0 else "-"
                bullets.append(f"Storage snapshot {mp}: used {sign}{self._bytes_to_human(abs(delta))}.")
        # Top-dir delta (root/home) if present.
        for scope, label in (("root_top", "Top / dirs snapshot"), ("home_top", "Top home dirs snapshot")):
            latest = self.db.get_latest_dir_size_samples(scope)
            if not latest:
                continue
            prev = self.db.get_previous_dir_size_samples(scope, str(latest.get("taken_at") or ""))
            if not prev:
                continue
            prev_map = {p: int(b) for p, b in (prev.get("samples") or [])}
            best = None
            for path, b in (latest.get("samples") or [])[:10]:
                if path not in prev_map:
                    continue
                try:
                    delta = int(b) - int(prev_map.get(path) or 0)
                except Exception:
                    continue
                if best is None or abs(delta) > abs(best[1]):
                    best = (path, delta)
            if best and abs(int(best[1])) >= 1024**3:
                sign = "+" if int(best[1]) >= 0 else "-"
                bullets.append(f"{label}: {best[0]} {sign}{self._bytes_to_human(abs(int(best[1])))}.")
        return bullets[:2]

    def _brief_network_deltas(self) -> list[str]:
        """
        Optional: DB-only deltas from stored network snapshots (no probes).
        """
        latest = self.db.get_latest_network_snapshot()
        if not latest:
            return []
        prev = self.db.get_previous_network_snapshot(str(latest.get("taken_at") or ""))
        if not prev:
            return []
        bullets: list[str] = []
        if str(latest.get("default_gateway") or "") != str(prev.get("default_gateway") or ""):
            bullets.append(
                "Network default gateway changed: {} -> {}.".format(
                    str(prev.get("default_gateway") or ""),
                    str(latest.get("default_gateway") or ""),
                )
            )
        if str(latest.get("default_iface") or "") != str(prev.get("default_iface") or ""):
            bullets.append(
                "Network default interface changed: {} -> {}.".format(
                    str(prev.get("default_iface") or ""),
                    str(latest.get("default_iface") or ""),
                )
            )
        try:
            nameservers = self.db.get_network_nameservers(str(latest.get("taken_at") or ""))
            prev_nameservers = self.db.get_network_nameservers(str(prev.get("taken_at") or ""))
            if [r.get("nameserver") for r in (nameservers or [])] != [r.get("nameserver") for r in (prev_nameservers or [])]:
                bullets.append("DNS nameservers changed.")
        except Exception:
            pass
        return bullets[:2]

    def _run_brief(self, user_id: str, question: str) -> OrchestratorResponse:
        """
        Capture a fresh system_facts snapshot (via observe_now) and present a short, delta-focused brief.
        No slash-command instructions; "show details" followup remains available.
        """
        observe = self._call_skill(
            user_id,
            "observe_now",
            "observe_now",
            {},
            ["db:write", "sys:read"],
            action_type="insert",
        )
        facts_uid = self._facts_user_id_for_request(user_id)
        report = build_changed_report_from_system_facts(
            self.db,
            user_id=facts_uid,
            timezone=self.timezone,
        )

        # Build short bullets from structured deltas when available.
        bullets: list[str] = []
        if report.delta_summary is None or report.machine_summary is None:
            bullets.append("Baseline created. Nothing to compare yet.")
        else:
            ds = report.delta_summary or {}
            ms = report.machine_summary or {}
            if bool(ds.get("facts_partial")):
                bullets.append("Some checks were partial; changes may be incomplete.")

            if bool(ds.get("rebooted")):
                bullets.append("System restarted since last check.")

            kc = ds.get("kernel_changed") if isinstance(ds.get("kernel_changed"), dict) else None
            if isinstance(kc, dict) and kc.get("from") and kc.get("to"):
                bullets.append(f"Kernel changed: {kc.get('from')} -> {kc.get('to')}.")

            mounts_changed = ds.get("mounts_changed") if isinstance(ds.get("mounts_changed"), list) else []
            mounts_changed = [str(x) for x in mounts_changed if str(x).strip()]
            if mounts_changed:
                shown = ", ".join(mounts_changed[:3])
                more = f" (+{len(mounts_changed) - 3} more)" if len(mounts_changed) > 3 else ""
                bullets.append(f"Mounts changed: {shown}{more}.")

            sig = ms.get("signals") if isinstance(ms.get("signals"), dict) else {}
            disk_from = sig.get("disk_used_pct_from")
            disk_to = sig.get("disk_used_pct_to")
            try:
                if disk_from is not None and disk_to is not None:
                    df = float(disk_from)
                    dt = float(disk_to)
                    if abs(dt - df) >= 1.0:
                        bullets.append(f"Disk /: {df:.0f}% -> {dt:.0f}% ({dt - df:+.0f}%).")
            except Exception:
                pass

            try:
                m_from = sig.get("mem_avail_bytes_from")
                m_to = sig.get("mem_avail_bytes_to")
                if m_from is not None and m_to is not None:
                    bf = int(m_from)
                    bt = int(m_to)
                    delta = bt - bf
                    if abs(delta) >= 512 * 1024**2:
                        sign = "+" if delta >= 0 else "-"
                        bullets.append(
                            f"RAM available: {self._bytes_to_human(bf)} -> {self._bytes_to_human(bt)} ({sign}{self._bytes_to_human(abs(delta))})."
                        )
            except Exception:
                pass

            try:
                l_from = sig.get("load_1m_from")
                l_to = sig.get("load_1m_to")
                cores = int(sig.get("cores") or 1)
                if l_from is not None and l_to is not None:
                    lf = float(l_from)
                    lt = float(l_to)
                    # Only emit load changes when notable.
                    if abs(lt - lf) >= 1.0 or lt >= float(cores) * 1.5:
                        bullets.append(f"Load (1m): {lf:.2f} -> {lt:.2f} ({lt - lf:+.2f}).")
            except Exception:
                pass

            if bool(ds.get("top_rss_changed")):
                bullets.append("Top RAM processes changed.")

            # Optional DB-only appenders (no new probes).
            bullets.extend(self._brief_storage_deltas())
            bullets.extend(self._brief_network_deltas())

        # If nothing high-signal changed, be explicit.
        bullets = [b for b in bullets if isinstance(b, str) and b.strip()]
        if bullets == []:
            bullets = ["No notable changes since last check."]
        if len(bullets) > 7:
            bullets = bullets[:7]

        summary_text = "\n".join([f"- {b}" for b in bullets]).strip() + '\n\nWant details? Say "show details".'

        # Cache detailed output for followups.
        raw = "\n".join(
            [
                "[ObserveNow]",
                observe.text or "",
                "",
                report.raw_text or "",
            ]
        ).strip()
        cache_key = self._cache_last_report(user_id, kind="system_delta", raw_text=raw)
        if cache_key:
            self._register_last_report(
                user_id=user_id,
                kind_hint="delta",
                details_cache_key=cache_key,
                details_text=raw,
            )
        return OrchestratorResponse(summary_text)

    def _run_health_summary(self, user_id: str, question: str) -> OrchestratorResponse:
        # Observer-only: DB reads + deterministic formatting only. No probes/collectors/samplers here.
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            latest_report = self.db.get_latest_unexpired_report(user_id, now_iso=now_iso)
        except Exception:
            latest_report = None

        source = "none"
        source_kind = None
        source_created_at = None
        facts_partial = False
        snapshots_used: list[str] = []
        taken_at: list[str] = []

        disk_used_pct: float | None = None
        mem_avail_bytes: int | None = None
        mem_total_bytes: int | None = None
        mem_avail_pct: float | None = None
        load_1m: float | None = None
        cores: int | None = None
        rebooted: bool | None = None
        load_high: bool | None = None
        load_high_15_count: int | None = None
        load_high_20_count: int | None = None

        if isinstance(latest_report, dict) and (latest_report.get("machine_summary_json") or "").strip():
            source = "report_history"
            source_kind = str(latest_report.get("kind") or "") or None
            source_created_at = str(latest_report.get("created_at") or "") or None
            try:
                ms = json.loads(str(latest_report.get("machine_summary_json") or "{}"))
            except Exception:
                ms = {}
            if isinstance(ms, dict):
                facts_partial = bool(ms.get("facts_partial"))
                snapshots_used = [str(x) for x in (ms.get("snapshots_used") or []) if str(x).strip()]
                taken_at = [str(x) for x in (ms.get("taken_at") or []) if str(x).strip()]
                sig = ms.get("signals") if isinstance(ms.get("signals"), dict) else {}

                def _f(key: str) -> Any:
                    return sig.get(key) if isinstance(sig, dict) else None

                # Disk: prefer "to"/"last"/direct.
                for k in ("disk_used_pct", "disk_used_pct_to", "disk_used_pct_last"):
                    try:
                        v = _f(k)
                        if v is not None:
                            disk_used_pct = float(v)
                            break
                    except Exception:
                        continue

                # Memory bytes: prefer direct/to/last variants.
                for k in ("mem_avail_bytes", "mem_avail_bytes_to", "mem_avail_bytes_last"):
                    try:
                        v = _f(k)
                        if v is not None:
                            mem_avail_bytes = int(v)
                            break
                    except Exception:
                        continue
                for k in ("mem_total_bytes", "mem_total_bytes_to", "mem_total_bytes_last"):
                    try:
                        v = _f(k)
                        if v is not None:
                            mem_total_bytes = int(v)
                            break
                    except Exception:
                        continue
                for k in ("mem_avail_pct", "mem_avail_pct_to", "mem_avail_pct_last", "mem_avail_pct_min"):
                    try:
                        v = _f(k)
                        if v is not None:
                            mem_avail_pct = float(v)
                            break
                    except Exception:
                        continue

                for k in ("load_1m", "load_1m_to", "load_peak_1m"):
                    try:
                        v = _f(k)
                        if v is not None:
                            load_1m = float(v)
                            break
                    except Exception:
                        continue
                try:
                    v = _f("cores")
                    if v is not None:
                        cores = int(v)
                except Exception:
                    cores = None

                rebooted = bool(_f("rebooted")) if _f("rebooted") is not None else None

                # Sustained load from counts when present.
                try:
                    high20 = int(_f("load_high_20_count") or 0)
                except Exception:
                    high20 = 0
                try:
                    high15 = int(_f("load_high_15_count") or 0)
                except Exception:
                    high15 = 0
                load_high_15_count = high15
                load_high_20_count = high20
                if high20 >= 3 or high15 >= 3:
                    load_high = True

        if source == "none":
            # Fallback: latest saved system_facts snapshot only.
            try:
                facts_uid = self._facts_user_id_for_request(user_id)
                row = self.db.get_latest_system_facts_snapshot(facts_uid)
            except Exception:
                row = None
            if not row:
                bullets = [
                    "I don't have any saved system checks yet.",
                    'To create a baseline, ask: "What changed since last time?"',
                ]
                text = "\n".join([f"- {b}" for b in bullets]) + '\n\nWant details? Say "show details".'
                # Cache minimal details for show-details parity (no history write; no data to base it on).
                raw = "\n".join(["[SystemHealthSummary]", "source=none", 'note="no saved data"']).strip()
                self._cache_last_report(user_id, kind="system_health_summary", raw_text=raw)
                return OrchestratorResponse(text)

            source = "system_facts_snapshot"
            source_kind = "system_facts_v1"
            source_created_at = str(row.get("taken_at") or "") or None
            facts_partial = bool(row.get("partial"))
            try:
                facts = json.loads(row.get("facts_json") or "{}")
            except Exception:
                facts = {}
            snap = facts.get("snapshot") if isinstance(facts, dict) else {}
            snapshots_used = [str((snap or {}).get("snapshot_id") or row.get("id") or "")]
            taken_at = [str((snap or {}).get("taken_at") or row.get("taken_at") or "")]
            sig_fs = (facts.get("filesystems") or {}) if isinstance(facts.get("filesystems"), dict) else {}
            mounts = sig_fs.get("mounts") if isinstance(sig_fs.get("mounts"), list) else []
            for m in mounts:
                if isinstance(m, dict) and (m.get("mountpoint") or "") == "/":
                    try:
                        disk_used_pct = float(m.get("used_pct"))
                    except Exception:
                        disk_used_pct = None
                    break
            ram = ((facts.get("memory") or {}).get("ram_bytes") or {}) if isinstance(facts.get("memory"), dict) else {}
            try:
                mem_total_bytes = int(ram.get("total") or 0)
                mem_avail_bytes = int(ram.get("available") or 0)
            except Exception:
                mem_total_bytes, mem_avail_bytes = None, None
            if mem_total_bytes and mem_total_bytes > 0 and mem_avail_bytes is not None:
                mem_avail_pct = (float(mem_avail_bytes) / float(mem_total_bytes)) * 100.0
            cpu = facts.get("cpu") if isinstance(facts, dict) else {}
            load = ((cpu or {}).get("load") or {}) if isinstance(cpu, dict) else {}
            try:
                load_1m = float(load.get("load_1m") or 0.0)
            except Exception:
                load_1m = None
            try:
                cores = int((cpu or {}).get("logical_cores") or 0)
            except Exception:
                cores = None
            if cores is not None and cores <= 0:
                cores = None

        # Compute severity (same conservative rules as system_opinion).
        order = {"ok": 0, "watch": 1, "act_soon": 2}
        disk_sev = "ok"
        if isinstance(disk_used_pct, (int, float)):
            if float(disk_used_pct) >= 90.0:
                disk_sev = "act_soon"
            elif float(disk_used_pct) >= 80.0:
                disk_sev = "watch"

        mem_sev = "ok"
        if isinstance(mem_total_bytes, int) and mem_total_bytes > 0 and isinstance(mem_avail_bytes, int):
            watch_thr = max(1024**3, int(float(mem_total_bytes) * 0.05))
            act_thr = max(512 * 1024**2, int(float(mem_total_bytes) * 0.02))
            watch_thr = min(watch_thr, mem_total_bytes)
            act_thr = min(act_thr, mem_total_bytes)
            if mem_avail_bytes < act_thr:
                mem_sev = "act_soon"
            elif mem_avail_bytes < watch_thr:
                mem_sev = "watch"

        cpu_sev = "ok"
        # If we have sustained-load marker from a report, use it. Otherwise fall back to latest load/cores.
        if isinstance(load_high_20_count, int) and load_high_20_count >= 3:
            cpu_sev = "act_soon"
        elif isinstance(load_high_15_count, int) and load_high_15_count >= 3:
            cpu_sev = "watch"
        elif load_high is True:
            cpu_sev = "watch"
        if isinstance(load_1m, (int, float)) and isinstance(cores, int) and cores > 0:
            if float(load_1m) > float(cores) * 2.0:
                cpu_sev = "act_soon"
            elif float(load_1m) > float(cores) * 1.5 and cpu_sev == "ok":
                cpu_sev = "watch"

        severity = max([disk_sev, mem_sev, cpu_sev], key=lambda s: order.get(s, 0))

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

        # Bullets (2-4 total).
        overall = "Overall health looks OK."
        if severity == "watch":
            overall = "Overall health looks OK, with a few things to watch."
        elif severity == "act_soon":
            overall = "Overall health looks OK overall, but one thing may need attention soon."
        if facts_partial:
            overall = overall.rstrip(".") + " (some checks were partial)."

        bullets: list[str] = [overall]

        # Key signals: pick 1-2 most relevant. Prefer the highest-severity domain(s).
        candidates: list[tuple[int, str]] = []
        if cpu_sev != "ok":
            candidates.append((order.get(cpu_sev, 0), "System load has been high in recent checks."))
        if mem_sev != "ok":
            if isinstance(mem_avail_bytes, int) and isinstance(mem_total_bytes, int) and mem_total_bytes > 0:
                pct = (float(mem_avail_bytes) / float(mem_total_bytes)) * 100.0
                candidates.append((order.get(mem_sev, 0), f"Memory available is {_bytes_to_human(mem_avail_bytes)} ({pct:.0f}% of RAM)."))
            elif isinstance(mem_avail_pct, (int, float)):
                candidates.append((order.get(mem_sev, 0), f"Memory available is about {float(mem_avail_pct):.0f}% of RAM."))
        if disk_sev != "ok":
            if isinstance(disk_used_pct, (int, float)):
                candidates.append((order.get(disk_sev, 0), f"Disk usage on / is {float(disk_used_pct):.0f}%."))

        # If everything is OK, still surface up to two calm facts (disk + memory).
        if severity == "ok":
            if isinstance(disk_used_pct, (int, float)):
                candidates.append((0, f"Disk usage on / is {float(disk_used_pct):.0f}%."))
            if isinstance(mem_avail_bytes, int) and isinstance(mem_total_bytes, int) and mem_total_bytes > 0:
                pct = (float(mem_avail_bytes) / float(mem_total_bytes)) * 100.0
                candidates.append((0, f"Memory available is {_bytes_to_human(mem_avail_bytes)} ({pct:.0f}% of RAM)."))
            elif isinstance(mem_avail_pct, (int, float)):
                candidates.append((0, f"Memory available is about {float(mem_avail_pct):.0f}% of RAM."))

        if rebooted:
            candidates.append((0, "A recent check indicates the system rebooted."))

        # Sort by severity desc, then stable text.
        candidates.sort(key=lambda it: (-int(it[0]), it[1]))
        for _sev, b in candidates:
            if len(bullets) >= 3:
                break
            if b not in bullets:
                bullets.append(b)

        # Offer-only next step only when severity != ok.
        if severity != "ok" and len(bullets) < 4:
            cmd = None
            if disk_sev != "ok":
                cmd = "`df -hT /`"
            elif mem_sev != "ok":
                cmd = "`ps -eo pid,comm,rss,%mem --sort=-rss | head`"
            elif cpu_sev != "ok":
                cmd = "`ps -eo pid,comm,%cpu --sort=-%cpu | head`"
            if cmd:
                bullets.append(f"If you want, you can run: {cmd}.")

        # Ensure 2-4 bullets, deterministic.
        bullets = [b for b in bullets if b]
        if len(bullets) < 2:
            bullets.append("No recent saved measurements to summarize.")
        if len(bullets) > 4:
            bullets = bullets[:4]

        summary_text = "\n".join([f"- {b}" for b in bullets]) + '\n\nWant details? Say "show details".'

        machine_summary = {
            "kind": "health_summary",
            "severity": severity,
            "facts_partial": bool(facts_partial),
            "signals": {
                "disk_used_pct": (round(float(disk_used_pct), 1) if isinstance(disk_used_pct, (int, float)) else None),
                "mem_avail_pct": (round(float(mem_avail_pct), 1) if isinstance(mem_avail_pct, (int, float)) else None),
                "mem_avail_bytes": int(mem_avail_bytes) if isinstance(mem_avail_bytes, int) and isinstance(mem_total_bytes, int) and mem_total_bytes > 0 else None,
                "mem_total_bytes": int(mem_total_bytes) if isinstance(mem_total_bytes, int) and mem_total_bytes > 0 else None,
                "load_1m": (round(float(load_1m), 2) if isinstance(load_1m, (int, float)) else None),
                "cores": (int(cores) if isinstance(cores, int) and cores > 0 else None),
                "load_high": bool(cpu_sev != "ok"),
            },
            "snapshots_used": snapshots_used,
            "taken_at": taken_at,
        }
        ms_json = json.dumps(machine_summary, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

        raw_lines = [
            "[SystemHealthSummary]",
            f"source={source}",
            f"source_kind={source_kind or ''}",
            f"source_created_at={source_created_at or ''}",
            f"machine_summary_json={ms_json}",
        ]
        raw_lines.append(self._provider_status_block(user_id))
        raw = "\n".join(raw_lines).strip()

        cache_key = self._cache_last_report(user_id, kind="system_health_summary", raw_text=raw)

        # Insert into report history with TTL, best-effort. Use now as created_at so TTL reflects "recent context".
        try:
            ttl_s = int(os.getenv("EXPIRES_AFTER_S", "1800") or 1800)
        except Exception:
            ttl_s = 1800
        try:
            created_at = now_iso
            dt = datetime.fromisoformat(created_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            dt = datetime.now(timezone.utc)
            created_at = dt.isoformat()
        expires_at = (dt + timedelta(seconds=int(ttl_s))).isoformat()
        try:
            report_id = str(uuid.uuid4())
            self.db.insert_report_history(
                id=report_id,
                user_id=user_id,
                kind="health_summary",
                created_at=created_at,
                expires_at=expires_at,
                details_cache_key=cache_key,
                machine_summary_json=ms_json,
                facts_snapshot_id=(snapshots_used[-1] if snapshots_used else None),
            )
            try:
                keep = int(os.getenv("MAX_REPORT_HISTORY", "5") or 5)
            except Exception:
                keep = 5
            self.db.delete_reports_older_than_limit(user_id, keep=max(keep, 1))

            # Phase 10B: change journal entry for health summary (best-effort).
            try:
                created_at_journal = created_at
                try:
                    ms_obj = json.loads(ms_json or "{}")
                except Exception:
                    ms_obj = {}
                if isinstance(ms_obj, dict):
                    taken = ms_obj.get("taken_at")
                    if isinstance(taken, list) and taken:
                        t = str(taken[-1] or "").strip()
                        if t:
                            created_at_journal = t
                line = make_journal_line("health_summary", ms_obj if isinstance(ms_obj, dict) else {})
                self.db.insert_journal_entry(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    kind="health_summary",
                    severity=str(severity),
                    created_at=created_at_journal,
                    line=line or f"Health: {severity}",
                    machine_summary_json=ms_json,
                )
            except Exception as exc2:
                try:
                    log_audit_event(
                        self.log_path,
                        event="journal_insert_failed",
                        user_id=user_id,
                        snapshot_id=str((snapshots_used[-1] if snapshots_used else "") or ""),
                        error=str(exc2) or "insert_failed",
                        probe="change_journal",
                        target="db",
                        severity="warn",
                    )
                except Exception:
                    pass
        except Exception as exc:
            try:
                log_audit_event(
                    self.log_path,
                    event="health_summary_history_insert_failed",
                    user_id=user_id,
                    snapshot_id=str((snapshots_used[-1] if snapshots_used else "") or ""),
                    error=str(exc) or "insert_failed",
                    probe="report_history",
                    target="db",
                    severity="warn",
                )
            except Exception:
                pass

        return OrchestratorResponse(summary_text)

    def _looks_like_slow_check(self, text: str) -> bool:
        lowered = (text or "").strip().lower()
        if not lowered:
            return False
        triggers = (
            "why is it slow",
            "why is my computer slow",
            "why is my pc slow",
            "computer is slow",
            "pc is slow",
            "system is slow",
            "everything is slow",
            "running slow",
            "feels slow",
            "laggy",
            "sluggish",
            "stuttering",
            "freezing",
            "unresponsive",
            "takes forever",
        )
        if any(t in lowered for t in triggers):
            return True
        if "slow" in lowered and any(word in lowered for word in ("computer", "system", "pc", "machine", "laptop", "desktop")):
            return True
        return False

    def _help_text(self, user_id: str) -> str:
        role = "observer"
        try:
            role = role_for_user(user_id)
        except Exception:
            role = "observer"

        # Conversational help: no command spam by default.
        base = [
            "Things you can ask in plain English:",
            '- "How\'s my computer doing?" (runs a read-only health check)',
            '- "Why is it slow?" (quick read-only diagnosis: RAM/CPU/disk)',
            '- "What changed since last time?" (captures snapshots and shows deltas)',
            '- "What GPU do I have?"',
            '- "Check disk usage now"',
            '- "Show my network status"',
            "",
            f"Role: {role}",
            '- "/whoami" shows your user_id (set OWNER_USER_ID in env to enable admin commands).',
            'After any report: say "show details" to see full output.',
        ]
        if role == "admin":
            base += [
                "",
                "Admin commands:",
                "- /settings_ui",
                "- /doctor",
                "- /settings_export",
                "- /digest_status",
            ]
        if _ui_mode() == "cli":
            return "\n".join(base)
        if os.getenv("SHOW_COMMANDS_IN_HELP", "0").strip().lower() in {"1", "true", "yes", "y", "on"}:
            base += [
                "",
                "Commands:",
                "- /resource_report",
                "- /hardware_report",
                "- /storage_live_report",
                "- /storage_report",
                "- /network_report",
                "- /runtime_status",
            ]
        return "\n".join(base)

    def _extract_bullets(self, text: str) -> list[str]:
        items: list[str] = []
        for line in (text or "").splitlines():
            ln = line.strip()
            if ln.startswith("- "):
                item = ln[2:].strip()
                if item:
                    items.append(item)
        return items

    def _run_health_check(self, user_id: str, question: str) -> OrchestratorResponse:
        # Deterministic: canonical facts endpoint (live, read-only).
        facts = collect_system_facts({"timezone": self.timezone, "user_id": user_id})
        snap = facts.get("snapshot") if isinstance(facts, dict) else None
        if not isinstance(snap, dict):
            snap = {}
        sid = snap.get("snapshot_id") or "unknown"
        taken_at = snap.get("taken_at") or "unknown"
        raw = (
            "[system_facts_v1]\n"
            + f"snapshot_id={sid}\n"
            + f"taken_at={taken_at}\n\n"
            + json.dumps(facts, ensure_ascii=True, sort_keys=True, indent=2)
        )

        # Map facts into the existing presenters to keep UX stable.
        cpu_load = ((facts.get("cpu") or {}).get("load") or {}) if isinstance(facts.get("cpu"), dict) else {}
        mem = ((facts.get("memory") or {}).get("ram_bytes") or {}) if isinstance(facts.get("memory"), dict) else {}
        swap = ((facts.get("memory") or {}).get("swap_bytes") or {}) if isinstance(facts.get("memory"), dict) else {}
        proc = (facts.get("process_summary") or {}) if isinstance(facts.get("process_summary"), dict) else {}
        fs = (facts.get("filesystems") or {}) if isinstance(facts.get("filesystems"), dict) else {}

        resource_payload = {
            "loads": {
                "1m": float(cpu_load.get("load_1m") or 0.0),
                "5m": float(cpu_load.get("load_5m") or 0.0),
                "15m": float(cpu_load.get("load_15m") or 0.0),
            },
            "memory": {
                "total": int(mem.get("total") or 0),
                "used": int(mem.get("used") or 0),
                "free": int(mem.get("free") or 0),
                "available": int(mem.get("available") or 0),
            },
            "swap": {
                "total": int(swap.get("total") or 0),
                "used": int(swap.get("used") or 0),
            },
            "rss_samples": [
                {"pid": int(r.get("pid") or 0), "name": r.get("name") or "unknown", "rss_bytes": int(r.get("rss_bytes") or 0), "cpu_ticks": 0}
                for r in (proc.get("top_mem") or [])
                if isinstance(r, dict)
            ],
            "cpu_samples": [],
        }
        storage_payload = {
            "mounts": [
                {
                    "mountpoint": m.get("mountpoint") or "?",
                    "total_bytes": int(m.get("total_bytes") or 0),
                    "used_bytes": int(m.get("used_bytes") or 0),
                    "free_bytes": int(m.get("avail_bytes") or 0),
                    "used_pct": float(m.get("used_pct") or 0.0),
                }
                for m in (fs.get("mounts") or [])
                if isinstance(m, dict)
            ]
        }

        res_summary = present_report(
            kind="resource_report",
            raw_text="",
            payload=resource_payload,
            question=question,
        ).summary_text
        sto_summary = present_report(
            kind="storage_live_report",
            raw_text="",
            payload=storage_payload,
            question=question,
        ).summary_text
        bullets = self._extract_bullets(res_summary)[:4] + self._extract_bullets(sto_summary)[:3]

        flags = []
        # Disk hysteresis: warn at 90%, clear at 88%.
        warn_pct = float(os.getenv("DISK_WARN_PCT", "90") or 90.0)
        clear_pct = float(os.getenv("DISK_CLEAR_PCT", "88") or 88.0)
        mounts = (storage_payload or {}).get("mounts") if isinstance(storage_payload, dict) else None
        if isinstance(mounts, list):
            disk_state = self._disk_full_active.setdefault(user_id, {})
            for m in mounts:
                if not isinstance(m, dict):
                    continue
                mp = m.get("mountpoint") or "?"
                pct = float(m.get("used_pct") or 0.0)
                active = bool(disk_state.get(mp))
                should_warn = pct >= warn_pct or (active and pct >= clear_pct)
                disk_state[mp] = should_warn
                if should_warn:
                    flags += flag_disk_full({"mounts": [m]}, pct_threshold=warn_pct)

        flags += flag_high_load(resource_payload)
        # If we have persisted snapshots, we can detect "high load across the last two samples".
        try:
            latest = self.db.get_latest_resource_snapshot()
            prev = self.db.get_previous_resource_snapshot((latest or {}).get("taken_at") or "") if latest else None
        except Exception:
            latest = None
            prev = None
        flags += flag_high_load_sustained(latest, prev)
        flags += flag_ram_hog(resource_payload)
        flags = self._cooldown_flags(user_id, flags, cooldown_s=float(os.getenv("FLAG_COOLDOWN_S", "600") or 600.0))
        for f in flags[:3]:
            bullets.append(f.message)
        if len(bullets) < 3:
            bullets = [
                "Health check ran.",
                "Resource and storage details are available on request.",
                'Say "show details" to see the full output.',
            ]

        cmds = build_helpful_commands(flags)
        cmds_block = ""
        if cmds:
            cmds_block = "\n\nIf you want, you can run:\n" + "\n".join([f"- `{c}`" for c in cmds[:8]])
            if commands_need_warning(cmds):
                cmds_block += (
                    "\n\nNote: commands with `sudo` or `vacuum` delete data (old logs). If you want, ask what they do before running them."
                )

        summary = "\n".join([f"- {b}" for b in bullets[:7]]) + "\n\nWant the full report? Say \"show details\"." + cmds_block
        self._cache_last_report(user_id, kind="computer_health", raw_text=raw)
        return OrchestratorResponse(summary, {"raw": raw})

    def _run_slow_diagnosis(self, user_id: str, question: str) -> OrchestratorResponse:
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

        facts = collect_system_facts({"timezone": self.timezone, "user_id": user_id})
        snap = facts.get("snapshot") if isinstance(facts, dict) else None
        if not isinstance(snap, dict):
            snap = {}
        sid = snap.get("snapshot_id") or "unknown"
        taken_at = snap.get("taken_at") or "unknown"
        collector = (snap.get("collector") or {}) if isinstance(snap.get("collector"), dict) else {}
        facts_partial = bool(collector.get("partial"))

        cpu_obj = facts.get("cpu") if isinstance(facts, dict) else {}
        mem_obj = facts.get("memory") if isinstance(facts, dict) else {}
        fs_obj = facts.get("filesystems") if isinstance(facts, dict) else {}
        proc_obj = facts.get("process_summary") if isinstance(facts, dict) else {}

        load = ((cpu_obj or {}).get("load") or {}) if isinstance(cpu_obj, dict) else {}
        load_1m = float(load.get("load_1m") or 0.0)
        try:
            cores = int((cpu_obj or {}).get("logical_cores") or 0) if isinstance(cpu_obj, dict) else 0
        except Exception:
            cores = 0
        if cores <= 0:
            cores = int(os.cpu_count() or 1)

        ram = ((mem_obj or {}).get("ram_bytes") or {}) if isinstance(mem_obj, dict) else {}
        swap = ((mem_obj or {}).get("swap_bytes") or {}) if isinstance(mem_obj, dict) else {}
        mem_total = int(ram.get("total") or 0)
        mem_avail = int(ram.get("available") or 0)
        mem_used = int(ram.get("used") or 0)
        swap_used = int(swap.get("used") or 0)
        swap_total = int(swap.get("total") or 0)

        # Conservative: flag only when available RAM is below max(1GiB, 5% of total).
        if mem_total > 0:
            mem_low_threshold = max(1024**3, int(float(mem_total) * 0.05))
            mem_low_threshold = min(mem_low_threshold, mem_total)
        else:
            mem_low_threshold = 0
        mem_pressure = bool(mem_total > 0 and mem_avail <= mem_low_threshold)
        cpu_saturation = bool(cores > 0 and load_1m > float(cores) * 1.5)
        warn_pct = float(os.getenv("DISK_WARN_PCT", "90") or 90.0)

        mounts = (fs_obj or {}).get("mounts") if isinstance(fs_obj, dict) else []
        root_mount = None
        if isinstance(mounts, list):
            for m in mounts:
                if isinstance(m, dict) and (m.get("mountpoint") or "") == "/":
                    root_mount = m
                    break
        disk_full = False
        disk_used_pct: float | None = None
        if isinstance(root_mount, dict):
            try:
                disk_used_pct = float(root_mount.get("used_pct") or 0.0)
                disk_full = float(disk_used_pct) >= warn_pct
            except Exception:
                disk_full = False
                disk_used_pct = None

        mem_avail_pct = (float(mem_avail) / float(mem_total) * 100.0) if mem_total > 0 else None
        # Severity is intentionally conservative: "watch" for early signals, "act_soon" only for clear risk.
        severity = "ok"
        try:
            if disk_used_pct is not None and float(disk_used_pct) >= 95.0:
                severity = "act_soon"
            elif disk_full:
                severity = "watch"
        except Exception:
            pass
        try:
            if mem_total > 0:
                mem_act = max(512 * 1024**2, int(float(mem_total) * 0.02))
                mem_act = min(mem_act, mem_total)
                if mem_avail <= mem_act:
                    severity = "act_soon"
                elif mem_pressure and severity == "ok":
                    severity = "watch"
        except Exception:
            pass
        try:
            if cores > 0:
                if load_1m > float(cores) * 2.0:
                    severity = "act_soon"
                elif cpu_saturation and severity == "ok":
                    severity = "watch"
        except Exception:
            pass

        bullets: list[str] = []
        hit_ram = False
        hit_cpu = False
        hit_disk = False
        categories = 0

        # 1) RAM pressure.
        if mem_pressure:
            hit_ram = True
            categories += 1
            pct_avail = (float(mem_avail) / float(mem_total) * 100.0) if mem_total > 0 else 0.0
            pct_thr = (float(mem_low_threshold) / float(mem_total) * 100.0) if mem_total > 0 else 0.0
            bullets.append(
                "RAM pressure: {avail} available ({pct:.0f}% of {total}); threshold {thr} ({thr_pct:.0f}%).".format(
                    avail=_bytes_to_human(mem_avail),
                    pct=pct_avail,
                    total=_bytes_to_human(mem_total),
                    thr=_bytes_to_human(mem_low_threshold),
                    thr_pct=pct_thr,
                )
            )
            top_mem = (proc_obj or {}).get("top_mem") if isinstance(proc_obj, dict) else []
            if isinstance(top_mem, list) and top_mem:
                parts = []
                for r in top_mem[:3]:
                    if not isinstance(r, dict):
                        continue
                    parts.append(f"{r.get('name') or 'unknown'} at {_bytes_to_human(int(r.get('rss_bytes') or 0))}")
                if parts:
                    bullets.append("Top RAM users: " + "; ".join(parts) + ".")
            if swap_total > 0 and swap_used > 0:
                bullets.append(
                    "Swap is in use ({used} of {total}); swapping can make the system feel very slow.".format(
                        used=_bytes_to_human(swap_used),
                        total=_bytes_to_human(swap_total),
                    )
                )

        # 2) CPU saturation (only sample top CPU when this is the strongest signal).
        cpu_rows: list[dict[str, Any]] = []
        sample_taken_at = None
        interval_ms = None
        top_n = None
        if categories < 2 and cpu_saturation:
            hit_cpu = True
            categories += 1
            bullets.append(f"CPU saturation: load (1m) is {load_1m:.2f} on {cores} cores (work is queueing).")
            sample_taken_at = datetime.now(timezone.utc).isoformat()
            try:
                interval_ms = int(os.getenv("CPU_TOP_SAMPLE_MS", "350") or 350)
            except Exception:
                interval_ms = 350
            try:
                top_n = int(os.getenv("CPU_TOP_N", "5") or 5)
            except Exception:
                top_n = 5
            try:
                cpu_rows = sample_top_process_cpu_pct(interval_ms=interval_ms, top_n=top_n)
            except Exception:
                cpu_rows = []
            if cpu_rows:
                parts = []
                for r in cpu_rows[:3]:
                    try:
                        parts.append(
                            "{name} (pid {pid}): {pct:.1f}%".format(
                                name=r.get("name") or "unknown",
                                pid=int(r.get("pid") or 0),
                                pct=float(r.get("cpu_pct") or 0.0),
                            )
                        )
                    except Exception:
                        continue
                if parts:
                    bullets.append(
                        "Top CPU processes (sampled; % is percent of one core, can exceed 100% on multi-core): "
                        + "; ".join(parts)
                        + "."
                    )

        # 3) Disk full.
        if categories < 2 and disk_full and isinstance(root_mount, dict):
            hit_disk = True
            categories += 1
            try:
                pct = float(root_mount.get("used_pct") or 0.0)
            except Exception:
                pct = 0.0
            avail = int(root_mount.get("avail_bytes") or 0)
            bullets.append(f"Disk space is tight: / is {pct:.0f}% used ({_bytes_to_human(avail)} free).")
            bullets.append("Low free disk can slow installs/updates and cause stalls when apps write temp files.")

        if not bullets:
            bullets.append("No single obvious bottleneck in a quick snapshot.")
            bullets.append(f"CPU: load_1m={load_1m:.2f} on {cores} cores.")
            if mem_total > 0:
                bullets.append(f"RAM: {_bytes_to_human(mem_avail)} available (of {_bytes_to_human(mem_total)}).")
            if isinstance(root_mount, dict):
                try:
                    pct = float(root_mount.get("used_pct") or 0.0)
                except Exception:
                    pct = 0.0
                bullets.append(f"Disk / usage: {pct:.0f}%.")

        checked_normals: list[str] = []
        if not hit_ram:
            checked_normals.append("RAM pressure (no strong signal)")
        if not hit_cpu:
            checked_normals.append("CPU saturation (no strong signal)")
        if not hit_disk:
            checked_normals.append("disk-full (no strong signal)")
        if checked_normals:
            prefix = "Also checked: " if (hit_ram or hit_cpu or hit_disk) else "Checks looked normal: "
            bullets.append(prefix + ", ".join(checked_normals) + ".")

        bullets = [b for b in bullets if b]
        # Keep the "checked" bullet as the last line if we have to trim.
        if len(bullets) > 5:
            keep_last = bullets[-1]
            bullets = bullets[:4] + [keep_last]
        if len(bullets) < 3:
            bullets.append("If you want, I can show the full report to dig deeper.")

        notes: list[str] = []
        if cpu_rows:
            notes.append("cpu_sampled")
        if facts_partial:
            notes.append("facts_partial")
        machine_summary = {
            "kind": "slow_diagnosis",
            "severity": severity,
            "facts_partial": bool(facts_partial),
            "snapshots_used": [str(sid)],
            "taken_at": [str(taken_at)],
            "signals": {
                "disk_used_pct": (round(float(disk_used_pct), 1) if isinstance(disk_used_pct, (int, float)) else None),
                "mem_avail_pct": (round(float(mem_avail_pct), 1) if isinstance(mem_avail_pct, (int, float)) else None),
                "mem_avail_bytes": int(mem_avail) if mem_total > 0 else None,
                "mem_total_bytes": int(mem_total) if mem_total > 0 else None,
                "load_1m": round(float(load_1m), 2),
                "cores": int(cores),
                "mem_pressure": bool(mem_pressure),
                "cpu_saturation": bool(cpu_saturation),
                "disk_full": bool(disk_full),
            },
            "notes": notes,
        }
        machine_summary_json = json.dumps(machine_summary, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

        summary = "\n".join([f"- {b}" for b in bullets]) + "\n\nWant the full report? Say \"show details\"."

        raw_lines = [
            "[slow_diagnosis]",
            f"snapshot_id={sid}",
            f"taken_at={taken_at}",
            f"machine_summary_json={machine_summary_json}",
            f"question={question}",
            f"load_1m={load_1m}",
            f"cores={cores}",
            f"mem_available_bytes={mem_avail}",
            f"mem_total_bytes={mem_total}",
            f"swap_used_bytes={swap_used}",
            "",
        ]
        if cpu_rows and sample_taken_at and interval_ms is not None and top_n is not None:
            raw_lines += [
                "top_cpu_sampled=true",
                f"sample_taken_at={sample_taken_at}",
                f"interval_ms={interval_ms}",
                f"top_n={top_n}",
                "cpu_pct_scale=percent_of_one_core (may exceed 100% on multi-core)",
            ]
            for r in cpu_rows[:10]:
                try:
                    raw_lines.append(
                        "- pid={pid} {name} cpu_pct={pct:.2f} rss_bytes={rss}".format(
                            pid=int(r.get("pid") or 0),
                            name=r.get("name") or "unknown",
                            pct=float(r.get("cpu_pct") or 0.0),
                            rss=int(r.get("rss_bytes") or 0),
                        )
                    )
                except Exception:
                    continue
            raw_lines.append("")
        raw_lines.append("[system_facts_v1]")
        raw_lines.append(json.dumps(facts, ensure_ascii=True, sort_keys=True, indent=2))
        raw = "\n".join(raw_lines).strip()
        cache_key = self._cache_last_report(user_id, kind="slow_diagnosis", raw_text=raw)
        self._register_last_report(
            user_id=user_id,
            kind_hint="slow_diagnosis",
            details_cache_key=cache_key,
            details_text=raw,
            facts_snapshot_id_hint=str(sid) if sid else None,
        )
        return OrchestratorResponse(summary, {"raw": raw})

    def _severity_phrase(self, severity: str) -> str:
        sev = (severity or "ok").strip().lower()
        if sev == "act_soon":
            return "needs attention soon"
        if sev == "watch":
            return "worth keeping an eye on"
        return "looks OK"

    def _explain_last_report(self, user_id: str, user_text: str) -> OrchestratorResponse:
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            rows = self.db.list_recent_reports(user_id, limit=int(os.getenv("MAX_REPORT_HISTORY", "5") or 5), include_expired=False, now_iso=now_iso)
        except Exception:
            rows = []
        if not rows:
            return OrchestratorResponse('I don\'t have a recent report yet. Try "what changed?" or "why is it slow?".')

        row = self._select_report_for_followup(rows, user_text=user_text)
        ms_json = (row.get("machine_summary_json") or "").strip()
        try:
            ms = json.loads(ms_json) if ms_json else {}
        except Exception:
            ms = {}

        kind = str(ms.get("kind") or row.get("kind") or "report")
        severity = str(ms.get("severity") or "ok")
        facts_partial = bool(ms.get("facts_partial"))
        snapshots_used = ms.get("snapshots_used") if isinstance(ms.get("snapshots_used"), list) else []
        n = len(snapshots_used)
        based_on = f" (based on last {n} checks)" if n > 1 else ""

        signals = ms.get("signals") if isinstance(ms.get("signals"), dict) else {}

        def _first_present(keys: tuple[str, ...]) -> Any:
            for k in keys:
                if k in signals and signals.get(k) is not None:
                    return signals.get(k)
            return None

        disk_pct = _first_present(("disk_used_pct", "disk_used_pct_to", "disk_used_pct_last", "disk_used_pct_first"))
        mem_pct = _first_present(("mem_avail_pct", "mem_avail_pct_to", "mem_avail_pct_last", "mem_avail_pct_min"))
        load_1m = _first_present(("load_1m", "load_1m_to", "load_peak_1m"))
        cores = _first_present(("cores",))

        bullets: list[str] = []
        bullets.append(f"Last report was a {kind} report: {severity} ({self._severity_phrase(severity)}).{based_on}")

        if kind == "delta":
            rebooted = bool(signals.get("rebooted"))
            kernel_changed = bool(signals.get("kernel_changed"))
            if rebooted:
                bullets.append("It looks like the system rebooted between checks (boot ID changed).")
            if kernel_changed:
                bullets.append("Kernel version changed between checks (likely an update).")
        elif kind == "trend":
            d0 = signals.get("disk_used_pct_first")
            d1 = signals.get("disk_used_pct_last")
            if isinstance(d0, (int, float)) and isinstance(d1, (int, float)):
                bullets.append(f"Disk / usage trend: {d0:.1f}% -> {d1:.1f}%.")
            mmin = signals.get("mem_avail_pct_min")
            mlast = signals.get("mem_avail_pct_last")
            if isinstance(mmin, (int, float)) and isinstance(mlast, (int, float)):
                bullets.append(f"Memory available stayed around {mlast:.1f}% (min {mmin:.1f}%).")
            high = signals.get("load_high_15_count")
            if isinstance(high, int) and n > 0:
                bullets.append(f"High load checks: {high}/{n} over the window.")

        # Generic signal interpretations.
        if isinstance(disk_pct, (int, float)):
            bullets.append(f"Disk / is at {float(disk_pct):.1f}% used.")
        if isinstance(mem_pct, (int, float)):
            bullets.append(f"Memory available is about {float(mem_pct):.1f}% of RAM (Linux uses cache; 'available' is the key number).")
        if isinstance(load_1m, (int, float)):
            if isinstance(cores, int) and cores > 0:
                bullets.append(f"Load (1m) is {float(load_1m):.2f} with {cores} cores; sustained load much above cores can feel slow.")
            else:
                bullets.append(f"Load (1m) is {float(load_1m):.2f}.")

        if facts_partial:
            bullets.append("Some checks were partial; treat this as best-effort.")

        # Trim and finish.
        bullets = [b for b in bullets if b]
        if len(bullets) > 6:
            bullets = bullets[:6]
        text = "\n".join([f"- {b}" for b in bullets]) + "\n\nWant the full report? Say \"show details\"."
        # Pin subsequent "show details" to the chosen report (not necessarily the newest one).
        self._set_selected_details_cache_key(user_id, str(row.get("details_cache_key") or ""))
        return OrchestratorResponse(text)

    def _advise_from_last_report(self, user_id: str, user_text: str) -> OrchestratorResponse:
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            rows = self.db.list_recent_reports(user_id, limit=int(os.getenv("MAX_REPORT_HISTORY", "5") or 5), include_expired=False, now_iso=now_iso)
        except Exception:
            rows = []
        if not rows:
            return OrchestratorResponse('I don\'t have a recent report yet. Try "what changed?" or "why is it slow?".')

        row = self._select_report_for_followup(rows, user_text=user_text)
        ms_json = (row.get("machine_summary_json") or "").strip()
        try:
            ms = json.loads(ms_json) if ms_json else {}
        except Exception:
            ms = {}

        kind = str(ms.get("kind") or row.get("kind") or "report")
        severity = str(ms.get("severity") or "ok")
        facts_partial = bool(ms.get("facts_partial"))
        signals = ms.get("signals") if isinstance(ms.get("signals"), dict) else {}

        def _first_present(keys: tuple[str, ...]) -> Any:
            for k in keys:
                if k in signals and signals.get(k) is not None:
                    return signals.get(k)
            return None

        disk_pct = _first_present(("disk_used_pct", "disk_used_pct_to", "disk_used_pct_last"))
        mem_pct = _first_present(("mem_avail_pct", "mem_avail_pct_to", "mem_avail_pct_last", "mem_avail_pct_min"))
        load_1m = _first_present(("load_1m", "load_1m_to", "load_peak_1m"))
        cores = _first_present(("cores",))

        bullets: list[str] = []
        bullets.append(f"Based on the last {kind} report: {severity} ({self._severity_phrase(severity)}).")
        if facts_partial:
            bullets.append("Note: some checks were partial, so treat this as best-effort.")

        cmds: list[str] = []
        try:
            if isinstance(disk_pct, (int, float)) and float(disk_pct) >= 80.0:
                cmds += ["df -hT /", "du -xh --max-depth=1 /home | sort -h | tail"]
        except Exception:
            pass
        try:
            if isinstance(mem_pct, (int, float)) and float(mem_pct) <= 10.0:
                cmds += ["ps -eo pid,comm,rss,%mem --sort=-rss | head"]
        except Exception:
            pass
        try:
            if isinstance(load_1m, (int, float)) and isinstance(cores, int) and cores > 0 and float(load_1m) > float(cores) * 1.5:
                cmds += ["ps -eo pid,comm,%cpu --sort=-%cpu | head", "top -o %CPU"]
        except Exception:
            pass

        # Fallback: if we don't have strong signals, still offer safe quick checks.
        if not cmds:
            cmds = ["df -hT /", "ps -eo pid,comm,rss,%mem --sort=-rss | head", "ps -eo pid,comm,%cpu --sort=-%cpu | head"]

        # De-dupe while preserving order.
        seen: set[str] = set()
        cmds2: list[str] = []
        for c in cmds:
            if c in seen:
                continue
            seen.add(c)
            cmds2.append(c)
        cmds2 = cmds2[:4]

        bullets.append("If you want, you can run: " + "; ".join([f"`{c}`" for c in cmds2]) + ".")
        if kind in {"slow_diagnosis", "delta", "trend"}:
            bullets.append("If you tell me what feels slow (apps, browser, boot, installs), I can narrow it down without running anything new.")

        bullets = [b for b in bullets if b]
        if len(bullets) > 6:
            bullets = bullets[:6]
        text = "\n".join([f"- {b}" for b in bullets]) + "\n\nWant the full report? Say \"show details\"."
        self._set_selected_details_cache_key(user_id, str(row.get("details_cache_key") or ""))
        return OrchestratorResponse(text)

    def _select_report_for_followup(self, rows: list[dict[str, Any]], *, user_text: str) -> dict[str, Any]:
        # Rows are expected most-recent-first. Selection must be deterministic.
        if not rows:
            return {}
        if len(rows) == 1:
            return rows[0]

        lowered = (user_text or "").lower()
        # These hint checks are used by callers who pass the *user message*.
        # For now, if the caller passes a label, we just pick most recent.
        hint_kind = None
        if any(w in lowered for w in ("trend", "over time", "since when")):
            hint_kind = "trend"
        elif any(w in lowered for w in ("changed", "different")):
            hint_kind = "delta"
        elif "slow" in lowered:
            hint_kind = "slow_diagnosis"
        elif any(w in lowered for w in ("worry", "concern", "recommend", "next steps", "advice", "opinion")):
            hint_kind = "opinion"

        if hint_kind:
            for r in rows:
                if str(r.get("kind") or "") == hint_kind:
                    return r
        return rows[0]

    # Note: Tool choice must stay deterministic. Do not add LLM-based tool execution here.

    def _llm_mode_text(self) -> str:
        config = getattr(self.llm_client, "config", None)
        selector = self._llm_selector()
        fallback_reason = self._llm_fallback_reason()

        allow_remote_chat = bool(getattr(config, "llm_allow_remote", False)) if config else False
        intent_assist = bool(getattr(config, "llm_intent_assist", False)) if config else False

        diagnostics = {}
        if self.llm_client and hasattr(self.llm_client, "diagnostics"):
            try:
                diagnostics = self.llm_client.diagnostics()
            except Exception:
                diagnostics = {}
        resolved = diagnostics.get("resolved") if isinstance(diagnostics, dict) else {}
        chat_provider = (resolved or {}).get("provider") or "none"
        chat_model = (resolved or {}).get("model") or "none"

        narration_enabled = os.getenv("ENABLE_NARRATION", "").strip().lower() in {"1", "true", "yes", "y", "on"}
        narration_routing = os.getenv("LLM_ROUTING", "auto").strip().lower() or "auto"
        narration_allow_remote = os.getenv("LLM_NARRATION_ALLOW_REMOTE", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

        effective = (
            "effective: chat selector={selector} provider={provider} model={model} "
            "remote_chat={remote_chat} remote_narration={remote_narr} intent_assist_local_only={intent_assist}"
        ).format(
            selector=selector,
            provider=chat_provider,
            model=chat_model,
            remote_chat="true" if allow_remote_chat else "false",
            remote_narr="true" if (allow_remote_chat and narration_allow_remote) else "false",
            intent_assist="true" if intent_assist else "false",
        )

        lines = ["LLM mode:", effective]
        if (chat_provider or "none") == "none" and fallback_reason:
            lines.append(f"last_fallback_reason={fallback_reason}")
        lines += [
            f"narration_enabled={'true' if narration_enabled else 'false'}",
            f"narration_routing={narration_routing}",
            f"narration_allow_remote={'true' if narration_allow_remote else 'false'}",
        ]
        return "\n".join(lines)

    def _format_opinion_reply(self, facts_text: str, opinion_text: str) -> str:
        return f"Facts:\n{facts_text}\n\nOpinion (opt-in):\n{opinion_text}"

    def _llm_selector(self) -> str:
        config = getattr(self.llm_client, "config", None)
        requested = getattr(config, "llm_selector_requested", None) if config else None
        fallback_reason = getattr(config, "llm_broker_fallback_reason", None) if config else None
        if requested == "broker" and fallback_reason:
            return "direct"
        if self._llm_broker or self._llm_broker_error:
            return "broker"
        selector = getattr(config, "llm_selector", None) if config else None
        return (selector or os.getenv("LLM_SELECTOR", "single") or "single").strip().lower()

    def _llm_fallback_reason(self) -> str | None:
        config = getattr(self.llm_client, "config", None)
        requested = getattr(config, "llm_selector_requested", None) if config else None
        fallback_reason = getattr(config, "llm_broker_fallback_reason", None) if config else None
        if requested == "broker" and fallback_reason:
            return fallback_reason
        if self._llm_broker_error:
            return self._llm_broker_error
        return None

    def _broker_diagnostics(self) -> dict[str, str]:
        if not self._llm_broker:
            if self._llm_broker_error:
                return {
                    "broker_choice": "none",
                    "broker_reason": self._llm_broker_error,
                    "broker_provider": "none",
                    "broker_model": "none",
                }
            return {}
        try:
            _, decision = self._llm_broker.select(TaskSpec(task="presentation_rewrite"))
        except Exception as exc:  # pragma: no cover - defensive
            return {
                "broker_choice": "none",
                "broker_reason": f"error:{exc}",
                "broker_provider": "none",
                "broker_model": "none",
            }
        winner = decision.get("winner") or {}
        return {
            "broker_choice": decision.get("winner_id") or "none",
            "broker_reason": decision.get("failure_reason") or "selected",
            "broker_provider": winner.get("provider") or "none",
            "broker_model": winner.get("model") or "none",
        }

    def _llm_status_text(self) -> str:
        config = getattr(self.llm_client, "config", None)
        selector = self._llm_selector()
        fallback_reason = self._llm_fallback_reason()
        raw_llm_selector = os.getenv("LLM_SELECTOR")
        raw_llm_provider = os.getenv("LLM_PROVIDER")
        raw_llm_model = os.getenv("LLM_MODEL")
        raw_llm_allow_remote = os.getenv("LLM_ALLOW_REMOTE")

        provider = (getattr(config, "llm_provider", None) if config else None) or (
            raw_llm_provider if raw_llm_provider is not None else "none"
        )
        provider = provider.strip().lower() if provider else "none"
        allow_remote = (
            getattr(config, "llm_allow_remote", None)
            if config is not None
            else os.getenv("LLM_ALLOW_REMOTE", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
        )

        openai_key_present = bool(
            getattr(config, "openai_api_key", None) if config else os.getenv("OPENAI_API_KEY", "")
        )
        openrouter_key_present = bool(
            getattr(config, "openrouter_api_key", None)
            if config
            else os.getenv("OPENROUTER_API_KEY", "")
        )
        anthropic_key_present = bool(
            getattr(config, "anthropic_api_key", None)
            if config
            else os.getenv("ANTHROPIC_API_KEY", "")
        )

        diagnostics = {}
        if self.llm_client and hasattr(self.llm_client, "diagnostics"):
            try:
                diagnostics = self.llm_client.diagnostics()
            except Exception:
                diagnostics = {}

        resolved_provider = (
            ((diagnostics.get("resolved") or {}).get("provider") or "").strip().lower()
            if isinstance(diagnostics, dict)
            else ""
        )
        resolved_model = (
            (diagnostics.get("resolved") or {}).get("model")
            if isinstance(diagnostics, dict)
            else None
        )
        resolve_reason = (
            (diagnostics.get("resolved") or {}).get("reason")
            if isinstance(diagnostics, dict)
            else None
        )
        providers_loaded = diagnostics.get("providers_loaded") if isinstance(diagnostics, dict) else None
        openrouter_provider_loaded = (
            bool(diagnostics.get("openrouter_provider_loaded"))
            if isinstance(diagnostics, dict) and "openrouter_provider_loaded" in diagnostics
            else None
        )

        # "provider/model" is what /llm_ping will actually attempt to use (if chat routing is implemented).
        model = "none"
        if resolved_provider and resolved_provider != "none":
            provider = resolved_provider
            model = resolved_model or "none"
        else:
            if provider == "openai":
                model = (
                    getattr(config, "openai_model", None) if config else os.getenv("OPENAI_MODEL", "none")
                )
            elif provider == "openrouter":
                model = (
                    getattr(config, "openrouter_model", None)
                    if config
                    else os.getenv("OPENROUTER_MODEL", "none")
                )
            elif provider == "ollama":
                model = (
                    getattr(config, "ollama_model", None) if config else os.getenv("OLLAMA_MODEL", "none")
                )

        ollama_url = None
        if config:
            ollama_url = config.ollama_base_url or config.ollama_host
        if not ollama_url:
            ollama_url = os.getenv("OLLAMA_BASE_URL", "").strip() or os.getenv("OLLAMA_HOST", "").strip()

        ollama_ok, ollama_reason = ping_ollama_with_reason(ollama_url or "", timeout_s=2)
        ollama_status = "ok" if ollama_ok else "fail"

        broker_info = self._broker_diagnostics() if selector == "broker" else {}

        lines = [
            "LLM status:",
            f"openrouter_key_present={'true' if openrouter_key_present else 'false'}",
            f"raw_LLM_SELECTOR={raw_llm_selector!r}",
            f"raw_LLM_PROVIDER={raw_llm_provider!r}",
            f"raw_LLM_MODEL={raw_llm_model!r}",
            f"raw_LLM_ALLOW_REMOTE={raw_llm_allow_remote!r}",
            f"selector={selector}",
            f"provider={provider}",
            f"model={model or 'none'}",
            f"allow_remote={'true' if allow_remote else 'false'}",
        ]
        if fallback_reason:
            lines.append(f"fallback_reason={fallback_reason}")
        if resolve_reason and resolve_reason != fallback_reason:
            lines.append(f"resolver_reason={resolve_reason}")
        if providers_loaded is not None:
            lines.append(f"providers_loaded={providers_loaded}")
        if openrouter_provider_loaded is not None:
            lines.append(f"openrouter_provider_loaded={'true' if openrouter_provider_loaded else 'false'}")
        if openrouter_key_present and allow_remote and openrouter_provider_loaded is False:
            lines.append("error=openrouter_provider_not_loaded")
        if selector == "broker":
            lines.append(f"broker_choice={broker_info.get('broker_choice', 'none')}")
            lines.append(f"broker_reason={broker_info.get('broker_reason', 'unknown')}")
        lines.append(f"openai_api_key_present={'true' if openai_key_present else 'false'}")
        lines.append(f"openrouter_api_key_present={'true' if openrouter_key_present else 'false'}")
        lines.append(f"anthropic_api_key_present={'true' if anthropic_key_present else 'false'}")
        lines.append(f"ollama_base_url_present={'true' if bool(ollama_url) else 'false'}")
        if ollama_ok:
            lines.append("ollama_ping=ok")
        else:
            reason = ollama_reason or "ollama_unreachable"
            lines.append(f"ollama_ping=fail reason={reason}")
        return "\n".join(lines)


    def _deliver_pending_opinion(
        self, user_id: str, pending: opinion_gate.PendingOpinion
    ) -> OrchestratorResponse:
        facts = pending.context.get("facts")
        facts_text = pending.context.get("facts_text") or "No facts available."
        context_note = pending.context.get("context_note")
        if not isinstance(facts, dict) or not facts:
            opinion_gate.clear_pending(self.db, user_id)
            return OrchestratorResponse("No facts available to form an opinion.")
        opinion_resp = self._call_skill(
            user_id,
            "opinion_on_report",
            "opinion_on_report",
            {"facts": facts, "context_note": context_note},
            [],
        )
        opinion_text = opinion_resp.text or "I can’t form a reliable opinion from the provided facts."
        final_text = self._format_opinion_reply(facts_text, opinion_text)
        opinion_gate.clear_pending(self.db, user_id)
        log_event(
            self.log_path,
            "opinion_delivered",
            {"user_id": user_id, "topic_key": pending.topic_key, "facts_hash": facts_hash(facts)},
        )
        return OrchestratorResponse(final_text, opinion_resp.data)

    def _call_skill(
        self,
        user_id: str,
        skill_name: str,
        function_name: str,
        args: dict[str, Any],
        requested_permissions: list[str],
        action_type: str | None = None,
        confirmed: bool = False,
    ) -> OrchestratorResponse:
        skill = self.skills.get(skill_name)
        if not skill:
            return OrchestratorResponse("Skill not found.")

        func = skill.functions.get(function_name)
        if not func:
            return OrchestratorResponse("Function not found.")

        action = {"action_type": action_type or ""}
        decision = evaluate_policy(skill.permissions, requested_permissions, action)
        if not decision.allowed:
            return OrchestratorResponse("Permission denied.")

        if decision.requires_confirmation and not confirmed:
            pending = PendingAction(
                user_id=user_id,
                action={
                    "skill": skill_name,
                    "function": function_name,
                    "args": args,
                    "requested_permissions": requested_permissions,
                    "action_type": action_type,
                },
                message="This will delete or overwrite data. Reply /confirm to proceed.",
            )
            self.confirmations.set(pending)
            return OrchestratorResponse(pending.message)

        ctx = dict(self._context())
        ctx["user_id"] = user_id
        result = func.handler(ctx, **args)
        log_event(self.log_path, "skill_call", {"skill": skill_name, "function": function_name})
        if self._runner and self._runner.mode in {"sandbox", "live"}:
            if not self._audit_runner_result(skill_name, user_id):
                return OrchestratorResponse(AUDIT_HARD_FAIL_MSG)
        response_text = "OK"
        if isinstance(result, dict) and result.get("text"):
            response_text = str(result["text"])
        if skill_name == "knowledge_query" and isinstance(result, dict):
            data = result.get("data", {})
            facts = data.get("facts")
            intent = data.get("intent")
            query = args.get("query") or ""
            if isinstance(facts, dict):
                entry = self._knowledge_cache.set(user_id, query, facts, intent)
                log_event(
                    self.log_path,
                    "knowledge_query_cached",
                    {
                        "user_id": user_id,
                        "facts_hash": entry.facts_hash,
                        "intent": (intent or {}).get("name") if isinstance(intent, dict) else None,
                        "query_len": len(query),
                    },
                )
                opinion_gate.store_pending(
                    self.db,
                    user_id,
                    topic_key="knowledge_query",
                    context={
                        "facts": facts,
                        "facts_text": response_text,
                        "context_note": (intent or {}).get("name") if isinstance(intent, dict) else None,
                    },
                    log_path=self.log_path,
                )
                response_text = f"{response_text}\n---\n{opinion_gate.OPINION_GATE_PROMPT}"
        if skill_name == "disk_report" and isinstance(result, dict):
            response_text = self._maybe_add_narration("disk_report", result, response_text)
        memory_text = response_text
        kind = self._report_kind(skill_name, function_name)
        if kind and _ui_mode() == "conversational":
            # Keep long/raw reports out of conversation memory by default. This also reduces the
            # risk of raw tool output leaking into any downstream LLM context.
            payload = None
            if isinstance(result, dict) and isinstance(result.get("payload"), dict):
                payload = result.get("payload")
            memory_text = present_report(kind=kind, raw_text=response_text, payload=payload, question=None).summary_text
        try:
            memory_ingest.ingest_event(
                self.db,
                user_id,
                "skill",
                memory_text,
                [function_name or skill_name],
                override=None,
            )
        except Exception:
            pass
        return OrchestratorResponse(response_text, result)

    def _format_projects(self, projects: list[dict[str, Any]]) -> str:
        if not projects:
            return "No projects yet. Use /project_new to add one."
        lines = ["Projects:"]
        for proj in projects:
            status = proj.get("status", "")
            pitch = proj.get("pitch") or ""
            suffix = f" - {pitch}" if pitch else ""
            lines.append(f"- {proj['name']} ({status}){suffix}")
        return "\n".join(lines)

    def handle_intent(self, decision: dict[str, Any], user_id: str) -> OrchestratorResponse:
        self._runner = Runner()
        try:
            if decision.get("type") != "skill_call":
                return OrchestratorResponse(decision.get("text", ""))

            skill_name = decision.get("skill", "")
            function_name = decision.get("function", "")
            args = decision.get("args") or {}

            permissions_map: dict[str, tuple[list[str], str | None]] = {
                "remember_note": (["db:write"], "insert"),
                "list_projects": (["db:read"], None),
                "add_reminder": (["db:write"], "insert"),
                "set_reminder": (["db:write"], "insert"),
                "next_best_task": (["db:read"], None),
                "daily_plan": (["db:read"], None),
                "weekly_review": (["db:read"], None),
                "restart_agent": (["ops:supervisor"], "ops_restart"),
                "service_status": (["ops:supervisor"], None),
                "service_logs": (["ops:supervisor"], None),
            }
            requested_permissions, action_type = permissions_map.get(function_name, ([], None))
            result = self._call_skill(
                user_id,
                skill_name,
                function_name,
                args,
                requested_permissions,
                action_type=action_type,
            )

            if function_name == "remember_note":
                return OrchestratorResponse("Saved.") if result.data else result
            if function_name in {"add_reminder", "set_reminder"}:
                return OrchestratorResponse("Reminder set.") if result.data else result
            if function_name == "list_projects":
                if result.data:
                    return OrchestratorResponse(self._format_projects(result.data.get("projects", [])))
                return result
            if function_name == "next_best_task":
                return OrchestratorResponse("Next-best-task suggestions are coming soon.")
            if function_name == "daily_plan":
                return OrchestratorResponse("Planning is coming soon.")
            if function_name == "weekly_review":
                return OrchestratorResponse("Weekly review is coming soon.")

            return result
        finally:
            self._runner = None

    def handle_message(self, text: str, user_id: str) -> OrchestratorResponse:
        self._runner = Runner()
        try:
            override, cleaned_text = memory_ingest.parse_memory_override(text)
            # "Show details" followup for the most recent read-only report.
            if not (text or "").lstrip().startswith("/") and wants_raw_details(text):
                raw = self._last_report_raw(user_id)
                if raw:
                    return OrchestratorResponse(f"Full report:\n{raw}")
                return OrchestratorResponse("I don't have a recent report to show. Ask me to run a check first.")
            if not (text or "").lstrip().startswith("/"):
                lowered = (text or "").strip().lower()
                if lowered in {"help", "commands", "command", "menu", "what can you do", "what do you do"}:
                    return OrchestratorResponse(self._help_text(user_id))
            cmd = parse_command(text)
            if cmd and cmd.name == "nomem":
                cmd = None
                text = cleaned_text
            if cmd:
                if cmd.name != "open_loops":
                    try:
                        memory_ingest.ingest_event(
                            self.db,
                            user_id,
                            "user",
                            cleaned_text if override else text,
                            [cmd.name],
                            override=override,
                        )
                    except Exception:
                        pass
                if cmd.name == "confirm":
                    pending = self.confirmations.pop(user_id)
                    if not pending:
                        return OrchestratorResponse("No pending action to confirm.")
                    action = pending.action
                    return self._call_skill(
                        user_id,
                        action["skill"],
                    action["function"],
                    action["args"],
                    action["requested_permissions"],
                    action.get("action_type"),
                    confirmed=True,
                )

                if cmd.name == "help":
                    return OrchestratorResponse(self._help_text(user_id))

                if cmd.name == "remember":
                    result = self._call_skill(
                        user_id,
                        "core",
                        "remember_note",
                        {"text": cmd.args},
                        ["db:write"],
                        action_type="insert",
                    )
                    return OrchestratorResponse("Saved.") if result.data else result

                if cmd.name == "projects":
                    result = self._call_skill(
                        user_id,
                        "core",
                        "list_projects",
                        {},
                        ["db:read"],
                    )
                    if result.data:
                        return OrchestratorResponse(self._format_projects(result.data.get("projects", [])))
                    return result

                if cmd.name == "project_new":
                    name, pitch = split_pipe_args(cmd.args, 2)
                    result = self._call_skill(
                        user_id,
                        "core",
                        "add_project",
                        {"name": name, "pitch": pitch or None},
                        ["db:write"],
                        action_type="insert",
                    )
                    return OrchestratorResponse("Project created.") if result.data else result

                if cmd.name == "task_add":
                    project, title, effort, impact = split_pipe_args(cmd.args, 4)
                    effort_mins = int(effort) if effort else None
                    impact_1to5 = int(impact) if impact else None
                    result = self._call_skill(
                        user_id,
                        "core",
                        "add_task",
                        {
                            "project": project or None,
                            "title": title,
                            "effort_mins": effort_mins,
                            "impact_1to5": impact_1to5,
                        },
                        ["db:write"],
                        action_type="insert",
                    )
                    return OrchestratorResponse("Task added.") if result.data else result

                if cmd.name == "remind":
                    when_local, text = split_pipe_args(cmd.args, 2)
                    result = self._call_skill(
                        user_id,
                        "core",
                        "add_reminder",
                        {"when_local": when_local, "text": text},
                        ["db:write"],
                        action_type="insert",
                    )
                    return OrchestratorResponse("Reminder set.") if result.data else result

                if cmd.name == "next":
                    return OrchestratorResponse("Next-best-task suggestions are coming soon.")

                if cmd.name == "plan":
                    return OrchestratorResponse("Planning is coming soon.")

                if cmd.name == "weekly":
                    return OrchestratorResponse("Weekly review is coming soon.")

                if cmd.name == "done":
                    return OrchestratorResponse("Logged. Mapping to tasks coming soon.")

                if cmd.name == "audit":
                    entries = self.db.audit_log_list_recent(user_id, limit=10)
                    if not entries:
                        return OrchestratorResponse("No audit entries yet.")
                    lines = ["Recent audit entries:"]
                    for entry in entries:
                        lines.append(
                            f"- {entry['created_at']} {entry['action_type']}:{entry['action_id']} {entry['status']}"
                        )
                    return OrchestratorResponse("\n".join(lines))

                if cmd.name == "status":
                    writes_flag = "on" if self.enable_writes else "off"
                    actions = ["apt_cache", "journald_vacuum", "home_cache"]
                    last_report = self.db.activity_log_latest("disk_report") or "none"
                    audit_entries = self.db.audit_log_list_recent(user_id, limit=5)
                    lines = [
                        f"ENABLE_WRITES: {writes_flag}",
                        "Write actions: " + ", ".join(actions),
                        "Sudo: no",
                        f"Last disk_report: {last_report}",
                    ]
                    if audit_entries:
                        lines.append("Recent audits:")
                        for entry in audit_entries:
                            lines.append(f"- {entry['id']} {entry['status']}")
                    return OrchestratorResponse("\n".join(lines))

                if cmd.name == "restart":
                    return self._call_skill(
                        user_id,
                        "ops_supervisor",
                        "restart_agent",
                        {},
                        ["ops:supervisor"],
                        action_type="ops_restart",
                    )

                if cmd.name == "service_status":
                    return self._call_skill(
                        user_id,
                        "ops_supervisor",
                        "service_status",
                        {},
                        ["ops:supervisor"],
                    )

                if cmd.name == "logs":
                    lines_arg = cmd.args.strip() if cmd.args else ""
                    lines_val = int(lines_arg) if lines_arg.isdigit() else None
                    return self._call_skill(
                        user_id,
                        "ops_supervisor",
                        "service_logs",
                        {"lines": lines_val},
                        ["ops:supervisor"],
                    )

                if cmd.name == "runtime_status":
                    response = self._call_skill(
                        user_id,
                        "runtime_status",
                        "runtime_status",
                        {},
                        ["db:read"],
                    )
                    if _ui_mode() == "conversational":
                        response.text = self._present_readonly_report(
                            user_id=user_id,
                            kind="runtime_status",
                            raw_text=response.text,
                            result_dict=response.data if isinstance(response.data, dict) else None,
                            question=text,
                        )
                    return response

                if cmd.name == "doctor":
                    report = run_doctor(
                        self.db,
                        user_id=user_id,
                        now_iso=datetime.now(timezone.utc).isoformat(),
                        env_path=(os.getenv("AGENT_ENV_PATH", "/etc/personal-agent/agent.env") or "/etc/personal-agent/agent.env"),
                        token_path=(os.getenv("SETTINGS_UI_TOKEN_PATH", "/etc/personal-agent/ui.token") or "/etc/personal-agent/ui.token"),
                        pid_path=(os.getenv("SETTINGS_UI_PID_PATH", "/run/personal-agent/settings_ui.pid") or "/run/personal-agent/settings_ui.pid"),
                        log_path=self.log_path,
                    )
                    text_out = "\n".join(report.lines).strip()
                    if text_out:
                        text_out += "\n\nWant details? Say \"show details\"."
                    raw = report.details_text
                    self._cache_last_report(user_id, kind="doctor", raw_text=raw)
                    return OrchestratorResponse(text_out, {"severity": report.severity})

                if cmd.name == "autonomy_status":
                    return self._call_skill(
                        user_id,
                        "core",
                        "autonomy_status",
                        {},
                        [],
                    )

                if cmd.name == "autonomy_simulate":
                    return self._call_skill(
                        user_id,
                        "core",
                        "autonomy_simulate",
                        {"capability": cmd.args.strip() if cmd.args else ""},
                        [],
                    )

                if cmd.name == "llm_status":
                    return OrchestratorResponse(self._llm_status_text())

                if cmd.name == "llm_mode":
                    return OrchestratorResponse(self._llm_mode_text())

                if cmd.name == "llm_ping":
                    selector = self._llm_selector()
                    fallback_reason = self._llm_fallback_reason()
                    selector_display = selector
                    if selector == "direct" and fallback_reason:
                        selector_display = f"direct (fallback from broker: {fallback_reason})"
                    broker_info = self._broker_diagnostics() if selector == "broker" else {}
                    router = self.llm_client
                    if not router or not hasattr(router, "chat"):
                        reason = "unavailable"
                        if selector == "broker":
                            return OrchestratorResponse(
                                "LLM ping: selector={selector} provider={provider} model={model} status=FAIL duration_ms=0 broker_choice={broker_choice} broker_reason={broker_reason} reason={reason}".format(
                                    selector=selector_display,
                                    provider=broker_info.get("broker_provider", "none"),
                                    model=broker_info.get("broker_model", "none"),
                                    broker_choice=broker_info.get("broker_choice", "none"),
                                    broker_reason=broker_info.get("broker_reason", "unknown"),
                                    reason=reason,
                                )
                            )
                        return OrchestratorResponse(
                            "LLM ping: selector={selector} provider=none model=none status=FAIL duration_ms=0 reason={reason}".format(
                                selector=selector_display,
                                reason=reason,
                            )
                        )
                    provider_override = None
                    if cmd.args:
                        provider_override = cmd.args.strip().lower() or None
                    result = router.chat(
                        [
                            {"role": "system", "content": "Reply with the single word PONG."},
                            {"role": "user", "content": "ping"},
                        ],
                        purpose="diagnostics",
                        compute_tier="low",
                        provider_override=provider_override,
                    )
                    status = "OK" if result.get("ok") else "FAIL"
                    reason = result.get("error_class") or ""
                    reason_part = f" reason={reason}" if status == "FAIL" and reason else ""
                    base = (
                        "LLM ping: selector={selector} provider={provider} model={model} status={status} duration_ms={duration_ms}{reason}".format(
                            selector=selector_display,
                            provider=result.get("provider") or "none",
                            model=result.get("model") or "none",
                            status=status,
                            duration_ms=result.get("duration_ms") or 0,
                            reason=reason_part,
                        )
                    )
                    if selector == "broker":
                        base += " broker_choice={choice} broker_reason={reason}".format(
                            choice=broker_info.get("broker_choice", "none"),
                            reason=broker_info.get("broker_reason", "unknown"),
                        )
                    return OrchestratorResponse(base)

                if cmd.name == "observe_now":
                    self._record_conversation_topic(user_id, "observe_now", "command")
                    return self._call_skill(
                        user_id,
                        "observe_now",
                        "observe_now",
                        {},
                        ["db:write", "sys:read"],
                        action_type="insert",
                    )

                if cmd.name == "brief":
                    self._record_conversation_topic(user_id, "brief", "command")
                    return self._run_brief(user_id, text)

                if cmd.name == "what_if":
                    self._record_conversation_topic(user_id, "what_if", "question")
                    question = (cmd.args or "").strip()
                    if not question:
                        return OrchestratorResponse("Usage: /what_if <free text>")
                    return self._call_skill(
                        user_id,
                        "what_if",
                        "what_if",
                        {"text": question},
                        [],
                    )

                if cmd.name == "compare_now":
                    self._record_conversation_topic(user_id, "compare_now", "question")
                    question = (cmd.args or "").strip()
                    if not question:
                        pending = self._get_pending_compare(user_id)
                        if not pending or not pending.get("what_if_text"):
                            return OrchestratorResponse("Usage: /compare_now <what-if text>")
                        question = pending.get("what_if_text") or ""
                    return OrchestratorResponse(compare_now_to_what_if(question))

                if cmd.name == "disk_changes":
                    report = self._disk_changes_report(user_id)
                    text_out = self._maybe_add_narration("disk_changes", report["payload"], report["text"])
                    return OrchestratorResponse(text_out, report["payload"])

                if cmd.name == "disk_baseline":
                    return OrchestratorResponse(self._disk_baseline(user_id))

                if cmd.name == "disk_grow":
                    path = cmd.args.strip() if cmd.args else ""
                    return OrchestratorResponse(self._disk_grow(user_id, path))

                if cmd.name == "disk_digest":
                    report = self._disk_digest_report(user_id)
                    text_out = self._maybe_add_narration("disk_digest", report["payload"], report["text"])
                    return OrchestratorResponse(text_out, report["payload"])

                if cmd.name == "chat":
                    prompt = (cmd.args or "").strip()
                    if not prompt:
                        return OrchestratorResponse("Usage: /chat <text>")
                    return self._conversation_response(user_id, prompt, ["chat"])

                if cmd.name == "open_loops":
                    return OrchestratorResponse(build_open_loops_report(self.db, user_id, self.timezone))

                if cmd.name == "ask":
                    question = (cmd.args or "").strip()
                    if not question:
                        return OrchestratorResponse("Usage: /ask <question>")
                    if self._ask_contains_advice(question):
                        refusal = (
                            "I can only provide factual recall from existing snapshots. "
                            "Please ask for observations, not advice or actions."
                        )
                        try:
                            self.db.audit_log_create(
                                user_id=user_id,
                                action_type="ask_query",
                                action_id="ask_query",
                                status="refused",
                                details={
                                    "command": "/ask",
                                    "question": question[:200],
                                    "reason": "advice_request",
                                },
                            )
                        except Exception:
                            return OrchestratorResponse(AUDIT_HARD_FAIL_MSG)
                        return OrchestratorResponse(refusal)

                    parsed = parse_timeframe(question, self.db, self.timezone)
                    if parsed.clarify:
                        question_text = (
                            "What timeframe should I use? (last 7 days, last 72 hours, or last week)"
                        )
                        options = ["last 7 days", "last 72 hours", "last week"]
                        try:
                            self.db.audit_log_create(
                                user_id=user_id,
                                action_type="ask_query",
                                action_id="ask_query",
                                status="clarification",
                                details={
                                    "command": "/ask",
                                    "question": question[:200],
                                    "clarification_required": True,
                                },
                            )
                        except Exception:
                            return OrchestratorResponse(AUDIT_HARD_FAIL_MSG)
                        self._store_pending_clarification(
                            user_id,
                            user_id,
                            "ask_query",
                            {"question": question},
                            question_text,
                            options,
                        )
                        return OrchestratorResponse(question_text)
                    if not parsed.ok:
                        return OrchestratorResponse("No snapshots found yet.")

                    timeframe = {
                        "label": parsed.label,
                        "start_date": parsed.start_date,
                        "end_date": parsed.end_date,
                        "start_ts": parsed.start_ts,
                        "end_ts": parsed.end_ts,
                        "user_id": user_id,
                        "clarification_required": False,
                    }
                    return self._call_skill(
                        user_id,
                        "recall",
                        "ask_query",
                        {"question": question, "timeframe": timeframe},
                        ["db:read"],
                    )

                if cmd.name == "ask_opinion":
                    question = (cmd.args or "").strip()
                    if not question:
                        return OrchestratorResponse("Usage: /ask_opinion <question>")
                    if self._ask_contains_advice(question):
                        refusal = (
                            "I can provide bounded opinions about historical data, but not advice or actions. "
                            "Please ask for observations or opinions only."
                        )
                        try:
                            self.db.audit_log_create(
                                user_id=user_id,
                                action_type="ask_opinion",
                                action_id="ask_opinion",
                                status="refused",
                                details={
                                    "command": "/ask_opinion",
                                    "question": question[:200],
                                    "reason": "advice_request",
                                },
                            )
                        except Exception:
                            return OrchestratorResponse(AUDIT_HARD_FAIL_MSG)
                        return OrchestratorResponse(refusal)

                    trigger = self._opinion_trigger(question)
                    if not trigger:
                        return OrchestratorResponse(
                            "This can be answered factually. Use /ask for factual recall."
                        )

                    parsed = parse_timeframe(question, self.db, self.timezone)
                    if parsed.clarify:
                        question_text = (
                            "What timeframe should I use? (last 7 days, last 72 hours, or last week)"
                        )
                        options = ["last 7 days", "last 72 hours", "last week"]
                        try:
                            self.db.audit_log_create(
                                user_id=user_id,
                                action_type="ask_opinion",
                                action_id="ask_opinion",
                                status="clarification",
                                details={
                                    "command": "/ask_opinion",
                                    "question": question[:200],
                                    "clarification_required": True,
                                },
                            )
                        except Exception:
                            return OrchestratorResponse(AUDIT_HARD_FAIL_MSG)
                        self._store_pending_clarification(
                            user_id,
                            user_id,
                            "ask_opinion",
                            {"question": question, "trigger": trigger},
                            question_text,
                            options,
                        )
                        return OrchestratorResponse(question_text)
                    if not parsed.ok:
                        return OrchestratorResponse("No snapshots found yet.")

                    timeframe = {
                        "label": parsed.label,
                        "start_date": parsed.start_date,
                        "end_date": parsed.end_date,
                        "start_ts": parsed.start_ts,
                        "end_ts": parsed.end_ts,
                        "user_id": user_id,
                        "clarification_required": False,
                    }
                    return self._call_skill(
                        user_id,
                        "opinion",
                        "ask_opinion",
                        {"question": question, "timeframe": timeframe, "trigger": trigger},
                        ["db:read"],
                    )

                if cmd.name == "storage_snapshot":
                    return self._call_skill(
                        user_id,
                        "storage_governor",
                        "storage_snapshot",
                        {"user_id": user_id},
                        ["db:write", "sys:read"],
                        action_type="insert",
                    )

                if cmd.name == "storage_report":
                    response = self._call_skill(
                        user_id,
                        "storage_governor",
                        "storage_report",
                        {"user_id": user_id},
                        ["db:read"],
                    )
                    kind = "storage_report"
                    if _ui_mode() == "conversational":
                        response.text = self._present_readonly_report(
                            user_id=user_id,
                            kind=kind,
                            raw_text=response.text,
                            result_dict=response.data if isinstance(response.data, dict) else None,
                            question=text,
                        )
                    return response

                if cmd.name == "resource_report":
                    response = self._call_skill(
                        user_id,
                        "resource_governor",
                        "resource_report",
                        {"user_id": user_id},
                        ["sys:read"],
                    )
                    kind = "resource_report"
                    if _ui_mode() == "conversational":
                        response.text = self._present_readonly_report(
                            user_id=user_id,
                            kind=kind,
                            raw_text=response.text,
                            result_dict=response.data if isinstance(response.data, dict) else None,
                            question=text,
                        )
                    return response

                if cmd.name == "hardware_report":
                    response = self._call_skill(
                        user_id,
                        "hardware_report",
                        "hardware_report",
                        {"user_id": user_id},
                        ["sys:read"],
                    )
                    kind = "hardware_report"
                    if _ui_mode() == "conversational":
                        response.text = self._present_readonly_report(
                            user_id=user_id,
                            kind=kind,
                            raw_text=response.text,
                            result_dict=response.data if isinstance(response.data, dict) else None,
                            question=text,
                        )
                    return response

                if cmd.name == "network_report":
                    response = self._call_skill(
                        user_id,
                        "network_governor",
                        "network_report",
                        {"user_id": user_id},
                        ["db:read"],
                    )
                    kind = "network_report"
                    if _ui_mode() == "conversational":
                        response.text = self._present_readonly_report(
                            user_id=user_id,
                            kind=kind,
                            raw_text=response.text,
                            result_dict=response.data if isinstance(response.data, dict) else None,
                            question=text,
                        )
                    return response

                if cmd.name == "weekly_reflection":
                    return self._call_skill(
                        user_id,
                        "reflection",
                        "weekly_reflection",
                        {"user_id": user_id},
                        ["db:read"],
                    )

            reply_action, pending = opinion_gate.handle_reply(self.db, user_id, text, self.log_path)
            if reply_action == "expired":
                return OrchestratorResponse("Opinion request expired.")
            if reply_action == "declined":
                return OrchestratorResponse("Okay — sticking to facts.")
            if reply_action == "accepted" and pending:
                return self._deliver_pending_opinion(user_id, pending)

            if override:
                text = cleaned_text
            opinion_requested = opinion_gate.is_opinion_request(text)

            gate_result = handle_action_text(self.db, user_id, text, self.enable_writes)
            if gate_result:
                return OrchestratorResponse(gate_result.get("message", ""))

            if self.llm_client and getattr(self.llm_client, "enabled", lambda: False)():
                intent = self.llm_client.intent_from_text(text)
                if intent:
                    return OrchestratorResponse("LLM intent parsing not wired yet.")

            # Rule-based intent routing (v0.1)
            decision = route_message(user_id, text, self._intent_context())
            try:
                topic_tags = self._topic_tags_for_decision(decision)
                memory_ingest.ingest_event(
                    self.db,
                    user_id,
                    "user",
                    text,
                    topic_tags,
                    override=override,
                )
            except Exception:
                pass
            try:
                # Audit-friendly and safe: log decision metadata but not raw user text.
                snippet = redact_for_llm(text)
                if len(snippet) > 120:
                    snippet = snippet[:120].rstrip()
                log_event(
                    self.log_path,
                    "router_decision",
                    {
                        "user_id": user_id,
                        "decision_type": decision.get("type"),
                        "skill": decision.get("skill"),
                        "function": decision.get("function"),
                        "confidence": decision.get("confidence"),
                        "reason": decision.get("reason"),
                        "text_snippet": snippet,
                    },
                )
            except Exception:
                pass

            if decision.get("type") == "disk_changes":
                report = self._disk_changes_report(user_id)
                text_out = self._maybe_add_narration("disk_changes", report["payload"], report["text"])
                return OrchestratorResponse(text_out, report["payload"])

            if decision.get("type") == "disk_baseline":
                return OrchestratorResponse(self._disk_baseline(user_id))

            if decision.get("type") == "disk_grow":
                return OrchestratorResponse(self._disk_grow(user_id, decision.get("path", "")))

            if decision.get("type") == "brief":
                return self._run_brief(user_id, text)

            if decision.get("type") == "system_delta":
                return self._run_brief(user_id, text)

            if decision.get("type") == "system_trend":
                facts_uid = self._facts_user_id_for_request(user_id)
                report = build_trend_report_from_system_facts(
                    self.db,
                    user_id=facts_uid,
                    limit=int(os.getenv("SYSTEM_TREND_LIMIT", "5") or 5),
                    question=text,
                )
                raw = "\n".join(
                    [
                        "[SystemTrend]",
                        report.raw_text or "",
                    ]
                ).strip()
                cache_key = self._cache_last_report(user_id, kind="system_trend", raw_text=raw)
                self._register_last_report(
                    user_id=user_id,
                    kind_hint="trend",
                    details_cache_key=cache_key,
                    details_text=raw,
                )
                return OrchestratorResponse(report.summary_text)

            if decision.get("type") == "system_health_summary":
                return self._run_health_summary(user_id, text)

            if decision.get("type") == "doctor":
                report = run_doctor(
                    self.db,
                    user_id=user_id,
                    now_iso=datetime.now(timezone.utc).isoformat(),
                    env_path=(os.getenv("AGENT_ENV_PATH", "/etc/personal-agent/agent.env") or "/etc/personal-agent/agent.env"),
                    token_path=(os.getenv("SETTINGS_UI_TOKEN_PATH", "/etc/personal-agent/ui.token") or "/etc/personal-agent/ui.token"),
                    pid_path=(os.getenv("SETTINGS_UI_PID_PATH", "/run/personal-agent/settings_ui.pid") or "/run/personal-agent/settings_ui.pid"),
                    log_path=self.log_path,
                )
                text_out = "\n".join(report.lines).strip()
                if text_out:
                    text_out += "\n\nWant details? Say \"show details\"."
                self._cache_last_report(user_id, kind="doctor", raw_text=report.details_text)
                return OrchestratorResponse(text_out, {"severity": report.severity})

            if decision.get("type") == "system_opinion":
                facts_uid = self._facts_user_id_for_request(user_id)
                report = build_system_opinion(
                    self.db,
                    facts_uid,
                    text,
                    limit=int(os.getenv("SYSTEM_OPINION_LIMIT", "5") or 5),
                )
                summary = "\n".join([f"- {b}" for b in (report.bullets or []) if b]).strip()
                if not summary:
                    summary = "- Overall (ok): no opinion available."
                summary = summary + "\n\nWant the full report? Say \"show details\"."
                raw = (report.details_text or "").strip()
                raw = self._append_provider_status(user_id, raw)
                cache_key = self._cache_last_report(user_id, kind="system_opinion", raw_text=raw)
                self._register_last_report(
                    user_id=user_id,
                    kind_hint="opinion",
                    details_cache_key=cache_key,
                    details_text=raw,
                )
                return OrchestratorResponse(summary)

            if decision.get("type") == "explain_last_report":
                return self._explain_last_report(user_id, text)

            if decision.get("type") == "advise_from_last_report":
                return self._advise_from_last_report(user_id, text)

            if decision.get("type") == "action_proposal":
                message = propose_action(
                    self.db,
                    user_id,
                    decision.get("action_type", ""),
                    decision.get("action_id", ""),
                    decision.get("details", {}) or {},
                )
                return OrchestratorResponse(message)

            if decision.get("type") == "respond":
                return OrchestratorResponse(decision.get("text", ""))

            if decision.get("type") == "command_alias":
                command = decision.get("command") or ""
                if command:
                    return self.handle_message(command, user_id)
                return OrchestratorResponse("Say 'help' to see what I can do.")

            if decision.get("type") == "compare_now_pending":
                pending = self._get_pending_compare(user_id)
                if not pending:
                    return OrchestratorResponse("No what-if scenario to compare. Ask a what-if question first.")
                text = pending.get("what_if_text") or ""
                if not text:
                    return OrchestratorResponse("No what-if scenario to compare. Ask a what-if question first.")
                return OrchestratorResponse(compare_now_to_what_if(text))

            if decision.get("type") == "resource_followup":
                question = decision.get("question") or ""
                response_text = resource_followup(self.db, user_id, _resource_followup_kind(question), self.timezone)
                return OrchestratorResponse(response_text)

            if decision.get("type") == "skill_call":
                if decision.get("skill") == "opinion_on_report" and not pending:
                    return OrchestratorResponse(
                        "I can share an opinion after a report. Ask for a report first, then opt in."
                    )
                if decision.get("skill") == "opinion_on_report":
                    facts = (decision.get("args") or {}).get("facts")
                    if isinstance(facts, dict):
                        log_event(
                            self.log_path,
                            "opinion_request_detected",
                            {"user_id": user_id, "facts_hash": facts_hash(facts)},
                        )
                skill_name = decision.get("skill", "core")
                default_scopes = ["db:read"] if skill_name == "core" else []
                response = self._call_skill(
                    user_id,
                    skill_name,
                    decision["function"],
                    decision.get("args", {}),
                    decision.get("scopes", default_scopes),
                    action_type="auto",
                )
                kind = self._report_kind(skill_name, decision.get("function", ""))
                if kind and _ui_mode() == "conversational":
                    response.text = self._present_readonly_report(
                        user_id=user_id,
                        kind=kind,
                        raw_text=response.text,
                        result_dict=response.data if isinstance(response.data, dict) else None,
                        question=text,
                    )
                if skill_name == "diagnostics":
                    topic = (decision.get("args") or {}).get("topic")
                    self._record_conversation_topic(user_id, topic, "question")
                if skill_name == "reflection" and decision.get("function") == "weekly_reflection":
                    self._record_conversation_topic(user_id, "weekly_reflection", "question")
                if skill_name == "what_if":
                    self._record_conversation_topic(user_id, "what_if", "question")
                    what_if_text = (decision.get("args") or {}).get("text") or ""
                    if what_if_text:
                        self._store_pending_compare(user_id, what_if_text)
                if opinion_requested and skill_name != "knowledge_query" and isinstance(response.data, dict):
                    facts, context_note = self._extract_opinion_facts(
                        skill_name, decision.get("function", ""), response.data
                    )
                    if isinstance(facts, dict) and facts:
                        opinion_gate.store_pending(
                            self.db,
                            user_id,
                            topic_key=decision.get("function", "") or skill_name,
                            context={
                                "facts": facts,
                                "facts_text": response.text,
                                "context_note": context_note,
                            },
                            log_path=self.log_path,
                        )
                        response.text = f"{response.text}\n---\n{opinion_gate.OPINION_GATE_PROMPT}"
                if decision.get("skill") == "opinion_on_report":
                    facts = (decision.get("args") or {}).get("facts")
                    if isinstance(facts, dict):
                        log_event(
                            self.log_path,
                            "opinion_on_report_invoked",
                            {"user_id": user_id, "facts_hash": facts_hash(facts)},
                        )
                return response

            if decision.get("type") == "clarification_request":
                return OrchestratorResponse(decision["prompt"])

            if decision.get("type") == "noop" and not cmd:
                if self._looks_like_slow_check(text):
                    if _ui_mode() == "conversational":
                        return self._run_slow_diagnosis(user_id, text)
                    return OrchestratorResponse(self._help_text(user_id))
                if self._looks_like_system_check(text):
                    if _ui_mode() == "conversational":
                        return self._run_health_check(user_id, text)
                    return OrchestratorResponse(self._help_text(user_id))
                topic_tags = self._topic_tags_for_decision(decision)
                return self._conversation_response(user_id, text, topic_tags)

            return OrchestratorResponse("Say 'help' to see what I can do.")
        finally:
            self._runner = None

    def _disk_changes(self, user_id: str) -> str:
        report = self._disk_changes_report(user_id)
        return report["text"]

    def _disk_changes_report(self, user_id: str) -> dict[str, Any]:
        baseline = self.db.disk_baseline_get(user_id)
        entries = self.db.activity_log_list_recent("disk_report", limit=2)
        if not entries:
            text = "No disk reports found. Run /disk_report first."
            return {"text": text, "payload": {"message": text}}
        new_entry = entries[0]
        new_snapshot = (new_entry.get("payload") or {}).get("snapshot", {})

        label = ""
        old_snapshot = None
        label_kind = ""
        if baseline:
            old_snapshot = baseline.get("snapshot")
            label = "Since baseline (set {}):".format(time_since(baseline.get("created_at")))
            label_kind = "baseline"
        elif len(entries) >= 2:
            old_entry = entries[1]
            old_snapshot = (old_entry.get("payload") or {}).get("snapshot", {})
            label = "Since last report ({}):".format(time_since(old_entry.get("ts")))
            label_kind = "last_report"
        else:
            text = "Not enough disk reports to compare yet. Run /disk_report again later."
            return {"text": text, "payload": {"message": text}}

        diff = diff_disk_reports(old_snapshot or {}, new_snapshot)
        lines: list[str] = []
        if not diff.get("has_changes"):
            lines.append("No significant disk changes {}.".format(label.replace(":", "").lower()))
        else:
            lines.append(label)
            for line in diff.get("grew", []):
                lines.append(f"- {line}")
            for line in diff.get("shrank", []):
                lines.append(f"- {line}")
            top_growth = diff.get("top_growth", [])
            if top_growth:
                lines.append("Top growth:")
                for line in top_growth:
                    lines.append(f"- {line}")
        anomalies = detect_anomalies(old_snapshot or {}, new_snapshot)
        if anomalies:
            observed_at = new_entry.get("ts") or datetime.now(timezone.utc).isoformat()
            self._persist_anomalies(user_id, observed_at, anomalies)
        if anomalies:
            lines.append("Notable changes:")
            for flag in anomalies:
                lines.append(f"- {flag}")
        payload = {
            "label": label,
            "label_kind": label_kind,
            "has_changes": bool(diff.get("has_changes")),
            "grew": diff.get("grew", []),
            "shrank": diff.get("shrank", []),
            "top_growth": diff.get("top_growth", []),
            "anomalies": anomalies,
        }
        return {"text": "\n".join(lines), "payload": payload}

    def _disk_baseline(self, user_id: str) -> str:
        entries = self.db.activity_log_list_recent("disk_report", limit=1)
        if not entries:
            return "No disk reports found. Run /disk_report first."
        latest = entries[0]
        snapshot = (latest.get("payload") or {}).get("snapshot", {})
        snapshot_hash = (latest.get("payload") or {}).get("snapshot_hash", "")
        if not snapshot:
            return "No disk reports found. Run /disk_report first."
        self.db.disk_baseline_set(user_id, snapshot, snapshot_hash)
        return "Disk baseline set."

    def _disk_grow(self, user_id: str, raw_path: str) -> str:
        entries = self.db.activity_log_list_recent("disk_report", limit=2)
        if not entries:
            return "No disk reports found. Run /disk_report first."
        latest = entries[0]
        latest_snapshot = (latest.get("payload") or {}).get("snapshot", {})
        home_path = latest_snapshot.get("home_path") or os.path.expanduser("~")
        target = resolve_allowed_path(raw_path, home_path)
        if not target:
            return "That path is not allowed. Use /, /var, or your home directory."

        current_map = _run_du(target)
        if not current_map:
            return "Unable to read disk usage for that path."

        baseline = self.db.disk_baseline_get(user_id)
        if baseline:
            label = "baseline (set {})".format(time_since(baseline.get("created_at")))
            return build_growth_report(baseline.get("snapshot") or {}, current_map, target, label)

        if len(entries) < 2:
            return "Not enough disk reports to compare yet. Run /disk_report again later."
        previous = entries[1]
        label = "last report ({})".format(time_since(previous.get("ts")))
        return build_growth_report((previous.get("payload") or {}).get("snapshot", {}), current_map, target, label)

    def _audit_runner_result(self, skill_name: str, user_id: str) -> bool:
        runner = self._runner
        if not runner or not hasattr(runner, "last_result"):
            return True
        result = runner.last_result
        if not result or result.mode not in {"sandbox", "live"}:
            return True
        payload = {
            "event_type": "runner_exec",
            "mode": result.mode,
            "skill": skill_name,
            "allowlist_tag": result.allowlist_tag,
            "cmd": result.would_run,
            "blocked_reason": result.blocked_reason,
            "returncode": result.returncode,
            "stdout": (result.stdout or "")[:2000],
            "stderr": (result.stderr or "")[:2000],
            "actor_id": user_id,
            "source": "runner",
        }
        try:
            self.db.audit_log_create(
                user_id=user_id,
                action_type="runner_exec",
                action_id=skill_name,
                status="executed" if result.ok else "failed",
                details=payload,
                error=result.error,
            )
        except Exception:
            return False
        return True

    def _disk_digest(self, user_id: str) -> str:
        report = self._disk_digest_report(user_id)
        return report["text"]

    def _disk_digest_report(self, user_id: str) -> dict[str, Any]:
        entries = self.db.activity_log_list_recent("disk_report", limit=10)
        scheduled = [
            entry for entry in entries if (entry.get("payload") or {}).get("source") == "scheduled"
        ]
        if len(scheduled) < 2:
            text = (
                "Not enough scheduled snapshots yet. Run /disk_report manually or set "
                "ENABLE_SCHEDULED_SNAPSHOTS=true and restart."
            )
            return {"text": text, "payload": {"message": text}}
        new_entry = scheduled[0]
        old_entry = scheduled[1]
        new_snapshot = (new_entry.get("payload") or {}).get("snapshot", {})
        old_snapshot = (old_entry.get("payload") or {}).get("snapshot", {})
        diff = diff_disk_reports(old_snapshot, new_snapshot)
        label = "last snapshot ({})".format(time_since(old_entry.get("ts")))
        lines: list[str] = []
        if not diff.get("has_changes"):
            lines.append("No significant disk changes since {}.".format(label))
        else:
            lines.append(f"Since {label}:")
            for line in diff.get("grew", []):
                lines.append(f"- {line}")
            for line in diff.get("shrank", []):
                lines.append(f"- {line}")
            top_growth = diff.get("top_growth", [])
            if top_growth:
                lines.append("Top growth:")
                for line in top_growth:
                    lines.append(f"- {line}")
        anomalies = detect_anomalies(old_snapshot, new_snapshot)
        if anomalies:
            observed_at = new_entry.get("ts") or datetime.now(timezone.utc).isoformat()
            self._persist_anomalies(user_id, observed_at, anomalies)
        if anomalies:
            lines.append("Notable changes:")
            for flag in anomalies:
                lines.append(f"- {flag}")
        payload = {
            "label": label,
            "has_changes": bool(diff.get("has_changes")),
            "grew": diff.get("grew", []),
            "shrank": diff.get("shrank", []),
            "top_growth": diff.get("top_growth", []),
            "anomalies": anomalies,
        }
        return {"text": "\n".join(lines), "payload": payload}

    def _persist_anomalies(self, user_id: str, observed_at: str, flags: list[str]) -> None:
        events = []
        for flag in flags:
            key = re.sub(r"[^a-z0-9]+", "_", (flag or "").lower()).strip("_") or "anomaly"
            events.append(
                {
                    "source": "disk_anomalies",
                    "anomaly_key": key,
                    "severity": "warn",
                    "message": flag,
                    "context": {"raw": flag},
                }
            )
        inserted = self.db.insert_anomaly_events(user_id, observed_at, events)
        log_event(
            self.log_path,
            "anomaly_events_persisted",
            {"user_id": user_id, "observed_at": observed_at, "count": inserted},
        )


def _resource_followup_kind(question: str) -> str:
    lowered = (question or "").lower()
    if "using the most memory" in lowered or "using memory" in lowered:
        return "top_memory"
    if "using cpu" in lowered:
        return "top_cpu"
    if "compare to last time" in lowered:
        return "compare"
    if "what changed since last snapshot" in lowered:
        return "changed"
    if "is this normal" in lowered:
        return "is_normal"
    if "ram high" in lowered:
        return "ram_high"
    return "ram_high"
