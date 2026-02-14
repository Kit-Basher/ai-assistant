from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
import os
import json
import uuid
import subprocess
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
from agent.logging_utils import log_event, redact_payload
from agent.knowledge_cache import KnowledgeQueryCache, facts_hash
from agent.conversation_memory import record_event
import agent.memory_ingest as memory_ingest
from agent.compare_mode import compare_now_to_what_if
from agent.report_followups import resource_followup
from agent.changed_report import build_changed_report_from_system_facts
from agent.policy import evaluate_policy
import agent.opinion_gate as opinion_gate
from agent.skills_loader import SkillLoader
from agent.ask_timeframe import parse_timeframe
from agent.cards import render_cards_markdown
from agent.nl_router import build_cards_payload, nl_route
from agent.nl_policy import can_run_nl_skill
from agent.friction import compute_next_action, compute_options, compute_plan, compute_summary
from agent.prefs import (
    ALLOWED_PREF_KEYS,
    get_pref_effective,
    get_pref_effective_with_source,
    list_prefs,
    reset_prefs,
    reset_thread_prefs,
    set_pref,
    set_thread_pref,
)
from agent.epistemics import (
    CandidateContract,
    Claim,
    ContextPack,
    EpistemicMonitor,
    MessageTurn,
    apply_epistemic_gate,
    build_plain_answer_candidate,
    build_epistemics_report,
    parse_candidate_json,
)
from memory.db import MemoryDB


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
_COMMAND_STARTERS = {
    "pytest",
    "python",
    "python3",
    "git",
    "npm",
    "pnpm",
    "yarn",
    "make",
    "cargo",
    "go",
    "uv",
    "pip",
    "poetry",
    "bash",
    "sh",
    "node",
    "npx",
    "ruff",
    "mypy",
}


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
        self._last_offer_topic: dict[str, str] = {}
        self._started_at = datetime.now(__import__("datetime").timezone.utc)
        self._epistemic_monitor = EpistemicMonitor(db)
        self._epistemic_history: dict[tuple[str, str], list[MessageTurn]] = {}
        self._epistemic_thread_state: dict[str, dict[str, str | None]] = {}

    @staticmethod
    def _next_action_enabled() -> bool:
        raw = (os.getenv("FRiction_NEXT_ACTION", "") or "").strip().lower()
        if not raw:
            raw = (os.getenv("FRICTION_NEXT_ACTION", "") or "").strip().lower()
        if not raw:
            return True
        return raw not in {"0", "false", "off", "no"}

    @staticmethod
    def _summary_enabled() -> bool:
        raw = (os.getenv("FRICTION_SUMMARY", "") or "").strip().lower()
        if not raw:
            return True
        return raw not in {"0", "false", "off", "no"}

    @staticmethod
    def _plan_enabled() -> bool:
        raw = (os.getenv("FRICTION_PLAN", "") or "").strip().lower()
        if not raw:
            return True
        return raw not in {"0", "false", "off", "no"}

    @staticmethod
    def _options_enabled() -> bool:
        raw = (os.getenv("FRICTION_OPTIONS", "") or "").strip().lower()
        if not raw:
            return True
        return raw not in {"0", "false", "off", "no"}

    @staticmethod
    def _normalize_friction_text(text: str | None) -> str:
        if not text:
            return ""
        value = " ".join(text.lower().replace("\n", " ").split())
        return re.sub(r"[^a-z0-9 ./_\\-]+", " ", value).strip()

    @staticmethod
    def _is_command_line(line: str) -> bool:
        stripped = (line or "").strip()
        if not stripped or stripped.startswith("- ") or stripped.startswith("In short:") or stripped.startswith("Next:"):
            return False
        command = stripped[2:].strip() if stripped.startswith("$ ") else stripped
        if not command:
            return False
        first = command.split(" ", 1)[0].strip().lower()
        if first in _COMMAND_STARTERS:
            return True
        if command.startswith("./") or command.startswith("/") or command.endswith(".sh") or command.endswith(".py"):
            return True
        return False

    def _apply_commands_in_codeblock_pref(self, text: str, enabled: bool) -> str:
        if not enabled:
            return text
        lines = text.splitlines()
        indices = [idx for idx, line in enumerate(lines) if self._is_command_line(line)]
        if len(indices) != 1:
            return text
        idx = indices[0]
        command = lines[idx].strip()
        if command.startswith("$ "):
            command = command[2:].strip()
        if not command:
            return text
        replacement = f"```bash\n{command}\n```"
        updated = list(lines)
        updated[idx] = replacement
        return "\n".join(updated)

    @staticmethod
    def _apply_terse_mode_pref(body: str, enabled: bool) -> str:
        if not enabled:
            return body
        paragraphs = [segment.strip() for segment in re.split(r"\n\s*\n", body or "") if segment.strip()]
        if len(paragraphs) <= 1:
            return body
        return paragraphs[0]

    def _active_thread_id_for_user(self, user_id: str) -> str:
        state = self._epistemic_thread_state.get(user_id) or {}
        value = state.get("active_thread_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return self._default_thread_id(user_id)

    def _formatting_prefs(self, thread_id: str | None) -> dict[str, bool]:
        return {
            "show_next_action": bool(get_pref_effective(self.db, thread_id, "show_next_action", True)),
            "show_summary": bool(get_pref_effective(self.db, thread_id, "show_summary", True)),
            "terse_mode": bool(get_pref_effective(self.db, thread_id, "terse_mode", False)),
            "commands_in_codeblock": bool(get_pref_effective(self.db, thread_id, "commands_in_codeblock", False)),
        }

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
        result = router.summarize(kind, payload)
        if not result or not result.text:
            return text
        provider = result.provider or "unknown"
        scope = "local" if provider == "ollama" else "cloud" if provider == "openai" else provider
        header = f"Narration ({scope})"
        return f"{header}\n{result.text}\n\n{text}"

    def _maybe_add_narration_from_text(self, kind: str, text: str) -> str:
        router = LLMNarrationRouter()
        result = router.summarize(kind, {"report_text": text})
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

    def _format_opinion_reply(self, facts_text: str, opinion_text: str) -> str:
        return f"Facts:\n{facts_text}\n\nOpinion (opt-in):\n{opinion_text}"

    def _default_thread_id(self, user_id: str) -> str:
        return f"user:{user_id}"

    def _resolve_epistemic_thread(self, user_id: str, response: OrchestratorResponse) -> tuple[str, str, str | None]:
        data = response.data if isinstance(response.data, dict) else {}
        explicit_thread_id = None
        if isinstance(data.get("active_thread_id"), str) and data.get("active_thread_id", "").strip():
            explicit_thread_id = data.get("active_thread_id", "").strip()
        elif isinstance(data.get("thread_id"), str) and data.get("thread_id", "").strip():
            explicit_thread_id = data.get("thread_id", "").strip()

        explicit_label = data.get("thread_label")
        next_label = explicit_label.strip() if isinstance(explicit_label, str) and explicit_label.strip() else None

        prev = self._epistemic_thread_state.get(user_id) or {}
        active_thread_id = (
            explicit_thread_id
            or (prev.get("active_thread_id").strip() if isinstance(prev.get("active_thread_id"), str) else None)
            or self._default_thread_id(user_id)
        )

        now_iso = datetime.now(timezone.utc).isoformat()
        prev_thread_id = prev.get("active_thread_id")
        if prev_thread_id != active_thread_id:
            created_at = now_iso
        else:
            prev_created_at = prev.get("thread_created_at")
            created_at = prev_created_at if isinstance(prev_created_at, str) and prev_created_at else now_iso
            if next_label is None:
                prev_label = prev.get("thread_label")
                next_label = prev_label if isinstance(prev_label, str) and prev_label else None

        self._epistemic_thread_state[user_id] = {
            "active_thread_id": active_thread_id,
            "thread_created_at": created_at,
            "thread_label": next_label,
        }
        return active_thread_id, created_at, next_label

    def _epistemic_recent_messages(self, user_id: str, thread_id: str, limit: int = 8) -> tuple[MessageTurn, ...]:
        turns = self._epistemic_history.get((user_id, thread_id), [])
        return tuple(turns[-max(1, int(limit)):])

    def _next_turn_id(self, user_id: str, thread_id: str, role: str) -> str:
        turns = self._epistemic_history.get((user_id, thread_id), [])
        seq = len(turns) + 1
        role_tag = "u" if role == "user" else "a"
        return f"{thread_id}:{role_tag}:{seq}"

    def _epistemic_append_turn(
        self,
        user_id: str,
        thread_id: str,
        role: str,
        text: str,
        turn_id: str | None = None,
    ) -> None:
        cleaned = " ".join((text or "").split())
        if not cleaned or role not in {"user", "assistant"}:
            return
        resolved_turn_id = turn_id.strip() if isinstance(turn_id, str) and turn_id.strip() else self._next_turn_id(
            user_id, thread_id, role
        )
        turns = self._epistemic_history.setdefault((user_id, thread_id), [])
        turns.append(MessageTurn(role=role, text=cleaned, turn_id=resolved_turn_id))
        if len(turns) > 12:
            del turns[:-12]
        try:
            self.db.log_activity(
                "epistemic_turn",
                {
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "turn_id": resolved_turn_id,
                    "role": role,
                    "text": cleaned,
                },
            )
        except Exception:
            pass

    def _epistemic_extract_referents(self, user_id: str, thread_id: str) -> tuple[str, ...]:
        refs: list[str] = []
        seen: set[str] = set()
        for turn in self._epistemic_recent_messages(user_id, thread_id, limit=8):
            if turn.role != "assistant":
                continue
            for match in re.finditer(r"\[(\d+)\]\s+([^|\n]+)", turn.text):
                candidate = f"[{match.group(1)}] {match.group(2).strip()}"
                if candidate not in seen:
                    seen.add(candidate)
                    refs.append(candidate)
            for line in turn.text.splitlines():
                if not line.strip().startswith("- "):
                    continue
                candidate = line.strip()[2:].strip()
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    refs.append(candidate)
        return tuple(refs[:5])

    def _epistemic_tool_failures(self, response: OrchestratorResponse) -> tuple[str, ...]:
        lower = (response.text or "").strip().lower()
        signals: list[str] = []
        patterns = (
            "runner_not_configured",
            "permission denied",
            "skill not found",
            "function not found",
            "error:",
            "traceback",
        )
        for pattern in patterns:
            if pattern in lower:
                signals.append(pattern)
        if isinstance(response.data, dict):
            error = response.data.get("error")
            if isinstance(error, str) and error.strip():
                signals.append(error.strip().lower())
        return tuple(sorted(set(signals)))

    def _epistemic_tool_event_ids(self, response: OrchestratorResponse) -> tuple[str, ...]:
        ids: set[str] = set()
        containers: list[dict[str, Any]] = []
        if isinstance(response.data, dict):
            containers.append(response.data)
            nested = response.data.get("data")
            if isinstance(nested, dict):
                containers.append(nested)
        for container in containers:
            for key in ("tool_event_id", "audit_ref", "audit_id", "action_id"):
                value = container.get(key)
                if isinstance(value, (str, int)) and not isinstance(value, bool):
                    normalized = str(value).strip()
                    if normalized:
                        ids.add(normalized)
            audit = container.get("audit")
            if isinstance(audit, dict):
                for key in ("action_id", "audit_id"):
                    value = audit.get(key)
                    if isinstance(value, (str, int)) and not isinstance(value, bool):
                        normalized = str(value).strip()
                        if normalized:
                            ids.add(normalized)
        return tuple(sorted(ids))

    def _epistemic_memory_signals(
        self,
        response: OrchestratorResponse,
        active_thread_id: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...], bool, tuple[str, ...], tuple[str, ...], bool, tuple[str, ...]]:
        text = (response.text or "").strip()
        lower = text.lower()
        hits: list[str] = []
        ambiguous: list[str] = []
        miss = False
        in_scope: list[str] = []
        in_scope_ids: list[str] = []
        out_of_scope: list[str] = []
        out_of_scope_relevant = False

        if "no snapshots found yet" in lower or "no facts available" in lower:
            miss = True
        if "what timeframe should i use" in lower:
            ambiguous.append("timeframe")
        if "clarification required" in lower:
            ambiguous.append("clarification_required")

        if isinstance(response.data, dict):
            data = response.data.get("data")
            if isinstance(data, dict) and isinstance(data.get("facts"), dict):
                hits.append("facts")
                in_scope.append("facts")
            if "baseline_created" in response.data:
                hits.append("system_facts")
                in_scope.append("system_facts")
            if response.data.get("clarification_required") is True:
                ambiguous.append("clarification_required")
            memory_items = response.data.get("memory_items")
            if isinstance(memory_items, list):
                for item in memory_items:
                    if not isinstance(item, dict):
                        continue
                    raw_id = item.get("id")
                    memory_id = (
                        str(raw_id).strip()
                        if isinstance(raw_id, (str, int)) and not isinstance(raw_id, bool) and str(raw_id).strip()
                        else None
                    )
                    raw_ref = item.get("ref")
                    if memory_id is None and (not isinstance(raw_ref, str) or not raw_ref.strip()):
                        continue
                    ref = raw_ref.strip() if isinstance(raw_ref, str) and raw_ref.strip() else memory_id
                    if ref is None:
                        continue
                    is_global = bool(item.get("global")) or str(item.get("scope", "")).strip().lower() == "global"
                    raw_thread = item.get("thread_id")
                    item_thread = raw_thread.strip() if isinstance(raw_thread, str) and raw_thread.strip() else None
                    relevant = item.get("relevant")
                    is_relevant = True if relevant is None else bool(relevant)
                    if is_global or item_thread == active_thread_id:
                        in_scope.append(ref)
                        in_scope_ids.append(memory_id or ref)
                    else:
                        out_of_scope.append(memory_id or ref)
                        if is_relevant:
                            out_of_scope_relevant = True

        return (
            tuple(sorted(set(hits))),
            tuple(sorted(set(ambiguous))),
            bool(miss),
            tuple(sorted(set(in_scope))),
            tuple(sorted(set(out_of_scope))),
            bool(out_of_scope_relevant),
            tuple(sorted(set(in_scope_ids))),
        )

    def _build_epistemic_context(self, user_id: str, response: OrchestratorResponse) -> ContextPack:
        active_thread_id, thread_created_at, thread_label = self._resolve_epistemic_thread(user_id, response)
        recent_messages = self._epistemic_recent_messages(user_id, active_thread_id, limit=8)
        hits, ambiguous, miss, in_scope, out_of_scope, out_of_scope_relevant, in_scope_ids = self._epistemic_memory_signals(
            response, active_thread_id
        )
        recent_turn_ids = tuple(
            turn_id
            for turn_id in (turn.turn_id for turn in recent_messages)
            if isinstance(turn_id, str) and turn_id.strip()
        )
        pending_user_turn_id = self._next_turn_id(user_id, active_thread_id, "user")
        if pending_user_turn_id not in recent_turn_ids:
            recent_turn_ids = tuple(list(recent_turn_ids) + [pending_user_turn_id])
        return ContextPack(
            user_id=user_id,
            active_thread_id=active_thread_id,
            thread_created_at=thread_created_at,
            thread_label=thread_label,
            recent_messages=recent_messages,
            recent_turn_ids=recent_turn_ids,
            memory_hits=hits,
            memory_ambiguous=ambiguous,
            memory_miss=miss,
            in_scope_memory=in_scope,
            in_scope_memory_ids=in_scope_ids,
            out_of_scope_memory=out_of_scope,
            out_of_scope_relevant_memory=out_of_scope_relevant,
            thread_turn_count=len(recent_messages),
            tools_available=tuple(sorted(self.skills.keys())),
            tool_event_ids=self._epistemic_tool_event_ids(response),
            tool_failures=self._epistemic_tool_failures(response),
            referents=self._epistemic_extract_referents(user_id, active_thread_id),
        )

    def _build_epistemic_candidate(self, response: OrchestratorResponse, ctx: ContextPack) -> CandidateContract | str:
        if isinstance(response.data, dict):
            candidate_json = response.data.get("epistemic_candidate_json")
            if isinstance(candidate_json, str) and candidate_json.strip():
                parsed, errors = parse_candidate_json(candidate_json.strip())
                if parsed is not None and not errors:
                    return self._populate_candidate_provenance(parsed, ctx)
                return candidate_json.strip()
        text = (response.text or "").strip()
        if text.startswith("{") and '"kind"' in text:
            parsed, errors = parse_candidate_json(text)
            if parsed is not None and not errors:
                return self._populate_candidate_provenance(parsed, ctx)
            return text
        return self._populate_candidate_provenance(build_plain_answer_candidate(response.text or ""), ctx)

    def _populate_candidate_provenance(self, candidate: CandidateContract, ctx: ContextPack) -> CandidateContract:
        if not candidate.claims:
            return candidate
        default_user_turn_id = ctx.recent_turn_ids[-1] if ctx.recent_turn_ids else None
        default_memory_id = ctx.in_scope_memory_ids[0] if len(ctx.in_scope_memory_ids) == 1 else None
        default_tool_event_id = ctx.tool_event_ids[0] if len(ctx.tool_event_ids) == 1 else None
        hydrated_claims: list[Claim] = []
        for claim in candidate.claims:
            if claim.support == "none":
                hydrated_claims.append(
                    Claim(
                        text=claim.text,
                        support="none",
                        ref=None,
                        user_turn_id=None,
                        memory_id=None,
                        tool_event_id=None,
                    )
                )
                continue
            if claim.support == "user":
                user_turn_id = claim.user_turn_id or default_user_turn_id
                if not user_turn_id:
                    hydrated_claims.append(
                        Claim(
                            text=claim.text,
                            support="none",
                            ref=None,
                            user_turn_id=None,
                            memory_id=None,
                            tool_event_id=None,
                        )
                    )
                    continue
                hydrated_claims.append(
                    Claim(
                        text=claim.text,
                        support="user",
                        ref=claim.ref,
                        user_turn_id=user_turn_id,
                        memory_id=None,
                        tool_event_id=None,
                    )
                )
                continue
            if claim.support == "memory":
                memory_id = claim.memory_id
                if memory_id is None and isinstance(claim.ref, str) and claim.ref.strip():
                    candidate_memory_id = claim.ref.strip()
                    if candidate_memory_id in set(ctx.in_scope_memory_ids):
                        memory_id = candidate_memory_id
                if memory_id is None:
                    memory_id = default_memory_id
                if memory_id is None:
                    hydrated_claims.append(
                        Claim(
                            text=claim.text,
                            support="none",
                            ref=None,
                            user_turn_id=None,
                            memory_id=None,
                            tool_event_id=None,
                        )
                    )
                    continue
                hydrated_claims.append(
                    Claim(
                        text=claim.text,
                        support="memory",
                        ref=claim.ref,
                        user_turn_id=None,
                        memory_id=memory_id,
                        tool_event_id=None,
                    )
                )
                continue
            if claim.support == "tool":
                tool_event_id = claim.tool_event_id or default_tool_event_id
                if not tool_event_id:
                    hydrated_claims.append(
                        Claim(
                            text=claim.text,
                            support="none",
                            ref=None,
                            user_turn_id=None,
                            memory_id=None,
                            tool_event_id=None,
                        )
                    )
                    continue
                hydrated_claims.append(
                    Claim(
                        text=claim.text,
                        support="tool",
                        ref=claim.ref,
                        user_turn_id=None,
                        memory_id=None,
                        tool_event_id=tool_event_id,
                    )
                )
                continue
            hydrated_claims.append(claim)
        return CandidateContract(
            kind=candidate.kind,
            final_answer=candidate.final_answer,
            clarifying_question=candidate.clarifying_question,
            claims=tuple(hydrated_claims),
            assumptions=candidate.assumptions,
            unresolved_refs=candidate.unresolved_refs,
            thread_refs=candidate.thread_refs,
            raw_json=candidate.raw_json,
        )

    def _apply_epistemic_layer(self, user_id: str, user_text: str, response: OrchestratorResponse) -> OrchestratorResponse:
        ctx = self._build_epistemic_context(user_id, response)
        candidate = self._build_epistemic_candidate(response, ctx)
        decision = apply_epistemic_gate(user_text, ctx, candidate)
        user_visible_text = decision.user_text
        skip_friction_formatting = bool(
            isinstance(response.data, dict) and response.data.get("skip_friction_formatting") is True
        )
        if not decision.intercepted and isinstance(candidate, CandidateContract) and not skip_friction_formatting:
            prefs = self._formatting_prefs(ctx.active_thread_id)
            show_summary = prefs["show_summary"] and self._summary_enabled()
            show_next = prefs["show_next_action"] and self._next_action_enabled()
            show_plan = self._plan_enabled()
            show_options = self._options_enabled()
            body_text = self._apply_commands_in_codeblock_pref(
                user_visible_text,
                prefs["commands_in_codeblock"],
            )
            summary_line = compute_summary(candidate, body_text) if show_summary else None
            body_text = self._apply_terse_mode_pref(body_text, prefs["terse_mode"])
            if prefs["terse_mode"] and summary_line:
                show_plan = False
            plan_steps = compute_plan(user_text, candidate, body_text) if show_plan else None
            if prefs["terse_mode"] and plan_steps:
                show_options = False
            next_action = compute_next_action(user_text, ctx, candidate) if show_next else None
            options = compute_options(user_text, candidate, body_text) if show_options else None
            if options:
                plan_norm = {
                    self._normalize_friction_text(step)
                    for step in (plan_steps or [])
                    if self._normalize_friction_text(step)
                }
                next_norm = self._normalize_friction_text(next_action)
                filtered_options: list[str] = []
                seen: set[str] = set()
                for option in options:
                    normalized = self._normalize_friction_text(option)
                    if not normalized:
                        continue
                    if normalized in seen:
                        continue
                    if normalized in plan_norm:
                        continue
                    if next_norm and (normalized == next_norm or normalized in next_norm or next_norm in normalized):
                        continue
                    seen.add(normalized)
                    filtered_options.append(option.replace("?", "").strip())
                    if len(filtered_options) >= 3:
                        break
                options = filtered_options if len(filtered_options) >= 2 else None
            parts: list[str] = []
            if summary_line:
                parts.append(summary_line)
            parts.append(body_text)
            if plan_steps:
                lines = [f"{idx}. {step}" for idx, step in enumerate(plan_steps, start=1)]
                parts.append("Plan:\n" + "\n".join(lines))
            if options:
                labels = ("A", "B", "C")
                option_lines = [f"{labels[idx]}) {option}" for idx, option in enumerate(options)]
                option_lines.append("Choose A, B, or C.")
                parts.append("Options:\n" + "\n".join(option_lines))
            if next_action:
                parts.append(f"Next: {next_action}")
            user_visible_text = "\n\n".join(part for part in parts if part and part.strip())
        try:
            self._epistemic_monitor.record(user_id, decision, active_thread_id=ctx.active_thread_id)
        except Exception:
            pass
        thread_id = ctx.active_thread_id or self._default_thread_id(user_id)
        user_turn_id = ctx.recent_turn_ids[-1] if ctx.recent_turn_ids else self._next_turn_id(user_id, thread_id, "user")
        self._epistemic_append_turn(user_id, thread_id, "user", user_text, turn_id=user_turn_id)
        assistant_turn_id = self._next_turn_id(user_id, thread_id, "assistant")
        self._epistemic_append_turn(user_id, thread_id, "assistant", user_visible_text, turn_id=assistant_turn_id)
        return OrchestratorResponse(user_visible_text, response.data)

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
        read_only_mode: bool = False,
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
        if read_only_mode:
            ctx["read_only_mode"] = True
        result = func.handler(ctx, **args)
        if read_only_mode:
            details = redact_payload(
                {
                    "mode": "read_only",
                    "skill": skill_name,
                    "function": function_name,
                    "args": args,
                }
            )
            try:
                self.db.audit_log_create(
                    user_id=user_id,
                    action_type="nl_read",
                    action_id=f"{skill_name}.{function_name}",
                    status="executed",
                    details=details,
                )
                self.db.log_activity(
                    "nl_read",
                    {
                        "user_id": user_id,
                        "skill": skill_name,
                        "function": function_name,
                        "mode": "read_only",
                    },
                )
            except Exception:
                return OrchestratorResponse(AUDIT_HARD_FAIL_MSG)
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
                opinion_gate.store_pending(self.db, user_id, "knowledge_query")
                response_text = (
                    f"{response_text}\n---\nWant my opinion?\n{opinion_gate.OPINION_GATE_PROMPT}"
                )
        if skill_name == "disk_report" and isinstance(result, dict):
            response_text = self._maybe_add_narration("disk_report", result, response_text)
        if not read_only_mode:
            try:
                memory_ingest.ingest_event(
                    self.db,
                    user_id,
                    "skill",
                    response_text,
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
        response = self._handle_message_impl(text, user_id)
        return self._apply_epistemic_layer(user_id, text, response)

    def _handle_message_impl(self, text: str, user_id: str) -> OrchestratorResponse:
        self._runner = Runner()
        try:
            override, cleaned_text = memory_ingest.parse_memory_override(text)
            cmd = parse_command(text)
            if cmd and cmd.name == "nomem":
                cmd = None
                text = cleaned_text
            if cmd:
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
                    raw_args = (cmd.args or "").strip()
                    if "|" in raw_args:
                        project, title, effort, impact = split_pipe_args(raw_args, 4)
                    else:
                        project, title, effort, impact = "", raw_args, "", ""
                    title = (title or "").strip()
                    if not title:
                        cards_payload = build_cards_payload(
                            [
                                {
                                    "key": "task-add-usage",
                                    "title": "Task add",
                                    "lines": ["Usage: /task_add <title>"],
                                    "severity": "warn",
                                }
                            ],
                            raw_available=False,
                            summary="Could not add task.",
                            confidence=1.0,
                            next_questions=["Try: /task_add Write report"],
                        )
                        return self._cards_response(user_id, cards_payload)
                    effort_clean = (effort or "").strip()
                    impact_clean = (impact or "").strip()
                    if effort_clean and not effort_clean.isdigit():
                        cards_payload = build_cards_payload(
                            [
                                {
                                    "key": "task-add-effort-invalid",
                                    "title": "Task add",
                                    "lines": ["Effort must be an integer number of minutes."],
                                    "severity": "warn",
                                }
                            ],
                            raw_available=False,
                            summary="Could not add task.",
                            confidence=1.0,
                            next_questions=["Try: /task_add Project|Title|30|4"],
                        )
                        return self._cards_response(user_id, cards_payload)
                    if impact_clean and not impact_clean.isdigit():
                        cards_payload = build_cards_payload(
                            [
                                {
                                    "key": "task-add-impact-invalid",
                                    "title": "Task add",
                                    "lines": ["Impact must be an integer from 1 to 5."],
                                    "severity": "warn",
                                }
                            ],
                            raw_available=False,
                            summary="Could not add task.",
                            confidence=1.0,
                            next_questions=["Try: /task_add Project|Title|30|4"],
                        )
                        return self._cards_response(user_id, cards_payload)
                    effort_mins = int(effort_clean) if effort_clean else None
                    impact_1to5 = int(impact_clean) if impact_clean else None
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
                    if not result.data:
                        return result
                    task_id = result.data.get("task_id") if isinstance(result.data, dict) else None
                    if isinstance(task_id, int):
                        return OrchestratorResponse(f"Task added: [{task_id}] {title}")
                    return OrchestratorResponse("Task added.")

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
                    arg = (cmd.args or "").strip()
                    if not re.fullmatch(r"\d+", arg):
                        cards_payload = build_cards_payload(
                            [
                                {
                                    "key": "task-done-usage",
                                    "title": "Task done",
                                    "lines": ["Usage: /done <id>"],
                                    "severity": "warn",
                                }
                            ],
                            raw_available=False,
                            summary="Could not mark task done.",
                            confidence=1.0,
                            next_questions=["Try: /done 1"],
                        )
                        return self._cards_response(user_id, cards_payload)
                    task_id = int(arg)
                    task = self.db.get_task(task_id)
                    if not task:
                        cards_payload = build_cards_payload(
                            [
                                {
                                    "key": "task-done-not-found",
                                    "title": "Task done",
                                    "lines": [f"Task not found: {task_id}"],
                                    "severity": "warn",
                                }
                            ],
                            raw_available=False,
                            summary="Could not mark task done.",
                            confidence=1.0,
                            next_questions=["Use /task_add <title> to add a task."],
                        )
                        return self._cards_response(user_id, cards_payload)
                    title = str(task.get("title") or "").strip()
                    status = str(task.get("status") or "").strip().lower()
                    if status == "done":
                        cards_payload = build_cards_payload(
                            [
                                {
                                    "key": "task-done-already",
                                    "title": "Task done",
                                    "lines": [f"Already done: [{task_id}] {title}"],
                                    "severity": "ok",
                                }
                            ],
                            raw_available=False,
                            summary="Task already completed.",
                            confidence=1.0,
                            next_questions=["Use /task_add <title> for a new task."],
                        )
                        return self._cards_response(user_id, cards_payload)
                    updated = self.db.mark_task_done(task_id)
                    if not updated:
                        cards_payload = build_cards_payload(
                            [
                                {
                                    "key": "task-done-not-found-race",
                                    "title": "Task done",
                                    "lines": [f"Task not found: {task_id}"],
                                    "severity": "warn",
                                }
                            ],
                            raw_available=False,
                            summary="Could not mark task done.",
                            confidence=1.0,
                            next_questions=["Try again with /done <id>."],
                        )
                        return self._cards_response(user_id, cards_payload)
                    cards_payload = build_cards_payload(
                        [
                            {
                                "key": "task-done",
                                "title": "Task done",
                                "lines": [f"Done: [{task_id}] {title}"],
                                "severity": "ok",
                            }
                        ],
                        raw_available=False,
                        summary="Task updated.",
                        confidence=1.0,
                        next_questions=["Use /today to review remaining tasks."],
                    )
                    return self._cards_response(user_id, cards_payload)

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

                if cmd.name == "prefs":
                    prefs = list_prefs(self.db)
                    lines = []
                    for key in ALLOWED_PREF_KEYS:
                        value = "on" if prefs.get(key, "off") == "on" else "off"
                        lines.append(f"{key}: {value}")
                    return OrchestratorResponse(
                        "\n".join(lines),
                        {"skip_friction_formatting": True, "thread_id": self._active_thread_id_for_user(user_id)},
                    )

                if cmd.name == "prefs_set":
                    parts = (cmd.args or "").strip().split()
                    if len(parts) != 2:
                        return OrchestratorResponse(
                            "Usage: /prefs_set <key> <on|off>",
                            {"skip_friction_formatting": True, "thread_id": self._active_thread_id_for_user(user_id)},
                        )
                    key = parts[0].strip()
                    value = parts[1].strip().lower()
                    if key not in ALLOWED_PREF_KEYS:
                        return OrchestratorResponse(
                            "Unknown preference key. Allowed: " + ", ".join(ALLOWED_PREF_KEYS),
                            {"skip_friction_formatting": True, "thread_id": self._active_thread_id_for_user(user_id)},
                        )
                    if value not in {"on", "off"}:
                        return OrchestratorResponse(
                            "Usage: /prefs_set <key> <on|off>",
                            {"skip_friction_formatting": True, "thread_id": self._active_thread_id_for_user(user_id)},
                        )
                    set_pref(self.db, key, value)
                    return OrchestratorResponse(
                        f"{key}: {value}",
                        {"skip_friction_formatting": True, "thread_id": self._active_thread_id_for_user(user_id)},
                    )

                if cmd.name == "prefs_reset":
                    reset_prefs(self.db)
                    return OrchestratorResponse(
                        "Preferences reset to defaults.",
                        {"skip_friction_formatting": True, "thread_id": self._active_thread_id_for_user(user_id)},
                    )

                if cmd.name == "prefs_thread":
                    thread_id = self._active_thread_id_for_user(user_id)
                    lines = []
                    for key in ALLOWED_PREF_KEYS:
                        value, source = get_pref_effective_with_source(self.db, thread_id, key)
                        lines.append(f"{key}: {value} (source: {source})")
                    return OrchestratorResponse(
                        "\n".join(lines),
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "prefs_thread_set":
                    thread_id = self._active_thread_id_for_user(user_id)
                    parts = (cmd.args or "").strip().split()
                    if len(parts) != 2:
                        return OrchestratorResponse(
                            "Usage: /prefs_thread_set <key> <on|off>",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    key = parts[0].strip()
                    value = parts[1].strip().lower()
                    if key not in ALLOWED_PREF_KEYS:
                        return OrchestratorResponse(
                            "Unknown preference key. Allowed: " + ", ".join(ALLOWED_PREF_KEYS),
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    if value not in {"on", "off"}:
                        return OrchestratorResponse(
                            "Usage: /prefs_thread_set <key> <on|off>",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    set_thread_pref(self.db, thread_id, key, value)
                    return OrchestratorResponse(
                        f"{key}: {value} (thread)",
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "prefs_thread_reset":
                    thread_id = self._active_thread_id_for_user(user_id)
                    reset_thread_prefs(self.db, thread_id)
                    return OrchestratorResponse(
                        "Thread preferences reset.",
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "epistemics_report":
                    return OrchestratorResponse(build_epistemics_report(self.db))

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
                    return self._call_skill(
                        user_id,
                        "runtime_status",
                        "runtime_status",
                        {},
                        ["db:read"],
                    )

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
                    return self._call_skill(
                        user_id,
                        "core",
                        "llm_status",
                        {},
                        [],
                    )

                if cmd.name == "llm_ping":
                    router = self.llm_client
                    if not router or not hasattr(router, "chat"):
                        return OrchestratorResponse(
                            "LLM ping: provider=none model=none status=FAIL reason=unavailable"
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
                    return OrchestratorResponse(
                        "LLM ping: provider={provider} model={model} status={status} duration_ms={duration_ms}{reason}".format(
                            provider=result.get("provider") or "none",
                            model=result.get("model") or "none",
                            status=status,
                            duration_ms=result.get("duration_ms") or 0,
                            reason=reason_part,
                        )
                    )

                if cmd.name == "observe_now":
                    self._record_conversation_topic(user_id, "observe_now", "command")
                    return self._call_skill(
                        user_id,
                        "observe_now",
                        "observe_now",
                        {},
                        [],
                    )

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

                if cmd.name == "brief":
                    return self._run_brief(user_id)

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
                    response.text = self._maybe_add_narration_from_text("storage_report", response.text)
                    return response

                if cmd.name == "resource_report":
                    response = self._call_skill(
                        user_id,
                        "resource_governor",
                        "resource_report",
                        {"user_id": user_id},
                        ["db:read"],
                    )
                    response.text = self._maybe_add_narration_from_text("resource_report", response.text)
                    return response

                if cmd.name == "network_report":
                    response = self._call_skill(
                        user_id,
                        "network_governor",
                        "network_report",
                        {"user_id": user_id},
                        ["db:read"],
                    )
                    response.text = self._maybe_add_narration_from_text("network_report", response.text)
                    return response

                if cmd.name == "weekly_reflection":
                    return self._call_skill(
                        user_id,
                        "reflection",
                        "weekly_reflection",
                        {"user_id": user_id},
                        ["db:read"],
                    )

                if cmd.name == "today":
                    cards_payload = self._today_cards_payload(cmd.args or "")
                    return self._cards_response(user_id, cards_payload)

                if cmd.name == "open_loops":
                    mode = (cmd.args or "").strip().lower()
                    if mode == "all":
                        return self._cards_response(user_id, self._open_loops_payload(status="all", order="due"))
                    if mode == "due":
                        return self._cards_response(user_id, self._open_loops_payload(status="open", order="due"))
                    if mode == "important":
                        return self._cards_response(user_id, self._open_loops_payload(status="open", order="important"))
                    return self._cards_response(user_id, self._open_loops_payload(status="open", order="created"))

                if cmd.name == "daily_brief_status":
                    return self._cards_response(user_id, self._daily_brief_status_payload(user_id))

                if cmd.name == "health":
                    return self._cards_response(user_id, self._health_payload(user_id))

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

            if not text.strip().startswith("/"):
                nl_decision = nl_route(text)
                nl_intent = nl_decision.get("intent")
                if nl_intent in {"OBSERVE_PC", "EXPLAIN_PREVIOUS"}:
                    return self._handle_nl_observe(user_id, text, nl_decision)
                if nl_intent == "MEMORY_WRITE_REQUEST":
                    memory_update = self._parse_memory_write(text)
                    changed: list[str] = []
                    if memory_update.get("response_style"):
                        self.db.set_preference("response_style", str(memory_update["response_style"]))
                        changed.append("response_style")
                    if memory_update.get("max_cards") is not None:
                        self.db.set_preference("max_cards", str(memory_update["max_cards"]))
                        changed.append("max_cards")
                    if memory_update.get("default_compare"):
                        self.db.set_preference("default_compare", str(memory_update["default_compare"]))
                        changed.append("default_compare")
                    if memory_update.get("default_compare_enabled"):
                        self.db.set_preference(
                            "default_compare_enabled", str(memory_update["default_compare_enabled"])
                        )
                        changed.append("default_compare_enabled")
                    if memory_update.get("show_confidence"):
                        self.db.set_preference("show_confidence", str(memory_update["show_confidence"]))
                        changed.append("show_confidence")
                    if memory_update.get("daily_brief_enabled"):
                        self.db.set_preference("daily_brief_enabled", str(memory_update["daily_brief_enabled"]))
                        changed.append("daily_brief_enabled")
                    if memory_update.get("daily_brief_time"):
                        self.db.set_preference("daily_brief_time", str(memory_update["daily_brief_time"]))
                        changed.append("daily_brief_time")
                    if memory_update.get("daily_brief_quiet_mode"):
                        self.db.set_preference(
                            "daily_brief_quiet_mode", str(memory_update["daily_brief_quiet_mode"])
                        )
                        changed.append("daily_brief_quiet_mode")
                    if memory_update.get("disk_delta_threshold_mb"):
                        self.db.set_preference(
                            "disk_delta_threshold_mb", str(memory_update["disk_delta_threshold_mb"])
                        )
                        changed.append("disk_delta_threshold_mb")
                    if memory_update.get("only_send_if_service_unhealthy"):
                        self.db.set_preference(
                            "only_send_if_service_unhealthy",
                            str(memory_update["only_send_if_service_unhealthy"]),
                        )
                        changed.append("only_send_if_service_unhealthy")
                    if memory_update.get("include_open_loops_due_within_days"):
                        self.db.set_preference(
                            "include_open_loops_due_within_days",
                            str(memory_update["include_open_loops_due_within_days"]),
                        )
                        changed.append("include_open_loops_due_within_days")
                    if memory_update.get("important_paths") is not None:
                        self.db.set_preference(
                            "important_paths",
                            json.dumps(memory_update["important_paths"], ensure_ascii=True),
                        )
                        changed.append("important_paths")
                    if memory_update.get("baseline_window"):
                        self.db.set_preference("baseline_window", str(memory_update["baseline_window"]))
                        changed.append("baseline_window")
                    if changed:
                        cards_payload = build_cards_payload(
                            [
                                {
                                    "key": "memory-updated",
                                    "title": "Saved preference anchors",
                                    "lines": [f"Updated: {', '.join(changed)}"],
                                    "severity": "ok",
                                }
                            ],
                            raw_available=False,
                            summary="Stored your explicit preference/anchor request.",
                            confidence=1.0,
                            next_questions=["Show my current preferences.", "Plan my day."],
                        )
                        return self._cards_response(user_id, cards_payload)
                    cards_payload = build_cards_payload(
                        [
                            {
                                "key": "memory-write-request",
                                "title": "Memory write request detected",
                                "lines": ["Write actions are not auto-run in free text mode."],
                                "severity": "warn",
                            }
                        ],
                        raw_available=False,
                        summary="Write requests are blocked in NL read-only mode.",
                        confidence=1.0,
                        next_questions=["Ask for disk/resource status instead."],
                    )
                    return self._cards_response(user_id, cards_payload)
                if nl_intent == "PLAN_DAY":
                    cards_payload = self._today_cards_payload(text)
                    return self._cards_response(user_id, cards_payload)
                if nl_intent == "SHOW_PREFERENCES":
                    cards_payload = self._preferences_cards_payload()
                    return self._cards_response(user_id, cards_payload)
                if nl_intent == "MEMORY_INSPECT":
                    cards_payload = self._memory_inspection_payload()
                    return self._cards_response(user_id, cards_payload)
                if nl_intent == "OPEN_LOOPS_LIST":
                    return self._cards_response(user_id, self._open_loops_payload(status="open", order="due"))
                if nl_intent == "DAILY_BRIEF_STATUS":
                    return self._cards_response(user_id, self._daily_brief_status_payload(user_id))
                if nl_intent == "OPEN_LOOP_ADD":
                    title, due, priority = self._parse_open_loop_add(text)
                    if title:
                        self.db.add_open_loop(title, due, priority=priority)
                        payload = build_cards_payload(
                            [
                                {
                                    "key": "open-loop-added",
                                    "title": "Open loop added",
                                    "lines": [f"{title} (due {due or 'unspecified'}, P{priority})"],
                                    "severity": "ok",
                                }
                            ],
                            raw_available=False,
                            summary="Stored your open loop.",
                            confidence=1.0,
                            next_questions=["Show open loops", "Mark one done"],
                        )
                        return self._cards_response(user_id, payload)
                if nl_intent == "OPEN_LOOP_DONE":
                    title_fragment = self._parse_open_loop_done(text)
                    if title_fragment:
                        count = self.db.complete_open_loop_by_title(title_fragment)
                        payload = build_cards_payload(
                            [
                                {
                                    "key": "open-loop-done",
                                    "title": "Open loop updated",
                                    "lines": [f"Marked done: {title_fragment}" if count else "No matching open loop found."],
                                    "severity": "ok" if count else "warn",
                                }
                            ],
                            raw_available=False,
                            summary="Updated open loop status." if count else "Could not find an open loop to mark done.",
                            confidence=1.0 if count else 0.8,
                            next_questions=["Show open loops", "Add another open loop"],
                        )
                        return self._cards_response(user_id, payload)
                if nl_intent == "CHITCHAT":
                    if re.match(r"^(hi|hello|hey)(\\b|[!.?]|$)", text.strip().lower()):
                        pass
                    else:
                        cards_payload = build_cards_payload(
                            [
                                {
                                    "key": "chitchat",
                                    "title": "Personal Agent",
                                    "lines": ["Ask about disk, CPU, memory, or process changes."],
                                    "severity": "ok",
                                }
                            ],
                            raw_available=False,
                            summary="Ready for a read-only system check.",
                            confidence=1.0,
                            next_questions=["What changed on my disk?", "How are CPU and memory right now?"],
                        )
                        return self._cards_response(user_id, cards_payload)

            gate_result = handle_action_text(self.db, user_id, text, self.enable_writes)
            if gate_result:
                return OrchestratorResponse(gate_result.get("message", ""))

            if self.llm_client and getattr(self.llm_client, "enabled", lambda: False)():
                intent = self.llm_client.intent_from_text(text)
                if intent:
                    return OrchestratorResponse("LLM intent parsing not wired yet.")

            # Rule-based intent routing (v0.1)
            intent_ctx = self._intent_context()
            intent_ctx["last_topic"] = self._last_offer_topic.get(user_id)
            decision = route_message(user_id, text, intent_ctx)
            if decision.get("type") != "greeting":
                self._last_offer_topic.pop(user_id, None)
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

            if decision.get("type") == "disk_changes":
                report = self._disk_changes_report(user_id)
                text_out = self._maybe_add_narration("disk_changes", report["payload"], report["text"])
                return OrchestratorResponse(text_out, report["payload"])

            if decision.get("type") == "disk_baseline":
                return OrchestratorResponse(self._disk_baseline(user_id))

            if decision.get("type") == "disk_grow":
                return OrchestratorResponse(self._disk_grow(user_id, decision.get("path", "")))

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

            if decision.get("type") == "brief":
                return self._run_brief(user_id)

            if decision.get("type") == "command_alias":
                command = decision.get("command") or ""
                if command:
                    return self._handle_message_impl(command, user_id)
                return OrchestratorResponse("Try /help")

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
                if decision.get("skill") == "opinion_on_report":
                    facts = (decision.get("args") or {}).get("facts")
                    if not isinstance(facts, dict) or not facts:
                        return OrchestratorResponse(
                            "I can share an opinion after a report. Ask for a report first, then opt in."
                        )
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
                            decision.get("function", "") or skill_name,
                        )
                        response.text = (
                            f"{response.text}\n---\nWant my opinion?\n{opinion_gate.OPINION_GATE_PROMPT}"
                        )
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

            if decision.get("type") == "greeting":
                self._last_offer_topic[user_id] = "brief_offer"
                return OrchestratorResponse('Hey 🙂 Want a quick /brief, or ask “anything new on my PC?”')

            return OrchestratorResponse("I didn’t understand that. Try /brief, or ask “anything new on my PC?”")
        finally:
            self._runner = None

    def _run_brief(self, user_id: str) -> OrchestratorResponse:
        # Fresh snapshot (observe-only; uses existing governors).
        self._call_skill(
            user_id,
            "observe_now",
            "observe_now",
            {"user_id": user_id},
            ["db:write", "sys:read"],
            action_type="insert",
        )

        report = build_changed_report_from_system_facts(self.db, user_id)
        if report.baseline_created:
            text = "Baseline created. I'll report changes next time."
            return OrchestratorResponse(text, {"baseline_created": True})

        lines: list[str] = []
        if report.machine_summary:
            lines.append(report.machine_summary)
        if report.delta_lines:
            lines.extend(report.delta_lines)
        else:
            lines.append("No notable changes since last snapshot.")

        # Keep it short and delta-focused.
        return OrchestratorResponse("\n".join(lines[:12]), {"baseline_created": False})

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

    def _parse_memory_write(self, text: str) -> dict[str, Any]:
        lowered = (text or "").strip().lower()
        raw = (text or "").strip()
        result = {
            "response_style": None,
            "max_cards": None,
            "default_compare": None,
            "default_compare_enabled": None,
            "show_confidence": None,
            "important_paths": None,
            "baseline_window": None,
            "daily_brief_enabled": None,
            "daily_brief_time": None,
            "daily_brief_quiet_mode": None,
            "disk_delta_threshold_mb": None,
            "only_send_if_service_unhealthy": None,
            "include_open_loops_due_within_days": None,
        }
        explicit_prefix = (
            lowered.startswith("remember that")
            or lowered.startswith("from now on")
            or lowered.startswith("remember this")
            or lowered.startswith("set max cards to")
            or lowered.startswith("turn confidence ")
            or lowered.startswith("default compare ")
            or lowered.startswith("daily brief ")
            or lowered.startswith("set disk delta threshold to")
            or lowered.startswith("only send if service unhealthy ")
            or lowered.startswith("set open loops due window to")
        )
        if not explicit_prefix:
            return result
        if "concise" in lowered:
            result["response_style"] = "concise"
        elif "detailed" in lowered:
            result["response_style"] = "detailed"
        m_cards = re.search(r"max cards\s*(?:is|=)?\s*(\d+)", lowered)
        if m_cards:
            result["max_cards"] = max(1, min(12, int(m_cards.group(1))))
        if "set max cards to" in lowered:
            m_set = re.search(r"set max cards to\s*(\d+)", lowered)
            if m_set:
                result["max_cards"] = max(1, min(12, int(m_set.group(1))))
        if "default compare" in lowered:
            if "last snapshot" in lowered:
                result["default_compare"] = "last_snapshot"
            elif "baseline" in lowered:
                result["default_compare"] = "baseline"
            if lowered.endswith(" on") or " on " in lowered:
                result["default_compare_enabled"] = "on"
            if lowered.endswith(" off") or " off " in lowered:
                result["default_compare_enabled"] = "off"
        if "turn confidence off" in lowered:
            result["show_confidence"] = "off"
        elif "turn confidence on" in lowered:
            result["show_confidence"] = "on"
        if lowered.startswith("daily brief "):
            if " off" in lowered or lowered.endswith("off"):
                result["daily_brief_enabled"] = "off"
            if " on" in lowered or lowered.endswith("on") or " at " in lowered:
                result["daily_brief_enabled"] = "on"
            match_time = re.search(r"\bat\s*([0-2]?\d:[0-5]\d)\b", lowered)
            if match_time:
                result["daily_brief_time"] = match_time.group(1)
            if "quiet on" in lowered:
                result["daily_brief_quiet_mode"] = "on"
            elif "quiet off" in lowered:
                result["daily_brief_quiet_mode"] = "off"
        m_disk_delta = re.search(r"set disk delta threshold to\s*(\d+)\s*mb", lowered)
        if m_disk_delta:
            result["disk_delta_threshold_mb"] = str(max(1, min(102400, int(m_disk_delta.group(1)))))
        if lowered.startswith("only send if service unhealthy "):
            if lowered.endswith("on"):
                result["only_send_if_service_unhealthy"] = "on"
            elif lowered.endswith("off"):
                result["only_send_if_service_unhealthy"] = "off"
        m_due = re.search(r"set open loops due window to\s*(\d+)\s*day", lowered)
        if m_due:
            result["include_open_loops_due_within_days"] = str(max(1, min(30, int(m_due.group(1)))))
        m_path = re.search(r"important path[s]?:\s*(.+)$", raw, re.IGNORECASE)
        if m_path:
            parts = [p.strip() for p in m_path.group(1).split(",") if p.strip()]
            result["important_paths"] = parts[:10]
        m_window = re.search(r"baseline window\s*(?:is|=)?\s*([a-z0-9_ -]+)$", lowered)
        if m_window:
            result["baseline_window"] = m_window.group(1).strip()[:50]
        return result

    def _today_cards_payload(self, request_text: str = "") -> dict[str, Any]:
        tasks_md = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tasks.md"))
        tasks: list[tuple[str, int | None]] = []
        lowered = (request_text or "").strip().lower()
        if os.path.isfile(tasks_md):
            try:
                with open(tasks_md, "r", encoding="utf-8") as handle:
                    for raw_line in handle.readlines():
                        t = raw_line.strip()
                        if t.startswith("- [ ]") or t.startswith("- [x]"):
                            body = t[5:].strip()
                            mins = None
                            match = re.search(r"\[(\d+)m\]", body, re.IGNORECASE)
                            if match:
                                mins = int(match.group(1))
                                body = re.sub(r"\s*\[\d+m\]\s*", " ", body, flags=re.IGNORECASE).strip()
                            tasks.append((body, mins))
            except Exception:
                tasks = []
        if not tasks:
            rows = self.db.list_tasks(limit=8)
            for row in rows:
                if str(row.get("status") or "").lower() in {"todo", "doing"}:
                    effort = row.get("effort_mins")
                    mins = int(effort) if isinstance(effort, int) else None
                    tasks.append((str(row.get("title") or "").strip(), mins))
        if "quick wins" in lowered:
            quick = [item for item in tasks if item[1] is not None and item[1] <= 20]
            tasks = quick if quick else tasks
        if "top 3" in lowered or "top three" in lowered or "priorities" in lowered:
            tasks = tasks[:3]
        lines = [
            f"{title} ({mins}m)" if mins is not None else title
            for title, mins in tasks[:6]
            if title
        ]
        if not lines:
            lines = ["No tasks found in tasks.md or DB tasks table."]
        cards = [{"key": "today-plan", "title": "Today plan", "lines": lines[:6], "severity": "ok"}]
        return build_cards_payload(
            cards,
            raw_available=True,
            summary="Here is your read-only plan for today.",
            confidence=0.90,
            next_questions=["Show top 3 priorities.", "Show only quick wins."],
        )

    def _apply_card_preferences(self, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        out = dict(payload or {})
        cards = list(out.get("cards") or [])
        next_questions = list(out.get("next_questions") or [])
        max_cards_pref = self.db.get_preference("max_cards")
        max_cards = int(max_cards_pref) if (max_cards_pref and max_cards_pref.isdigit()) else 4
        max_cards = max(1, min(12, max_cards))
        out["cards"] = cards[:max_cards]
        next_q_pref = self.db.get_preference("next_questions_limit")
        next_q_limit = int(next_q_pref) if (next_q_pref and next_q_pref.isdigit()) else 2
        next_q_limit = max(0, min(5, next_q_limit))
        out["next_questions"] = next_questions[:next_q_limit]
        show_conf_pref = (self.db.get_preference("show_confidence") or "on").strip().lower()
        out["show_confidence"] = show_conf_pref not in {"off", "false", "0", "no"}
        return out

    def _cards_response(self, user_id: str, payload: dict[str, Any]) -> OrchestratorResponse:
        adjusted = self._apply_card_preferences(user_id, payload)
        return OrchestratorResponse(render_cards_markdown(adjusted), adjusted)

    def _preferences_cards_payload(self) -> dict[str, Any]:
        keys = [
            "response_style",
            "max_cards",
            "default_compare",
            "default_compare_enabled",
            "show_confidence",
            "daily_brief_enabled",
            "daily_brief_time",
            "daily_brief_quiet_mode",
            "disk_delta_threshold_mb",
            "only_send_if_service_unhealthy",
            "include_open_loops_due_within_days",
            "important_paths",
            "baseline_window",
        ]
        lines = []
        for key in keys:
            value = self.db.get_preference(key)
            if value is not None and str(value).strip():
                lines.append(f"{key}: {value}")
        if not lines:
            lines = ["No saved preferences yet."]
        return build_cards_payload(
            [{"key": "prefs", "title": "Your preferences", "lines": lines, "severity": "ok"}],
            raw_available=False,
            summary="Current saved assistant preferences.",
            confidence=1.0,
            next_questions=self._preferences_followups(),
        )

    def _preferences_followups(self) -> list[str]:
        followups: list[str] = []
        max_cards = self.db.get_preference("max_cards")
        show_conf = (self.db.get_preference("show_confidence") or "on").strip().lower()
        if max_cards and max_cards.isdigit():
            next_val = 4 if int(max_cards) != 4 else 3
            followups.append(f"Set max cards to {next_val}")
        else:
            followups.append("Set max cards to 4")
        if show_conf in {"off", "false", "0", "no"}:
            followups.append("Turn confidence on")
        else:
            followups.append("Turn confidence off")
        followups.append("Change max cards")
        return followups

    def _memory_inspection_payload(self) -> dict[str, Any]:
        rows = self.db.list_preferences()
        lines: list[str] = []
        for row in rows:
            key = str(row.get("key") or "")
            if not key:
                continue
            value = str(row.get("value") or "")
            updated = str(row.get("updated_at") or "unknown")
            lines.append(f"{key}: {value} (updated {updated})")
        if not lines:
            lines = ["No remembered preferences or anchors yet."]
        return build_cards_payload(
            [{"key": "memory-inspect", "title": "What I remember about you", "lines": lines, "severity": "ok"}],
            raw_available=False,
            summary="Only explicit preferences and anchors are stored.",
            confidence=1.0,
            next_questions=self._preferences_followups(),
        )

    def _parse_open_loop_add(self, text: str) -> tuple[str | None, str | None, int]:
        match = re.match(r"^remember that (?:i need to )?(.+?) by (.+)$", (text or "").strip(), re.IGNORECASE)
        if not match:
            return None, None, 3
        title = match.group(1).strip()
        priority = 3
        if title.startswith("!"):
            priority = 1
            title = title.lstrip("!").strip()
        return title, match.group(2).strip(), priority

    def _parse_open_loop_done(self, text: str) -> str | None:
        match = re.match(r"^mark (.+) done$", (text or "").strip(), re.IGNORECASE)
        if not match:
            return None
        return match.group(1).strip()

    def _open_loops_payload(
        self,
        due_soon_only: bool = False,
        status: str = "open",
        order: str = "created",
        due_within_days: int = 2,
    ) -> dict[str, Any]:
        rows = self.db.list_open_loops(status=status, limit=20, order=order)
        lines: list[str] = []
        if due_soon_only:
            now = datetime.now(timezone.utc).date()
            filtered = []
            for row in rows:
                due = (row.get("due_date") or "").strip()
                try:
                    due_date = datetime.fromisoformat(due).date()
                except Exception:
                    continue
                if (due_date - now).days <= due_within_days:
                    filtered.append(row)
            rows = filtered
        for row in rows:
            due = row.get("due_date") or "no due date"
            priority = int(row.get("priority") or 3)
            lines.append(f"P{priority} {row.get('title')}: due {due} ({row.get('status')})")
        if not lines:
            lines = ["No open loops."]
        label = "all" if status == "all" else "open"
        return build_cards_payload(
            [{"key": "open-loops", "title": "Open loops", "lines": lines[:8], "severity": "ok"}],
            raw_available=False,
            summary=f"Tracked {label} loops.",
            confidence=1.0,
            next_questions=["Mark one done", "Add a new loop with due date"],
        )

    def _brief_decision_reasons(self, decision_reason: str, signals: dict[str, Any], threshold_mb: float) -> list[str]:
        lines: list[str] = []
        if decision_reason == "send":
            lines.append("Brief should send now.")
            return lines
        if decision_reason == "already_sent_today":
            lines.append("Already sent today.")
        if decision_reason == "before_time":
            lines.append("Scheduled time has not been reached.")
        if decision_reason == "disabled":
            lines.append("Daily brief is disabled.")
        if decision_reason == "service_healthy_gate":
            lines.append("Suppressed: service is healthy and service-only gate is on.")
        if decision_reason == "quiet_no_signals":
            disk_delta = signals.get("disk_delta_mb")
            lines.append(
                f"Suppressed: disk delta below threshold ({disk_delta}MB < {threshold_mb:.0f}MB)."
            )
            lines.append("Suppressed: no due open loops in window.")
        if decision_reason == "invalid_time":
            lines.append("Configured brief time is invalid.")
        return lines or [f"Decision: {decision_reason}"]

    def _daily_brief_status_payload(self, user_id: str) -> dict[str, Any]:
        from agent.daily_brief import should_send_daily_brief

        enabled = (self.db.get_preference("daily_brief_enabled") or "off").strip().lower() in {"on", "true", "1", "yes"}
        local_time = (self.db.get_preference("daily_brief_time") or "09:00").strip()
        last_sent = self.db.get_preference("daily_brief_last_sent_date")
        quiet_mode = (self.db.get_preference("daily_brief_quiet_mode") or "off").strip().lower() in {"on", "true", "1", "yes"}
        threshold_pref = self.db.get_preference("disk_delta_threshold_mb")
        threshold_mb = float(threshold_pref) if (threshold_pref and threshold_pref.isdigit()) else 250.0
        svc_gate = (self.db.get_preference("only_send_if_service_unhealthy") or "off").strip().lower() in {"on", "true", "1", "yes"}
        include_due_pref = self.db.get_preference("include_open_loops_due_within_days")
        include_due_days = int(include_due_pref) if (include_due_pref and include_due_pref.isdigit()) else 2
        brief_payload = self.build_daily_brief_cards(user_id)
        signals = brief_payload.get("daily_brief_signals") if isinstance(brief_payload, dict) else {}
        decision = should_send_daily_brief(
            now_utc=datetime.now(timezone.utc),
            timezone_name=self.timezone,
            enabled=enabled,
            local_time_hhmm=local_time,
            last_sent_local_date=last_sent,
            quiet_mode=quiet_mode,
            disk_delta_mb=(signals or {}).get("disk_delta_mb"),
            disk_delta_threshold_mb=threshold_mb,
            service_unhealthy=bool((signals or {}).get("service_unhealthy")),
            only_send_if_service_unhealthy=svc_gate,
            has_due_open_loops=int((signals or {}).get("due_open_loops_count") or 0) > 0 and include_due_days > 0,
        )
        reasons = self._brief_decision_reasons(decision.reason, signals or {}, threshold_mb)
        cards = [
            {
                "key": "brief-status",
                "title": "Daily brief status",
                "lines": [f"should_send: {decision.should_send}", f"reason: {decision.reason}", *reasons],
                "severity": "ok" if decision.should_send else "warn",
            },
            {
                "key": "brief-signals",
                "title": "Signals",
                "lines": [
                    f"disk_delta_mb: {(signals or {}).get('disk_delta_mb')}",
                    f"service_unhealthy: {bool((signals or {}).get('service_unhealthy'))}",
                    f"due_open_loops_count: {int((signals or {}).get('due_open_loops_count') or 0)}",
                ],
                "severity": "ok",
            },
        ]
        return build_cards_payload(
            cards,
            raw_available=True,
            summary="Daily brief send decision explained.",
            confidence=1.0,
            next_questions=["Enable daily brief at 09:00", "Set disk delta threshold to 300 mb"],
        )

    def _systemd_active(self, unit: str) -> str:
        try:
            proc = subprocess.run(
                ["systemctl", "is-active", unit],
                check=False,
                capture_output=True,
                text=True,
                timeout=1.5,
            )
            text = (proc.stdout or proc.stderr or "").strip()
            return text or "unknown"
        except Exception:
            return "unknown"

    def _health_payload(self, user_id: str) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        uptime_sec = int((datetime.now(timezone.utc) - self._started_at).total_seconds())
        observe_audit = self.db.audit_log_latest_by_type("observe_now_scheduled")
        observe_last = observe_audit.get("created_at") if isinstance(observe_audit, dict) else "none"
        observe_status = observe_audit.get("status") if isinstance(observe_audit, dict) else "unknown"
        daily_enabled = self.db.get_preference("daily_brief_enabled") or "off"
        daily_time = self.db.get_preference("daily_brief_time") or "09:00"
        daily_last_sent = self.db.get_preference("daily_brief_last_sent_date") or "never"
        last_failed = None
        for row in self.db.audit_log_list_recent(user_id, limit=20):
            if str(row.get("status") or "").lower() == "failed":
                last_failed = row
                break
        error_line = "none"
        if last_failed:
            err = redact_payload({"error": str(last_failed.get("error") or "")}).get("error") or ""
            error_line = f"{last_failed.get('action_type')}:{last_failed.get('action_id')} {err[:140]}"
        cards = [
            {
                "key": "health-bot",
                "title": "Bot status",
                "lines": [f"now_utc: {now}", f"uptime_sec: {uptime_sec}", f"user: {user_id}"],
                "severity": "ok",
            },
            {
                "key": "health-observe",
                "title": "Observe scheduler",
                "lines": [
                    f"service active: {self._systemd_active('personal-agent-observe.service')}",
                    f"timer active: {self._systemd_active('personal-agent-observe.timer')}",
                    f"last run: {observe_last} ({observe_status})",
                ],
                "severity": "ok" if observe_status in {"executed", "started", "unknown"} else "warn",
            },
            {
                "key": "health-db",
                "title": "Database",
                "lines": [f"path: {self.db.db_path}", f"schema_version: {self.db.get_schema_version()}"],
                "severity": "ok",
            },
            {
                "key": "health-brief",
                "title": "Daily brief config",
                "lines": [
                    f"enabled: {daily_enabled}",
                    f"time: {daily_time}",
                    f"last_sent_date: {daily_last_sent}",
                ],
                "severity": "ok",
            },
            {
                "key": "health-error",
                "title": "Last error summary",
                "lines": [error_line],
                "severity": "warn" if error_line != "none" else "ok",
            },
        ]
        return build_cards_payload(
            cards,
            raw_available=False,
            summary="Health snapshot of bot, scheduler, DB, and daily brief.",
            confidence=1.0,
            next_questions=["Show daily brief status", "Show open loops due"],
        )

    def build_daily_brief_cards(self, user_id: str) -> dict[str, Any]:
        cards: list[dict[str, Any]] = []
        include_due_days_pref = self.db.get_preference("include_open_loops_due_within_days")
        include_due_days = int(include_due_days_pref) if (include_due_days_pref and include_due_days_pref.isdigit()) else 2
        disk_delta_mb: float | None = None
        service_unhealthy = False
        due_open_loops_count = 0
        # Disk delta + top growth.
        storage_resp = self._call_skill(
            user_id,
            "storage_governor",
            "storage_report",
            {"user_id": user_id},
            ["db:read"],
            action_type="observe",
            read_only_mode=True,
        )
        storage_data = storage_resp.data if isinstance(storage_resp.data, dict) else {}
        storage_cards = ((storage_data.get("cards_payload") or {}).get("cards") or []) if isinstance(storage_data, dict) else []
        disk_usage = next((c for c in storage_cards if isinstance(c, dict) and str(c.get("title", "")).startswith("Disk usage")), None)
        top_growth = next((c for c in storage_cards if isinstance(c, dict) and str(c.get("title", "")).startswith("Top growing paths")), None)
        if isinstance(disk_usage, dict):
            cards.append(disk_usage)
        if isinstance(top_growth, dict):
            cards.append(top_growth)
        mounts = (storage_data.get("payload") or {}).get("mounts") if isinstance(storage_data, dict) else None
        if isinstance(mounts, list):
            delta_bytes = 0
            for mount in mounts:
                if not isinstance(mount, dict):
                    continue
                value = mount.get("delta_used")
                if value is None:
                    continue
                try:
                    delta_bytes += int(value)
                except Exception:
                    continue
            disk_delta_mb = round(delta_bytes / (1024.0 * 1024.0), 2)
        # Service verdict.
        service_resp = self._call_skill(
            user_id,
            "service_health_report",
            "service_health_report",
            {"user_id": user_id},
            ["db:read"],
            action_type="observe",
            read_only_mode=True,
        )
        service_data = service_resp.data if isinstance(service_resp.data, dict) else {}
        service_cards = ((service_data.get("cards_payload") or {}).get("cards") or []) if isinstance(service_data, dict) else []
        if service_cards and isinstance(service_cards[0], dict):
            cards.append(service_cards[0])
            sev = str(service_cards[0].get("severity") or "ok").lower()
            service_unhealthy = sev in {"warn", "bad"}
        # Today plan.
        today_payload = self._today_cards_payload()
        today_cards = today_payload.get("cards") or []
        if today_cards and isinstance(today_cards[0], dict):
            cards.append(today_cards[0])
        # Due soon loops (only in daily brief flow).
        loops_payload = self._open_loops_payload(due_soon_only=True, order="due", due_within_days=include_due_days)
        loops_cards = loops_payload.get("cards") or []
        if loops_cards and isinstance(loops_cards[0], dict):
            cards.append(loops_cards[0])
            loop_lines = [str(item) for item in (loops_cards[0].get("lines") or [])]
            if loop_lines == ["No open loops."]:
                due_open_loops_count = 0
            else:
                due_open_loops_count = len(loop_lines)
        payload = self._apply_card_preferences(
            user_id,
            build_cards_payload(
                cards,
                raw_available=True,
                summary="Daily brief: key changes and today’s focus.",
                confidence=0.95,
                next_questions=["Show full disk pressure report", "Show all open loops"],
            ),
        )
        payload["daily_brief_signals"] = {
            "disk_delta_mb": disk_delta_mb,
            "service_unhealthy": bool(service_unhealthy),
            "due_open_loops_count": int(due_open_loops_count),
        }
        return payload

    def _domain_summary_and_followups(
        self, skill_name: str, function_name: str, data: dict[str, Any], text: str
    ) -> tuple[str, list[str]]:
        payload = data.get("payload") if isinstance(data, dict) else {}
        if skill_name == "storage_governor" and function_name == "storage_report":
            mounts = (payload or {}).get("mounts") or []
            growth_lines = []
            top_used = None
            if mounts:
                top_used = sorted(mounts, key=lambda m: float(m.get("used_pct", 0.0)), reverse=True)[0]
            root_top = ((payload or {}).get("root_top") or {}).get("samples") or []
            home_top = ((payload or {}).get("home_top") or {}).get("samples") or []
            top_path = None
            if root_top or home_top:
                combined = sorted(list(root_top) + list(home_top), key=lambda t: int(t[1]), reverse=True)
                if combined:
                    top_path = combined[0][0]
            if top_used:
                growth_lines.append(
                    f"Disk {top_used.get('mountpoint')} is {float(top_used.get('used_pct', 0.0)):.1f}% used"
                )
            if top_path:
                growth_lines.append(f"largest tracked dir: {top_path}")
            summary = "; ".join(growth_lines) if growth_lines else "Disk status snapshot ready."
            followups = ["Show only top growing paths", "Show largest files in /var/log"]
            has_delta = any(m.get("delta_used") is not None for m in mounts)
            if has_delta:
                summary = summary + ". Deltas are included."
            return summary, followups
        if skill_name == "disk_pressure_report":
            largest_files = (payload or {}).get("largest_files") or []
            growth = (payload or {}).get("growth") or []
            culprit = largest_files[0][0] if largest_files else (growth[0][0] if growth else None)
            summary = f"Disk pressure culprit: {culprit}" if culprit else "Disk pressure is currently stable."
            return summary, ["Show only top growing paths", "Show largest files in /var/log"]
        if skill_name == "network_governor":
            cards_payload = data.get("cards_payload") if isinstance(data, dict) else {}
            lines = []
            if isinstance(cards_payload, dict):
                cards = cards_payload.get("cards") or []
                if cards and isinstance(cards[0], dict):
                    lines = [str(x) for x in (cards[0].get("lines") or [])]
            latency = next((line.split(":", 1)[1].strip() for line in lines if line.startswith("ping latency:")), "unavailable")
            route = next((line.split(":", 1)[1].strip() for line in lines if line.startswith("default route:")), "n/a")
            verdict = "Network looks healthy" if "unavailable" not in latency else "Network health is partial"
            return f"{verdict}; latency {latency}; route {route}", ["Show DNS changes", "Show interface errors only"]
        if skill_name == "service_health_report":
            report = (payload or {}).get("report") or ""
            lower = str(report).lower()
            running = "running" if "- status: active" in lower else "stopped/degraded"
            has_error = "error" in lower or "failed" in lower
            verdict = f"Service is {running}"
            if has_error:
                verdict += "; recent errors found"
            return verdict, ["Show last 20 service logs", "Check runtime status details"]
        if skill_name == "resource_governor":
            loads = (payload or {}).get("loads") or {}
            mem = (payload or {}).get("memory") or {}
            used = int(mem.get("used", 0))
            total = int(mem.get("total", 0))
            pct = (used / total * 100.0) if total else 0.0
            return f"CPU load 1m {float(loads.get('1m', 0.0)):.2f}; memory {pct:.1f}% used", [
                "Show only CPU deltas",
                "Show only memory deltas",
            ]
        return "Status snapshot ready.", ["Show details", "What changed since last snapshot?"]

    def _handle_nl_observe(self, user_id: str, text: str, decision: dict[str, Any]) -> OrchestratorResponse:
        selected = decision.get("skills") or []
        cards: list[dict[str, Any]] = []
        raw_available = False
        blocked_count = 0
        summary_parts: list[str] = []
        followups: list[str] = []
        permissions_map = {
            "storage_report": ["db:read"],
            "resource_report": ["db:read"],
            "network_report": ["db:read"],
            "service_health_report": ["db:read"],
            "disk_pressure_report": ["db:read", "sys:read"],
        }
        for idx, selected_skill in enumerate(selected):
            skill_name = selected_skill.get("skill")
            function_name = selected_skill.get("function")
            requested_permissions = permissions_map.get(str(function_name or ""), ["db:read"])
            allowed, _reason = can_run_nl_skill(
                self.skills,
                str(skill_name or ""),
                str(function_name or ""),
                requested_permissions=requested_permissions,
            )
            if not allowed:
                blocked_count += 1
                continue
            response = self._call_skill(
                user_id,
                skill_name,
                function_name,
                {"user_id": user_id},
                requested_permissions,
                action_type="observe",
                read_only_mode=True,
            )
            data = response.data if isinstance(response.data, dict) else {}
            summary, domain_followups = self._domain_summary_and_followups(
                str(skill_name or ""), str(function_name or ""), data, text
            )
            if summary:
                summary_parts.append(summary)
            for item in domain_followups:
                if item not in followups:
                    followups.append(item)
            cards_payload = data.get("cards_payload") if isinstance(data, dict) else None
            if isinstance(cards_payload, dict):
                for card in cards_payload.get("cards", []):
                    if not isinstance(card, dict):
                        continue
                    card_copy = dict(card)
                    card_copy.setdefault("key", f"{skill_name}:{function_name}:{idx}:{len(cards)}")
                    cards.append(card_copy)
                raw_available = raw_available or bool(cards_payload.get("raw_available"))
            elif response.text:
                cards.append(
                    {
                        "key": f"{skill_name}:{function_name}:{idx}",
                        "title": function_name.replace("_", " ").title(),
                        "lines": [response.text],
                        "severity": "ok",
                    }
                )
                raw_available = True

        if blocked_count and not cards:
            cards.append(
                {
                    "key": "nl-read-only-refused",
                    "title": "Read-only guard",
                    "lines": ["NL path refused non read-only skill execution."],
                    "severity": "warn",
                }
            )

        summary_text = "; ".join([part for part in summary_parts if part]).strip() or "Status snapshot ready."
        cards_payload = build_cards_payload(
            cards,
            raw_available=raw_available,
            summary=summary_text,
            confidence=0.95 if cards else 0.40,
            next_questions=followups or ["Show details", "What changed since last snapshot?"],
        )
        return self._cards_response(user_id, cards_payload)

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
