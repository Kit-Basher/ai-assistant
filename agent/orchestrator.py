from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
import os
import json
import uuid
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
from agent.logging_utils import log_event
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
                    return self.handle_message(command, user_id)
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

            return OrchestratorResponse(
                "I can help with /ask, /ask_opinion, /weekly_reflection, /storage_report, /resource_report, "
                "/brief, /network_report, /storage_snapshot, /observe_now, /what_if, /compare_now, /remind, /status, "
                "/runtime_status, /disk_grow, /audit, /llm_ping, /llm_status, /mem, /nomem. "
                "Use slash commands for now."
            )
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
