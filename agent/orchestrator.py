from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from functools import partial
import platform
import re
from typing import Any
import os
import json
import shlex
import uuid
import subprocess
import sqlite3
from datetime import datetime, timezone, timedelta
import time
from zoneinfo import ZoneInfo
from pathlib import Path

from agent.intent_router import route_message
from agent.intent.low_confidence import detect_low_confidence
from agent.disk_diff import diff_disk_reports, time_since
from agent.disk_anomalies import detect_anomalies
from agent.doctor import run_doctor_report
from agent.runner import Runner
from agent.disk_grow import resolve_allowed_path, build_growth_report, _run_du
from agent.action_gate import handle_action_text, propose_action
from agent.commands import parse_command, split_pipe_args
from agent.confirmations import ConfirmationStore, PendingAction
from agent.logging_utils import log_event, redact_payload
from agent.knowledge_cache import KnowledgeQueryCache, facts_hash
from agent.conversation_memory import record_event
import agent.memory_ingest as memory_ingest
from agent.packs.policy import PackPermissionDenied, enforce_iface_allowed
from agent.packs.store import PackStore
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
from agent.identity import (
    assistant_identity_label,
    get_public_identity,
    normalize_identity_name,
    user_identity_label,
)
from agent.llm.chat_preflight import build_chat_selection_policy_meta
from agent.llm.inference_router import route_inference
from agent.onboarding_contract import onboarding_next_action, onboarding_summary
from agent.recovery_contract import recovery_next_action, recovery_summary
from agent.runtime_contract import get_effective_llm_identity, normalize_user_facing_status
from agent.error_response_ux import compose_actionable_message, deterministic_error_message
from agent.runtime_truth_service import RuntimeTruthService
from agent.skill_governance import (
    ALLOWED_EXECUTION_MODES,
    DECLARABLE_CAPABILITIES,
    DEFAULT_EXECUTION_MODE,
    evaluate_skill_execution_request,
)
from agent.skill_governance_store import SkillGovernanceStore
from agent.setup_chat_flow import (
    _looks_like_current_model_query,
    _looks_like_local_model_inventory_query,
    _looks_like_model_availability_query,
    _looks_like_model_lifecycle_query,
    _looks_like_setup_explanation_query,
    _looks_like_runtime_status_query,
    classify_runtime_chat_route,
    normalize_setup_text,
)
from agent.skills.system_health_analyzer import build_system_health_report
from agent.skills.system_health import collect_system_health
from agent.skills.system_health_summary import render_system_health_summary
from agent.tool_contract import normalize_tool_request
from agent.tool_executor import ToolExecutor
from agent.memory_runtime import MemoryRuntime
from agent.working_memory import (
    append_turn as append_working_memory_turn,
    build_hot_messages,
    build_working_memory_context_text,
    default_budget as default_working_memory_budget,
    manage_working_memory,
    rebuild_state_from_messages,
)
from agent.memory_contract import (
    PENDING_STATUS_ABORTED,
    PENDING_STATUS_DONE,
    PENDING_STATUS_EXPIRED,
    PENDING_STATUS_READY_TO_RESUME,
    PENDING_STATUS_WAITING_FOR_USER,
    deterministic_memory_snapshot,
)
from agent.anchors import create_anchor, list_anchors, parse_anchor_input, reset_anchors
from agent.prefs import (
    ALLOWED_PREF_KEYS,
    get_project_mode,
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
from agent.perception import analyze_snapshot, collect_snapshot, summarize_inventory
from memory.db import MemoryDB


AUDIT_HARD_FAIL_MSG = "Audit logging failed. Operation aborted."
_TELEGRAM_LATENCY_ROUTE_TIMEOUT_SECONDS = 5.0
_TELEGRAM_LATENCY_FALLBACK_TIMEOUT_SECONDS = 4.0
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

_AUTHORITATIVE_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "system.performance": (
        "slow",
        "lag",
        "lagging",
        "stutter",
        "stuttering",
        "fps",
        "bottleneck",
        "throttle",
        "throttling",
        "temps",
        "temperature",
        "hot",
        "overheating",
        "cpu",
        "gpu",
        "vram",
        "ram",
    ),
    "system.health": (
        "crash",
        "crashing",
        "black screen",
        "boot loop",
        "boot",
        "service",
        "systemd",
        "failed unit",
        "failed units",
        "journal",
        "error loop",
    ),
    "system.storage": (
        "disk full",
        "out of space",
        "space left",
        "storage",
        "ssd full",
        "drive full",
        "what s eating space",
        "what is eating space",
        "largest folders",
        "disk space",
    ),
}
_LOCAL_OBS_SNAPSHOT_MARKERS = (
    '"ts":',
    '"cpu":',
    '"memory":',
    '"disk":',
)
_LOCAL_OBS_METRICS_MARKERS = (
    '"cpu_usage":',
    '"mem_available":',
    '"root_disk_used_pct":',
)
_AUTHORITATIVE_DOMAIN_TO_TOOL = {
    "system.performance": "sys_metrics_snapshot",
    "system.health": "sys_health_report",
    "system.storage": "sys_inventory_summary",
}
_LLM_RUN_DIRECTIVE_RE = re.compile(r"\[\[RUN:(/[a-z_]+)\]\]", re.IGNORECASE)
_LLM_RUN_DIRECTIVE_ALLOWLIST = {"/brief", "/status", "/health"}
_VENDOR_IDENTITY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("anthropic", ("created by anthropic", "i am anthropic", "i'm anthropic")),
    ("openai", ("created by openai", "i am openai", "i'm openai", "as an openai")),
    ("google", ("created by google", "i am google", "i'm google")),
)
_ASSISTANT_IDENTITY_REWRITE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?i)\bI\s+(?:am|'m|’m)\s+"
            r"(?:an?\s+)?(?:AI\s+assistant\s+)?(?:from\s+)?"
            r"(?:OpenAI|Anthropic|Google|DeepSeek|OpenRouter|Ollama|"
            r"GPT[-\w.]*|Claude[-\w.]*|Gemini[-\w.]*|Qwen[-\w.]*|Llama[-\w.]*|Mistral[-\w.]*)\b"
        ),
        "I am your Personal Agent assistant",
    ),
    (
        re.compile(r"(?i)\bcreated by\s+(?:OpenAI|Anthropic|Google)\b"),
        "running inside your Personal Agent",
    ),
    (
        re.compile(r"(?i)\b(?:created|built|made|developed)\s+by\s+[^.]{1,80}\.?"),
        "running locally in your Personal Agent runtime.",
    ),
    (
        re.compile(
            r"(?i)\bas\s+an?\s+"
            r"(?:OpenAI|Anthropic|Google|DeepSeek|OpenRouter|Ollama|"
            r"GPT[-\w.]*|Claude[-\w.]*|Gemini[-\w.]*|Qwen[-\w.]*|Llama[-\w.]*|Mistral[-\w.]*)"
            r"[^,]*,\s*"
        ),
        "",
    ),
    (
        re.compile(
            r"(?i)\bI\s+(?:am|'m|’m)\s+running\s+in\s+an?\s+environment\s+managed\s+by\s+"
            r"(?:Alibaba\s+Cloud|AWS|Amazon|Azure|Google\s+Cloud|GCP|OpenAI|Anthropic)\b[^.]*\.?"
        ),
        "I am your Personal Agent assistant running locally in your Personal Agent runtime.",
    ),
    (
        re.compile(
            r"(?i)\b(?:managed|hosted|run|operated)\s+by\s+"
            r"(?:Alibaba\s+Cloud|AWS|Amazon|Azure|Google\s+Cloud|GCP|OpenAI|Anthropic)\b[^.]*\.?"
        ),
        "running locally in your Personal Agent runtime",
    ),
)
_ASSISTANT_IDENTITY_LEAK_RE = re.compile(
    r"(?i)\b(?:i\s+(?:am|'m|’m)|created by|as\s+an?|as\s+a)\b.*\b"
    r"(?:openai|anthropic|google|gemini|claude|deepseek|openrouter|ollama|"
    r"gpt[-\w.]*|qwen[-\w.]*|llama[-\w.]*|mistral[-\w.]*)\b"
)
_ASSISTANT_GENERIC_ORIGIN_LEAK_RE = re.compile(
    r"(?i)\b(?:created|built|made|developed|provided|powered)\s+by\b.{0,80}"
)
_ASSISTANT_ORIGIN_LEAK_RE = re.compile(
    r"(?i)\b(?:running\s+in\s+an?\s+environment\s+managed\s+by|"
    r"managed\s+by|hosted\s+by|operated\s+by|run\s+by)\b.*\b"
    r"(?:alibaba\s+cloud|aws|amazon|azure|google\s+cloud|gcp|openai|anthropic|cloud|vendor|platform|company)\b"
)
_GROUNDED_QUERY_ESCAPE_RE = re.compile(
    r"(?i)(?:"
    r"i\s+do\s+not\s+have\s+real[- ]time\s+access|"
    r"i\s+don'?t\s+have\s+real[- ]time\s+access|"
    r"running\s+in\s+an\s+environment\s+managed\s+by|"
    r"managed\s+by\s+alibaba\s+cloud|"
    r"alibaba\s+cloud"
    r")"
)
_INTERPRETATION_FOLLOWUP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("top_memory", re.compile(r"\b(what is using(?: up)? my memory the most|using the most memory|using up my memory|memory the most)\b", re.IGNORECASE)),
    ("explain", re.compile(r"\b(explain it to me|explain that|what does that mean|summari[sz]e that|what'?s the important part)\b", re.IGNORECASE)),
    ("concern", re.compile(r"\b(should i worry|do i need to worry|is that bad|is that normal|anything unusual|is there anything to be concerned about there)\b", re.IGNORECASE)),
    ("action", re.compile(r"\b(what should i do)\b", re.IGNORECASE)),
)
_DEEP_SYSTEM_FOLLOWUP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(can you learn more|learn more|show me more|what else can you find|dig deeper)\b", re.IGNORECASE),
    re.compile(r"\b(run a check and see if you can learn more)\b", re.IGNORECASE),
)
_MODEL_SCOUT_FOLLOWUP_RE = re.compile(
    r"\b(run it|check them|check it|evaluate them|test them|look into them|try them|compare them|see if they(?: re|'re)? good)\b",
    re.IGNORECASE,
)
_DIRECT_MODEL_SWITCH_TOKEN_RE = re.compile(
    r"\b(?:[a-z0-9._-]+:)?[a-z0-9][a-z0-9./_-]*:[a-z0-9][a-z0-9./_-]*\b",
    re.IGNORECASE,
)
_INTERPRETATION_DEBUG_REFLEX_RE = re.compile(
    r"(?i)(?:^|\b)(?:run|use|check|try)\s+"
    r"(?:`[^`]+`|free\b|top\b|htop\b|ps\b|ps\s+aux\b|vmstat\b|smem\b|iotop\b|df\b|du\b)"
)
_INTERPRETATION_SHELL_SNIPPET_RE = re.compile(
    r"(?i)(`[^`]+`|\b(?:free|top|htop|ps|vmstat|smem|iotop|df|du)\s+-[a-z])"
)
_INTERPRETABLE_RESULT_TTL_SECONDS = 900
_RUNTIME_SETUP_STATE_TTL_SECONDS = 900
_MODEL_TRIAL_STATE_TTL_SECONDS = 86400
_MODEL_SCOUT_ACTION_VERBS = (
    "run",
    "check",
    "scan",
    "inspect",
    "evaluate",
    "test",
    "try",
    "compare",
    "look into",
    "see if",
)
_MODEL_SCOUT_QUALITY_HINTS = (
    "good for us",
    "any good",
    "worth trying",
    "worth using",
    "useful",
    "compare",
    "evaluate",
)
_MODEL_SCOUT_STRATEGY_PHRASES = (
    "is there a better model i should use",
    "should we switch to a better model",
    "try a better model",
    "use a better model",
    "should i use a better model",
    "better model should use",
    "what better local models could i try",
    "better local models could i try",
)
_MODEL_SCOUT_REMOTE_ROLE_PHRASES = (
    "cheap cloud",
    "low cost cloud",
    "budget cloud",
    "cheap remote",
    "low cost remote",
    "budget remote",
    "premium model",
    "premium coding model",
    "premium research model",
)
_MODEL_SCOUT_DISCOVERY_PHRASES = (
    "better model we should download",
    "look on hugging face for a better model",
    "look on huggingface for a better model",
    "find promising local models",
    "find promising models to download",
    "check for a better model we should download",
    "look for new better models",
    "find new models",
    "find new local models",
    "find small models",
    "find smol models",
    "find tiny models",
    "find lightweight models",
    "show me smol models",
    "show me small models",
    "show me tiny models",
    "show me lightweight models",
    "look for smol models",
    "look for small models",
    "look for tiny models",
    "look for lightweight models",
    "search hugging face for models",
    "search huggingface for models",
)
_MODEL_SCOUT_SWITCH_BACK_PHRASES = (
    "switch back",
    "switch back to the previous model",
    "go back to the previous model",
    "use the previous model again",
    "switch back to the last model",
)
_RUNTIME_REPAIR_ACTION_PHRASES = (
    "repair it",
    "fix it",
    "repair ollama",
    "fix ollama",
    "repair openrouter",
    "fix openrouter",
    "get it working",
    "get this working",
    "help me get it working",
    "help me get this working",
    "help me fix that",
    "help me fix it",
)
_MODEL_READY_NOW_PHRASES = (
    "what models are ready now",
    "which models are ready now",
    "what models are usable right now",
    "which models are usable right now",
)
_MODEL_CONTROLLER_TEST_PHRASES = (
    "test this model without adopting it",
    "test that model without adopting it",
    "test this model without switching",
    "test that model without switching",
)
_MODEL_CONTROLLER_TRIAL_SWITCH_PHRASES = (
    "switch temporarily",
    "switch to it temporarily",
    "use it temporarily",
    "try it temporarily",
)
_MODEL_CONTROLLER_PROMOTE_PHRASES = (
    "make this model the default",
    "make that model the default",
    "make this the default",
    "make that the default",
)
_MODEL_SCOUT_TERM_STOPWORDS = {
    "the",
    "any",
    "that",
    "those",
    "these",
    "other",
    "available",
    "actual",
    "actually",
    "good",
    "models",
    "model",
    "chat",
    "local",
    "remote",
    "provider",
    "providers",
}


def _normalize_authoritative_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    return f" {cleaned.strip()} "


def _contains_keyword(normalized_text: str, keyword: str) -> bool:
    token = keyword.strip().lower()
    if not token:
        return False
    token = re.sub(r"[^a-z0-9]+", " ", token).strip()
    if not token:
        return False
    return f" {token} " in normalized_text


def classify_authoritative_domain(text: str) -> set[str]:
    normalized = _normalize_authoritative_text(text)
    if normalized.strip() == "":
        return set()
    domains: set[str] = set()
    for domain, keywords in _AUTHORITATIVE_DOMAIN_KEYWORDS.items():
        if any(_contains_keyword(normalized, keyword) for keyword in keywords):
            domains.add(domain)
    return domains


def has_local_observations_block(text: str) -> bool:
    lowered = (text or "").lower()
    if "local_observations" in lowered:
        return True
    compact = "".join(ch for ch in lowered if not ch.isspace())
    if all(marker in compact for marker in _LOCAL_OBS_SNAPSHOT_MARKERS):
        return True
    if all(marker in compact for marker in _LOCAL_OBS_METRICS_MARKERS):
        return True
    return False


@dataclass


class OrchestratorResponse:
    text: str
    data: dict[str, Any] | None = None


class Orchestrator:
    """Core runtime orchestration layer.

    Product/business logic belongs here (or shared core modules), not in transport adapters.
    """

    def __init__(
        self,
        db: MemoryDB,
        skills_path: str,
        log_path: str,
        timezone: str,
        llm_client: Any,
        enable_writes: bool = False,
        perception_enabled: bool = True,
        perception_roots: tuple[str, ...] | None = None,
        perception_interval_seconds: int = 5,
        runtime_truth_service: RuntimeTruthService | None = None,
        chat_runtime_adapter: Any | None = None,
        semantic_memory_service: Any | None = None,
    ) -> None:
        self.db = db
        self._skill_loader = SkillLoader(skills_path)
        self.skills = self._skill_loader.load_all()
        self._skill_governance_store = SkillGovernanceStore(db.db_path)
        self._skill_governance_decisions: dict[str, dict[str, Any]] = {}
        self._blocked_skill_governance: dict[str, dict[str, Any]] = {}
        self._pack_store = PackStore(db.db_path)
        for skill in self.skills.values():
            if str(skill.pack_trust).strip().lower() != "native":
                continue
            permissions = skill.pack_permissions if isinstance(skill.pack_permissions, dict) else {"ifaces": []}
            self._pack_store.ensure_native_pack(
                pack_id=str(skill.pack_id or skill.name),
                version=str(skill.version or "0.1.0"),
                permissions=permissions,
                manifest_path=skill.pack_manifest_path,
            )
        self._refresh_skill_governance_state()
        self.log_path = log_path
        self.timezone = timezone
        self.llm_client = llm_client
        self.confirmations = ConfirmationStore()
        self.enable_writes = enable_writes
        self._runner: Runner | None = None
        self.perception_enabled = bool(perception_enabled)
        self.perception_roots = tuple(
            part.strip()
            for part in (perception_roots or ("/home", "/data/projects"))
            if part and part.strip()
        ) or ("/home", "/data/projects")
        self.perception_interval_seconds = max(1, int(perception_interval_seconds))
        self._knowledge_cache = KnowledgeQueryCache()
        self._pending_compare: dict[str, dict[str, str]] = {}
        self._last_offer_topic: dict[str, str] = {}
        self._started_at = datetime.now(__import__("datetime").timezone.utc)
        self._epistemic_monitor = EpistemicMonitor(db)
        self._epistemic_history: dict[tuple[str, str], list[MessageTurn]] = {}
        self._epistemic_thread_state: dict[str, dict[str, str | None]] = {}
        self._memory_runtime = MemoryRuntime(db)
        self._runtime_truth_service = runtime_truth_service
        self._chat_runtime_adapter = chat_runtime_adapter
        self._semantic_memory_service = semantic_memory_service
        if self._semantic_memory_service is not None:
            try:
                setattr(self.db, "_semantic_memory_service", self._semantic_memory_service)
            except Exception:
                pass
        self._runtime_setup_state: dict[str, dict[str, Any]] = {}
        self._model_trial_state: dict[str, dict[str, Any]] = {}
        self._last_interpretable_result: dict[str, dict[str, Any]] = {}
        self._tool_executor = ToolExecutor(
            handlers={
                "brief": self._tool_handler_brief,
                "status": self._tool_handler_status,
                "health": self._tool_handler_health,
                "doctor": self._tool_handler_doctor,
                "observe_system_health": self._tool_handler_observe_system_health,
                "observe_now": self._tool_handler_observe_now,
            },
            emit_log=self._emit_tool_log,
            component="orchestrator.tool_executor",
        )

    def _emit_tool_log(self, event: str, payload: dict[str, Any]) -> None:
        log_event(self.log_path, event, payload)

    def _refresh_skill_governance_state(self) -> None:
        self._skill_governance_decisions = {}
        self._blocked_skill_governance = {}
        for row in self._skill_loader.blocked_skills:
            skill_id = str(row.get("skill_id") or "").strip()
            if not skill_id:
                continue
            recorded = self._skill_governance_store.record_skill_governance(
                skill_id=skill_id,
                skill_type=str(row.get("skill_type") or "general"),
                requested_execution_mode=str(row.get("requested_execution_mode") or DEFAULT_EXECUTION_MODE),
                requested_capabilities=[
                    str(item).strip()
                    for item in (row.get("requested_capabilities") if isinstance(row.get("requested_capabilities"), list) else [])
                    if str(item).strip()
                ],
                persistence_requested=bool(row.get("persistence_requested", False)),
                allowed=False,
                requires_user_approval=False,
                reason=str(row.get("reason") or "forbidden_persistence_pattern"),
                source_issues=[
                    str(item).strip()
                    for item in (row.get("source_issues") if isinstance(row.get("source_issues"), list) else [])
                    if str(item).strip()
                ],
                source_pack=str(row.get("source_pack") or "").strip() or None,
            )
            self._blocked_skill_governance[skill_id] = recorded

        for skill in self.skills.values():
            request = skill.execution_request
            if request is None:
                continue
            decision = evaluate_skill_execution_request(
                request,
                source_issues=skill.governance_source_issues,
                managed_background_task_approved=self._skill_governance_store.has_approved_background_task_for_skill(skill.name),
                managed_adapter_approved=self._skill_governance_store.has_approved_adapter_for_skill(skill.name),
            )
            recorded = self._skill_governance_store.record_skill_governance(
                skill_id=skill.name,
                skill_type=skill.skill_type,
                requested_execution_mode=request.requested_execution_mode,
                requested_capabilities=list(request.requested_capabilities),
                persistence_requested=bool(request.persistence_requested),
                allowed=bool(decision.allowed),
                requires_user_approval=bool(decision.requires_user_approval),
                reason=str(decision.reason or "unknown"),
                source_issues=list(decision.source_issues),
                source_pack=str(skill.pack_id or skill.name).strip() or None,
            )
            self._skill_governance_decisions[skill.name] = recorded

    def skill_governance_status(self) -> dict[str, Any]:
        return {
            "policy": {
                "allowed_execution_modes": sorted(ALLOWED_EXECUTION_MODES),
                "default_execution_mode": DEFAULT_EXECUTION_MODE,
                "declarable_capabilities": sorted(DECLARABLE_CAPABILITIES),
                "persistent_execution_default": "deny",
            },
            "skills": self._skill_governance_store.list_skill_governance(),
            "managed_adapters": self._skill_governance_store.list_managed_adapters(),
            "background_tasks": self._skill_governance_store.list_background_tasks(),
        }

    def register_managed_adapter(self, **kwargs: Any) -> dict[str, Any]:
        return self._skill_governance_store.register_managed_adapter(**kwargs)

    def register_background_task(self, **kwargs: Any) -> dict[str, Any]:
        return self._skill_governance_store.register_background_task(**kwargs)

    def _skill_governance_denied_response(self, skill_id: str, governance: dict[str, Any]) -> OrchestratorResponse:
        mode = str(governance.get("requested_execution_mode") or DEFAULT_EXECUTION_MODE).strip().lower() or DEFAULT_EXECUTION_MODE
        reason = str(governance.get("reason") or "execution_governance_denied").strip().lower() or "execution_governance_denied"
        if bool(governance.get("requires_user_approval", False)):
            if mode == "managed_background_task":
                message = (
                    f"Skill {skill_id} requested managed background task execution and needs platform approval first."
                )
            elif mode == "managed_adapter":
                message = f"Skill {skill_id} requested managed adapter execution and needs platform approval first."
            else:
                message = f"Skill {skill_id} needs platform approval before it can run."
        else:
            message = f"Skill {skill_id} is blocked by execution governance ({reason})."
        issues = governance.get("source_issues") if isinstance(governance.get("source_issues"), list) else []
        data = {
            "skill_governance": governance,
            "source_issues": [str(item).strip() for item in issues if str(item).strip()],
        }
        return OrchestratorResponse(message, data)

    def _llm_chat_available(self) -> bool:
        adapter = self._chat_runtime_adapter
        if (
            callable(getattr(adapter, "_safe_mode_enabled", None))
            and adapter._safe_mode_enabled()
            and callable(getattr(adapter, "assistant_chat_available", None))
        ):
            try:
                return bool(adapter.assistant_chat_available())
            except Exception:
                pass
        client = self.llm_client
        if not client:
            return False
        if not hasattr(client, "chat") or not callable(getattr(client, "chat", None)):
            return False
        enabled_fn = getattr(client, "enabled", None)
        if not callable(enabled_fn):
            return False
        try:
            return bool(enabled_fn())
        except Exception:
            return False

    @staticmethod
    def _bootstrap_no_chat_text() -> str:
        state = "LLM_MISSING"
        return (
            f"{onboarding_summary(state)}\n"
            "1) Start Ollama locally at http://127.0.0.1:11434.\n"
            "2) Install a local chat model (for example qwen2.5:3b-instruct).\n"
            f"Next: {onboarding_next_action(state)}"
        )

    def _bootstrap_no_chat_response(self) -> OrchestratorResponse:
        return OrchestratorResponse(self._bootstrap_no_chat_text())

    def _llm_error_fallback_response(self, user_id: str, text: str) -> OrchestratorResponse:
        heuristic_command = self._heuristic_llm_command(text)
        if heuristic_command:
            try:
                tool_request = self._command_to_tool_request(heuristic_command, reason="heuristic_command")
                if tool_request is not None:
                    return self._execute_tool_request(
                        tool_request=tool_request,
                        user_id=user_id,
                        surface="llm",
                        runtime_mode="DEGRADED",
                    )
            except Exception:
                pass
        mode = "LLM_UNAVAILABLE"
        return OrchestratorResponse(f"{recovery_summary(mode)}\nNext: {recovery_next_action(mode)}")

    @staticmethod
    def _trace_id(prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex[:10]}"

    def _continuity_error_response(
        self,
        *,
        title: str,
        failure_code: str,
        next_action: str,
    ) -> OrchestratorResponse:
        trace_id = self._trace_id("cont")
        return OrchestratorResponse(
            deterministic_error_message(
                title=title,
                trace_id=trace_id,
                component="orchestrator.continuity",
                failure_code=failure_code,
                next_action=next_action,
            ),
            {
                "trace_id": trace_id,
                "failure_code": failure_code,
            },
        )

    def _tool_runtime_mode(self) -> str:
        return "READY" if self._llm_chat_available() else "BOOTSTRAP_REQUIRED"

    @staticmethod
    def _command_to_tool_request(command: str, *, reason: str) -> dict[str, Any] | None:
        normalized = str(command or "").strip().lower()
        command_map = {
            "/brief": "brief",
            "/status": "status",
            "/health": "health",
            "/doctor": "doctor",
            "/health_system": "observe_system_health",
            "/observe_now": "observe_now",
        }
        tool_name = command_map.get(normalized)
        if not tool_name:
            return None
        return normalize_tool_request(
            {
                "tool": tool_name,
                "args": {},
                "reason": str(reason or "").strip().lower() or "llm_tool_request",
                "confidence": 1.0,
            }
        )

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        value = str(text or "").strip()
        if not value:
            return None
        candidate = value
        if value.startswith("```"):
            lines = [line for line in value.splitlines() if line.strip()]
            if len(lines) >= 2:
                candidate = "\n".join(line for line in lines[1:] if line.strip() != "```").strip()
        try:
            parsed = json.loads(candidate)
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    def _parse_llm_tool_request(self, llm_text: str) -> dict[str, Any] | None:
        parsed = self._extract_json_object(llm_text)
        if not isinstance(parsed, dict):
            return None
        if "tool" not in parsed:
            return None
        return normalize_tool_request(parsed)

    def _execute_tool_request(
        self,
        *,
        tool_request: dict[str, Any],
        user_id: str,
        surface: str,
        runtime_mode: str | None = None,
    ) -> OrchestratorResponse:
        mode = str(runtime_mode or "").strip().upper() or self._tool_runtime_mode()
        result = self._tool_executor.execute(
            request=tool_request,
            user_id=user_id,
            surface=surface,
            runtime_mode=mode,
            enable_writes=bool(self.enable_writes),
            safe_mode=False,
        )
        if bool(result.get("ok", False)):
            self._memory_runtime.set_last_tool(user_id, str(result.get("tool") or ""))
            return OrchestratorResponse(
                str(result.get("user_text") or "Done.").strip() or "Done.",
                {
                    "tool_result": {
                        "tool": result.get("tool"),
                        "trace_id": result.get("trace_id"),
                        "component": result.get("component"),
                        "data": result.get("data") if isinstance(result.get("data"), dict) else {},
                    }
                },
            )
        text = deterministic_error_message(
            title=f"❌ {str(result.get('user_text') or 'Tool request failed.').strip()}",
            trace_id=str(result.get("trace_id") or ""),
            component=str(result.get("component") or "orchestrator.tool_executor"),
            next_action=str(result.get("next_action") or "Run: python -m agent doctor"),
            failure_code=str(result.get("error_code") or "tool_request_failed"),
        )
        return OrchestratorResponse(text)

    def _status_response(self, user_id: str) -> OrchestratorResponse:
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

    def _memory_summary_response(self, user_id: str) -> OrchestratorResponse:
        continuity_health = self._memory_runtime.inspect_user_state(user_id)
        if bool(continuity_health.get("healthy", True)):
            thread_id = self._active_thread_id_for_user(user_id)
            snapshot = self._memory_runtime.deterministic_snapshot(user_id, thread_id=thread_id)
        else:
            runtime_state = self._memory_runtime.get_thread_state(user_id)
            active_state = self._epistemic_thread_state.get(user_id) or {}
            thread_id = (
                str(active_state.get("active_thread_id") or "").strip()
                or str(continuity_health.get("active_thread_id") or "").strip()
                or str(runtime_state.get("thread_id") or "").strip()
                or self._default_thread_id(user_id)
            )
            pending_items = self._memory_runtime.list_pending_items(
                user_id,
                thread_id=thread_id,
                include_expired=True,
            )
            snapshot = deterministic_memory_snapshot(
                thread_state=runtime_state,
                pending_items=pending_items,
                last_meaningful_user_request=self.db.get_user_pref(
                    self._memory_runtime._last_request_key(user_id)
                ),
                last_agent_action=self.db.get_user_pref(
                    self._memory_runtime._last_action_key(user_id)
                ),
            )
        summary = snapshot.get("memory_summary") if isinstance(snapshot.get("memory_summary"), dict) else {}
        pending_items = snapshot.get("pending_items") if isinstance(snapshot.get("pending_items"), list) else []
        lines = [
            f"Memory summary (thread {thread_id}):",
            f"Current topic: {str(summary.get('current_topic') or 'none')}",
            f"Pending items: {int(summary.get('pending_count') or 0)}",
            f"Last tool: {str(summary.get('last_tool') or 'none')}",
            f"Resumable: {'yes' if bool(summary.get('resumable', False)) else 'no'}",
        ]
        if not bool(continuity_health.get("healthy", True)):
            corrupt_entries = (
                continuity_health.get("corrupt_entries")
                if isinstance(continuity_health.get("corrupt_entries"), list)
                else []
            )
            corrupt_count = len(corrupt_entries)
            lines.append(
                (
                    "Continuity memory is degraded right now, so resume state may be incomplete."
                    if corrupt_count <= 0
                    else (
                        f"Continuity memory is degraded right now ({corrupt_count} corrupted "
                        f"{'entry' if corrupt_count == 1 else 'entries'} ignored), so resume state may be incomplete."
                    )
                )
            )
        last_request = str(summary.get("last_meaningful_user_request") or "").strip()
        if last_request:
            lines.append(f"Last request: {last_request}")
        last_action = str(summary.get("last_agent_action") or "").strip()
        if last_action:
            lines.append(f"Last action: {last_action}")
        if pending_items:
            first = pending_items[0] if isinstance(pending_items[0], dict) else {}
            question = str(first.get("question") or "").strip()
            if question:
                lines.append(f"Next pending: {question}")
        return OrchestratorResponse(
            "\n".join(lines),
            {
                "memory_snapshot": snapshot,
                "thread_id": thread_id,
                "skip_runtime_thread_persist": not bool(continuity_health.get("healthy", True)),
                "skip_friction_formatting": True,
            },
        )

    @staticmethod
    def _assistant_memory_focus(query_text: str | None) -> str:
        normalized = " ".join(str(query_text or "").strip().lower().split())
        if any(
            phrase in normalized
            for phrase in (
                "what are we working on",
                "what were we working on",
                "what were we doing before",
                "thing we were doing before",
            )
        ):
            return "working_context"
        if any(
            phrase in normalized
            for phrase in (
                "what do you know about my system",
                "what do you know about this system",
                "what do you know about my machine",
                "what do you know about this machine",
            )
        ):
            return "system_context"
        return "memory_summary"

    def _assistant_memory_state(self, user_id: str) -> dict[str, Any]:
        thread_id = self._active_thread_id_for_user(user_id)
        prefs = self.db.list_preferences()
        anchors = self.db.list_thread_anchors(thread_id, limit=5)
        open_loops = self.db.list_open_loops(status="open", limit=5, order="due")
        tasks = [
            row
            for row in self.db.list_tasks(limit=8)
            if str(row.get("status") or "").strip().lower() in {"todo", "doing"}
        ]
        projects = self.db.list_projects()[:5]
        continuity_health = self._memory_runtime.inspect_user_state(user_id)
        snapshot = self._memory_runtime.deterministic_snapshot(user_id, thread_id=thread_id)
        summary = snapshot.get("memory_summary") if isinstance(snapshot.get("memory_summary"), dict) else {}
        truth = self._runtime_truth()
        target_truth = (
            truth.chat_target_truth()
            if truth is not None and callable(getattr(truth, "chat_target_truth", None))
            else {}
        )
        return {
            "thread_id": thread_id,
            "preferences": prefs,
            "anchors": anchors,
            "open_loops": open_loops,
            "tasks": tasks,
            "projects": projects,
            "snapshot": snapshot,
            "summary": summary,
            "continuity_health": continuity_health if isinstance(continuity_health, dict) else {},
            "target_truth": target_truth if isinstance(target_truth, dict) else {},
        }

    @staticmethod
    def _assistant_preference_hints(preferences: list[dict[str, Any]]) -> list[str]:
        hints: list[str] = []
        for row in preferences:
            if not isinstance(row, dict):
                continue
            key = str(row.get("key") or "").strip().lower()
            value = str(row.get("value") or "").strip()
            if not key or not value:
                continue
            if key == "response_style":
                hints.append(f"you prefer {value} replies")
                continue
            if key == "daily_brief_enabled":
                hints.append("daily brief is on" if value.lower() in {"on", "true", "1"} else "daily brief is off")
                continue
            if key == "default_compare":
                hints.append(f"default compare mode is {value}")
                continue
            if key == "show_confidence":
                hints.append("confidence display is on" if value.lower() in {"on", "true", "1"} else "confidence display is off")
                continue
            hints.append(f"there is a saved preference for {key.replace('_', ' ')}")
        return hints[:3]

    def _assistant_memory_overview_response(
        self,
        user_id: str,
        *,
        query_text: str | None = None,
    ) -> OrchestratorResponse:
        focus = self._assistant_memory_focus(query_text)
        state = self._assistant_memory_state(user_id)
        summary = state.get("summary") if isinstance(state.get("summary"), dict) else {}
        continuity_health = state.get("continuity_health") if isinstance(state.get("continuity_health"), dict) else {}
        preferences = state.get("preferences") if isinstance(state.get("preferences"), list) else []
        anchors = state.get("anchors") if isinstance(state.get("anchors"), list) else []
        open_loops = state.get("open_loops") if isinstance(state.get("open_loops"), list) else []
        tasks = state.get("tasks") if isinstance(state.get("tasks"), list) else []
        projects = state.get("projects") if isinstance(state.get("projects"), list) else []
        target_truth = state.get("target_truth") if isinstance(state.get("target_truth"), dict) else {}

        preference_hints = self._assistant_preference_hints(
            [row for row in preferences if isinstance(row, dict)]
        )
        anchor_titles = [
            str(row.get("title") or "").strip()
            for row in anchors
            if isinstance(row, dict) and str(row.get("title") or "").strip()
        ][:2]
        open_loop_titles = [
            str(row.get("title") or "").strip()
            for row in open_loops
            if isinstance(row, dict) and str(row.get("title") or "").strip()
        ][:3]
        task_titles = [
            str(row.get("title") or "").strip()
            for row in tasks
            if isinstance(row, dict) and str(row.get("title") or "").strip()
        ][:3]
        project_names = [
            str(getattr(row, "name", "") or "").strip()
            for row in projects
            if str(getattr(row, "name", "") or "").strip()
        ][:2]
        current_topic = str(summary.get("current_topic") or "").strip()
        last_request = str(summary.get("last_meaningful_user_request") or "").strip()
        last_action = str(summary.get("last_agent_action") or "").strip()
        effective_model = str(target_truth.get("effective_model") or "").strip()
        effective_provider = str(target_truth.get("effective_provider") or "").strip().lower()
        os_name = str(platform.system() or "").strip()
        arch = str(platform.machine() or "").strip()
        os_label = " ".join(part for part in (os_name, arch) if part)

        has_saved_memory = bool(
            preference_hints
            or anchor_titles
            or open_loop_titles
            or task_titles
            or project_names
            or (current_topic and current_topic != "none")
        )
        has_work_context = bool(has_saved_memory or last_request or last_action)
        continuity_warning = ""
        if not bool(continuity_health.get("healthy", True)):
            corrupt_entries = (
                continuity_health.get("corrupt_entries")
                if isinstance(continuity_health.get("corrupt_entries"), list)
                else []
            )
            corrupt_count = len(corrupt_entries)
            if corrupt_count > 0:
                continuity_warning = (
                    f"Continuity memory is degraded right now ({corrupt_count} corrupted "
                    f"{'entry' if corrupt_count == 1 else 'entries'} ignored), so resume state may be incomplete."
                )
            else:
                continuity_warning = (
                    "Continuity memory is degraded right now, so resume state may be incomplete."
                )

        if focus == "working_context":
            if not has_work_context:
                message = (
                    "I do not have a strong saved working context yet. "
                    "I can remember preferences, open loops, and active tasks as we go."
                )
                if continuity_warning:
                    message = f"{continuity_warning} {message}"
            else:
                lines: list[str] = []
                if continuity_warning:
                    lines.append(continuity_warning)
                if current_topic and current_topic != "none":
                    lines.append(f"It looks like we were focused on {current_topic}.")
                if last_request:
                    lines.append(f"The last concrete request I have saved is: {last_request}.")
                if open_loop_titles:
                    lines.append(f"Open loops I am tracking include {', '.join(open_loop_titles)}.")
                if task_titles:
                    lines.append(f"Active tasks I can see include {', '.join(task_titles)}.")
                elif project_names:
                    lines.append(f"Related project context I can see includes {', '.join(project_names)}.")
                if not lines and last_action:
                    lines.append(f"The last notable action I recorded was: {last_action}.")
                lines.append("If you want, I can pick up from that context.")
                message = " ".join(lines)
        elif focus == "system_context":
            lines = []
            if continuity_warning:
                lines.append(continuity_warning)
            if os_label:
                lines.append(f"I know this assistant is running locally on {os_label}.")
            if effective_model and effective_provider:
                lines.append(f"Right now chat is using {effective_model} on {effective_provider}.")
            if preference_hints:
                lines.append(f"I also remember preferences like {', '.join(preference_hints)}.")
            if not has_saved_memory:
                lines.append("I do not have a richer saved system profile beyond the live runtime context yet.")
            else:
                if anchor_titles:
                    lines.append(f"Saved thread context includes {', '.join(anchor_titles)}.")
                if open_loop_titles:
                    lines.append(f"Relevant open loops include {', '.join(open_loop_titles)}.")
            message = " ".join(lines).strip() or "I do not have a richer saved system profile yet."
        else:
            if not has_saved_memory:
                message = (
                    "I do not have much saved about you yet. "
                    "Right now I can see the live runtime context, but I do not have saved preferences, open loops, or project notes to summarize."
                )
                if continuity_warning:
                    message = f"{continuity_warning} {message}"
            else:
                lines = ["Here is the useful memory I have right now."]
                if continuity_warning:
                    lines.insert(0, continuity_warning)
                if preference_hints:
                    lines.append(f"Preferences I know: {', '.join(preference_hints)}.")
                if current_topic and current_topic != "none":
                    lines.append(f"Current working context: {current_topic}.")
                if open_loop_titles:
                    lines.append(f"Open loops I am tracking: {', '.join(open_loop_titles)}.")
                if task_titles:
                    lines.append(f"Active tasks I can see: {', '.join(task_titles)}.")
                elif project_names:
                    lines.append(f"Project context I can see: {', '.join(project_names)}.")
                if anchor_titles:
                    lines.append(f"Saved thread anchors include {', '.join(anchor_titles)}.")
                if not any(item.endswith(".") for item in lines[-1:]):
                    lines[-1] = f"{lines[-1]}."
                lines.append("If you want, I can unpack any of those areas next.")
                message = " ".join(lines)

        return self._runtime_truth_response(
            text=message,
            route="agent_memory",
            used_runtime_state=False,
            used_memory=True,
            used_tools=["memory_store"],
            payload={
                "type": "agent_memory",
                "kind": focus,
                "summary": message,
                "preferences_count": len(preferences),
                "anchor_count": len(anchors),
                "open_loop_count": len(open_loops),
                "task_count": len(tasks),
                "project_count": len(projects),
                "current_topic": current_topic or None,
                "last_request": last_request or None,
                "effective_model": effective_model or None,
                "effective_provider": effective_provider or None,
            },
        )

    def _selective_chat_memory_context(self, user_id: str, text: str) -> str:
        normalized = " ".join(str(text or "").strip().lower().split())
        if not normalized:
            return ""
        wants_context = any(
            token in normalized
            for token in (
                "before",
                "again",
                "continue",
                "next step",
                "working on",
                "prefer",
                "preference",
                "same thing",
                "that task",
            )
        )
        if not wants_context:
            return ""
        state = self._assistant_memory_state(user_id)
        summary = state.get("summary") if isinstance(state.get("summary"), dict) else {}
        preferences = state.get("preferences") if isinstance(state.get("preferences"), list) else []
        open_loops = state.get("open_loops") if isinstance(state.get("open_loops"), list) else []
        preference_hints = self._assistant_preference_hints(
            [row for row in preferences if isinstance(row, dict)]
        )
        lines: list[str] = []
        current_topic = str(summary.get("current_topic") or "").strip()
        last_request = str(summary.get("last_meaningful_user_request") or "").strip()
        if current_topic and current_topic != "none":
            lines.append(f"Current topic: {current_topic}")
        if last_request:
            lines.append(f"Recent request: {last_request}")
        if preference_hints:
            lines.append(f"Preferences: {', '.join(preference_hints)}")
        open_loop_titles = [
            str(row.get("title") or "").strip()
            for row in open_loops
            if isinstance(row, dict) and str(row.get("title") or "").strip()
        ][:2]
        if open_loop_titles:
            lines.append(f"Open loops: {', '.join(open_loop_titles)}")
        return "\n".join(lines[:4]).strip()

    def _working_memory_budget(self, payload: dict[str, Any]) -> Any:
        max_context_tokens = int(payload.get("max_context_tokens") or payload.get("min_context_tokens") or 0) or 0
        adapter = self._chat_runtime_adapter
        if max_context_tokens <= 0 and adapter is not None:
            try:
                defaults = (
                    dict(adapter.get_defaults())
                    if callable(getattr(adapter, "get_defaults", None))
                    else {}
                )
            except Exception:
                defaults = {}
            model_id = (
                str(payload.get("model") or "").strip()
                or str(defaults.get("chat_model") or defaults.get("default_model") or "").strip()
            )
            registry = getattr(getattr(adapter, "_router", None), "registry", None)
            models = getattr(registry, "models", None)
            model_row = models.get(model_id) if isinstance(models, dict) and model_id else None
            max_context_tokens = int(getattr(model_row, "max_context_tokens", 0) or 0) or 0
        return default_working_memory_budget(max_context_tokens or None)

    def _working_memory_durable_ingestor(self, *, user_id: str, thread_id: str | None) -> Any:
        def _ingest(payload: dict[str, Any]) -> None:
            service = self._semantic_memory_service
            if service is None:
                return
            source_ref = str(payload.get("source_ref") or "").strip() or f"working-memory:{user_id}"
            scope = f"thread:{thread_id}" if str(thread_id or "").strip() else f"user:{user_id}"
            metadata = {
                "user_id": str(user_id or "").strip() or "unknown",
                "thread_id": str(thread_id or "").strip() or None,
                "working_memory": True,
                "source_ref": source_ref,
            }
            raw_text = str(payload.get("raw_text") or "").strip()
            note_text = str(payload.get("text") or "").strip()
            if raw_text:
                try:
                    service.ingest_conversation_text(
                        source_ref=source_ref,
                        text=raw_text,
                        scope=scope,
                        thread_id=str(thread_id or "").strip() or None,
                        pinned=False,
                        metadata={**metadata, "ingest_kind": "working_memory_chunk"},
                    )
                except Exception:
                    pass
            if note_text:
                try:
                    service.ingest_note_text(
                        source_ref=f"{source_ref}:durable",
                        text=note_text,
                        scope=scope,
                        pinned=True,
                        metadata={**metadata, "ingest_kind": "working_memory_summary"},
                    )
                except Exception:
                    pass

        return _ingest

    def _record_chat_working_memory_turn(
        self,
        *,
        user_id: str,
        role: str,
        text: str,
        chat_context: dict[str, Any] | None,
    ) -> None:
        if not isinstance(chat_context, dict):
            return
        cleaned = str(text or "").strip()
        if not cleaned:
            return
        state, issue = self._memory_runtime.load_working_memory_state(user_id)
        if issue is not None:
            return
        thread_state = self._memory_runtime.get_thread_state(user_id)
        append_working_memory_turn(
            state,
            role=str(role or "assistant").strip().lower() or "assistant",
            text=cleaned,
            topic_hint=str(thread_state.get("current_topic") or "").strip() or None,
        )
        self._memory_runtime.save_working_memory_state(
            user_id,
            state,
            refuse_if_corrupt=True,
        )

    def _prepare_working_memory_for_chat(
        self,
        *,
        user_id: str,
        text: str,
        payload: dict[str, Any],
        thread_id: str | None,
        messages: list[dict[str, str]],
        memory_context_text: str,
    ) -> dict[str, Any]:
        system_messages = [
            {"role": "system", "content": str(row.get("content") or "").strip()}
            for row in messages
            if isinstance(row, dict)
            and str(row.get("role") or "").strip().lower() == "system"
            and str(row.get("content") or "").strip()
        ]
        transcript_messages = [
            {"role": str(row.get("role") or "").strip().lower(), "content": str(row.get("content") or "").strip()}
            for row in messages
            if isinstance(row, dict)
            and str(row.get("role") or "").strip().lower() in {"user", "assistant"}
            and str(row.get("content") or "").strip()
        ]
        state, issue = self._memory_runtime.load_working_memory_state(user_id)
        if len(transcript_messages) > 1:
            state = rebuild_state_from_messages(transcript_messages, previous_state=state)
        elif transcript_messages:
            thread_state = self._memory_runtime.get_thread_state(user_id)
            for row in transcript_messages:
                append_working_memory_turn(
                    state,
                    role=str(row.get("role") or "user").strip().lower() or "user",
                    text=str(row.get("content") or "").strip(),
                    topic_hint=str(thread_state.get("current_topic") or "").strip() or None,
                )
        elif str(text or "").strip():
            thread_state = self._memory_runtime.get_thread_state(user_id)
            append_working_memory_turn(
                state,
                role="user",
                text=str(text or "").strip(),
                topic_hint=str(thread_state.get("current_topic") or "").strip() or None,
            )
        budget = self._working_memory_budget(payload)
        manage_working_memory(
            state,
            budget=budget,
            user_id=user_id,
            thread_id=thread_id,
            durable_ingestor=self._working_memory_durable_ingestor(user_id=user_id, thread_id=thread_id),
        )
        effective_memory_context_text = build_working_memory_context_text(
            state,
            current_query=str(text or ""),
            extra_context_text=memory_context_text,
        )
        hot_messages = build_hot_messages(state)
        effective_messages = [*system_messages, *hot_messages] if hot_messages else [*system_messages, *transcript_messages]
        if issue is None:
            self._memory_runtime.save_working_memory_state(
                user_id,
                state,
                refuse_if_corrupt=True,
            )
        return {
            "messages": effective_messages,
            "memory_context_text": effective_memory_context_text,
            "issue": issue,
            "used_working_memory": bool(state.warm_summaries or state.cold_state_blocks or state.hot_turns),
        }

    def _runtime_truth(self) -> RuntimeTruthService | None:
        return self._runtime_truth_service

    @staticmethod
    def _response_data(response: OrchestratorResponse) -> dict[str, Any]:
        return dict(response.data) if isinstance(response.data, dict) else {}

    @staticmethod
    def _merge_response_data(response: OrchestratorResponse, **updates: Any) -> OrchestratorResponse:
        data = Orchestrator._response_data(response)
        for key, value in updates.items():
            if value is None and key not in data:
                continue
            data[key] = value
        return OrchestratorResponse(response.text, data)

    @staticmethod
    def _chat_channel(source_surface: str | None) -> str:
        normalized = str(source_surface or "").strip().lower()
        if normalized in {"telegram", "api", "cli"}:
            return normalized
        return "api"

    def _record_runtime_event(self, event_name: str, **fields: Any) -> None:
        adapter = self._chat_runtime_adapter
        record = getattr(adapter, "record_runtime_event", None)
        if callable(record):
            try:
                record(event_name, **fields)
                return
            except Exception:
                pass
        try:
            log_event(self.log_path, event_name, fields)
        except Exception:
            pass

    @staticmethod
    def _router_error_kind(result: dict[str, Any]) -> str | None:
        if not isinstance(result, dict):
            return None
        return str(result.get("error_kind") or result.get("error_class") or "").strip().lower() or None

    @staticmethod
    def _router_attempt_model(result: dict[str, Any]) -> str | None:
        if not isinstance(result, dict):
            return None
        direct = str(result.get("model") or "").strip()
        if direct:
            return direct
        attempts = result.get("attempts") if isinstance(result.get("attempts"), list) else []
        for row in attempts:
            if not isinstance(row, dict):
                continue
            model = str(row.get("model") or "").strip()
            if model:
                return model
        return None

    def _runtime_truth_response(
        self,
        *,
        text: str,
        route: str,
        payload: dict[str, Any] | None = None,
        used_memory: bool = False,
        used_runtime_state: bool = True,
        used_llm: bool = False,
        used_tools: list[str] | None = None,
        error_kind: str | None = None,
        ok: bool = True,
        next_question: str | None = None,
    ) -> OrchestratorResponse:
        data: dict[str, Any] = {
            "route": str(route or "runtime_status").strip().lower() or "runtime_status",
            "used_runtime_state": bool(used_runtime_state),
            "used_llm": bool(used_llm),
            "used_memory": bool(used_memory),
            "used_tools": [str(item).strip() for item in (used_tools or []) if str(item).strip()],
            "skip_friction_formatting": True,
            "skip_epistemic_gate": True,
            "ok": bool(ok),
        }
        if error_kind:
            data["error_kind"] = str(error_kind).strip()
        if next_question:
            data["next_question"] = str(next_question).strip()
        if isinstance(payload, dict):
            data["runtime_payload"] = dict(payload)
        return OrchestratorResponse(str(text or "").strip() or "Done.", data)

    def _runtime_state_unavailable_response(
        self,
        *,
        route: str,
        used_memory: bool = False,
        reason: str | None = None,
    ) -> OrchestratorResponse:
        message = "I can't read a clean runtime status from the current state yet."
        return self._runtime_truth_response(
            text=message,
            route=route,
            used_memory=used_memory,
            error_kind="runtime_state_unavailable",
            payload={
                "type": "runtime_state_unavailable",
                "summary": message,
                "reason": str(reason or "runtime_state_unavailable").strip() or "runtime_state_unavailable",
            },
        )

    def _clear_pending_confirmation(
        self,
        user_id: str,
        *,
        status: str,
    ) -> PendingAction | None:
        pending = self.confirmations.pop(user_id)
        if pending is None:
            return None
        pending_id = str((pending.action if isinstance(pending.action, dict) else {}).get("pending_id") or "").strip()
        if pending_id:
            self._memory_runtime.set_pending_status(user_id, pending_id, status)
        return pending

    def _confirmation_preview_response(
        self,
        user_id: str,
        *,
        route: str,
        question: str,
        used_tools: list[str],
        action: dict[str, Any],
        title: str,
        preview_payload: dict[str, Any] | None = None,
        used_memory: bool = False,
        used_runtime_state: bool = True,
    ) -> OrchestratorResponse:
        self._clear_pending_confirmation(user_id, status=PENDING_STATUS_ABORTED)
        now_epoch = int(datetime.now(timezone.utc).timestamp())
        pending_id = f"confirm-{uuid.uuid4().hex[:10]}"
        action_payload = {
            **dict(action),
            "kind": "native_mutation",
            "pending_id": pending_id,
        }
        pending = PendingAction(
            user_id=user_id,
            action=action_payload,
            message=question,
        )
        self.confirmations.set(pending)
        self._memory_runtime.add_pending_item(
            user_id,
            {
                "pending_id": pending_id,
                "kind": "confirmation",
                "origin_tool": (used_tools[0] if used_tools else str(action.get("operation") or "native_mutation")),
                "question": question,
                "options": ["yes", "no"],
                "created_at": now_epoch,
                "expires_at": now_epoch + 600,
                "thread_id": self._active_thread_id_for_user(user_id),
                "status": PENDING_STATUS_WAITING_FOR_USER,
                "context": {
                    "operation": str(action.get("operation") or "").strip() or None,
                },
            },
        )
        payload = dict(preview_payload) if isinstance(preview_payload, dict) else {}
        payload.update(
            {
                "type": "action_confirmation_required",
                "title": title,
                "summary": question,
                "requires_confirmation": True,
                "confirmation_token": pending_id,
                "confirm_command": "yes",
                "cancel_command": "no",
                "mutating": True,
                "action": str(action.get("operation") or "").strip() or None,
            }
        )
        return self._runtime_truth_response(
            text=question,
            route=route,
            used_tools=used_tools,
            used_memory=used_memory,
            used_runtime_state=used_runtime_state,
            next_question=question,
            payload=payload,
        )

    def _execute_confirmed_native_mutation(self, user_id: str, action: dict[str, Any]) -> OrchestratorResponse:
        operation = str(action.get("operation") or "").strip().lower()
        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        if operation == "shell_install_package":
            return self._shell_install_package_response(
                user_id=user_id,
                manager=str(params.get("manager") or "").strip() or None,
                package=str(params.get("package") or "").strip() or None,
                scope=str(params.get("scope") or "").strip() or None,
                dry_run=bool(params.get("dry_run", False)),
                confirmed=True,
            )
        if operation == "shell_create_directory":
            return self._shell_create_directory_response(
                user_id,
                str(params.get("path") or "").strip() or None,
                confirmed=True,
            )
        if operation == "model_trial_switch":
            return self._execute_model_controller_trial_switch(
                user_id,
                model_id=str(params.get("model_id") or "").strip() or None,
                provider_id=str(params.get("provider_id") or "").strip().lower() or None,
            )
        if operation == "model_set_target":
            return self._execute_model_set_target(
                user_id,
                model_id=str(params.get("model_id") or "").strip() or None,
                provider_id=str(params.get("provider_id") or "").strip().lower() or None,
                promote_default=bool(params.get("promote_default", False)),
                used_memory=bool(params.get("used_memory", False)),
            )
        if operation == "model_acquire":
            return self._execute_model_acquire(
                model_id=str(params.get("model_id") or "").strip() or None,
                provider_id=str(params.get("provider_id") or "").strip().lower() or None,
            )
        if operation == "switch_better_local_model":
            return self._execute_switch_better_local_model(
                user_id,
                model_id=str(params.get("model_id") or "").strip() or None,
            )
        if operation == "model_switch_back":
            return self._execute_model_controller_switch_back(user_id)
        return self._runtime_truth_response(
            text="That pending confirmation target is no longer available.",
            route="action_tool",
            used_runtime_state=False,
            ok=False,
            error_kind="resumable_missing",
            payload={
                "type": "runtime_state_unavailable",
                "summary": "That pending confirmation target is no longer available.",
                "reason": "resumable_missing",
            },
        )

    def _assistant_capabilities_response(self) -> OrchestratorResponse:
        truth = self._runtime_truth()
        adapter = self._chat_runtime_adapter
        safe_mode_enabled = bool(
            callable(getattr(adapter, "_safe_mode_enabled", None))
            and adapter._safe_mode_enabled()
        )
        areas: list[dict[str, Any]] = [
            {
                "key": "system_inspection",
                "title": "System inspection",
                "summary": "I can inspect current memory, RAM, storage, and run an agent doctor check on this machine.",
                "available": True,
            },
            {
                "key": "local_memory",
                "title": "Local memory",
                "summary": "I can read and update your local memory, preferences, anchors, and open loops.",
                "available": True,
            },
        ]
        if truth is not None:
            target_truth = (
                truth.chat_target_truth()
                if callable(getattr(truth, "chat_target_truth", None))
                else {}
            )
            effective_model = str(target_truth.get("effective_model") or "").strip()
            effective_provider = str(target_truth.get("effective_provider") or "").strip().lower()
            current_target = (
                f"{effective_model} on {effective_provider}"
                if effective_model and effective_provider
                else "the current chat target"
            )
            areas.extend(
                [
                    {
                        "key": "runtime_status",
                        "title": "Runtime and model status",
                        "summary": (
                            f"I can tell you which model and provider are active, like {current_target}, "
                            "and whether the runtime is healthy."
                        ),
                        "available": True,
                    },
                    {
                        "key": "provider_setup",
                        "title": "Provider repair and switching",
                        "summary": "I can help repair, configure, or switch chat providers such as Ollama and OpenRouter.",
                        "available": True,
                    },
                    {
                        "key": "scheduler_status",
                        "title": "Scheduler and daily brief",
                        "summary": "I can inspect scheduler, daily brief, and managed background-task status.",
                        "available": True,
                    },
                ]
            )
        else:
            areas.append(
                {
                    "key": "runtime_status",
                    "title": "Runtime and model status",
                    "summary": "Runtime and provider inspection is limited right now because I cannot read the control-plane state.",
                    "available": False,
                }
            )

        lines = ["Here is what I can help with right now:"]
        for row in areas:
            title = str(row.get("title") or "Capability").strip()
            summary = str(row.get("summary") or "").strip()
            if not summary:
                continue
            lines.append(f"- {title}: {summary}")
        if safe_mode_enabled:
            lines.append("- Safe mode: background automation and remote fallback are currently paused on purpose.")
        lines.append("")
        lines.append("If you want, I can check the runtime or inspect your system resources now.")
        return self._runtime_truth_response(
            text="\n".join(lines).strip(),
            route="assistant_capabilities",
            payload={
                "type": "assistant_capabilities",
                "areas": [dict(row) for row in areas],
                "safe_mode": safe_mode_enabled,
            },
        )

    def _assistant_frontdoor_engaged(self, text: str) -> bool:
        adapter = self._chat_runtime_adapter
        frontdoor = getattr(adapter, "should_use_assistant_frontdoor", None)
        if not callable(frontdoor):
            return False
        try:
            return bool(
                frontdoor(
                    text=text,
                    route_decision=None,
                    is_user_chat=True,
                )
            )
        except Exception:
            return False

    def _configured_identity_names(self) -> tuple[str | None, str | None]:
        assistant_name = None
        user_name = None
        try:
            assistant_name = normalize_identity_name(self.db.get_preference("assistant_name"))
        except Exception:
            assistant_name = None
        try:
            user_name = normalize_identity_name(self.db.get_preference("user_name"))
        except Exception:
            user_name = None
        config = getattr(self._chat_runtime_adapter, "config", None)
        if assistant_name is None:
            assistant_name = normalize_identity_name(getattr(config, "assistant_name", None))
        if user_name is None:
            user_name = normalize_identity_name(getattr(config, "user_name", None))
        return assistant_name, user_name

    def _assistant_identity_prompt_lines(self) -> list[str]:
        assistant_name, user_name = self._configured_identity_names()
        assistant_label = assistant_identity_label(assistant_name=assistant_name)
        user_label = user_identity_label(user_name=user_name)
        lines = [
            f"You are {assistant_label}, the user's local-first personal assistant.",
            "Identity rule: you run locally inside the user's Personal Agent runtime.",
            "Always identify as the local Personal Agent.",
            "Never claim you were created by a company, vendor, model maker, or external platform.",
            "Never claim to be hosted, managed, or operated by any cloud or vendor environment.",
            "Do not invent a name, persona, company, or origin story.",
            f"If asked who you are, answer as {assistant_label} only.",
            "If no explicit user name is configured, refer to the user as 'you'.",
            "Keep the tone practical, direct, calm, and non-corporate.",
        ]
        if user_label != "you":
            lines.append(f"You may refer to the user as {user_label} sparingly when it helps clarity.")
        return lines

    @staticmethod
    def _assistant_identity_leaked(text: str) -> bool:
        prefix = "\n".join(
            line.strip()
            for line in str(text or "").strip().splitlines()[:3]
            if line.strip()
        )[:400]
        return bool(
            _ASSISTANT_IDENTITY_LEAK_RE.search(prefix)
            or _ASSISTANT_GENERIC_ORIGIN_LEAK_RE.search(prefix)
            or _ASSISTANT_ORIGIN_LEAK_RE.search(prefix)
        )

    def _translate_assistant_internal_error(
        self,
        text: str,
        *,
        response_data: dict[str, Any],
    ) -> str:
        normalized = str(text or "").strip()
        if not normalized:
            return normalized
        lowered = normalized.lower()
        route = str(response_data.get("route") or "").strip().lower()
        error_kind = str(response_data.get("error_kind") or "").strip().lower()
        if "i couldn't read that from the runtime state." in lowered:
            return "I'm having trouble reading the current runtime state right now."
        if "chat llm is unavailable." in lowered or "run: python -m agent setup" in lowered:
            return "Something went wrong while answering that. I'm having trouble accessing my language model right now."
        if all(marker in normalized for marker in ("trace_id:", "component:", "next_action:")):
            if route in {
                "generic_chat",
                "runtime_status",
                "provider_status",
                "model_status",
                "setup_flow",
                "operational_status",
            } or error_kind:
                return "Something went wrong while handling that request. I'm having trouble accessing that right now."
        return normalized

    def _apply_assistant_response_guard(
        self,
        *,
        user_id: str,
        user_text: str,
        response: OrchestratorResponse,
    ) -> OrchestratorResponse:
        response_data = self._response_data(response)
        route = str(response_data.get("route") or "").strip().lower()
        used_llm = bool(response_data.get("used_llm", False))
        error_kind = str(response_data.get("error_kind") or "").strip().lower()
        guarded_text = str(response.text or "")
        if used_llm or route == "generic_chat":
            provider = str(response_data.get("provider") or "").strip() or None
            model = str(response_data.get("model") or "").strip() or None
            guarded_text = self._sanitize_vendor_identity_claim(
                guarded_text,
                provider=provider,
                model=model,
            )
        if self._assistant_frontdoor_engaged(user_text):
            lowered_guarded_text = str(guarded_text or "").strip().lower()
            bootstrap_recovery_text = any(
                marker in lowered_guarded_text
                for marker in (
                    "start ollama locally",
                    "install a local chat model",
                    "no chat model available right now",
                )
            )
            generic_fallback_used = bool(response_data.get("generic_fallback_used", False))
            generic_chat_like = route in {"", "generic_chat"}
            preserve_raw_recovery = bool(
                generic_chat_like
                and not used_llm
                and not error_kind
                and (
                    generic_fallback_used
                    or not self._llm_chat_available()
                    or bootstrap_recovery_text
                )
            )
            guarded_text = self._translate_assistant_internal_error(
                guarded_text if not preserve_raw_recovery else str(response.text or ""),
                response_data=response_data,
            ) if not preserve_raw_recovery else str(response.text or "")
            if not preserve_raw_recovery and (
                (used_llm or route == "generic_chat")
                and self._looks_like_grounded_system_query(user_text)
                and _GROUNDED_QUERY_ESCAPE_RE.search(str(guarded_text or ""))
            ):
                fallback = self._grounded_system_fallback_response(
                    user_id,
                    user_text,
                    allow_actions=False,
                )
                if fallback is not None:
                    return fallback
                guarded_text = "I couldn't verify that from the current runtime state."
        if guarded_text == response.text:
            return response
        return OrchestratorResponse(guarded_text, response_data)

    @staticmethod
    def _mentioned_provider_id(text: str) -> str | None:
        normalized = normalize_setup_text(text)
        for provider_id in ("ollama", "openrouter", "openai"):
            if provider_id in normalized:
                return provider_id
        return None

    def _looks_like_grounded_system_query(self, text: str) -> bool:
        normalized = normalize_setup_text(text)
        if not normalized:
            return False
        normalized_space = normalized.replace("/", " ")
        if _looks_like_current_model_query(normalized):
            return True
        if _looks_like_model_availability_query(normalized):
            return True
        if _looks_like_model_lifecycle_query(normalized):
            return True
        if _looks_like_local_model_inventory_query(normalized):
            return True
        if _looks_like_runtime_status_query(normalized):
            return True
        if self._mentioned_provider_id(normalized) and any(
            token in normalized_space
            for token in (
                "status",
                "health",
                "configured",
                "working",
                "ready",
                "models",
                "model",
                "downloaded",
                "installed",
                "local",
            )
        ):
            return True
        if any(token in normalized_space for token in ("switch back", "go back", "revert that", "previous model")):
            return True
        if any(phrase in normalized_space for phrase in ("switch to ", "switch chat to ", "change to ", "use ")):
            if ":" in normalized_space or any(
                token in normalized_space for token in ("ollama", "openrouter", "openai", "model", "models")
            ):
                return True
        if any(token in normalized_space for token in ("model", "models")) and any(
            token in normalized_space
            for token in ("downloaded", "installed", "local", "available", "usable", "switch to")
        ):
            return True
        return False

    @staticmethod
    def _interpretation_followup_kind(text: str) -> str | None:
        normalized = str(text or "").strip()
        if not normalized:
            return None
        for kind, pattern in _INTERPRETATION_FOLLOWUP_PATTERNS:
            if pattern.search(normalized):
                return kind
        return None

    def _current_interpretable_result(self, user_id: str) -> dict[str, Any]:
        state = self._last_interpretable_result.get(user_id)
        if not isinstance(state, dict):
            return {}
        created_ts = int(state.get("created_ts") or 0)
        if created_ts and int(time.time()) - created_ts > _INTERPRETABLE_RESULT_TTL_SECONDS:
            self._last_interpretable_result.pop(user_id, None)
            return {}
        return dict(state)

    @staticmethod
    def _result_card_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
        cards = payload.get("cards") if isinstance(payload.get("cards"), list) else []
        return [dict(card) for card in cards if isinstance(card, dict)]

    def _remember_interpretable_result(
        self,
        *,
        user_id: str,
        user_text: str,
        response: OrchestratorResponse,
    ) -> None:
        response_data = self._response_data(response)
        route = str(response_data.get("route") or "").strip().lower()
        payload = response_data.get("runtime_payload") if isinstance(response_data.get("runtime_payload"), dict) else {}
        payload_type = str(payload.get("type") or "").strip().lower()
        remember_setup_flow = route == "setup_flow" and payload_type in {"provider_repair", "provider_repair_options"}
        if route not in {"operational_status", "runtime_status", "provider_status", "model_status"} and not remember_setup_flow and not (
            route == "action_tool" and payload_type in {"model_scout", "model_controller", "external_pack_knowledge"}
        ):
            return
        used_tools = [str(item).strip() for item in (response_data.get("used_tools") if isinstance(response_data.get("used_tools"), list) else []) if str(item).strip()]
        summary = str(payload.get("summary") or response.text or "").strip()
        if not summary:
            return
        self._last_interpretable_result[user_id] = {
            "created_ts": int(time.time()),
            "route": route,
            "kind": str(payload.get("kind") or payload.get("type") or "").strip() or None,
            "user_text": str(user_text or "").strip(),
            "response_text": str(response.text or "").strip(),
            "summary": summary,
            "payload": dict(payload),
            "used_tools": used_tools,
        }

    @staticmethod
    def _context_fact_lines(context: dict[str, Any]) -> list[str]:
        payload = context.get("payload") if isinstance(context.get("payload"), dict) else {}
        lines: list[str] = []
        summary = str(context.get("summary") or payload.get("summary") or "").strip()
        if summary:
            lines.append(f"Summary: {summary}")
        for card in Orchestrator._result_card_rows(payload)[:3]:
            title = str(card.get("title") or "").strip() or "Detail"
            card_lines = [
                str(item).strip()
                for item in (card.get("lines") if isinstance(card.get("lines"), list) else [])
                if str(item).strip()
            ][:4]
            if card_lines:
                lines.append(f"{title}: {' | '.join(card_lines)}")
        response_text = str(context.get("response_text") or "").strip()
        if response_text and not lines:
            first_block = response_text.split("\n\n", 1)[0].strip()
            if first_block:
                lines.append(first_block)
        return lines

    @staticmethod
    def _top_memory_items_from_context(context: dict[str, Any]) -> list[str]:
        payload = context.get("payload") if isinstance(context.get("payload"), dict) else {}
        candidates: list[str] = []
        for card in Orchestrator._result_card_rows(payload):
            for line in (card.get("lines") if isinstance(card.get("lines"), list) else []):
                text = str(line or "").strip()
                lowered = text.lower()
                if not text:
                    continue
                if "rss" in lowered or "memory" in lowered or "ram" in lowered or "gib" in lowered or "mib" in lowered:
                    candidates.append(text.lstrip("- ").strip())
        if candidates:
            return candidates[:3]
        response_text = str(context.get("response_text") or "").strip()
        lines = [line.strip().lstrip("- ").strip() for line in response_text.splitlines() if line.strip()]
        return [line for line in lines if any(token in line.lower() for token in ("rss", "memory", "ram", "gib", "mib"))][:3]

    @staticmethod
    def _memory_usage_pct_from_context(context: dict[str, Any]) -> float | None:
        payload = context.get("payload") if isinstance(context.get("payload"), dict) else {}
        candidate_lines: list[str] = []
        for card in Orchestrator._result_card_rows(payload):
            candidate_lines.extend(
                str(item).strip()
                for item in (card.get("lines") if isinstance(card.get("lines"), list) else [])
                if str(item).strip()
            )
        candidate_lines.append(str(context.get("summary") or "").strip())
        for line in candidate_lines:
            match = re.search(r"(\d+(?:\.\d+)?)%\s+of\s+(?:ram|memory)", line, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _memory_process_meaning(process_label: str) -> str:
        lowered = str(process_label or "").strip().lower()
        if any(token in lowered for token in ("chrome", "firefox", "browser", "edge", "brave")):
            return "That is likely your web browser, so heavy tabs, media, or extensions are the usual cause."
        if any(token in lowered for token in ("postgres", "mysql", "mariadb", "redis", "mongod")):
            return "That looks like a database process, which usually means cached data or active queries are holding memory."
        if any(token in lowered for token in ("python", "node", "java", "code", "electron")):
            return "That is likely an application runtime, so the memory use is usually coming from the app workload rather than the operating system itself."
        return "That looks like an application process rather than a low-level system component."

    @staticmethod
    def _parse_process_line(line: str) -> tuple[str, str] | None:
        text = str(line or "").strip().lstrip("- ").strip()
        if not text or ":" not in text:
            return None
        name, value = text.split(":", 1)
        process_name = str(name or "").strip()
        metric = str(value or "").strip()
        if not process_name or not metric:
            return None
        return process_name, metric

    @staticmethod
    def _interpretation_has_assessment(text: str) -> bool:
        lowered = str(text or "").strip().lower()
        return any(
            token in lowered
            for token in (
                "normal",
                "mildly high",
                "worth watching",
                "concerning",
                "not urgent",
                "pay attention",
                "should worry",
                "shouldn't worry",
                "not automatically an emergency",
            )
        )

    @staticmethod
    def _interpretation_has_meaning(text: str) -> bool:
        lowered = str(text or "").strip().lower()
        return any(
            token in lowered
            for token in (
                "likely",
                "usually",
                "means",
                "main driver",
                "coming from",
                "rather than",
            )
        )

    @staticmethod
    def _context_warns(context: dict[str, Any]) -> bool:
        payload = context.get("payload") if isinstance(context.get("payload"), dict) else {}
        for card in Orchestrator._result_card_rows(payload):
            severity = str(card.get("severity") or "").strip().lower()
            if severity in {"warn", "error", "critical"}:
                return True
        summary = str(context.get("summary") or "").strip().lower()
        return any(token in summary for token in ("warn", "high", "elevated", "pressure", "low free", "attention"))

    def _grounded_interpretation_summary(self, context: dict[str, Any], followup_kind: str) -> str:
        top_memory = [
            parsed
            for parsed in (
                self._parse_process_line(line)
                for line in self._top_memory_items_from_context(context)
            )
            if parsed is not None
        ]
        memory_pct = self._memory_usage_pct_from_context(context)
        summary = str(context.get("summary") or "I still have the last system result.").strip()
        assessment = "normal"
        if memory_pct is not None:
            if memory_pct >= 90:
                assessment = "concerning"
            elif memory_pct >= 80:
                assessment = "mildly high"
            else:
                assessment = "normal"
        elif self._context_warns(context):
            assessment = "mildly high"
        if top_memory:
            top_name, top_value = top_memory[0]
            compare = ""
            if len(top_memory) > 1:
                compare_name, compare_value = top_memory[1]
                compare = f" That is clearly above {compare_name} at {compare_value}, so it is the main driver of the current memory load."
            meaning = self._memory_process_meaning(top_name)
            if followup_kind == "top_memory":
                action = ""
                if assessment in {"mildly high", "concerning"}:
                    action = f" If the machine feels slow, the single most useful next step is to trim the workload in {top_name} first."
                return (
                    f"The biggest memory user right now is {top_name} at {top_value}.{compare} "
                    f"{meaning} I would describe this as {assessment}.{action}"
                )
            if followup_kind == "concern":
                action = ""
                if assessment in {"mildly high", "concerning"}:
                    action = f" If you want to act on one thing, start with {top_name}."
                return (
                    f"The main issue is that {top_name} is using {top_value}, which is driving most of the memory pressure.{compare} "
                    f"{meaning} Based on this snapshot, I would call that {assessment}, not automatically a crisis.{action}"
                )
            if followup_kind == "action":
                return (
                    f"The key point is that {top_name} is the main memory consumer at {top_value}.{compare} "
                    f"{meaning} I would describe the current state as {assessment}. "
                    f"If you take one action, start with {top_name}."
                )
            return (
                f"The important part is that {top_name} is using {top_value}, so most of the pressure is concentrated there rather than spread evenly across the system."
                f"{compare} {meaning} I would describe the current state as {assessment}."
            )
        if followup_kind == "action":
            return (
                f"The important part is: {summary} I would describe the current state as {assessment}. "
                "If you take one action, focus on the largest workload mentioned in the report."
            )
        return f"The important part is: {summary} I would describe the current state as {assessment}."

    def _fallback_interpretation_summary(self, context: dict[str, Any], followup_kind: str) -> str:
        grounded = self._grounded_interpretation_summary(context, followup_kind)
        return f"I can still tell you the basics from the data I already gathered. {grounded}"

    def _interpretation_response_needs_fallback(self, text: str, context: dict[str, Any]) -> bool:
        normalized = str(text or "").strip()
        if not normalized:
            return True
        lowered = normalized.lower()
        if _INTERPRETATION_DEBUG_REFLEX_RE.search(normalized) or _INTERPRETATION_SHELL_SNIPPET_RE.search(normalized):
            return True
        if "*resource report*" in lowered or "*storage report*" in lowered or "memory snapshot ready" in lowered:
            return True
        top_memory_lines = [line.lower() for line in self._top_memory_items_from_context(context)]
        if top_memory_lines:
            raw_matches = sum(1 for line in top_memory_lines if line and line in lowered)
            if raw_matches >= min(2, len(top_memory_lines)) and not self._interpretation_has_meaning(normalized):
                return True
        if not self._interpretation_has_assessment(normalized):
            return True
        if not self._interpretation_has_meaning(normalized):
            return True
        return False

    def _explanation_chat_target(self) -> tuple[str | None, str | None]:
        truth = self._runtime_truth()
        configured_provider: str | None = None
        configured_model: str | None = None
        if truth is not None:
            try:
                target_truth = (
                    truth.chat_target_truth()
                    if callable(getattr(truth, "chat_target_truth", None))
                    else {}
                )
            except Exception:
                target_truth = {}
            configured_provider = str(target_truth.get("effective_provider") or target_truth.get("configured_provider") or "").strip().lower() or None
            configured_model = str(target_truth.get("effective_model") or target_truth.get("configured_model") or "").strip() or None
        adapter = self._chat_runtime_adapter
        registry = adapter.registry_document if isinstance(getattr(adapter, "registry_document", None), dict) else {}
        models_doc = registry.get("models") if isinstance(registry.get("models"), dict) else {}
        providers_doc = registry.get("providers") if isinstance(registry.get("providers"), dict) else {}
        health_monitor = getattr(adapter, "_health_monitor", None)
        health_state = health_monitor.state if isinstance(getattr(health_monitor, "state", None), dict) else {}
        provider_health = health_state.get("providers") if isinstance(health_state.get("providers"), dict) else {}
        model_health = health_state.get("models") if isinstance(health_state.get("models"), dict) else {}
        current_rank = 0
        if configured_model and isinstance(models_doc.get(configured_model), dict):
            current_rank = int(models_doc[configured_model].get("quality_rank") or 0)
        candidates: list[tuple[int, str, str]] = []
        for model_id, model_payload in models_doc.items():
            if not isinstance(model_payload, dict):
                continue
            provider_id = str(model_payload.get("provider") or "").strip().lower()
            if not provider_id:
                continue
            provider_payload = providers_doc.get(provider_id) if isinstance(providers_doc.get(provider_id), dict) else {}
            if not bool(provider_payload.get("local", provider_id == "ollama")):
                continue
            capabilities = {
                str(item).strip().lower()
                for item in (model_payload.get("capabilities") if isinstance(model_payload.get("capabilities"), list) else [])
                if str(item).strip()
            }
            if "chat" not in capabilities:
                continue
            if not bool(model_payload.get("enabled", True)) or not bool(model_payload.get("available", True)):
                continue
            provider_status = str((provider_health.get(provider_id) if isinstance(provider_health.get(provider_id), dict) else {}).get("status") or "unknown").strip().lower()
            model_status = str((model_health.get(model_id) if isinstance(model_health.get(model_id), dict) else {}).get("status") or "unknown").strip().lower()
            if provider_status not in {"", "ok", "unknown"} or model_status not in {"", "ok", "unknown"}:
                continue
            quality_rank = int(model_payload.get("quality_rank") or 0)
            candidates.append((quality_rank, provider_id, str(model_id)))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        for quality_rank, provider_id, model_id in candidates:
            if configured_model and model_id == configured_model:
                continue
            if quality_rank > current_rank:
                return provider_id, model_id
        return configured_provider, configured_model

    def _interpret_previous_result_followup(
        self,
        user_id: str,
        text: str,
        *,
        chat_context: dict[str, Any] | None = None,
    ) -> OrchestratorResponse | None:
        followup_kind = self._interpretation_followup_kind(text)
        if not followup_kind:
            return None
        context = self._current_interpretable_result(user_id)
        if not context:
            return None
        fact_lines = self._context_fact_lines(context)
        fallback_text = self._fallback_interpretation_summary(context, followup_kind)
        if not self._llm_chat_available():
            return self._merge_response_data(
                OrchestratorResponse(fallback_text),
                route="interpretation_followup",
                used_runtime_state=False,
                used_llm=False,
                used_memory=True,
                used_tools=[],
                ok=True,
            )
        provider_override, model_override = self._explanation_chat_target()
        source_surface = str((chat_context or {}).get("source_surface") or "api").strip().lower() or "api"
        channel = self._chat_channel(source_surface)
        trace_id = self._trace_id("interp")
        prompt_facts = "\n".join(f"- {line}" for line in fact_lines) or f"- {fallback_text}"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Personal Agent, a local-first assistant.\n"
                    "Interpret the prior tool-backed result in plain English.\n"
                    "Use only the supplied facts. Do not invent values. If the facts are insufficient, say that clearly.\n"
                    "Do not identify yourself as a model or vendor.\n"
                    "Do not suggest shell commands, rerunning tools, or generic Linux debugging steps.\n"
                    "Assume the existing data is enough for this answer.\n"
                    "Structure the answer as: key finding, brief comparison, plain-English meaning, light assessment, and at most one gentle next action if useful.\n"
                    "Answer directly and briefly."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Previous request: {str(context.get('user_text') or 'unknown').strip()}\n"
                    f"Previous result summary: {str(context.get('summary') or '').strip()}\n"
                    f"Known facts:\n{prompt_facts}\n\n"
                    f"Follow-up request: {str(text or '').strip()}"
                ),
            },
        ]
        try:
            result = route_inference(
                messages=messages,
                user_text=str(text or "").strip(),
                purpose="chat",
                task_hint="interpret previous tool result",
                trace_id=trace_id,
                provider_override=provider_override,
                model_override=model_override,
                metadata={
                    "trace_id": trace_id,
                    "source_surface": source_surface,
                    "channel": channel,
                    "interpretation_followup": True,
                },
            )
        except Exception:
            result = {"ok": False, "error_kind": "interpretation_exception"}
        if not bool(result.get("ok")):
            return self._merge_response_data(
                OrchestratorResponse(fallback_text),
                route="interpretation_followup",
                used_runtime_state=False,
                used_llm=False,
                used_memory=True,
                used_tools=[],
                ok=True,
                error_kind=str(result.get("error_kind") or "").strip() or None,
            )
        explanation_text = str(result.get("text") or "").strip()
        if not explanation_text or self._interpretation_response_needs_fallback(explanation_text, context):
            explanation_text = fallback_text
        return self._merge_response_data(
            OrchestratorResponse(explanation_text),
            route="interpretation_followup",
            used_runtime_state=False,
            used_llm=True,
            used_memory=True,
            used_tools=[],
            ok=True,
            provider=str(result.get("provider") or provider_override or "").strip() or None,
            model=str(result.get("model") or model_override or "").strip() or None,
        )

    @staticmethod
    def _deep_system_followup_requested(text: str) -> bool:
        normalized = str(text or "").strip()
        if not normalized:
            return False
        return any(pattern.search(normalized) for pattern in _DEEP_SYSTEM_FOLLOWUP_PATTERNS)

    @staticmethod
    def _is_machine_observe_context(context: dict[str, Any]) -> bool:
        if not isinstance(context, dict):
            return False
        if str(context.get("route") or "").strip().lower() != "operational_status":
            return False
        payload = context.get("payload") if isinstance(context.get("payload"), dict) else {}
        kind = str(payload.get("kind") or "").strip().lower()
        used_tools = {
            str(item).strip().lower()
            for item in (context.get("used_tools") if isinstance(context.get("used_tools"), list) else [])
            if str(item).strip()
        }
        if used_tools & {"hardware_report", "resource_report", "storage_report", "disk_pressure_report", "observe_system_health"}:
            return True
        if kind in {"observe_pc", "observe_system_health", "deep_system_observe"}:
            return True
        user_text = str(context.get("user_text") or "").strip().lower()
        return any(token in user_text for token in ("pc", "cpu", "gpu", "ram", "storage", "machine", "system"))

    def _deep_system_followup_response(self, user_id: str, text: str) -> OrchestratorResponse | None:
        if not self._deep_system_followup_requested(text):
            return None
        context = self._current_interpretable_result(user_id)
        if not self._is_machine_observe_context(context):
            return None
        decision = {
            "intent": "OBSERVE_PC",
            "skills": [
                {"skill": "hardware_report", "function": "hardware_report"},
                {"skill": "resource_governor", "function": "resource_report"},
                {"skill": "storage_governor", "function": "storage_report"},
            ],
        }
        response = self._handle_nl_observe(user_id, text, decision)
        payload = dict(response.data) if isinstance(response.data, dict) else {}
        summary = str(payload.get("summary") or response.text or "").strip() or "Deeper system inspection is ready."
        return self._runtime_truth_response(
            text=str(response.text or "").strip() or summary,
            route="operational_status",
            used_runtime_state=False,
            used_tools=["hardware_report", "resource_report", "storage_report"],
            payload={
                "type": "operational_status",
                "kind": "deep_system_observe",
                "summary": summary,
                **payload,
            },
        )

    @staticmethod
    def _time_date_intent_kind(text: str) -> str | None:
        normalized = normalize_setup_text(text).replace("/", " ")
        if not normalized:
            return None
        time_phrases = (
            "what time is it",
            "what time is it right now",
            "what s the current time",
            "what is the current time",
            "what time is it here",
            "current time",
        )
        date_phrases = (
            "what day is it",
            "what day is it today",
            "what s today s date",
            "what is today s date",
            "what s the date",
            "what is the date",
            "today s date",
        )
        if any(phrase in normalized for phrase in time_phrases):
            return "local_time"
        if any(phrase in normalized for phrase in date_phrases):
            return "local_date"
        if "time" in normalized and "timeframe" not in normalized:
            if any(phrase in normalized for phrase in ("check the time", "tell me the time", "the time right now", "time right now")):
                return "local_time"
            if "current time" in normalized:
                return "local_time"
        if "date" in normalized or "day" in normalized:
            if any(phrase in normalized for phrase in ("today s date", "today date", "day is it today")):
                return "local_date"
        return None

    def _assistant_local_now(self) -> datetime:
        timezone_name = str(self.timezone or "UTC").strip() or "UTC"
        try:
            tzinfo = ZoneInfo(timezone_name)
        except Exception:
            tzinfo = timezone.utc
        return datetime.now(tzinfo)

    @staticmethod
    def _format_local_clock(now: datetime) -> str:
        hour = int(now.strftime("%I") or "0")
        minute = now.strftime("%M")
        suffix = now.strftime("%p")
        return f"{hour}:{minute} {suffix}"

    def _local_time_response(self, kind: str) -> OrchestratorResponse:
        now = self._assistant_local_now()
        timezone_label = (
            str(getattr(now.tzinfo, "key", "") or "").strip()
            or str(now.tzname() or "").strip()
            or str(self.timezone or "").strip()
            or "local time"
        )
        date_label = f"{now.strftime('%A')}, {now.strftime('%B')} {now.day}, {now.year}"
        if str(kind or "").strip().lower() == "local_date":
            message = f"Today is {date_label} in {timezone_label}."
        else:
            message = f"It's {self._format_local_clock(now)} in {timezone_label} on {date_label}."
        return self._runtime_truth_response(
            text=message,
            route="action_tool",
            used_runtime_state=False,
            used_tools=["local_time"],
            payload={
                "type": "local_time",
                "kind": str(kind or "local_time").strip().lower() or "local_time",
                "summary": message,
                "timezone": timezone_label,
                "timestamp": now.isoformat(),
            },
        )

    @staticmethod
    def _is_model_context(context: dict[str, Any]) -> bool:
        if not isinstance(context, dict):
            return False
        route = str(context.get("route") or "").strip().lower()
        payload = context.get("payload") if isinstance(context.get("payload"), dict) else {}
        payload_type = str(payload.get("type") or "").strip().lower()
        if route == "model_status":
            return True
        return route == "action_tool" and payload_type in {"model_scout", "model_controller"}

    @staticmethod
    def _model_ready_now_requested(text: str) -> bool:
        normalized = normalize_setup_text(text).replace("/", " ")
        if not normalized:
            return False
        if any(phrase in normalized for phrase in _MODEL_READY_NOW_PHRASES):
            return True
        return bool(
            any(token in normalized for token in ("model", "models"))
            and "ready" in normalized
            and "right now" in normalized
        )

    @staticmethod
    def _model_controller_test_requested(text: str) -> bool:
        normalized = normalize_setup_text(text).replace("/", " ")
        if not normalized:
            return False
        if any(phrase in normalized for phrase in _MODEL_CONTROLLER_TEST_PHRASES):
            return True
        has_explicit_target = _DIRECT_MODEL_SWITCH_TOKEN_RE.search(normalized) is not None
        return bool(
            "test" in normalized
            and any(token in normalized for token in ("without adopting", "without switching", "without using"))
            and ("model" in normalized or has_explicit_target)
        )

    @staticmethod
    def _model_controller_trial_switch_requested(text: str) -> bool:
        normalized = normalize_setup_text(text).replace("/", " ")
        if not normalized:
            return False
        if any(phrase in normalized for phrase in _MODEL_CONTROLLER_TRIAL_SWITCH_PHRASES):
            return True
        has_explicit_target = _DIRECT_MODEL_SWITCH_TOKEN_RE.search(normalized) is not None
        return bool(
            "temporarily" in normalized
            and any(token in normalized for token in ("switch", "use", "try"))
            and ("model" in normalized or has_explicit_target)
        )

    @staticmethod
    def _model_controller_promote_requested(text: str) -> bool:
        normalized = normalize_setup_text(text).replace("/", " ")
        if not normalized:
            return False
        if any(phrase in normalized for phrase in _MODEL_CONTROLLER_PROMOTE_PHRASES):
            return True
        return bool(
            normalized.startswith("make ")
            and " default" in normalized
            and (_DIRECT_MODEL_SWITCH_TOKEN_RE.search(normalized) is not None or "model" in normalized)
        )

    @staticmethod
    def _parse_control_mode_intent(text: str) -> dict[str, str] | None:
        normalized = normalize_setup_text(text).replace("/", " ")
        if not normalized:
            return None
        mentions_mode = bool(re.search(r"\bmode\b", normalized)) or any(
            re.search(rf"\b{token}\b", normalized) is not None
            for token in ("controlled", "safe", "baseline")
        )
        if not mentions_mode:
            return None
        if re.search(r"^\s*(what|which|why|how)\b", normalized):
            return {"kind": "get_mode"}
        if re.search(r"^\s*(are|is)\s+(we|you)\b", normalized):
            return {"kind": "get_mode"}
        if re.search(r"^\s*should\s+i\b", normalized):
            return {"kind": "get_mode"}
        if re.search(r"^\s*(tell me about|explain|can you explain)\b", normalized):
            return {"kind": "get_mode"}
        if re.search(r"\bwhat does (?:this|your) mode allow\b", normalized):
            return {"kind": "get_mode"}
        if re.search(r"\bwhat (?:requires|would need) my approval\b", normalized):
            return {"kind": "get_mode"}

        request_prefix = r"^\s*(?:please\s+)?(?:(?:can|could|would)\s+you\s+|let'?s\s+)?"

        def _matches(pattern: str) -> bool:
            return re.search(pattern, normalized) is not None

        if _matches(request_prefix + r"(?:exit|leave|disable|turn off|stop using)\b.*\bcontrolled mode\b"):
            return {"kind": "set_mode", "mode": "baseline"}
        if _matches(request_prefix + r"(?:return|go back|switch|use|follow|revert)\b.*\bbaseline(?: mode)?\b"):
            return {"kind": "set_mode", "mode": "baseline"}
        if _matches(request_prefix + r"(?:switch|go|enter|return|turn|enable|use)\b.*\bsafe mode\b"):
            return {"kind": "set_mode", "mode": "safe"}
        if _matches(request_prefix + r"(?:switch|go|enter|put|set|turn|enable|use)\b.*\bcontrolled mode\b"):
            return {"kind": "set_mode", "mode": "controlled"}
        return None

    def _control_mode_intent_response(self, text: str) -> OrchestratorResponse | None:
        intent = self._parse_control_mode_intent(text)
        if not isinstance(intent, dict):
            return None
        kind = str(intent.get("kind") or "").strip().lower()
        if kind == "get_mode":
            return self._model_controller_policy_response()
        if kind == "set_mode":
            requested_mode = str(intent.get("mode") or "").strip().lower()
            if requested_mode in {"safe", "controlled", "baseline"}:
                return self._control_mode_change_response(requested_mode)
        return None

    @staticmethod
    def _repair_followup_requested(text: str) -> bool:
        normalized = normalize_setup_text(text).replace("/", " ")
        if not normalized:
            return False
        if any(phrase in normalized for phrase in _RUNTIME_REPAIR_ACTION_PHRASES):
            return True
        return bool(
            any(token in normalized for token in ("repair", "fix", "working"))
            and any(token in normalized for token in ("ollama", "openrouter", "provider", "model", "that", "it"))
        )

    @staticmethod
    def _repair_context_handoff_requested(text: str) -> bool:
        normalized = normalize_setup_text(text).replace("/", " ")
        if not normalized:
            return False
        if Orchestrator._repair_followup_requested(text):
            return True
        if normalized in {"needs attention", "what needs attention"}:
            return True
        return _looks_like_setup_explanation_query(normalized)

    def _repair_option_choice_response(self, user_id: str, text: str) -> OrchestratorResponse | None:
        normalized = normalize_setup_text(text).replace("/", " ")
        if normalized not in {"1", "2"}:
            return None
        context = self._current_interpretable_result(user_id)
        if not isinstance(context, dict) or not context:
            return None
        if str(context.get("route") or "").strip().lower() != "setup_flow":
            return None
        payload = context.get("payload") if isinstance(context.get("payload"), dict) else {}
        if str(payload.get("type") or "").strip().lower() != "provider_repair_options":
            return None
        if normalized == "1":
            return self._repair_followup_response(user_id, "repair it")
        option_kind = str(payload.get("option_2_kind") or "").strip().lower()
        if option_kind == "switch_back":
            return self._model_controller_switch_back_response(user_id)
        if option_kind != "switch_model":
            return None
        target_model = str(payload.get("option_2_model_id") or "").strip() or None
        target_provider = str(payload.get("option_2_provider") or "").strip().lower() or None
        if not target_model:
            return None
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="setup_flow",
                used_memory=True,
                reason="runtime_truth_service_unavailable",
            )
        current_target = truth.current_chat_target_status()
        previous_provider, previous_model = self._target_snapshot_from_truth(current_target)
        prompt = f"I can switch chat to {target_model} now. Do you want me to do that?"
        self._save_runtime_setup_state(
            user_id,
            {
                "step": "awaiting_switch_confirm",
                "action_type": "confirm_model_switch",
                "provider": target_provider,
                "model_id": target_model,
                "previous_provider": previous_provider,
                "previous_model": previous_model,
            },
        )
        return self._runtime_truth_response(
            text=prompt,
            route="setup_flow",
            used_memory=True,
            next_question=prompt,
            payload={
                "type": "confirm_switch_model",
                "provider": target_provider,
                "model_id": target_model,
                "title": "Use this model for chat?",
                "prompt": prompt,
                "approve_label": "Switch model",
                "approve_command": "yes",
                "cancel_label": "Keep current",
                "cancel_command": "no",
                "summary": prompt,
            },
        )

    def _recent_unhealthy_runtime_context(self, user_id: str) -> dict[str, Any]:
        context = self._current_interpretable_result(user_id)
        if not isinstance(context, dict) or not context:
            return {}
        route = str(context.get("route") or "").strip().lower()
        if route not in {"model_status", "provider_status", "runtime_status", "setup_flow"}:
            return {}
        payload = context.get("payload") if isinstance(context.get("payload"), dict) else {}
        provider = (
            str(payload.get("provider") or payload.get("configured_provider") or "").strip().lower()
            or None
        )
        model_id = (
            str(payload.get("model_id") or payload.get("configured_model") or "").strip()
            or None
        )
        provider_health_status = (
            str(payload.get("provider_health_status") or "").strip().lower()
            or (
                str(payload.get("health_status") or "").strip().lower()
                if route == "provider_status"
                else None
            )
        )
        model_health_status = (
            str(payload.get("model_health_status") or "").strip().lower()
            or (
                str(payload.get("health_status") or "").strip().lower()
                if route == "model_status"
                else None
            )
        )
        if provider_health_status not in {"down", "degraded"} and model_health_status not in {"down", "degraded"}:
            return {}
        return {
            "route": route,
            "provider": provider,
            "model_id": model_id,
            "provider_health_status": provider_health_status,
            "model_health_status": model_health_status,
        }

    def assistant_followup_hint(self, user_id: str, text: str) -> dict[str, Any]:
        thread_id = self._active_thread_id_for_user(user_id)
        followup = self._memory_runtime.resolve_followup(user_id, text, thread_id)
        followup_type = str(followup.get("type") or "").strip().lower()
        if followup_type in {"match", "ambiguous", "expired"}:
            pending_item = followup.get("pending_item") if isinstance(followup.get("pending_item"), dict) else {}
            pending_kind = str(pending_item.get("kind") or "").strip().lower() or None
            return {
                "kind": f"pending_{pending_kind or 'followup'}",
                "followup_type": followup_type,
                "followup_intent": str(followup.get("intent") or "").strip().lower() or None,
                "pending_kind": pending_kind,
            }
        normalized = normalize_setup_text(text).replace("/", " ")
        if self._model_trial_switch_back_requested(normalized):
            return {"kind": "model_switch_back"}
        if self._repair_context_handoff_requested(text):
            context = self._recent_unhealthy_runtime_context(user_id)
            if context:
                return {
                    "kind": "runtime_repair_followup",
                    **context,
                }
        return {}

    def _repair_context_handoff_response(self, user_id: str, text: str) -> OrchestratorResponse | None:
        if not self._repair_context_handoff_requested(text):
            return None
        context = self._recent_unhealthy_runtime_context(user_id)
        if not context:
            return None
        if self._repair_followup_requested(text):
            return self._repair_followup_response(user_id, text)
        provider_hint = str(context.get("provider") or "").strip().lower()
        synthetic_text = f"repair {provider_hint}" if provider_hint else "repair it"
        return self._repair_followup_response(user_id, synthetic_text)

    @staticmethod
    def _model_context_terms(context: dict[str, Any]) -> list[str]:
        if not isinstance(context, dict):
            return []
        payload = context.get("payload") if isinstance(context.get("payload"), dict) else {}
        rows: list[dict[str, Any]] = []
        for key in (
            "usable_models",
            "other_usable_models",
            "ready_now_models",
            "other_ready_now_models",
            "local_installed_models",
            "not_ready_models",
            "suggestions",
            "better_candidates",
            "candidate_rows",
        ):
            value = payload.get(key) if isinstance(payload.get(key), list) else []
            rows.extend(row for row in value if isinstance(row, dict))
        raw_terms = [
            str(payload.get("active_model") or "").strip(),
            str(payload.get("configured_model") or "").strip(),
            str(payload.get("model_id") or "").strip(),
            str(payload.get("effective_model_id") or "").strip(),
        ]
        for row in rows:
            raw_terms.extend(
                [
                    str(row.get("model_id") or "").strip(),
                    str(row.get("repo_id") or "").strip(),
                    str(row.get("provider_id") or "").strip(),
                ]
            )
        seen: set[str] = set()
        terms: list[str] = []
        for item in raw_terms:
            candidate = str(item or "").strip().lower()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            terms.append(candidate)
        return terms[:12]

    @staticmethod
    def _model_scout_focus_terms(text: str, context: dict[str, Any] | None = None) -> list[str]:
        normalized = normalize_setup_text(text).replace("/", " ")
        terms: list[str] = []
        for match in re.finditer(r"\b([a-z0-9._:-]{3,})\s+models?\b", normalized):
            candidate = str(match.group(1) or "").strip().lower()
            if candidate and candidate not in _MODEL_SCOUT_TERM_STOPWORDS:
                terms.append(candidate)
        if not terms and any(token in normalized for token in ("them", "those", "they")) and isinstance(context, dict):
            terms.extend(Orchestrator._model_context_terms(context))
        seen: set[str] = set()
        deduped: list[str] = []
        for term in terms:
            normalized_term = str(term or "").strip().lower()
            if not normalized_term or normalized_term in seen:
                continue
            seen.add(normalized_term)
            deduped.append(normalized_term)
        return deduped[:12]

    @staticmethod
    def _preferred_model_context_target(context: dict[str, Any]) -> tuple[str | None, str | None]:
        if not isinstance(context, dict):
            return None, None
        payload = context.get("payload") if isinstance(context.get("payload"), dict) else {}

        preferred_rows: list[dict[str, Any]] = []
        for key in ("recommended_candidate",):
            row = payload.get(key)
            if isinstance(row, dict):
                preferred_rows.append(dict(row))
        for key in ("better_candidates", "candidate_rows", "ready_now_models", "other_ready_now_models", "usable_models", "other_usable_models"):
            rows = payload.get(key) if isinstance(payload.get(key), list) else []
            preferred_rows.extend(dict(row) for row in rows if isinstance(row, dict))

        seen: set[str] = set()
        for row in preferred_rows:
            model_id = str(row.get("model_id") or "").strip() or None
            provider_id = str(row.get("provider_id") or "").strip().lower() or None
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            if bool(row.get("active", False)):
                continue
            return model_id, provider_id

        direct_model = str(payload.get("model_id") or payload.get("active_model") or "").strip() or None
        direct_provider = str(payload.get("provider") or payload.get("active_provider") or "").strip().lower() or None
        return direct_model, direct_provider

    @staticmethod
    def _model_scout_row_search_text(row: dict[str, Any]) -> str:
        fields = (
            row.get("id"),
            row.get("kind"),
            row.get("repo_id"),
            row.get("provider_id"),
            row.get("model_id"),
            row.get("rationale"),
        )
        return " ".join(str(value or "").strip().lower() for value in fields if str(value or "").strip())

    @staticmethod
    def _model_scout_row_label(row: dict[str, Any]) -> str:
        return (
            str(row.get("model_id") or "").strip()
            or str(row.get("repo_id") or "").strip()
            or str(row.get("id") or "").strip()
            or "model"
        )

    def _model_scout_inventory_response(self, *, focus_terms: list[str]) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="action_tool",
                reason="runtime_truth_service_unavailable",
            )
        payload = self._canonical_model_inventory_snapshot(truth)
        rows = [
            dict(row)
            for row in (payload.get("models") if isinstance(payload.get("models"), list) else [])
            if isinstance(row, dict)
        ]
        active_model = str(payload.get("active_model") or "").strip() or None
        active_provider = str(payload.get("active_provider") or "").strip().lower() or None
        installed_rows = [dict(row) for row in rows if bool(row.get("available", False))]
        usable_rows = [dict(row) for row in rows if bool(row.get("usable_now", False))]
        suggested_rows = [dict(row) for row in usable_rows if not bool(row.get("active", False))][:2]
        not_ready_rows = [dict(row) for row in rows if not bool(row.get("usable_now", False))][:2]

        visible_rows = rows
        if focus_terms:
            visible_rows = [
                row
                for row in rows
                if any(term in self._model_scout_row_search_text(row) for term in focus_terms)
            ]
        if not rows:
            message = (
                "I do not currently see any chat-capable models in the runtime registry. "
                "If you want, I can help you configure a provider or add a local model."
            )
        elif focus_terms and not visible_rows:
            focus_label = ", ".join(term for term in focus_terms[:3] if term)
            active_label = active_model or "the current chat target"
            message = (
                f"I do not currently see any installed or registered chat models matching {focus_label}. "
                f"Right now chat is using {active_label}. If you want, I can still list the models that are available now."
            )
        else:
            visible_installed = [dict(row) for row in visible_rows if bool(row.get("available", False))]
            visible_usable = [dict(row) for row in visible_rows if bool(row.get("usable_now", False))]
            labels = [self._model_scout_row_label(row) for row in visible_installed[:4]]
            suggested_labels = [self._model_scout_row_label(row) for row in suggested_rows if not focus_terms or row in visible_rows]
            active_label = active_model or "no active chat model"
            if focus_terms:
                focus_label = ", ".join(term for term in focus_terms[:3] if term)
                if visible_usable:
                    message_parts = [
                        f"For {focus_label}, I can currently see {', '.join(labels[:3])}. "
                        f"The best usable match right now is {self._model_scout_row_label(visible_usable[0])}."
                    ]
                    if active_label and active_label not in labels:
                        message_parts.append(f"Right now chat is using {active_label}.")
                    alternative_labels = [
                        self._model_scout_row_label(row)
                        for row in suggested_rows
                        if self._model_scout_row_label(row) not in labels and self._model_scout_row_label(row) != active_label
                    ][:2]
                    if alternative_labels:
                        message_parts.append(
                            f"Other usable models I can see right now are {', '.join(alternative_labels)}."
                        )
                    message = " ".join(message_parts)
                else:
                    reasons = ", ".join(
                        f"{self._model_scout_row_label(row)} ({str(row.get('availability_reason') or '').strip()})"
                        for row in visible_rows[:2]
                    )
                    message = f"For {focus_label}, I can see {reasons}."
                    if active_label:
                        message = f"{message} Right now chat is using {active_label}."
            elif labels:
                message = f"Right now chat is using {active_label}. Other chat models I can currently see are {', '.join(labels)}."
                if suggested_labels:
                    message = f"{message} The most useful alternatives right now are {', '.join(suggested_labels[:2])}."
            else:
                message = f"Right now chat is using {active_label}, but I do not see any other installed chat models."
            if not focus_terms and not_ready_rows:
                not_ready = ", ".join(
                    f"{self._model_scout_row_label(row)} ({str(row.get('availability_reason') or '').strip()})"
                    for row in not_ready_rows
                )
                if not_ready:
                    message = f"{message} I can also see models that are present but not ready yet: {not_ready}."

        return self._runtime_truth_response(
            text=message,
            route="action_tool",
            used_runtime_state=True,
            used_tools=["model_scout"],
            payload={
                "type": "model_scout",
                "summary": message,
                "focus_terms": focus_terms,
                "active_model": active_model,
                "active_provider": active_provider,
                "installed_models": installed_rows,
                "usable_models": usable_rows,
                "suggested_models": suggested_rows,
                "source": "runtime_truth.model_inventory_status+model_readiness_status",
            },
        )

    @staticmethod
    def _model_scout_strategy_requested(normalized_text: str) -> bool:
        normalized = str(normalized_text or "").strip().lower()
        if not normalized:
            return False
        recommendation_hints = (
            "should i use",
            "should we use",
            "would you use",
            "would you choose",
            "do you recommend",
            "recommend",
        )
        if any(phrase in normalized for phrase in _MODEL_SCOUT_STRATEGY_PHRASES):
            return True
        if any(
            phrase in normalized
            for phrase in ("premium model", "use premium", "best model", "upgrade model", "stronger model")
        ):
            return True
        if Orchestrator._model_scout_remote_role_requested(normalized):
            return bool(
                re.search(r"\bmodels?\b", normalized)
                and any(phrase in normalized for phrase in recommendation_hints)
            )
        if re.search(r"\bmodels?\b", normalized) and any(phrase in normalized for phrase in recommendation_hints):
            if any(token in normalized for token in ("coding", "code", "debug", "refactor", "review")):
                return True
            if any(token in normalized for token in ("research", "reasoning", "analysis", "analyze")):
                return True
        if re.search(r"\blocal models?\b", normalized) and any(
            phrase in normalized
            for phrase in (
                "recommend",
                "do you recommend",
                "should i use",
                "should we use",
                "would you choose",
            )
        ):
            return True
        if "better model" in normalized and any(token in normalized for token in ("use", "switch", "should", "try")):
            return True
        return False

    @staticmethod
    def _model_scout_remote_role_requested(normalized_text: str) -> str | None:
        normalized = str(normalized_text or "").strip().lower().replace("-", " ")
        if not normalized:
            return None
        if re.search(r"\blocal models?\b", normalized) and any(
            phrase in normalized
            for phrase in (
                "recommend",
                "do you recommend",
                "should i use",
                "should we use",
                "would you choose",
            )
        ):
            return "best_local"
        if any(phrase in normalized for phrase in _MODEL_SCOUT_REMOTE_ROLE_PHRASES):
            if "coding" in normalized:
                return "premium_coding" if "premium" in normalized else "cheap_cloud"
            if "research" in normalized or "reasoning" in normalized or "analysis" in normalized:
                return "premium_research" if "premium" in normalized else "cheap_cloud"
            if "premium" in normalized:
                return "premium_general"
            return "cheap_cloud"
        return None

    @staticmethod
    def _model_scout_discovery_requested(normalized_text: str) -> bool:
        normalized = str(normalized_text or "").strip().lower()
        if not normalized:
            return False
        if any(phrase in normalized for phrase in _MODEL_SCOUT_DISCOVERY_PHRASES):
            return True
        if ("hugging face" in normalized or "huggingface" in normalized) and (
            "model" in normalized or "models" in normalized
        ):
            return True
        if any(token in normalized for token in ("smol", "small", "tiny", "lightweight")) and (
            "model" in normalized or "models" in normalized
        ):
            return True
        if any(token in normalized for token in ("find", "look", "search", "discover", "show")) and (
            "new model" in normalized
            or "new models" in normalized
            or "downloadable model" in normalized
            or "downloadable models" in normalized
        ):
            return True
        return "download" in normalized and "model" in normalized

    @staticmethod
    def _model_trial_switch_back_requested(normalized_text: str) -> bool:
        normalized = str(normalized_text or "").strip().lower()
        return any(phrase in normalized for phrase in _MODEL_SCOUT_SWITCH_BACK_PHRASES)

    @staticmethod
    def _model_switch_advisory_requested(normalized_text: str) -> bool:
        normalized = str(normalized_text or "").strip().lower().replace("/", " ")
        return bool(re.search(r"\bshould (?:i|we) switch models?\b", normalized))

    @staticmethod
    def _model_acquisition_requested(normalized_text: str) -> bool:
        normalized = str(normalized_text or "").strip().lower()
        if not normalized:
            return False
        if normalized.startswith("please "):
            normalized = normalized[len("please "):].strip()
        if any(
            normalized.startswith(prefix)
            for prefix in ("install ", "download ", "pull ", "acquire ", "import ")
        ):
            return True
        return any(
            phrase in normalized
            for phrase in (
                "install it first",
                "download it first",
                "pull it first",
                "acquire it first",
                "import it first",
                "install this model",
                "download this model",
                "pull this model",
                "acquire this model",
                "import this model",
            )
        )

    @staticmethod
    def _model_acquisition_candidate(
        rows: list[dict[str, Any]],
        *,
        exclude_model: str | None = None,
    ) -> dict[str, Any] | None:
        state_priority = {
            "acquirable": 0,
            "installed_not_ready": 1,
            "queued": 2,
            "downloading": 3,
            "blocked_by_policy": 4,
        }
        candidates = [
            dict(row)
            for row in rows
            if isinstance(row, dict)
            and str(row.get("model_id") or "").strip()
            and str(row.get("model_id") or "").strip() != str(exclude_model or "").strip()
            and str(row.get("acquisition_state") or "").strip().lower() in state_priority
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda row: (
                state_priority.get(str(row.get("acquisition_state") or "").strip().lower(), 99),
                0 if bool(row.get("local", False)) else 1,
                -int(row.get("quality_rank") or 0),
                str(row.get("model_id") or ""),
            )
        )
        return candidates[0]

    @staticmethod
    def _model_scout_tier_human(candidate: dict[str, Any]) -> str:
        tier = str(candidate.get("tier") or "").strip().lower()
        if tier == "free_remote":
            return "a free remote model"
        if tier == "cheap_remote":
            return "a very cheap remote model"
        if bool(candidate.get("local", False)):
            return "a local model"
        return "a usable model"

    @staticmethod
    def _model_scout_task_request(text: str) -> dict[str, Any]:
        normalized = normalize_setup_text(text).replace("/", " ")
        if any(token in normalized for token in ("code", "coding", "debug", "refactor", "review")):
            return {
                "task_type": "coding",
                "requirements": ["chat"],
                "preferred_local": True,
            }
        if any(token in normalized for token in ("research", "deeper", "deep analysis", "reasoning", "analyze")):
            return {
                "task_type": "reasoning",
                "requirements": ["chat", "long_context"],
                "preferred_local": True,
            }
        return {
            "task_type": "chat",
            "requirements": ["chat"],
            "preferred_local": True,
        }

    @staticmethod
    def _model_scout_task_label(task_type: str) -> str:
        normalized = str(task_type or "chat").strip().lower() or "chat"
        if normalized == "coding":
            return "coding work"
        if normalized == "reasoning":
            return "deeper analysis"
        return "everyday chat"

    @staticmethod
    def _model_scout_role_line(title: str, candidate: dict[str, Any] | None) -> str | None:
        if not isinstance(candidate, dict):
            return None
        model_id = str(candidate.get("model_id") or "").strip()
        if not model_id:
            return None
        return f"{title}: {model_id}."

    @staticmethod
    def _model_scout_role_blocked_line(
        title: str,
        candidate: dict[str, Any] | None,
        *,
        mode_label: str,
    ) -> str | None:
        if not isinstance(candidate, dict):
            return None
        model_id = str(candidate.get("model_id") or "").strip()
        if not model_id:
            return None
        return f"{title}: {model_id} ({mode_label} does not allow it right now)."

    @staticmethod
    def _model_scout_discovery_query_focus(query: str | None) -> dict[str, str]:
        normalized = normalize_setup_text(query).replace("-", " ")
        if not normalized:
            return {
                "summary": "The closest matches look like",
                "basis": "I compared family fit, locality, and task fit.",
            }
        if any(phrase in normalized for phrase in ("newer than", "newest than", "more recent than", "better than", "latest")) and any(
            token in normalized for token in ("chat", "coding", "code", "vision", "reasoning", "local")
        ):
            return {
                "summary": "For a newer chat option, the closest upgrades look like",
                "basis": "I compared family freshness, size, and chat fit against the model you named.",
            }
        if any(token in normalized for token in ("vision", "image")):
            return {
                "summary": "For a lightweight local vision model, the closest fits look like",
                "basis": "I compared vision capability, local practicality, and size.",
            }
        if any(token in normalized for token in ("coding", "code", "coder")):
            return {
                "summary": "For a small local coding model, the closest local fits look like",
                "basis": "I compared coding fit, local practicality, and size.",
            }
        if "gemma" in normalized:
            return {
                "summary": "For Gemma, the closest family matches look like",
                "basis": "I compared Gemma-family fit, recency, and size.",
            }
        if any(token in normalized for token in ("local", "ollama")):
            return {
                "summary": "For a local model, the closest local fits look like",
                "basis": "I compared local practicality and task fit.",
            }
        return {
            "summary": "The closest matches look like",
            "basis": "I compared family fit, locality, and task fit.",
        }

    @staticmethod
    def _model_scout_discovery_group_lines(
        *,
        query: str | None,
        models: list[dict[str, Any]],
    ) -> tuple[str | None, list[str]]:
        normalized = normalize_setup_text(query).replace("-", " ").lower()
        comparative = any(token in normalized for token in ("newer than", "newest than", "more recent than", "better than", "latest"))
        local_preferred = any(token in normalized for token in ("local", "ollama"))
        baseline_tokens: list[str] = []
        baseline_signature = ""
        if comparative:
            match = re.search(
                r"\b(?:newer than|newest than|more recent than|better than|latest)\s+(?P<baseline>.+?)(?:\s+for\s+|\s+than\s+|$)",
                normalized,
            )
            if match is not None:
                baseline = str(match.group("baseline") or "").strip().lower()
                baseline_tokens = [token for token in re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)?", baseline) if token]
                baseline_signature = re.sub(r"[^a-z0-9]+", "", baseline)
        likely_family: list[str] = []
        practical_local: list[str] = []
        related: list[str] = []
        row_lookup: dict[str, dict[str, Any]] = {}
        query_terms = [
            term
            for term in ("gemma", "qwen", "llama", "mistral", "deepseek", "coder", "vision", "chat")
            if term in normalized
        ]
        for row in models:
            model_id = str(row.get("id") or row.get("model_name") or row.get("model") or "").strip()
            if not model_id:
                continue
            model_text = model_id.lower()
            model_signature = re.sub(r"[^a-z0-9]+", "", model_text)
            match_band = str(row.get("match_band") or "").strip().lower()
            row_lookup[model_id] = row
            if comparative and baseline_signature and baseline_signature in model_signature:
                continue
            if comparative and baseline_tokens and all(token in model_text for token in baseline_tokens):
                continue
            if any(term in model_text for term in query_terms) and model_id not in likely_family:
                likely_family.append(model_id)
            if bool(row.get("local", False)) and model_id not in practical_local:
                practical_local.append(model_id)
            if model_id not in likely_family and model_id not in practical_local and match_band in {"likely", "related"} and model_id not in related:
                related.append(model_id)
            if len(likely_family) >= 3 and len(practical_local) >= 3 and len(related) >= 3:
                break

        if not likely_family and models:
            likely_family = [
                str(row.get("id") or row.get("model_name") or row.get("model") or "").strip()
                for row in models[:3]
                if str(row.get("id") or row.get("model_name") or row.get("model") or "").strip()
            ]
        if not practical_local:
            practical_local = [
                str(row.get("id") or row.get("model_name") or row.get("model") or "").strip()
                for row in models
                if isinstance(row, dict) and bool(row.get("local", False))
            ][:3]
        if not related:
            related = [
                str(row.get("id") or row.get("model_name") or row.get("model") or "").strip()
                for row in models[:5]
                if str(row.get("id") or row.get("model_name") or row.get("model") or "").strip()
            ]

        def _size_b(model_id: str) -> float | None:
            match = re.search(r"(?<!\w)(\d+(?:\.\d+)?)\s*[bB](?!\w)", model_id)
            if match is None:
                return None
            try:
                return float(match.group(1))
            except (TypeError, ValueError):
                return None

        def _task_hint() -> str | None:
            if any(token in normalized for token in ("vision", "image")):
                return "vision"
            if any(token in normalized for token in ("coding", "code", "coder")):
                return "coding"
            if "chat" in normalized:
                return "chat"
            if any(token in normalized for token in ("reasoning", "research", "analysis", "analyze", "deeper")):
                return "reasoning"
            return None

        def _score(model_id: str, *, local: bool = False) -> tuple[int, float, str]:
            model_text = model_id.lower()
            row = row_lookup.get(model_id) if isinstance(row_lookup.get(model_id), dict) else {}
            capability_text = " ".join(
                str(item).strip().lower()
                for item in (row.get("capabilities") if isinstance(row, dict) and isinstance(row.get("capabilities"), list) else [])
                if str(item).strip()
            )
            task_hint = _task_hint()
            score = 0
            if comparative:
                model_signature = re.sub(r"[^a-z0-9]+", "", model_text)
                if baseline_signature and baseline_signature in model_signature:
                    score -= 1000
                elif baseline_tokens and all(token in model_text for token in baseline_tokens):
                    score -= 1000
                if any(token in model_text for token in ("qwen3", "gemma3", "llama3.2", "llama-3.2", "mistral-nemo")):
                    score += 80
                if any(token in model_text for token in ("qwen3.5", "gemma-3", "llava", "moondream", "deepseek")):
                    score += 50
                if any(token in model_text for token in ("qwen2.5", "gemma-2", "llama3", "mistral")):
                    score += 20
            if local_preferred:
                score += 70 if local else -20
                if task_hint == "vision" and (any(token in capability_text for token in ("vision", "image", "multimodal")) or any(token in model_text for token in ("vision", "llava", "moondream"))):
                    score += 55
                elif task_hint == "coding" and (any(token in capability_text for token in ("code", "coding", "coder")) or any(token in model_text for token in ("coder", "code", "qwen2.5-coder", "codellama"))):
                    score += 45
                elif task_hint == "chat" and ("chat" in capability_text or "chat" in model_text):
                    score += 20
                elif task_hint == "reasoning" and (any(token in capability_text for token in ("reasoning", "analysis")) or any(token in model_text for token in ("r1", "reasoning", "analysis", "deepseek"))):
                    score += 35
                size_b = _size_b(model_id)
                if size_b is not None:
                    if size_b <= 4.0:
                        score += 20
                    elif size_b <= 8.0:
                        score += 10
                    else:
                        score -= 10
            else:
                size_b = _size_b(model_id)
                if size_b is not None and size_b <= 8.0:
                    score += 5
            return (-score, _size_b(model_id) or 999.0, model_text)

        likely_family.sort(key=lambda item: _score(item, local=local_preferred))
        practical_local.sort(key=lambda item: _score(item, local=True))
        related.sort(key=lambda item: _score(item, local=bool(item in practical_local)))

        lead_candidate = None
        if comparative and likely_family:
            lead_candidate = likely_family[0]
        elif local_preferred and practical_local:
            lead_candidate = practical_local[0]
        elif likely_family:
            lead_candidate = likely_family[0]
        elif practical_local:
            lead_candidate = practical_local[0]
        elif related:
            lead_candidate = related[0]

        lines: list[str] = []
        if local_preferred and practical_local:
            lines.append(f"Practical local fit: {', '.join(practical_local[:3])}.")
        if likely_family:
            lines.append(f"Likely family match: {', '.join(likely_family[:3])}.")
        if not local_preferred and practical_local:
            lines.append(f"Practical local fit: {', '.join(practical_local[:3])}.")
        if related:
            related_filtered = [item for item in related[:3] if item not in likely_family[:3] and item not in practical_local[:3]]
            if related_filtered:
                lines.append(f"Related alternative: {', '.join(related_filtered)}.")
        return lead_candidate, lines

    @staticmethod
    def _model_scout_acquisition_line(
        candidate: dict[str, Any] | None,
        *,
        mode_label: str,
    ) -> tuple[str | None, str | None]:
        if not isinstance(candidate, dict):
            return None, None
        model_id = str(candidate.get("model_id") or "").strip()
        if not model_id:
            return None, None
        acquisition_state = str(candidate.get("acquisition_state") or "").strip().lower()
        acquisition_reason = str(
            candidate.get("acquisition_reason")
            or candidate.get("availability_reason")
            or ""
        ).strip()
        if acquisition_state == "acquirable":
            return (
                f"Available to acquire first: {model_id}. It is not ready yet, but I can get it if you approve.",
                model_id,
            )
        if acquisition_state == "installed_not_ready":
            return f"Installed but not ready yet: {model_id}.", None
        if acquisition_state == "queued":
            return f"Waiting for approval: {model_id}.", None
        if acquisition_state == "downloading":
            return f"Downloading now: {model_id}.", None
        if acquisition_state == "blocked_by_policy":
            return f"Not available in this mode: {model_id}. {mode_label} is blocking install/download/import right now.", None
        if acquisition_reason:
            return f"Not ready yet: {model_id}. {acquisition_reason}.", None
        return None, None

    @staticmethod
    def _model_scout_mode_line(
        *,
        control_mode: str,
        mode_label: str,
        allow_install_pull: bool,
        local_only: bool,
    ) -> str:
        if control_mode == "safe":
            return "Mode: SAFE MODE. Local-only right now; cloud suggestions are advisory only."
        if local_only:
            if allow_install_pull:
                return f"Mode: {mode_label}. Recommendations are local-only right now. I only act after you approve."
            return f"Mode: {mode_label}. Recommendations are local-only right now, and installs are paused by policy. I only act after you approve."
        if allow_install_pull:
            return f"Mode: {mode_label}. I can recommend local and cloud options. I only act after you approve."
        return f"Mode: {mode_label}. I can recommend local and cloud options, but installs are paused by policy. I only act after you approve."

    @staticmethod
    def _model_scout_action_note(
        *,
        advisory_actions: dict[str, Any] | None,
    ) -> str:
        prefix = "No change has been made."
        if not isinstance(advisory_actions, dict):
            return prefix

        labels = {
            "test": "test it",
            "switch_temporarily": "switch to it temporarily",
            "make_default": "make it the default",
            "acquire": "acquire/install it first",
        }
        ordered_names = ("test", "switch_temporarily", "make_default", "acquire")
        available = [
            labels[name]
            for name in ordered_names
            if isinstance(advisory_actions.get(name), dict)
            and str((advisory_actions.get(name) or {}).get("state") or "").strip().lower() == "available"
        ]
        if available:
            if len(available) == 1:
                actions_text = available[0]
            elif len(available) == 2:
                actions_text = f"{available[0]} or {available[1]}"
            else:
                actions_text = f"{', '.join(available[:-1])}, or {available[-1]}"
            return f"{prefix} You can {actions_text} if you want."

        blocked_reasons = {
            str((advisory_actions.get(name) or {}).get("reason_code") or "").strip().lower()
            for name in ordered_names
            if isinstance(advisory_actions.get(name), dict)
            and str((advisory_actions.get(name) or {}).get("state") or "").strip().lower() == "blocked"
        }
        blocked_reasons.discard("")
        if blocked_reasons == {"safe_mode_remote_block"}:
            return f"{prefix} In SAFE MODE, remote actions are blocked."
        if blocked_reasons <= {"remote_switch_disabled", "remote_recommendation_disabled"} and blocked_reasons:
            return f"{prefix} Remote actions are blocked by policy right now."
        if blocked_reasons <= {"safe_mode_install_block", "install_disabled_by_policy"} and blocked_reasons:
            return f"{prefix} Install/download actions are blocked in this mode."
        return prefix

    @staticmethod
    def _model_scout_requested_role_title(requested_remote_role: str | None) -> str | None:
        normalized = str(requested_remote_role or "").strip().lower()
        if normalized == "best_local":
            return "Best local option"
        if normalized == "cheap_cloud":
            return "Cheap cloud recommendation"
        if normalized == "premium_coding":
            return "Premium coding recommendation"
        if normalized == "premium_research":
            return "Premium research recommendation"
        if normalized == "premium_general":
            return "Premium remote recommendation"
        return None

    @staticmethod
    def _model_scout_requested_role_reason(requested_remote_role: str | None) -> str | None:
        normalized = str(requested_remote_role or "").strip().lower()
        if normalized == "best_local":
            return "no usable local model is currently available."
        if normalized == "cheap_cloud":
            return "no usable low-cost remote model is currently available."
        if normalized == "premium_research":
            return "no remote model currently meets the required premium quality and context thresholds."
        if normalized in {"premium_coding", "premium_general"}:
            return "no remote model currently meets the required premium quality threshold."
        return None

    @staticmethod
    def _model_scout_requested_role_candidate(
        requested_remote_role: str | None,
        *,
        local_primary_candidate: dict[str, Any] | None,
        cheap_cloud_candidate: dict[str, Any] | None,
        premium_coding_candidate: dict[str, Any] | None,
        premium_research_candidate: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        normalized = str(requested_remote_role or "").strip().lower()
        if normalized == "best_local":
            return (
                {
                    **dict(local_primary_candidate),
                    "recommendation_basis": str(local_primary_candidate.get("recommendation_basis") or "best_local"),
                    "recommendation_explanation": (
                        str(local_primary_candidate.get("recommendation_explanation") or "").strip()
                        or "strongest local option currently available"
                    ),
                }
                if isinstance(local_primary_candidate, dict)
                else None
            )
        if normalized == "cheap_cloud":
            return dict(cheap_cloud_candidate) if isinstance(cheap_cloud_candidate, dict) else None
        if normalized == "premium_coding":
            return dict(premium_coding_candidate) if isinstance(premium_coding_candidate, dict) else None
        if normalized == "premium_research":
            return dict(premium_research_candidate) if isinstance(premium_research_candidate, dict) else None
        if normalized == "premium_general":
            if isinstance(premium_research_candidate, dict):
                return dict(premium_research_candidate)
            if isinstance(premium_coding_candidate, dict):
                return dict(premium_coding_candidate)
        return None

    @staticmethod
    def _model_scout_primary_role_resolution(
        *,
        recommendation_roles: dict[str, Any],
        requested_remote_role: str | None,
        task_type: str,
        primary_candidate: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not isinstance(recommendation_roles, dict):
            return None
        model_id = str((primary_candidate or {}).get("model_id") or "").strip()
        normalized_request = str(requested_remote_role or "").strip().lower()
        normalized_task = str(task_type or "chat").strip().lower() or "chat"
        preferred_keys: list[str] = []
        if normalized_request == "best_local":
            preferred_keys = ["best_local"]
        elif normalized_request == "cheap_cloud":
            preferred_keys = ["cheap_cloud"]
        elif normalized_request == "premium_coding":
            preferred_keys = ["premium_coding"]
        elif normalized_request == "premium_research":
            preferred_keys = ["premium_research"]
        elif normalized_request == "premium_general":
            preferred_keys = ["premium_research", "premium_coding"]
        elif normalized_task == "coding":
            preferred_keys = ["best_task_coding"]
        elif normalized_task == "reasoning":
            preferred_keys = ["best_task_research"]
        else:
            preferred_keys = ["best_task_chat"]
        for key in preferred_keys:
            resolution = recommendation_roles.get(key)
            if not isinstance(resolution, dict):
                continue
            resolution_model = str(resolution.get("model_id") or "").strip()
            if not model_id or not resolution_model or resolution_model == model_id:
                return dict(resolution)
        if model_id:
            for resolution in recommendation_roles.values():
                if not isinstance(resolution, dict):
                    continue
                if str(resolution.get("state") or "").strip().lower() != "selected":
                    continue
                if str(resolution.get("model_id") or "").strip() == model_id:
                    return dict(resolution)
        return None

    @staticmethod
    def _model_scout_default_heading(task_type: str) -> str:
        normalized = str(task_type or "chat").strip().lower() or "chat"
        if normalized == "coding":
            return "Best coding option"
        if normalized == "reasoning":
            return "Best research option"
        return "Best chat option"

    @staticmethod
    def _model_scout_included_role_keys(
        *,
        task_type: str,
        requested_remote_role: str | None,
    ) -> list[str] | None:
        normalized_task = str(task_type or "chat").strip().lower() or "chat"
        normalized_request = str(requested_remote_role or "").strip().lower()
        if normalized_request == "cheap_cloud":
            return ["best_local", "cheap_cloud"]
        if normalized_request == "premium_coding":
            return ["best_local", "cheap_cloud", "premium_coding"]
        if normalized_request == "premium_research":
            return ["best_local", "cheap_cloud", "premium_research"]
        if normalized_request == "best_local":
            return ["best_local"]
        if normalized_task == "coding":
            return ["best_local", "cheap_cloud", "premium_coding", "best_task_coding"]
        if normalized_task == "reasoning":
            return ["best_local", "cheap_cloud", "premium_research", "best_task_research"]
        return None

    @staticmethod
    def _model_scout_secondary_role_lines(
        *,
        task_type: str,
        requested_remote_role: str | None,
        active_model: str | None,
        primary_model: str | None,
        local_only: bool,
        mode_label: str,
        local_primary_candidate: dict[str, Any] | None,
        cheap_cloud_candidate: dict[str, Any] | None,
        premium_coding_candidate: dict[str, Any] | None,
        premium_research_candidate: dict[str, Any] | None,
    ) -> list[str]:
        titles = {
            "best_local": "Best local option (fast, no cost)",
            "cheap_cloud": "Cheap cloud option",
            "premium_coding": "Premium coding option",
            "premium_research": "Premium research option",
        }
        candidates = {
            "best_local": local_primary_candidate,
            "cheap_cloud": cheap_cloud_candidate,
            "premium_coding": premium_coding_candidate,
            "premium_research": premium_research_candidate,
        }
        normalized_task = str(task_type or "chat").strip().lower() or "chat"
        normalized_request = str(requested_remote_role or "").strip().lower()
        if normalized_request in {"cheap_cloud", "premium_coding", "premium_research", "premium_general"}:
            order = ["best_local", "cheap_cloud"]
        elif normalized_task == "coding":
            order = ["best_local", "cheap_cloud", "premium_coding"]
        elif normalized_task == "reasoning":
            order = ["best_local", "cheap_cloud", "premium_research"]
        else:
            order = ["best_local", "cheap_cloud"]
        seen_models = {str(primary_model or "").strip()}
        lines: list[str] = []
        for key in order:
            candidate = candidates.get(key)
            if not isinstance(candidate, dict):
                continue
            model_id = str(candidate.get("model_id") or "").strip()
            if not model_id:
                continue
            if model_id in seen_models:
                continue
            if key == "best_local" and model_id == str(active_model or "").strip():
                continue
            if local_only and key != "best_local":
                line = Orchestrator._model_scout_role_blocked_line(
                    titles[key],
                    candidate,
                    mode_label=mode_label,
                )
            elif not local_only:
                line = Orchestrator._model_scout_role_line(titles[key], candidate)
            else:
                line = None
            if line:
                lines.append(line)
                seen_models.add(model_id)
        return lines

    def _model_scout_strategy_response(
        self,
        user_id: str,
        text: str,
        *,
        requested_role_override: str | None = None,
    ) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="action_tool",
                reason="runtime_truth_service_unavailable",
            )
        payload_fn = getattr(truth, "model_scout_v2_status", None)
        if not callable(payload_fn):
            return self._model_scout_inventory_response(focus_terms=[])
        task_request = self._model_scout_task_request(text)
        requested_remote_role = str(requested_role_override or "").strip().lower() or self._model_scout_remote_role_requested(
            normalize_setup_text(text).replace("/", " ")
        )
        included_role_keys = self._model_scout_included_role_keys(
            task_type=str(task_request.get("task_type") or "chat").strip().lower() or "chat",
            requested_remote_role=requested_remote_role,
        )
        try:
            payload = payload_fn(task_request=task_request, included_role_keys=included_role_keys)
        except TypeError:
            payload = payload_fn(task_request=task_request)
        active_model = str(payload.get("active_model") or "").strip() or None
        active_provider = str(payload.get("active_provider") or "").strip().lower() or None
        current_candidate = payload.get("current_candidate") if isinstance(payload.get("current_candidate"), dict) else None
        recommended_candidate = payload.get("recommended_candidate") if isinstance(payload.get("recommended_candidate"), dict) else None
        task_recommendation = payload.get("task_recommendation") if isinstance(payload.get("task_recommendation"), dict) else None
        better_candidates = [
            dict(row)
            for row in (payload.get("better_candidates") if isinstance(payload.get("better_candidates"), list) else [])
            if isinstance(row, dict)
        ]
        candidate_rows = [
            dict(row)
            for row in (payload.get("candidate_rows") if isinstance(payload.get("candidate_rows"), list) else [])
            if isinstance(row, dict)
        ]
        recommendation_roles = (
            payload.get("recommendation_roles") if isinstance(payload.get("recommendation_roles"), dict) else {}
        )
        not_ready_rows = [
            dict(row)
            for row in (payload.get("not_ready_models") if isinstance(payload.get("not_ready_models"), list) else [])
            if isinstance(row, dict)
        ]
        policy = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
        role_candidates = payload.get("role_candidates") if isinstance(payload.get("role_candidates"), dict) else {}
        advisory_only = bool(payload.get("advisory_only", False))
        control_mode = str(policy.get("mode") or ("safe" if advisory_only else "controlled")).strip().lower() or ("safe" if advisory_only else "controlled")
        mode_label = str(policy.get("mode_label") or ("SAFE MODE" if control_mode == "safe" else "Controlled Mode")).strip() or ("SAFE MODE" if control_mode == "safe" else "Controlled Mode")
        task_type = str((payload.get("task_request") if isinstance(payload.get("task_request"), dict) else {}).get("task_type") or task_request.get("task_type") or "chat").strip().lower() or "chat"
        allow_remote_recommendation = bool(
            policy.get("allow_remote_recommendation", policy.get("allow_remote_fallback", True))
        )
        local_only = bool(policy.get("safe_mode", False) or not allow_remote_recommendation)
        if local_only:
            candidate_rows = [dict(row) for row in candidate_rows if bool(row.get("local", False))]
            better_candidates = [dict(row) for row in better_candidates if bool(row.get("local", False))]
            not_ready_rows = [dict(row) for row in not_ready_rows if bool(row.get("local", False))]
            if isinstance(recommended_candidate, dict) and not bool(recommended_candidate.get("local", False)):
                recommended_candidate = None
            if isinstance(task_recommendation, dict) and not bool(task_recommendation.get("local", False)):
                task_recommendation = None

        if not candidate_rows and not active_model:
            message = (
                "I do not currently see any ready chat models to scout. "
                "If you want, I can help you configure a provider or add a local model."
            )
            return self._runtime_truth_response(
                text=message,
                route="action_tool",
                used_runtime_state=True,
                used_tools=["model_scout"],
                payload={
                    "type": "model_scout",
                    "mode": "strategy",
                    "summary": message,
                    "active_model": active_model,
                    "active_provider": active_provider,
                    "candidate_rows": candidate_rows,
                    "better_candidates": better_candidates,
                    "source": "runtime_truth.model_scout_v2",
                },
            )

        active_label = active_model or "the current chat target"
        cheap_cloud_candidate = role_candidates.get("cheap_cloud") if isinstance(role_candidates.get("cheap_cloud"), dict) else None
        premium_coding_candidate = (
            role_candidates.get("premium_coding_cloud")
            if isinstance(role_candidates.get("premium_coding_cloud"), dict)
            else None
        )
        premium_research_candidate = (
            role_candidates.get("premium_research_cloud")
            if isinstance(role_candidates.get("premium_research_cloud"), dict)
            else None
        )
        local_primary_candidate = (
            role_candidates.get("comfortable_local_default")
            if isinstance(role_candidates.get("comfortable_local_default"), dict)
            else None
        )
        acquisition_candidate = self._model_acquisition_candidate(
            not_ready_rows,
            exclude_model=active_model,
        )
        allow_install_pull = bool(policy.get("allow_install_pull", True))
        acquisition_line, acquisition_model = self._model_scout_acquisition_line(
            acquisition_candidate,
            mode_label=mode_label,
        )
        mode_line = self._model_scout_mode_line(
            control_mode=control_mode,
            mode_label=mode_label,
            allow_install_pull=allow_install_pull,
            local_only=local_only,
        )
        requested_role_title = self._model_scout_requested_role_title(requested_remote_role)
        requested_role_reason = self._model_scout_requested_role_reason(requested_remote_role)
        requested_role_candidate = (
            self._model_scout_requested_role_candidate(
                requested_remote_role,
                local_primary_candidate=local_primary_candidate,
                cheap_cloud_candidate=cheap_cloud_candidate,
                premium_coding_candidate=premium_coding_candidate,
                premium_research_candidate=premium_research_candidate,
            )
            if not local_only
            else (
                self._model_scout_requested_role_candidate(
                    requested_remote_role,
                    local_primary_candidate=local_primary_candidate,
                    cheap_cloud_candidate=None,
                    premium_coding_candidate=None,
                    premium_research_candidate=None,
                )
                if str(requested_remote_role or "").strip().lower() == "best_local"
                else None
            )
        )
        primary_heading = requested_role_title or self._model_scout_default_heading(task_type)
        explicit_requested_role = bool(requested_role_title)
        primary_candidate = (
            requested_role_candidate
            if explicit_requested_role
            else (task_recommendation if isinstance(task_recommendation, dict) else recommended_candidate)
        )
        primary_resolution = self._model_scout_primary_role_resolution(
            recommendation_roles=recommendation_roles,
            requested_remote_role=requested_remote_role,
            task_type=task_type,
            primary_candidate=primary_candidate,
        )
        secondary_lines = self._model_scout_secondary_role_lines(
            task_type=task_type,
            requested_remote_role=requested_remote_role,
            active_model=active_model,
            primary_model=(primary_candidate.get("model_id") if isinstance(primary_candidate, dict) else None),
            local_only=local_only,
            mode_label=mode_label,
            local_primary_candidate=local_primary_candidate,
            cheap_cloud_candidate=cheap_cloud_candidate,
            premium_coding_candidate=premium_coding_candidate,
            premium_research_candidate=premium_research_candidate,
        )
        if isinstance(primary_candidate, dict):
            target_model = str(primary_candidate.get("model_id") or "").strip() or "that model"
            reason = str(primary_candidate.get("recommendation_explanation") or "").strip() or "it looks like the strongest practical option right now"
            comparison = (
                primary_resolution.get("comparison")
                if isinstance(primary_resolution, dict) and isinstance(primary_resolution.get("comparison"), dict)
                else None
            )
            advisory_actions = (
                primary_resolution.get("advisory_actions")
                if isinstance(primary_resolution, dict) and isinstance(primary_resolution.get("advisory_actions"), dict)
                else None
            )
            comparison_text = str((comparison or {}).get("explanation") or "").strip()
            lines = [
                f"Current model: {active_label}.",
                f"{primary_heading}: {target_model}.",
                f"Why: {reason}.",
            ]
            if comparison_text:
                lines.append(f"Compared with current: {comparison_text}.")
            lines.extend(secondary_lines)
            if acquisition_line:
                lines.append(acquisition_line)
            lines.append(mode_line)
            lines.append(self._model_scout_action_note(advisory_actions=advisory_actions))
            prompt = "\n".join(lines)
            return self._runtime_truth_response(
                text=prompt,
                route="action_tool",
                used_runtime_state=True,
                used_tools=["model_scout"],
                payload={
                    "type": "model_scout",
                    "mode": "strategy",
                    "summary": prompt,
                    "active_model": active_model,
                    "active_provider": active_provider,
                    "current_candidate": current_candidate,
                    "recommended_candidate": recommended_candidate,
                    "task_recommendation": task_recommendation,
                    "better_candidates": better_candidates,
                    "candidate_rows": candidate_rows,
                    "role_candidates": role_candidates,
                    "recommendation_roles": recommendation_roles,
                    "policy": dict(policy),
                    "advisory_only": advisory_only,
                    "source": "runtime_truth.model_scout_v2",
                },
            )

        if requested_role_title and local_only:
            lines = [
                f"Current model: {active_label}.",
                f"{primary_heading}: not available in {mode_label} right now.",
            ]
            lines.append("Reason: remote recommendations are not usable in this mode.")
            lines.extend(secondary_lines)
            if acquisition_line:
                lines.append(acquisition_line)
            lines.append(mode_line)
            lines.append(
                self._model_scout_action_note(
                    advisory_actions=(
                        primary_resolution.get("advisory_actions")
                        if isinstance(primary_resolution, dict) and isinstance(primary_resolution.get("advisory_actions"), dict)
                        else None
                    ),
                )
            )
            message = "\n".join(lines)
            return self._runtime_truth_response(
                text=message,
                route="action_tool",
                used_runtime_state=True,
                used_tools=["model_scout"],
                payload={
                    "type": "model_scout",
                    "mode": "strategy",
                    "summary": message,
                    "active_model": active_model,
                    "active_provider": active_provider,
                    "current_candidate": current_candidate,
                    "recommended_candidate": recommended_candidate,
                    "task_recommendation": task_recommendation,
                    "better_candidates": better_candidates,
                    "candidate_rows": candidate_rows,
                    "role_candidates": role_candidates,
                    "recommendation_roles": recommendation_roles,
                    "policy": dict(policy),
                    "advisory_only": advisory_only,
                    "source": "runtime_truth.model_scout_v2",
                },
            )

        if explicit_requested_role and not local_only:
            lines = [
                f"Current model: {active_label}.",
                f"{primary_heading}: none currently qualifies.",
            ]
            if requested_role_reason:
                lines.append(f"Reason: {requested_role_reason}")
            lines.extend(secondary_lines)
            if acquisition_line:
                lines.append(acquisition_line)
            lines.append(mode_line)
            lines.append(
                self._model_scout_action_note(
                    advisory_actions=(
                        primary_resolution.get("advisory_actions")
                        if isinstance(primary_resolution, dict) and isinstance(primary_resolution.get("advisory_actions"), dict)
                        else None
                    ),
                )
            )
            message = "\n".join(lines)
            return self._runtime_truth_response(
                text=message,
                route="action_tool",
                used_runtime_state=True,
                used_tools=["model_scout"],
                payload={
                    "type": "model_scout",
                    "mode": "strategy",
                    "summary": message,
                    "active_model": active_model,
                    "active_provider": active_provider,
                    "current_candidate": current_candidate,
                    "recommended_candidate": recommended_candidate,
                    "task_recommendation": task_recommendation,
                    "better_candidates": better_candidates,
                    "candidate_rows": candidate_rows,
                    "role_candidates": role_candidates,
                    "recommendation_roles": recommendation_roles,
                    "policy": dict(policy),
                    "advisory_only": advisory_only,
                    "source": "runtime_truth.model_scout_v2",
                },
            )

        lines = [
            f"Current model: {active_label}.",
            f"{primary_heading}: your current model looks fine right now.",
        ]
        lines.extend(secondary_lines)
        if acquisition_line:
            lines.append(acquisition_line)
        lines.append(mode_line)
        lines.append(
            self._model_scout_action_note(
                advisory_actions=(
                    primary_resolution.get("advisory_actions")
                    if isinstance(primary_resolution, dict) and isinstance(primary_resolution.get("advisory_actions"), dict)
                    else None
                ),
            )
        )
        message = "\n".join(lines)
        return self._runtime_truth_response(
            text=message,
            route="action_tool",
            used_runtime_state=True,
            used_tools=["model_scout"],
            payload={
                "type": "model_scout",
                "mode": "strategy",
                "summary": message,
                "active_model": active_model,
                "active_provider": active_provider,
                "current_candidate": current_candidate,
                "recommended_candidate": recommended_candidate,
                "task_recommendation": task_recommendation,
                "better_candidates": better_candidates,
                "candidate_rows": candidate_rows,
                "role_candidates": role_candidates,
                "policy": dict(policy),
                "advisory_only": advisory_only,
                "source": "runtime_truth.model_scout_v2",
            },
        )

    def _model_scout_discovery_response(self, query: str | None = None) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="action_tool",
                reason="runtime_truth_service_unavailable",
        )
        discovery = truth.model_discovery_query(query=query, filters={})
        models = [dict(row) for row in (discovery.get("models") if isinstance(discovery.get("models"), list) else []) if isinstance(row, dict)]
        sources = [dict(row) for row in (discovery.get("sources") if isinstance(discovery.get("sources"), list) else []) if isinstance(row, dict)]
        focus = self._model_scout_discovery_query_focus(query)
        lead_candidate, candidate_lines = self._model_scout_discovery_group_lines(query=query, models=models)
        debug = discovery.get("debug") if isinstance(discovery, dict) else {}
        ranking = debug.get("ranking") if isinstance(debug, dict) and isinstance(debug.get("ranking"), dict) else {}
        broadening_used = bool(ranking.get("broadening_used", False))
        broadening_variants = [
            str(item).strip()
            for item in (ranking.get("broadening_variants") if isinstance(ranking.get("broadening_variants"), list) else [])
            if str(item).strip()
        ]
        source_names = [
            str(row.get("source") or "").strip()
            for row in sources
            if str(row.get("source") or "").strip()
        ]
        source_summary = ", ".join(sorted(dict.fromkeys(source_names)))
        source_errors = debug.get("source_errors") if isinstance(debug, dict) and isinstance(debug.get("source_errors"), dict) else {}
        source_error_text = ", ".join(
            f"{source}: {detail.get('error') or detail.get('error_kind') or 'unknown_error'}"
            for source, detail in source_errors.items()
            if isinstance(detail, dict)
        )
        first_line = focus["summary"]
        if lead_candidate:
            first_line = f"{first_line} {lead_candidate}"
        elif query:
            first_line = f"{first_line} {str(query).strip()}."
        else:
            first_line = f"{first_line}."
        message_lines = [first_line]
        message_lines.extend(candidate_lines)
        if focus.get("basis"):
            message_lines.append(str(focus["basis"]))
        if broadening_used and broadening_variants:
            message_lines.append(f"Broadened search: {', '.join(broadening_variants[:4])}.")
        if source_summary:
            message_lines.append(f"Sources checked: {source_summary} ({len(source_names)} queried).")
        if source_error_text:
            message_lines.append(f"Source errors: {source_error_text}.")
        message = "\n".join(line for line in message_lines if str(line).strip())
        return self._runtime_truth_response(
            text=message,
            route="action_tool",
            used_runtime_state=True,
            used_tools=["model_discovery_manager"],
            ok=bool(discovery.get("ok", False)),
            payload={
                "type": "model_discovery",
                "mode": "external_discovery",
                "summary": message,
                "query": query,
                "models": models[:10],
                "sources": sources,
                "debug": discovery.get("debug") if isinstance(discovery, dict) else {},
                "source": "runtime_truth.model_discovery_query",
            },
        )

    def _model_ready_now_response(self) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="model_status",
                reason="runtime_truth_service_unavailable",
            )
        payload_fn = getattr(truth, "model_readiness_status", None)
        if not callable(payload_fn):
            return self._runtime_state_unavailable_response(
                route="model_status",
                reason="model_readiness_unavailable",
            )
        payload = payload_fn()
        active_model = str(payload.get("active_model") or "").strip() or None
        active_label = active_model or "no active chat model"
        ready_rows = [
            dict(row)
            for row in (payload.get("ready_now_models") if isinstance(payload.get("ready_now_models"), list) else payload.get("usable_models") if isinstance(payload.get("usable_models"), list) else [])
            if isinstance(row, dict)
        ]
        other_ready_rows = [
            dict(row)
            for row in (payload.get("other_ready_now_models") if isinstance(payload.get("other_ready_now_models"), list) else payload.get("other_usable_models") if isinstance(payload.get("other_usable_models"), list) else [])
            if isinstance(row, dict)
        ]
        not_ready_rows = [
            dict(row)
            for row in (payload.get("not_ready_models") if isinstance(payload.get("not_ready_models"), list) else [])
            if isinstance(row, dict)
        ]
        ready_preview = self._inventory_preview(other_ready_rows if active_model else ready_rows)
        not_ready_preview = self._inventory_reason_preview(not_ready_rows)
        if ready_preview:
            message = f"Right now chat is using {active_label}. Other models ready to use now: {ready_preview}."
        elif active_model and ready_rows:
            message = f"Right now chat is using {active_label}, and it is the only model that looks ready now."
        elif ready_rows:
            message = f"Models ready to use right now: {self._inventory_preview(ready_rows)}."
        else:
            message = "I do not see a chat model that is ready to use right now."
        if not_ready_preview:
            message = f"{message} Present but not ready: {not_ready_preview}."
        return self._runtime_truth_response(
            text=message,
            route="model_status",
            payload={
                **dict(payload),
                "type": "model_readiness_inventory",
                "title": "Ready chat models",
                "summary": message,
            },
        )

    def _execute_model_acquire(
        self,
        *,
        model_id: str | None,
        provider_id: str | None,
    ) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="action_tool",
                reason="runtime_truth_service_unavailable",
            )
        matched_model = str(model_id or "").strip() or None
        normalized_provider = str(provider_id or "").strip().lower() or None
        if not matched_model:
            message = "I need one exact model before I can acquire it."
            return self._runtime_truth_response(
                text=message,
                route="action_tool",
                payload={
                    "type": "model_acquisition",
                    "action": "acquire_target",
                    "ok": False,
                    "summary": message,
                },
            )
        acquire_fn = getattr(truth, "acquire_chat_model_target", None)
        if not callable(acquire_fn):
            return self._runtime_state_unavailable_response(
                route="action_tool",
                reason="model_acquisition_unavailable",
            )
        ok, body = acquire_fn(matched_model, provider_id=normalized_provider)
        response_body = body if isinstance(body, dict) else {}
        message = str(response_body.get("message") or "").strip()
        if not message:
            if ok:
                message = f"I started acquiring {matched_model} through the canonical model manager."
            else:
                message = f"I couldn't acquire {matched_model} right now."
        return self._runtime_truth_response(
            text=message,
            route="action_tool",
            ok=bool(ok),
            error_kind=None if ok else str(response_body.get("error_kind") or response_body.get("error") or "model_acquisition_failed").strip() or "model_acquisition_failed",
            used_runtime_state=True,
            used_tools=["model_manager"],
            payload={
                "type": "model_acquisition",
                "action": "acquire_target",
                "provider": str(response_body.get("provider") or normalized_provider).strip().lower() or normalized_provider,
                "model_id": str(response_body.get("model_id") or matched_model).strip() or matched_model,
                "ok": bool(ok),
                "summary": message,
                "result": dict(response_body),
            },
        )

    def _model_acquire_response(self, user_id: str, text: str, *, confirmed: bool = False) -> OrchestratorResponse:
        resolution = self._resolve_controller_model_target(user_id, text)
        status = str(resolution.get("status") or "").strip().lower()
        if status == "ambiguous":
            matches = [
                str(item).strip()
                for item in (resolution.get("matches") if isinstance(resolution.get("matches"), list) else [])
                if str(item).strip()
            ]
            requested = str(resolution.get("requested") or "that model").strip() or "that model"
            message = f"I can acquire more than one match for {requested}: {', '.join(matches[:3])}. Which exact model do you want?"
            return self._runtime_truth_response(
                text=message,
                route="action_tool",
                next_question=message,
                payload={
                    "type": "model_acquisition",
                    "action": "acquire_target",
                    "ok": False,
                    "summary": message,
                    "matches": matches,
                },
            )
        matched_model = str(resolution.get("model_id") or "").strip() or None
        provider_id = str(resolution.get("provider_id") or "").strip().lower() or None
        if not matched_model:
            return self._execute_model_acquire(model_id=None, provider_id=None)
        if not confirmed:
            question = (
                f"I will ask the canonical model manager to acquire {matched_model}. "
                "This mutates the available model inventory. Reply yes to proceed or no to cancel."
            )
            return self._confirmation_preview_response(
                user_id,
                route="action_tool",
                question=question,
                used_tools=["model_manager"],
                action={
                    "operation": "model_acquire",
                    "params": {
                        "model_id": matched_model,
                        "provider_id": provider_id,
                    },
                },
                title="Model acquisition confirmation",
                preview_payload={
                    "provider": provider_id,
                    "model_id": matched_model,
                    "preview": {
                        "provider": provider_id,
                        "model_id": matched_model,
                    },
                },
            )
        return self._execute_model_acquire(model_id=matched_model, provider_id=provider_id)

    def _model_controller_test_response(self, user_id: str, text: str) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="action_tool",
                reason="runtime_truth_service_unavailable",
            )
        resolution = self._resolve_controller_model_target(user_id, text)
        status = str(resolution.get("status") or "").strip().lower()
        if status == "ambiguous":
            matches = [
                str(item).strip()
                for item in (resolution.get("matches") if isinstance(resolution.get("matches"), list) else [])
                if str(item).strip()
            ]
            requested = str(resolution.get("requested") or "that model").strip() or "that model"
            message = f"I can test more than one match for {requested}: {', '.join(matches[:3])}. Which exact model do you want me to test?"
            return self._runtime_truth_response(
                text=message,
                route="action_tool",
                next_question=message,
                payload={
                    "type": "model_controller",
                    "action": "test_target",
                    "ok": False,
                    "summary": message,
                    "matches": matches,
                },
            )
        model_id = str(resolution.get("model_id") or "").strip() or None
        provider_id = str(resolution.get("provider_id") or "").strip().lower() or None
        if not model_id:
            message = "I need one exact model to test. If you want, I can show the ready models or the local installed models first."
            return self._runtime_truth_response(
                text=message,
                route="action_tool",
                payload={
                    "type": "model_controller",
                    "action": "test_target",
                    "ok": False,
                    "summary": message,
                },
            )
        test_fn = getattr(truth, "test_chat_model_target", None)
        if not callable(test_fn):
            return self._runtime_state_unavailable_response(
                route="action_tool",
                reason="model_controller_test_unavailable",
            )
        ok, body = test_fn(model_id, provider_id=provider_id)
        provider_label = self._setup_provider_label(str((body if isinstance(body, dict) else {}).get("provider") or provider_id or "").strip().lower() or None)
        reason = str((body if isinstance(body, dict) else {}).get("reason") or (body if isinstance(body, dict) else {}).get("error") or "").strip()
        if ok:
            message = f"I tested {model_id} without switching. It responded successfully on {provider_label}."
        else:
            message = f"I tested {model_id} without switching, and it is not ready right now."
            if reason:
                message = f"{message.rstrip('.')} Reason: {reason}."
        return self._runtime_truth_response(
            text=message,
            route="action_tool",
            ok=bool(ok),
            error_kind=None if ok else str((body if isinstance(body, dict) else {}).get("error") or "model_test_failed").strip() or "model_test_failed",
            used_runtime_state=True,
            used_tools=["model_controller"],
            payload={
                "type": "model_controller",
                "action": "test_target",
                "provider": str((body if isinstance(body, dict) else {}).get("provider") or provider_id).strip().lower() or provider_id,
                "model_id": str((body if isinstance(body, dict) else {}).get("model_id") or model_id).strip() or model_id,
                "ok": bool(ok),
                "summary": message,
            },
        )

    def _execute_model_controller_trial_switch(
        self,
        user_id: str,
        *,
        model_id: str | None,
        provider_id: str | None,
    ) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="model_status",
                reason="runtime_truth_service_unavailable",
            )
        matched_model = str(model_id or "").strip() or None
        normalized_provider = str(provider_id or "").strip().lower() or None
        if not matched_model:
            message = "I need one exact model before I can switch temporarily."
            return self._runtime_truth_response(
                text=message,
                route="model_status",
                payload={
                    "type": "model_controller",
                    "action": "trial_switch",
                    "ok": False,
                    "summary": message,
                },
            )
        current_target = truth.current_chat_target_status()
        previous_provider, previous_model = self._target_snapshot_from_truth(current_target)
        explicit_setter = getattr(truth, "set_temporary_chat_model_target", None)
        if callable(explicit_setter):
            switch_ok, switch_body = explicit_setter(matched_model, provider_id=normalized_provider)
        else:
            fallback_setter = getattr(truth, "set_confirmed_chat_model_target", None)
            if callable(fallback_setter):
                switch_ok, switch_body = fallback_setter(matched_model, provider_id=normalized_provider)
            else:
                switch_ok, switch_body = truth.set_default_chat_model(matched_model)
        applied_provider = str((switch_body if isinstance(switch_body, dict) else {}).get("provider") or normalized_provider).strip().lower() or None
        applied_model = str((switch_body if isinstance(switch_body, dict) else {}).get("model_id") or matched_model).strip() or None
        if bool(switch_ok):
            self._record_model_trial_switch(
                user_id,
                previous_provider=previous_provider,
                previous_model=previous_model,
                applied_provider=applied_provider,
                applied_model=applied_model,
                source="temporary_switch",
            )
        return self._post_switch_response(
            truth=truth,
            route="model_status",
            used_memory=False,
            used_tools=["model_controller"],
            ok=bool(switch_ok),
            body=switch_body if isinstance(switch_body, dict) else {},
            applied_provider=applied_provider,
            applied_model=applied_model,
            success_type="model_controller",
            success_title="Temporary model switch",
            failure_title="Temporary switch failed",
        )

    def _model_controller_trial_switch_response(self, user_id: str, text: str, *, confirmed: bool = False) -> OrchestratorResponse:
        resolution = self._resolve_controller_model_target(user_id, text)
        status = str(resolution.get("status") or "").strip().lower()
        if status == "ambiguous":
            matches = [
                str(item).strip()
                for item in (resolution.get("matches") if isinstance(resolution.get("matches"), list) else [])
                if str(item).strip()
            ]
            requested = str(resolution.get("requested") or "that model").strip() or "that model"
            message = f"I can switch temporarily to more than one match for {requested}: {', '.join(matches[:3])}. Which exact model do you want?"
            return self._runtime_truth_response(
                text=message,
                route="model_status",
                next_question=message,
                payload={
                    "type": "model_controller",
                    "action": "trial_switch",
                    "ok": False,
                    "summary": message,
                    "matches": matches,
                },
            )
        matched_model = str(resolution.get("model_id") or "").strip() or None
        provider_id = str(resolution.get("provider_id") or "").strip().lower() or None
        if not matched_model:
            return self._execute_model_controller_trial_switch(user_id, model_id=None, provider_id=None)
        if not confirmed:
            question = (
                f"I will switch chat temporarily to {matched_model}. "
                "This mutates the active chat target. Reply yes to proceed or no to cancel."
            )
            return self._confirmation_preview_response(
                user_id,
                route="model_status",
                question=question,
                used_tools=["model_controller"],
                action={
                    "operation": "model_trial_switch",
                    "params": {
                        "model_id": matched_model,
                        "provider_id": provider_id,
                    },
                },
                title="Temporary switch confirmation",
                preview_payload={
                    "provider": provider_id,
                    "model_id": matched_model,
                    "preview": {
                        "provider": provider_id,
                        "model_id": matched_model,
                        "switch_kind": "temporary",
                    },
                },
            )
        return self._execute_model_controller_trial_switch(user_id, model_id=matched_model, provider_id=provider_id)

    def _execute_model_set_target(
        self,
        user_id: str,
        *,
        model_id: str | None,
        provider_id: str | None,
        promote_default: bool,
        used_memory: bool,
    ) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="model_status",
                used_memory=used_memory,
                reason="runtime_truth_service_unavailable",
            )
        matched_model = str(model_id or "").strip() or None
        normalized_provider = str(provider_id or "").strip().lower() or None
        if not matched_model:
            message = (
                "I couldn't find that model in the current runtime registry. "
                "If you want, I can list the models that are ready now or the local installed models."
            )
            return self._runtime_truth_response(
                text=message,
                route="model_status",
                used_memory=used_memory,
                next_question=message,
                payload={
                    "type": "action_required",
                    "title": "Which model?",
                    "summary": message,
                },
            )
        current_target = truth.current_chat_target_status()
        previous_provider, previous_model = self._target_snapshot_from_truth(current_target)
        if promote_default:
            policy_status = (
                truth.model_controller_policy_status()
                if callable(getattr(truth, "model_controller_policy_status", None))
                else {}
            )
            use_explicit_controller = not bool(policy_status.get("safe_mode", False))
            explicit_setter = getattr(truth, "set_confirmed_chat_model_target", None)
            default_setter = getattr(truth, "set_default_chat_model", None)
            if use_explicit_controller and callable(explicit_setter):
                default_ok, default_body = explicit_setter(
                    matched_model,
                    provider_id=normalized_provider,
                )
            elif callable(default_setter):
                default_ok, default_body = default_setter(matched_model)
            elif callable(explicit_setter):
                default_ok, default_body = explicit_setter(
                    matched_model,
                    provider_id=normalized_provider,
                )
            else:
                default_ok, default_body = truth.set_default_chat_model(matched_model)
        else:
            explicit_setter = getattr(truth, "set_confirmed_chat_model_target", None)
            if callable(explicit_setter):
                default_ok, default_body = explicit_setter(
                    matched_model,
                    provider_id=normalized_provider,
                )
            else:
                default_ok, default_body = truth.set_default_chat_model(matched_model)
        self._clear_runtime_setup_state(user_id)
        if promote_default:
            applied_provider = normalized_provider
            applied_model = matched_model
        else:
            applied_provider = str((default_body if isinstance(default_body, dict) else {}).get("provider") or normalized_provider).strip().lower() or None
            applied_model = str((default_body if isinstance(default_body, dict) else {}).get("model_id") or matched_model).strip() or None
        if bool(default_ok):
            if promote_default:
                self._clear_model_trial_state(user_id)
                if isinstance(default_body, dict):
                    current_after = truth.current_chat_target_status()
                    active_provider_after, active_model_after = self._target_snapshot_from_truth(current_after)
                    if active_model_after == applied_model and (not applied_provider or active_provider_after == applied_provider):
                        default_body["message"] = f"{applied_model} is now the default chat model, and chat is now using it."
                    elif active_model_after:
                        default_body["message"] = (
                            f"{applied_model} is now the default chat model. "
                            f"Chat is still using {active_model_after}."
                        )
                    else:
                        default_body["message"] = f"{applied_model} is now the default chat model."
            else:
                self._record_model_trial_switch(
                    user_id,
                    previous_provider=previous_provider,
                    previous_model=previous_model,
                    applied_provider=applied_provider,
                    applied_model=applied_model,
                    source="direct_switch",
                )
        if promote_default:
            body_dict = dict(default_body) if isinstance(default_body, dict) else {}
            message = str(body_dict.get("message") or f"{applied_model or matched_model} is now the default chat model.").strip()
            return self._runtime_truth_response(
                text=message,
                route="model_status",
                used_memory=used_memory,
                used_tools=["model_controller"],
                error_kind=None if default_ok else str(body_dict.get("error") or "set_default_failed").strip() or "set_default_failed",
                ok=bool(default_ok),
                payload={
                    "type": "model_controller" if default_ok else "provider_test_result",
                    "provider": applied_provider,
                    "model_id": applied_model,
                    "ok": bool(default_ok),
                    "title": "Default model updated" if default_ok else "Default model update failed",
                    "summary": message,
                },
            )
        return self._post_switch_response(
            truth=truth,
            route="model_status",
            used_memory=used_memory,
            used_tools=["model_controller"],
            ok=bool(default_ok),
            body=default_body if isinstance(default_body, dict) else {},
            applied_provider=applied_provider,
            applied_model=applied_model,
            success_type="setup_complete",
            success_title="Default model updated",
            failure_title="Model switch failed",
        )

    def _model_controller_promote_default_response(self, user_id: str, text: str) -> OrchestratorResponse:
        state = self._current_runtime_setup_state(user_id)
        return self._set_default_model_response(user_id, text, state)

    def _execute_model_controller_switch_back(self, user_id: str) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="model_status",
                reason="runtime_truth_service_unavailable",
            )
        state = self._current_model_trial_state(user_id)
        previous_model = str(state.get("previous_model") or "").strip() or None
        previous_provider = str(state.get("previous_provider") or "").strip().lower() or None
        source = str(state.get("source") or "").strip().lower()
        if not previous_model:
            current = truth.current_chat_target_status()
            current_model = str(current.get("model") or "").strip() or "the current model"
            message = (
                f"I do not have a recent trial model switch to roll back. Right now chat is using {current_model}. "
                "Switch back only undoes a recent temporary trial switch. Changing the default alone does not create a trial rollback."
            )
            return self._runtime_truth_response(
                text=message,
                route="model_status",
                payload={
                    "type": "model_switch",
                    "ok": False,
                    "title": "No previous model",
                    "summary": message,
                },
            )
        if source == "temporary_switch":
            temporary_restorer = getattr(truth, "restore_temporary_chat_model_target", None)
            if callable(temporary_restorer):
                switch_ok, switch_body = temporary_restorer(previous_model, provider_id=previous_provider)
            else:
                explicit_setter = getattr(truth, "set_confirmed_chat_model_target", None)
                if callable(explicit_setter):
                    switch_ok, switch_body = explicit_setter(previous_model, provider_id=previous_provider)
                else:
                    switch_ok, switch_body = truth.set_default_chat_model(previous_model)
        else:
            explicit_setter = getattr(truth, "set_confirmed_chat_model_target", None)
            if callable(explicit_setter):
                switch_ok, switch_body = explicit_setter(previous_model, provider_id=previous_provider)
            else:
                switch_ok, switch_body = truth.set_default_chat_model(previous_model)
        if switch_ok:
            self._clear_model_trial_state(user_id)
        message = str((switch_body if isinstance(switch_body, dict) else {}).get("message") or f"Now using {previous_model} for chat.")
        return self._runtime_truth_response(
            text=message,
            route="model_status",
            used_tools=["model_controller"],
            error_kind=None if switch_ok else str((switch_body if isinstance(switch_body, dict) else {}).get("error") or "switch_back_failed").strip() or "switch_back_failed",
            ok=bool(switch_ok),
            payload={
                "type": "model_switch",
                "provider": str((switch_body if isinstance(switch_body, dict) else {}).get("provider") or previous_provider).strip().lower() or previous_provider,
                "model_id": str((switch_body if isinstance(switch_body, dict) else {}).get("model_id") or previous_model).strip() or previous_model,
                "ok": bool(switch_ok),
                "title": "Previous model restored" if switch_ok else "Switch back failed",
                "summary": message,
            },
        )

    @staticmethod
    def _model_scout_followup_requested(normalized_text: str) -> bool:
        return bool(_MODEL_SCOUT_FOLLOWUP_RE.search(str(normalized_text or "")))

    def _looks_like_model_scout_action(self, text: str, *, context: dict[str, Any] | None = None) -> bool:
        normalized = normalize_setup_text(text).replace("/", " ")
        if not normalized:
            return False
        if "model scout" in normalized:
            return True
        if self._model_scout_strategy_requested(normalized):
            return True
        if self._model_scout_discovery_requested(normalized):
            return True
        if self._model_scout_followup_requested(normalized):
            return self._is_model_context(context or {})
        has_action_verb = any(phrase in normalized for phrase in _MODEL_SCOUT_ACTION_VERBS)
        has_model_noun = any(token in normalized for token in ("model", "models"))
        has_quality_hint = any(phrase in normalized for phrase in _MODEL_SCOUT_QUALITY_HINTS)
        return bool(has_action_verb and has_model_noun and has_quality_hint)

    def _model_scout_action_response(self, user_id: str, text: str) -> OrchestratorResponse | None:
        context = self._current_interpretable_result(user_id)
        if not self._looks_like_model_scout_action(text, context=context):
            return None
        normalized = normalize_setup_text(text).replace("/", " ")
        focus_terms = self._model_scout_focus_terms(text, context)
        if self._model_scout_discovery_requested(normalized):
            return self._model_scout_discovery_response(text)
        if self._model_scout_followup_requested(normalized) and not focus_terms and not self._is_model_context(context):
            question = "Do you want me to run Model Scout on the models we were just discussing?"
            return self._runtime_truth_response(
                text=question,
                route="action_tool",
                used_runtime_state=False,
                payload={
                    "type": "action_clarification",
                    "kind": "model_scout",
                    "summary": question,
                    "next_question": question,
                },
                next_question=question,
            )
        if self._model_scout_strategy_requested(normalized) or ("model scout" in normalized and not focus_terms):
            return self._model_scout_strategy_response(user_id, text)
        return self._model_scout_inventory_response(focus_terms=focus_terms)

    def _handle_action_tool_intent(self, user_id: str, text: str) -> OrchestratorResponse | None:
        intent_kind = self._time_date_intent_kind(text)
        if intent_kind is not None:
            return self._local_time_response(intent_kind)
        control_mode_response = self._control_mode_intent_response(text)
        if control_mode_response is not None:
            return control_mode_response
        normalized = normalize_setup_text(text).replace("/", " ")
        if self._model_switch_advisory_requested(normalized):
            return self._model_switch_advisory_response()
        if self._model_trial_switch_back_requested(normalized):
            return self._model_controller_switch_back_response(user_id)
        if self._model_acquisition_requested(normalized):
            return self._model_acquire_response(user_id, text)
        if self._model_ready_now_requested(text):
            return self._model_ready_now_response()
        if self._model_controller_test_requested(text):
            return self._model_controller_test_response(user_id, text)
        if self._model_controller_trial_switch_requested(text):
            return self._model_controller_trial_switch_response(user_id, text)
        if self._model_controller_promote_requested(text):
            return self._model_controller_promote_default_response(user_id, text)
        model_scout_response = self._model_scout_action_response(user_id, text)
        if model_scout_response is not None:
            return model_scout_response
        return None

    def _grounded_system_fallback_response(
        self,
        user_id: str,
        text: str,
        *,
        allow_actions: bool,
    ) -> OrchestratorResponse | None:
        normalized = normalize_setup_text(text)
        if not normalized or not self._looks_like_grounded_system_query(text):
            return None
        provider_hint = self._mentioned_provider_id(normalized)
        state = self._current_runtime_setup_state(user_id)
        if _looks_like_model_lifecycle_query(normalized):
            return self._model_lifecycle_response(text)
        if _looks_like_local_model_inventory_query(normalized):
            return self._model_inventory_response(local_only=True, provider_id=provider_hint)
        if _looks_like_model_availability_query(normalized):
            normalized_space = normalized.replace("/", " ")
            remote_only = any(token in normalized_space for token in ("cloud", "remote")) and any(
                token in normalized_space for token in ("model", "models")
            )
            return self._model_inventory_response(local_only=False, remote_only=remote_only)
        if _looks_like_current_model_query(normalized):
            return self._current_model_response()
        if _looks_like_runtime_status_query(normalized):
            return self._runtime_status_response("runtime_status")
        if provider_hint and any(
            token in normalized
            for token in ("status", "health", "configured", "working", "ready")
        ):
            return self._provider_status_response(provider_hint)
        if allow_actions and self._model_trial_switch_back_requested(normalized.replace("/", " ")):
            return self._model_controller_switch_back_response(user_id)
        if allow_actions and any(
            phrase in normalized.replace("/", " ")
            for phrase in ("switch to ", "switch chat to ", "change to ", "use ")
        ):
            resolution = self._resolve_runtime_model_target(text)
            if str(resolution.get("status") or "").strip().lower() in {"unique", "ambiguous"}:
                return self._set_default_model_response(user_id, text, state)
        return None

    def _safe_mode_containment_response(
        self,
        user_id: str,
        text: str,
    ) -> OrchestratorResponse | None:
        if str(text or "").strip().startswith("/"):
            return None
        if not self._assistant_frontdoor_engaged(text):
            return None
        if self._runtime_truth() is None:
            return None

        state = self._current_runtime_setup_state(user_id)
        normalized = normalize_setup_text(text)

        if self._repair_context_handoff_requested(text) and self._recent_unhealthy_runtime_context(user_id):
            repair_response = self._repair_context_handoff_response(user_id, text)
            if repair_response is not None:
                return repair_response
            return self._setup_explanation_response(used_memory=bool(state))

        if self._repair_context_handoff_requested(text):
            return self._setup_explanation_response(used_memory=bool(state))

        if _looks_like_setup_explanation_query(normalized):
            return self._setup_explanation_response(used_memory=bool(state))

        if self._looks_like_grounded_system_query(text):
            grounded = self._grounded_system_fallback_response(
                user_id,
                text,
                allow_actions=True,
            )
            if grounded is not None:
                return grounded
            return self._runtime_state_unavailable_response(
                route="runtime_status",
                used_memory=bool(state),
                reason="safe_mode_containment_blocked_generic_escape",
            )
        return None

    @staticmethod
    def _assistant_unmatched_words(text: str) -> list[str]:
        normalized = normalize_setup_text(text).replace("/", " ")
        return [piece for piece in normalized.split(" ") if piece]

    def _looks_like_runtime_overview_prompt(self, text: str) -> bool:
        normalized = normalize_setup_text(text).replace("/", " ")
        words = set(self._assistant_unmatched_words(text))
        if not normalized:
            return False
        if any(
            phrase in normalized
            for phrase in (
                "what is happening",
                "whats happening",
                "what is going on",
                "whats going on",
            )
        ):
            return True
        return bool(words & {"system", "runtime", "agent"} and words & {"status", "report", "health"})

    def _looks_like_short_help_prompt(self, text: str) -> bool:
        words = self._assistant_unmatched_words(text)
        if not words or len(words) > 3:
            return False
        return bool(set(words) & {"help", "fix", "repair"})

    def _looks_like_placeholder_action_prompt(self, text: str) -> bool:
        words = set(self._assistant_unmatched_words(text))
        if not words or len(words) > 4:
            return False
        placeholders = {"thing", "things", "this", "that", "it", "stuff", "something"}
        action_verbs = {"do", "run", "make", "fix", "repair", "handle", "check"}
        if "status" in words and words & placeholders:
            return True
        return bool(words & action_verbs and words & placeholders)

    def _assistant_unmatched_clarification_response(
        self,
        *,
        used_memory: bool,
        reason: str,
    ) -> OrchestratorResponse:
        question = "Do you want runtime status, model status, setup help, or a direct task?"
        return self._runtime_truth_response(
            text=question,
            route="assistant_clarification",
            used_memory=used_memory,
            used_runtime_state=False,
            next_question=question,
            payload={
                "type": "assistant_unmatched_clarification",
                "reason": str(reason or "unclear_input").strip() or "unclear_input",
                "summary": question,
            },
        )

    def _assistant_unmatched_fallback_response(
        self,
        *,
        used_memory: bool,
        reason: str,
    ) -> OrchestratorResponse:
        message = (
            "I’m not sure what you want yet. Ask about runtime, models, setup, "
            "system status, or tell me the task directly."
        )
        return self._runtime_truth_response(
            text=message,
            route="assistant_fallback",
            used_memory=used_memory,
            used_runtime_state=False,
            payload={
                "type": "assistant_unmatched_fallback",
                "reason": str(reason or "unclear_input").strip() or "unclear_input",
                "summary": message,
            },
        )

    def _assistant_unmatched_input_response(
        self,
        user_id: str,
        text: str,
    ) -> OrchestratorResponse | None:
        if str(text or "").strip().startswith("/"):
            return None
        if not self._assistant_frontdoor_engaged(text):
            return None
        used_memory = bool(self._current_runtime_setup_state(user_id))
        external_pack_response = self._external_pack_knowledge_response(user_id, text)
        if external_pack_response is not None:
            return external_pack_response
        truth = self._runtime_truth()
        normalized = normalize_setup_text(text).replace("/", " ")

        if self._model_scout_strategy_requested(normalized):
            return self._model_scout_strategy_response(user_id, text)

        if self._looks_like_runtime_overview_prompt(text) and truth is not None:
            return self._runtime_status_response("runtime_status")

        if self._looks_like_short_help_prompt(text):
            if truth is not None:
                setup = truth.setup_status()
                setup_state = str(setup.get("setup_state") or "").strip().lower() or "unavailable"
                if setup_state != "ready":
                    return self._setup_explanation_response(used_memory=used_memory)
            return self._assistant_unmatched_clarification_response(
                used_memory=used_memory,
                reason="short_help_prompt",
            )

        if self._looks_like_placeholder_action_prompt(text):
            return self._assistant_unmatched_clarification_response(
                used_memory=used_memory,
                reason="placeholder_action_prompt",
            )

        low_confidence = detect_low_confidence(text)
        nl_intent = str((nl_route(text) or {}).get("intent") or "").strip().upper()
        if low_confidence.is_low_confidence and nl_intent != "CHITCHAT":
            if str(low_confidence.reason or "").strip().lower() in {"no_semantic_tokens", "repetition_spam"}:
                return self._assistant_unmatched_fallback_response(
                    used_memory=used_memory,
                    reason=str(low_confidence.reason or "unclear_input"),
                )
            return self._assistant_unmatched_clarification_response(
                used_memory=used_memory,
                reason=str(low_confidence.reason or "unclear_input"),
            )
        return None

    @staticmethod
    def _external_pack_snippet(text: str | None, *, max_chars: int = 260) -> str:
        cleaned = " ".join(str(text or "").strip().split())
        if not cleaned:
            return ""
        if len(cleaned) <= max_chars:
            return cleaned
        cut = cleaned[:max_chars].rsplit(" ", 1)[0].strip()
        return cut + "..."

    @staticmethod
    def _external_pack_display_name(value: str | None) -> str:
        cleaned = " ".join(str(value or "").strip().replace("_", "-").split())
        if not cleaned:
            return "Imported pack"
        if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)+", cleaned):
            return " ".join(part.capitalize() for part in cleaned.split("-"))
        return cleaned

    @staticmethod
    def _external_pack_query_tokens(text: str | None) -> list[str]:
        normalized = normalize_setup_text(text).replace("/", " ")
        stopwords = {
            "a",
            "an",
            "and",
            "for",
            "how",
            "i",
            "is",
            "it",
            "me",
            "my",
            "of",
            "on",
            "or",
            "should",
            "the",
            "to",
            "use",
            "what",
            "when",
            "with",
            "you",
        }
        return [
            token
            for token in re.findall(r"[a-z0-9][a-z0-9._-]*", normalized)
            if token and token not in stopwords
        ]

    @staticmethod
    def _external_pack_sections(skill_text: str) -> dict[str, str]:
        sections: dict[str, list[str]] = {}
        current: str | None = None
        for raw_line in str(skill_text or "").splitlines():
            match = re.match(r"^#{1,3}\s+(?P<title>.+?)\s*$", raw_line.strip())
            if match is not None:
                current = str(match.group("title") or "").strip().lower()
                sections.setdefault(current, [])
                continue
            if current is not None:
                sections.setdefault(current, []).append(raw_line)
        return {
            key: " ".join(line.strip() for line in lines if line.strip()).strip()
            for key, lines in sections.items()
        }

    def _external_pack_knowledge_response(self, user_id: str, text: str) -> OrchestratorResponse | None:
        query_tokens = self._external_pack_query_tokens(text)
        if not query_tokens:
            return None
        normalized_query = normalize_setup_text(text).replace("/", " ")
        removed_packs = self._pack_store.list_external_pack_removals()
        removed_best: dict[str, Any] | None = None
        removed_best_score = 0.0
        for row in removed_packs:
            if not isinstance(row, dict):
                continue
            review_envelope = row.get("review_envelope") if isinstance(row.get("review_envelope"), dict) else {}
            canonical_pack = row.get("canonical_pack") if isinstance(row.get("canonical_pack"), dict) else {}
            canonical_source = canonical_pack.get("source") if isinstance(canonical_pack.get("source"), dict) else {}
            skill_text = str(row.get("skill_text") or "").strip()
            pack_name_raw = str(
                review_envelope.get("pack_name")
                or canonical_pack.get("display_name")
                or canonical_pack.get("name")
                or canonical_source.get("display_name")
                or canonical_source.get("name")
                or canonical_source.get("title")
                or canonical_source.get("repo")
                or ""
            ).strip()
            pack_name = self._external_pack_display_name(pack_name_raw)
            summary = str(
                canonical_pack.get("capabilities", {}).get("summary")
                if isinstance(canonical_pack.get("capabilities"), dict)
                else ""
            ).strip()
            sections = self._external_pack_sections(skill_text) if skill_text else {}
            corpus = " ".join(
                part
                for part in (
                    pack_name,
                    summary,
                    sections.get("purpose"),
                    sections.get("when to use"),
                    sections.get("inputs"),
                    sections.get("behavior"),
                    sections.get("constraints"),
                    sections.get("response style"),
                    sections.get("example prompts"),
                    str(canonical_source.get("display_name") or ""),
                    str(canonical_source.get("name") or ""),
                    str(canonical_source.get("title") or ""),
                    str(canonical_source.get("repo") or ""),
                    str(canonical_source.get("origin") or ""),
                    str(row.get("reason") or ""),
                    str(row.get("removed_by") or ""),
                )
                if part
            ).lower()
            score = 0.0
            if pack_name and pack_name.lower() in normalized_query:
                score += 1.0
            score += min(0.4, 0.1 * sum(1 for token in query_tokens if token in corpus))
            if score > removed_best_score:
                removed_best_score = score
                removed_best = {
                    "row": row,
                    "pack_name": pack_name,
                    "summary": summary,
                }
        if removed_best is not None and removed_best_score >= 0.35:
            row = removed_best["row"]
            pack_name = str(removed_best.get("pack_name") or "").strip() or "That pack"
            summary = str(removed_best.get("summary") or "").strip()
            lines = [
                f"{pack_name} was removed, so I can’t use it through chat anymore.",
            ]
            if summary:
                lines.append(f"Removed pack summary: {summary}.")
            lines.append("Reinstall it if you want to use it again, or ask /packs to see what is still installed.")
            message = " ".join(lines).strip()
            return self._runtime_truth_response(
                text=message,
                route="action_tool",
                used_runtime_state=False,
                used_memory=bool(self._current_runtime_setup_state(user_id)),
                used_tools=["external_pack_lookup"],
                payload={
                    "type": "external_pack_removed_notice",
                    "summary": message,
                    "pack_id": str(row.get("pack_id") or row.get("canonical_id") or "").strip() or None,
                    "pack_name": pack_name,
                    "status": "removed",
                },
            )
        packs = self._pack_store.list_external_packs()
        best: dict[str, Any] | None = None
        best_score = 0.0
        for row in packs:
            if not isinstance(row, dict):
                continue
            pack_id = str(row.get("pack_id") or row.get("canonical_id") or "").strip()
            if pack_id and callable(getattr(self._pack_store, "get_external_pack_removal", None)):
                try:
                    if self._pack_store.get_external_pack_removal(pack_id) is not None:
                        continue
                except Exception:
                    pass
            review_envelope = row.get("review_envelope") if isinstance(row.get("review_envelope"), dict) else {}
            pack_name_raw = str(
                review_envelope.get("pack_name")
                or (row.get("canonical_pack") if isinstance(row.get("canonical_pack"), dict) else {}).get("display_name")
                or row.get("name")
                or ""
            ).strip()
            pack_name = self._external_pack_display_name(pack_name_raw)
            canonical_pack = row.get("canonical_pack") if isinstance(row.get("canonical_pack"), dict) else {}
            summary = str(
                (canonical_pack.get("capabilities") if isinstance(canonical_pack.get("capabilities"), dict) else {}).get("summary")
                or row.get("summary")
                or row.get("review_summary")
                or ""
            ).strip()
            normalized_path = str(row.get("normalized_path") or "").strip()
            skill_text = ""
            if normalized_path:
                try:
                    skill_path = Path(normalized_path) / "SKILL.md"
                    skill_text = skill_path.read_text(encoding="utf-8")
                except OSError:
                    skill_text = ""
            sections = self._external_pack_sections(skill_text)
            corpus = " ".join(
                part
                for part in (
                    pack_name,
                    summary,
                    sections.get("purpose"),
                    sections.get("when to use"),
                    sections.get("inputs"),
                    sections.get("behavior"),
                    sections.get("constraints"),
                    sections.get("response style"),
                    sections.get("example prompts"),
                    str(row.get("classification") or ""),
                )
                if part
            ).lower()
            if not corpus:
                continue
            score = 0.0
            if pack_name and pack_name.lower() in normalized_query:
                score += 0.75
            if summary and any(token in summary.lower() for token in query_tokens):
                score += 0.18
            score += min(0.35, 0.08 * sum(1 for token in query_tokens if token in corpus))
            if str(row.get("status") or "").strip().lower() == "blocked":
                score -= 0.06
            if score > best_score:
                best_score = score
                best = {
                    "row": row,
                    "sections": sections,
                    "summary": summary,
                    "score": score,
                }
        if best is None or best_score < 0.22:
            return None
        row = best["row"]
        sections = best["sections"] if isinstance(best.get("sections"), dict) else {}
        review_envelope = row.get("review_envelope") if isinstance(row.get("review_envelope"), dict) else {}
        pack_name = self._external_pack_display_name(
            str(
                review_envelope.get("pack_name")
                or (row.get("canonical_pack") if isinstance(row.get("canonical_pack"), dict) else {}).get("display_name")
                or row.get("name")
                or ""
            ).strip()
        )
        status = str(row.get("status") or "").strip().lower()
        summary = str(best.get("summary") or "").strip()
        purpose = self._external_pack_snippet(sections.get("purpose") or summary or f"{pack_name} is a safe imported text pack.")
        when_to_use = self._external_pack_snippet(sections.get("when to use"))
        inputs = self._external_pack_snippet(sections.get("inputs"))
        behavior = self._external_pack_snippet(sections.get("behavior"))
        constraints = self._external_pack_snippet(sections.get("constraints"))
        examples = self._external_pack_snippet(sections.get("example prompts"))
        lines: list[str] = []
        if status == "blocked":
            lines.append(f"{pack_name} is blocked, so I cannot use it through chat.")
            if summary:
                lines.append(f"Reason: {summary}.")
        else:
            lines.append(f"{pack_name} is a safe imported text pack for {purpose}.")
            if when_to_use:
                lines.append(f"Use it when {when_to_use}.")
            if inputs:
                lines.append(f"Inputs: {inputs}.")
            if behavior:
                lines.append(f"Behavior: {behavior}.")
            if constraints:
                lines.append(f"Constraints: {constraints}.")
            if examples:
                lines.append(f"Example prompts: {examples}.")
            if status == "partial_safe_import":
                lines.append("Some unsafe files were stripped during import.")
        message = " ".join(lines).strip()
        return self._runtime_truth_response(
            text=message,
            route="action_tool",
            used_runtime_state=False,
            used_memory=bool(self._current_runtime_setup_state(user_id)),
            used_tools=["external_pack_lookup"],
            payload={
                "type": "external_pack_knowledge",
                "summary": message,
                "pack_id": str(row.get("pack_id") or row.get("canonical_id") or "").strip() or None,
                "pack_name": pack_name,
                "status": status or None,
            },
        )

    def forget_external_pack_activation(self, *, pack_id: str | None = None, pack_name: str | None = None) -> int:
        normalized_pack_id = str(pack_id or "").strip()
        normalized_pack_name = self._external_pack_display_name(pack_name).lower()
        if not normalized_pack_id and not normalized_pack_name:
            return 0
        removed = 0
        for user_key, state in list(self._last_interpretable_result.items()):
            if not isinstance(state, dict):
                continue
            payload = state.get("payload") if isinstance(state.get("payload"), dict) else {}
            if str(payload.get("type") or "").strip().lower() != "external_pack_knowledge":
                continue
            payload_pack_id = str(payload.get("pack_id") or "").strip()
            payload_pack_name = self._external_pack_display_name(str(payload.get("pack_name") or "")).lower()
            response_text = str(state.get("response_text") or "").strip().lower()
            if normalized_pack_id and payload_pack_id == normalized_pack_id:
                self._last_interpretable_result.pop(user_key, None)
                removed += 1
                continue
            if normalized_pack_name and (
                payload_pack_name == normalized_pack_name or normalized_pack_name in response_text
            ):
                self._last_interpretable_result.pop(user_key, None)
                removed += 1
        return removed

    def _current_runtime_setup_state(self, user_id: str) -> dict[str, Any]:
        state = self._runtime_setup_state.get(user_id)
        if not isinstance(state, dict):
            return {}
        created_ts = int(state.get("created_ts") or 0)
        if created_ts and int(time.time()) - created_ts > _RUNTIME_SETUP_STATE_TTL_SECONDS:
            self._runtime_setup_state.pop(user_id, None)
            return {}
        return dict(state)

    def _save_runtime_setup_state(self, user_id: str, state: dict[str, Any]) -> None:
        self._runtime_setup_state[user_id] = {
            **dict(state),
            "created_ts": int(time.time()),
        }

    def _clear_runtime_setup_state(self, user_id: str) -> None:
        self._runtime_setup_state.pop(user_id, None)

    def _current_model_trial_state(self, user_id: str) -> dict[str, Any]:
        state = self._model_trial_state.get(user_id)
        if not isinstance(state, dict):
            return {}
        created_ts = int(state.get("created_ts") or 0)
        if created_ts and int(time.time()) - created_ts > _MODEL_TRIAL_STATE_TTL_SECONDS:
            self._model_trial_state.pop(user_id, None)
            return {}
        return dict(state)

    def _save_model_trial_state(self, user_id: str, state: dict[str, Any]) -> None:
        self._model_trial_state[user_id] = {
            **dict(state),
            "created_ts": int(time.time()),
        }

    def _clear_model_trial_state(self, user_id: str) -> None:
        self._model_trial_state.pop(user_id, None)

    @staticmethod
    def _target_snapshot_from_truth(current: dict[str, Any] | None) -> tuple[str | None, str | None]:
        payload = dict(current) if isinstance(current, dict) else {}
        provider = (
            str(payload.get("effective_provider") or payload.get("provider") or "").strip().lower()
            or None
        )
        model = str(payload.get("effective_model") or payload.get("model") or "").strip() or None
        return provider, model

    def _record_model_trial_switch(
        self,
        user_id: str,
        *,
        previous_provider: str | None,
        previous_model: str | None,
        applied_provider: str | None,
        applied_model: str | None,
        source: str,
    ) -> None:
        previous_target = str(previous_model or "").strip() or None
        applied_target = str(applied_model or "").strip() or None
        if not previous_target or not applied_target or previous_target == applied_target:
            return
        self._save_model_trial_state(
            user_id,
            {
                "trial_switch_active": True,
                "previous_provider": str(previous_provider or "").strip().lower() or None,
                "previous_model": previous_target,
                "current_provider": str(applied_provider or "").strip().lower() or None,
                "current_model": applied_target,
                "source": source,
            },
        )

    @staticmethod
    def _post_switch_health_note(
        *,
        provider_label: str,
        provider_health_status: str | None,
        model_health_status: str | None,
    ) -> str:
        provider_state = str(provider_health_status or "").strip().lower()
        model_state = str(model_health_status or "").strip().lower()
        if provider_state == "down":
            return f" {provider_label} is not responding right now."
        if provider_state == "degraded":
            return f" {provider_label} needs attention right now."
        if model_state == "down":
            return " That model is not responding properly right now."
        if model_state == "degraded":
            return " That model looks degraded right now."
        return ""

    def _post_switch_response(
        self,
        *,
        truth: Any,
        route: str,
        used_memory: bool,
        used_tools: list[str] | None = None,
        ok: bool,
        body: dict[str, Any] | None,
        applied_provider: str | None,
        applied_model: str | None,
        success_type: str,
        success_title: str,
        failure_title: str,
    ) -> OrchestratorResponse:
        body_dict = dict(body) if isinstance(body, dict) else {}
        message = str(body_dict.get("message") or "I switched the chat model.").strip() or "I switched the chat model."
        error_kind = None if ok else str(body_dict.get("error") or "bad_request").strip() or "bad_request"
        next_question = None
        provider_health_status = None
        model_health_status = None
        if ok and applied_model:
            current = truth.current_chat_target_status()
            current_provider, current_model = self._target_snapshot_from_truth(current)
            provider_health_status = str(current.get("provider_health_status") or "").strip().lower() or None
            model_health_status = str(current.get("health_status") or "").strip().lower() or None
            ready = bool(
                current_model == applied_model
                and (not applied_provider or current_provider == applied_provider)
                and bool(current.get("ready", False))
            )
            if not ready:
                provider_label = self._setup_provider_label(applied_provider)
                message = (
                    f"I switched to {applied_model}, but it isn't responding properly right now."
                    f"{self._post_switch_health_note(provider_label=provider_label, provider_health_status=provider_health_status, model_health_status=model_health_status)} "
                    "Do you want me to switch back?"
                ).strip()
                next_question = message
                success_title = "Model switched but unhealthy"
        return self._runtime_truth_response(
            text=message,
            route=route,
            used_memory=used_memory,
            used_tools=used_tools,
            error_kind=error_kind,
            ok=bool(ok),
            next_question=next_question,
            payload={
                "type": success_type if ok else "provider_test_result",
                "provider": applied_provider,
                "model_id": applied_model,
                "ok": bool(ok),
                "title": success_title if ok else failure_title,
                "summary": message,
                "provider_health_status": provider_health_status,
                "model_health_status": model_health_status,
            },
        )

    def runtime_setup_state_hint(self, user_id: str) -> dict[str, Any]:
        state = self._current_runtime_setup_state(user_id)
        step = str(state.get("step") or "").strip().lower()
        if not step:
            return {}
        awaiting_secret = step == "awaiting_openrouter_key"
        awaiting_confirmation = step in {"awaiting_switch_confirm", "awaiting_openrouter_reuse_confirm"}
        if not awaiting_secret and not awaiting_confirmation:
            return {}
        return {
            "route": "setup_flow",
            "step": step,
            "awaiting_secret": awaiting_secret,
            "awaiting_confirmation": awaiting_confirmation,
        }

    @staticmethod
    def _setup_provider_label(provider_id: str | None) -> str:
        normalized = str(provider_id or "").strip().lower()
        if normalized == "openrouter":
            return "OpenRouter"
        if normalized == "openai":
            return "OpenAI"
        if normalized == "ollama":
            return "Ollama"
        if not normalized:
            return "Unknown provider"
        return normalized.replace("_", " ").title()

    def _runtime_model_catalog(self) -> list[str]:
        truth = self._runtime_truth()
        if truth is None:
            return []
        model_ids: list[str] = []
        seen: set[str] = set()
        payload = self._canonical_model_inventory_snapshot(truth)
        rows = payload.get("models") if isinstance(payload.get("models"), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            candidate = str(row.get("model_id") or row.get("id") or "").strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            model_ids.append(candidate)
        if model_ids:
            return model_ids
        status_payload = truth.providers_status()
        rows = status_payload.get("providers") if isinstance(status_payload.get("providers"), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            for model_id in (row.get("model_ids") if isinstance(row.get("model_ids"), list) else []):
                candidate = str(model_id or "").strip()
                if not candidate or candidate in seen:
                    continue
                seen.add(candidate)
                model_ids.append(candidate)
        return model_ids

    def _resolve_runtime_model_target(self, text: str) -> dict[str, Any]:
        normalized = normalize_setup_text(text)
        if not normalized:
            return {"status": "none", "requested": None, "model_id": None, "matches": []}
        model_ids = self._runtime_model_catalog()
        if not model_ids:
            return {"status": "none", "requested": None, "model_id": None, "matches": []}

        candidate_tokens: list[str] = []
        seen_tokens: set[str] = set()
        for match in _DIRECT_MODEL_SWITCH_TOKEN_RE.finditer(normalized):
            candidate = str(match.group(0) or "").strip().lower()
            if not candidate or candidate in seen_tokens:
                continue
            seen_tokens.add(candidate)
            candidate_tokens.append(candidate)

        def _unique(values: list[str]) -> list[str]:
            seen_values: set[str] = set()
            deduped: list[str] = []
            for value in values:
                candidate = str(value or "").strip()
                if not candidate or candidate in seen_values:
                    continue
                seen_values.add(candidate)
                deduped.append(candidate)
            return deduped

        for token in candidate_tokens:
            exact_matches = _unique(
                [model_id for model_id in model_ids if normalize_setup_text(model_id) == token]
            )
            if len(exact_matches) == 1:
                provider_id = str(exact_matches[0].split(":", 1)[0]).strip().lower() if ":" in exact_matches[0] else None
                return {
                    "status": "unique",
                    "requested": token,
                    "model_id": exact_matches[0],
                    "provider_id": provider_id,
                    "matches": exact_matches,
                }
            bare_matches = _unique(
                [
                    model_id
                    for model_id in model_ids
                    if ":" in model_id and normalize_setup_text(model_id.split(":", 1)[1]) == token
                ]
            )
            if len(bare_matches) == 1:
                provider_id = str(bare_matches[0].split(":", 1)[0]).strip().lower() if ":" in bare_matches[0] else None
                return {
                    "status": "unique",
                    "requested": token,
                    "model_id": bare_matches[0],
                    "provider_id": provider_id,
                    "matches": bare_matches,
                }
            if len(bare_matches) > 1:
                return {
                    "status": "ambiguous",
                    "requested": token,
                    "model_id": None,
                    "matches": bare_matches,
                }

        fallback = self._match_runtime_model_from_text(text)
        if fallback:
            provider_id = str(fallback.split(":", 1)[0]).strip().lower() if ":" in fallback else None
            return {
                "status": "unique",
                "requested": None,
                "model_id": fallback,
                "provider_id": provider_id,
                "matches": [fallback],
            }
        return {"status": "none", "requested": None, "model_id": None, "provider_id": None, "matches": []}

    def _resolve_controller_model_target(self, user_id: str, text: str) -> dict[str, Any]:
        resolution = self._resolve_runtime_model_target(text)
        status = str(resolution.get("status") or "").strip().lower()
        if status in {"unique", "ambiguous"}:
            return resolution
        context = self._current_interpretable_result(user_id)
        if self._is_model_context(context):
            model_id, provider_id = self._preferred_model_context_target(context)
            if model_id:
                return {
                    "status": "unique",
                    "requested": "context_model",
                    "model_id": model_id,
                    "provider_id": provider_id,
                    "matches": [model_id],
                    "source": "recent_model_context",
                }
        return resolution

    def _match_runtime_model_from_text(self, text: str) -> str | None:
        normalized = normalize_setup_text(text)
        if not normalized:
            return None
        for candidate_id in self._runtime_model_catalog():
            candidate_model = candidate_id.split(":", 1)[1] if ":" in candidate_id else candidate_id
            for token in (candidate_id, candidate_model):
                token_normalized = normalize_setup_text(token)
                if token_normalized and token_normalized in normalized:
                    return candidate_id
        return None

    def _provider_status_message(
        self,
        *,
        provider_key: str,
        provider_label: str,
        provider_known: bool,
        provider_enabled: bool,
        model_id: str | None,
        current_model_id: str | None,
        configured: bool,
        active: bool,
        health_status: str,
        health_reason: str | None,
        secret_present: bool,
        effective_provider: str | None,
        effective_model_id: str | None,
    ) -> str:
        if provider_key == "ollama":
            current_label = model_id or current_model_id or "none"
            if health_status == "ok":
                message = f"Ollama is reachable. Chat is configured for {current_label}, and it looks healthy."
            elif health_status == "degraded":
                message = (
                    f"Ollama is reachable but degraded. Chat is configured for {current_label}, "
                    "and it needs attention right now."
                )
            elif health_status == "down":
                message = (
                    f"Ollama is currently down. Chat is configured for {current_label}, "
                    "but it is not responding right now."
                )
            elif configured:
                message = (
                    f"Ollama is configured for {current_label}, but I couldn't verify its health right now."
                )
            else:
                message = f"Ollama is not set up for chat yet. Health status: {health_status}."
            if health_reason:
                message = f"{message.rstrip('.')} Reason: {health_reason}."
            return message

        if configured and active and model_id and health_status == "ok":
            message = f"Yes. Chat is currently using {provider_label} with {model_id}."
        elif configured and active and model_id and effective_model_id and effective_provider and effective_provider != provider_key:
            effective_provider_label = self._setup_provider_label(effective_provider)
            message = (
                f"{provider_label} is configured for chat with {model_id}, but it is not healthy right now. "
                f"The best healthy target would be {effective_model_id} on {effective_provider_label}."
            )
        elif configured and active and model_id:
            message = f"{provider_label} is configured for chat with {model_id}, but it is not healthy right now."
        elif configured and model_id and current_model_id:
            message = f"{provider_label} is set up with {model_id}. Chat is currently using {current_model_id}."
        elif configured and model_id:
            message = f"{provider_label} is set up with {model_id}."
        elif configured:
            message = f"{provider_label} is set up for chat."
        elif provider_key == "openrouter" and not secret_present:
            message = "OpenRouter is not set up yet. I still need your API key."
        elif provider_known and provider_enabled:
            message = f"{provider_label} is partly set up, but I do not have a chat model ready for it yet."
        else:
            message = f"{provider_label} is not set up for chat yet."

        if configured and health_status in {"down", "degraded"}:
            health_note = (
                f"{provider_label} is configured, but it is not responding right now."
                if health_status == "down"
                else f"{provider_label} is configured, but it needs attention right now."
            )
            if health_note not in message:
                message = f"{message.rstrip('.')} {health_note}"
        if health_reason and health_reason not in message:
            message = f"{message.rstrip('.')} Reason: {health_reason}."
        return message

    def _repair_followup_response(self, user_id: str, text: str) -> OrchestratorResponse | None:
        if not self._repair_followup_requested(text):
            return None
        context = self._recent_unhealthy_runtime_context(user_id)
        if not context:
            return None
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="setup_flow",
                used_memory=True,
                reason="runtime_truth_service_unavailable",
            )
        provider_hint = self._mentioned_provider_id(text) or str(context.get("provider") or "").strip().lower() or None
        if not provider_hint:
            return None

        adapter = self._chat_runtime_adapter
        current_snapshot = truth.provider_status(provider_hint)
        current_target = truth.current_chat_target_status()
        current_provider = str(current_target.get("provider") or "").strip().lower() or None
        current_model_id = str(current_target.get("model") or "").strip() or None
        current_model_health_status = str(current_target.get("health_status") or "").strip().lower() or "unknown"
        model_id = str(
            (current_model_id if current_provider == provider_hint and current_model_id else None)
            or current_snapshot.get("model_id")
            or current_snapshot.get("current_model_id")
            or ""
        ).strip() or None
        repair_attempted = False
        if callable(getattr(adapter, "test_provider", None)):
            try:
                repair_attempted = True
                test_payload: dict[str, Any] = {}
                if model_id:
                    test_payload["model"] = model_id
                adapter.test_provider(provider_hint, test_payload)
            except Exception:
                repair_attempted = True
        snapshot = truth.provider_status(provider_hint)
        provider_label = str(snapshot.get("provider_label") or self._setup_provider_label(provider_hint))
        model_label = str(snapshot.get("model_id") or snapshot.get("current_model_id") or model_id or "none").strip() or "none"
        provider_health_status = str(snapshot.get("health_status") or "unknown").strip().lower() or "unknown"
        provider_health_reason = str(snapshot.get("health_reason") or "").strip() or None
        current_target = truth.current_chat_target_status()
        current_provider = str(current_target.get("provider") or "").strip().lower() or current_provider
        current_model_id = str(current_target.get("model") or "").strip() or current_model_id
        model_health_status = (
            str(current_target.get("health_status") or "").strip().lower()
            if current_provider == provider_hint and current_model_id
            else current_model_health_status
        ) or "unknown"
        model_label = str(current_model_id or model_label or "none").strip() or "none"
        inventory = self._canonical_model_inventory_snapshot(truth)
        inventory_rows = inventory.get("models") if isinstance(inventory.get("models"), list) else []
        local_installed_rows = [
            dict(row)
            for row in inventory_rows
            if isinstance(row, dict)
            and bool(row.get("local", False))
            and bool(row.get("available", False))
            and str(row.get("provider_id") or "").strip().lower() == provider_hint
        ]
        local_other_rows = [
            dict(row)
            for row in local_installed_rows
            if str(row.get("model_id") or "").strip() != str(model_label or "")
        ]
        suggested_row = next(
            (
                row
                for row in local_other_rows
                if bool(row.get("usable_now", False))
            ),
            local_other_rows[0] if local_other_rows else None,
        )
        suggested_model = str((suggested_row or {}).get("model_id") or "").strip() or None
        trial_state = self._current_model_trial_state(user_id)
        previous_model = str(trial_state.get("previous_model") or "").strip() or None

        if provider_health_status == "ok" and model_health_status in {"down", "degraded"}:
            concern = (
                "is not healthy right now"
                if model_health_status == "down"
                else "looks degraded right now"
            )
            message = f"{provider_label} is reachable, but the current chat model {model_label} {concern}."
            if previous_model:
                message = (
                    f"{message.rstrip('.')} I can:\n"
                    f"1) Recheck {model_label} now.\n"
                    f"2) Switch back to {previous_model}.\n"
                    "Reply 1 or 2."
                )
                return self._runtime_truth_response(
                    text=message,
                    route="setup_flow",
                    used_memory=True,
                    used_runtime_state=True,
                    used_tools=["provider_repair_check"] if repair_attempted else [],
                    payload={
                        "type": "provider_repair_options",
                        "provider": provider_hint,
                        "model_id": model_label,
                        "provider_health_status": provider_health_status,
                        "model_health_status": model_health_status,
                        "summary": message,
                        "option_1_kind": "recheck_model",
                        "option_2_kind": "switch_back",
                        "previous_model": previous_model,
                    },
                )
            if suggested_model:
                message = (
                    f"{message.rstrip('.')} I can:\n"
                    f"1) Recheck {model_label} now.\n"
                    f"2) Switch to {suggested_model}.\n"
                    "Reply 1 or 2."
                )
                return self._runtime_truth_response(
                    text=message,
                    route="setup_flow",
                    used_memory=True,
                    used_runtime_state=True,
                    used_tools=["provider_repair_check"] if repair_attempted else [],
                    payload={
                        "type": "provider_repair_options",
                        "provider": provider_hint,
                        "model_id": model_label,
                        "provider_health_status": provider_health_status,
                        "model_health_status": model_health_status,
                        "summary": message,
                        "option_1_kind": "recheck_model",
                        "option_2_kind": "switch_model",
                        "option_2_model_id": suggested_model,
                        "option_2_provider": provider_hint,
                    },
                )
            message = f"{message.rstrip('.')} I can explain the failure or help you choose another installed local model."
            return self._runtime_truth_response(
                text=message,
                route="setup_flow",
                used_memory=True,
                used_runtime_state=True,
                used_tools=["provider_repair_check"] if repair_attempted else [],
                payload={
                    "type": "provider_repair",
                    "provider": provider_hint,
                    "model_id": model_label,
                    "provider_health_status": provider_health_status,
                    "model_health_status": model_health_status,
                    "summary": message,
                    "repair_attempted": repair_attempted,
                },
            )

        if provider_health_status == "ok":
            message = f"{provider_label} is reachable again. Chat is configured for {model_label}, and it looks healthy now."
        elif provider_health_status == "degraded":
            message = f"{provider_label} is reachable, but it still needs attention. Chat is configured for {model_label}."
        else:
            message = f"{provider_label} is currently down. Chat is configured for {model_label}, but it is not responding right now."

        if provider_health_reason and provider_health_status in {"down", "degraded"}:
            message = f"{message.rstrip('.')} Reason: {provider_health_reason}."
        if provider_health_status in {"down", "degraded"} and previous_model:
            message = f"{message.rstrip('.')} I can switch back to {previous_model} if you want."
        elif provider_health_status in {"down", "degraded"}:
            message = f"{message.rstrip('.')} I can help you reconfigure {provider_label} if you want."
        elif repair_attempted:
            message = f"{message.rstrip('.')} I rechecked it just now."

        return self._runtime_truth_response(
            text=message,
            route="setup_flow",
            used_memory=True,
            used_runtime_state=True,
            used_tools=["provider_repair_check"] if repair_attempted else [],
            payload={
                "type": "provider_repair",
                "provider": provider_hint,
                "model_id": model_label,
                "health_status": provider_health_status,
                "health_reason": provider_health_reason,
                "provider_health_status": provider_health_status,
                "model_health_status": model_health_status,
                "summary": message,
                "repair_attempted": repair_attempted,
                "switch_back_available": bool(previous_model),
                "previous_model": previous_model,
            },
        )

    def _provider_status_response(self, provider_id: str) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="provider_status",
                reason="runtime_truth_service_unavailable",
            )
        snapshot = truth.provider_status(provider_id)
        target_truth = (
            truth.chat_target_truth()
            if callable(getattr(truth, "chat_target_truth", None))
            else {}
        )
        provider_key = str(snapshot.get("provider") or "").strip().lower()
        provider_label = str(snapshot.get("provider_label") or self._setup_provider_label(provider_key))
        model_id = str(snapshot.get("model_id") or "").strip() or None
        current_model_id = str(snapshot.get("current_model_id") or "").strip() or None
        configured = bool(snapshot.get("configured", False))
        active = bool(snapshot.get("active", False))
        health_status = str(snapshot.get("health_status") or "unknown").strip().lower() or "unknown"
        health_reason = str(snapshot.get("health_reason") or "").strip() or None
        secret_present = bool(snapshot.get("secret_present", False))
        effective_provider = str(snapshot.get("effective_provider") or target_truth.get("effective_provider") or "").strip().lower() or None
        effective_model_id = str(snapshot.get("effective_model_id") or target_truth.get("effective_model") or "").strip() or None
        message = self._provider_status_message(
            provider_key=provider_key,
            provider_label=provider_label,
            provider_known=bool(snapshot.get("known", False)),
            provider_enabled=bool(snapshot.get("enabled", False)),
            model_id=model_id,
            current_model_id=current_model_id,
            configured=configured,
            active=active,
            health_status=health_status,
            health_reason=health_reason,
            secret_present=secret_present,
            effective_provider=effective_provider,
            effective_model_id=effective_model_id,
        )

        return self._runtime_truth_response(
            text=message,
            route="provider_status",
            payload={
                "type": "provider_status",
                "provider": provider_key,
                "configured": configured,
                "active": active,
                "model_id": model_id,
                "current_model_id": current_model_id,
                "health_status": health_status,
                "health_reason": health_reason,
                "title": f"{provider_label} status",
                "summary": message,
            },
        )

    def _providers_status_response(self) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="provider_status",
                reason="runtime_truth_service_unavailable",
            )
        status_payload = truth.providers_status()
        rows = status_payload.get("providers") if isinstance(status_payload.get("providers"), list) else []
        configured_rows = [row for row in rows if isinstance(row, dict) and bool(row.get("configured", False))]
        active_row = next((row for row in rows if isinstance(row, dict) and bool(row.get("active", False))), None)
        target_truth = (
            truth.chat_target_truth()
            if callable(getattr(truth, "chat_target_truth", None))
            else {}
        )
        active_model_id = str(target_truth.get("effective_model") or (active_row or {}).get("model_id") or "").strip()
        active_provider_label = str(
            self._setup_provider_label(
                str(target_truth.get("effective_provider") or (active_row or {}).get("provider") or "")
            )
        )
        active_health_status = str((active_row or {}).get("health_status") or "unknown").strip().lower() or "unknown"
        if active_model_id and (
            str(target_truth.get("effective_provider") or "").strip().lower()
            != str(target_truth.get("configured_provider") or "").strip().lower()
        ):
            message = (
                f"Chat is configured to use {str(target_truth.get('configured_model') or 'the configured model')}, "
                f"but the best healthy target would be {active_model_id} on {active_provider_label}."
            )
        elif active_row and active_model_id and active_health_status == "ok":
            message = f"Chat is currently using {active_model_id}."
        elif active_row and active_model_id:
            message = f"Chat is configured to use {active_model_id}, but {active_provider_label} is not healthy right now."
        else:
            message = "I do not have a chat provider ready yet."
        if configured_rows:
            detail_rows = [
                row
                for row in configured_rows
                if not active_row
                or str(row.get("provider") or "").strip().lower()
                != str(active_row.get("provider") or "").strip().lower()
            ]
            details = ", ".join(
                f"{str(row.get('provider_label') or self._setup_provider_label(str(row.get('provider') or '')))}"
                + (
                    f" ({str(row.get('model_id') or '').strip()})"
                    if str(row.get("model_id") or "").strip()
                    else ""
                )
                for row in detail_rows
            )
            if active_row and str(active_row.get("model_id") or "").strip() and details:
                message = f"{message.rstrip('.')} I also have {details} ready."
            elif details:
                message = f"I have {details} ready for chat."
        return self._runtime_truth_response(
            text=message,
            route="provider_status",
            payload={
                "type": "providers_status",
                "title": "Provider status",
                "summary": message,
                "providers": rows,
            },
        )

    def _runtime_status_response(self, kind: str) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="runtime_status",
                reason="runtime_truth_service_unavailable",
            )
        status_payload = truth.runtime_status(kind)
        message = str(status_payload.get("summary") or "").strip() or "I can't read a clean runtime status from the current state yet."
        return self._runtime_truth_response(
            text=message,
            route="runtime_status",
            payload={
                "type": "runtime_status",
                **dict(status_payload),
                "title": "Runtime status",
                "summary": message,
            },
        )

    @staticmethod
    def _governance_component_label(identifier: str | None) -> str:
        normalized = str(identifier or "").strip()
        if not normalized:
            return "unknown"
        if normalized == "runtime_scheduler":
            return "runtime scheduler"
        if normalized == "telegram":
            return "Telegram"
        return normalized.replace("_", " ")

    def _governance_adapters_response(self) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="governance_status",
                reason="runtime_truth_service_unavailable",
            )
        payload = truth.list_managed_adapters()
        rows = payload.get("managed_adapters") if isinstance(payload.get("managed_adapters"), list) else []
        active_rows = payload.get("active_adapters") if isinstance(payload.get("active_adapters"), list) else []
        if not rows:
            message = "I do not have any managed adapters registered right now."
        else:
            details = ", ".join(
                f"{self._governance_component_label(str(row.get('adapter_id') or ''))}"
                + (
                    " (approved and enabled)"
                    if bool(row.get("approved", False)) and bool(row.get("enabled", False))
                    else " (not fully approved)"
                )
                for row in rows
                if isinstance(row, dict)
            )
            if active_rows:
                message = f"Managed adapters: {details}."
            else:
                message = f"I have managed adapters registered, but none are active right now: {details}."
        return self._runtime_truth_response(
            text=message,
            route="governance_status",
            payload={
                "type": "governance_managed_adapters",
                "title": "Managed adapters",
                "summary": message,
                **dict(payload),
            },
        )

    def _governance_background_tasks_response(self) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="governance_status",
                reason="runtime_truth_service_unavailable",
            )
        payload = truth.list_background_tasks()
        rows = payload.get("background_tasks") if isinstance(payload.get("background_tasks"), list) else []
        active_rows = payload.get("active_tasks") if isinstance(payload.get("active_tasks"), list) else []
        if active_rows:
            detail = ", ".join(
                self._governance_component_label(str(row.get("task_id") or ""))
                for row in active_rows
                if isinstance(row, dict)
            )
            message = f"Active background tasks: {detail}."
        elif rows:
            detail = ", ".join(
                self._governance_component_label(str(row.get("task_id") or ""))
                for row in rows
                if isinstance(row, dict)
            )
            message = f"I have background tasks registered, but none are active right now: {detail}."
        else:
            message = "I do not have any governed background tasks registered right now."
        return self._runtime_truth_response(
            text=message,
            route="governance_status",
            payload={
                "type": "governance_background_tasks",
                "title": "Background tasks",
                "summary": message,
                **dict(payload),
            },
        )

    def _governance_blocks_response(self) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="governance_status",
                reason="runtime_truth_service_unavailable",
            )
        payload = truth.list_governance_blocks()
        rows = payload.get("blocked_skills") if isinstance(payload.get("blocked_skills"), list) else []
        if not rows:
            message = "No skills are currently blocked by execution governance."
        else:
            detail = ", ".join(
                f"{str(row.get('skill_id') or '').strip()} ({str(row.get('reason') or 'blocked').strip().replace('_', ' ')})"
                for row in rows
                if isinstance(row, dict) and str(row.get("skill_id") or "").strip()
            )
            message = f"Execution governance blocked these skills: {detail}."
        return self._runtime_truth_response(
            text=message,
            route="governance_status",
            payload={
                "type": "governance_blocks",
                "title": "Blocked skills",
                "summary": message,
                **dict(payload),
            },
        )

    def _governance_pending_response(self) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="governance_status",
                reason="runtime_truth_service_unavailable",
            )
        payload = truth.list_pending_governance_requests()
        pending_skills = payload.get("pending_skills") if isinstance(payload.get("pending_skills"), list) else []
        pending_adapters = payload.get("pending_adapters") if isinstance(payload.get("pending_adapters"), list) else []
        pending_tasks = payload.get("pending_background_tasks") if isinstance(payload.get("pending_background_tasks"), list) else []
        parts: list[str] = []
        if pending_skills:
            parts.append(
                "skills: "
                + ", ".join(
                    str(row.get("skill_id") or "").strip()
                    for row in pending_skills
                    if isinstance(row, dict) and str(row.get("skill_id") or "").strip()
                )
            )
        if pending_adapters:
            parts.append(
                "adapters: "
                + ", ".join(
                    self._governance_component_label(str(row.get("adapter_id") or ""))
                    for row in pending_adapters
                    if isinstance(row, dict)
                )
            )
        if pending_tasks:
            parts.append(
                "background tasks: "
                + ", ".join(
                    self._governance_component_label(str(row.get("task_id") or ""))
                    for row in pending_tasks
                    if isinstance(row, dict)
                )
            )
        if parts:
            message = "Waiting for governance approval: " + "; ".join(parts) + "."
        else:
            message = "Nothing is waiting for governance approval right now."
        return self._runtime_truth_response(
            text=message,
            route="governance_status",
            payload={
                "type": "governance_pending",
                "title": "Pending governance approvals",
                "summary": message,
                **dict(payload),
            },
        )

    def _governance_overview_response(self) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="governance_status",
                reason="runtime_truth_service_unavailable",
            )
        adapters_payload = truth.list_managed_adapters()
        tasks_payload = truth.list_background_tasks()
        pending_payload = truth.list_pending_governance_requests()
        active_adapters = adapters_payload.get("active_adapters") if isinstance(adapters_payload.get("active_adapters"), list) else []
        active_tasks = tasks_payload.get("active_tasks") if isinstance(tasks_payload.get("active_tasks"), list) else []
        allowed_parts: list[str] = []
        if active_adapters:
            allowed_parts.append(
                "managed adapters: "
                + ", ".join(
                    self._governance_component_label(str(row.get("adapter_id") or ""))
                    for row in active_adapters
                    if isinstance(row, dict)
                )
            )
        if active_tasks:
            allowed_parts.append(
                "background tasks: "
                + ", ".join(
                    self._governance_component_label(str(row.get("task_id") or ""))
                    for row in active_tasks
                    if isinstance(row, dict)
                )
            )
        pending_count = 0
        for key in ("pending_skills", "pending_adapters", "pending_background_tasks"):
            rows = pending_payload.get(key) if isinstance(pending_payload.get(key), list) else []
            pending_count += len(rows)
        if allowed_parts:
            message = "Allowed persistent components right now: " + "; ".join(allowed_parts) + "."
        else:
            message = "No persistent background components are currently allowed beyond the main runtime."
        if pending_count:
            message = f"{message.rstrip('.')} {pending_count} governance request(s) still need approval."
        return self._runtime_truth_response(
            text=message,
            route="governance_status",
            payload={
                "type": "governance_overview",
                "title": "Execution governance",
                "summary": message,
                "managed_adapters": adapters_payload.get("managed_adapters") if isinstance(adapters_payload, dict) else [],
                "background_tasks": tasks_payload.get("background_tasks") if isinstance(tasks_payload, dict) else [],
                "pending": dict(pending_payload),
            },
        )

    def _governance_skill_status_response(self, skill_id: str | None) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="governance_status",
                reason="runtime_truth_service_unavailable",
            )
        payload = truth.get_skill_governance_status(skill_id)
        if bool(payload.get("needs_skill_id", False)):
            message = "Tell me the skill name you want me to inspect."
            return self._runtime_truth_response(
                text=message,
                route="governance_status",
                next_question=message,
                payload={
                    "type": "governance_skill_status",
                    "found": False,
                    "needs_skill_id": True,
                    "summary": message,
                },
            )
        skill = payload.get("skill") if isinstance(payload.get("skill"), dict) else {}
        if not skill:
            missing_id = str(payload.get("skill_id") or skill_id or "").strip() or "that skill"
            message = f"I couldn't find governance information for {missing_id}."
            return self._runtime_truth_response(
                text=message,
                route="governance_status",
                payload={
                    "type": "governance_skill_status",
                    "found": False,
                    "skill_id": missing_id,
                    "summary": message,
                },
            )
        execution_mode = str(skill.get("requested_execution_mode") or "in_process").strip().lower() or "in_process"
        if bool(skill.get("allowed", False)):
            message = f"{str(skill.get('skill_id') or 'That skill').strip()} uses {execution_mode} execution and is currently allowed."
        elif bool(skill.get("requires_user_approval", False)):
            message = f"{str(skill.get('skill_id') or 'That skill').strip()} requests {execution_mode} execution and is waiting for approval."
        else:
            message = f"{str(skill.get('skill_id') or 'That skill').strip()} is blocked by execution governance."
        return self._runtime_truth_response(
            text=message,
            route="governance_status",
            payload={
                "type": "governance_skill_status",
                "title": "Skill governance",
                "summary": message,
                **dict(payload),
            },
        )

    def _governance_execution_mode_response(self, target_id: str | None) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="governance_status",
                reason="runtime_truth_service_unavailable",
            )
        normalized_target = str(target_id or "").strip() or None
        if not normalized_target:
            message = "Tell me which skill or managed component you want me to inspect."
            return self._runtime_truth_response(
                text=message,
                route="governance_status",
                next_question=message,
                payload={
                    "type": "governance_execution_mode",
                    "found": False,
                    "target_id": None,
                    "summary": message,
                },
            )

        adapter_payload = truth.get_managed_adapter_status(normalized_target)
        adapter = adapter_payload.get("adapter") if isinstance(adapter_payload.get("adapter"), dict) else {}
        if adapter:
            adapter_label = self._governance_component_label(str(adapter.get("adapter_id") or normalized_target))
            message = f"{adapter_label} uses managed_adapter mode."
            return self._runtime_truth_response(
                text=message,
                route="governance_status",
                payload={
                    "type": "governance_execution_mode",
                    "title": "Execution mode",
                    "summary": message,
                    "found": True,
                    "target_id": normalized_target,
                    "component_kind": "managed_adapter",
                    "execution_mode": "managed_adapter",
                    "adapter": dict(adapter),
                },
            )

        task_payload = truth.get_background_task_status(normalized_target)
        task = task_payload.get("task") if isinstance(task_payload.get("task"), dict) else {}
        if task:
            task_label = self._governance_component_label(str(task.get("task_id") or normalized_target))
            message = f"{task_label} uses managed_background_task mode."
            return self._runtime_truth_response(
                text=message,
                route="governance_status",
                payload={
                    "type": "governance_execution_mode",
                    "title": "Execution mode",
                    "summary": message,
                    "found": True,
                    "target_id": normalized_target,
                    "component_kind": "background_task",
                    "execution_mode": "managed_background_task",
                    "background_task": dict(task),
                },
            )

        skill_payload = truth.get_skill_governance_status(normalized_target)
        skill = skill_payload.get("skill") if isinstance(skill_payload.get("skill"), dict) else {}
        if skill:
            execution_mode = (
                str(skill.get("requested_execution_mode") or "in_process").strip().lower() or "in_process"
            )
            skill_label = self._governance_component_label(str(skill.get("skill_id") or normalized_target))
            message = f"{skill_label} uses {execution_mode} mode."
            return self._runtime_truth_response(
                text=message,
                route="governance_status",
                payload={
                    "type": "governance_execution_mode",
                    "title": "Execution mode",
                    "summary": message,
                    "found": True,
                    "target_id": normalized_target,
                    "component_kind": "skill",
                    "execution_mode": execution_mode,
                    "skill": dict(skill),
                },
            )

        message = f"I couldn't find governance information for {normalized_target}."
        return self._runtime_truth_response(
            text=message,
            route="governance_status",
            payload={
                "type": "governance_execution_mode",
                "title": "Execution mode",
                "summary": message,
                "found": False,
                "target_id": normalized_target,
            },
        )

    def _governance_adapter_detail_response(self, adapter_id: str | None) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="governance_status",
                reason="runtime_truth_service_unavailable",
            )
        payload = truth.get_managed_adapter_status(adapter_id)
        adapter = payload.get("adapter") if isinstance(payload.get("adapter"), dict) else {}
        if not adapter:
            message = "Tell me which managed adapter you want me to explain."
            return self._runtime_truth_response(
                text=message,
                route="governance_status",
                next_question=message,
                payload={
                    "type": "governance_adapter_detail",
                    "found": False,
                    "adapter_id": str(adapter_id or "").strip() or None,
                    "summary": message,
                },
            )
        adapter_label = self._governance_component_label(str(adapter.get("adapter_id") or ""))
        reason = str(adapter.get("reason") or "").strip() or "It is an explicit managed adapter owned by the runtime."
        requested_by = str(adapter.get("requested_by") or "").strip() or None
        owner = str(adapter.get("owner") or "").strip() or None
        detail_parts = [reason]
        if requested_by:
            detail_parts.append(f"Requested by {requested_by}.")
        if owner:
            detail_parts.append(f"Owned by {owner}.")
        message = f"{adapter_label} exists as a managed adapter. " + " ".join(detail_parts)
        return self._runtime_truth_response(
            text=message,
            route="governance_status",
            payload={
                "type": "governance_adapter_detail",
                "title": f"{adapter_label} adapter",
                "summary": message,
                **dict(payload),
            },
        )

    def _current_model_response(self) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="model_status",
                reason="runtime_truth_service_unavailable",
            )
        current = truth.current_chat_target_status()
        target_truth = (
            truth.chat_target_truth()
            if callable(getattr(truth, "chat_target_truth", None))
            else {}
        )
        configured_provider = str(current.get("provider") or target_truth.get("configured_provider") or "").strip().lower() or None
        configured_model = str(current.get("model") or target_truth.get("configured_model") or "").strip() or None
        effective_provider = str(target_truth.get("effective_provider") or configured_provider or "").strip().lower() or None
        effective_model = str(target_truth.get("effective_model") or configured_model or "").strip() or None
        if not configured_model and not effective_model:
            return self._runtime_state_unavailable_response(
                route="model_status",
                reason="current_model_missing",
            )
        provider_label = self._setup_provider_label(configured_provider)
        effective_provider_label = self._setup_provider_label(effective_provider)
        provider_health_status = str(current.get("provider_health_status") or "").strip().lower() or "unknown"
        model_health_status = str(current.get("health_status") or "").strip().lower() or "unknown"
        ready = bool(current.get("ready", False))
        if ready:
            message = f"Chat is currently using {configured_model} on {provider_label}."
        elif effective_model and effective_provider and effective_model != configured_model:
            message = (
                f"Chat is configured to use {configured_model} on {provider_label}, but it is not healthy right now. "
                f"The best healthy target would be {effective_model} on {effective_provider_label}."
            )
        elif provider_health_status == "down":
            message = f"Chat is configured to use {configured_model} on {provider_label}, but {provider_label} is not responding right now."
        elif provider_health_status == "degraded":
            message = f"Chat is configured to use {configured_model} on {provider_label}, but {provider_label} needs attention right now."
        elif model_health_status == "down":
            message = f"Chat is configured to use {configured_model} on {provider_label}, but that model is not healthy right now."
        else:
            message = f"Chat is configured to use {configured_model or effective_model} on {provider_label}, but it is not ready right now."
        return self._runtime_truth_response(
            text=message,
            route="model_status",
            payload={
                "type": "model_status",
                "provider": effective_provider or configured_provider,
                "model_id": effective_model or configured_model,
                "configured_provider": configured_provider,
                "configured_model": configured_model,
                "title": "Current model",
                "ready": ready,
                "health_status": model_health_status,
                "provider_health_status": provider_health_status,
                "summary": message,
            },
        )

    @staticmethod
    def _inventory_preview(rows: list[dict[str, Any]], *, limit: int = 6) -> str:
        model_ids = [
            str(row.get("model_id") or "").strip()
            for row in rows
            if isinstance(row, dict) and str(row.get("model_id") or "").strip()
        ]
        if not model_ids:
            return ""
        preview = ", ".join(model_ids[:limit])
        extra = max(0, len(model_ids) - min(len(model_ids), limit))
        if extra > 0:
            preview = f"{preview}, and {extra} more"
        return preview

    @staticmethod
    def _inventory_reason_preview(rows: list[dict[str, Any]], *, limit: int = 3) -> str:
        items: list[str] = []
        for row in rows[:limit]:
            model_id = str(row.get("model_id") or "").strip()
            reason = str(row.get("availability_reason") or "").strip()
            if not model_id:
                continue
            items.append(f"{model_id} ({reason})" if reason else model_id)
        return ", ".join(items)

    @staticmethod
    def _lifecycle_row_label(row: dict[str, Any]) -> str:
        for key in ("model_id", "repo_id", "artifact_id", "target_key"):
            value = str(row.get(key) or "").strip()
            if value:
                return value
        return "unknown target"

    @staticmethod
    def _lifecycle_row_reason(row: dict[str, Any]) -> str | None:
        message = str(row.get("message") or "").strip()
        if message:
            return message
        acquisition_reason = str(row.get("acquisition_reason") or row.get("availability_reason") or "").strip()
        if acquisition_reason:
            return acquisition_reason
        error_kind = str(row.get("error_kind") or "").strip()
        if error_kind:
            return error_kind.replace("_", " ")
        return None

    @staticmethod
    def _lifecycle_rows_preview(
        rows: list[dict[str, Any]],
        *,
        limit: int = 4,
        include_reason: bool = False,
    ) -> str:
        items: list[str] = []
        for row in rows[:limit]:
            label = Orchestrator._lifecycle_row_label(row)
            if not label:
                continue
            if include_reason:
                reason = Orchestrator._lifecycle_row_reason(row)
                items.append(f"{label} ({reason})" if reason else label)
            else:
                items.append(label)
        preview = ", ".join(items)
        extra = max(0, len(rows) - min(len(rows), limit))
        if preview and extra > 0:
            preview = f"{preview}, and {extra} more"
        return preview

    @staticmethod
    def _lifecycle_row_search_terms(row: dict[str, Any]) -> list[str]:
        raw_terms = [
            str(row.get("model_id") or "").strip(),
            (
                str(str(row.get("model_id") or "").strip().split(":", 1)[1]).strip()
                if ":" in str(row.get("model_id") or "").strip()
                else ""
            ),
            str(row.get("repo_id") or "").strip(),
            str(row.get("artifact_id") or "").strip(),
            str(row.get("target_key") or "").strip(),
        ]
        terms: list[str] = []
        seen: set[str] = set()
        for item in raw_terms:
            normalized = normalize_setup_text(item)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            terms.append(normalized)
        return terms

    def _resolve_lifecycle_target_row(self, text: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        resolution = self._resolve_runtime_model_target(text)
        status = str(resolution.get("status") or "").strip().lower()
        if status == "unique":
            model_id = str(resolution.get("model_id") or "").strip()
            match = next(
                (
                    dict(row)
                    for row in rows
                    if str(row.get("model_id") or "").strip() == model_id
                ),
                None,
            )
            if isinstance(match, dict):
                return {
                    "status": "unique",
                    "requested": str(resolution.get("requested") or model_id).strip() or model_id,
                    "row": match,
                    "matches": [match],
                }
        normalized = normalize_setup_text(text)
        matches: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            if not any(term and term in normalized for term in self._lifecycle_row_search_terms(row)):
                continue
            key = str(row.get("target_key") or row.get("model_id") or row.get("artifact_id") or "").strip()
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            matches.append(dict(row))
        if len(matches) == 1:
            return {
                "status": "unique",
                "requested": self._lifecycle_row_label(matches[0]),
                "row": matches[0],
                "matches": matches,
            }
        if len(matches) > 1:
            return {
                "status": "ambiguous",
                "requested": None,
                "row": None,
                "matches": matches,
            }
        return {
            "status": "none",
            "requested": str(resolution.get("requested") or "").strip() or None,
            "row": None,
            "matches": [],
        }

    def _model_lifecycle_response(self, text: str) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="model_status",
                reason="runtime_truth_service_unavailable",
            )
        payload_fn = getattr(truth, "model_lifecycle_status", None)
        if not callable(payload_fn):
            return self._runtime_state_unavailable_response(
                route="model_status",
                reason="model_lifecycle_status_unavailable",
            )
        payload = payload_fn()
        rows = [
            dict(row)
            for row in (payload.get("models") if isinstance(payload.get("models"), list) else [])
            if isinstance(row, dict)
        ]
        downloading_rows = [
            dict(row)
            for row in (payload.get("downloading_targets") if isinstance(payload.get("downloading_targets"), list) else [])
            if isinstance(row, dict)
        ]
        queued_rows = [
            dict(row)
            for row in (payload.get("queued_targets") if isinstance(payload.get("queued_targets"), list) else [])
            if isinstance(row, dict)
        ]
        failed_rows = [
            dict(row)
            for row in (payload.get("failed_targets") if isinstance(payload.get("failed_targets"), list) else [])
            if isinstance(row, dict)
        ]
        normalized = normalize_setup_text(text).replace("/", " ")
        resolution = self._resolve_lifecycle_target_row(text, rows)
        if str(resolution.get("status") or "").strip().lower() == "ambiguous":
            matches = [
                self._lifecycle_row_label(row)
                for row in (resolution.get("matches") if isinstance(resolution.get("matches"), list) else [])
                if isinstance(row, dict)
            ]
            message = f"I found more than one tracked install target that matches that request: {', '.join(matches[:4])}."
            return self._runtime_truth_response(
                text=message,
                route="model_status",
                next_question="Tell me the full model id you want me to check.",
                payload={
                    "type": "model_lifecycle_status",
                    "query_kind": "ambiguous_target",
                    "title": "Model lifecycle status",
                    "summary": message,
                    "matches": matches,
                    "counts": dict(payload.get("counts") or {}),
                },
            )
        if str(resolution.get("status") or "").strip().lower() == "unique":
            row = resolution.get("row") if isinstance(resolution.get("row"), dict) else {}
            label = self._lifecycle_row_label(row)
            lifecycle_state = str(row.get("lifecycle_state") or "not_installed").strip().lower() or "not_installed"
            if lifecycle_state == "ready":
                message = f"{label} is ready to use now."
            elif lifecycle_state == "installed_not_ready":
                message = f"{label} is already installed, but it is not ready yet."
            elif lifecycle_state == "installed":
                message = f"{label} is installed."
            elif lifecycle_state == "downloading":
                message = f"{label} is downloading now."
            elif lifecycle_state == "queued":
                message = f"{label} is queued. I am waiting for approval before I start."
            elif lifecycle_state == "failed":
                message = f"The last attempt to get {label} failed."
            else:
                acquisition_state = str(row.get("acquisition_state") or "").strip().lower()
                if acquisition_state == "acquirable":
                    message = f"{label} is not installed yet, but I can get it for you if you approve."
                elif acquisition_state == "blocked_by_policy":
                    message = f"{label} is not installed yet, and this mode does not let me download or install it."
                elif acquisition_state == "not_acquirable":
                    message = f"{label} is not installed, and it is not on the supported acquire/install path."
                else:
                    message = f"{label} is not installed."
            reason = self._lifecycle_row_reason(row)
            if reason and reason.lower() not in message.lower():
                message = f"{message.rstrip('.')} Reason: {reason}."
            return self._runtime_truth_response(
                text=message,
                route="model_status",
                payload={
                    "type": "model_lifecycle_status",
                    "query_kind": "target",
                    "title": "Model lifecycle status",
                    "summary": message,
                    "model_id": row.get("model_id"),
                    "target_key": row.get("target_key"),
                    "lifecycle_state": lifecycle_state,
                    "row": dict(row),
                    "counts": dict(payload.get("counts") or {}),
                },
            )
        requested_target = str(resolution.get("requested") or "").strip() or None
        if not requested_target:
            explicit_target = re.search(
                r"\b(?:[a-z0-9._-]+:)?[a-z0-9][a-z0-9./_-]*:[a-z0-9][a-z0-9./_-]*\b",
                normalized,
            )
            if explicit_target is not None:
                requested_target = str(explicit_target.group(0) or "").strip() or None
        if requested_target and any(token in normalized for token in ("install", "download", "import", "pull")):
            policy = (
                truth.model_controller_policy_status()
                if callable(getattr(truth, "model_controller_policy_status", None))
                else {}
            )
            if not bool(policy.get("allow_install_pull", True)):
                message = (
                    f"{requested_target} is not installed. This mode does not let me download or install it here."
                )
            else:
                message = (
                    f"{requested_target} is not installed. I can only get it if it is on the supported acquire/install path."
                )
            return self._runtime_truth_response(
                text=message,
                route="model_status",
                payload={
                    "type": "model_lifecycle_status",
                    "query_kind": "target_missing_acquisition",
                    "title": "Model lifecycle status",
                    "summary": message,
                    "model_id": requested_target,
                    "target_key": requested_target,
                    "lifecycle_state": "not_installed",
                    "row": None,
                    "counts": dict(payload.get("counts") or {}),
                },
            )
        if "fail" in normalized:
            if failed_rows:
                message = f"Failed model installs or downloads: {self._lifecycle_rows_preview(failed_rows, include_reason=True)}."
            else:
                message = "I do not see any failed model installs or downloads in the canonical manager state."
            return self._runtime_truth_response(
                text=message,
                route="model_status",
                payload={
                    "type": "model_lifecycle_status",
                    "query_kind": "failed",
                    "title": "Failed installs",
                    "summary": message,
                    "matches": failed_rows,
                    "counts": dict(payload.get("counts") or {}),
                },
            )
        if any(token in normalized for token in ("downloading", "installing", "in progress", "queued", "pending")):
            if downloading_rows:
                message = f"Models downloading right now: {self._lifecycle_rows_preview(downloading_rows)}."
                if queued_rows:
                    message = f"{message} Queued and waiting approval: {self._lifecycle_rows_preview(queued_rows)}."
            elif queued_rows:
                message = f"I do not see a download in progress right now. Queued and waiting approval: {self._lifecycle_rows_preview(queued_rows)}."
            else:
                message = "I do not see any model downloads in progress right now."
            return self._runtime_truth_response(
                text=message,
                route="model_status",
                payload={
                    "type": "model_lifecycle_status",
                    "query_kind": "active_operations",
                    "title": "Active installs",
                    "summary": message,
                    "matches": [*downloading_rows, *queued_rows],
                    "counts": dict(payload.get("counts") or {}),
                },
            )
        counts = dict(payload.get("counts") or {})
        message = (
            "Canonical model lifecycle status: "
            f"{int(counts.get('ready', 0) or 0)} ready, "
            f"{int(counts.get('downloading', 0) or 0)} downloading, "
            f"{int(counts.get('queued', 0) or 0)} queued, and "
            f"{int(counts.get('failed', 0) or 0)} failed."
        )
        return self._runtime_truth_response(
            text=message,
            route="model_status",
            payload={
                "type": "model_lifecycle_status",
                "query_kind": "summary",
                "title": "Model lifecycle status",
                "summary": message,
                "counts": counts,
                "active_operations": [
                    dict(row)
                    for row in (payload.get("active_operations") if isinstance(payload.get("active_operations"), list) else [])
                    if isinstance(row, dict)
                ],
                "failed_targets": failed_rows,
            },
        )

    def _canonical_model_inventory_snapshot(self, truth: Any) -> dict[str, Any]:
        inventory_fn = getattr(truth, "model_inventory_status", None)
        readiness_fn = getattr(truth, "model_readiness_status", None)
        inventory = inventory_fn() if callable(inventory_fn) else {}
        readiness = readiness_fn() if callable(readiness_fn) else {}
        inventory = dict(inventory) if isinstance(inventory, dict) else {}
        readiness = dict(readiness) if isinstance(readiness, dict) else {}

        if not inventory and not readiness:
            return {
                "active_provider": None,
                "active_model": None,
                "configured_provider": None,
                "configured_model": None,
                "models": [],
                "ready_now_models": [],
                "usable_models": [],
                "other_ready_now_models": [],
                "other_usable_models": [],
                "not_ready_models": [],
                "local_installed_models": [],
                "remote_registered_models": [],
                "inventory": {},
                "readiness": {},
                "source": "canonical_inventory+readiness",
            }

        inventory_rows = [
            dict(row)
            for row in (inventory.get("models") if isinstance(inventory.get("models"), list) else [])
            if isinstance(row, dict)
        ]
        readiness_rows = [
            dict(row)
            for row in (readiness.get("models") if isinstance(readiness.get("models"), list) else [])
            if isinstance(row, dict)
        ]
        readiness_by_model = {
            str(row.get("model_id") or "").strip(): dict(row)
            for row in readiness_rows
            if str(row.get("model_id") or "").strip()
        }

        merged_rows: list[dict[str, Any]] = []
        seen_model_ids: set[str] = set()
        for inventory_row in inventory_rows:
            model_id = str(inventory_row.get("model_id") or "").strip()
            seen_model_ids.add(model_id)
            merged_rows.append(
                {
                    **dict(inventory_row),
                    **dict(readiness_by_model.get(model_id) or {}),
                }
            )
        for readiness_row in readiness_rows:
            model_id = str(readiness_row.get("model_id") or "").strip()
            if model_id in seen_model_ids:
                continue
            merged_rows.append(dict(readiness_row))

        ready_rows = [
            dict(row)
            for row in (
                readiness.get("ready_now_models")
                if isinstance(readiness.get("ready_now_models"), list)
                else readiness.get("usable_models")
                if isinstance(readiness.get("usable_models"), list)
                else []
            )
            if isinstance(row, dict)
        ]
        not_ready_rows = [
            dict(row)
            for row in (readiness.get("not_ready_models") if isinstance(readiness.get("not_ready_models"), list) else [])
            if isinstance(row, dict)
        ]
        local_installed_rows = [
            dict(row)
            for row in (
                inventory.get("local_installed_models")
                if isinstance(inventory.get("local_installed_models"), list)
                else []
            )
            if isinstance(row, dict)
        ]
        if not local_installed_rows:
            local_installed_rows = [
                dict(row)
                for row in merged_rows
                if bool(row.get("local", False))
                and bool(row.get("installed_local", row.get("available", False)))
            ]
        remote_registered_rows = [
            dict(row)
            for row in (
                inventory.get("remote_registered_models")
                if isinstance(inventory.get("remote_registered_models"), list)
                else []
            )
            if isinstance(row, dict)
        ]
        if not remote_registered_rows:
            remote_registered_rows = [
                dict(row)
                for row in merged_rows
                if not bool(row.get("local", False))
            ]
        return {
            "active_provider": str(readiness.get("active_provider") or inventory.get("active_provider") or "").strip().lower() or None,
            "active_model": str(readiness.get("active_model") or inventory.get("active_model") or "").strip() or None,
            "configured_provider": str(readiness.get("configured_provider") or inventory.get("configured_provider") or "").strip().lower() or None,
            "configured_model": str(readiness.get("configured_model") or inventory.get("configured_model") or "").strip() or None,
            "models": merged_rows,
            "ready_now_models": ready_rows,
            "usable_models": [dict(row) for row in ready_rows],
            "other_ready_now_models": [
                dict(row) for row in ready_rows if not bool(row.get("active", False))
            ],
            "other_usable_models": [
                dict(row) for row in ready_rows if not bool(row.get("active", False))
            ],
            "not_ready_models": not_ready_rows,
            "local_installed_models": local_installed_rows,
            "remote_registered_models": remote_registered_rows,
            "inventory": dict(inventory),
            "readiness": dict(readiness),
            "source": "canonical_inventory+readiness",
        }

    def _model_inventory_response(
        self,
        *,
        local_only: bool,
        remote_only: bool = False,
        provider_id: str | None = None,
    ) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="model_status",
                reason="runtime_truth_service_unavailable",
            )
        payload = self._canonical_model_inventory_snapshot(truth)
        provider_key = str(provider_id or "").strip().lower() or None
        active_model = str(payload.get("active_model") or "").strip() or None
        active_provider = str(payload.get("active_provider") or "").strip().lower() or None
        active_label = active_model or "the current chat target"
        if active_provider and active_model:
            active_label = f"{active_model} on {self._setup_provider_label(active_provider)}"
        rows = [
            dict(row)
            for row in (payload.get("models") if isinstance(payload.get("models"), list) else [])
            if isinstance(row, dict)
        ]
        if provider_key:
            rows = [
                row
                for row in rows
                if str(row.get("provider_id") or "").strip().lower() == provider_key
            ]
        if local_only:
            rows = [row for row in rows if bool(row.get("local", False))]
        if remote_only:
            rows = [row for row in rows if not bool(row.get("local", False))]
        usable_rows = [row for row in rows if bool(row.get("usable_now", False))]
        other_usable_rows = [row for row in usable_rows if not bool(row.get("active", False))]
        not_ready_rows = [row for row in rows if not bool(row.get("usable_now", False))]
        installed_preview = self._inventory_preview(rows)
        usable_preview = self._inventory_preview(other_usable_rows if active_model else usable_rows)
        not_ready_preview = self._inventory_reason_preview(not_ready_rows)
        response_payload = {
            "active_provider": payload.get("active_provider"),
            "active_model": payload.get("active_model"),
            "configured_provider": payload.get("configured_provider"),
            "configured_model": payload.get("configured_model"),
            "models": rows,
            "usable_models": usable_rows,
            "other_usable_models": other_usable_rows,
            "not_ready_models": not_ready_rows,
            "source": payload.get("source"),
        }

        if local_only:
            scope_label = "Ollama local" if provider_key == "ollama" else "local"
            if rows:
                parts = [f"Right now chat is using {active_label}."]
                parts.append(f"{scope_label} installed chat models: {installed_preview}.")
                if usable_preview:
                    parts.append(f"Other {scope_label.lower()} models ready to use now: {usable_preview}.")
                if not_ready_preview:
                    parts.append(f"Present but not ready: {not_ready_preview}.")
                message = " ".join(parts)
            else:
                message = (
                    "I do not currently see any installed Ollama chat models."
                    if provider_key == "ollama"
                    else "I do not currently see any installed local chat models."
                )
            payload_type = "local_model_inventory"
            title = "Local chat models"
        elif remote_only:
            if rows:
                parts: list[str] = []
                if usable_preview:
                    parts.append(f"Cloud models available to use now: {usable_preview}.")
                else:
                    parts.append("I do not currently see a cloud model that is usable right now.")
                if not_ready_preview:
                    parts.append(f"Cloud models present but not ready: {not_ready_preview}.")
                message = " ".join(parts)
            else:
                message = "I do not currently see any cloud chat models in the runtime inventory."
            payload_type = "model_availability"
            title = "Available cloud models"
        else:
            if usable_preview:
                message = f"Right now chat is using {active_label}. Other models available to use now: {usable_preview}."
            elif active_model:
                message = f"Right now chat is using {active_label}. I do not see another healthy model available to switch to right now."
            else:
                all_usable_preview = self._inventory_preview(usable_rows)
                message = (
                    f"The models I can use right now are {all_usable_preview}."
                    if all_usable_preview
                    else "I do not see a healthy usable chat model right now."
                )
            if not_ready_preview:
                message = f"{message} Also present but not ready: {not_ready_preview}."
            if usable_preview:
                message = f"{message} If you want, I can switch you to one of those usable models."
            payload_type = "model_availability"
            title = "Available chat models"
        return self._runtime_truth_response(
            text=message,
            route="model_status",
            payload={
                **response_payload,
                "type": payload_type,
                "title": title,
                "inventory_scope": "local" if local_only else "remote" if remote_only else "available",
                "inventory_provider": provider_key,
                "summary": message,
            },
        )

    def _available_models_response(self) -> OrchestratorResponse:
        return self._model_inventory_response(local_only=False)

    @staticmethod
    def _filesystem_entry_label(entry: dict[str, Any]) -> str:
        name = str(entry.get("name") or "").strip()
        entry_type = str(entry.get("type") or "").strip().lower()
        if not name:
            return "unknown"
        if entry_type == "dir":
            return f"{name}/"
        if entry_type == "symlink":
            return f"{name} [symlink]"
        return name

    @staticmethod
    def _filesystem_target_label(payload: dict[str, Any]) -> str:
        return str(payload.get("resolved_path") or payload.get("path") or "that path").strip() or "that path"

    @staticmethod
    def _filesystem_error_message(payload: dict[str, Any]) -> str:
        target = Orchestrator._filesystem_target_label(payload)
        error_kind = str(payload.get("error_kind") or "").strip().lower() or "filesystem_error"
        if error_kind == "outside_allowed_roots":
            return compose_actionable_message(
                what_happened=f"I can't access {target}",
                why="It is outside the allowed local file roots.",
                next_action="Choose a path under the allowed local roots.",
            )
        if error_kind == "sensitive_path_blocked":
            return compose_actionable_message(
                what_happened=f"I can't access {target}",
                why="That path is blocked by the local privacy policy.",
                next_action="Choose a non-sensitive path instead.",
            )
        if error_kind == "binary_file_not_supported":
            return compose_actionable_message(
                what_happened=f"I can't read {target}",
                why="It looks binary, and this skill only supports text files.",
                next_action="Point me at a text file instead.",
            )
        if error_kind == "not_found":
            return compose_actionable_message(
                what_happened=f"I can't access {target}",
                why="It does not exist.",
                next_action="Check the path and try again.",
            )
        if error_kind == "not_readable":
            return compose_actionable_message(
                what_happened=f"I can't access {target}",
                why="It is not readable.",
                next_action="Choose a readable path instead.",
            )
        if error_kind == "not_directory":
            return compose_actionable_message(
                what_happened=f"I can't list {target}",
                why="It is not a directory.",
                next_action="Choose a directory path instead.",
            )
        if error_kind == "not_file":
            return compose_actionable_message(
                what_happened=f"I can't read {target}",
                why="It is not a regular text file.",
                next_action="Choose a regular text file instead.",
            )
        return str(payload.get("message") or "I couldn't complete that filesystem request.").strip() or "I couldn't complete that filesystem request."

    def _filesystem_clarification_response(
        self,
        *,
        kind: str,
        question: str,
    ) -> OrchestratorResponse:
        return self._runtime_truth_response(
            text=question,
            route="action_tool",
            used_tools=["filesystem"],
            used_runtime_state=False,
            next_question=question,
            payload={
                "type": "action_clarification",
                "kind": kind,
                "summary": question,
                "next_question": question,
            },
        )

    def _filesystem_list_directory_response(self, path_hint: str | None) -> OrchestratorResponse:
        if not str(path_hint or "").strip():
            return self._filesystem_clarification_response(
                kind="filesystem_list_directory",
                question="Tell me the exact directory path you want me to list.",
            )
        truth = self._runtime_truth()
        if truth is None or not callable(getattr(truth, "filesystem_list_directory", None)):
            return self._runtime_state_unavailable_response(
                route="action_tool",
                reason="filesystem_list_directory_unavailable",
            )
        payload = truth.filesystem_list_directory(path_hint)
        payload = dict(payload) if isinstance(payload, dict) else {}
        if not bool(payload.get("ok", False)):
            message = self._filesystem_error_message(payload)
            return self._runtime_truth_response(
                text=message,
                route="action_tool",
                used_tools=["filesystem"],
                ok=False,
                error_kind=str(payload.get("error_kind") or "filesystem_error").strip() or "filesystem_error",
                payload={
                    **payload,
                    "title": "Directory listing",
                    "summary": message,
                },
            )
        entries = [
            dict(row)
            for row in (payload.get("entries") if isinstance(payload.get("entries"), list) else [])
            if isinstance(row, dict)
        ]
        target = self._filesystem_target_label(payload)
        if not entries:
            message = f"{target} is empty."
        else:
            preview = ", ".join(self._filesystem_entry_label(row) for row in entries[:12])
            message = f"{target} contains {len(entries)} entries: {preview}."
            if bool(payload.get("truncated", False)):
                message = f"{message} Showing the first {len(entries)} entries."
        return self._runtime_truth_response(
            text=message,
            route="action_tool",
            used_tools=["filesystem"],
            payload={
                **payload,
                "title": "Directory listing",
                "summary": message,
            },
        )

    def _filesystem_stat_path_response(self, path_hint: str | None) -> OrchestratorResponse:
        if not str(path_hint or "").strip():
            return self._filesystem_clarification_response(
                kind="filesystem_stat_path",
                question="Tell me the exact file or directory path you want me to inspect.",
            )
        truth = self._runtime_truth()
        if truth is None or not callable(getattr(truth, "filesystem_stat_path", None)):
            return self._runtime_state_unavailable_response(
                route="action_tool",
                reason="filesystem_stat_path_unavailable",
            )
        payload = truth.filesystem_stat_path(path_hint)
        payload = dict(payload) if isinstance(payload, dict) else {}
        if not bool(payload.get("ok", False)):
            message = self._filesystem_error_message(payload)
            return self._runtime_truth_response(
                text=message,
                route="action_tool",
                used_tools=["filesystem"],
                ok=False,
                error_kind=str(payload.get("error_kind") or "filesystem_error").strip() or "filesystem_error",
                payload={
                    **payload,
                    "title": "Path status",
                    "summary": message,
                },
            )
        target = self._filesystem_target_label(payload)
        modified_time = payload.get("modified_time")
        modified_text = None
        if modified_time is not None:
            try:
                modified_text = datetime.fromtimestamp(float(modified_time), tz=timezone.utc).isoformat()
            except Exception:
                modified_text = None
        size_text = f"{int(payload.get('size') or 0)} bytes"
        message = f"{target} is a {str(payload.get('type') or 'path').strip().lower()} with size {size_text}."
        if modified_text:
            message = f"{message} Modified: {modified_text}."
        if not bool(payload.get("readable", False)):
            message = f"{message} It is not readable."
        return self._runtime_truth_response(
            text=message,
            route="action_tool",
            used_tools=["filesystem"],
            payload={
                **payload,
                "title": "Path status",
                "summary": message,
            },
        )

    def _filesystem_read_text_file_response(self, path_hint: str | None) -> OrchestratorResponse:
        if not str(path_hint or "").strip():
            return self._filesystem_clarification_response(
                kind="filesystem_read_text_file",
                question="Tell me the exact text file path you want me to read.",
            )
        truth = self._runtime_truth()
        if truth is None or not callable(getattr(truth, "filesystem_read_text_file", None)):
            return self._runtime_state_unavailable_response(
                route="action_tool",
                reason="filesystem_read_text_file_unavailable",
            )
        payload = truth.filesystem_read_text_file(path_hint)
        payload = dict(payload) if isinstance(payload, dict) else {}
        if not bool(payload.get("ok", False)):
            message = self._filesystem_error_message(payload)
            return self._runtime_truth_response(
                text=message,
                route="action_tool",
                used_tools=["filesystem"],
                ok=False,
                error_kind=str(payload.get("error_kind") or "filesystem_error").strip() or "filesystem_error",
                payload={
                    **payload,
                    "title": "Text file preview",
                    "summary": message,
                },
            )
        target = self._filesystem_target_label(payload)
        bytes_read = int(payload.get("bytes_read") or 0)
        total_size = int(payload.get("total_size") or bytes_read)
        if bool(payload.get("truncated", False)):
            header = f"Text preview from {target} ({bytes_read} bytes shown of {total_size})."
        else:
            header = f"Text from {target} ({bytes_read} bytes)."
        text_body = str(payload.get("text") or "")
        message = f"{header}\n{text_body}" if text_body else header
        return self._runtime_truth_response(
            text=message,
            route="action_tool",
            used_tools=["filesystem"],
            payload={
                **payload,
                "title": "Text file preview",
                "summary": header,
            },
        )

    def _filesystem_search_filenames_response(
        self,
        *,
        root_hint: str | None,
        query: str | None,
    ) -> OrchestratorResponse:
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return self._filesystem_clarification_response(
                kind="filesystem_search_filenames",
                question="Tell me the filename you want me to search for.",
            )
        truth = self._runtime_truth()
        if truth is None or not callable(getattr(truth, "filesystem_search_filenames", None)):
            return self._runtime_state_unavailable_response(
                route="action_tool",
                reason="filesystem_search_filenames_unavailable",
            )
        payload = truth.filesystem_search_filenames(root_hint or ".", normalized_query)
        payload = dict(payload) if isinstance(payload, dict) else {}
        target = str(payload.get("resolved_root") or payload.get("root") or root_hint or ".").strip() or "."
        if not bool(payload.get("ok", False)):
            error_kind = str(payload.get("error_kind") or "filesystem_error").strip() or "filesystem_error"
            if error_kind == "no_matches":
                message = f"I didn't find any files or directories named like {normalized_query!r} under {target}."
            else:
                message = self._filesystem_error_message(payload)
            return self._runtime_truth_response(
                text=message,
                route="action_tool",
                used_tools=["filesystem"],
                ok=False,
                error_kind=error_kind,
                payload={
                    **payload,
                    "title": "Filename search",
                    "summary": message,
                },
            )
        results = [
            dict(row)
            for row in (payload.get("results") if isinstance(payload.get("results"), list) else [])
            if isinstance(row, dict)
        ]
        preview = ", ".join(str(row.get("path") or "").strip() for row in results[:8] if str(row.get("path") or "").strip())
        message = f"Filename matches for {normalized_query!r} under {target}: {preview}."
        if bool(payload.get("truncated", False)):
            message = f"{message} Showing the first {len(results)} matches."
        return self._runtime_truth_response(
            text=message,
            route="action_tool",
            used_tools=["filesystem"],
            payload={
                **payload,
                "title": "Filename search",
                "summary": message,
            },
        )

    def _filesystem_search_text_response(
        self,
        *,
        root_hint: str | None,
        query: str | None,
    ) -> OrchestratorResponse:
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return self._filesystem_clarification_response(
                kind="filesystem_search_text",
                question="Tell me the text you want me to search for.",
            )
        truth = self._runtime_truth()
        if truth is None or not callable(getattr(truth, "filesystem_search_text", None)):
            return self._runtime_state_unavailable_response(
                route="action_tool",
                reason="filesystem_search_text_unavailable",
            )
        payload = truth.filesystem_search_text(root_hint or ".", normalized_query)
        payload = dict(payload) if isinstance(payload, dict) else {}
        target = str(payload.get("resolved_root") or payload.get("root") or root_hint or ".").strip() or "."
        if not bool(payload.get("ok", False)):
            error_kind = str(payload.get("error_kind") or "filesystem_error").strip() or "filesystem_error"
            if error_kind == "no_matches":
                message = f"I didn't find any text matches for {normalized_query!r} under {target}."
            else:
                message = self._filesystem_error_message(payload)
            return self._runtime_truth_response(
                text=message,
                route="action_tool",
                used_tools=["filesystem"],
                ok=False,
                error_kind=error_kind,
                payload={
                    **payload,
                    "title": "Text search",
                    "summary": message,
                },
            )
        results = [
            dict(row)
            for row in (payload.get("results") if isinstance(payload.get("results"), list) else [])
            if isinstance(row, dict)
        ]
        preview_lines: list[str] = []
        for row in results[:4]:
            path = str(row.get("path") or "").strip()
            snippet = str(row.get("snippet") or "").strip()
            line_number = row.get("line_number")
            if not path:
                continue
            if line_number is not None:
                preview_lines.append(f"{path}:{int(line_number)} {snippet}")
            else:
                preview_lines.append(f"{path} {snippet}")
        message = f"Text matches for {normalized_query!r} under {target}:\n" + "\n".join(preview_lines)
        if bool(payload.get("truncated", False)):
            message = f"{message}\nShowing the first {len(results)} matches."
        return self._runtime_truth_response(
            text=message,
            route="action_tool",
            used_tools=["filesystem"],
            payload={
                **payload,
                "title": "Text search",
                "summary": f"Text matches for {normalized_query!r} under {target}.",
            },
        )

    @staticmethod
    def _shell_blocked_message(payload: dict[str, Any]) -> str:
        blocked_reason = str(payload.get("blocked_reason") or payload.get("error_kind") or "").strip().lower()
        if blocked_reason == "shell_interpolation_blocked":
            return compose_actionable_message(
                what_happened="I can't run shell-style chaining, pipes, or interpolation through this bounded shell skill",
                why="This shell surface only accepts validated structured operations.",
                next_action="Ask for one supported shell action at a time.",
            )
        if blocked_reason == "unsupported_command":
            return compose_actionable_message(
                what_happened="I can't run that shell command here",
                why="This skill only supports a small allowlisted set of native shell operations.",
                next_action="Ask for a supported read-only shell command or a bounded install/create action.",
            )
        if blocked_reason == "destructive_operation_blocked":
            return compose_actionable_message(
                what_happened="I can't run that shell operation here",
                why="This bounded shell skill blocks destructive and privilege-changing actions.",
                next_action="Use a non-destructive path instead.",
            )
        if blocked_reason == "operation_not_supported":
            return compose_actionable_message(
                what_happened="I can't do that file-management action here",
                why="This bounded shell skill does not support that operation.",
                next_action="Use one of the supported shell or filesystem actions instead.",
            )
        if blocked_reason == "unsupported_manager":
            return compose_actionable_message(
                what_happened="I can't use that package manager here",
                why="This bounded shell skill supports only a narrow allowlisted install path.",
                next_action="Use a supported package manager instead.",
            )
        if blocked_reason == "unsupported_scope":
            return compose_actionable_message(
                what_happened="I can't use that install scope here",
                why="This bounded shell skill only supports specific validated install scopes.",
                next_action="Use a supported install scope instead.",
            )
        if blocked_reason == "invalid_package_name":
            return compose_actionable_message(
                what_happened="I can't start that install request",
                why="The package name is not valid.",
                next_action="Retry with a simple valid package name.",
            )
        if blocked_reason == "invalid_argument":
            return compose_actionable_message(
                what_happened="I can't run that shell request",
                why="The argument is not valid for this bounded shell surface.",
                next_action="Retry with a simple validated argument.",
            )
        if blocked_reason == "outside_allowed_roots":
            return compose_actionable_message(
                what_happened="I can't use that path here",
                why="It is outside the allowed local roots.",
                next_action="Choose a path under the allowed local roots.",
            )
        if blocked_reason == "sensitive_path_blocked":
            return compose_actionable_message(
                what_happened="I can't use that path here",
                why="It is blocked by the local privacy policy.",
                next_action="Choose a non-sensitive path instead.",
            )
        if blocked_reason == "not_directory":
            return compose_actionable_message(
                what_happened="I can't use that path for this request",
                why="It is not a directory.",
                next_action="Choose a directory path instead.",
            )
        if blocked_reason == "not_found":
            return compose_actionable_message(
                what_happened="I can't use that path for this request",
                why="It does not exist.",
                next_action="Check the path and try again.",
            )
        if blocked_reason == "path_exists":
            return compose_actionable_message(
                what_happened="I can't create that directory",
                why="A non-directory path already exists there.",
                next_action="Choose a different target path.",
            )
        if blocked_reason == "parent_not_directory":
            return compose_actionable_message(
                what_happened="I can't create that directory",
                why="The parent directory does not exist or is not a directory.",
                next_action="Create or choose a valid parent directory first.",
            )
        if blocked_reason == "not_writable":
            return compose_actionable_message(
                what_happened="I can't write there",
                why="That path is not writable.",
                next_action="Choose a writable directory instead.",
            )
        return str(payload.get("message") or "I couldn't complete that shell request.").strip() or "I couldn't complete that shell request."

    @staticmethod
    def _shell_command_label(payload: dict[str, Any]) -> str:
        command_name = str(payload.get("command_name") or "").strip().lower()
        labels = {
            "python_version": "Python version",
            "pip_version": "pip version",
            "which": "Executable lookup",
            "apt_search": "APT search",
            "apt_cache_policy": "APT policy",
            "ollama_list": "Ollama models",
            "ollama_ps": "Ollama running models",
            "ollama_show": "Ollama model details",
            "pwd": "Working directory",
            "uname": "System information",
        }
        return labels.get(command_name, "Shell command")

    def _shell_command_payload_response(self, payload: dict[str, Any]) -> OrchestratorResponse:
        ok = bool(payload.get("ok", False))
        label = self._shell_command_label(payload)
        stdout = str(payload.get("stdout") or "").strip()
        stderr = str(payload.get("stderr") or "").strip()
        if not ok:
            error_kind = str(payload.get("error_kind") or payload.get("blocked_reason") or "shell_error").strip() or "shell_error"
            if error_kind == "permission_denied":
                message = compose_actionable_message(
                    what_happened=f"{label} did not run",
                    why="It needs more privileges than this bounded shell skill can use.",
                    next_action="Use a supported non-privileged action instead.",
                )
            elif error_kind == "command_not_available":
                message = compose_actionable_message(
                    what_happened=f"{label} is not available in this environment",
                    why="The required command is missing from this runtime.",
                    next_action="Install it through a supported path, or choose another supported command.",
                )
            elif error_kind == "timeout":
                message = compose_actionable_message(
                    what_happened=f"{label} timed out",
                    why="The bounded shell timeout was reached before it finished.",
                    next_action="Retry with a smaller request or check the environment first.",
                )
            else:
                message = self._shell_blocked_message(payload)
            if stderr:
                message = f"{message}\n{stderr}"
            return self._runtime_truth_response(
                text=message,
                route="action_tool",
                used_tools=["shell"],
                ok=False,
                error_kind=error_kind,
                payload={
                    **payload,
                    "title": label,
                    "summary": message.splitlines()[0],
                },
            )
        output = stdout or stderr or "Command completed."
        message = f"{label}:\n{output}"
        if bool(payload.get("truncated", False)):
            message = f"{message}\nOutput was truncated."
        return self._runtime_truth_response(
            text=message,
            route="action_tool",
            used_tools=["shell"],
            payload={
                **payload,
                "title": label,
                "summary": label,
            },
        )

    def _shell_blocked_request_response(self, *, blocked_reason: str, request_text: str | None = None) -> OrchestratorResponse:
        payload = {
            "ok": False,
            "type": "shell_blocked_request",
            "blocked_reason": str(blocked_reason or "unsupported_command").strip() or "unsupported_command",
            "request_text": str(request_text or "").strip() or None,
        }
        message = self._shell_blocked_message(payload)
        return self._runtime_truth_response(
            text=message,
            route="action_tool",
            used_tools=["shell"],
            used_runtime_state=False,
            ok=False,
            error_kind=str(payload["blocked_reason"]),
            payload={
                **payload,
                "title": "Shell request blocked",
                "summary": message,
            },
        )

    def _shell_execute_safe_command_response(
        self,
        *,
        command_name: str | None,
        subject: str | None = None,
        query: str | None = None,
        cwd: str | None = None,
    ) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None or not callable(getattr(truth, "shell_execute_safe_command", None)):
            return self._runtime_state_unavailable_response(
                route="action_tool",
                reason="shell_execute_safe_command_unavailable",
            )
        payload = truth.shell_execute_safe_command(
            command_name,
            subject=subject,
            query=query,
            cwd=cwd,
        )
        payload = dict(payload) if isinstance(payload, dict) else {}
        return self._shell_command_payload_response(payload)

    def _shell_install_package_response(
        self,
        user_id: str,
        *,
        manager: str | None,
        package: str | None,
        scope: str | None = None,
        dry_run: bool = False,
        confirmed: bool = False,
    ) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="action_tool",
                reason="shell_install_package_unavailable",
            )
        if not confirmed:
            preview_fn = getattr(truth, "shell_preview_install_package", None)
            if not callable(preview_fn):
                return self._runtime_state_unavailable_response(
                    route="action_tool",
                    reason="shell_install_package_preview_unavailable",
                )
            preview = preview_fn(
                manager=manager,
                package=package,
                scope=scope,
                dry_run=dry_run,
            )
            preview = dict(preview) if isinstance(preview, dict) else {}
            if not bool(preview.get("ok", False)):
                return self._shell_command_payload_response(preview)
            command_preview = " ".join(str(item) for item in (preview.get("argv") if isinstance(preview.get("argv"), list) else []) if str(item).strip())
            package_label = str(preview.get("package") or package or "that package").strip() or "that package"
            question = (
                f"I will install {package_label} using {command_preview}. "
                "This mutates the local system. Reply yes to proceed or no to cancel."
            )
            return self._confirmation_preview_response(
                user_id=user_id,
                route="action_tool",
                question=question,
                used_tools=["shell"],
                action={
                    "operation": "shell_install_package",
                    "params": {
                        "manager": preview.get("manager"),
                        "package": preview.get("package"),
                        "scope": preview.get("scope"),
                        "dry_run": bool(preview.get("dry_run", False)),
                    },
                },
                title="Install package confirmation",
                preview_payload={
                    **preview,
                    "preview": {
                        "command": command_preview,
                        "manager": preview.get("manager"),
                        "package": preview.get("package"),
                        "scope": preview.get("scope"),
                    },
                },
            )
        if not callable(getattr(truth, "shell_install_package", None)):
            return self._runtime_state_unavailable_response(
                route="action_tool",
                reason="shell_install_package_unavailable",
            )
        payload = truth.shell_install_package(
            manager=manager,
            package=package,
            scope=scope,
            dry_run=dry_run,
        )
        payload = dict(payload) if isinstance(payload, dict) else {}
        return self._shell_command_payload_response(payload)

    def _model_controller_switch_back_response(self, user_id: str, *, confirmed: bool = False) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="model_status",
                reason="runtime_truth_service_unavailable",
            )
        state = self._current_model_trial_state(user_id)
        previous_model = str(state.get("previous_model") or "").strip() or None
        previous_provider = str(state.get("previous_provider") or "").strip().lower() or None
        if not previous_model:
            return self._execute_model_controller_switch_back(user_id)
        if not confirmed:
            question = (
                f"I will switch chat back to {previous_model}. "
                "This mutates the active chat target. Reply yes to proceed or no to cancel."
            )
            return self._confirmation_preview_response(
                user_id,
                route="model_status",
                question=question,
                used_tools=["model_controller"],
                action={
                    "operation": "model_switch_back",
                    "params": {
                        "model_id": previous_model,
                        "provider_id": previous_provider,
                    },
                },
                title="Switch back confirmation",
                preview_payload={
                    "provider": previous_provider,
                    "model_id": previous_model,
                    "preview": {
                        "provider": previous_provider,
                        "model_id": previous_model,
                        "switch_kind": "restore_previous",
                    },
                },
            )
        return self._execute_model_controller_switch_back(user_id)
        if not callable(getattr(truth, "shell_install_package", None)):
            return self._runtime_state_unavailable_response(
                route="action_tool",
                reason="shell_install_package_unavailable",
            )
        payload = truth.shell_install_package(
            manager=manager,
            package=package,
            scope=scope,
            dry_run=dry_run,
        )
        payload = dict(payload) if isinstance(payload, dict) else {}
        return self._shell_command_payload_response(payload)

    def _shell_create_directory_response(self, user_id: str, path_hint: str | None, *, confirmed: bool = False) -> OrchestratorResponse:
        if not str(path_hint or "").strip():
            return self._runtime_truth_response(
                text="Tell me the exact directory path you want me to create.",
                route="action_tool",
                used_tools=["shell"],
                used_runtime_state=False,
                ok=False,
                error_kind="missing_path",
                next_question="Tell me the exact directory path you want me to create.",
                payload={
                    "type": "action_clarification",
                    "kind": "shell_create_directory",
                    "summary": "Tell me the exact directory path you want me to create.",
                    "next_question": "Tell me the exact directory path you want me to create.",
                },
            )
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="action_tool",
                reason="shell_create_directory_unavailable",
            )
        if not confirmed:
            preview_fn = getattr(truth, "shell_preview_create_directory", None)
            if not callable(preview_fn):
                return self._runtime_state_unavailable_response(
                    route="action_tool",
                    reason="shell_create_directory_preview_unavailable",
                )
            preview = preview_fn(path_hint)
            preview = dict(preview) if isinstance(preview, dict) else {}
            if not bool(preview.get("ok", False)):
                message = self._shell_blocked_message(preview)
                return self._runtime_truth_response(
                    text=message,
                    route="action_tool",
                    used_tools=["shell"],
                    ok=False,
                    error_kind=str(preview.get("blocked_reason") or preview.get("error_kind") or "shell_error").strip() or "shell_error",
                    payload={
                        **preview,
                        "title": "Create directory",
                        "summary": message,
                    },
                )
            target = str(preview.get("resolved_path") or preview.get("path") or path_hint).strip() or "that directory"
            if not bool(preview.get("mutated", False)):
                message = f"Directory {target} already exists."
                return self._runtime_truth_response(
                    text=message,
                    route="action_tool",
                    used_tools=["shell"],
                    payload={
                        **preview,
                        "type": "shell_create_directory_preview",
                        "title": "Create directory",
                        "summary": message,
                    },
                )
            question = (
                f"I will create the directory {target}. "
                "This mutates the local filesystem. Reply yes to proceed or no to cancel."
            )
            return self._confirmation_preview_response(
                user_id=user_id,
                route="action_tool",
                question=question,
                used_tools=["shell"],
                action={
                    "operation": "shell_create_directory",
                    "params": {
                        "path": str(preview.get("path") or path_hint).strip() or path_hint,
                    },
                },
                title="Create directory confirmation",
                preview_payload={
                    **preview,
                    "preview": {
                        "path": str(preview.get("resolved_path") or preview.get("path") or path_hint).strip() or None,
                    },
                },
            )
        if not callable(getattr(truth, "shell_create_directory", None)):
            return self._runtime_state_unavailable_response(
                route="action_tool",
                reason="shell_create_directory_unavailable",
            )
        payload = truth.shell_create_directory(path_hint)
        payload = dict(payload) if isinstance(payload, dict) else {}
        if not bool(payload.get("ok", False)):
            message = self._shell_blocked_message(payload)
            return self._runtime_truth_response(
                text=message,
                route="action_tool",
                used_tools=["shell"],
                ok=False,
                error_kind=str(payload.get("blocked_reason") or payload.get("error_kind") or "shell_error").strip() or "shell_error",
                payload={
                    **payload,
                    "title": "Create directory",
                    "summary": message,
                },
            )
        target = str(payload.get("resolved_path") or payload.get("path") or path_hint).strip() or "that directory"
        if bool(payload.get("created", False)):
            message = f"Created directory {target}."
        else:
            message = f"Directory {target} already exists."
        return self._runtime_truth_response(
            text=message,
            route="action_tool",
            used_tools=["shell"],
            payload={
                **payload,
                "title": "Create directory",
                "summary": message,
            },
        )

    def _model_switch_advisory_response(self) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="action_tool",
                reason="runtime_truth_service_unavailable",
            )
        policy_selection = (
            truth.model_policy_status()
            if callable(getattr(truth, "model_policy_status", None))
            else {}
        )
        candidate_payload = (
            truth.model_policy_candidate(status=policy_selection)
            if callable(getattr(truth, "model_policy_candidate", None))
            else {}
        )
        policy_payload = (
            truth.model_controller_policy_status(
                target_truth={
                    "effective_provider": (
                        str((policy_selection if isinstance(policy_selection, dict) else {}).get("current_active_provider") or "").strip().lower()
                        or None
                    ),
                    "effective_model": (
                        str((policy_selection if isinstance(policy_selection, dict) else {}).get("current_active_model") or "").strip()
                        or None
                    ),
                    "configured_provider": (
                        str((policy_selection if isinstance(policy_selection, dict) else {}).get("current_default_provider") or "").strip().lower()
                        or None
                    ),
                    "configured_model": (
                        str((policy_selection if isinstance(policy_selection, dict) else {}).get("current_default_model") or "").strip()
                        or None
                    ),
                }
            )
            if callable(getattr(truth, "model_controller_policy_status", None))
            else {}
        )
        candidate_payload = dict(candidate_payload) if isinstance(candidate_payload, dict) else {}
        policy_payload = dict(policy_payload) if isinstance(policy_payload, dict) else {}
        selection = candidate_payload.get("selection") if isinstance(candidate_payload.get("selection"), dict) else {}
        candidate = candidate_payload.get("candidate") if isinstance(candidate_payload.get("candidate"), dict) else None
        current_label = self._model_policy_candidate_label(
            selection.get("selected_candidate") if isinstance(selection.get("selected_candidate"), dict) else None
        ) or "the current model"
        candidate_label = self._model_policy_candidate_label(candidate)

        if bool(candidate_payload.get("switch_recommended", False)) and candidate_label:
            detail = str(candidate_payload.get("decision_detail") or "").strip()
            message = f"I would recommend {candidate_label} right now."
            if detail:
                message = f"{message} Reason: {detail}"
        else:
            detail = str(candidate_payload.get("decision_detail") or "").strip().rstrip(".")
            message = (
                f"I would keep {current_label} right now because {detail.lower()}."
                if detail
                else f"I would keep {current_label} right now."
            )

        approval_line = (
            "No change has been made. I still need your approval before I test a model, switch temporarily, or make a default change."
        )
        full_message = f"{message}\n{approval_line}"
        return self._runtime_truth_response(
            text=full_message,
            route="action_tool",
            used_runtime_state=True,
            used_tools=["model_controller"],
            payload={
                "type": "model_controller",
                "mode": "switch_advisory",
                "title": "Switch advisory",
                "summary": full_message,
                "switch_recommended": bool(candidate_payload.get("switch_recommended", False)),
                "candidate": dict(candidate) if isinstance(candidate, dict) else None,
                "selection": dict(selection),
                "policy": policy_payload,
                "source": "runtime_truth.model_policy_candidate+model_controller_policy_status",
            },
        )

    def _agent_memory_response(
        self,
        user_id: str,
        kind: str,
        *,
        query_text: str | None = None,
    ) -> OrchestratorResponse:
        normalized_kind = str(kind or "").strip().lower() or "agent_memory_inspect"
        if normalized_kind == "agent_memory_preferences":
            payload = self._preferences_cards_payload()
            response = self._cards_response(user_id, payload)
            return self._merge_response_data(
                response,
                route="agent_memory",
                used_runtime_state=False,
                used_llm=False,
                used_memory=True,
                used_tools=["memory_store"],
                ok=True,
                skip_friction_formatting=True,
                skip_epistemic_gate=True,
                runtime_payload={
                    "type": "agent_memory",
                    "kind": "preferences",
                    "summary": str(payload.get("summary") or "Current saved assistant preferences.").strip(),
                    **dict(payload),
                },
            )
        if normalized_kind == "agent_memory_open_loops":
            payload = self._open_loops_payload(status="open", order="due")
            response = self._cards_response(user_id, payload)
            return self._merge_response_data(
                response,
                route="agent_memory",
                used_runtime_state=False,
                used_llm=False,
                used_memory=True,
                used_tools=["memory_store"],
                ok=True,
                skip_friction_formatting=True,
                skip_epistemic_gate=True,
                runtime_payload={
                    "type": "agent_memory",
                    "kind": "open_loops",
                    "summary": str(payload.get("summary") or "Tracked open loops.").strip(),
                    **dict(payload),
                },
            )
        if normalized_kind == "agent_memory_inspect":
            return self._assistant_memory_overview_response(user_id, query_text=query_text)
        thread_id = self._active_thread_id_for_user(user_id)
        prefs = self.db.list_preferences()
        anchors = self.db.list_thread_anchors(thread_id, limit=5)
        open_loops = self.db.list_open_loops(status="open", limit=5, order="due")
        snapshot = self._memory_runtime.deterministic_snapshot(user_id, thread_id=thread_id)
        summary = snapshot.get("memory_summary") if isinstance(snapshot.get("memory_summary"), dict) else {}
        pref_keys = [str(row.get("key") or "").strip() for row in prefs if isinstance(row, dict) and str(row.get("key") or "").strip()][:3]
        open_loop_titles = [str(row.get("title") or "").strip() for row in open_loops if isinstance(row, dict) and str(row.get("title") or "").strip()][:2]
        anchor_titles = [str(row.get("title") or "").strip() for row in anchors if isinstance(row, dict) and str(row.get("title") or "").strip()][:2]
        lines = ["When you say memory here, I mean my local saved memory rather than system RAM."]
        if prefs:
            lines.append(
                f"I currently have {len(prefs)} saved preference entries"
                + (f", including {', '.join(pref_keys)}." if pref_keys else ".")
            )
        else:
            lines.append("I do not currently have saved preference entries.")
        if anchor_titles:
            lines.append(f"For this thread I also have anchors such as {', '.join(anchor_titles)}.")
        if open_loops:
            lines.append(
                f"I am tracking {len(open_loops)} open loops"
                + (f", including {', '.join(open_loop_titles)}." if open_loop_titles else ".")
            )
        else:
            lines.append("I am not tracking any open loops right now.")
        current_topic = str(summary.get("current_topic") or "").strip()
        if current_topic and current_topic != "none":
            lines.append(f"My current conversation topic is {current_topic}.")
        lines.append("If you want, I can show your preferences or open loops next.")
        message = " ".join(lines)
        return self._runtime_truth_response(
            text=message,
            route="agent_memory",
            used_runtime_state=False,
            used_memory=True,
            used_tools=["memory_store"],
            payload={
                "type": "agent_memory",
                "kind": "memory_inspect",
                "summary": message,
                "preferences_count": len(prefs),
                "anchor_count": len(anchors),
                "open_loop_count": len(open_loops),
                "thread_id": thread_id,
            },
        )

    def _operational_status_response(self, user_id: str, text: str, kind: str) -> OrchestratorResponse:
        normalized_kind = str(kind or "").strip().lower()
        if normalized_kind == "operational_doctor":
            result = self._tool_handler_doctor({}, user_id)
            return self._runtime_truth_response(
                text=str(result.get("user_text") or "").strip() or "Doctor report ready.",
                route="operational_status",
                used_runtime_state=False,
                used_tools=["doctor"],
                payload={
                    "type": "operational_status",
                    "kind": "doctor",
                    "summary": str(result.get("user_text") or "").strip() or "Doctor report ready.",
                },
            )
        if normalized_kind == "operational_agent_status":
            result = self._tool_handler_status({}, user_id)
            return self._runtime_truth_response(
                text=str(result.get("user_text") or "").strip() or "Agent status ready.",
                route="operational_status",
                used_runtime_state=False,
                used_tools=["status"],
                payload={
                    "type": "operational_status",
                    "kind": "status",
                    "summary": str(result.get("user_text") or "").strip() or "Agent status ready.",
                },
            )
        nl_decision = nl_route(text)
        if str(nl_decision.get("intent") or "") not in {"OBSERVE_PC", "EXPLAIN_PREVIOUS"}:
            result = self._tool_handler_observe_system_health({}, user_id)
            return self._runtime_truth_response(
                text=str(result.get("user_text") or "").strip() or "System health report ready.",
                route="operational_status",
                used_runtime_state=False,
                used_tools=["observe_system_health"],
                payload={
                    "type": "operational_status",
                    "kind": "observe_system_health",
                    "summary": str(result.get("user_text") or "").strip() or "System health report ready.",
                },
            )
        response = self._handle_nl_observe(user_id, text, nl_decision)
        payload = dict(response.data) if isinstance(response.data, dict) else {}
        summary = str(payload.get("summary") or response.text or "").strip() or "Status snapshot ready."
        return self._runtime_truth_response(
            text=str(response.text or "").strip() or summary,
            route="operational_status",
            used_runtime_state=False,
            used_tools=[
                str(item.get("function") or "").strip()
                for item in (nl_decision.get("skills") if isinstance(nl_decision.get("skills"), list) else [])
                if isinstance(item, dict) and str(item.get("function") or "").strip()
            ],
            payload={
                "type": "operational_status",
                "kind": str(nl_decision.get("intent") or "OBSERVE_PC"),
                "summary": summary,
                **payload,
            },
        )

    @staticmethod
    def _format_cost_per_1m(value: Any) -> str:
        return f"${float(value or 0.0):.2f} per 1M tokens"

    @staticmethod
    def _model_policy_candidate_label(candidate: dict[str, Any] | None) -> str | None:
        if not isinstance(candidate, dict):
            return None
        model_id = str(candidate.get("model_id") or "").strip()
        if model_id:
            return model_id
        provider_id = str(candidate.get("provider_id") or "").strip().lower()
        if provider_id:
            return provider_id
        return None

    @staticmethod
    def _model_controller_policy_message(payload: dict[str, Any]) -> str:
        mode = str(payload.get("mode") or "safe").strip().lower() or "safe"
        mode_label = str(payload.get("mode_label") or ("SAFE MODE" if mode == "safe" else "Controlled Mode")).strip() or ("SAFE MODE" if mode == "safe" else "Controlled Mode")
        mode_source = str(payload.get("mode_source") or "config_default").strip().lower() or "config_default"
        allow_remote_recommendation = bool(payload.get("allow_remote_recommendation", payload.get("allow_remote_fallback", True)))
        status_line = (
            f"Status: {mode_label} is active because you explicitly turned it on."
            if mode_source == "explicit_override"
            else f"Status: {mode_label} is the current baseline."
        )
        if mode == "safe":
            lines = [
                "Mode: SAFE MODE.",
                status_line,
                "Allowed: I can inspect runtime, setup, and model status, and I can recommend local models.",
                "Blocked: remote switching and install/download/import.",
                "Transition: Controlled Mode only starts if you turn it on explicitly.",
                "Approval: I still need your approval before I test a model, switch temporarily, make a default change, or switch back.",
            ]
        else:
            allowed_line = (
                "Allowed: I can recommend ready local models and usable cloud models when provider health and policy allow them."
                if allow_remote_recommendation
                else "Allowed: I can recommend ready local models. Cloud recommendations are paused by policy right now."
            )
            install_line = (
                "Approval: I need your explicit approval before I test a model, switch temporarily, make a model the default, or acquire/install it through the canonical model manager."
                if bool(payload.get("allow_install_pull", True))
                else "Approval: I need your explicit approval before I test a model, switch temporarily, or make a model the default. Install/download is paused by policy right now."
            )
            lines = [
                f"Mode: {mode_label}.",
                status_line,
                allowed_line,
                "Automatic actions: none. I will not switch or install on my own.",
                "Transition: You can return to SAFE MODE at any time.",
                install_line,
            ]
        return "\n".join(lines)

    def _model_controller_policy_payload_response(
        self,
        payload: dict[str, Any],
        *,
        route: str,
        used_tools: list[str] | None = None,
        ok: bool = True,
        error_kind: str | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> OrchestratorResponse:
        message = self._model_controller_policy_message(payload)
        runtime_payload = {
            **dict(payload),
            "type": "model_controller_policy",
            "title": "Model control mode",
            "summary": message,
        }
        if isinstance(extra_payload, dict):
            runtime_payload.update(extra_payload)
        return self._runtime_truth_response(
            text=message,
            route=route,
            payload=runtime_payload,
            used_tools=used_tools,
            ok=ok,
            error_kind=error_kind,
        )

    def _model_controller_policy_response(self) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="model_policy_status",
                reason="runtime_truth_service_unavailable",
            )
        payload = truth.model_controller_policy_status()
        return self._model_controller_policy_payload_response(
            dict(payload) if isinstance(payload, dict) else {},
            route="model_policy_status",
        )

    def _control_mode_change_response(self, requested_mode: str) -> OrchestratorResponse:
        adapter = self._chat_runtime_adapter
        set_mode = getattr(adapter, "llm_control_mode_set", None)
        if not callable(set_mode):
            return self._runtime_state_unavailable_response(
                route="action_tool",
                reason="control_mode_unavailable",
            )
        ok, body = set_mode(
            {
                "mode": str(requested_mode or "").strip().lower(),
                "confirm": True,
                "actor": "assistant",
            }
        )
        response_body = dict(body) if isinstance(body, dict) else {}
        policy_payload = (
            dict(response_body.get("policy"))
            if isinstance(response_body.get("policy"), dict)
            else {}
        )
        if not policy_payload and callable(getattr(adapter, "llm_control_mode_status", None)):
            status_payload = adapter.llm_control_mode_status()
            policy_payload = dict(status_payload) if isinstance(status_payload, dict) else {}
        if not policy_payload:
            truth = self._runtime_truth()
            if truth is not None and callable(getattr(truth, "model_controller_policy_status", None)):
                status_payload = truth.model_controller_policy_status()
                policy_payload = dict(status_payload) if isinstance(status_payload, dict) else {}
        if not ok:
            message = str(response_body.get("message") or "").strip() or "I couldn't change the control mode right now."
            return self._runtime_truth_response(
                text=message,
                route="action_tool",
                ok=False,
                error_kind=str(response_body.get("error_kind") or response_body.get("error") or "control_mode_change_failed").strip() or "control_mode_change_failed",
                used_tools=["model_controller"],
                payload={
                    "type": "model_controller_policy",
                    "action": "control_mode_set",
                    "requested_mode": str(requested_mode or "").strip().lower() or None,
                    "summary": message,
                    "ok": False,
                    "result": response_body,
                    **policy_payload,
                },
            )
        return self._model_controller_policy_payload_response(
            policy_payload,
            route="action_tool",
            used_tools=["model_controller"],
            extra_payload={
                "action": "control_mode_set",
                "requested_mode": str(requested_mode or "").strip().lower() or None,
                "result": response_body,
            },
        )

    def _model_policy_status_response(self) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="model_policy_status",
                reason="runtime_truth_service_unavailable",
            )
        payload = truth.model_policy_status()
        cheap_cap = self._format_cost_per_1m(payload.get("cheap_remote_cap_per_1m"))
        general_cap = self._format_cost_per_1m(payload.get("general_remote_cap_per_1m"))
        selected = self._model_policy_candidate_label(
            payload.get("selected_candidate") if isinstance(payload.get("selected_candidate"), dict) else None
        )
        if bool(payload.get("switch_recommended", False)):
            recommended = self._model_policy_candidate_label(
                payload.get("recommended_candidate") if isinstance(payload.get("recommended_candidate"), dict) else None
            )
            message = (
                "Your model-selection policy is local first: I keep the strongest healthy approved local chat model "
                f"when it is good enough, then free remote models, then cheap remote models under {cheap_cap}. "
                f"Ordinary routing still uses the broader default cap of {general_cap}. "
                f"Right now I would recommend {recommended or 'a different model'}, but I will not switch automatically."
            )
        else:
            message = (
                "Your model-selection policy is local first: I keep the strongest healthy approved local chat model "
                f"when it is good enough, then free remote models, then cheap remote models under {cheap_cap}. "
                f"Ordinary routing still uses the broader default cap of {general_cap}. "
                f"Right now I would keep {selected or 'the current default'}."
            )
        return self._runtime_truth_response(
            text=message,
            route="model_policy_status",
            payload={
                **dict(payload),
                "type": "model_policy_status",
                "title": "Model selection policy",
                "summary": message,
            },
        )

    def _model_policy_cap_response(self) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="model_policy_status",
                reason="runtime_truth_service_unavailable",
            )
        payload = truth.model_policy_status()
        cheap_cap = self._format_cost_per_1m(payload.get("cheap_remote_cap_per_1m"))
        general_cap = self._format_cost_per_1m(payload.get("general_remote_cap_per_1m"))
        message = (
            f"Your cheap remote recommendation cap is {cheap_cap}. "
            f"Ordinary routing still uses the broader default cap of {general_cap}."
        )
        return self._runtime_truth_response(
            text=message,
            route="model_policy_status",
            payload={
                **dict(payload),
                "type": "model_policy_status",
                "title": "Cheap remote cap",
                "summary": message,
            },
        )

    def _model_policy_current_choice_response(self) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="model_policy_status",
                reason="runtime_truth_service_unavailable",
            )
        payload = truth.model_policy_status()
        current_candidate = payload.get("current_candidate") if isinstance(payload.get("current_candidate"), dict) else {}
        selected_candidate = payload.get("selected_candidate") if isinstance(payload.get("selected_candidate"), dict) else {}
        free_remote = (
            payload.get("tier_candidates").get("free_remote")
            if isinstance(payload.get("tier_candidates"), dict)
            and isinstance(payload.get("tier_candidates").get("free_remote"), dict)
            else None
        )
        current_label = self._model_policy_candidate_label(current_candidate) or self._model_policy_candidate_label(selected_candidate) or "the current model"
        if bool(payload.get("switch_recommended", False)):
            recommended_label = self._model_policy_candidate_label(
                payload.get("recommended_candidate") if isinstance(payload.get("recommended_candidate"), dict) else None
            )
            message = (
                f"I am still using {current_label} because it is the current default, but policy would recommend "
                f"{recommended_label or 'another model'} because {str(payload.get('decision_detail') or '').strip().lower()}. "
                "I still need your approval before I try that model."
            )
        else:
            detail = str(payload.get("decision_detail") or "").strip().rstrip(".")
            message = f"I am using {current_label} because {detail.lower()}."
            if bool((current_candidate or selected_candidate).get("local", False)) and free_remote is not None:
                message = f"{message.rstrip('.')} Free remote candidates were not preferred because local-first retained the current local model."
        return self._runtime_truth_response(
            text=message,
            route="model_policy_status",
            payload={
                **dict(payload),
                "type": "model_policy_explanation",
                "title": "Current model choice",
                "summary": message,
            },
        )

    def _model_policy_provider_explanation_response(self, provider_id: str | None) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="model_policy_status",
                reason="runtime_truth_service_unavailable",
            )
        provider_key = str(provider_id or "").strip().lower() or None
        provider_payload = truth.model_policy_provider_candidate(provider_key)
        provider_status = provider_payload.get("provider_status") if isinstance(provider_payload.get("provider_status"), dict) else {}
        provider_label = self._setup_provider_label(provider_key)
        selection = provider_payload.get("selection") if isinstance(provider_payload.get("selection"), dict) else {}
        provider_selection = provider_payload.get("provider_selection") if isinstance(provider_payload.get("provider_selection"), dict) else {}
        candidate = provider_payload.get("candidate") if isinstance(provider_payload.get("candidate"), dict) else None
        current_label = self._model_policy_candidate_label(
            selection.get("selected_candidate") if isinstance(selection.get("selected_candidate"), dict) else None
        ) or "the current model"

        if not bool(provider_status.get("configured", False)):
            message = f"I didn't switch to {provider_label} because it is not configured for chat right now."
        elif candidate is None:
            rejected = next(
                (
                    row
                    for row in (
                        provider_selection.get("rejected_candidates")
                        if isinstance(provider_selection.get("rejected_candidates"), list)
                        else []
                    )
                    if isinstance(row, dict)
                ),
                {},
            )
            rejected_reason = str(rejected.get("reason") or "").strip().lower()
            if rejected_reason == "cheap_remote_cap_exceeded":
                message = (
                    f"I didn't switch to {provider_label} because its paid models are above the automatic cheap cap of "
                    f"{self._format_cost_per_1m(selection.get('cheap_remote_cap_per_1m'))}."
                )
            elif rejected_reason == "auth_missing":
                message = f"I didn't switch to {provider_label} because its API key is not available."
            else:
                message = f"I didn't switch to {provider_label} because it does not have a policy-allowed chat model right now."
        elif bool(selection.get("switch_recommended", False)) and str((candidate or {}).get("provider_id") or "").strip().lower() == provider_key:
            message = f"I would recommend {provider_label} with {self._model_policy_candidate_label(candidate) or 'its best candidate'} right now, but I would still ask first."
        else:
            message = (
                f"I didn't switch to {provider_label} because {str(selection.get('decision_detail') or '').strip().lower()}. "
                f"The best {provider_label} candidate right now is {self._model_policy_candidate_label(candidate) or 'unavailable'}, "
                f"and local-first kept {current_label}."
            )
        return self._runtime_truth_response(
            text=message,
            route="model_policy_status",
            payload={
                **dict(provider_payload),
                "type": "model_policy_explanation",
                "title": f"{provider_label} policy explanation",
                "summary": message,
                "provider_id": provider_key,
            },
        )

    def _model_policy_switch_candidate_response(self) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="model_policy_status",
                reason="runtime_truth_service_unavailable",
            )
        payload = truth.model_policy_candidate()
        candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else None
        selection = payload.get("selection") if isinstance(payload.get("selection"), dict) else {}
        current_label = self._model_policy_candidate_label(
            selection.get("selected_candidate") if isinstance(selection.get("selected_candidate"), dict) else None
        ) or "the current model"
        candidate_label = self._model_policy_candidate_label(candidate)
        if bool(payload.get("switch_recommended", False)) and candidate_label:
            message = f"I would recommend {candidate_label} right now, but I will not switch automatically."
        else:
            message = f"I would keep {current_label} right now because {str(payload.get('decision_detail') or '').strip().lower()}."
        return self._runtime_truth_response(
            text=message,
            route="model_policy_status",
            payload={
                **dict(payload),
                "type": "model_policy_candidate",
                "title": "Best current candidate",
                "summary": message,
            },
        )

    def _model_policy_tier_candidate_response(self, tier: str | None) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="model_policy_status",
                reason="runtime_truth_service_unavailable",
            )
        tier_key = str(tier or "").strip().lower() or None
        payload = truth.model_policy_candidate(tier_key)
        candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else None
        tier_label = "free remote" if tier_key == "free_remote" else "cheap remote"
        if candidate is None:
            if tier_key == "cheap_remote":
                message = (
                    f"I do not have a cheap remote model under the automatic cap of "
                    f"{self._format_cost_per_1m(payload.get('cheap_remote_cap_per_1m'))} right now."
                )
            else:
                message = f"I do not have a {tier_label} model available right now."
        else:
            message = f"I would choose {self._model_policy_candidate_label(candidate)} as the best {tier_label} model right now."
        return self._runtime_truth_response(
            text=message,
            route="model_policy_status",
            payload={
                **dict(payload),
                "type": "model_policy_candidate",
                "title": f"{tier_label.title()} candidate",
                "summary": message,
            },
        )

    def _find_ollama_models_response(self, query: str | None = None) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="model_status",
                reason="runtime_truth_service_unavailable",
            )
        discovery = truth.model_discovery_query(query=query, filters={"sources": ["ollama"]})
        models = [dict(row) for row in (discovery.get("models") if isinstance(discovery.get("models"), list) else []) if isinstance(row, dict)]
        preview = ", ".join(
            str(row.get("id") or row.get("model_name") or row.get("model") or "").strip()
            for row in models[:5]
            if str(row.get("id") or row.get("model_name") or row.get("model") or "").strip()
        )
        message = str(discovery.get("message") or "").strip()
        if models and preview and preview not in message:
            message = f"{message} Installed Ollama models: {preview}."
        return self._runtime_truth_response(
            text=message,
            route="model_status",
            used_runtime_state=True,
            used_tools=["model_discovery_manager"],
            ok=bool(discovery.get("ok", False)),
            payload={
                "type": "model_discovery",
                "mode": "local_only",
                "summary": message,
                "query": query,
                "models": models[:10],
                "sources": discovery.get("sources") if isinstance(discovery, dict) else [],
                "debug": discovery.get("debug") if isinstance(discovery, dict) else {},
                "source": "runtime_truth.model_discovery_query",
            },
        )

    def _execute_switch_better_local_model(
        self,
        user_id: str,
        *,
        model_id: str | None,
    ) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="model_status",
                reason="runtime_truth_service_unavailable",
            )
        candidate_model_id = str(model_id or "").strip() or None
        if not candidate_model_id:
            message = "I could not find a better local model right now."
            return self._runtime_truth_response(
                text=message,
                route="model_status",
                ok=False,
                error_kind="local_model_unavailable",
                payload={
                    "type": "model_switch",
                    "provider": "ollama",
                    "ok": False,
                    "title": "No local model found",
                    "summary": message,
                },
            )
        configure_ok, configure_body = truth.configure_local_chat_model(candidate_model_id)
        configured_model = str((configure_body if isinstance(configure_body, dict) else {}).get("model_id") or candidate_model_id or "").strip() or None
        message = str((configure_body if isinstance(configure_body, dict) else {}).get("message") or "I switched to a local model.")
        return self._runtime_truth_response(
            text=message,
            route="model_status",
            used_memory=False,
            used_tools=["model_controller"],
            error_kind=None if configure_ok else str((configure_body if isinstance(configure_body, dict) else {}).get("error") or "local_model_switch_failed").strip() or "local_model_switch_failed",
            ok=bool(configure_ok),
            payload={
                "type": "model_switch",
                "provider": "ollama",
                "ok": bool(configure_ok),
                "model_id": configured_model,
                "title": "Local model update",
                "summary": message,
            },
        )

    def _switch_better_local_model_response(self, user_id: str, *, confirmed: bool = False) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="model_status",
                reason="runtime_truth_service_unavailable",
            )
        choose_ok, choose_body = truth.choose_best_local_chat_model({"refresh": True})
        if not choose_ok:
            message = str((choose_body if isinstance(choose_body, dict) else {}).get("message") or "I could not find a better local model right now.")
            return self._runtime_truth_response(
                text=message,
                route="model_status",
                error_kind=str((choose_body if isinstance(choose_body, dict) else {}).get("error") or "").strip() or None,
                ok=False,
                payload={
                    "type": "model_switch",
                    "provider": "ollama",
                    "ok": False,
                    "title": "No local model found",
                    "summary": message,
                },
            )
        candidate = choose_body.get("candidate") if isinstance(choose_body.get("candidate"), dict) else {}
        candidate_model_id = str(candidate.get("model_id") or "").strip() or None
        current = truth.current_chat_target_status()
        if candidate_model_id and str(current.get("provider") or "").strip().lower() == "ollama" and str(current.get("model") or "").strip() == candidate_model_id:
            message = f"Chat is already using the best available local model: {candidate_model_id}."
            return self._runtime_truth_response(
                text=message,
                route="model_status",
                payload={
                    "type": "model_switch",
                    "provider": "ollama",
                    "ok": True,
                    "model_id": candidate_model_id,
                    "title": "Local model unchanged",
                    "summary": message,
                },
            )
        if not confirmed:
            question = (
                f"I will switch chat to the best available local model {candidate_model_id}. "
                "This mutates the configured chat model. Reply yes to proceed or no to cancel."
            )
            return self._confirmation_preview_response(
                user_id,
                route="model_status",
                question=question,
                used_tools=["model_controller"],
                action={
                    "operation": "switch_better_local_model",
                    "params": {
                        "model_id": candidate_model_id,
                    },
                },
                title="Best local switch confirmation",
                preview_payload={
                    "provider": "ollama",
                    "model_id": candidate_model_id,
                    "preview": {
                        "provider": "ollama",
                        "model_id": candidate_model_id,
                        "switch_kind": "best_local",
                    },
                },
            )
        return self._execute_switch_better_local_model(user_id, model_id=candidate_model_id)

    def _configure_ollama_response(
        self,
        *,
        make_default: bool,
        used_memory: bool,
    ) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="setup_flow",
                used_memory=used_memory,
                reason="runtime_truth_service_unavailable",
            )
        snapshot = truth.provider_status("ollama")
        current = truth.current_chat_target_status()
        choose_ok, choose_body = truth.choose_best_local_chat_model({"refresh": True})
        if not choose_ok:
            message = str((choose_body if isinstance(choose_body, dict) else {}).get("message") or "I could not find a usable Ollama chat model right now.")
            return self._runtime_truth_response(
                text=message,
                route="setup_flow",
                used_memory=used_memory,
                error_kind=str((choose_body if isinstance(choose_body, dict) else {}).get("error") or "local_model_unavailable").strip() or "local_model_unavailable",
                ok=False,
                payload={
                    "type": "provider_test_result",
                    "provider": "ollama",
                    "ok": False,
                    "title": "Ollama setup",
                    "summary": message,
                },
            )
        candidate = choose_body.get("candidate") if isinstance(choose_body.get("candidate"), dict) else {}
        candidate_model_id = str(candidate.get("model_id") or snapshot.get("model_id") or "").strip() or None
        if candidate_model_id and str(current.get("provider") or "").strip().lower() == "ollama" and str(current.get("model") or "").strip() == candidate_model_id:
            message = f"Ollama is already configured for chat with {candidate_model_id}."
            return self._runtime_truth_response(
                text=message,
                route="setup_flow",
                used_memory=used_memory,
                payload={
                    "type": "setup_complete",
                    "provider": "ollama",
                    "model_id": candidate_model_id,
                    "ok": True,
                    "title": "Ollama ready",
                    "summary": message,
                },
            )
        if not candidate_model_id:
            message = "I could not find a usable Ollama chat model right now."
            return self._runtime_truth_response(
                text=message,
                route="setup_flow",
                used_memory=used_memory,
                ok=False,
                error_kind="local_model_unavailable",
                payload={
                    "type": "provider_test_result",
                    "provider": "ollama",
                    "ok": False,
                    "title": "Ollama setup",
                    "summary": message,
                },
            )
        configure_ok, configure_body = truth.configure_local_chat_model(candidate_model_id)
        message = str((configure_body if isinstance(configure_body, dict) else {}).get("message") or "Ollama is ready for chat.")
        configured_model = str((configure_body if isinstance(configure_body, dict) else {}).get("model_id") or candidate_model_id).strip() or candidate_model_id
        return self._runtime_truth_response(
            text=message,
            route="setup_flow",
            used_memory=used_memory,
            error_kind=None if configure_ok else str((configure_body if isinstance(configure_body, dict) else {}).get("error") or "local_model_switch_failed").strip() or "local_model_switch_failed",
            ok=bool(configure_ok),
            payload={
                "type": "setup_complete" if configure_ok else "provider_test_result",
                "provider": "ollama",
                "model_id": configured_model,
                "ok": bool(configure_ok),
                "title": "Ollama ready" if configure_ok else "Ollama setup",
                "summary": message,
                "make_default": bool(make_default),
            },
        )

    def _setup_explanation_response(self, *, used_memory: bool) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="setup_flow",
                used_memory=used_memory,
                reason="runtime_truth_service_unavailable",
            )
        setup = truth.setup_status()
        active_provider = str(setup.get("active_provider") or "").strip().lower() or None
        active_model = str(setup.get("active_model") or "").strip() or None
        provider_label = self._setup_provider_label(active_provider)
        setup_state = str(setup.get("setup_state") or "unavailable").strip().lower() or "unavailable"
        attention_kind = str(setup.get("attention_kind") or "").strip().lower() or None
        provider_health_status = (
            str(setup.get("provider_health_status") or "unknown").strip().lower() or "unknown"
        )
        health_reason = str(setup.get("provider_health_reason") or "").strip() or None
        local_installed_rows = [
            dict(row)
            for row in (
                setup.get("local_installed_models")
                if isinstance(setup.get("local_installed_models"), list)
                else []
            )
            if isinstance(row, dict)
        ]
        other_local_rows = [
            dict(row)
            for row in (
                setup.get("other_local_models")
                if isinstance(setup.get("other_local_models"), list)
                else []
            )
            if isinstance(row, dict)
        ]

        if setup_state == "ready" and active_model:
            message = f"Setup looks okay right now. Chat is using {active_model} on {provider_label}."
            if active_provider == "ollama" and provider_health_status == "ok":
                message = f"{message.rstrip('.')} Ollama is reachable."
            elif active_provider:
                message = f"{message.rstrip('.')} {provider_label} looks healthy."
            if other_local_rows:
                preview = self._inventory_preview(other_local_rows, limit=4)
                if preview:
                    message = f"{message.rstrip('.')} Other local chat models I can see are {preview}."
            ok = True
        elif setup_state == "attention" and active_model and active_provider:
            if attention_kind == "provider_down":
                message = (
                    f"Setup needs attention right now. Chat is configured for {active_model} on {provider_label}, "
                    f"but {provider_label} is not responding right now."
                )
            elif attention_kind == "provider_degraded":
                message = (
                    f"Setup needs attention right now. Chat is configured for {active_model} on {provider_label}, "
                    f"but {provider_label} needs attention right now."
                )
            elif attention_kind == "model_unhealthy":
                message = (
                    f"Setup needs attention right now. Chat is configured for {active_model} on {provider_label}, "
                    "but that model is not healthy right now."
                )
            else:
                message = (
                    f"Setup needs attention right now. Chat is configured for {active_model} on {provider_label}, "
                    "but it is not ready right now."
                )
            if health_reason:
                message = f"{message.rstrip('.')} Reason: {health_reason}."
            ok = False
        elif setup_state == "inventory_only":
            preview = self._inventory_preview(local_installed_rows, limit=4)
            message = (
                f"I can see local chat models {preview}, but none is active right now."
                if preview
                else "I can see local chat models, but none is active right now."
            )
            ok = False
        else:
            message = (
                "No chat model is available right now. "
                "If you want, I can help you start Ollama or switch to another configured provider."
            )
            ok = False

        return self._runtime_truth_response(
            text=message,
            route="setup_flow",
            used_memory=used_memory,
            used_runtime_state=True,
            payload={
                "type": "setup_explanation",
                "ok": ok,
                "provider": active_provider,
                "model_id": active_model,
                "health_status": provider_health_status,
                "health_reason": health_reason,
                "summary": message,
            },
        )

    def _set_default_model_response(
        self,
        user_id: str,
        text: str,
        state: dict[str, Any],
        *,
        confirmed: bool = False,
    ) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="model_status",
                used_memory=bool(state),
                reason="runtime_truth_service_unavailable",
            )
        promote_default = self._model_controller_promote_requested(text)
        resolution = self._resolve_controller_model_target(user_id, text)
        if str(resolution.get("status") or "") == "ambiguous":
            matches = [
                str(item).strip()
                for item in (resolution.get("matches") if isinstance(resolution.get("matches"), list) else [])
                if str(item).strip()
            ]
            requested = str(resolution.get("requested") or "that model").strip() or "that model"
            options = ", ".join(matches[:3])
            message = (
                f"I can switch to {requested} on more than one provider: {options}. "
                "Which exact model do you want?"
            )
            return self._runtime_truth_response(
                text=message,
                route="model_status",
                used_memory=bool(state),
                next_question=message,
                payload={
                    "type": "action_required",
                    "title": "Which exact model?",
                    "summary": message,
                    "matches": matches,
                },
            )
        matched_model = (
            str(resolution.get("model_id") or "").strip()
            or str(state.get("model_id") or "").strip()
            or None
        )
        provider_id = str(
            resolution.get("provider_id")
            or (matched_model.split(":", 1)[0] if matched_model and ":" in matched_model else "")
            or ""
        ).strip().lower() or None
        if not confirmed:
            if not matched_model:
                message = (
                    "I couldn't find that model in the current runtime registry. "
                    "If you want, I can list the models that are ready now or the local installed models."
                )
                return self._runtime_truth_response(
                    text=message,
                    route="model_status",
                    used_memory=bool(state),
                    next_question=message,
                    payload={
                        "type": "action_required",
                        "title": "Which model?",
                        "summary": message,
                    },
                )
            if promote_default:
                question = (
                    f"I will make {matched_model} the default chat model. "
                    "This mutates the configured chat target. Reply yes to proceed or no to cancel."
                )
            else:
                question = (
                    f"I will switch chat to {matched_model}. "
                    "This mutates the active chat target. Reply yes to proceed or no to cancel."
                )
            return self._confirmation_preview_response(
                user_id,
                route="model_status",
                question=question,
                used_tools=["model_controller"],
                action={
                    "operation": "model_set_target",
                    "params": {
                        "model_id": matched_model,
                        "provider_id": provider_id,
                        "promote_default": promote_default,
                        "used_memory": bool(state),
                    },
                },
                title="Default model confirmation" if promote_default else "Model switch confirmation",
                preview_payload={
                    "provider": provider_id,
                    "model_id": matched_model,
                    "preview": {
                        "provider": provider_id,
                        "model_id": matched_model,
                        "switch_kind": "make_default" if promote_default else "direct_switch",
                    },
                },
                used_memory=bool(state),
            )
        return self._execute_model_set_target(
            user_id,
            model_id=matched_model,
            provider_id=provider_id,
            promote_default=promote_default,
            used_memory=bool(state),
        )

    def _cancel_pending_setup_response(self, user_id: str, state: dict[str, Any]) -> OrchestratorResponse:
        had_pending = bool(state)
        self._clear_runtime_setup_state(user_id)
        step = str(state.get("step") or "").strip().lower()
        if had_pending and step == "awaiting_openrouter_reuse_confirm":
            message = "Okay, I won't use the stored OpenRouter key right now."
        else:
            message = "Okay, I will keep the current chat model." if had_pending else "There is no pending setup right now."
        return self._runtime_truth_response(
            text=message,
            route="setup_flow",
            used_memory=had_pending,
            payload={
                "type": "action_required",
                "title": "No change made",
                "summary": message,
            },
        )

    def _confirm_pending_setup_response(self, user_id: str, state: dict[str, Any]) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="setup_flow",
                used_memory=bool(state),
                reason="runtime_truth_service_unavailable",
            )
        step = str(state.get("step") or "").strip().lower()
        if step == "awaiting_openrouter_reuse_confirm":
            current = truth.current_chat_target_status()
            make_default = bool(state.get("make_default"))
            configure_ok, configure_body = truth.configure_openrouter(
                None,
                {
                    "make_default": make_default,
                    "defer_model_refresh": True,
                },
            )
            configured_model = str((configure_body if isinstance(configure_body, dict) else {}).get("model_id") or "").strip() or None
            if not configure_ok:
                self._save_runtime_setup_state(
                    user_id,
                    {
                        "step": "awaiting_openrouter_key",
                        "provider": "openrouter",
                        "make_default": make_default,
                    },
                )
                message = str((configure_body if isinstance(configure_body, dict) else {}).get("message") or "OpenRouter setup did not succeed.")
                return self._runtime_truth_response(
                    text=message,
                    route="setup_flow",
                    used_memory=True,
                    error_kind=str((configure_body if isinstance(configure_body, dict) else {}).get("error") or "upstream_down").strip() or "upstream_down",
                    ok=False,
                    payload={
                        "type": "provider_test_result",
                        "provider": "openrouter",
                        "model_id": configured_model,
                        "ok": False,
                        "title": "OpenRouter test failed",
                        "summary": message,
                    },
                )
            self._clear_runtime_setup_state(user_id)
            if configured_model and not make_default and bool(current.get("ready", False)):
                prompt = f"OpenRouter is ready. Do you want me to switch chat to {configured_model} now?"
                self._save_runtime_setup_state(
                    user_id,
                    {
                        "step": "awaiting_switch_confirm",
                        "action_type": "confirm_model_switch",
                        "provider": "openrouter",
                        "model_id": configured_model,
                    },
                )
                return self._runtime_truth_response(
                    text=prompt,
                    route="setup_flow",
                    used_memory=True,
                    next_question=prompt,
                    payload={
                        "type": "confirm_switch_model",
                        "provider": "openrouter",
                        "model_id": configured_model,
                        "title": "Use OpenRouter for chat?",
                        "prompt": prompt,
                        "approve_label": "Use OpenRouter",
                        "approve_command": "yes",
                        "cancel_label": "Keep current",
                        "cancel_command": "no",
                        "summary": prompt,
                    },
                )

            message = str((configure_body if isinstance(configure_body, dict) else {}).get("message") or "OpenRouter is ready.")
            return self._runtime_truth_response(
                text=message,
                route="setup_flow",
                used_memory=True,
                payload={
                    "type": "setup_complete",
                    "provider": "openrouter",
                    "model_id": configured_model,
                    "ok": True,
                    "title": "OpenRouter ready",
                    "summary": message,
                },
            )

        pending_model = str(state.get("model_id") or "").strip()
        pending_provider = str(state.get("provider") or "").strip().lower() or None
        pending_action = str(state.get("action_type") or "").strip().lower() or None
        if step != "awaiting_switch_confirm" or pending_action not in {"", "confirm_model_switch", "confirm_model_scout_switch"} or not pending_model:
            self._clear_runtime_setup_state(user_id)
            message = "There is no pending model switch right now."
            return self._runtime_truth_response(
                text=message,
                route="setup_flow",
                used_memory=bool(state),
                payload={
                    "type": "action_required",
                    "title": "No pending switch",
                    "summary": message,
                },
            )
        current_target = truth.current_chat_target_status()
        previous_provider, previous_model = self._target_snapshot_from_truth(current_target)
        previous_provider = str(state.get("previous_provider") or previous_provider or "").strip().lower() or None
        previous_model = str(state.get("previous_model") or previous_model or "").strip() or None
        explicit_setter = getattr(truth, "set_confirmed_chat_model_target", None)
        if callable(explicit_setter):
            default_ok, default_body = explicit_setter(
                pending_model,
                provider_id=pending_provider,
            )
        else:
            default_ok, default_body = truth.set_default_chat_model(pending_model)
        self._clear_runtime_setup_state(user_id)
        applied_provider = str((default_body if isinstance(default_body, dict) else {}).get("provider") or pending_provider).strip().lower() or None
        applied_model = str((default_body if isinstance(default_body, dict) else {}).get("model_id") or pending_model).strip() or None
        if bool(default_ok):
            self._record_model_trial_switch(
                user_id,
                previous_provider=previous_provider,
                previous_model=previous_model,
                applied_provider=applied_provider,
                applied_model=applied_model,
                source="model_scout_v2" if pending_action == "confirm_model_scout_switch" else "confirmed_switch",
            )
        return self._post_switch_response(
            truth=truth,
            route="setup_flow",
            used_memory=True,
            used_tools=["model_controller"],
            ok=bool(default_ok),
            body=default_body if isinstance(default_body, dict) else {},
            applied_provider=applied_provider,
            applied_model=applied_model,
            success_type="setup_complete",
            success_title="Chat model updated",
            failure_title="Model switch failed",
        )

    def _configure_openrouter_response(
        self,
        user_id: str,
        decision: dict[str, Any],
        state: dict[str, Any],
    ) -> OrchestratorResponse:
        truth = self._runtime_truth()
        if truth is None:
            return self._runtime_state_unavailable_response(
                route="setup_flow",
                used_memory=bool(state),
                reason="runtime_truth_service_unavailable",
            )
        make_default = bool(decision.get("make_default") or state.get("make_default"))
        current = truth.current_chat_target_status()
        if not bool(current.get("ready", False)):
            if not str(current.get("model") or "").strip():
                make_default = True
        provider_snapshot = truth.provider_status("openrouter")
        api_key = str(decision.get("api_key") or "").strip()
        if not api_key and not bool(provider_snapshot.get("secret_present", False)):
            self._save_runtime_setup_state(
                user_id,
                {
                    "step": "awaiting_openrouter_key",
                    "provider": "openrouter",
                    "make_default": make_default,
                },
            )
            message = "Paste your OpenRouter API key and I will finish the setup."
            return self._runtime_truth_response(
                text=message,
                route="setup_flow",
                used_memory=bool(state),
                next_question=message,
                payload={
                    "type": "request_secret",
                    "provider": "openrouter",
                    "secret_kind": "api_key",
                    "title": "OpenRouter key needed",
                    "prompt": message,
                    "submit_hint": "Paste the key directly in chat.",
                    "summary": message,
                },
            )
        if not api_key and not make_default and bool(provider_snapshot.get("secret_present", False)):
            self._save_runtime_setup_state(
                user_id,
                {
                    "step": "awaiting_openrouter_reuse_confirm",
                    "provider": "openrouter",
                    "make_default": False,
                },
            )
            message = "I already have an OpenRouter API key stored. Reply yes and I will test it now, or paste a new key to replace it."
            return self._runtime_truth_response(
                text=message,
                route="setup_flow",
                used_memory=bool(state),
                next_question=message,
                payload={
                    "type": "confirm_reuse_secret",
                    "provider": "openrouter",
                    "title": "Stored OpenRouter key available",
                    "prompt": message,
                    "approve_label": "Use stored key",
                    "approve_command": "yes",
                    "cancel_label": "Cancel",
                    "cancel_command": "no",
                    "submit_hint": "Or paste a new OpenRouter API key.",
                    "summary": message,
                },
            )

        configure_ok, configure_body = truth.configure_openrouter(
            api_key,
            {
                "make_default": make_default,
                "defer_model_refresh": True,
            },
        )
        configured_model = str((configure_body if isinstance(configure_body, dict) else {}).get("model_id") or "").strip() or None
        if not configure_ok:
            self._save_runtime_setup_state(
                user_id,
                {
                    "step": "awaiting_openrouter_key",
                    "provider": "openrouter",
                    "make_default": make_default,
                },
            )
            message = str((configure_body if isinstance(configure_body, dict) else {}).get("message") or "OpenRouter setup did not succeed.")
            return self._runtime_truth_response(
                text=message,
                route="setup_flow",
                used_memory=bool(state) or bool(api_key),
                error_kind=str((configure_body if isinstance(configure_body, dict) else {}).get("error") or "upstream_down").strip() or "upstream_down",
                ok=False,
                payload={
                    "type": "provider_test_result",
                    "provider": "openrouter",
                    "model_id": configured_model,
                    "ok": False,
                    "title": "OpenRouter test failed",
                    "summary": message,
                },
            )

        self._clear_runtime_setup_state(user_id)
        if configured_model and not make_default and bool(current.get("ready", False)):
            prompt = f"OpenRouter is ready. Do you want me to switch chat to {configured_model} now?"
            self._save_runtime_setup_state(
                user_id,
                {
                    "step": "awaiting_switch_confirm",
                    "action_type": "confirm_model_switch",
                    "provider": "openrouter",
                    "model_id": configured_model,
                },
            )
            return self._runtime_truth_response(
                text=prompt,
                route="setup_flow",
                used_memory=bool(state),
                next_question=prompt,
                payload={
                    "type": "confirm_switch_model",
                    "provider": "openrouter",
                    "model_id": configured_model,
                    "title": "Use OpenRouter for chat?",
                    "prompt": prompt,
                    "approve_label": "Use OpenRouter",
                    "approve_command": "yes",
                    "cancel_label": "Keep current",
                    "cancel_command": "no",
                    "summary": prompt,
                },
            )

        message = str((configure_body if isinstance(configure_body, dict) else {}).get("message") or "OpenRouter is ready.")
        return self._runtime_truth_response(
            text=message,
            route="setup_flow",
            used_memory=bool(state),
            payload={
                "type": "setup_complete",
                "provider": "openrouter",
                "model_id": configured_model,
                "ok": True,
                "title": "OpenRouter ready",
                "summary": message,
            },
        )

    def _handle_runtime_truth_chat(self, user_id: str, text: str) -> OrchestratorResponse | None:
        state = self._current_runtime_setup_state(user_id)
        repair_choice = self._repair_option_choice_response(user_id, text)
        if repair_choice is not None:
            return repair_choice
        repair_followup = self._repair_context_handoff_response(user_id, text)
        if repair_followup is not None:
            return repair_followup
        decision = classify_runtime_chat_route(
            text,
            awaiting_secret=str(state.get("step") or "") == "awaiting_openrouter_key",
            awaiting_confirmation=str(state.get("step") or "") in {"awaiting_switch_confirm", "awaiting_openrouter_reuse_confirm"},
        )
        route = str(decision.get("route") or "generic_chat").strip().lower() or "generic_chat"
        kind = str(decision.get("kind") or "none").strip().lower()
        if kind == "product_specific_guard":
            adapter = self._chat_runtime_adapter
            if bool(
                callable(getattr(adapter, "should_use_assistant_frontdoor", None))
                and adapter.should_use_assistant_frontdoor(
                    text=text,
                    route_decision=decision,
                    is_user_chat=True,
                )
            ):
                return None
            action_tool_response = self._handle_action_tool_intent(user_id, text)
            if action_tool_response is not None:
                return action_tool_response
            return self._runtime_state_unavailable_response(
                route="runtime_status",
                used_memory=bool(state),
                reason="product_specific_message_unclassified",
            )
        mode_safe_model_kinds = {
            "describe_current_model",
            "local_model_inventory",
            "model_lifecycle_status",
            "find_ollama_models",
            "recommend_local_model",
            "switch_better_local_model",
            "model_ready_now",
            "model_availability",
            "model_scout_strategy",
            "model_scout_discovery",
        }
        control_mode_response = (
            None if kind in mode_safe_model_kinds else self._control_mode_intent_response(text)
        )
        if control_mode_response is not None:
            return control_mode_response
        if route == "generic_chat" or kind in {"none", "generic_chat"}:
            return None
        if kind in {"operational_doctor", "operational_agent_status", "operational_observe"}:
            return self._operational_status_response(user_id, text, kind)
        if kind == "assistant_capabilities":
            return self._assistant_capabilities_response()
        if kind in {"agent_memory_inspect", "agent_memory_preferences", "agent_memory_open_loops"}:
            return self._agent_memory_response(user_id, kind, query_text=text)
        if self._runtime_truth() is None:
            return self._runtime_state_unavailable_response(
                route=route,
                used_memory=bool(state),
                reason="runtime_truth_service_unavailable",
            )
        if kind == "providers_status":
            return self._providers_status_response()
        if kind == "provider_status":
            return self._provider_status_response(str(decision.get("provider_id") or ""))
        if kind in {"runtime_status", "telegram_status"}:
            return self._runtime_status_response(kind)
        if kind == "model_controller_policy":
            return self._model_controller_policy_response()
        if kind == "model_policy_status":
            return self._model_policy_status_response()
        if kind == "model_policy_cap":
            return self._model_policy_cap_response()
        if kind == "model_policy_current_choice":
            return self._model_policy_current_choice_response()
        if kind == "model_policy_provider_explanation":
            return self._model_policy_provider_explanation_response(str(decision.get("provider_id") or "").strip() or None)
        if kind == "model_policy_switch_candidate":
            return self._model_policy_switch_candidate_response()
        if kind == "model_policy_tier_candidate":
            return self._model_policy_tier_candidate_response(str(decision.get("tier") or "").strip() or None)
        if kind == "governance_adapters":
            return self._governance_adapters_response()
        if kind == "governance_background_tasks":
            return self._governance_background_tasks_response()
        if kind == "governance_blocks":
            return self._governance_blocks_response()
        if kind == "governance_pending":
            return self._governance_pending_response()
        if kind == "governance_overview":
            return self._governance_overview_response()
        if kind == "governance_skill_status":
            return self._governance_skill_status_response(str(decision.get("skill_id") or "").strip() or None)
        if kind == "governance_execution_mode":
            return self._governance_execution_mode_response(str(decision.get("target_id") or "").strip() or None)
        if kind == "governance_adapter_detail":
            return self._governance_adapter_detail_response(str(decision.get("adapter_id") or "").strip() or None)
        if kind == "describe_current_model":
            return self._current_model_response()
        if kind == "filesystem_list_directory":
            return self._filesystem_list_directory_response(str(decision.get("path_hint") or "").strip() or None)
        if kind == "filesystem_stat_path":
            return self._filesystem_stat_path_response(str(decision.get("path_hint") or "").strip() or None)
        if kind == "filesystem_read_text_file":
            return self._filesystem_read_text_file_response(str(decision.get("path_hint") or "").strip() or None)
        if kind == "filesystem_search_filenames":
            return self._filesystem_search_filenames_response(
                root_hint=str(decision.get("path_hint") or "").strip() or None,
                query=str(decision.get("query") or "").strip() or None,
            )
        if kind == "filesystem_search_text":
            return self._filesystem_search_text_response(
                root_hint=str(decision.get("path_hint") or "").strip() or None,
                query=str(decision.get("query") or "").strip() or None,
            )
        if kind == "shell_safe_command":
            return self._shell_execute_safe_command_response(
                command_name=str(decision.get("command_name") or "").strip() or None,
                subject=str(decision.get("subject") or "").strip() or None,
                query=str(decision.get("query") or "").strip() or None,
                cwd=str(decision.get("cwd") or "").strip() or None,
            )
        if kind == "shell_install_package":
            return self._shell_install_package_response(
                user_id,
                manager=str(decision.get("manager") or "").strip() or None,
                package=str(decision.get("package") or "").strip() or None,
                scope=str(decision.get("scope") or "").strip() or None,
                dry_run=bool(decision.get("dry_run", False)),
            )
        if kind == "shell_create_directory":
            return self._shell_create_directory_response(user_id, str(decision.get("path_hint") or "").strip() or None)
        if kind == "shell_blocked_request":
            return self._shell_blocked_request_response(
                blocked_reason=str(decision.get("blocked_reason") or "").strip() or "unsupported_command",
                request_text=str(decision.get("request_text") or "").strip() or None,
            )
        if kind == "model_lifecycle_status":
            return self._model_lifecycle_response(text)
        if kind == "model_scout_strategy":
            return self._model_scout_strategy_response(user_id, text)
        if kind == "model_scout_discovery":
            return self._model_scout_discovery_response(text)
        if kind == "recommend_local_model":
            return self._model_scout_strategy_response(
                user_id,
                text,
                requested_role_override="best_local",
            )
        if kind == "model_switch_advisory":
            return self._model_switch_advisory_response()
        if kind == "model_ready_now":
            return self._model_ready_now_response()
        if kind == "model_availability":
            return self._model_inventory_response(
                local_only=False,
                remote_only=str(decision.get("inventory_scope") or "").strip().lower() == "remote",
            )
        if kind == "local_model_inventory":
            provider_hint = str(decision.get("provider_id") or "").strip().lower() or None
            return self._model_inventory_response(local_only=True, provider_id=provider_hint)
        if kind == "find_ollama_models":
            return self._find_ollama_models_response(text)
        if kind == "switch_better_local_model":
            return self._switch_better_local_model_response(user_id)
        if kind == "model_acquisition_request":
            return self._model_acquire_response(user_id, text)
        if kind == "set_default_model":
            return self._set_default_model_response(user_id, text, state)
        if kind == "cancel_pending_setup":
            return self._cancel_pending_setup_response(user_id, state)
        if kind == "confirm_pending_setup":
            return self._confirm_pending_setup_response(user_id, state)
        if kind == "setup_explanation":
            return self._setup_explanation_response(used_memory=bool(state))
        if kind == "configure_ollama":
            return self._configure_ollama_response(
                make_default=bool(decision.get("make_default")),
                used_memory=bool(state),
            )
        if kind in {"configure_openrouter", "provide_openrouter_key"}:
            return self._configure_openrouter_response(user_id, decision, state)
        return self._runtime_state_unavailable_response(
            route=route,
            used_memory=bool(state),
            reason="unhandled_runtime_truth_route",
        )

    def _tool_handler_brief(self, request: dict[str, Any], user_id: str) -> dict[str, Any]:
        _ = request
        response = self._handle_message_impl("/brief", user_id)
        return {"ok": True, "user_text": response.text, "data": response.data if isinstance(response.data, dict) else {}}

    def _tool_handler_status(self, request: dict[str, Any], user_id: str) -> dict[str, Any]:
        _ = request
        response = self._handle_message_impl("/status", user_id)
        return {"ok": True, "user_text": response.text, "data": response.data if isinstance(response.data, dict) else {}}

    def _tool_handler_health(self, request: dict[str, Any], user_id: str) -> dict[str, Any]:
        _ = request
        response = self._handle_message_impl("/health", user_id)
        return {"ok": True, "user_text": response.text, "data": response.data if isinstance(response.data, dict) else {}}

    def _tool_handler_doctor(self, request: dict[str, Any], user_id: str) -> dict[str, Any]:
        _ = request
        _ = user_id
        report = run_doctor_report(online=False, fix=False)
        pass_count = sum(1 for item in report.checks if item.status == "OK")
        warn_count = sum(1 for item in report.checks if item.status == "WARN")
        fail_count = sum(1 for item in report.checks if item.status == "FAIL")
        next_action = report.next_action or "none"
        text = (
            f"Doctor: {report.summary_status} (trace {report.trace_id})\n"
            f"PASS {pass_count} · WARN {warn_count} · FAIL {fail_count}\n"
            f"Next: {next_action}\n"
            "Run: python -m agent doctor --json for details."
        )
        return {"ok": True, "user_text": text, "data": {"trace_id": report.trace_id, "summary_status": report.summary_status}}

    def _tool_handler_observe_system_health(self, request: dict[str, Any], user_id: str) -> dict[str, Any]:
        _ = request
        _ = user_id
        report = build_system_health_report(collect_system_health())
        return {
            "ok": True,
            "user_text": render_system_health_summary(
                report.get("observed") if isinstance(report.get("observed"), dict) else {},
                report.get("analysis") if isinstance(report.get("analysis"), dict) else {},
            ),
            "data": {"system_health": report},
        }

    def _tool_handler_observe_now(self, request: dict[str, Any], user_id: str) -> dict[str, Any]:
        _ = request
        response = self._handle_message_impl("/observe_now", user_id)
        return {"ok": True, "user_text": response.text, "data": response.data if isinstance(response.data, dict) else {}}

    def _log_llm_selection(
        self,
        *,
        trace_id: str,
        provider: str | None,
        model: str | None,
        reason: str,
        fallback_used: bool,
        task_type: str | None = None,
        fallback_count: int | None = None,
    ) -> None:
        identity = get_effective_llm_identity(
            provider=provider,
            model=model,
            local_providers={"ollama"},
            reason=reason,
        )
        runtime_status = normalize_user_facing_status(
            ready=bool(identity.get("known", False) and not fallback_used),
            bootstrap_required=False,
            failure_code=(None if not fallback_used else "llm_runtime_error"),
            phase=None if not fallback_used else "degraded",
            provider=provider,
            model=model,
            local_providers={"ollama"},
        )
        try:
            log_event(
                self.log_path,
                "llm.selection",
                {
                    "trace_id": trace_id,
                    "surface": "orchestrator",
                    "route": "chat",
                    "runtime_mode": str(runtime_status.get("runtime_mode") or "DEGRADED"),
                    "selected_provider": str(provider or "").strip().lower() or None,
                    "selected_model": str(model or "").strip() or None,
                    "known": bool(identity.get("known", False)),
                    "task_type": str(task_type or "").strip().lower() or None,
                    "reason": str(reason or "").strip().lower() or "unknown",
                    "fallback_used": bool(fallback_used),
                    "fallback_count": int(fallback_count or 0),
                },
            )
        except Exception:
            return

    @staticmethod
    def _normalize_chat_messages(messages: Any, fallback_text: str) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for item in (messages if isinstance(messages, list) else []):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower() or "user"
            if role not in {"system", "user", "assistant"}:
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            normalized.append({"role": role, "content": content})
        if normalized:
            return normalized
        fallback = str(fallback_text or "").strip()
        if not fallback:
            return []
        return [{"role": "user", "content": fallback}]

    def _llm_chat(
        self,
        user_id: str,
        text: str,
        *,
        chat_context: dict[str, Any] | None = None,
    ) -> OrchestratorResponse:
        context = dict(chat_context) if isinstance(chat_context, dict) else {}
        trace_id = str(context.get("trace_id") or "").strip() or f"orch-{uuid.uuid4().hex[:10]}"
        source_surface = str(context.get("source_surface") or "orchestrator").strip().lower() or "orchestrator"
        purpose = str(context.get("purpose") or "chat").strip() or "chat"
        task_type_override = str(context.get("task_type") or purpose).strip() or "chat"
        payload = context.get("payload") if isinstance(context.get("payload"), dict) else {}
        normalized_messages = self._normalize_chat_messages(context.get("messages"), text)
        memory_context_text = str(context.get("memory_context_text") or "").strip()
        if not memory_context_text:
            memory_context_text = self._selective_chat_memory_context(user_id, text)
        used_memory = bool(memory_context_text) or len(normalized_messages) > 1
        runtime_adapter_used = bool(self._chat_runtime_adapter)
        # Final SAFE MODE/frontdoor containment guard. Deterministic assistant
        # handling should still win even if a grounded request drifted this far.
        containment_response = self._safe_mode_containment_response(user_id, text)
        if containment_response is not None:
            return containment_response
        external_pack_response = self._external_pack_knowledge_response(user_id, text)
        if external_pack_response is not None:
            return external_pack_response
        unmatched_response = self._assistant_unmatched_input_response(user_id, text)
        if unmatched_response is not None:
            return unmatched_response
        working_memory_payload = self._prepare_working_memory_for_chat(
            user_id=user_id,
            text=text,
            payload=dict(payload),
            thread_id=str(context.get("thread_id") or "").strip() or None,
            messages=normalized_messages,
            memory_context_text=memory_context_text,
        )
        normalized_messages = (
            list(working_memory_payload.get("messages"))
            if isinstance(working_memory_payload.get("messages"), list)
            else normalized_messages
        )
        memory_context_text = str(working_memory_payload.get("memory_context_text") or "").strip()
        used_memory = bool(memory_context_text) or len(normalized_messages) > 1 or bool(
            working_memory_payload.get("used_working_memory")
        )
        payload = dict(payload)
        payload["memory_context_text"] = memory_context_text
        prepared = None
        defaults: dict[str, Any] = {}
        if runtime_adapter_used and callable(
            getattr(self._chat_runtime_adapter, "prepare_orchestrator_chat_request", None)
        ):
            adapter_payload = self._chat_runtime_adapter.prepare_orchestrator_chat_request(
                {
                    "payload": dict(payload),
                    "messages": normalized_messages,
                    "request_started_epoch": int(context.get("request_started_epoch") or 0) or int(time.time()),
                    "trace_id": trace_id,
                    "source_surface": source_surface,
                }
            )
            if isinstance(adapter_payload, dict):
                prepared = adapter_payload.get("prepared")
                defaults = (
                    dict(adapter_payload.get("defaults"))
                    if isinstance(adapter_payload.get("defaults"), dict)
                    else {}
                )
        heuristic_command = self._heuristic_llm_command(text)
        if heuristic_command:
            try:
                tool_request = self._command_to_tool_request(heuristic_command, reason="heuristic_command")
                if tool_request is None:
                    raise RuntimeError("heuristic_tool_missing")
                response = self._execute_tool_request(
                    tool_request=tool_request,
                    user_id=user_id,
                    surface="llm",
                    runtime_mode="READY",
                )
                self._log_llm_selection(
                    trace_id=trace_id,
                    provider=None,
                    model=None,
                    reason="heuristic_command",
                    fallback_used=True,
                    task_type="tool_use",
                    fallback_count=0,
                )
                return self._merge_response_data(
                    response,
                    route="generic_chat",
                    used_runtime_state=runtime_adapter_used,
                    used_llm=False,
                    used_memory=used_memory,
                    used_tools=[str(tool_request.get("tool") or "").strip()],
                    ok=True,
                )
            except Exception:
                pass
        if not self._llm_chat_available():
            self._log_llm_selection(
                trace_id=trace_id,
                provider=None,
                model=None,
                reason="llm_unavailable",
                fallback_used=True,
                task_type="chat",
                fallback_count=0,
            )
            return self._merge_response_data(
                self._bootstrap_no_chat_response(),
                route="generic_chat",
                used_runtime_state=runtime_adapter_used,
                used_llm=False,
                used_memory=used_memory,
                used_tools=[],
                ok=True,
            )
        system_prompt = (
            "\n".join(self._assistant_identity_prompt_lines())
            + "\n"
            + "Be friendly, calm, competent, concise, and practical.\n"
            + "Avoid generic filler, canned support-bot language, and vendor/model self-descriptions.\n"
            + "Sound like one consistent assistant, not a shell wrapper or support bot.\n"
            + "Ask one clarifying question when needed.\n"
            + "Never claim you ran checks/actions unless they actually ran.\n"
            + "If system state is unknown, say so and offer to check.\n"
            + "IF YOU NEED SYSTEM FACTS, YOU MUST reply with ONLY ONE LINE and NOTHING ELSE:\n"
            + "[[RUN:/brief]] or [[RUN:/status]] or [[RUN:/health]]\n"
            + "Use at most one RUN directive.\n"
            + "DO NOT suggest slash commands.\n"
            + "DO NOT print '/brief /status /help' in normal chat."
        )
        if memory_context_text:
            system_prompt = (
                f"{system_prompt}\n"
                "Relevant remembered context (use only if it directly helps; never invent beyond it):\n"
                f"{memory_context_text}"
            )
        messages = normalized_messages or [{"role": "user", "content": str(text or "").strip()}]
        channel = self._chat_channel(context.get("channel") or source_surface)
        explicit_timeout_seconds = float(payload.get("timeout_seconds") or 0) or None
        request_id = str(context.get("request_id") or "").strip() or None

        def _route_inference_kwargs(*, latency_fallback: bool) -> dict[str, Any]:
            routed_messages = (
                list(getattr(prepared, "messages", []))
                if prepared is not None and isinstance(getattr(prepared, "messages", None), list)
                else [{"role": "system", "content": system_prompt}, *messages]
            )
            effective_timeout_seconds = explicit_timeout_seconds
            if channel == "telegram":
                budget = (
                    _TELEGRAM_LATENCY_FALLBACK_TIMEOUT_SECONDS
                    if latency_fallback
                    else _TELEGRAM_LATENCY_ROUTE_TIMEOUT_SECONDS
                )
                if effective_timeout_seconds is None:
                    effective_timeout_seconds = budget
                else:
                    effective_timeout_seconds = min(float(effective_timeout_seconds), budget)
            return {
                "llm_client": self.llm_client,
                "messages": routed_messages,
                "user_text": (
                    str(getattr(prepared, "last_user_text", "") or "").strip()
                    if prepared is not None
                    else str(text or "").strip()
                ),
                "task_hint": (
                    str(getattr(prepared, "last_user_text", "") or "").strip()
                    if prepared is not None
                    else str(text or "").strip()
                ),
                "purpose": purpose,
                "task_type": task_type_override,
                "trace_id": trace_id,
                "provider_override": (
                    str(getattr(prepared, "provider_override", "") or "").strip().lower() or None
                    if prepared is not None
                    else None
                ),
                "model_override": (
                    str(getattr(prepared, "model_override", "") or "").strip() or None
                    if prepared is not None
                    else None
                ),
                "require_tools": bool(getattr(prepared, "require_tools", False)) if prepared is not None else False,
                "require_json": bool(payload.get("require_json")),
                "require_vision": bool(payload.get("require_vision")),
                "min_context_tokens": int(payload.get("min_context_tokens") or 0) or None,
                "timeout_seconds": effective_timeout_seconds,
                "compute_tier": "low",
                "metadata": {
                    "trace_id": trace_id,
                    "source_surface": source_surface,
                    "channel": channel,
                    "latency_fallback": bool(latency_fallback),
                    "selection_reason": (
                        str(getattr(prepared, "selection_reason", "") or "").strip()
                        if prepared is not None
                        else None
                    ),
                },
            }
        try:
            if prepared is not None and isinstance(getattr(prepared, "direct_result", None), dict):
                router_result = dict(prepared.direct_result)
                router_data = {}
                used_llm = False
            else:
                router_result = route_inference(**_route_inference_kwargs(latency_fallback=False))
                if channel == "telegram" and self._router_error_kind(router_result) == "timeout":
                    slow_model = self._router_attempt_model(router_result)
                    initial_duration_ms = int(router_result.get("duration_ms") or 0)
                    self._record_runtime_event(
                        "telegram_latency_guard",
                        request_id=request_id,
                        trace_id=trace_id,
                        source=source_surface,
                        model_selected=slow_model,
                        duration_ms=initial_duration_ms,
                        fallback_used=True,
                    )
                    fallback_result = route_inference(**_route_inference_kwargs(latency_fallback=True))
                    fallback_model = self._router_attempt_model(fallback_result)
                    self._record_runtime_event(
                        "telegram_latency_fallback",
                        request_id=request_id,
                        trace_id=trace_id,
                        source=source_surface,
                        slow_model=slow_model,
                        fallback_model=fallback_model,
                        duration_ms=int(fallback_result.get("duration_ms") or initial_duration_ms),
                        fallback_used=True,
                        ok=bool(fallback_result.get("ok")),
                    )
                    router_result = fallback_result
                router_data = (
                    router_result.get("data")
                    if isinstance(router_result.get("data"), dict)
                    else {}
                )
                used_llm = True
        except Exception:
            self._log_llm_selection(
                trace_id=trace_id,
                provider=None,
                model=None,
                reason="llm_router_exception",
                fallback_used=True,
                task_type="chat",
                fallback_count=0,
            )
            return self._merge_response_data(
                self._llm_error_fallback_response(user_id, text),
                route="generic_chat",
                used_runtime_state=runtime_adapter_used,
                used_llm=False,
                used_memory=used_memory,
                used_tools=[],
                ok=False,
                error_kind="llm_router_exception",
            )
        task_request = router_data.get("task_request") if isinstance(router_data.get("task_request"), dict) else {}
        task_type = str(task_request.get("task_type") or router_result.get("task_type") or task_type_override)
        selection = router_data.get("selection") if isinstance(router_data.get("selection"), dict) else {}
        selection_reason = str(router_result.get("selection_reason") or selection.get("reason") or "router_default").strip() or "router_default"
        selection_fallbacks = [
            str(item).strip()
            for item in (selection.get("fallbacks") if isinstance(selection.get("fallbacks"), list) else [])
            if str(item).strip()
        ]
        selected_provider = str(router_result.get("provider") or "").strip() or None
        selected_model = str(router_result.get("model") or "").strip() or None
        selection_policy = (
            build_chat_selection_policy_meta(prepared=prepared, result=router_result, defaults=defaults)
            if prepared is not None and defaults
            else None
        )
        response_common: dict[str, Any] = {
            "route": "generic_chat",
            "used_runtime_state": runtime_adapter_used,
            "used_llm": bool(used_llm),
            "used_memory": used_memory,
            "used_tools": [],
            "ok": bool(router_result.get("ok")),
            "provider": selected_provider,
            "model": selected_model,
            "fallback_used": bool(router_result.get("fallback_used", False)),
            "attempts": router_result.get("attempts") or [],
            "duration_ms": int(router_result.get("duration_ms") or 0),
            "error_kind": str(router_result.get("error_kind") or router_result.get("error_class") or "").strip() or None,
        }
        if selection_policy is not None:
            response_common["selection_policy"] = selection_policy
        if not bool(router_result.get("ok")):
            self._log_llm_selection(
                trace_id=trace_id,
                provider=selected_provider,
                model=selected_model,
                reason=str(router_result.get("error_kind") or selection_reason or "llm_inference_failed"),
                fallback_used=True,
                task_type=task_type,
                fallback_count=len(selection_fallbacks),
            )
            llm_text = str(router_result.get("text") or "").strip()
            if llm_text:
                return self._merge_response_data(
                    OrchestratorResponse(
                    llm_text,
                    {
                        "llm_control": {
                            "trace_id": trace_id,
                            **router_data,
                        }
                    },
                    ),
                    **response_common,
                )
            return self._merge_response_data(
                self._llm_error_fallback_response(user_id, text),
                **response_common,
            )
        llm_text = str(router_result.get("text") or "").strip()
        if llm_text:
            llm_text = self._sanitize_vendor_identity_claim(
                llm_text,
                provider=selected_provider,
                model=selected_model,
            )
            tool_request = self._parse_llm_tool_request(llm_text)
            if tool_request is not None:
                response = self._execute_tool_request(
                    tool_request=tool_request,
                    user_id=user_id,
                    surface="llm",
                    runtime_mode="READY",
                )
                self._log_llm_selection(
                    trace_id=trace_id,
                    provider=selected_provider,
                    model=selected_model,
                    reason="tool_request",
                    fallback_used=True,
                    task_type=task_type,
                    fallback_count=len(selection_fallbacks),
                )
                return self._merge_response_data(
                    response,
                    **{
                        **response_common,
                        "ok": True,
                        "used_tools": [str(tool_request.get("tool") or "").strip()],
                    },
                )
            directive_command = self._parse_llm_run_directive(llm_text)
            if directive_command:
                try:
                    tool_request = self._command_to_tool_request(
                        directive_command,
                        reason="run_directive",
                    )
                    if tool_request is None:
                        raise RuntimeError("directive_tool_missing")
                    response = self._execute_tool_request(
                        tool_request=tool_request,
                        user_id=user_id,
                        surface="llm",
                        runtime_mode="READY",
                    )
                    self._log_llm_selection(
                        trace_id=trace_id,
                        provider=selected_provider,
                        model=selected_model,
                        reason="run_directive",
                        fallback_used=True,
                        task_type=task_type,
                        fallback_count=len(selection_fallbacks),
                    )
                    return self._merge_response_data(
                        response,
                        **{
                            **response_common,
                            "ok": True,
                            "used_tools": [str(tool_request.get("tool") or "").strip()],
                        },
                    )
                except Exception:
                    self._log_llm_selection(
                        trace_id=trace_id,
                        provider=None,
                        model=None,
                        reason="run_directive_failed",
                        fallback_used=True,
                        task_type=task_type,
                        fallback_count=len(selection_fallbacks),
                    )
                    return self._merge_response_data(
                        self._llm_error_fallback_response(user_id, text),
                        **{
                            **response_common,
                            "ok": False,
                            "error_kind": "run_directive_failed",
                        },
                    )
            self._log_llm_selection(
                trace_id=trace_id,
                provider=selected_provider,
                model=selected_model,
                reason=selection_reason if selected_model else "llm_chat",
                fallback_used=bool(router_result.get("fallback_used", False)),
                task_type=task_type,
                fallback_count=len(selection_fallbacks),
            )
            return self._merge_response_data(
                OrchestratorResponse(
                llm_text,
                {
                    "llm_chat": {
                        "trace_id": trace_id,
                        "route": "generic_chat",
                        "source_surface": source_surface,
                        "provider": selected_provider,
                        "model": selected_model,
                        "task_type": task_type,
                    }
                },
                ),
                **{
                    **response_common,
                    "ok": True,
                },
            )

        self._log_llm_selection(
            trace_id=trace_id,
            provider=None,
            model=None,
            reason="llm_empty_response",
            fallback_used=True,
            task_type=task_type,
            fallback_count=len(selection_fallbacks),
        )
        return self._merge_response_data(
            self._llm_error_fallback_response(user_id, text),
            **{
                **response_common,
                "ok": False,
                "error_kind": "llm_empty_response",
            },
        )

    @staticmethod
    def _vendor_claimed_in_text(text: str) -> str | None:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return None
        for vendor, patterns in _VENDOR_IDENTITY_PATTERNS:
            if any(pattern in lowered for pattern in patterns):
                return vendor
        return None

    def _sanitize_vendor_identity_claim(
        self,
        llm_text: str,
        *,
        provider: str | None,
        model: str | None,
    ) -> str:
        original_text = str(llm_text or "")
        original_claimed_vendor = self._vendor_claimed_in_text(original_text)
        original_identity_leaked = self._assistant_identity_leaked(original_text)
        sanitized = original_text
        for pattern, replacement in _ASSISTANT_IDENTITY_REWRITE_PATTERNS:
            sanitized = pattern.sub(replacement, sanitized, count=1)
        sanitized = re.sub(r"[ \t]{2,}", " ", sanitized)
        sanitized = re.sub(r"\s+([,.!?])", r"\1", sanitized).strip()
        claimed_vendor = self._vendor_claimed_in_text(sanitized)
        if (
            not original_claimed_vendor
            and not original_identity_leaked
            and not claimed_vendor
            and not self._assistant_identity_leaked(sanitized)
        ):
            return sanitized
        assistant_name, user_name = self._configured_identity_names()
        identity = get_public_identity(
            provider=provider,
            model=model,
            local_providers={"ollama"},
            assistant_name=assistant_name,
            user_name=user_name,
        )
        summary = str(identity.get("summary") or "").strip()
        return summary or f"I’m {assistant_identity_label(assistant_name=assistant_name)}."

    @staticmethod
    def _parse_llm_run_directive(llm_text: str) -> str | None:
        lines = [line.strip() for line in str(llm_text or "").strip().splitlines() if line.strip()][:2]
        for line in lines:
            match = _LLM_RUN_DIRECTIVE_RE.search(line)
            if not match:
                continue
            command = str(match.group(1) or "").strip().lower()
            if command not in _LLM_RUN_DIRECTIVE_ALLOWLIST:
                return None
            return command
        return None

    @staticmethod
    def _heuristic_llm_command(user_text: str) -> str | None:
        normalized = " ".join(str(user_text or "").lower().split()).strip()
        if not normalized:
            return None
        if normalized == "health":
            return "/health_system"
        if re.search(r"(how is my pc|check system|how is the computer running)", normalized):
            return "/health_system"
        if re.search(r"(system health|pc health|computer health)", normalized):
            return "/health_system"
        if re.search(r"(changed|what changed).*(pc|computer|system)", normalized):
            return "/brief"
        if re.search(r"(status|uptime|agent status|bot status)", normalized):
            return "/status"
        if re.search(
            r"(how is the bot health|bot health|health check|system health|health|unhealthy|running ok|stats)",
            normalized,
        ):
            return "/health"
        return None

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
    def _normalize_thread_label(label: str) -> str:
        cleaned = " ".join((label or "").replace("?", "").split()).strip()
        if len(cleaned) > 60:
            cleaned = cleaned[:60].rstrip()
        return cleaned

    @staticmethod
    def _normalize_graph_node_id(node_id: str) -> str:
        raw = (node_id or "").strip().lower()
        return "".join(ch for ch in raw if ch.isalnum() or ch == "_")

    @staticmethod
    def _normalize_graph_text(value: str, limit: int, lower: bool = False) -> str:
        text = " ".join((value or "").replace("?", "").split()).strip()
        if lower:
            text = text.lower()
        if len(text) > limit:
            text = text[:limit].rstrip()
        return text

    @staticmethod
    def _render_pretty_json(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=False)

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
        runtime_state = self._memory_runtime.get_thread_state(user_id)
        runtime_thread_id = str(runtime_state.get("thread_id") or "").strip()
        if runtime_thread_id:
            self._set_active_thread_id_for_user(user_id, runtime_thread_id)
            return runtime_thread_id
        return self._default_thread_id(user_id)

    def _set_active_thread_id_for_user(self, user_id: str, thread_id: str) -> None:
        normalized = (thread_id or "").strip()
        if not normalized:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        self._epistemic_thread_state[user_id] = {
            "active_thread_id": normalized,
            "thread_created_at": now_iso,
            "thread_label": None,
        }
        self._memory_runtime.set_thread_state(
            user_id,
            thread_id=normalized,
            updated_at=int(datetime.now(timezone.utc).timestamp()),
        )

    def _create_new_thread_id_for_user(self, user_id: str) -> str:
        base = self._default_thread_id(user_id)
        prefix = f"{base}:t"
        max_idx = 0

        for row in self.db.list_recent_threads(limit=1000):
            if not isinstance(row, dict):
                continue
            thread_id = str(row.get("thread_id") or "").strip()
            if not thread_id.startswith(prefix):
                continue
            suffix = thread_id[len(prefix):]
            if suffix.isdigit():
                max_idx = max(max_idx, int(suffix))

        for (known_user, thread_id) in self._epistemic_history.keys():
            if known_user != user_id:
                continue
            if not thread_id.startswith(prefix):
                continue
            suffix = thread_id[len(prefix):]
            if suffix.isdigit():
                max_idx = max(max_idx, int(suffix))

        state = self._epistemic_thread_state.get(user_id) or {}
        current_thread = str(state.get("active_thread_id") or "").strip()
        if current_thread.startswith(prefix):
            suffix = current_thread[len(prefix):]
            if suffix.isdigit():
                max_idx = max(max_idx, int(suffix))

        return f"{prefix}{max_idx + 1}"

    def _parse_thread_new_args(self, raw_args: str) -> tuple[str, dict[str, str], str]:
        args = raw_args or ""
        header_line, _, body = args.partition("\n")
        header_line = header_line.strip()
        body_text = body if body else ""

        try:
            tokens = shlex.split(header_line)
        except ValueError:
            tokens = header_line.split()

        label_tokens: list[str] = []
        pref_flags: dict[str, str] = {}
        flag_map = {
            "terse": "terse_mode",
            "summary": "show_summary",
            "next": "show_next_action",
            "codeblock": "commands_in_codeblock",
        }
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if token.startswith("--"):
                flag = token[2:].strip().lower()
                next_token = tokens[idx + 1].strip().lower() if idx + 1 < len(tokens) else None
                if flag in flag_map and next_token in {"on", "off"}:
                    pref_flags[flag_map[flag]] = next_token
                    idx += 2
                    continue
                if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
                    idx += 2
                else:
                    idx += 1
                continue
            label_tokens.append(token)
            idx += 1

        raw_label = " ".join(label_tokens).strip()
        normalized_label = self._normalize_thread_label(raw_label) if raw_label else ""
        final_label = normalized_label if normalized_label else "Untitled"
        return final_label, pref_flags, body_text

    def _formatting_prefs(self, thread_id: str | None) -> dict[str, bool]:
        project_mode = bool(get_project_mode(self.db, thread_id))
        prefs = {
            "show_next_action": bool(get_pref_effective(self.db, thread_id, "show_next_action", True)),
            "show_summary": bool(get_pref_effective(self.db, thread_id, "show_summary", True)),
            "terse_mode": bool(get_pref_effective(self.db, thread_id, "terse_mode", False)),
            "commands_in_codeblock": bool(get_pref_effective(self.db, thread_id, "commands_in_codeblock", False)),
            "project_mode": project_mode,
        }
        if project_mode:
            prefs["show_summary"] = False
            prefs["show_next_action"] = True
            prefs["terse_mode"] = False
            prefs["commands_in_codeblock"] = True
        return prefs

    def _context(self) -> dict[str, Any]:
        ctx = {"db": self.db, "timezone": self.timezone, "log_path": self.log_path}
        if self._semantic_memory_service is not None:
            ctx["semantic_memory"] = self._semantic_memory_service
        if self._runner:
            ctx["runner"] = self._runner
        if self.llm_client:
            ctx["route_inference"] = partial(route_inference, llm_client=self.llm_client)
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
        expires_ts = int((now_dt + timedelta(minutes=minutes)).timestamp())
        self._memory_runtime.add_pending_item(
            user_id,
            {
                "pending_id": pending_id,
                "kind": "clarification",
                "origin_tool": intent_type,
                "question": question,
                "options": list(options),
                "created_at": int(now_dt.timestamp()),
                "expires_at": expires_ts,
                "thread_id": self._active_thread_id_for_user(user_id),
                "status": PENDING_STATUS_WAITING_FOR_USER,
                "context": {"chat_id": str(chat_id)},
            },
        )

    def _intent_context(self, chat_id: str | None = None) -> dict[str, Any]:
        context = dict(self._context())
        if chat_id:
            context["chat_id"] = chat_id
        context["knowledge_cache"] = self._knowledge_cache
        return context

    @staticmethod
    def _perception_event_payloads(events: list[Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for event in events:
            rows.append(
                {
                    "kind": str(getattr(event, "kind", "") or ""),
                    "severity": str(getattr(event, "severity", "") or ""),
                    "summary": str(getattr(event, "summary", "") or ""),
                    "evidence_json": dict(getattr(event, "evidence_json", {}) or {}),
                }
            )
        return rows

    def _collect_perception_snapshot(self) -> dict[str, Any]:
        return collect_snapshot(list(self.perception_roots))

    def _store_perception_snapshot_and_events(self, snapshot: dict[str, Any]) -> tuple[int, list[int], list[dict[str, Any]]]:
        events = analyze_snapshot(snapshot)
        snapshot_id = self.db.insert_metrics_snapshot(snapshot)
        event_ids: list[int] = []
        for event in events:
            event_id = self.db.insert_event(
                int(snapshot.get("ts") or datetime.now(timezone.utc).timestamp()),
                event.kind,
                event.severity,
                event.summary,
                event.evidence_json,
            )
            event_ids.append(event_id)
        return snapshot_id, event_ids, self._perception_event_payloads(events)

    @staticmethod
    def _metrics_row_payload(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if not row:
            return None
        return {
            "id": int(row.get("id") or 0),
            "ts": int(row.get("ts") or 0),
            "cpu_usage": float(row.get("cpu_usage") or 0.0),
            "cpu_freq": float(row.get("cpu_freq") or 0.0),
            "mem_used": int(row.get("mem_used") or 0),
            "mem_available": int(row.get("mem_available") or 0),
            "root_disk_used_pct": float(row.get("root_disk_used_pct") or 0.0),
            "gpu_usage": float(row.get("gpu_usage")) if row.get("gpu_usage") is not None else None,
            "gpu_mem_used": int(row.get("gpu_mem_used")) if row.get("gpu_mem_used") is not None else None,
            "gpu_temp": float(row.get("gpu_temp")) if row.get("gpu_temp") is not None else None,
        }

    @staticmethod
    def _json_payload_from_response(response: OrchestratorResponse, tool_name: str) -> dict[str, Any]:
        raw = (response.text or "").strip()
        if not raw.startswith("{"):
            raise RuntimeError(f"{tool_name} returned non-JSON output")
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{tool_name} returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"{tool_name} returned non-object JSON")
        return parsed

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _collect_authoritative_observations(self, domains: set[str]) -> dict[str, Any]:
        observations: dict[str, Any] = {}
        refs: dict[str, dict[str, Any]] = {}

        for domain in sorted(domains):
            tool_name = _AUTHORITATIVE_DOMAIN_TO_TOOL.get(domain, "")
            if domain == "system.performance":
                payload = self._json_payload_from_response(self._sys_metrics_snapshot(), tool_name)
                observations[domain] = payload
                stored = payload.get("stored") if isinstance(payload.get("stored"), dict) else {}
                snapshot = payload.get("snapshot") if isinstance(payload.get("snapshot"), dict) else {}
                refs[domain] = {
                    "tool": tool_name,
                    "snapshot_id": self._coerce_int(stored.get("snapshot_id")),
                    "ts": self._coerce_int(snapshot.get("ts")),
                }
                continue

            if domain == "system.health":
                payload = self._json_payload_from_response(self._sys_health_report(), tool_name)
                observations[domain] = payload
                latest = payload.get("latest_metrics") if isinstance(payload.get("latest_metrics"), dict) else {}
                refs[domain] = {
                    "tool": tool_name,
                    "snapshot_id": self._coerce_int(latest.get("id")),
                    "ts": self._coerce_int(latest.get("ts")),
                }
                continue

            if domain == "system.storage":
                payload = self._json_payload_from_response(self._sys_inventory_summary(), tool_name)
                observations[domain] = payload
                latest = self.db.get_latest_metrics_snapshot() or {}
                refs[domain] = {
                    "tool": tool_name,
                    "snapshot_id": self._coerce_int(latest.get("id")),
                    "ts": self._coerce_int(latest.get("ts")),
                }
                continue

        return {
            "domains": sorted(domains),
            "grounding": {
                "collected_at_ts": int(datetime.now(timezone.utc).timestamp()),
                "observation_refs": refs,
            },
            "observations": observations,
        }

    @staticmethod
    def _authoritative_summary_lines(local_observations: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        observations = (
            local_observations.get("observations")
            if isinstance(local_observations.get("observations"), dict)
            else {}
        )

        perf = observations.get("system.performance") if isinstance(observations.get("system.performance"), dict) else {}
        perf_snapshot = perf.get("snapshot") if isinstance(perf.get("snapshot"), dict) else {}
        perf_cpu = perf_snapshot.get("cpu") if isinstance(perf_snapshot.get("cpu"), dict) else {}
        perf_gpu = perf_snapshot.get("gpu") if isinstance(perf_snapshot.get("gpu"), dict) else {}
        if perf_snapshot:
            try:
                cpu_usage = float(perf_cpu.get("usage_pct") or 0.0)
            except (TypeError, ValueError):
                cpu_usage = 0.0
            try:
                gpu_usage = float(perf_gpu.get("usage_pct") or 0.0)
            except (TypeError, ValueError):
                gpu_usage = 0.0
            try:
                gpu_temp = float(perf_gpu.get("temperature_c") or 0.0)
            except (TypeError, ValueError):
                gpu_temp = 0.0
            lines.append(
                "Performance: cpu={cpu}% gpu={gpu}% gpu_temp={temp}C".format(
                    cpu=cpu_usage,
                    gpu=gpu_usage,
                    temp=gpu_temp,
                )
            )

        health = observations.get("system.health") if isinstance(observations.get("system.health"), dict) else {}
        if health:
            latest = health.get("latest_metrics") if isinstance(health.get("latest_metrics"), dict) else {}
            events = health.get("recent_events") if isinstance(health.get("recent_events"), list) else []
            lines.append(
                "Health: latest_snapshot_id={sid} recent_events={events}".format(
                    sid=int(latest.get("id") or 0),
                    events=len(events),
                )
            )

        storage = observations.get("system.storage") if isinstance(observations.get("system.storage"), dict) else {}
        inventory = storage.get("inventory") if isinstance(storage.get("inventory"), dict) else {}
        if inventory:
            try:
                root_used_pct = float(inventory.get("root_disk_used_pct") or 0.0)
            except (TypeError, ValueError):
                root_used_pct = 0.0
            lines.append(
                "Storage: root_used_pct={used}% top_dirs={count}".format(
                    used=root_used_pct,
                    count=len(inventory.get("top_dirs") or []),
                )
            )

        if not lines:
            lines.append("Collected local observations for authoritative domain query.")
        return lines

    def _answer_with_authoritative_observations(self, user_text: str, local_observations: dict[str, Any]) -> OrchestratorResponse:
        observations_json = json.dumps(local_observations, ensure_ascii=True, sort_keys=True)
        llm_text: str | None = None

        if self.llm_client and hasattr(self.llm_client, "chat"):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Answer using only LOCAL_OBSERVATIONS. "
                        "If evidence is missing, say \"I’m not sure.\" and ask one focused question."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question:\n{user_text}\n\nLOCAL_OBSERVATIONS\n{observations_json}",
                },
            ]
            try:
                result = route_inference(
                    llm_client=self.llm_client,
                    messages=messages,
                    user_text=user_text,
                    task_hint=user_text,
                    purpose="authoritative_domain",
                    task_type="chat",
                    compute_tier="low",
                    require_tools=True,
                    trace_id=self._trace_id("llm"),
                )
            except Exception:
                result = None
            if isinstance(result, dict) and result.get("ok") and str(result.get("text") or "").strip():
                llm_text = str(result.get("text") or "").strip()

        answer_body = llm_text or "\n".join(self._authoritative_summary_lines(local_observations))
        final_text = f"{answer_body}\n\nLOCAL_OBSERVATIONS\n{observations_json}"
        return OrchestratorResponse(
            final_text,
            {
                "skip_friction_formatting": True,
                "local_observations": local_observations,
            },
        )

    def _authoritative_tool_failure_response(self, domains: set[str], error: Exception) -> OrchestratorResponse:
        primary = sorted(domains)[0] if domains else "system.performance"
        tool_name = _AUTHORITATIVE_DOMAIN_TO_TOOL.get(primary, "sys_metrics_snapshot")
        reason = " ".join(str(error).replace("?", "").split()) or "unknown error"
        text = (
            "I’m not sure.\n\n"
            f"I couldn’t read local system data via {tool_name} ({reason}). "
            f"Do you want me to retry {tool_name} now?"
        )
        return OrchestratorResponse(text, {"skip_friction_formatting": True})

    def _enforce_authoritative_domain_gate(self, user_text: str) -> OrchestratorResponse | None:
        domains = classify_authoritative_domain(user_text)
        if not domains:
            return None
        if has_local_observations_block(user_text):
            return None
        try:
            local_observations = self._collect_authoritative_observations(domains)
        except Exception as exc:  # pragma: no cover - defensive safety
            return self._authoritative_tool_failure_response(domains, exc)
        return self._answer_with_authoritative_observations(user_text, local_observations)

    def _sys_metrics_snapshot(self) -> OrchestratorResponse:
        if not self.perception_enabled:
            return OrchestratorResponse("Perception is disabled.")
        snapshot = self._collect_perception_snapshot()
        snapshot_id, event_ids, events = self._store_perception_snapshot_and_events(snapshot)
        payload = {
            "ok": True,
            "source": "fresh",
            "snapshot": snapshot,
            "events": events,
            "stored": {
                "snapshot_id": snapshot_id,
                "event_ids": event_ids,
            },
        }
        return OrchestratorResponse(self._render_pretty_json(payload))

    def _sys_health_report(self) -> OrchestratorResponse:
        if not self.perception_enabled:
            return OrchestratorResponse("Perception is disabled.")
        latest = self.db.get_latest_metrics_snapshot()
        source = "sqlite" if latest else "fresh"
        snapshot: dict[str, Any] | None = None
        if not latest:
            snapshot = self._collect_perception_snapshot()
            self._store_perception_snapshot_and_events(snapshot)
            latest = self.db.get_latest_metrics_snapshot()
        payload = {
            "ok": True,
            "source": source,
            "latest_metrics": self._metrics_row_payload(latest),
            "recent_events": self.db.list_recent_events(limit=10),
            "system_health": ((snapshot or {}).get("system_health") if snapshot else None),
        }
        return OrchestratorResponse(self._render_pretty_json(payload))

    def _sys_inventory_summary(self) -> OrchestratorResponse:
        if not self.perception_enabled:
            return OrchestratorResponse("Perception is disabled.")
        latest = self.db.get_latest_metrics_snapshot()
        source = "sqlite" if latest else "fresh"
        if latest:
            snapshot = {
                "cpu": {"freq_mhz": latest.get("cpu_freq") or 0.0, "load_avg": {"1m": 0.0}},
                "memory": {
                    "total": 0,
                    "used": int(latest.get("mem_used") or 0),
                    "available": int(latest.get("mem_available") or 0),
                    "swap_total": 0,
                },
                "disk": {
                    "root": {
                        "total": 0,
                        "used_pct": float(latest.get("root_disk_used_pct") or 0.0),
                    },
                    "top_dirs": [],
                },
                "gpu": {
                    "available": latest.get("gpu_usage") is not None,
                },
            }
        else:
            snapshot = self._collect_perception_snapshot()
            self._store_perception_snapshot_and_events(snapshot)

        summary = summarize_inventory(snapshot, list(self.perception_roots))
        payload = {"ok": True, "source": source, "inventory": summary}
        return OrchestratorResponse(self._render_pretty_json(payload))

    def _maybe_add_narration(self, kind: str, payload: dict[str, Any], text: str) -> str:
        if not self._narration_enabled():
            return text
        if not self.llm_client or not hasattr(self.llm_client, "chat"):
            return text
        result = route_inference(
            llm_client=self.llm_client,
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the payload in 2-4 concise bullet lines with no recommendations.",
                },
                {"role": "user", "content": json.dumps({"kind": kind, "payload": payload}, ensure_ascii=True)},
            ],
            task_hint=kind,
            purpose="narration",
            task_type="chat",
            compute_tier="low",
            trace_id=self._trace_id("llm"),
        )
        if not result.get("ok") or not result.get("text"):
            return text
        provider = result.get("provider") or "unknown"
        scope = "local" if provider == "ollama" else "cloud" if provider == "openai" else provider
        header = f"Narration ({scope})"
        return f"{header}\n{result.get('text')}\n\n{text}"

    def _maybe_add_narration_from_text(self, kind: str, text: str) -> str:
        if not self._narration_enabled():
            return text
        if not self.llm_client or not hasattr(self.llm_client, "chat"):
            return text
        result = route_inference(
            llm_client=self.llm_client,
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the report in 2-4 concise bullet lines with no recommendations.",
                },
                {"role": "user", "content": json.dumps({"kind": kind, "report_text": text}, ensure_ascii=True)},
            ],
            task_hint=kind,
            purpose="narration",
            task_type="chat",
            compute_tier="low",
            trace_id=self._trace_id("llm"),
        )
        if not result.get("ok") or not result.get("text"):
            return text
        provider = result.get("provider") or "unknown"
        scope = "local" if provider == "ollama" else "cloud" if provider == "openai" else provider
        header = f"Narration ({scope})"
        return f"{header}\n{result.get('text')}\n\n{text}"

    @staticmethod
    def _narration_enabled() -> bool:
        narration_flag = os.getenv("ENABLE_NARRATION", "").strip().lower()
        legacy_flag = os.getenv("LLM_NARRATION_ENABLED", "").strip().lower()
        return (narration_flag or legacy_flag) in {"1", "true", "yes", "y", "on"}

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
        record_event(self.db, user_id, topic, intent_type)
        self._memory_runtime.set_current_topic(
            user_id,
            topic=topic,
            last_tool=intent_type,
        )

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
        now_epoch = int(datetime.now(timezone.utc).timestamp())
        self._memory_runtime.add_pending_item(
            user_id,
            {
                "kind": "followup",
                "origin_tool": "compare_now",
                "question": "Run compare-now on the latest what-if scenario?",
                "options": ["yes", "no"],
                "created_at": now_epoch,
                "expires_at": now_epoch + 600,
                "thread_id": self._active_thread_id_for_user(user_id),
                "status": PENDING_STATUS_READY_TO_RESUME,
                "context": {"what_if_text": what_if_text},
            },
        )

    def _get_pending_compare(self, user_id: str) -> dict[str, Any] | None:
        row = self._pending_compare.get(user_id)
        if not row:
            return None
        try:
            if row.get("expires_at") and datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc):
                self._pending_compare.pop(user_id, None)
                pending_items = self._memory_runtime.list_pending_items(user_id, include_expired=True)
                for pending in pending_items:
                    if str(pending.get("origin_tool") or "") != "compare_now":
                        continue
                    self._memory_runtime.set_pending_status(
                        user_id,
                        str(pending.get("pending_id") or ""),
                        PENDING_STATUS_EXPIRED,
                    )
                return None
        except (TypeError, ValueError):
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
        if not bool(data.get("skip_runtime_thread_persist", False)):
            self._memory_runtime.set_thread_state(
                user_id,
                thread_id=active_thread_id,
                status="active",
                updated_at=int(datetime.now(timezone.utc).timestamp()),
            )
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
        except (sqlite3.Error, OSError, TypeError, ValueError):
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
            plan_steps = (
                compute_plan(
                    user_text,
                    candidate,
                    body_text,
                    min_imperative_sentences=1 if prefs["project_mode"] else 2,
                    min_steps=1 if prefs["project_mode"] else 2,
                )
                if show_plan
                else None
            )
            if prefs["terse_mode"] and plan_steps:
                show_options = False
            next_action = compute_next_action(user_text, ctx, candidate) if show_next else None
            options = (
                compute_options(
                    user_text,
                    candidate,
                    body_text,
                    project_mode=prefs["project_mode"],
                )
                if show_options
                else None
            )
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
        except (sqlite3.Error, OSError, TypeError, ValueError):
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
            blocked = self._blocked_skill_governance.get(skill_name)
            if isinstance(blocked, dict):
                return self._skill_governance_denied_response(skill_name, blocked)
            return OrchestratorResponse("Skill not found.")

        func = skill.functions.get(function_name)
        if not func:
            return OrchestratorResponse("Function not found.")

        governance = self._skill_governance_decisions.get(skill_name)
        if governance is None:
            governance = self._skill_governance_store.get_skill_governance(skill_name)
        if isinstance(governance, dict):
            if not bool(governance.get("allowed", False)):
                log_event(
                    self.log_path,
                    "skill_governance_denied",
                    {
                        "skill": skill_name,
                        "function": function_name,
                        "reason": str(governance.get("reason") or "execution_governance_denied"),
                    },
                )
                return self._skill_governance_denied_response(skill_name, governance)
            effective_mode = str(governance.get("requested_execution_mode") or DEFAULT_EXECUTION_MODE).strip().lower() or DEFAULT_EXECUTION_MODE
            if effective_mode != DEFAULT_EXECUTION_MODE:
                return OrchestratorResponse(
                    f"Skill {skill_name} is governed as {effective_mode} and must be run by the main runtime, not directly from chat.",
                    {"skill_governance": governance},
                )

        pack_id = str(skill.pack_id or skill_name).strip() or skill_name
        iface = f"{skill_name}.{function_name}"
        try:
            enforce_iface_allowed(
                pack_id=pack_id,
                iface=iface,
                fallback_iface=function_name,
                pack_record=self._pack_store.get_pack(pack_id),
                trust=str(skill.pack_trust or "native").strip().lower() or "native",
                expected_permissions_hash=str(skill.pack_permissions_hash or "").strip(),
            )
        except PackPermissionDenied as exc:
            log_event(
                self.log_path,
                "pack_permission_denied",
                {
                    "pack_id": pack_id,
                    "skill": skill_name,
                    "function": function_name,
                    "reason": exc.reason,
                },
            )
            return OrchestratorResponse(
                f"This skill pack is not allowed to call {function_name}. "
                f"Approve pack {pack_id} for {function_name}?"
            )

        action = {"action_type": action_type or ""}
        decision = evaluate_policy(skill.permissions, requested_permissions, action)
        if not decision.allowed:
            return OrchestratorResponse("Permission denied.")

        if decision.requires_confirmation and not confirmed:
            now_epoch = int(datetime.now(timezone.utc).timestamp())
            pending_id = f"confirm-{uuid.uuid4().hex[:10]}"
            pending = PendingAction(
                user_id=user_id,
                action={
                    "pending_id": pending_id,
                    "skill": skill_name,
                    "function": function_name,
                    "args": args,
                    "requested_permissions": requested_permissions,
                    "action_type": action_type,
                },
                message="This will delete or overwrite data. Reply /confirm to proceed.",
            )
            self.confirmations.set(pending)
            self._memory_runtime.add_pending_item(
                user_id,
                {
                    "pending_id": pending_id,
                    "kind": "confirmation",
                    "origin_tool": function_name or skill_name,
                    "question": pending.message,
                    "options": ["/confirm", "cancel"],
                    "created_at": now_epoch,
                    "expires_at": now_epoch + 600,
                    "thread_id": self._active_thread_id_for_user(user_id),
                    "status": PENDING_STATUS_WAITING_FOR_USER,
                },
            )
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
            except (sqlite3.Error, OSError, TypeError, ValueError):
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
            except (TypeError, ValueError, sqlite3.Error, OSError):
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

    def handle_message(
        self,
        text: str,
        user_id: str,
        *,
        chat_context: dict[str, Any] | None = None,
    ) -> OrchestratorResponse:
        response = self._handle_message_impl(text, user_id, chat_context=chat_context)
        response_data = response.data if isinstance(response.data, dict) else {}
        if bool(response_data.get("skip_epistemic_gate", False)):
            final_response = response
        else:
            final_response = self._apply_epistemic_layer(user_id, text, response)
        final_response = self._apply_assistant_response_guard(
            user_id=user_id,
            user_text=text,
            response=final_response,
        )
        self._remember_interpretable_result(
            user_id=user_id,
            user_text=text,
            response=final_response,
        )
        try:
            self._record_chat_working_memory_turn(
                user_id=user_id,
                role="assistant",
                text=final_response.text,
                chat_context=chat_context,
            )
        except Exception:
            pass
        try:
            self._memory_runtime.record_agent_action(
                user_id,
                final_response.text,
                action_kind=self._memory_action_kind_for_response(text=text, response=final_response),
            )
        except Exception:
            pass
        return final_response

    @staticmethod
    def _memory_action_kind_for_response(*, text: str, response: OrchestratorResponse) -> str | None:
        command = parse_command(str(text or "").strip())
        if command is not None and str(command.name or "").strip():
            return str(command.name).strip().lower()
        response_data = response.data if isinstance(response.data, dict) else {}
        if isinstance(response_data.get("memory_snapshot"), dict):
            return "memory_summary"
        normalized = " ".join(str(text or "").strip().lower().split())
        if normalized in {"memory", "resume", "what are we doing?", "where were we?"}:
            return "memory_summary"
        return None

    def _handle_message_impl(
        self,
        text: str,
        user_id: str,
        *,
        chat_context: dict[str, Any] | None = None,
    ) -> OrchestratorResponse:
        self._runner = Runner()
        try:
            context = dict(chat_context) if isinstance(chat_context, dict) else {}
            requested_thread_id = str(context.get("thread_id") or "").strip()
            if requested_thread_id:
                self._set_active_thread_id_for_user(user_id, requested_thread_id)
            override, cleaned_text = memory_ingest.parse_memory_override(text)
            cmd = parse_command(text)
            continuity_health_before = self._memory_runtime.inspect_user_state(user_id)
            if cmd and cmd.name == "nomem":
                cmd = None
                text = cleaned_text
            effective_user_text = cleaned_text if override else text
            self._memory_runtime.clear_expired_pending_items(user_id)
            if str(effective_user_text or "").strip() and not str(effective_user_text).strip().startswith("/"):
                self._memory_runtime.record_user_request(user_id, str(effective_user_text))
                try:
                    self._record_chat_working_memory_turn(
                        user_id=user_id,
                        role="user",
                        text=str(effective_user_text),
                        chat_context=context,
                    )
                except Exception:
                    pass
            skip_memory_thread_repair = bool(
                cmd is not None
                and cmd.name == "memory"
                and not bool(continuity_health_before.get("healthy", True))
            )
            if not skip_memory_thread_repair:
                self._memory_runtime.set_thread_state(
                    user_id,
                    runtime_mode=("READY" if self._llm_chat_available() else "BOOTSTRAP_REQUIRED"),
                )
            if not cmd:
                tool_command = self._heuristic_llm_command(effective_user_text)
                if tool_command == "/health_system":
                    tool_request = self._command_to_tool_request(tool_command, reason="deterministic_system_health")
                    if tool_request is not None:
                        return self._execute_tool_request(
                            tool_request=tool_request,
                            user_id=user_id,
                            surface="orchestrator",
                            runtime_mode=self._tool_runtime_mode(),
                        )
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
                except (TypeError, ValueError, sqlite3.Error, OSError):
                    pass
                skip_memory_tool_repair = bool(
                    cmd.name == "memory"
                    and not bool(continuity_health_before.get("healthy", True))
                )
                if not skip_memory_tool_repair:
                    self._memory_runtime.set_last_tool(user_id, cmd.name)
                if cmd.name == "confirm":
                    pending = self.confirmations.pop(user_id)
                    if not pending:
                        return OrchestratorResponse("No pending action to confirm.")
                    action = pending.action
                    pending_id = str(action.get("pending_id") or "").strip()
                    if pending_id:
                        self._memory_runtime.set_pending_status(user_id, pending_id, PENDING_STATUS_DONE)
                    if str(action.get("kind") or "").strip().lower() == "native_mutation":
                        return self._execute_confirmed_native_mutation(user_id, action)
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

                if cmd.name in {"anchor", "checkpoint"}:
                    thread_id = self._active_thread_id_for_user(user_id)
                    parsed = parse_anchor_input(cmd.args)
                    if parsed is None:
                        return OrchestratorResponse(
                            "Usage: /anchor <title>",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    title, bullets, open_line = parsed
                    anchor_id = create_anchor(self.db, thread_id, title, bullets, open_line)
                    return OrchestratorResponse(
                        f"Saved checkpoint {anchor_id}.",
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "anchors":
                    thread_id = self._active_thread_id_for_user(user_id)
                    anchors = list_anchors(self.db, thread_id, limit=5)
                    lines = [f"Anchors (thread {thread_id}):"]
                    if not anchors:
                        lines.append("(none)")
                    else:
                        latest = anchors[0]
                        focus_title = (latest.title or "").replace("?", "").strip() or "Checkpoint"
                        lines.append(f"Current focus: {focus_title}")
                        latest_open = (latest.open_line or "").replace("?", "").strip()
                        if latest_open:
                            if latest_open.lower().startswith("open:"):
                                latest_open = latest_open[5:].strip()
                            if latest_open:
                                lines.append(f"Next: {latest_open}")
                        lines.append("---")
                        for anchor in anchors:
                            title = (anchor.title or "").replace("?", "").strip()
                            created_at = (anchor.created_at or "").replace("?", "").strip()
                            lines.append(f"#{anchor.id} {created_at} — {title}")
                            for bullet in anchor.bullets:
                                bullet_text = (bullet or "").replace("?", "").strip()
                                if bullet_text:
                                    lines.append(f"  - {bullet_text}")
                            if anchor.open_line:
                                open_line = (anchor.open_line or "").replace("?", "").strip()
                                if open_line:
                                    lines.append(f"  {open_line}")
                    return OrchestratorResponse(
                        "\n".join(lines),
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "threads":
                    thread_id = self._active_thread_id_for_user(user_id)
                    rows = self.db.list_recent_threads(limit=10)
                    lines = ["Threads:"]
                    if not rows:
                        lines.append("(none)")
                    else:
                        thread_ids = [
                            str(row.get("thread_id") or "").strip()
                            for row in rows
                            if isinstance(row, dict) and str(row.get("thread_id") or "").strip()
                        ]
                        labels_by_thread = self.db.list_thread_labels(thread_ids)
                        for idx, row in enumerate(rows, start=1):
                            listed_thread_id = str(row.get("thread_id") or "").replace("?", "").strip()
                            last_ts = str(row.get("last_ts") or "").replace("?", "").strip()
                            if not listed_thread_id or not last_ts:
                                continue
                            label = labels_by_thread.get(listed_thread_id)
                            label_text = self._normalize_thread_label(str(label or "(none)")) or "(none)"
                            focus = self.db.get_latest_anchor_title(listed_thread_id)
                            focus_text = str(focus or "(none)").replace("?", "").strip() or "(none)"
                            lines.append(
                                f"{idx}) {listed_thread_id}  {last_ts}  Label: {label_text}  Focus: {focus_text}"
                            )
                    return OrchestratorResponse(
                        "\n".join(lines),
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "thread_use":
                    current_thread_id = self._active_thread_id_for_user(user_id)
                    target_thread_id = (cmd.args or "").replace("?", "").strip()
                    if not target_thread_id:
                        return OrchestratorResponse(
                            "Usage: /thread_use <thread_id>",
                            {"skip_friction_formatting": True, "thread_id": current_thread_id},
                        )
                    rows = self.db.list_recent_threads(limit=200)
                    recent_ids = {
                        str(row.get("thread_id") or "").strip()
                        for row in rows
                        if isinstance(row, dict) and str(row.get("thread_id") or "").strip()
                    }
                    has_anchor = self.db.get_latest_anchor_title(target_thread_id) is not None
                    if target_thread_id not in recent_ids and not has_anchor:
                        return OrchestratorResponse(
                            f"Unknown thread: {target_thread_id}.",
                            {"skip_friction_formatting": True, "thread_id": current_thread_id},
                        )
                    self._set_active_thread_id_for_user(user_id, target_thread_id)
                    return OrchestratorResponse(
                        f"Active thread set to {target_thread_id}.",
                        {"skip_friction_formatting": True, "thread_id": target_thread_id},
                    )

                if cmd.name == "node":
                    thread_id = self._active_thread_id_for_user(user_id)
                    try:
                        parts = shlex.split((cmd.args or "").strip())
                    except ValueError:
                        parts = (cmd.args or "").strip().split()
                    if len(parts) < 2:
                        return OrchestratorResponse(
                            'Usage: /node <node_id> "<label>"',
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    raw_node_id = parts[0]
                    raw_label = " ".join(parts[1:])
                    node_id = self._normalize_graph_node_id(raw_node_id)
                    label = self._normalize_graph_text(raw_label, 80, lower=False)
                    created = self.db.create_graph_node(thread_id, node_id, label)
                    if not created:
                        return OrchestratorResponse(
                            "Cannot create node.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    return OrchestratorResponse(
                        f"Node {node_id} created.",
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "link":
                    thread_id = self._active_thread_id_for_user(user_id)
                    try:
                        parts = shlex.split((cmd.args or "").strip())
                    except ValueError:
                        parts = (cmd.args or "").strip().split()
                    if len(parts) != 3:
                        return OrchestratorResponse(
                            "Usage: /link <from_node> <relation> <to_node>",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    from_node = self.db.resolve_graph_ref(thread_id, parts[0])
                    relation = self.db.normalize_relation(parts[1])
                    to_node = self.db.resolve_graph_ref(thread_id, parts[2])
                    if not self.db.validate_relation_allowed(thread_id, relation):
                        return OrchestratorResponse(
                            "Cannot create link.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    if self.db.has_relation_constraint(thread_id, relation, "acyclic") and self.db.would_create_cycle(
                        thread_id, relation, from_node, to_node
                    ):
                        return OrchestratorResponse(
                            "Cannot create link.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    created = self.db.create_graph_edge(thread_id, from_node, to_node, relation)
                    if not created:
                        return OrchestratorResponse(
                            "Cannot create link.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    return OrchestratorResponse(
                        "Link created.",
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "relation_add":
                    thread_id = self._active_thread_id_for_user(user_id)
                    raw = (cmd.args or "").strip()
                    normalized = self.db.normalize_relation(raw)
                    if not normalized:
                        return OrchestratorResponse(
                            "Invalid relation.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    existing = set(self.db.list_relation_types(thread_id))
                    if normalized in existing:
                        return OrchestratorResponse(
                            "Relation type already exists.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    added = self.db.add_relation_type(thread_id, normalized)
                    if not added:
                        return OrchestratorResponse(
                            "Invalid relation.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    return OrchestratorResponse(
                        f"Relation type added: {normalized}.",
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "relation_remove":
                    thread_id = self._active_thread_id_for_user(user_id)
                    raw = (cmd.args or "").strip()
                    normalized = self.db.normalize_relation(raw)
                    if not normalized:
                        return OrchestratorResponse(
                            "Relation type not found.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    removed = self.db.remove_relation_type(thread_id, normalized)
                    if not removed:
                        return OrchestratorResponse(
                            "Relation type not found.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    return OrchestratorResponse(
                        f"Relation type removed: {normalized}.",
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "relations":
                    thread_id = self._active_thread_id_for_user(user_id)
                    strict = self.db.get_relation_strict_mode(thread_id)
                    types = self.db.list_relation_types(thread_id)
                    lines = [
                        f"Relations (thread {thread_id}):",
                        f"Mode: {'strict' if strict else 'open'}",
                        "Types:",
                    ]
                    if not types:
                        lines.append("  (none)")
                    else:
                        for relation in types:
                            lines.append(f"  - {relation}")
                    return OrchestratorResponse(
                        "\n".join(lines).replace("?", ""),
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "relation_mode":
                    thread_id = self._active_thread_id_for_user(user_id)
                    raw = (cmd.args or "").strip().lower()
                    if raw == "strict":
                        self.db.set_relation_strict_mode(thread_id, True)
                        return OrchestratorResponse(
                            "Relation mode set to strict.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    if raw == "open":
                        self.db.set_relation_strict_mode(thread_id, False)
                        return OrchestratorResponse(
                            "Relation mode set to open.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    return OrchestratorResponse(
                        "Usage: /relation_mode strict|open",
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "relation_constraint_add":
                    thread_id = self._active_thread_id_for_user(user_id)
                    try:
                        parts = shlex.split((cmd.args or "").strip())
                    except ValueError:
                        parts = (cmd.args or "").strip().split()
                    if len(parts) != 2:
                        return OrchestratorResponse(
                            "Invalid constraint.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    relation = self.db.normalize_relation(parts[0])
                    constraint = (parts[1] or "").strip().lower()
                    if constraint != "acyclic" or not relation:
                        return OrchestratorResponse(
                            "Invalid constraint.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    if self.db.get_relation_strict_mode(thread_id) and relation not in set(
                        self.db.list_relation_types(thread_id)
                    ):
                        return OrchestratorResponse(
                            "Relation type not found.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    self.db.add_relation_constraint(thread_id, relation, constraint)
                    return OrchestratorResponse(
                        f"Constraint added: {relation} {constraint}.",
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "relation_constraint_remove":
                    thread_id = self._active_thread_id_for_user(user_id)
                    try:
                        parts = shlex.split((cmd.args or "").strip())
                    except ValueError:
                        parts = (cmd.args or "").strip().split()
                    if len(parts) != 2:
                        return OrchestratorResponse(
                            "Constraint not found.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    relation = self.db.normalize_relation(parts[0])
                    constraint = (parts[1] or "").strip().lower()
                    if constraint != "acyclic" or not relation:
                        return OrchestratorResponse(
                            "Constraint not found.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    removed = self.db.remove_relation_constraint(thread_id, relation, constraint)
                    if not removed:
                        return OrchestratorResponse(
                            "Constraint not found.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    return OrchestratorResponse(
                        f"Constraint removed: {relation} {constraint}.",
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "relation_constraints":
                    thread_id = self._active_thread_id_for_user(user_id)
                    items = self.db.list_relation_constraints(thread_id)
                    lines = [f"Relation constraints (thread {thread_id}):"]
                    if not items:
                        lines.append("(none)")
                    else:
                        for relation, constraint in items:
                            lines.append(f"- {relation} {constraint}")
                    return OrchestratorResponse(
                        "\n".join(lines).replace("?", ""),
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "graph":
                    thread_id = self._active_thread_id_for_user(user_id)
                    nodes = self.db.list_graph_nodes(thread_id)
                    aliases = self.db.list_graph_aliases(thread_id)
                    edges = self.db.list_graph_edges(thread_id)
                    strict = self.db.get_relation_strict_mode(thread_id)
                    relation_count = len(self.db.list_relation_types(thread_id))
                    lines = [
                        f"Graph (thread {thread_id}):",
                        f"Mode: {'strict' if strict else 'open'}",
                        f"Declared relations: {relation_count}",
                        "Nodes:",
                    ]
                    if not nodes:
                        lines.append("  - (none)")
                    else:
                        for node in nodes:
                            node_id = self._normalize_graph_node_id(str(node.get("node_id") or ""))
                            label = self._normalize_graph_text(str(node.get("label") or ""), 80, lower=False)
                            if node_id and label:
                                lines.append(f"  - {node_id}: {label}")
                    lines.append("Aliases:")
                    if not aliases:
                        lines.append("  (none)")
                    else:
                        for alias, node_id in aliases:
                            normalized_alias = self._normalize_graph_node_id(alias)
                            normalized_node_id = self._normalize_graph_node_id(node_id)
                            if normalized_alias and normalized_node_id:
                                lines.append(f"  - {normalized_alias} -> {normalized_node_id}")
                    lines.append("Edges:")
                    if not edges:
                        lines.append("  - (none)")
                    else:
                        for edge in edges:
                            from_node = self._normalize_graph_node_id(str(edge.get("from_node") or ""))
                            relation = self._normalize_graph_text(str(edge.get("relation") or ""), 40, lower=True)
                            to_node = self._normalize_graph_node_id(str(edge.get("to_node") or ""))
                            if from_node and relation and to_node:
                                lines.append(f"  - {from_node} --{relation}--> {to_node}")
                    out = "\n".join(lines).replace("?", "")
                    return OrchestratorResponse(
                        out,
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "graph_out":
                    thread_id = self._active_thread_id_for_user(user_id)
                    ref = (cmd.args or "").strip()
                    node_id = self.db.resolve_graph_ref(thread_id, ref)
                    if not node_id:
                        return OrchestratorResponse(
                            "Node not found.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    node_label = self._normalize_graph_text(
                        str(self.db.get_graph_node_label(thread_id, node_id) or "(none)"),
                        80,
                        lower=False,
                    ) or "(none)"
                    out_edges = self.db.list_out_edges(thread_id, node_id)
                    lines = [
                        f"Graph out (thread {thread_id}):",
                        f"Node: {node_id} ({node_label})",
                    ]
                    if not out_edges:
                        lines.append("  (none)")
                    else:
                        grouped: dict[str, list[str]] = defaultdict(list)
                        for relation, to_node in out_edges:
                            grouped[relation].append(to_node)
                        for relation in sorted(grouped):
                            lines.append(f"{relation}:")
                            for to_node in grouped[relation]:
                                to_label = self._normalize_graph_text(
                                    str(self.db.get_graph_node_label(thread_id, to_node) or "(none)"),
                                    80,
                                    lower=False,
                                ) or "(none)"
                                lines.append(f"  - {to_node} ({to_label})")
                    return OrchestratorResponse(
                        "\n".join(lines).replace("?", ""),
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "graph_in":
                    thread_id = self._active_thread_id_for_user(user_id)
                    ref = (cmd.args or "").strip()
                    node_id = self.db.resolve_graph_ref(thread_id, ref)
                    if not node_id:
                        return OrchestratorResponse(
                            "Node not found.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    node_label = self._normalize_graph_text(
                        str(self.db.get_graph_node_label(thread_id, node_id) or "(none)"),
                        80,
                        lower=False,
                    ) or "(none)"
                    in_edges = self.db.list_in_edges(thread_id, node_id)
                    lines = [
                        f"Graph in (thread {thread_id}):",
                        f"Node: {node_id} ({node_label})",
                    ]
                    if not in_edges:
                        lines.append("  (none)")
                    else:
                        grouped: dict[str, list[str]] = defaultdict(list)
                        for relation, from_node in in_edges:
                            grouped[relation].append(from_node)
                        for relation in sorted(grouped):
                            lines.append(f"{relation}:")
                            for from_node in grouped[relation]:
                                from_label = self._normalize_graph_text(
                                    str(self.db.get_graph_node_label(thread_id, from_node) or "(none)"),
                                    80,
                                    lower=False,
                                ) or "(none)"
                                lines.append(f"  - {from_node} ({from_label})")
                    return OrchestratorResponse(
                        "\n".join(lines).replace("?", ""),
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "graph_path":
                    thread_id = self._active_thread_id_for_user(user_id)
                    try:
                        parts = shlex.split((cmd.args or "").strip())
                    except ValueError:
                        parts = (cmd.args or "").strip().split()
                    if len(parts) < 2:
                        return OrchestratorResponse(
                            "Usage: /graph_path <from_ref> <to_ref> [--max <N>]",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    from_ref = parts[0]
                    to_ref = parts[1]
                    max_depth = 6
                    if len(parts) > 2:
                        if len(parts) == 4 and parts[2] == "--max":
                            try:
                                max_depth = int(parts[3])
                            except (TypeError, ValueError):
                                return OrchestratorResponse(
                                    "Usage: /graph_path <from_ref> <to_ref> [--max <N>]",
                                    {"skip_friction_formatting": True, "thread_id": thread_id},
                                )
                        else:
                            return OrchestratorResponse(
                                "Usage: /graph_path <from_ref> <to_ref> [--max <N>]",
                                {"skip_friction_formatting": True, "thread_id": thread_id},
                            )
                    max_depth = max(1, min(10, int(max_depth)))
                    from_node = self.db.resolve_graph_ref(thread_id, from_ref)
                    to_node = self.db.resolve_graph_ref(thread_id, to_ref)
                    if not from_node or not to_node:
                        return OrchestratorResponse(
                            "Node not found.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )

                    def _label(node_id: str) -> str:
                        return self._normalize_graph_text(
                            str(self.db.get_graph_node_label(thread_id, node_id) or "(none)"),
                            80,
                            lower=False,
                        ) or "(none)"

                    if from_node == to_node:
                        label = _label(from_node)
                        return OrchestratorResponse(
                            "\n".join(
                                [
                                    f"Graph path (thread {thread_id}):",
                                    f"From: {from_node} ({label})",
                                    f"To: {to_node} ({label})",
                                    "Depth: 0",
                                    "Path:",
                                    f"  1) {from_node} ({label})",
                                ]
                            ).replace("?", ""),
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )

                    all_edges = self.db.list_all_edges(thread_id)
                    adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
                    for src, relation, dst in all_edges:
                        adjacency[src].append((relation, dst))
                    for src in adjacency:
                        adjacency[src].sort(key=lambda item: (item[0], item[1]))

                    queue: deque[str] = deque([from_node])
                    visited: set[str] = {from_node}
                    depth_by_node: dict[str, int] = {from_node: 0}
                    parent: dict[str, tuple[str, str] | None] = {from_node: None}
                    found = False
                    while queue:
                        current = queue.popleft()
                        current_depth = depth_by_node[current]
                        if current_depth >= max_depth:
                            continue
                        for relation, neighbor in adjacency.get(current, []):
                            if neighbor in visited:
                                continue
                            visited.add(neighbor)
                            parent[neighbor] = (current, relation)
                            depth_by_node[neighbor] = current_depth + 1
                            if neighbor == to_node:
                                found = True
                                queue.clear()
                                break
                            queue.append(neighbor)

                    if not found:
                        return OrchestratorResponse(
                            "\n".join(
                                [
                                    f"Graph path (thread {thread_id}):",
                                    "No path found.",
                                ]
                            ).replace("?", ""),
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )

                    node_path: list[str] = [to_node]
                    edge_path: list[str] = []
                    cursor = to_node
                    while cursor != from_node:
                        link = parent.get(cursor)
                        if link is None:
                            break
                        prev_node, relation = link
                        edge_path.append(relation)
                        node_path.append(prev_node)
                        cursor = prev_node
                    node_path.reverse()
                    edge_path.reverse()
                    lines = [
                        f"Graph path (thread {thread_id}):",
                        f"From: {from_node} ({_label(from_node)})",
                        f"To: {to_node} ({_label(to_node)})",
                        f"Depth: {len(edge_path)}",
                        "Path:",
                        f"  1) {node_path[0]} ({_label(node_path[0])})",
                    ]
                    for idx, (relation, node_id) in enumerate(zip(edge_path, node_path[1:]), start=2):
                        lines.append(f"  {idx}) --{relation}--> {node_id} ({_label(node_id)})")
                    return OrchestratorResponse(
                        "\n".join(lines).replace("?", ""),
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "graph_export":
                    thread_id = self._active_thread_id_for_user(user_id)
                    payload = self.db.export_graph(thread_id)
                    text_out = self._render_pretty_json(payload).replace("?", "")
                    return OrchestratorResponse(
                        text_out,
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "graph_pack_export":
                    active_thread_id = self._active_thread_id_for_user(user_id)
                    raw_args = (cmd.args or "").strip()
                    if not raw_args:
                        thread_ids = [active_thread_id]
                    else:
                        try:
                            tokens = shlex.split(raw_args)
                        except ValueError:
                            tokens = raw_args.split()
                        if len(tokens) != 2 or tokens[0] != "--threads":
                            return OrchestratorResponse(
                                "Export failed.",
                                {"skip_friction_formatting": True, "thread_id": active_thread_id},
                            )
                        requested = [
                            item.replace("?", "").strip()
                            for item in tokens[1].split(",")
                            if item.replace("?", "").strip()
                        ]
                        thread_ids = sorted(set(requested))
                        if not thread_ids or len(thread_ids) > self.db.GRAPH_PACK_MAX_THREADS:
                            return OrchestratorResponse(
                                "Export failed.",
                                {"skip_friction_formatting": True, "thread_id": active_thread_id},
                            )
                        recent_ids = {
                            str(row.get("thread_id") or "").strip()
                            for row in self.db.list_recent_threads(limit=5000)
                            if isinstance(row, dict) and str(row.get("thread_id") or "").strip()
                        }
                        for target_thread_id in thread_ids:
                            if not self.db.thread_exists_for_graph_ops(target_thread_id, recent_ids):
                                return OrchestratorResponse(
                                    "Export failed.",
                                    {"skip_friction_formatting": True, "thread_id": active_thread_id},
                                )
                    payload = self.db.export_graph_pack(thread_ids)
                    text_out = self._render_pretty_json(payload).replace("?", "")
                    return OrchestratorResponse(
                        text_out,
                        {"skip_friction_formatting": True, "thread_id": active_thread_id},
                    )

                if cmd.name == "graph_import":
                    thread_id = self._active_thread_id_for_user(user_id)
                    raw_args = cmd.args or ""
                    stripped = raw_args.lstrip()
                    is_merge = False
                    if stripped.startswith("--merge"):
                        suffix = stripped[len("--merge"):]
                        if not suffix or suffix[0].isspace():
                            is_merge = True
                            payload_text = suffix.strip()
                        else:
                            payload_text = raw_args.strip()
                    else:
                        payload_text = raw_args.strip()
                    if not payload_text:
                        return OrchestratorResponse(
                            "Import failed.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    try:
                        payload = json.loads(payload_text)
                    except json.JSONDecodeError:
                        payload = None
                    if not isinstance(payload, dict):
                        return OrchestratorResponse(
                            "Import failed.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    imported = (
                        self.db.import_graph_merge(thread_id, payload)
                        if is_merge
                        else self.db.import_graph_replace(thread_id, payload)
                    )
                    if not imported:
                        return OrchestratorResponse(
                            "Import failed.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    return OrchestratorResponse(
                        "Graph merged." if is_merge else "Graph imported.",
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "graph_pack_import":
                    active_thread_id = self._active_thread_id_for_user(user_id)
                    raw_args = cmd.args or ""
                    stripped = raw_args.lstrip()
                    is_merge = False
                    if stripped.startswith("--merge"):
                        suffix = stripped[len("--merge"):]
                        if not suffix or suffix[0].isspace():
                            is_merge = True
                            payload_text = suffix.strip()
                        else:
                            payload_text = raw_args.strip()
                    else:
                        payload_text = raw_args.strip()
                    if not payload_text:
                        return OrchestratorResponse(
                            "Import failed.",
                            {"skip_friction_formatting": True, "thread_id": active_thread_id},
                        )
                    try:
                        payload = json.loads(payload_text)
                    except json.JSONDecodeError:
                        payload = None
                    if not isinstance(payload, dict):
                        return OrchestratorResponse(
                            "Import failed.",
                            {"skip_friction_formatting": True, "thread_id": active_thread_id},
                        )
                    imported = (
                        self.db.import_graph_pack_merge(payload)
                        if is_merge
                        else self.db.import_graph_pack_replace(payload)
                    )
                    if not imported:
                        return OrchestratorResponse(
                            "Import failed.",
                            {"skip_friction_formatting": True, "thread_id": active_thread_id},
                        )
                    return OrchestratorResponse(
                        "Pack merged." if is_merge else "Pack imported.",
                        {"skip_friction_formatting": True, "thread_id": active_thread_id},
                    )

                if cmd.name == "graph_clone":
                    active_thread_id = self._active_thread_id_for_user(user_id)
                    try:
                        parts = shlex.split((cmd.args or "").strip())
                    except ValueError:
                        parts = (cmd.args or "").strip().split()
                    if not parts:
                        return OrchestratorResponse(
                            "Clone failed.",
                            {"skip_friction_formatting": True, "thread_id": active_thread_id},
                        )
                    from_thread_id = parts[0].replace("?", "").strip()
                    is_merge = False
                    if len(parts) == 2:
                        if parts[1] == "--merge":
                            is_merge = True
                        else:
                            return OrchestratorResponse(
                                "Clone failed.",
                                {"skip_friction_formatting": True, "thread_id": active_thread_id},
                            )
                    elif len(parts) > 2:
                        return OrchestratorResponse(
                            "Clone failed.",
                            {"skip_friction_formatting": True, "thread_id": active_thread_id},
                        )
                    cloned = self.db.clone_graph(from_thread_id, active_thread_id, merge=is_merge)
                    if not cloned:
                        return OrchestratorResponse(
                            "Clone failed.",
                            {"skip_friction_formatting": True, "thread_id": active_thread_id},
                        )
                    return OrchestratorResponse(
                        f"Graph merged from {from_thread_id}." if is_merge else "Graph cloned.",
                        {"skip_friction_formatting": True, "thread_id": active_thread_id},
                    )

                if cmd.name == "node_rename":
                    thread_id = self._active_thread_id_for_user(user_id)
                    try:
                        parts = shlex.split((cmd.args or "").strip())
                    except ValueError:
                        parts = (cmd.args or "").strip().split()
                    if len(parts) < 2:
                        return OrchestratorResponse(
                            'Usage: /node_rename <node_or_alias> "<new label>"',
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    ref = parts[0]
                    new_label = self._normalize_graph_text(" ".join(parts[1:]), 80, lower=False)
                    node_id = self.db.resolve_graph_ref(thread_id, ref)
                    if not node_id:
                        return OrchestratorResponse(
                            "Node not found.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    updated = self.db.set_graph_node_label(thread_id, node_id, new_label)
                    if not updated:
                        return OrchestratorResponse(
                            "Node not found.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    return OrchestratorResponse(
                        f"Node {node_id} renamed.",
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "node_alias":
                    thread_id = self._active_thread_id_for_user(user_id)
                    try:
                        parts = shlex.split((cmd.args or "").strip())
                    except ValueError:
                        parts = (cmd.args or "").strip().split()
                    if len(parts) != 2:
                        return OrchestratorResponse(
                            "Usage: /node_alias <node_id> <alias>",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    node_id = self._normalize_graph_node_id(parts[0])
                    alias = self._normalize_graph_node_id(parts[1])
                    if not alias:
                        return OrchestratorResponse(
                            "Invalid alias.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    if not self.db.resolve_graph_ref(thread_id, node_id):
                        return OrchestratorResponse(
                            "Node not found.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    if self.db.resolve_graph_ref(thread_id, alias):
                        return OrchestratorResponse(
                            "Alias already exists.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    added = self.db.add_graph_alias(thread_id, alias, node_id)
                    if not added:
                        return OrchestratorResponse(
                            "Alias already exists.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    return OrchestratorResponse(
                        f"Alias {alias} added to {node_id}.",
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "node_unalias":
                    thread_id = self._active_thread_id_for_user(user_id)
                    alias = self._normalize_graph_node_id(cmd.args or "")
                    removed = self.db.remove_graph_alias(thread_id, alias)
                    if not removed:
                        return OrchestratorResponse(
                            "Alias not found.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    return OrchestratorResponse(
                        f"Alias {alias} removed.",
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "node_delete":
                    thread_id = self._active_thread_id_for_user(user_id)
                    ref = (cmd.args or "").strip()
                    node_id = self.db.resolve_graph_ref(thread_id, ref)
                    if not node_id:
                        return OrchestratorResponse(
                            "Node not found.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    deleted = self.db.delete_graph_node(thread_id, node_id)
                    if not deleted:
                        return OrchestratorResponse(
                            "Node not found.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    return OrchestratorResponse(
                        f"Node {node_id} deleted.",
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "focus_node":
                    thread_id = self._active_thread_id_for_user(user_id)
                    raw = (cmd.args or "").strip()
                    if not raw:
                        node_id = self.db.get_thread_focus_node(thread_id)
                        if not node_id:
                            return OrchestratorResponse(
                                "Focus node: (none)",
                                {"skip_friction_formatting": True, "thread_id": thread_id},
                            )
                        node = self.db.get_graph_node(thread_id, node_id)
                        label = self._normalize_graph_text(str((node or {}).get("label") or ""), 80, lower=False)
                        if not label:
                            label = "(none)"
                        return OrchestratorResponse(
                            f"Focus node: {node_id} ({label})",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    resolved = self.db.resolve_graph_ref(thread_id, raw)
                    if not resolved:
                        return OrchestratorResponse(
                            "Node not found.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    set_ok = self.db.set_thread_focus_node(thread_id, resolved)
                    if not set_ok:
                        return OrchestratorResponse(
                            "Node not found.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    return OrchestratorResponse(
                        f"Focus node set to {resolved}.",
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "focus_node_clear":
                    thread_id = self._active_thread_id_for_user(user_id)
                    self.db.clear_thread_focus_node(thread_id)
                    return OrchestratorResponse(
                        "Focus node cleared.",
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "graph_clear":
                    thread_id = self._active_thread_id_for_user(user_id)
                    self.db.clear_graph(thread_id)
                    return OrchestratorResponse(
                        "Graph cleared for this thread.",
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "thread_new":
                    label, pref_flags, body_text = self._parse_thread_new_args(cmd.args or "")
                    thread_id = self._create_new_thread_id_for_user(user_id)
                    self._set_active_thread_id_for_user(user_id, thread_id)
                    self.db.set_thread_label(thread_id, label)

                    for pref_key in (
                        "terse_mode",
                        "show_summary",
                        "show_next_action",
                        "commands_in_codeblock",
                    ):
                        value = pref_flags.get(pref_key)
                        if value in {"on", "off"}:
                            set_thread_pref(self.db, thread_id, pref_key, value)

                    anchor_created = False
                    if body_text.strip():
                        parsed_anchor = parse_anchor_input(f"{label}\n{body_text}")
                        if parsed_anchor is not None:
                            _, bullets, open_line = parsed_anchor
                            create_anchor(self.db, thread_id, label, bullets, open_line)
                            anchor_created = True

                    prefs = self._formatting_prefs(thread_id)
                    terse = "on" if prefs.get("terse_mode") else "off"
                    summary = "on" if prefs.get("show_summary") else "off"
                    next_action = "on" if prefs.get("show_next_action") else "off"
                    codeblock = "on" if prefs.get("commands_in_codeblock") else "off"
                    lines = [
                        "New thread created:",
                        f"Thread: {thread_id}",
                        f"Label: {label}",
                        f"Prefs: terse={terse} summary={summary} next={next_action} codeblock={codeblock}",
                    ]
                    if anchor_created:
                        lines.append("Anchor initialized.")
                    lines.append("Use /resume to continue.")
                    text_out = "\n".join(lines).replace("?", "")
                    return OrchestratorResponse(
                        text_out,
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "thread_label":
                    thread_id = self._active_thread_id_for_user(user_id)
                    normalized = self._normalize_thread_label(cmd.args or "")
                    if not normalized:
                        return OrchestratorResponse(
                            "Usage: /thread_label <label>",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    self.db.set_thread_label(thread_id, normalized)
                    return OrchestratorResponse(
                        f"Label set for {thread_id}.",
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "thread_unlabel":
                    thread_id = self._active_thread_id_for_user(user_id)
                    self.db.clear_thread_label(thread_id)
                    return OrchestratorResponse(
                        f"Label cleared for {thread_id}.",
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "resume":
                    thread_id = self._active_thread_id_for_user(user_id)
                    pending_items = self._memory_runtime.list_pending_items(
                        user_id,
                        thread_id=thread_id,
                        include_expired=False,
                    )
                    resumable_items = [
                        item
                        for item in pending_items
                        if item.get("status") in {PENDING_STATUS_WAITING_FOR_USER, PENDING_STATUS_READY_TO_RESUME}
                    ]
                    if resumable_items:
                        first = resumable_items[0]
                        question = str(first.get("question") or "").strip() or "Pending follow-up."
                        options = first.get("options") if isinstance(first.get("options"), list) else []
                        lines = [
                            f"Resume (thread {thread_id}):",
                            f"Pending: {question}",
                        ]
                        if options:
                            lines.append("Options: " + ", ".join(str(option) for option in options if str(option).strip()))
                        return OrchestratorResponse(
                            "\n".join(lines),
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    rows = self.db.list_thread_anchors(thread_id, limit=2)
                    lines = [f"Resume (thread {thread_id}):"]
                    if not rows:
                        lines.append("No checkpoints yet. Create one with: /anchor <title>")
                        return OrchestratorResponse(
                            "\n".join(lines),
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )

                    latest_row = rows[0]
                    focus = (str(latest_row.get("title") or "")).replace("?", "").strip() or "Checkpoint"
                    lines.append(f"Focus: {focus}")

                    next_text = (str(latest_row.get("open_line") or "")).replace("?", "").strip()
                    if next_text.lower().startswith("open:"):
                        next_text = next_text[5:].strip()
                    if next_text:
                        lines.append(f"Next: {next_text}")

                    def _parse_notes(raw_bullets: str) -> list[str]:
                        try:
                            decoded = json.loads(raw_bullets or "[]")
                        except json.JSONDecodeError:
                            decoded = []
                        out: list[str] = []
                        if isinstance(decoded, list):
                            for item in decoded:
                                if not isinstance(item, str):
                                    continue
                                value = " ".join(item.replace("?", "").split()).strip()
                                if not value:
                                    continue
                                out.append(value)
                                if len(out) >= 2:
                                    break
                        return out

                    notes = _parse_notes(str(latest_row.get("bullets") or "[]"))
                    if not notes and len(rows) > 1:
                        notes = _parse_notes(str(rows[1].get("bullets") or "[]"))
                    if notes:
                        lines.append("Notes:")
                        for note in notes[:2]:
                            lines.append(f"- {note}")

                    lines.append("Tip: Add a new checkpoint with /anchor when you make progress.")
                    focus_node = self.db.get_thread_focus_node(thread_id)
                    if focus_node:
                        related_nodes = self.db.list_related_nodes(thread_id, focus_node, limit=3)
                        lines.append("Related nodes:")
                        if not related_nodes:
                            lines.append("(none)")
                        else:
                            for node_id in related_nodes:
                                node = self.db.get_graph_node(thread_id, node_id)
                                label = self._normalize_graph_text(
                                    str((node or {}).get("label") or ""),
                                    80,
                                    lower=False,
                                )
                                label = label if label else "(none)"
                                lines.append(f"- {node_id} ({label})")
                    return OrchestratorResponse(
                        "\n".join(lines),
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

                if cmd.name == "memory":
                    return self._memory_summary_response(user_id)

                if cmd.name == "anchors_reset":
                    thread_id = self._active_thread_id_for_user(user_id)
                    reset_anchors(self.db, thread_id)
                    return OrchestratorResponse(
                        "Cleared anchors for this thread.",
                        {"skip_friction_formatting": True, "thread_id": thread_id},
                    )

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

                if cmd.name == "project_mode":
                    thread_id = self._active_thread_id_for_user(user_id)
                    raw = (cmd.args or "").strip().lower()
                    if not raw:
                        status = "on" if get_project_mode(self.db, thread_id) else "off"
                        return OrchestratorResponse(
                            f"Project mode: {status}",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    if raw not in {"on", "off"}:
                        return OrchestratorResponse(
                            "Usage: /project_mode <on|off>",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    set_thread_pref(self.db, thread_id, "project_mode", raw)
                    if raw == "on":
                        return OrchestratorResponse(
                            f"Project mode enabled for {thread_id}.",
                            {"skip_friction_formatting": True, "thread_id": thread_id},
                        )
                    return OrchestratorResponse(
                        f"Project mode disabled for {thread_id}.",
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
                    result = route_inference(
                        llm_client=router,
                        messages=[
                            {"role": "system", "content": "Reply with the single word PONG."},
                            {"role": "user", "content": "ping"},
                        ],
                        task_hint="ping diagnostics",
                        purpose="diagnostics",
                        task_type="chat",
                        compute_tier="low",
                        provider_override=provider_override,
                        trace_id=self._trace_id("llm"),
                    )
                    status = "OK" if result.get("ok") else "FAIL"
                    result_data = result.get("data") if isinstance(result.get("data"), dict) else {}
                    reason = result.get("error_kind") or result_data.get("error_kind") or ""
                    reason_part = f" reason={reason}" if status == "FAIL" and reason else ""
                    return OrchestratorResponse(
                        "LLM ping: provider={provider} model={model} status={status} duration_ms={duration_ms}{reason}".format(
                            provider=result.get("provider") or "none",
                            model=result.get("model") or "none",
                            status=status,
                            duration_ms=result.get("duration_ms") or result_data.get("duration_ms") or 0,
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

                if cmd.name == "health_system":
                    self._record_conversation_topic(user_id, "system_health", "command")
                    report = build_system_health_report(collect_system_health())
                    return OrchestratorResponse(
                        render_system_health_summary(
                            report.get("observed") if isinstance(report.get("observed"), dict) else {},
                            report.get("analysis") if isinstance(report.get("analysis"), dict) else {},
                        ),
                        {"system_health": report},
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
                        except (sqlite3.Error, OSError, TypeError, ValueError):
                            return OrchestratorResponse(AUDIT_HARD_FAIL_MSG)
                        return OrchestratorResponse(refusal)
                    gated = self._enforce_authoritative_domain_gate(question)
                    if gated is not None:
                        return gated

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
                        except (sqlite3.Error, OSError, TypeError, ValueError):
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
                        except (sqlite3.Error, OSError, TypeError, ValueError):
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
                        except (sqlite3.Error, OSError, TypeError, ValueError):
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

                if cmd.name == "sys_metrics_snapshot":
                    return self._sys_metrics_snapshot()

                if cmd.name == "sys_health_report":
                    return self._sys_health_report()

                if cmd.name == "sys_inventory_summary":
                    return self._sys_inventory_summary()

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

            runtime_text = cleaned_text if override else text
            if not runtime_text.strip().startswith("/"):
                interpretation_response = self._interpret_previous_result_followup(
                    user_id,
                    runtime_text,
                    chat_context=context,
                )
                if interpretation_response is not None:
                    return interpretation_response
                deeper_system_response = self._deep_system_followup_response(user_id, runtime_text)
                if deeper_system_response is not None:
                    return deeper_system_response
                runtime_response = self._handle_runtime_truth_chat(user_id, runtime_text)
                if runtime_response is not None:
                    return runtime_response
                action_tool_response = self._handle_action_tool_intent(user_id, runtime_text)
                if action_tool_response is not None:
                    return action_tool_response
                grounded_fallback = self._grounded_system_fallback_response(
                    user_id,
                    runtime_text,
                    allow_actions=True,
                )
                if grounded_fallback is not None:
                    return grounded_fallback
                containment_response = self._safe_mode_containment_response(user_id, runtime_text)
                if containment_response is not None:
                    return containment_response

            thread_id = self._active_thread_id_for_user(user_id)
            if (
                self._memory_runtime.should_start_new_thread(user_id, text, thread_id)
                and str(self._last_offer_topic.get(user_id) or "").strip() != "brief_offer"
            ):
                self._memory_runtime.abort_pending_for_thread(user_id, thread_id)

            followup = self._memory_runtime.resolve_followup(user_id, text, thread_id)
            followup_type = str(followup.get("type") or "")
            followup_reason = str(followup.get("reason") or "")
            followup_intent = str(followup.get("intent") or "")
            if followup_type == "ambiguous":
                return self._continuity_error_response(
                    title="❌ Follow-up is ambiguous.",
                    failure_code="followup_ambiguous",
                    next_action="Reply with the exact question you want to continue.",
                )
            if followup_type == "expired":
                return self._continuity_error_response(
                    title="❌ That pending step expired.",
                    failure_code="pending_expired",
                    next_action="Ask me to run it again.",
                )
            if followup_type == "none" and followup_reason == "no_pending":
                if str(self._last_offer_topic.get(user_id) or "").strip() != "brief_offer":
                    return self._continuity_error_response(
                        title="❌ No resumable work is active.",
                        failure_code="no_resumable_work",
                        next_action="Ask \"what are we doing?\" or start a new request.",
                    )
            if followup_type == "match":
                pending_item = followup.get("pending_item") if isinstance(followup.get("pending_item"), dict) else {}
                pending_id = str(pending_item.get("pending_id") or "").strip()
                pending_kind = str(pending_item.get("kind") or "").strip().lower()
                pending_status = str(pending_item.get("status") or "").strip().upper()
                if pending_status == PENDING_STATUS_EXPIRED:
                    if pending_id:
                        self._memory_runtime.set_pending_status(user_id, pending_id, PENDING_STATUS_EXPIRED)
                    return self._continuity_error_response(
                        title="❌ That pending step expired.",
                        failure_code="pending_expired",
                        next_action="Ask me to run it again.",
                    )
                if pending_kind == "followup" and str(pending_item.get("origin_tool") or "") == "compare_now":
                    if followup_intent in {"accept", "details"}:
                        compare_pending = self._get_pending_compare(user_id)
                        if not compare_pending or not compare_pending.get("what_if_text"):
                            if pending_id:
                                self._memory_runtime.set_pending_status(user_id, pending_id, PENDING_STATUS_ABORTED)
                            return self._continuity_error_response(
                                title="❌ Compare context is no longer available.",
                                failure_code="resumable_missing",
                                next_action="Ask a new what-if question first.",
                            )
                        what_if_text = str(compare_pending.get("what_if_text") or "").strip()
                        if pending_id:
                            self._memory_runtime.set_pending_status(user_id, pending_id, PENDING_STATUS_DONE)
                        self._pending_compare.pop(user_id, None)
                        return OrchestratorResponse(compare_now_to_what_if(what_if_text))
                    if followup_intent == "decline":
                        if pending_id:
                            self._memory_runtime.set_pending_status(user_id, pending_id, PENDING_STATUS_ABORTED)
                        self._pending_compare.pop(user_id, None)
                        return OrchestratorResponse("Okay — I cancelled that pending compare step.")
                if pending_kind == "confirmation":
                    if followup_intent in {"accept", "details"}:
                        pending_action = self.confirmations.pop(user_id)
                        if not pending_action:
                            if pending_id:
                                self._memory_runtime.set_pending_status(user_id, pending_id, PENDING_STATUS_ABORTED)
                            return self._continuity_error_response(
                                title="❌ Confirmation target is no longer available.",
                                failure_code="resumable_missing",
                                next_action="Re-run the action you want to confirm.",
                            )
                        if pending_id:
                            self._memory_runtime.set_pending_status(user_id, pending_id, PENDING_STATUS_DONE)
                        action = pending_action.action
                        if str(action.get("kind") or "").strip().lower() == "native_mutation":
                            return self._execute_confirmed_native_mutation(user_id, action)
                        return self._call_skill(
                            user_id,
                            action["skill"],
                            action["function"],
                            action["args"],
                            action["requested_permissions"],
                            action.get("action_type"),
                            confirmed=True,
                        )
                    if followup_intent == "decline":
                        self.confirmations.pop(user_id)
                        if pending_id:
                            self._memory_runtime.set_pending_status(user_id, pending_id, PENDING_STATUS_ABORTED)
                        return OrchestratorResponse("Okay — I cancelled that pending confirmation.")
                if pending_kind in {"clarification", "confirmation", "task"}:
                    question = str(pending_item.get("question") or "").strip() or "Please answer the pending prompt."
                    options = pending_item.get("options") if isinstance(pending_item.get("options"), list) else []
                    lines = [question]
                    if options:
                        lines.append("Options: " + ", ".join(str(item) for item in options if str(item).strip()))
                    lines.append("Reply with the exact option.")
                    return OrchestratorResponse("\n".join(lines))

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
                    return self._agent_memory_response(user_id, "agent_memory_inspect", query_text=text)
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
                    if not self._llm_chat_available():
                        return self._bootstrap_no_chat_response()

            gate_result = handle_action_text(self.db, user_id, text, self.enable_writes)
            if gate_result:
                return OrchestratorResponse(gate_result.get("message", ""))

            if self._llm_chat_available():
                return self._llm_chat(user_id, text, chat_context=context)

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
            except (TypeError, ValueError, sqlite3.Error, OSError):
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
                for pending_item in self._memory_runtime.list_pending_items(
                    user_id,
                    thread_id=self._active_thread_id_for_user(user_id),
                    include_expired=True,
                ):
                    if str(pending_item.get("origin_tool") or "") != "compare_now":
                        continue
                    self._memory_runtime.set_pending_status(
                        user_id,
                        str(pending_item.get("pending_id") or ""),
                        PENDING_STATUS_DONE,
                    )
                self._pending_compare.pop(user_id, None)
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
                if not self._llm_chat_available():
                    return self._bootstrap_no_chat_response()
                return OrchestratorResponse("Hi. I’m ready to help. Tell me what you want to do.")

            if not self._llm_chat_available():
                return self._bootstrap_no_chat_response()
            return OrchestratorResponse("I’m not sure what you need yet. Send 'help' for commands.")
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
            text = "I have a baseline now. I'll report changes next time."
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
        except (sqlite3.Error, OSError, TypeError, ValueError):
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
        open_loops = self.db.list_open_loops(status="open", limit=6, order="due")
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
            except (OSError, UnicodeError, ValueError):
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
            open_loops = [row for row in open_loops if int(row.get("priority") or 3) <= 2]
        if "top 3" in lowered or "top three" in lowered or "priorities" in lowered:
            tasks = tasks[:3]
            open_loops = open_loops[:3]
        task_lines = [
            f"{title} ({mins}m)" if mins is not None else title
            for title, mins in tasks[:6]
            if title
        ]
        open_loop_lines: list[str] = []
        for row in open_loops[:3]:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "").strip()
            if not title:
                continue
            due = str(row.get("due_date") or "").strip()
            priority = int(row.get("priority") or 3)
            open_loop_lines.append(f"P{priority} {title}" + (f" (due {due})" if due else ""))
        lines = [*open_loop_lines, *task_lines]
        if not lines:
            lines = ["I do not have any active tasks or urgent open loops saved right now."]
        cards = [{"key": "today-plan", "title": "Today priorities", "lines": lines[:6], "severity": "ok"}]
        if open_loop_lines:
            cards.append(
                {
                    "key": "today-open-loops",
                    "title": "Open loops I am tracking",
                    "lines": open_loop_lines[:3],
                    "severity": "warn" if any("(due " in line for line in open_loop_lines[:3]) else "ok",
                }
            )
        if open_loop_lines and task_lines:
            summary = "Here is a practical plan for today based on your open loops and active tasks."
        elif open_loop_lines:
            summary = "Here is a practical plan for today based on the open loops I am tracking."
        elif task_lines:
            summary = "Here is a practical plan for today based on your active tasks."
        else:
            summary = "I do not have any active tasks or urgent open loops saved right now."
        return build_cards_payload(
            cards,
            raw_available=True,
            summary=summary,
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
                except (TypeError, ValueError):
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
        except (OSError, subprocess.SubprocessError):
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
                except (TypeError, ValueError, OverflowError):
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
        if skill_name == "hardware_report":
            cards_payload = data.get("cards_payload") if isinstance(data, dict) else {}
            next_questions = [
                str(item).strip()
                for item in (
                    cards_payload.get("next_questions")
                    if isinstance(cards_payload, dict) and isinstance(cards_payload.get("next_questions"), list)
                    else []
                )
                if str(item).strip()
            ]
            payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
            memory = payload.get("memory") if isinstance(payload.get("memory"), dict) else {}
            gpu = payload.get("gpu") if isinstance(payload.get("gpu"), dict) else {}
            ram_total = float(memory.get("total_bytes") or 0.0)
            ram_available = float(memory.get("available_bytes") or 0.0)
            ram_total_gib = ram_total / float(1024**3) if ram_total else 0.0
            ram_available_gib = ram_available / float(1024**3) if ram_available else 0.0
            gpu_available = bool(gpu.get("available", False))
            gpu_rows = gpu.get("gpus") if isinstance(gpu.get("gpus"), list) else []
            first_gpu = gpu_rows[0] if gpu_rows and isinstance(gpu_rows[0], dict) else {}
            gpu_name = str(first_gpu.get("name") or "GPU").strip()
            vram_total_mb = int(first_gpu.get("memory_total_mb") or 0)
            vram_used_mb = int(first_gpu.get("memory_used_mb") or 0)
            vram_free_mb = max(vram_total_mb - vram_used_mb, 0) if vram_total_mb else 0
            if ram_total_gib > 0:
                summary = f"You have {ram_total_gib:.0f} GiB of RAM with {ram_available_gib:.0f} GiB available."
            else:
                summary = "RAM availability is unavailable right now."
            if gpu_available and vram_total_mb > 0:
                summary += f" VRAM is available on {gpu_name} with {vram_free_mb} MiB free out of {vram_total_mb} MiB total."
            else:
                summary += " VRAM is unavailable right now."
            return summary, next_questions or [
                "How much memory am I using?",
                "How is my storage?",
                "Can you see the GPU?",
            ]
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
            api_active = "- personal-agent-api.service: status=active" in lower
            telegram_active = "- personal-agent-telegram.service: status=active" in lower
            telegram_failed = "- personal-agent-telegram.service: status=failed" in lower
            verdict = "API service is running" if api_active else "API service is stopped/degraded"
            if telegram_failed:
                verdict += "; Telegram service failed"
            elif telegram_active:
                verdict += "; Telegram service active"
            return verdict, ["Show last 20 service logs", "Run status"]
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
            "hardware_report": ["sys:read", "net:none"],
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
