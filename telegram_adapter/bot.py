from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone, time
from zoneinfo import ZoneInfo
import json
import os
from pathlib import Path
import sys
import tempfile
import time as pytime
from typing import Any, Callable

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
except ModuleNotFoundError:  # pragma: no cover - testing without telegram installed
    Update = object  # type: ignore
    Application = object  # type: ignore
    CommandHandler = object  # type: ignore
    ContextTypes = object  # type: ignore
    MessageHandler = object  # type: ignore
    filters = object()  # type: ignore

from agent.config import Config, load_config
from agent.fallback_ladder import run_with_fallback
from agent.llm_router import LLMRouter
from agent.llm_client import build_llm_broker
from agent.logging_utils import log_event
from agent.scheduled_snapshots import (
    safe_run_scheduled_snapshot,
    safe_run_storage_snapshot,
    safe_run_resource_snapshot,
    safe_run_network_snapshot,
)
from agent.debug_protocol import DebugProtocol
from agent.orchestrator import Orchestrator
from agent.cards import render_cards_markdown, validate_cards_payload
from agent.daily_brief import should_send_daily_brief
from agent.model_scout import build_model_scout
from agent.audit_log import AuditLog
from agent.secret_store import SecretStore
from agent.permissions import PermissionStore
from agent.ux.clarify_suggest import (
    build_clarify_message,
    build_suggest_message,
    classify_ambiguity,
    parse_recovery_choice,
    recovery_options,
)
from agent.ux.llm_fixit_wizard import LLMFixitWizardStore
from memory.db import MemoryDB


_TELEGRAM_BOT_TOKEN_SECRET_KEY = "telegram:bot_token"
_TELEGRAM_FALLBACK_TEXT = "I hit an internal error, but I’m still running. Try one of these:"
_TELEGRAM_HELP_TEXT = (
    "Try one of these:\n"
    "1) /brief\n"
    "2) what model are we using?\n"
    "3) check for new models\n"
    "Commands: /brief, /model, /ask, /ask_opinion, /task_add, /done, /health, /scout, /help"
)
_TELEGRAM_UNKNOWN_FALLBACK_TEXT = (
    "I can help with system updates, tasks, and troubleshooting.\n"
    "Examples: /brief or \"anything new on my PC?\"\n"
    "For more options, send /help."
)
_TELEGRAM_STATUS_TOKENS = {"ping", "hello", "hi"}
_MODEL_PROVIDER_INTENTS = {
    "model_watch.run_now",
    "provider.status",
    "provider.setup.ollama",
    "provider.setup.openrouter",
    "provider.help",
    "none",
}
_MODEL_PROVIDER_HELP_TEXT = (
    "I can help with model/provider setup.\n"
    "Try: \"what model are we using\", \"check models\", \"setup ollama\", or \"setup openrouter\"."
)
_NO_ACTIVE_CHOICE_TEXT = "No active choice right now. Ask me to check models or set up a provider."
_WIZARD_CANCEL_TOKENS = {"cancel", "stop", "never mind", "nevermind", "quit"}
_WIZARD_YES_TOKENS = {"1", "yes", "y", "ok", "ready"}
_WIZARD_NO_TOKENS = {"2", "no", "n"}
_WIZARD_CHOICE_TOKENS = {"1", "2", "3", "yes", "no", "y", "n", "ok", "cancel"}
_SETUP_WIZARD_SCHEMA_VERSION = 1
_OLLAMA_TIER_CHOICES = [
    {"id": "small", "label": "Small (fastest)", "recommended": True},
    {"id": "medium", "label": "Medium (balanced)", "recommended": False},
    {"id": "large", "label": "Large (best quality)", "recommended": False},
]


def resolve_telegram_bot_token_with_source() -> tuple[str | None, str]:
    secret_store = SecretStore(path=os.getenv("AGENT_SECRET_STORE_PATH", "").strip() or None)
    secret_token = (secret_store.get_secret(_TELEGRAM_BOT_TOKEN_SECRET_KEY) or "").strip()
    if secret_token:
        return secret_token, "secret_store"
    env_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if env_token:
        return env_token, "env"
    return None, "missing"


def _resolve_telegram_bot_token() -> str | None:
    token, _source = resolve_telegram_bot_token_with_source()
    return token


def _safe_reply_text(text: str | None) -> str:
    value = str(text or "").strip()
    return value if value else "I’m still here. What should I do next?"


def _normalize_user_text(text: str | None) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _contains_any(normalized_text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in normalized_text for phrase in phrases)


def classify_model_provider_intent(text: str) -> str:
    normalized = _normalize_user_text(text)
    normalized_cmp = _normalize_user_text(normalized.replace("/", " ").replace("-", " "))
    if not normalized_cmp:
        return "none"
    if _contains_any(
        normalized_cmp,
        (
            "setup openrouter",
            "set up openrouter",
            "configure openrouter",
            "openrouter setup",
        ),
    ):
        return "provider.setup.openrouter"
    if _contains_any(
        normalized_cmp,
        (
            "setup ollama",
            "set up ollama",
            "configure ollama",
            "ollama setup",
        ),
    ):
        return "provider.setup.ollama"
    if _contains_any(
        normalized_cmp,
        (
            "check models",
            "check for new models",
            "check for better models",
            "new models",
            "better models",
            "run model watch",
            "model watch now",
            "scan models",
        ),
    ):
        return "model_watch.run_now"
    if _contains_any(
        normalized_cmp,
        (
            "what model are we using",
            "which model are we using",
            "what provider are we using",
            "which provider are we using",
            "current model",
            "model status",
            "provider status",
            "what model are you using",
        ),
    ):
        return "provider.status"
    if _contains_any(
        normalized_cmp,
        (
            "model help",
            "provider help",
            "llm help",
            "help with model",
            "help with provider",
        ),
    ):
        return "provider.help"
    return "none"


class TelegramModelProviderWizardStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or self.default_path()).expanduser().resolve()
        self.state = self.load()

    @staticmethod
    def default_path() -> str:
        env_path = os.getenv("AGENT_TELEGRAM_MODEL_WIZARD_STATE_PATH", "").strip()
        if env_path:
            return env_path
        return str(
            Path.home()
            / ".local"
            / "share"
            / "personal-agent"
            / "telegram_model_provider_wizard_state.json"
        )

    @staticmethod
    def empty_state() -> dict[str, Any]:
        return {
            "schema_version": _SETUP_WIZARD_SCHEMA_VERSION,
            "active": False,
            "flow": None,
            "step": "idle",
            "question": None,
            "choices": [],
            "chat_id": None,
            "issue_code": None,
            "pending": {},
            "created_ts": None,
            "updated_ts": None,
        }

    def _normalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        state = self.empty_state()
        state["schema_version"] = _SETUP_WIZARD_SCHEMA_VERSION
        state["active"] = bool(raw.get("active", False))
        flow = str(raw.get("flow") or "").strip().lower()
        state["flow"] = flow if flow in {"openrouter_setup", "ollama_setup", "recovery"} else None
        step = str(raw.get("step") or "").strip().lower()
        valid_steps = {
            "idle",
            "awaiting_openrouter_has_key",
            "awaiting_openrouter_key",
            "awaiting_openrouter_default",
            "awaiting_ollama_ready",
            "awaiting_ollama_size",
            "awaiting_ollama_pull",
            "awaiting_ollama_default",
            "awaiting_recovery_choice",
        }
        state["step"] = step if step in valid_steps else "idle"
        state["question"] = str(raw.get("question") or "").strip() or None
        state["chat_id"] = str(raw.get("chat_id") or "").strip() or None
        state["issue_code"] = str(raw.get("issue_code") or "").strip() or None
        pending = raw.get("pending") if isinstance(raw.get("pending"), dict) else {}
        state["pending"] = {str(key): pending[key] for key in sorted(pending.keys())}
        try:
            state["created_ts"] = int(raw.get("created_ts") or 0) or None
        except (TypeError, ValueError):
            state["created_ts"] = None
        try:
            state["updated_ts"] = int(raw.get("updated_ts") or 0) or None
        except (TypeError, ValueError):
            state["updated_ts"] = None
        choices_raw = raw.get("choices") if isinstance(raw.get("choices"), list) else []
        choices: list[dict[str, Any]] = []
        for row in choices_raw:
            if not isinstance(row, dict):
                continue
            choice_id = str(row.get("id") or "").strip()
            label = str(row.get("label") or "").strip()
            if not choice_id or not label:
                continue
            choices.append(
                {
                    "id": choice_id,
                    "label": label,
                    "recommended": bool(row.get("recommended", False)),
                }
            )
        state["choices"] = choices
        if not bool(state["active"]):
            state["flow"] = None
            state["step"] = "idle"
            state["question"] = None
            state["choices"] = []
            state["chat_id"] = None
            state["issue_code"] = None
            state["pending"] = {}
        return state

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return self.empty_state()
        try:
            parsed = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return self.empty_state()
        if not isinstance(parsed, dict):
            return self.empty_state()
        return self._normalize(parsed)

    def _write(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(state, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

    def save(self, raw: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize(raw if isinstance(raw, dict) else {})
        self._write(normalized)
        self.state = normalized
        return normalized

    def clear(self) -> dict[str, Any]:
        return self.save(self.empty_state())


def _format_commit_short(value: str | None) -> str:
    commit = str(value or "").strip()
    if not commit:
        return "unknown"
    return commit[:12]


def _runtime_status_text(bot_data: dict[str, Any]) -> str:
    runtime = bot_data.get("runtime")
    version = str(getattr(runtime, "version", "") or bot_data.get("runtime_version") or "0.1.0").strip() or "0.1.0"
    commit_value = str(getattr(runtime, "git_commit", "") or bot_data.get("runtime_git_commit") or "unknown").strip()
    commit_short = _format_commit_short(commit_value)

    started_at = getattr(runtime, "started_at", None)
    uptime_seconds = 0
    if isinstance(started_at, datetime):
        uptime_seconds = max(0, int((datetime.now(timezone.utc) - started_at).total_seconds()))
    else:
        started_ts_raw = bot_data.get("runtime_started_ts")
        try:
            started_ts = float(started_ts_raw)
        except Exception:
            started_ts = pytime.time()
        uptime_seconds = max(0, int(pytime.time() - started_ts))

    return (
        f"✅ Agent is running (v{version}, commit {commit_short}, uptime {uptime_seconds}s).\n"
        "Try /brief or ask 'anything new on my PC?'"
    )


def _smalltalk_preroute_reply(text: str, bot_data: dict[str, Any]) -> tuple[str | None, str | None]:
    normalized = _normalize_user_text(text)
    if normalized in {"/help", "help"}:
        return _TELEGRAM_HELP_TEXT, "help"
    if normalized in _TELEGRAM_STATUS_TOKENS:
        return _runtime_status_text(bot_data), "status"
    return None, None


def _runtime_llm_availability(bot_data: dict[str, Any]) -> tuple[bool, str]:
    runtime = bot_data.get("runtime")
    if runtime is None or not hasattr(runtime, "llm_availability_state"):
        return False, "llm_unavailable"
    try:
        state = runtime.llm_availability_state()  # type: ignore[attr-defined]
    except Exception:
        return False, "llm_unavailable"
    if not isinstance(state, dict):
        return False, "llm_unavailable"
    return bool(state.get("available", False)), str(state.get("reason") or "llm_unavailable").strip().lower()


def _set_recovery_wizard_state(
    *,
    wizard_store: TelegramModelProviderWizardStore | None,
    chat_id: str,
    reason: str,
) -> None:
    if wizard_store is None:
        return
    _save_setup_state(
        wizard_store,
        chat_id=chat_id,
        flow="recovery",
        step="awaiting_recovery_choice",
        question="Reply 1, 2, or 3.",
        issue_code="llm.recovery",
        choices=recovery_options(),
        pending={"reason": str(reason or "").strip().lower() or "llm_unavailable"},
    )


def _llm_status_payload(runtime: Any | None) -> dict[str, Any]:
    if runtime is None or not hasattr(runtime, "llm_status"):
        return {}
    try:
        payload = runtime.llm_status()  # type: ignore[attr-defined]
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _model_status_report(*, runtime: Any | None) -> str:
    status = _llm_status_payload(runtime)
    default_provider = str(status.get("default_provider") or "unknown").strip() or "unknown"
    resolved_model = (
        str(status.get("resolved_default_model") or "").strip()
        or str(status.get("default_model") or "").strip()
        or "unknown"
    )
    providers_rows = status.get("providers") if isinstance(status.get("providers"), list) else []
    provider_ids = sorted(
        str(row.get("id") or "").strip().lower()
        for row in providers_rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    )
    provider_line = ", ".join(provider_ids) if provider_ids else "none"

    model_watch_line = "Model Watch: no summary available."
    if runtime is not None and hasattr(runtime, "model_watch_latest"):
        try:
            latest = runtime.model_watch_latest()  # type: ignore[attr-defined]
        except Exception:
            latest = {}
        if isinstance(latest, dict):
            if bool(latest.get("found")) and isinstance(latest.get("batch"), dict):
                batch = latest.get("batch") if isinstance(latest.get("batch"), dict) else {}
                top = batch.get("top_pick") if isinstance(batch.get("top_pick"), dict) else {}
                top_model = str(top.get("model") or top.get("id") or "unknown").strip()
                top_score = float(top.get("score") or 0.0)
                model_watch_line = f"Model Watch: top candidate {top_model} (score {top_score:.2f})."
            elif str(latest.get("reason") or "").strip():
                model_watch_line = f"Model Watch: {str(latest.get('reason') or '').strip()}."

    return "\n".join(
        [
            f"Current provider/model: {default_provider} / {resolved_model}",
            f"Configured providers: {provider_line}",
            model_watch_line,
        ]
    )


def _is_ollama_reachable(*, runtime: Any | None) -> bool:
    status = _llm_status_payload(runtime)
    providers_rows = status.get("providers") if isinstance(status.get("providers"), list) else []
    for row in providers_rows:
        if not isinstance(row, dict):
            continue
        provider_id = str(row.get("id") or "").strip().lower()
        if provider_id != "ollama":
            continue
        health = row.get("health") if isinstance(row.get("health"), dict) else {}
        health_status = str(health.get("status") or "unknown").strip().lower()
        if health_status in {"ok", "unknown"}:
            return True
    return False


def _choose_openrouter_default_model(*, runtime: Any | None) -> str | None:
    status = _llm_status_payload(runtime)
    rows = status.get("models") if isinstance(status.get("models"), list) else []
    candidates: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("provider") or "").strip().lower() != "openrouter":
            continue
        model_id = str(row.get("id") or "").strip()
        if not model_id:
            continue
        if bool(row.get("enabled", False)) and bool(row.get("available", False)) and bool(row.get("routable", False)):
            candidates.append(model_id)
    if candidates:
        return sorted(candidates)[0]
    fallback = sorted(
        str(row.get("id") or "").strip()
        for row in rows
        if isinstance(row, dict)
        and str(row.get("provider") or "").strip().lower() == "openrouter"
        and str(row.get("id") or "").strip()
    )
    return fallback[0] if fallback else None


def _choose_ollama_model_for_tier(*, runtime: Any | None, tier: str) -> str | None:
    status = _llm_status_payload(runtime)
    rows = status.get("models") if isinstance(status.get("models"), list) else []
    models = sorted(
        str(row.get("id") or "").strip()
        for row in rows
        if isinstance(row, dict)
        and str(row.get("provider") or "").strip().lower() == "ollama"
        and str(row.get("id") or "").strip()
        and bool(row.get("enabled", False))
    )
    if not models:
        return None
    patterns = {
        "small": (":1b", ":2b", ":3b", ":4b"),
        "medium": (":6b", ":7b", ":8b", ":9b", ":10b", ":11b", ":12b", ":13b"),
        "large": (":14b", ":20b", ":30b", ":32b", ":70b", ":72b"),
    }
    tier_patterns = patterns.get(tier, ())
    for model_id in models:
        lower_model = model_id.lower()
        if any(pattern in lower_model for pattern in tier_patterns):
            return model_id
    return models[0]


def _choice_id_from_text(*, text: str, choices: list[dict[str, Any]]) -> str | None:
    normalized = _normalize_user_text(text)
    if not normalized:
        return None
    if normalized.isdigit():
        idx = int(normalized)
        if 1 <= idx <= len(choices):
            return str(choices[idx - 1].get("id") or "").strip() or None
    for row in choices:
        if not isinstance(row, dict):
            continue
        choice_id = str(row.get("id") or "").strip().lower()
        label = str(row.get("label") or "").strip().lower()
        if normalized == choice_id or (label and normalized == label):
            return str(row.get("id") or "").strip() or None
    return None


def _save_setup_state(
    store: TelegramModelProviderWizardStore,
    *,
    chat_id: str,
    flow: str,
    step: str,
    question: str,
    issue_code: str,
    choices: list[dict[str, Any]] | None = None,
    pending: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now_epoch = int(pytime.time())
    current = store.state if isinstance(store.state, dict) else store.empty_state()
    created_ts = int(current.get("created_ts") or now_epoch)
    state = {
        "active": True,
        "flow": flow,
        "step": step,
        "question": question,
        "choices": choices if isinstance(choices, list) else [],
        "chat_id": chat_id,
        "issue_code": issue_code,
        "pending": pending if isinstance(pending, dict) else {},
        "created_ts": created_ts,
        "updated_ts": now_epoch,
    }
    return store.save(state)


def _start_openrouter_setup(
    *,
    runtime: Any | None,
    wizard_store: TelegramModelProviderWizardStore,
    chat_id: str,
) -> str:
    if runtime is None:
        wizard_store.clear()
        return "OpenRouter setup is unavailable in this runtime."
    _save_setup_state(
        wizard_store,
        chat_id=chat_id,
        flow="openrouter_setup",
        step="awaiting_openrouter_has_key",
        question="Do you already have an OpenRouter API key? Reply 1 for Yes or 2 for No.",
        issue_code="provider.setup.openrouter",
        choices=[
            {"id": "has_key_yes", "label": "Yes", "recommended": True},
            {"id": "has_key_no", "label": "No", "recommended": False},
        ],
    )
    return "OpenRouter setup: Do you already have an OpenRouter API key? Reply 1 for Yes or 2 for No."


def _start_ollama_setup(
    *,
    runtime: Any | None,
    wizard_store: TelegramModelProviderWizardStore,
    chat_id: str,
) -> str:
    if runtime is None:
        wizard_store.clear()
        return "Ollama setup is unavailable in this runtime."
    if _is_ollama_reachable(runtime=runtime):
        _save_setup_state(
            wizard_store,
            chat_id=chat_id,
            flow="ollama_setup",
            step="awaiting_ollama_size",
            question="Which Ollama size tier do you want? Reply 1 small, 2 medium, or 3 large.",
            issue_code="provider.setup.ollama",
            choices=list(_OLLAMA_TIER_CHOICES),
        )
        return "Ollama is reachable. Which size tier do you want? Reply 1 small, 2 medium, or 3 large."
    _save_setup_state(
        wizard_store,
        chat_id=chat_id,
        flow="ollama_setup",
        step="awaiting_ollama_ready",
        question="Run this command: ollama list. Then reply READY.",
        issue_code="provider.setup.ollama",
        choices=[],
    )
    return "I can’t reach Ollama yet. Run this command: `ollama list` and then reply READY."


def _model_watch_run_summary(*, runtime: Any | None) -> str:
    if runtime is None or not hasattr(runtime, "run_model_watch_once"):
        return "Model watch is unavailable in this runtime."
    try:
        ok, body = runtime.run_model_watch_once(trigger="manual")  # type: ignore[attr-defined]
    except Exception:
        return "Model watch run failed."
    payload = body if isinstance(body, dict) else {}
    if not ok:
        message = str(payload.get("message") or payload.get("error") or "Model watch run failed.").strip()
        return message or "Model watch run failed."
    proposal = payload.get("proposal") if isinstance(payload.get("proposal"), dict) else None
    if not isinstance(proposal, dict):
        return "No new better models in configured providers."
    proposal_type = str(payload.get("proposal_type") or proposal.get("proposal_type") or "").strip().lower()
    if proposal_type == "local_download":
        repo_id = str(proposal.get("repo_id") or "unknown").strip()
        revision = str(proposal.get("revision") or "unknown").strip()
        return (
            f"Model watch found a local download candidate: {repo_id} @ {revision}.\n"
            "Use the current fix-it prompt to approve or snooze."
        )
    to_model = str(proposal.get("to_model") or "unknown").strip()
    score_delta = float(proposal.get("score_delta") or 0.0)
    return (
        "Model watch found a better default candidate.\n"
        f"Top candidate: {to_model} (+{score_delta:.2f}).\n"
        "Use the current fix-it prompt to apply or snooze."
    )


def _setup_wizard_route_reply(
    *,
    runtime: Any | None,
    wizard_store: TelegramModelProviderWizardStore | None,
    llm_fixit_fn: Callable[[dict[str, Any]], tuple[bool, dict[str, Any]]] | None,
    orchestrator: Orchestrator | None,
    chat_id: str,
    text: str,
) -> str | None:
    if runtime is None or wizard_store is None:
        return None
    try:
        state = wizard_store.load()
        wizard_store.state = state
    except Exception:
        state = wizard_store.empty_state()
    if not bool(state.get("active", False)):
        return None
    state_chat = str(state.get("chat_id") or "").strip()
    if state_chat and state_chat != chat_id:
        return None
    normalized = _normalize_user_text(text)
    if normalized in _WIZARD_CANCEL_TOKENS:
        wizard_store.clear()
        return "Cancelled."

    flow = str(state.get("flow") or "").strip().lower()
    step = str(state.get("step") or "").strip().lower()
    pending = state.get("pending") if isinstance(state.get("pending"), dict) else {}
    issue_code = str(state.get("issue_code") or "provider.setup").strip()

    if flow == "recovery" and step == "awaiting_recovery_choice":
        options = state.get("choices") if isinstance(state.get("choices"), list) else recovery_options()
        choice = parse_recovery_choice(text, options=options)
        if not choice:
            return "Reply 1, 2, or 3."
        wizard_store.clear()
        if choice == "status":
            return _model_status_report(runtime=runtime)
        if choice == "fixit":
            if llm_fixit_fn is None:
                return "LLM fix-it is unavailable right now."
            ok, body = llm_fixit_fn({"actor": "telegram"})
            message = str((body or {}).get("message") or (body or {}).get("next_question") or "").strip()
            if message:
                return message
            return "LLM fix-it is ready." if ok else "LLM fix-it is unavailable right now."
        if orchestrator is not None:
            response = orchestrator.handle_message("/brief", user_id=chat_id)
            reply = str((response.text if response else "") or "").strip()
            if reply:
                return reply
        return "Brief summary is unavailable right now."

    if flow == "openrouter_setup":
        if step == "awaiting_openrouter_has_key":
            if normalized in _WIZARD_YES_TOKENS:
                _save_setup_state(
                    wizard_store,
                    chat_id=chat_id,
                    flow=flow,
                    step="awaiting_openrouter_key",
                    question="Paste your OpenRouter API key now.",
                    issue_code=issue_code,
                    choices=[],
                    pending={},
                )
                return "Paste your OpenRouter API key now."
            if normalized in _WIZARD_NO_TOKENS:
                wizard_store.clear()
                return "No problem. Create an OpenRouter key, then send 'setup openrouter' again."
            return "Reply 1 for Yes or 2 for No."

        if step == "awaiting_openrouter_key":
            api_key = str(text or "").strip()
            if not api_key:
                return "Paste your OpenRouter API key now."
            ok_secret, _body_secret = runtime.set_provider_secret("openrouter", {"api_key": api_key})
            if not ok_secret:
                return "I couldn't save that key. Paste it again or reply cancel."
            ok_test, body_test = runtime.test_provider("openrouter", {})
            if not ok_test:
                reason = str((body_test or {}).get("message") or (body_test or {}).get("error") or "test_failed").strip()
                return f"Key saved, but OpenRouter test failed ({reason}). Paste another key or reply cancel."
            _save_setup_state(
                wizard_store,
                chat_id=chat_id,
                flow=flow,
                step="awaiting_openrouter_default",
                question="OpenRouter is working. Set it as default now? Reply 1 for Yes or 2 for No.",
                issue_code=issue_code,
                choices=[
                    {"id": "set_default_yes", "label": "Yes", "recommended": True},
                    {"id": "set_default_no", "label": "No", "recommended": False},
                ],
                pending={},
            )
            return "OpenRouter is working. Set it as default now? Reply 1 for Yes or 2 for No."

        if step == "awaiting_openrouter_default":
            if normalized in _WIZARD_YES_TOKENS:
                payload: dict[str, Any] = {"default_provider": "openrouter"}
                default_model = _choose_openrouter_default_model(runtime=runtime)
                if default_model:
                    payload["default_model"] = default_model
                ok_defaults, body_defaults = runtime.update_defaults(payload)
                wizard_store.clear()
                if ok_defaults:
                    model_text = f" / {default_model}" if default_model else ""
                    return f"OpenRouter configured. Default set to openrouter{model_text}."
                reason = str((body_defaults or {}).get("error") or "update_defaults_failed").strip()
                return f"OpenRouter is configured, but I couldn't set defaults ({reason})."
            if normalized in _WIZARD_NO_TOKENS:
                wizard_store.clear()
                return "OpenRouter is configured. Defaults unchanged."
            return "Reply 1 for Yes or 2 for No."

    if flow == "ollama_setup":
        if step == "awaiting_ollama_ready":
            if normalized in _WIZARD_YES_TOKENS:
                if not _is_ollama_reachable(runtime=runtime):
                    return "I still can't reach Ollama. Run `ollama list`, then reply READY."
                _save_setup_state(
                    wizard_store,
                    chat_id=chat_id,
                    flow=flow,
                    step="awaiting_ollama_size",
                    question="Which Ollama size tier do you want? Reply 1 small, 2 medium, or 3 large.",
                    issue_code=issue_code,
                    choices=list(_OLLAMA_TIER_CHOICES),
                    pending={},
                )
                return "Which Ollama size tier do you want? Reply 1 small, 2 medium, or 3 large."
            return "Run `ollama list`, then reply READY."

        if step == "awaiting_ollama_size":
            choice_id = _choice_id_from_text(text=text, choices=list(_OLLAMA_TIER_CHOICES))
            if choice_id is None:
                return "Reply 1 for small, 2 for medium, or 3 for large."
            candidate_model = _choose_ollama_model_for_tier(runtime=runtime, tier=choice_id)
            if not candidate_model:
                _save_setup_state(
                    wizard_store,
                    chat_id=chat_id,
                    flow=flow,
                    step="awaiting_ollama_pull",
                    question="No local Ollama models found. Run `ollama pull qwen2.5:3b-instruct`, then reply READY.",
                    issue_code=issue_code,
                    choices=[],
                    pending={"tier": choice_id},
                )
                return "No local Ollama models found. Run `ollama pull qwen2.5:3b-instruct`, then reply READY."
            _save_setup_state(
                wizard_store,
                chat_id=chat_id,
                flow=flow,
                step="awaiting_ollama_default",
                question=f"Use {candidate_model} as default now? Reply 1 for Yes or 2 for No.",
                issue_code=issue_code,
                choices=[
                    {"id": "set_default_yes", "label": "Yes", "recommended": True},
                    {"id": "set_default_no", "label": "No", "recommended": False},
                ],
                pending={"tier": choice_id, "model": candidate_model},
            )
            return f"Use {candidate_model} as default now? Reply 1 for Yes or 2 for No."

        if step == "awaiting_ollama_pull":
            if normalized in _WIZARD_YES_TOKENS:
                tier = str(pending.get("tier") or "small").strip().lower() or "small"
                candidate_model = _choose_ollama_model_for_tier(runtime=runtime, tier=tier)
                if not candidate_model:
                    return "I still don't see local models. Run `ollama pull qwen2.5:3b-instruct`, then reply READY."
                _save_setup_state(
                    wizard_store,
                    chat_id=chat_id,
                    flow=flow,
                    step="awaiting_ollama_default",
                    question=f"Use {candidate_model} as default now? Reply 1 for Yes or 2 for No.",
                    issue_code=issue_code,
                    choices=[
                        {"id": "set_default_yes", "label": "Yes", "recommended": True},
                        {"id": "set_default_no", "label": "No", "recommended": False},
                    ],
                    pending={"tier": tier, "model": candidate_model},
                )
                return f"Use {candidate_model} as default now? Reply 1 for Yes or 2 for No."
            return "Run `ollama pull qwen2.5:3b-instruct`, then reply READY."

        if step == "awaiting_ollama_default":
            if normalized in _WIZARD_YES_TOKENS:
                model_id = str(pending.get("model") or "").strip() or _choose_ollama_model_for_tier(
                    runtime=runtime,
                    tier=str(pending.get("tier") or "small").strip().lower() or "small",
                )
                payload: dict[str, Any] = {"default_provider": "ollama"}
                if model_id:
                    payload["default_model"] = model_id
                ok_defaults, body_defaults = runtime.update_defaults(payload)
                wizard_store.clear()
                if ok_defaults:
                    model_text = f" / {model_id}" if model_id else ""
                    return f"Ollama configured. Default set to ollama{model_text}."
                reason = str((body_defaults or {}).get("error") or "update_defaults_failed").strip()
                return f"Ollama is reachable, but I couldn't set defaults ({reason})."
            if normalized in _WIZARD_NO_TOKENS:
                wizard_store.clear()
                return "Ollama setup complete. Defaults unchanged."
            return "Reply 1 for Yes or 2 for No."

    wizard_store.clear()
    return _NO_ACTIVE_CHOICE_TEXT


def _handle_model_provider_intent(
    *,
    intent: str,
    bot_data: dict[str, Any],
    chat_id: str,
) -> tuple[str | None, str | None]:
    if intent not in _MODEL_PROVIDER_INTENTS or intent == "none":
        return None, None
    runtime = bot_data.get("runtime")
    wizard_store = bot_data.get("model_provider_wizard_store")
    setup_store = wizard_store if isinstance(wizard_store, TelegramModelProviderWizardStore) else None
    if intent == "provider.status":
        return _model_status_report(runtime=runtime), "status"
    if intent == "provider.help":
        return _MODEL_PROVIDER_HELP_TEXT, "help"
    if intent == "model_watch.run_now":
        return _model_watch_run_summary(runtime=runtime), "chat"
    if intent == "provider.setup.openrouter":
        if setup_store is None:
            return "Provider setup wizard is unavailable right now.", "fallback"
        return _start_openrouter_setup(runtime=runtime, wizard_store=setup_store, chat_id=chat_id), "chat"
    if intent == "provider.setup.ollama":
        if setup_store is None:
            return "Provider setup wizard is unavailable right now.", "fallback"
        return _start_ollama_setup(runtime=runtime, wizard_store=setup_store, chat_id=chat_id), "chat"
    return None, None


def _is_unknown_orchestrator_reply(text: str) -> bool:
    normalized = _normalize_user_text(text)
    return normalized.startswith("i didn’t understand that") or normalized.startswith("i didn't understand that")


def _trace_id_from_update(update: Update, *, fallback_prefix: str = "tg") -> str:
    chat_id = (
        str(getattr(getattr(update, "effective_chat", None), "id", "") or "").strip()
        if update is not None
        else ""
    )
    message_id = (
        str(getattr(getattr(update, "effective_message", None), "message_id", "") or "").strip()
        if update is not None
        else ""
    )
    if chat_id and message_id:
        return f"{fallback_prefix}-{chat_id}-{message_id}"
    if chat_id:
        return f"{fallback_prefix}-{chat_id}"
    return f"{fallback_prefix}-unknown"


def _envelope_from_exception(
    *,
    exc: Exception,
    intent: str,
    trace_id: str,
    log_path: str | None,
) -> dict[str, object]:
    def _raiser() -> dict[str, object]:
        raise exc

    return run_with_fallback(
        fn=_raiser,
        context={
            "intent": intent,
            "trace_id": trace_id,
            "log_path": log_path,
            "actions": [],
        },
    )


def _redact_chat_id(chat_id: str) -> str:
    value = str(chat_id or "").strip()
    if not value:
        return "unknown"
    if len(value) <= 4:
        return "***"
    return f"***{value[-4:]}"


def _safe_append_telegram_message_audit(
    *,
    audit_log: AuditLog | None,
    action: str,
    chat_id: str,
    message_kind: str,
    route: str,
    outcome: str,
    error_kind: str | None = None,
) -> None:
    if audit_log is None:
        return
    try:
        audit_log.append(
            actor="telegram",
            action=action,
            params={
                "chat_id_redacted": _redact_chat_id(chat_id),
                "message_kind": str(message_kind or "text"),
                "route": str(route or "ignored"),
            },
            decision="allow",
            reason=f"{route}:{outcome}",
            dry_run=False,
            outcome=str(outcome or "handled"),
            error_kind=str(error_kind or "") or None,
            duration_ms=0,
        )
    except Exception:
        pass


def _wizard_status_from_state(state: dict[str, Any]) -> str:
    if not isinstance(state, dict):
        return "idle"
    if not bool(state.get("active", False)):
        return "idle"
    step = str(state.get("step") or "").strip().lower()
    if step == "awaiting_choice":
        return "needs_user_choice"
    if step == "awaiting_confirm":
        return "needs_confirmation"
    return "idle"


def _map_fixit_reply_to_payload(state: dict[str, Any], text: str) -> tuple[dict[str, Any] | None, str | None]:
    status = _wizard_status_from_state(state)
    if status == "idle":
        return None, None
    normalized = " ".join(str(text or "").strip().lower().split())
    if not normalized:
        if status == "needs_user_choice":
            return None, "Reply 1, 2, or 3."
        return None, "Reply YES (1) or NO (2)."

    if status == "needs_user_choice":
        raw_choices = state.get("choices") if isinstance(state.get("choices"), list) else []
        choices: list[dict[str, str]] = []
        for row in raw_choices:
            if not isinstance(row, dict):
                continue
            choice_id = str(row.get("id") or "").strip()
            label = str(row.get("label") or "").strip()
            if not choice_id:
                continue
            choices.append({"id": choice_id, "label": label})
        if not choices:
            return None, "No pending choices right now. Ask me to run fix-it again."
        if normalized.isdigit():
            idx = int(normalized)
            if 1 <= idx <= len(choices):
                return {"answer": choices[idx - 1]["id"]}, None
        for choice in choices:
            choice_id = str(choice.get("id") or "").strip().lower()
            label = str(choice.get("label") or "").strip().lower()
            if normalized == choice_id or (label and normalized == label):
                return {"answer": choice.get("id")}, None
        return None, "Reply 1, 2, or 3."

    if status == "needs_confirmation":
        if normalized in {"1", "yes", "y", "apply", "ok"}:
            return {"confirm": True}, None
        if normalized in {"2", "no", "n", "cancel"}:
            return {"confirm": False}, None
        return None, "Reply YES (1) or NO (2)."

    return None, None


def maybe_handle_llm_fixit_reply(
    *,
    llm_fixit_fn: Callable[[dict[str, Any]], tuple[bool, dict[str, Any]]] | None,
    wizard_store: LLMFixitWizardStore | None,
    audit_log: AuditLog | None,
    chat_id: str,
    text: str,
    log_path: str | None,
) -> str | None:
    if llm_fixit_fn is None or wizard_store is None:
        return None
    try:
        state = wizard_store.load()
        wizard_store.state = state
    except Exception:
        state = wizard_store.empty_state()
    status = _wizard_status_from_state(state)
    payload, hint_message = _map_fixit_reply_to_payload(state, text)
    if payload is None:
        if hint_message is None:
            return None
        if audit_log is not None:
            try:
                audit_log.append(
                    actor="telegram",
                    action="llm.fixit.telegram",
                    params={
                        "chat_id_redacted": _redact_chat_id(chat_id),
                        "status": status,
                        "mapped": False,
                    },
                    decision="allow",
                    reason="reply_not_mapped",
                    dry_run=False,
                    outcome="handled",
                    error_kind="needs_clarification",
                    duration_ms=0,
                )
            except Exception:
                pass
        return hint_message

    payload = dict(payload)
    payload["actor"] = "telegram"
    ok, body = llm_fixit_fn(payload)
    message = str(body.get("message") or body.get("next_question") or "").strip()
    if not message:
        message = "I processed that fix-it step."
    if audit_log is not None:
        try:
            audit_log.append(
                actor="telegram",
                action="llm.fixit.telegram",
                params={
                    "chat_id_redacted": _redact_chat_id(chat_id),
                    "status": status,
                    "mapped": True,
                    "answer": payload.get("answer"),
                    "confirm": payload.get("confirm"),
                },
                decision="allow",
                reason="fixit_reply",
                dry_run=False,
                outcome="success" if ok else "failed",
                error_kind=str(body.get("error_kind") or "") or None,
                duration_ms=0,
            )
        except Exception:
            pass
    if log_path:
        log_event(
            log_path,
            "llm_fixit_telegram",
            {
                "chat_id_redacted": _redact_chat_id(chat_id),
                "status": status,
                "mapped": True,
                "ok": bool(ok),
                "error_kind": str(body.get("error_kind") or "") or None,
            },
        )
    return message


async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    audit_log_value = context.application.bot_data.get("audit_log")
    audit_log = audit_log_value if isinstance(audit_log_value, AuditLog) else None

    if update.effective_chat is None or update.effective_message is None:
        _safe_append_telegram_message_audit(
            audit_log=audit_log,
            action="telegram.message.ignored",
            chat_id="",
            message_kind="non_text",
            route="ignored",
            outcome="ignored",
        )
        return

    chat_id = str(update.effective_chat.id)
    text = (update.effective_message.text or "").strip()
    if not text:
        _safe_append_telegram_message_audit(
            audit_log=audit_log,
            action="telegram.message.ignored",
            chat_id=chat_id,
            message_kind="text",
            route="ignored",
            outcome="ignored",
        )
        return

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    db: MemoryDB = context.application.bot_data["db"]
    log_path: str = context.application.bot_data["log_path"]
    bot_data = context.application.bot_data
    trace_id = _trace_id_from_update(update)
    llm_fixit_fn = bot_data.get("llm_fixit_fn")
    wizard_store = bot_data.get("llm_fixit_store")
    model_provider_wizard_store = bot_data.get("model_provider_wizard_store")
    _safe_append_telegram_message_audit(
        audit_log=audit_log,
        action="telegram.message.received",
        chat_id=chat_id,
        message_kind="text",
        route="chat",
        outcome="received",
    )

    try:
        # Remember which chat we're talking to (useful for reminders/jobs).
        db.set_preference("telegram_chat_id", chat_id)

        fixit_reply = maybe_handle_llm_fixit_reply(
            llm_fixit_fn=llm_fixit_fn if callable(llm_fixit_fn) else None,
            wizard_store=wizard_store if isinstance(wizard_store, LLMFixitWizardStore) else None,
            audit_log=audit_log if isinstance(audit_log, AuditLog) else None,
            chat_id=chat_id,
            text=text,
            log_path=log_path,
        )
        if fixit_reply is not None:
            _safe_append_telegram_message_audit(
                audit_log=audit_log,
                action="telegram.message.handled",
                chat_id=chat_id,
                message_kind="text",
                route="fixit",
                outcome="handled",
            )
            await update.effective_message.reply_text(_safe_reply_text(fixit_reply))
            log_event(
                log_path,
                "telegram_fixit_intercept",
                {
                    "chat_id_redacted": _redact_chat_id(chat_id),
                    "route": "fixit",
                    "trace_id": trace_id,
                },
            )
            return

        setup_reply = _setup_wizard_route_reply(
            runtime=bot_data.get("runtime"),
            wizard_store=(
                model_provider_wizard_store
                if isinstance(model_provider_wizard_store, TelegramModelProviderWizardStore)
                else None
            ),
            llm_fixit_fn=llm_fixit_fn if callable(llm_fixit_fn) else None,
            orchestrator=orchestrator if isinstance(orchestrator, Orchestrator) else None,
            chat_id=chat_id,
            text=text,
        )
        if setup_reply is not None:
            _safe_append_telegram_message_audit(
                audit_log=audit_log,
                action="telegram.message.handled",
                chat_id=chat_id,
                message_kind="text",
                route="fixit",
                outcome="handled",
            )
            await update.effective_message.reply_text(_safe_reply_text(setup_reply))
            log_event(
                log_path,
                "telegram_fixit_intercept",
                {
                    "chat_id_redacted": _redact_chat_id(chat_id),
                    "route": "fixit",
                    "trace_id": trace_id,
                },
            )
            return

        normalized_text = _normalize_user_text(text)
        if normalized_text in _WIZARD_CHOICE_TOKENS:
            _safe_append_telegram_message_audit(
                audit_log=audit_log,
                action="telegram.message.handled",
                chat_id=chat_id,
                message_kind="text",
                route="fallback",
                outcome="handled",
            )
            await update.effective_message.reply_text(_NO_ACTIVE_CHOICE_TEXT)
            log_event(
                log_path,
                "telegram_message",
                {
                    "chat_id_redacted": _redact_chat_id(chat_id),
                    "route": "fallback",
                    "trace_id": trace_id,
                },
            )
            return

        model_provider_intent = classify_model_provider_intent(text)
        preroute_reply, preroute_route = _handle_model_provider_intent(
            intent=model_provider_intent,
            bot_data=bot_data,
            chat_id=chat_id,
        )
        if preroute_reply is not None and preroute_route is not None:
            _safe_append_telegram_message_audit(
                audit_log=audit_log,
                action="telegram.message.handled",
                chat_id=chat_id,
                message_kind="text",
                route=preroute_route,
                outcome="handled",
            )
            await update.effective_message.reply_text(_safe_reply_text(preroute_reply))
            log_event(
                log_path,
                "telegram_message",
                {
                    "chat_id_redacted": _redact_chat_id(chat_id),
                    "route": preroute_route,
                    "trace_id": trace_id,
                    "model_provider_intent": model_provider_intent,
                },
            )
            return

        smalltalk_reply, smalltalk_route = _smalltalk_preroute_reply(text, bot_data)
        if smalltalk_reply is not None and smalltalk_route is not None:
            _safe_append_telegram_message_audit(
                audit_log=audit_log,
                action="telegram.message.handled",
                chat_id=chat_id,
                message_kind="text",
                route=smalltalk_route,
                outcome="handled",
            )
            await update.effective_message.reply_text(_safe_reply_text(smalltalk_reply))
            log_event(
                log_path,
                "telegram_message",
                {
                    "chat_id_redacted": _redact_chat_id(chat_id),
                    "route": smalltalk_route,
                    "trace_id": trace_id,
                },
            )
            return

        ambiguity = classify_ambiguity(text)
        if ambiguity.ambiguous:
            llm_available, availability_reason = _runtime_llm_availability(bot_data)
            if llm_available:
                clarify_text = build_clarify_message(text)
                _safe_append_telegram_message_audit(
                    audit_log=audit_log,
                    action="telegram.message.handled",
                    chat_id=chat_id,
                    message_kind="text",
                    route="chat",
                    outcome="handled",
                )
                await update.effective_message.reply_text(_safe_reply_text(clarify_text))
                log_event(
                    log_path,
                    "telegram_message",
                    {
                        "chat_id_redacted": _redact_chat_id(chat_id),
                        "route": "chat",
                        "trace_id": trace_id,
                        "clarify_suggest_mode": "clarify",
                    },
                )
                return
            _set_recovery_wizard_state(
                wizard_store=(
                    model_provider_wizard_store
                    if isinstance(model_provider_wizard_store, TelegramModelProviderWizardStore)
                    else None
                ),
                chat_id=chat_id,
                reason=availability_reason,
            )
            suggest_text = build_suggest_message(availability_reason=availability_reason)
            _safe_append_telegram_message_audit(
                audit_log=audit_log,
                action="telegram.message.handled",
                chat_id=chat_id,
                message_kind="text",
                route="fallback",
                outcome="handled",
            )
            await update.effective_message.reply_text(_safe_reply_text(suggest_text))
            log_event(
                log_path,
                "telegram_message",
                {
                    "chat_id_redacted": _redact_chat_id(chat_id),
                    "route": "fallback",
                    "trace_id": trace_id,
                    "clarify_suggest_mode": "suggest",
                    "availability_reason": availability_reason,
                },
            )
            return

        # Route ALL non-command text through the orchestrator (single brain).
        response = orchestrator.handle_message(text, user_id=chat_id)
        reply_text = response.text.strip() if response and response.text else ""
        parse_mode = None
        route = "chat"
        if response and isinstance(response.data, dict):
            ok, _ = validate_cards_payload(response.data)
            if ok:
                reply_text = render_cards_markdown(response.data)
                parse_mode = "Markdown"
                route = "chat"
        if not reply_text or _is_unknown_orchestrator_reply(reply_text):
            reply_text = _TELEGRAM_UNKNOWN_FALLBACK_TEXT
            parse_mode = None
            route = "fallback"
        await update.effective_message.reply_text(_safe_reply_text(reply_text), parse_mode=parse_mode)
        _safe_append_telegram_message_audit(
            audit_log=audit_log,
            action="telegram.message.handled",
            chat_id=chat_id,
            message_kind="text",
            route=route,
            outcome="handled",
        )

        log_event(
            log_path,
            "telegram_message",
            {
                "chat_id_redacted": _redact_chat_id(chat_id),
                "route": route,
                "trace_id": trace_id,
            },
        )
    except Exception as exc:
        _safe_append_telegram_message_audit(
            audit_log=audit_log,
            action="telegram.message.handled",
            chat_id=chat_id,
            message_kind="text",
            route="chat",
            outcome="failed",
            error_kind=exc.__class__.__name__,
        )
        envelope = _envelope_from_exception(
            exc=exc,
            intent="telegram.message",
            trace_id=trace_id,
            log_path=log_path,
        )
        message = _safe_reply_text(str(envelope.get("message") or _TELEGRAM_FALLBACK_TEXT))
        await update.effective_message.reply_text(message)


def _command_payload(text: str, command: str) -> str:
    if not text:
        return ""
    parts = text.split(maxsplit=1)
    if not parts:
        return ""
    token = parts[0]
    if token.startswith(command):
        return parts[1] if len(parts) > 1 else ""
    return ""


async def _handle_remind(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    text = update.effective_message.text or ""
    content = _command_payload(text, "/remind")
    prompt = f"remind me {content}".strip()

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    log_path: str = context.application.bot_data["log_path"]

    response = orchestrator.handle_message(prompt, user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))
    log_event(log_path, "telegram_command", {"chat_id": chat_id, "text": text, "forwarded": prompt})


async def _handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]

    response = orchestrator.handle_message("/status", user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_runtime_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]

    response = orchestrator.handle_message("/runtime_status", user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_disk_grow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    text = update.effective_message.text or ""
    content = _command_payload(text, "/disk_grow")
    prompt = "/disk_grow {}".format(content).strip()

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message(prompt, user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_audit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    audit_log: AuditLog = context.application.bot_data["audit_log"]
    rows = audit_log.recent(limit=5)
    if not rows:
        await update.effective_message.reply_text("No ModelOps audit events yet.")
        return
    lines = ["Recent ModelOps audit events:"]
    for row in rows:
        lines.append(
            f"- {row.get('ts')} {row.get('action')} "
            f"[{row.get('decision')}/{row.get('outcome')}] reason={row.get('reason')}"
        )
    await update.effective_message.reply_text("\n".join(lines))


async def _handle_permissions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    permission_store: PermissionStore = context.application.bot_data["permission_store"]
    permissions = permission_store.load()
    mode = permissions.get("mode") or "manual_confirm"
    actions = permissions.get("actions") if isinstance(permissions.get("actions"), dict) else {}
    constraints = permissions.get("constraints") if isinstance(permissions.get("constraints"), dict) else {}

    lines = [f"ModelOps permissions mode: {mode}", "Actions:"]
    for action_name in sorted(actions.keys()):
        lines.append(f"- {action_name}: {'allow' if bool(actions[action_name]) else 'deny'}")
    lines.append("Constraints:")
    lines.append(f"- max_download_gb: {constraints.get('max_download_gb')}")
    lines.append(f"- allow_install_ollama: {constraints.get('allow_install_ollama')}")
    lines.append(f"- allow_remote_models: {constraints.get('allow_remote_models')}")
    lines.append(f"- allowed_providers: {', '.join(constraints.get('allowed_providers') or [])}")
    await update.effective_message.reply_text("\n".join(lines))


async def _handle_storage_snapshot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]

    response = orchestrator.handle_message("/storage_snapshot", user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_storage_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]

    response = orchestrator.handle_message("/storage_report", user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_resource_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]

    response = orchestrator.handle_message("/resource_report", user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_brief(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]

    response = orchestrator.handle_message("/brief", user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_brief_alias(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    audit_log_value = context.application.bot_data.get("audit_log")
    audit_log = audit_log_value if isinstance(audit_log_value, AuditLog) else None
    _safe_append_telegram_message_audit(
        audit_log=audit_log,
        action="telegram.message.handled",
        chat_id=chat_id,
        message_kind="command",
        route="alias",
        outcome="handled",
    )
    await _handle_brief(update, context)


async def _handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    audit_log_value = context.application.bot_data.get("audit_log")
    audit_log = audit_log_value if isinstance(audit_log_value, AuditLog) else None
    _safe_append_telegram_message_audit(
        audit_log=audit_log,
        action="telegram.message.handled",
        chat_id=chat_id,
        message_kind="command",
        route="help",
        outcome="handled",
    )
    await update.effective_message.reply_text(_TELEGRAM_HELP_TEXT)


async def _handle_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    audit_log_value = context.application.bot_data.get("audit_log")
    audit_log = audit_log_value if isinstance(audit_log_value, AuditLog) else None
    _safe_append_telegram_message_audit(
        audit_log=audit_log,
        action="telegram.message.handled",
        chat_id=chat_id,
        message_kind="command",
        route="status",
        outcome="handled",
    )
    runtime = context.application.bot_data.get("runtime")
    await update.effective_message.reply_text(_safe_reply_text(_model_status_report(runtime=runtime)))


async def _handle_network_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]

    response = orchestrator.handle_message("/network_report", user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_sys_metrics_snapshot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message("/sys_metrics_snapshot", user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_sys_health_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message("/sys_health_report", user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_sys_inventory_summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message("/sys_inventory_summary", user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_weekly_reflection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]

    response = orchestrator.handle_message("/weekly_reflection", user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message("/today", user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_task_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    text = update.effective_message.text or ""
    content = _command_payload(text, "/task_add")
    prompt = f"/task_add {content}".strip()
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message(prompt, user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_done(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    text = update.effective_message.text or ""
    content = _command_payload(text, "/done")
    prompt = f"/done {content}".strip()
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message(prompt, user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_open_loops(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    text = update.effective_message.text or ""
    content = _command_payload(text, "/open_loops")
    prompt = f"/open_loops {content}".strip()
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message(prompt, user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_health(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message("/health", user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_daily_brief_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message("/daily_brief_status", user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    text = update.effective_message.text or ""
    content = _command_payload(text, "/ask")
    prompt = "/ask {}".format(content).strip()

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message(prompt, user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_ask_opinion(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    text = update.effective_message.text or ""
    content = _command_payload(text, "/ask_opinion")
    prompt = "/ask_opinion {}".format(content).strip()

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message(prompt, user_id=chat_id)
    await update.effective_message.reply_text(_safe_reply_text(response.text))


async def _handle_scout(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    model_scout = context.application.bot_data.get("model_scout")
    if model_scout is None:
        await update.effective_message.reply_text("Model Scout is unavailable in this runtime.")
        return

    suggestions = model_scout.list_suggestions(status="new", limit=5)
    message = model_scout.format_scout_details(suggestions, limit=5)
    await update.effective_message.reply_text(message)


async def _handle_scout_dismiss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    model_scout = context.application.bot_data.get("model_scout")
    if model_scout is None:
        await update.effective_message.reply_text("Model Scout is unavailable in this runtime.")
        return

    text = update.effective_message.text or ""
    suggestion_id = _command_payload(text, "/scout_dismiss").strip()
    if not suggestion_id:
        await update.effective_message.reply_text("Usage: /scout_dismiss <suggestion_id>")
        return

    if model_scout.dismiss(suggestion_id):
        await update.effective_message.reply_text(f"Dismissed suggestion {suggestion_id}.")
    else:
        await update.effective_message.reply_text(f"Suggestion not found: {suggestion_id}")


async def _handle_scout_installed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    model_scout = context.application.bot_data.get("model_scout")
    if model_scout is None:
        await update.effective_message.reply_text("Model Scout is unavailable in this runtime.")
        return

    text = update.effective_message.text or ""
    suggestion_id = _command_payload(text, "/scout_installed").strip()
    if not suggestion_id:
        await update.effective_message.reply_text("Usage: /scout_installed <suggestion_id>")
        return

    if model_scout.mark_installed(suggestion_id):
        await update.effective_message.reply_text(f"Marked suggestion as installed: {suggestion_id}.")
    else:
        await update.effective_message.reply_text(f"Suggestion not found: {suggestion_id}")


async def _on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    import logging

    logger = logging.getLogger(__name__)
    error = context.error
    try:
        from telegram.error import Conflict
    except Exception:  # pragma: no cover - defensive import
        Conflict = None
    if Conflict is not None and isinstance(error, Conflict):
        logger.error(
            "Telegram polling conflict detected. Another instance may be running or getUpdates is active elsewhere.",
            exc_info=error,
        )
        return
    logger.exception("Telegram handler error", exc_info=error)
    message = getattr(update, "effective_message", None)
    if message is None:
        return
    trace_id = _trace_id_from_update(update) if isinstance(update, Update) else "tg-error"
    bot_data = getattr(getattr(context, "application", None), "bot_data", {}) or {}
    log_path = str(bot_data.get("log_path") or "").strip() or None
    envelope = _envelope_from_exception(
        exc=error if isinstance(error, Exception) else RuntimeError("TelegramHandlerError"),
        intent="telegram.error_handler",
        trace_id=trace_id,
        log_path=log_path,
    )
    try:
        await message.reply_text(_safe_reply_text(str(envelope.get("message") or _TELEGRAM_FALLBACK_TEXT)))
    except Exception:
        return


async def _check_reminders(context: ContextTypes.DEFAULT_TYPE) -> None:
    db: MemoryDB = context.application.bot_data["db"]
    log_path: str = context.application.bot_data["log_path"]
    debug_protocol: DebugProtocol | None = context.application.bot_data.get("debug_protocol")
    orchestrator: Orchestrator | None = context.application.bot_data.get("orchestrator")

    chat_id = db.get_preference("telegram_chat_id")
    if not chat_id:
        return

    now_ts = datetime.now(timezone.utc).isoformat()
    reminders = db.list_due_reminders(now_ts)
    for reminder in reminders:
        try:
            audit_id = db.audit_log_create(
                user_id=str(chat_id),
                action_type="reminder_send",
                action_id=str(reminder.id),
                status="attempted",
                details={
                    "event_type": "reminder_send",
                    "reminder_id": reminder.id,
                    "claim_attempted": True,
                    "claim_succeeded": False,
                    "send_succeeded": False,
                },
            )
        except Exception:
            return

        claim_ok = db.claim_reminder_sent(reminder.id, now_ts)
        if not claim_ok:
            try:
                db.audit_log_update_status(
                    audit_id,
                    "skipped",
                    details={
                        "event_type": "reminder_send",
                        "reminder_id": reminder.id,
                        "claim_attempted": True,
                        "claim_succeeded": False,
                        "send_succeeded": False,
                        "status_transition": "pending->skipped",
                    },
                )
            except Exception:
                return
            continue

        if debug_protocol and orchestrator:
            if debug_protocol.record_reminder(
                str(chat_id),
                reminder.text,
                datetime.now(timezone.utc),
            ):
                response = orchestrator.handle_message("/runtime_status", user_id=str(chat_id))
                await context.bot.send_message(chat_id=chat_id, text=response.text)
                db.mark_reminder_failed(reminder.id, "debug_protocol_triggered")
                try:
                    db.audit_log_update_status(
                        audit_id,
                        "failed",
                        details={
                            "event_type": "reminder_send",
                            "reminder_id": reminder.id,
                            "claim_attempted": True,
                            "claim_succeeded": True,
                            "send_succeeded": False,
                            "status_transition": "sent->failed",
                            "error": "debug_protocol_triggered",
                        },
                    )
                except Exception:
                    return
                continue

        try:
            await context.bot.send_message(chat_id=chat_id, text=f"Reminder: {reminder.text}")
            try:
                db.audit_log_update_status(
                    audit_id,
                    "executed",
                    details={
                        "event_type": "reminder_send",
                        "reminder_id": reminder.id,
                        "claim_attempted": True,
                        "claim_succeeded": True,
                        "send_succeeded": True,
                        "status_transition": "pending->sent",
                    },
                )
            except Exception:
                return
            log_event(log_path, "reminder_sent", {"reminder_id": reminder.id})
        except Exception as exc:
            db.mark_reminder_failed(reminder.id, str(exc))
            try:
                db.audit_log_update_status(
                    audit_id,
                    "failed",
                    details={
                        "event_type": "reminder_send",
                        "reminder_id": reminder.id,
                        "claim_attempted": True,
                        "claim_succeeded": True,
                        "send_succeeded": False,
                        "status_transition": "sent->failed",
                        "error": str(exc),
                    },
                )
            except Exception:
                return
            if debug_protocol and orchestrator:
                if debug_protocol.record_audit_event(
                    "reminder_send",
                    str(reminder.id),
                    "failed",
                    datetime.now(timezone.utc),
                ):
                    response = orchestrator.handle_message("/runtime_status", user_id=str(chat_id))
                    await context.bot.send_message(chat_id=chat_id, text=response.text)


async def _scheduled_disk_snapshot(context: ContextTypes.DEFAULT_TYPE) -> None:
    db: MemoryDB = context.application.bot_data["db"]
    home_path: str = context.application.bot_data["home_path"]
    safe_run_scheduled_snapshot(db, home_path, "/")


async def _scheduled_storage_snapshot(context: ContextTypes.DEFAULT_TYPE) -> None:
    db: MemoryDB = context.application.bot_data["db"]
    home_path: str = context.application.bot_data["home_path"]
    timezone: str = context.application.bot_data["timezone"]
    user_id = db.get_preference("telegram_chat_id") or "system"
    safe_run_storage_snapshot(db, timezone, home_path, user_id)


async def _scheduled_resource_snapshot(context: ContextTypes.DEFAULT_TYPE) -> None:
    db: MemoryDB = context.application.bot_data["db"]
    timezone: str = context.application.bot_data["timezone"]
    user_id = db.get_preference("telegram_chat_id") or "system"
    safe_run_resource_snapshot(db, timezone, user_id)


async def _scheduled_network_snapshot(context: ContextTypes.DEFAULT_TYPE) -> None:
    db: MemoryDB = context.application.bot_data["db"]
    timezone: str = context.application.bot_data["timezone"]
    user_id = db.get_preference("telegram_chat_id") or "system"
    safe_run_network_snapshot(db, timezone, user_id)


async def _scheduled_daily_brief(context: ContextTypes.DEFAULT_TYPE) -> None:
    db: MemoryDB = context.application.bot_data["db"]
    timezone_name: str = context.application.bot_data["timezone"]
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    chat_id = db.get_preference("telegram_chat_id")
    if not chat_id:
        return
    enabled = (db.get_preference("daily_brief_enabled") or "off").strip().lower() in {"on", "true", "1", "yes"}
    local_time = (db.get_preference("daily_brief_time") or "09:00").strip()
    quiet_mode = (db.get_preference("daily_brief_quiet_mode") or "off").strip().lower() in {"on", "true", "1", "yes"}
    threshold_pref = db.get_preference("disk_delta_threshold_mb")
    disk_delta_threshold_mb = float(threshold_pref) if (threshold_pref and threshold_pref.isdigit()) else 250.0
    svc_gate = (db.get_preference("only_send_if_service_unhealthy") or "off").strip().lower() in {
        "on",
        "true",
        "1",
        "yes",
    }
    include_due_pref = db.get_preference("include_open_loops_due_within_days")
    include_due_days = int(include_due_pref) if (include_due_pref and include_due_pref.isdigit()) else 2
    last_sent = db.get_preference("daily_brief_last_sent_date")
    payload = orchestrator.build_daily_brief_cards(str(chat_id))
    signals = payload.get("daily_brief_signals") if isinstance(payload, dict) else {}
    disk_delta_mb = signals.get("disk_delta_mb") if isinstance(signals, dict) else None
    service_unhealthy = bool(signals.get("service_unhealthy")) if isinstance(signals, dict) else False
    due_open_loops = int(signals.get("due_open_loops_count") or 0) if isinstance(signals, dict) else 0
    decision = should_send_daily_brief(
        now_utc=datetime.now(timezone.utc),
        timezone_name=timezone_name,
        enabled=enabled,
        local_time_hhmm=local_time,
        last_sent_local_date=last_sent,
        quiet_mode=quiet_mode,
        disk_delta_mb=disk_delta_mb,
        disk_delta_threshold_mb=disk_delta_threshold_mb,
        service_unhealthy=service_unhealthy,
        only_send_if_service_unhealthy=svc_gate,
        has_due_open_loops=due_open_loops > 0 and include_due_days > 0,
    )
    if not decision.should_send:
        return
    text = render_cards_markdown(payload)
    send_ok = False
    for attempt in (1, 2):
        try:
            await context.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
            send_ok = True
            break
        except Exception:
            if attempt == 1:
                await asyncio.sleep(1.0)
                continue
            return
    if send_ok:
        db.set_preference("daily_brief_last_sent_date", decision.local_date)


def register_handlers(app: Application) -> None:
    # Log exceptions to journalctl instead of swallowing them.
    app.add_error_handler(_on_error)

    # Explicit command handlers (commands should NOT go through the generic text handler).
    app.add_handler(CommandHandler("remind", _handle_remind))
    app.add_handler(CommandHandler("status", _handle_status))
    app.add_handler(CommandHandler("runtime_status", _handle_runtime_status))
    app.add_handler(CommandHandler("disk_grow", _handle_disk_grow))
    app.add_handler(CommandHandler("audit", _handle_audit))
    app.add_handler(CommandHandler("permissions", _handle_permissions))
    app.add_handler(CommandHandler("storage_snapshot", _handle_storage_snapshot))
    app.add_handler(CommandHandler("storage_report", _handle_storage_report))
    app.add_handler(CommandHandler("resource_report", _handle_resource_report))
    app.add_handler(CommandHandler("brief", _handle_brief))
    app.add_handler(CommandHandler("breif", _handle_brief_alias))
    app.add_handler(CommandHandler("help", _handle_help))
    app.add_handler(CommandHandler("model", _handle_model))
    app.add_handler(CommandHandler("network_report", _handle_network_report))
    app.add_handler(CommandHandler("sys_metrics_snapshot", _handle_sys_metrics_snapshot))
    app.add_handler(CommandHandler("sys_health_report", _handle_sys_health_report))
    app.add_handler(CommandHandler("sys_inventory_summary", _handle_sys_inventory_summary))
    app.add_handler(CommandHandler("weekly_reflection", _handle_weekly_reflection))
    app.add_handler(CommandHandler("today", _handle_today))
    app.add_handler(CommandHandler("task_add", _handle_task_add))
    app.add_handler(CommandHandler("done", _handle_done))
    app.add_handler(CommandHandler("open_loops", _handle_open_loops))
    app.add_handler(CommandHandler("health", _handle_health))
    app.add_handler(CommandHandler("daily_brief_status", _handle_daily_brief_status))
    app.add_handler(CommandHandler("ask", _handle_ask))
    app.add_handler(CommandHandler("ask_opinion", _handle_ask_opinion))
    app.add_handler(CommandHandler("scout", _handle_scout))
    app.add_handler(CommandHandler("scout_dismiss", _handle_scout_dismiss))
    app.add_handler(CommandHandler("scout_installed", _handle_scout_installed))

    # Non-command messages only.
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))


def build_app(
    *,
    config: Config | None = None,
    token: str | None = None,
    llm_fixit_fn: Callable[[dict[str, Any]], tuple[bool, dict[str, Any]]] | None = None,
    llm_fixit_store: LLMFixitWizardStore | None = None,
    model_provider_wizard_store: TelegramModelProviderWizardStore | None = None,
    audit_log: AuditLog | None = None,
    runtime: Any | None = None,
) -> Application:
    loaded = config if isinstance(config, Config) else load_config(require_telegram_token=False)
    resolved_token = str(token or _resolve_telegram_bot_token() or "").strip()
    if not resolved_token:
        print(
            "Missing Telegram bot token. Save it in Web UI (telegram:bot_token) or set TELEGRAM_BOT_TOKEN.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
    loaded = replace(loaded, telegram_bot_token=resolved_token)
    db = MemoryDB(loaded.db_path)

    schema_path = Path(__file__).resolve().parents[1] / "memory" / "schema.sql"
    db.init_schema(str(schema_path))

    llm_client = LLMRouter(loaded, log_path=loaded.log_path)

    llm_broker, llm_broker_error = build_llm_broker(loaded)
    model_scout = build_model_scout(loaded)
    permission_store = PermissionStore(path=os.getenv("AGENT_PERMISSIONS_PATH", "").strip() or None)
    effective_audit_log = audit_log if isinstance(audit_log, AuditLog) else AuditLog(
        path=os.getenv("AGENT_AUDIT_LOG_PATH", "").strip() or None
    )

    orchestrator = Orchestrator(
        db=db,
        skills_path=loaded.skills_path,
        log_path=loaded.log_path,
        timezone=loaded.agent_timezone,
        llm_client=llm_client,
        enable_writes=loaded.enable_writes,
        llm_broker=llm_broker,
        llm_broker_error=llm_broker_error,
        perception_enabled=loaded.perception_enabled,
        perception_roots=loaded.perception_roots,
        perception_interval_seconds=loaded.perception_interval_seconds,
    )
    debug_protocol = DebugProtocol()

    app = Application.builder().token(loaded.telegram_bot_token).build()
    register_handlers(app)

    effective_llm_fixit_fn = llm_fixit_fn
    if not callable(effective_llm_fixit_fn):
        def _llm_fixit_internal(payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
            from agent.api_server import build_runtime

            runtime = build_runtime(config=load_config(require_telegram_token=False))
            try:
                return runtime.llm_fixit(payload if isinstance(payload, dict) else {})
            finally:
                runtime.close()

        effective_llm_fixit_fn = _llm_fixit_internal
    effective_llm_fixit_store = (
        llm_fixit_store if isinstance(llm_fixit_store, LLMFixitWizardStore) else LLMFixitWizardStore()
    )
    effective_model_provider_wizard_store = (
        model_provider_wizard_store
        if isinstance(model_provider_wizard_store, TelegramModelProviderWizardStore)
        else TelegramModelProviderWizardStore()
    )

    app.bot_data["db"] = db
    app.bot_data["orchestrator"] = orchestrator
    app.bot_data["debug_protocol"] = debug_protocol
    app.bot_data["log_path"] = loaded.log_path
    app.bot_data["home_path"] = os.path.expanduser("~")
    app.bot_data["timezone"] = loaded.agent_timezone
    app.bot_data["model_scout"] = model_scout
    app.bot_data["permission_store"] = permission_store
    app.bot_data["audit_log"] = effective_audit_log
    app.bot_data["llm_fixit_store"] = effective_llm_fixit_store
    app.bot_data["model_provider_wizard_store"] = effective_model_provider_wizard_store
    app.bot_data["llm_fixit_fn"] = effective_llm_fixit_fn
    app.bot_data["runtime"] = runtime
    app.bot_data["runtime_version"] = str(
        getattr(runtime, "version", "") or getattr(loaded, "api_version", "") or "0.1.0"
    )
    app.bot_data["runtime_git_commit"] = str(getattr(runtime, "git_commit", "") or "unknown")
    app.bot_data["runtime_started_ts"] = float(pytime.time())

    app.job_queue.run_repeating(_check_reminders, interval=30, first=5)
    if loaded.enable_scheduled_snapshots:
        run_time = time(9, 0, tzinfo=ZoneInfo(loaded.agent_timezone))
        app.job_queue.run_daily(_scheduled_disk_snapshot, time=run_time, name="disk_snapshot_daily")
        app.job_queue.run_daily(_scheduled_storage_snapshot, time=run_time, name="storage_snapshot_daily")
        app.job_queue.run_daily(_scheduled_resource_snapshot, time=run_time, name="resource_snapshot_daily")
        app.job_queue.run_daily(_scheduled_network_snapshot, time=run_time, name="network_snapshot_daily")
    return app


def main() -> None:
    app = build_app()
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
