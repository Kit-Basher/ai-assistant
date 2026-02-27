from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any


_WIZARD_SCHEMA_VERSION = 1
_FAILURE_STREAK_THRESHOLDS = (3, 10, 50)


@dataclass(frozen=True)
class WizardChoice:
    id: str
    label: str
    recommended: bool = False


@dataclass(frozen=True)
class DeterministicAction:
    id: str
    kind: str
    action: str
    reason: str
    params: dict[str, Any]
    safe_to_execute: bool


@dataclass(frozen=True)
class WizardDecision:
    status: str
    issue_code: str
    message: str
    question: str | None
    choices: list[WizardChoice]
    plan: list[DeterministicAction]
    details: dict[str, Any]


class LLMFixitWizardStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or self.default_path()).expanduser().resolve()
        self.state = self.load()

    @staticmethod
    def default_path() -> str:
        env_path = os.getenv("AGENT_LLM_FIXIT_WIZARD_STATE_PATH", "").strip()
        if env_path:
            return env_path
        return str(Path.home() / ".local" / "share" / "personal-agent" / "llm_fixit_wizard_state.json")

    @staticmethod
    def empty_state() -> dict[str, Any]:
        return {
            "schema_version": _WIZARD_SCHEMA_VERSION,
            "active": False,
            "issue_hash": None,
            "issue_code": None,
            "step": "idle",
            "question": None,
            "choices": [],
            "pending_plan": [],
            "pending_confirm_token": None,
            "pending_created_ts": None,
            "pending_expires_ts": None,
            "pending_issue_code": None,
            "confirm_token": None,  # legacy compatibility
            "last_prompt_ts": None,
        }

    def _normalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        state = self.empty_state()
        state["schema_version"] = _WIZARD_SCHEMA_VERSION
        state["active"] = bool(raw.get("active", False))
        state["issue_hash"] = str(raw.get("issue_hash") or "").strip() or None
        state["issue_code"] = str(raw.get("issue_code") or "").strip() or None
        step = str(raw.get("step") or "").strip().lower()
        state["step"] = step if step in {"idle", "awaiting_choice", "awaiting_confirm"} else "idle"
        state["question"] = str(raw.get("question") or "").strip() or None
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
        plan_raw = raw.get("pending_plan") if isinstance(raw.get("pending_plan"), list) else []
        plan: list[dict[str, Any]] = []
        for row in plan_raw:
            if not isinstance(row, dict):
                continue
            action_id = str(row.get("id") or "").strip()
            action = str(row.get("action") or "").strip()
            kind = str(row.get("kind") or "").strip().lower()
            reason = str(row.get("reason") or "").strip()
            params = row.get("params") if isinstance(row.get("params"), dict) else {}
            if not action_id or not action:
                continue
            plan.append(
                {
                    "id": action_id,
                    "kind": kind or "safe_action",
                    "action": action,
                    "reason": reason,
                    "params": dict(sorted(params.items())),
                    "safe_to_execute": bool(row.get("safe_to_execute", False)),
                }
            )
        state["pending_plan"] = plan
        pending_confirm = (
            str(raw.get("pending_confirm_token") or "").strip()
            or str(raw.get("confirm_token") or "").strip()
            or None
        )
        state["pending_confirm_token"] = pending_confirm
        try:
            state["pending_created_ts"] = int(raw.get("pending_created_ts") or 0) or None
        except (TypeError, ValueError):
            state["pending_created_ts"] = None
        try:
            state["pending_expires_ts"] = int(raw.get("pending_expires_ts") or 0) or None
        except (TypeError, ValueError):
            state["pending_expires_ts"] = None
        state["pending_issue_code"] = str(raw.get("pending_issue_code") or "").strip() or None
        state["confirm_token"] = pending_confirm  # legacy compatibility
        try:
            state["last_prompt_ts"] = int(raw.get("last_prompt_ts") or 0) or None
        except (TypeError, ValueError):
            state["last_prompt_ts"] = None

        if not bool(state["active"]):
            state["step"] = "idle"
            state["question"] = None
            state["choices"] = []
            state["pending_plan"] = []
            state["pending_confirm_token"] = None
            state["pending_created_ts"] = None
            state["pending_expires_ts"] = None
            state["pending_issue_code"] = None
            state["confirm_token"] = None
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
        fd, tmp_path = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
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



def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default



def _provider_rows(status_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = status_payload.get("providers") if isinstance(status_payload.get("providers"), list) else []
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        provider_id = str(row.get("id") or "").strip().lower()
        if not provider_id:
            continue
        output[provider_id] = row
    return output



def _model_rows(status_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = status_payload.get("models") if isinstance(status_payload.get("models"), list) else []
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("id") or "").strip()
        if not model_id:
            continue
        output[model_id] = row
    return output



def _resolved_default_model(status_payload: dict[str, Any]) -> str | None:
    resolved = str(status_payload.get("resolved_default_model") or "").strip()
    if resolved:
        return resolved
    return str(status_payload.get("default_model") or "").strip() or None



def _resolved_default_provider(status_payload: dict[str, Any]) -> str | None:
    provider = str(status_payload.get("default_provider") or "").strip().lower()
    if provider:
        return provider
    model_id = _resolved_default_model(status_payload)
    if model_id and ":" in model_id:
        return model_id.split(":", 1)[0].strip().lower() or None
    return None



def _best_local_model(status_payload: dict[str, Any]) -> str | None:
    candidates: list[tuple[str, int, int, int]] = []
    for model_id, row in sorted(_model_rows(status_payload).items()):
        provider = str(row.get("provider") or "").strip().lower()
        if provider != "ollama":
            continue
        health = row.get("health") if isinstance(row.get("health"), dict) else {}
        health_status = str(health.get("status") or "unknown").strip().lower()
        candidates.append(
            (
                model_id,
                0 if bool(row.get("enabled", False)) else 1,
                0 if bool(row.get("available", False)) and bool(row.get("routable", False)) and health_status == "ok" else 1,
                0 if health_status == "ok" else 1,
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[1], item[2], item[3], item[0]))
    return candidates[0][0]



def _any_routable_model(status_payload: dict[str, Any]) -> bool:
    for row in _model_rows(status_payload).values():
        health = row.get("health") if isinstance(row.get("health"), dict) else {}
        if (
            bool(row.get("enabled", False))
            and bool(row.get("available", False))
            and bool(row.get("routable", False))
            and str(health.get("status") or "unknown").strip().lower() == "ok"
        ):
            return True
    return False



def _error_kind(row: dict[str, Any]) -> str:
    health = row.get("health") if isinstance(row.get("health"), dict) else {}
    return str(health.get("last_error_kind") or "").strip().lower()



def _status_code(row: dict[str, Any]) -> int | None:
    health = row.get("health") if isinstance(row.get("health"), dict) else {}
    value = _safe_int(health.get("status_code"), 0)
    return value or None



def _provider_health_status(row: dict[str, Any]) -> str:
    health = row.get("health") if isinstance(row.get("health"), dict) else {}
    return str(health.get("status") or "unknown").strip().lower()



def _detect_issue_code(status_payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    providers = _provider_rows(status_payload)
    models = _model_rows(status_payload)
    safe_mode = status_payload.get("safe_mode") if isinstance(status_payload.get("safe_mode"), dict) else {}
    default_provider = _resolved_default_provider(status_payload)
    default_model = _resolved_default_model(status_payload)

    debug: dict[str, Any] = {
        "default_provider": default_provider,
        "default_model": default_model,
        "safe_mode_paused": bool(safe_mode.get("paused", False)),
    }

    if bool(safe_mode.get("paused", False)):
        debug["reason"] = str(safe_mode.get("reason") or "paused")
        return "safe_mode_paused", debug

    openai = providers.get("openai") if isinstance(providers.get("openai"), dict) else {}
    openai_error_kind = _error_kind(openai)
    openai_status = _status_code(openai)
    if openai and (openai_error_kind in {"http_401", "auth_error", "unauthorized"} or openai_status == 401):
        debug["provider"] = "openai"
        debug["provider_error_kind"] = openai_error_kind
        debug["provider_status_code"] = openai_status
        return "openai_unauthorized", debug

    openrouter = providers.get("openrouter") if isinstance(providers.get("openrouter"), dict) else {}
    openrouter_status = _provider_health_status(openrouter)
    openrouter_health = openrouter.get("health") if isinstance(openrouter.get("health"), dict) else {}
    openrouter_cooldown = _safe_int(openrouter_health.get("cooldown_until"), 0)
    openrouter_streak = max(0, _safe_int(openrouter_health.get("failure_streak"), 0))
    now_epoch = int(time.time())
    if openrouter and (
        openrouter_status == "down"
        or (openrouter_cooldown > now_epoch and openrouter_streak >= 10)
    ):
        debug["provider"] = "openrouter"
        debug["provider_status"] = openrouter_status
        debug["provider_failure_streak"] = openrouter_streak
        return "openrouter_down", debug

    if not _any_routable_model(status_payload):
        debug["routable_models"] = 0
        return "no_routable_model", debug

    default_provider_row = providers.get(default_provider or "") if default_provider else None
    default_model_row = models.get(default_model or "") if default_model else None
    provider_down = bool(default_provider_row) and _provider_health_status(default_provider_row or {}) == "down"
    model_ok = bool(default_model_row) and str(
        ((default_model_row or {}).get("health") or {}).get("status") or ""
    ).strip().lower() == "ok"
    if provider_down and model_ok:
        debug["provider_status"] = _provider_health_status(default_provider_row or {})
        debug["model_status"] = str(
            ((default_model_row or {}).get("health") or {}).get("status") or ""
        ).strip().lower()
        return "provider_inconsistent", debug

    debug["routable_models"] = sum(
        1
        for row in models.values()
        if bool(row.get("enabled", False)) and bool(row.get("available", False)) and bool(row.get("routable", False))
    )
    return "ok", debug



def _choices_for_issue(issue_code: str) -> list[WizardChoice]:
    if issue_code == "openai_unauthorized":
        return [
            WizardChoice(id="local_only", label="Use local-only", recommended=True),
            WizardChoice(id="fix_openai", label="Fix OpenAI", recommended=False),
            WizardChoice(id="details", label="Show details", recommended=False),
        ]
    if issue_code == "openrouter_down":
        return [
            WizardChoice(id="local_only", label="Use local-only", recommended=True),
            WizardChoice(id="repair_openrouter", label="Repair OpenRouter", recommended=False),
            WizardChoice(id="details", label="Show details", recommended=False),
        ]
    if issue_code == "safe_mode_paused":
        return [
            WizardChoice(id="unpause_autopilot", label="Unpause autopilot", recommended=True),
            WizardChoice(id="keep_paused", label="Keep paused", recommended=False),
            WizardChoice(id="details", label="Show details", recommended=False),
        ]
    if issue_code == "no_routable_model":
        return [
            WizardChoice(id="fix_ollama", label="Fix local Ollama", recommended=True),
            WizardChoice(id="use_remote_fallback", label="Use remote fallback", recommended=False),
            WizardChoice(id="details", label="Show details", recommended=False),
        ]
    if issue_code == "provider_inconsistent":
        return [
            WizardChoice(id="run_health_probe", label="Refresh health", recommended=True),
            WizardChoice(id="local_only", label="Use local-only", recommended=False),
            WizardChoice(id="details", label="Show details", recommended=False),
        ]
    return []



def _message_for_issue(issue_code: str, status_payload: dict[str, Any], debug: dict[str, Any]) -> str:
    _ = status_payload
    if issue_code == "openai_unauthorized":
        return "OpenAI authentication failed, so remote OpenAI calls are blocked right now."
    if issue_code == "openrouter_down":
        return "OpenRouter is currently unavailable, so remote routing through OpenRouter is failing."
    if issue_code == "safe_mode_paused":
        reason = str(debug.get("reason") or "paused")
        return f"Autopilot safe mode is paused ({reason}), so automatic fix actions are blocked."
    if issue_code == "no_routable_model":
        return "No enabled and routable model is available right now."
    if issue_code == "provider_inconsistent":
        return "Default provider health and default model health disagree, so routing status is inconsistent."
    return "LLM setup looks healthy."



def _question_for_issue(issue_code: str) -> str | None:
    if issue_code == "ok":
        return None
    return "Which option should I take?"



def evaluate_wizard_decision(status_payload: dict[str, Any]) -> WizardDecision:
    issue_code, debug = _detect_issue_code(status_payload)
    message = _message_for_issue(issue_code, status_payload, debug)
    if issue_code == "ok":
        return WizardDecision(
            status="ok",
            issue_code="ok",
            message=message,
            question=None,
            choices=[],
            plan=[],
            details=debug,
        )
    choices = _choices_for_issue(issue_code)
    return WizardDecision(
        status="needs_user_choice",
        issue_code=issue_code,
        message=message,
        question=_question_for_issue(issue_code),
        choices=choices,
        plan=[],
        details=debug,
    )



def _append_action(actions: list[DeterministicAction], *, kind: str, action: str, reason: str, params: dict[str, Any], safe: bool) -> None:
    action_id = f"{len(actions) + 1:02d}_{action}"
    actions.append(
        DeterministicAction(
            id=action_id,
            kind=kind,
            action=action,
            reason=reason,
            params=dict(sorted((params or {}).items())),
            safe_to_execute=bool(safe),
        )
    )



def _local_only_actions(status_payload: dict[str, Any], issue_code: str) -> list[DeterministicAction]:
    actions: list[DeterministicAction] = []
    providers = _provider_rows(status_payload)
    if issue_code in {"openrouter_down", "provider_inconsistent"}:
        if bool((providers.get("openrouter") or {}).get("enabled", False)):
            _append_action(
                actions,
                kind="safe_action",
                action="provider.set_enabled",
                reason="Disable OpenRouter while it is failing.",
                params={"provider": "openrouter", "enabled": False},
                safe=True,
            )
    if issue_code == "openai_unauthorized":
        if bool((providers.get("openai") or {}).get("enabled", False)):
            _append_action(
                actions,
                kind="safe_action",
                action="provider.set_enabled",
                reason="Disable OpenAI until credentials are fixed.",
                params={"provider": "openai", "enabled": False},
                safe=True,
            )

    local_model = _best_local_model(status_payload)
    if local_model:
        _append_action(
            actions,
            kind="safe_action",
            action="defaults.set",
            reason="Route to a healthy local model.",
            params={
                "default_provider": "ollama",
                "default_model": local_model,
                "allow_remote_fallback": False,
            },
            safe=True,
        )
    else:
        _append_action(
            actions,
            kind="user_action",
            action="ollama.install_local_model",
            reason="No healthy local model is currently available.",
            params={"hint": "Install a small local chat model in the UI and retry fix-it."},
            safe=False,
        )
    return actions



def build_plan_for_choice(
    *,
    issue_code: str,
    choice_id: str,
    status_payload: dict[str, Any],
) -> list[DeterministicAction]:
    choice = str(choice_id or "").strip().lower()
    if not choice:
        return []

    if choice == "details":
        return []

    if choice == "local_only":
        return _local_only_actions(status_payload, issue_code)

    actions: list[DeterministicAction] = []
    if choice == "repair_openrouter":
        _append_action(
            actions,
            kind="safe_action",
            action="provider.test",
            reason="Run a deterministic OpenRouter connectivity test.",
            params={"provider": "openrouter"},
            safe=True,
        )
        return actions

    if choice == "fix_openai":
        _append_action(
            actions,
            kind="user_action",
            action="provider.set_secret",
            reason="OpenAI key is missing or invalid.",
            params={"provider": "openai", "hint": "Set a valid OpenAI API key in Providers."},
            safe=False,
        )
        _append_action(
            actions,
            kind="safe_action",
            action="provider.test",
            reason="Verify OpenAI after updating the key.",
            params={"provider": "openai"},
            safe=True,
        )
        return actions

    if choice == "unpause_autopilot":
        _append_action(
            actions,
            kind="safe_action",
            action="autopilot.unpause",
            reason="Clear safe mode pause so deterministic apply actions can resume.",
            params={"confirm": True},
            safe=True,
        )
        return actions

    if choice == "keep_paused":
        return actions

    if choice == "fix_ollama":
        providers = _provider_rows(status_payload)
        ollama = providers.get("ollama") if isinstance(providers.get("ollama"), dict) else {}
        if _provider_health_status(ollama or {}) == "down":
            _append_action(
                actions,
                kind="user_action",
                action="ollama.start_verify",
                reason="Ollama provider is down.",
                params={"hint": "Open Providers in the UI, start Ollama, then run Fix-it again."},
                safe=False,
            )
        else:
            _append_action(
                actions,
                kind="user_action",
                action="ollama.install_local_model",
                reason="No routable local model is currently available.",
                params={"hint": "Install one Ollama chat model, then rerun Fix-it."},
                safe=False,
            )
        return actions

    if choice == "use_remote_fallback":
        _append_action(
            actions,
            kind="safe_action",
            action="defaults.set",
            reason="Enable remote fallback while local models are unavailable.",
            params={"allow_remote_fallback": True},
            safe=True,
        )
        return actions

    if choice == "run_health_probe":
        _append_action(
            actions,
            kind="safe_action",
            action="health.run",
            reason="Refresh provider/model health from authoritative probes.",
            params={"trigger": "manual"},
            safe=True,
        )
        return actions

    return actions



def decision_issue_hash(decision: WizardDecision, status_payload: dict[str, Any]) -> str:
    payload = {
        "issue_code": decision.issue_code,
        "message": decision.message,
        "default_provider": _resolved_default_provider(status_payload),
        "default_model": _resolved_default_model(status_payload),
        "safe_mode_paused": bool((status_payload.get("safe_mode") or {}).get("paused", False)),
        "provider_states": sorted(
            [
                {
                    "id": provider_id,
                    "status": _provider_health_status(row),
                    "failure_streak": max(0, _safe_int((row.get("health") or {}).get("failure_streak"), 0)),
                }
                for provider_id, row in _provider_rows(status_payload).items()
            ],
            key=lambda item: str(item.get("id") or ""),
        ),
    }
    return sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()



def confirm_token_for_plan(plan: list[DeterministicAction]) -> str:
    payload = [
        {
            "id": item.id,
            "kind": item.kind,
            "action": item.action,
            "reason": item.reason,
            "params": dict(sorted(item.params.items())),
            "safe_to_execute": bool(item.safe_to_execute),
        }
        for item in plan
    ]
    return sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def confirm_token_for_plan_rows(plan_rows: list[dict[str, Any]]) -> str:
    payload = [
        {
            "id": str(item.get("id") or "").strip(),
            "kind": str(item.get("kind") or "").strip(),
            "action": str(item.get("action") or "").strip(),
            "reason": str(item.get("reason") or "").strip(),
            "params": dict(
                sorted(
                    (
                        params_key,
                        params_value,
                    )
                    for params_key, params_value in (
                        (item.get("params") if isinstance(item.get("params"), dict) else {}).items()
                    )
                )
            ),
            "safe_to_execute": bool(item.get("safe_to_execute", False)),
        }
        for item in sorted(
            [row for row in plan_rows if isinstance(row, dict)],
            key=lambda row: str(row.get("id") or ""),
        )
    ]
    return sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()



def parse_choice_answer(answer: str | None, choices: list[WizardChoice]) -> str | None:
    value = str(answer or "").strip()
    if not value:
        return None
    normalized = value.lower()
    ids = [choice.id for choice in choices]
    if normalized in ids:
        return normalized
    if normalized.isdigit():
        idx = int(normalized)
        if 1 <= idx <= len(ids):
            return ids[idx - 1]
    return None



def render_wizard_prompt(decision: WizardDecision) -> str:
    base = str(decision.message or "").strip() or "I found an LLM setup issue."
    if not decision.choices:
        return base
    lines = [base]
    for idx, choice in enumerate(decision.choices, start=1):
        suffix = " (recommended)" if choice.recommended else ""
        lines.append(f"{idx}) {choice.label}{suffix}")
    lines.append("Reply with 1, 2, or 3.")
    return "\n".join(lines)



def summarize_notification_message(raw_message: str) -> str:
    text = str(raw_message or "").strip()
    if not text:
        return "I detected an LLM state update."
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullet_lines = [line[2:].strip() for line in lines if line.startswith("- ")]
    summary: list[str] = []
    for line in bullet_lines:
        lowered = line.lower()
        if lowered.startswith("defaults: default_model ->"):
            after = line.split("->", 1)[1].split("(was", 1)[0].strip()
            summary.append(f"Default model changed to {after}.")
        elif lowered.startswith("defaults: default_provider ->"):
            after = line.split("->", 1)[1].split("(was", 1)[0].strip()
            summary.append(f"Default provider changed to {after}.")
        elif lowered.startswith("provider ") and "health.status" in lowered and "->" in line:
            prefix = line.split(":", 1)[0].strip()
            provider = prefix.split(" ", 1)[1].strip() if " " in prefix else prefix
            after = line.split("->", 1)[1].split("(was", 1)[0].strip().strip('"')
            if after == "down":
                summary.append(f"Provider {provider} is down.")
            elif after == "ok":
                summary.append(f"Provider {provider} recovered.")
            else:
                summary.append(f"Provider {provider} health is now {after}.")
        elif lowered.startswith("model ") and "health.status" in lowered and "->" in line:
            prefix = line.split(":", 1)[0].strip()
            model = prefix.split(" ", 1)[1].strip() if " " in prefix else prefix
            after = line.split("->", 1)[1].split("(was", 1)[0].strip().strip('"')
            if after == "down":
                summary.append(f"Model {model} is down.")
            elif after == "ok":
                summary.append(f"Model {model} recovered.")
            else:
                summary.append(f"Model {model} health is now {after}.")
    if not summary:
        return "I updated LLM settings in the background and the service is still running."
    return "\n".join(summary[:3])



def failure_streak_threshold_crossed(before_state: dict[str, Any], after_state: dict[str, Any]) -> bool:
    before_providers = before_state.get("providers") if isinstance(before_state.get("providers"), dict) else {}
    after_providers = after_state.get("providers") if isinstance(after_state.get("providers"), dict) else {}
    for provider_id in sorted(set(before_providers.keys()) | set(after_providers.keys())):
        before_row = before_providers.get(provider_id) if isinstance(before_providers.get(provider_id), dict) else {}
        after_row = after_providers.get(provider_id) if isinstance(after_providers.get(provider_id), dict) else {}
        before_streak = max(0, _safe_int((before_row.get("health") or {}).get("failure_streak"), 0))
        after_streak = max(0, _safe_int((after_row.get("health") or {}).get("failure_streak"), 0))
        for threshold in _FAILURE_STREAK_THRESHOLDS:
            if before_streak < threshold <= after_streak:
                return True
    return False



def provider_or_model_ok_down_transition(before_state: dict[str, Any], after_state: dict[str, Any]) -> bool:
    def _status_from_row(row: dict[str, Any]) -> str:
        health = row.get("health") if isinstance(row.get("health"), dict) else {}
        return str(health.get("status") or "unknown").strip().lower()

    before_providers = before_state.get("providers") if isinstance(before_state.get("providers"), dict) else {}
    after_providers = after_state.get("providers") if isinstance(after_state.get("providers"), dict) else {}
    for provider_id in sorted(set(before_providers.keys()) | set(after_providers.keys())):
        before_status = _status_from_row(before_providers.get(provider_id) if isinstance(before_providers.get(provider_id), dict) else {})
        after_status = _status_from_row(after_providers.get(provider_id) if isinstance(after_providers.get(provider_id), dict) else {})
        if before_status != after_status and {before_status, after_status} & {"ok", "down"}:
            return True

    before_models = before_state.get("models") if isinstance(before_state.get("models"), dict) else {}
    after_models = after_state.get("models") if isinstance(after_state.get("models"), dict) else {}
    for model_id in sorted(set(before_models.keys()) | set(after_models.keys())):
        before_status = _status_from_row(before_models.get(model_id) if isinstance(before_models.get(model_id), dict) else {})
        after_status = _status_from_row(after_models.get(model_id) if isinstance(after_models.get(model_id), dict) else {})
        if before_status != after_status and {before_status, after_status} & {"ok", "down"}:
            return True
    return False



def decision_to_json(decision: WizardDecision) -> dict[str, Any]:
    return {
        "status": decision.status,
        "issue_code": decision.issue_code,
        "message": decision.message,
        "question": decision.question,
        "choices": [asdict(choice) for choice in decision.choices],
        "plan": [asdict(step) for step in decision.plan],
        "details": decision.details,
    }
