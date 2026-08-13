from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
import re
import threading
import uuid
from typing import Any, Callable, Mapping

from agent.capability_registry import ApprovalPolicy, CapabilityMode, CapabilityRegistry
from agent.logging_utils import redact_payload
from agent.llm.inference_router import route_inference


TASK_SCHEMA_VERSION = "personal-agent.task.v1"
PLAN_SCHEMA_VERSION = "personal-agent.plan.v1"
MISSING_CAPABILITY_SCHEMA_VERSION = "personal-agent.missing-capability.v1"
MAX_PLAN_STEPS = 8
MAX_PLAN_REVISIONS = 3
MAX_REPLANS = 2
MAX_CAPABILITY_CALLS = 16
MAX_PLANNING_GENERATIONS_PER_VERSION = 1
MAX_TASK_WALL_SECONDS = 900
MAX_INPUT_BYTES = 16_384
MAX_RESULT_BYTES = 32_768
MAX_EVENT_BYTES = 8_192
MAX_RETAINED_TERMINAL_TASKS = 200
MAX_CONCURRENT_TASKS_PER_ACTOR = 3


class TaskState(str, Enum):
    PLANNING = "planning"
    PROPOSED = "proposed"
    READY = "ready"
    AWAITING_INFORMATION = "awaiting_information"
    AWAITING_APPROVAL = "awaiting_approval"
    RUNNING = "running"
    VERIFYING = "verifying"
    PAUSED = "paused"
    BLOCKED = "blocked"
    RECOVERING = "recovering"
    SUCCEEDED = "succeeded"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"
    DENIED = "denied"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    INDETERMINATE = "indeterminate"


TERMINAL_STATES = {
    TaskState.SUCCEEDED,
    TaskState.PARTIALLY_COMPLETED,
    TaskState.FAILED,
    TaskState.DENIED,
    TaskState.CANCELLED,
    TaskState.EXPIRED,
    TaskState.INDETERMINATE,
}

ALLOWED_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.PLANNING: frozenset({TaskState.PROPOSED, TaskState.AWAITING_INFORMATION, TaskState.BLOCKED, TaskState.FAILED}),
    TaskState.PROPOSED: frozenset({TaskState.READY, TaskState.AWAITING_INFORMATION, TaskState.AWAITING_APPROVAL, TaskState.CANCELLED, TaskState.FAILED}),
    TaskState.READY: frozenset({TaskState.RUNNING, TaskState.PAUSED, TaskState.CANCELLED, TaskState.EXPIRED}),
    TaskState.RUNNING: frozenset({TaskState.AWAITING_APPROVAL, TaskState.VERIFYING, TaskState.PAUSED, TaskState.BLOCKED, TaskState.RECOVERING, TaskState.PARTIALLY_COMPLETED, TaskState.CANCELLED, TaskState.FAILED, TaskState.INDETERMINATE}),
    TaskState.AWAITING_INFORMATION: frozenset({TaskState.PROPOSED, TaskState.CANCELLED, TaskState.EXPIRED}),
    TaskState.AWAITING_APPROVAL: frozenset({TaskState.RUNNING, TaskState.DENIED, TaskState.CANCELLED, TaskState.EXPIRED, TaskState.BLOCKED}),
    TaskState.VERIFYING: frozenset({TaskState.RUNNING, TaskState.SUCCEEDED, TaskState.PARTIALLY_COMPLETED, TaskState.RECOVERING, TaskState.BLOCKED, TaskState.FAILED, TaskState.INDETERMINATE}),
    TaskState.PAUSED: frozenset({TaskState.READY, TaskState.CANCELLED, TaskState.EXPIRED, TaskState.BLOCKED}),
    TaskState.BLOCKED: frozenset({TaskState.READY, TaskState.PARTIALLY_COMPLETED, TaskState.CANCELLED, TaskState.EXPIRED, TaskState.FAILED}),
    TaskState.RECOVERING: frozenset({TaskState.READY, TaskState.RUNNING, TaskState.AWAITING_APPROVAL, TaskState.BLOCKED, TaskState.PARTIALLY_COMPLETED, TaskState.FAILED, TaskState.INDETERMINATE}),
}

_OUTPUT_REFERENCE_RE = re.compile(r"^\$\{([a-z0-9][a-z0-9_-]{0,63})\.result\.([a-zA-Z0-9_.-]{1,120})\}$")
_DANGEROUS_PLAN_KEYS = {"approved", "approval", "command", "endpoint", "executor", "raw_shell", "shell", "tool", "url"}
_SENSITIVE_RESULT_KEYS = {
    "body", "content", "conversation", "document", "html", "message", "messages",
    "model_prompt", "prompt", "raw", "response_text", "secret", "summary", "text", "token",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _bounded_text(value: Any, *, maximum: int, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"task_{field}_required")
    if len(text.encode("utf-8")) > maximum:
        raise ValueError(f"task_{field}_too_large")
    return text


def _redacted(value: Any) -> Any:
    if isinstance(value, dict):
        value = redact_payload(value)
    encoded = _canonical_json(value)
    if len(encoded.encode("utf-8")) <= MAX_RESULT_BYTES:
        return value
    return {"truncated": True, "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(), "bytes": len(encoded.encode("utf-8"))}


def _content_fingerprint(value: Any) -> dict[str, Any]:
    encoded = str(value or "").encode("utf-8", errors="replace")
    return {"redacted": True, "bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def _audit_safe_result(value: Any, *, key: str = "") -> Any:
    """Keep structured routing evidence while excluding content and prompts.

    Paths and non-content status fields remain available for typed output
    references (for example search -> read). Raw file/model/web/pack text is
    represented only by a length and digest.
    """

    if key.lower() in _SENSITIVE_RESULT_KEYS:
        return _content_fingerprint(value)
    if isinstance(value, Mapping):
        clean = redact_payload(dict(value))
        return {str(item_key): _audit_safe_result(item_value, key=str(item_key)) for item_key, item_value in clean.items()}
    if isinstance(value, list):
        return [_audit_safe_result(item) for item in value[:100]]
    return value


def _declared_result_succeeded(result: Any) -> bool:
    """Interpret the canonical capability outcome without trusting prose."""
    payload = TaskCoordinator._result_payload(result)
    declared = payload.get("data") if isinstance(payload.get("data"), Mapping) else payload
    runtime_payload = declared.get("runtime_payload") if isinstance(declared.get("runtime_payload"), Mapping) else {}
    return not (
        declared.get("ok") is False
        or bool(declared.get("error_kind"))
        or runtime_payload.get("ok") is False
        or bool(runtime_payload.get("error_kind"))
    )


@dataclass(frozen=True)
class ValidatedPlan:
    payload: dict[str, Any]

    @property
    def task_id(self) -> str:
        return str(self.payload["task_id"])

    @property
    def version(self) -> int:
        return int(self.payload["version"])

    @property
    def plan_hash(self) -> str:
        return str(self.payload["plan_hash"])


def validate_plan(
    proposal: Mapping[str, Any],
    registry: CapabilityRegistry,
    *,
    actor_id: str,
    session_id: str,
    thread_id: str,
) -> ValidatedPlan:
    allowed = {
        "schema_version", "task_id", "version", "goal", "success_criteria", "steps",
        "created_by", "planning_generation_count", "plan_hash",
    }
    unknown = sorted(set(proposal) - allowed)
    if unknown:
        raise ValueError(f"task_plan_unknown_fields:{','.join(unknown)}")
    if str(proposal.get("schema_version") or "") != PLAN_SCHEMA_VERSION:
        raise ValueError("task_plan_schema_unsupported")
    task_id = str(proposal.get("task_id") or "").strip()
    if not re.fullmatch(r"task-[a-f0-9]{16,32}", task_id):
        raise ValueError("task_plan_id_invalid")
    version = int(proposal.get("version") or 0)
    if version < 1 or version > MAX_PLAN_REVISIONS:
        raise ValueError("task_plan_version_out_of_bounds")
    goal = _bounded_text(proposal.get("goal"), maximum=2_000, field="goal")
    criteria_raw = proposal.get("success_criteria")
    if not isinstance(criteria_raw, list) or not criteria_raw or len(criteria_raw) > MAX_PLAN_STEPS:
        raise ValueError("task_success_criteria_invalid")
    criteria = [_bounded_text(item, maximum=500, field="criterion") for item in criteria_raw]
    steps_raw = proposal.get("steps")
    if not isinstance(steps_raw, list) or not 1 <= len(steps_raw) <= MAX_PLAN_STEPS:
        raise ValueError("task_plan_step_count_invalid")
    planning_generations = int(proposal.get("planning_generation_count") or 0)
    if planning_generations < 0 or planning_generations > MAX_PLANNING_GENERATIONS_PER_VERSION:
        raise ValueError("task_planning_generation_limit_exceeded")
    normalized_steps: list[dict[str, Any]] = []
    known_ids: set[str] = set()
    for position, raw in enumerate(steps_raw):
        if not isinstance(raw, Mapping):
            raise ValueError("task_plan_step_invalid")
        allowed_step = {
            "step_id", "capability_id", "inputs", "depends_on", "expected_evidence",
            "verification", "timeout_seconds", "retry_limit", "compensation_capability_id",
        }
        extra = sorted(set(raw) - allowed_step)
        if extra:
            raise ValueError(f"task_plan_step_unknown_fields:{','.join(extra)}")
        step_id = str(raw.get("step_id") or f"step-{position + 1}").strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", step_id) or step_id in known_ids:
            raise ValueError("task_plan_step_id_invalid")
        capability_id = str(raw.get("capability_id") or "").strip().lower()
        definition = registry.require(capability_id)
        if not definition.task_composable:
            raise ValueError(f"task_capability_not_composable:{capability_id}")
        inputs_raw = raw.get("inputs")
        if not isinstance(inputs_raw, Mapping):
            raise ValueError("task_plan_step_inputs_invalid")
        if len(_canonical_json(inputs_raw).encode("utf-8")) > MAX_INPUT_BYTES:
            raise ValueError("task_plan_step_inputs_too_large")
        dangerous = sorted(key for key in inputs_raw if str(key).lower() in _DANGEROUS_PLAN_KEYS and key not in definition.input_contract.properties)
        if dangerous:
            raise ValueError(f"task_plan_dangerous_input:{','.join(dangerous)}")
        dependencies = raw.get("depends_on") if isinstance(raw.get("depends_on"), list) else []
        depends_on = [str(item).strip().lower() for item in dependencies]
        if len(depends_on) != len(set(depends_on)) or any(item not in known_ids for item in depends_on):
            raise ValueError("task_plan_dependency_invalid")
        prepared_inputs = dict(inputs_raw)
        prepared_inputs.setdefault("user_id", actor_id)
        prepared_inputs.setdefault("text", goal)
        for value in prepared_inputs.values():
            if isinstance(value, str) and value.startswith("${"):
                match = _OUTPUT_REFERENCE_RE.fullmatch(value)
                if not match or match.group(1) not in depends_on:
                    raise ValueError("task_plan_output_reference_invalid")
        # References are type-checked after resolution. All other values cross
        # the exact WP2 contract now.
        if not any(isinstance(value, str) and value.startswith("${") for value in prepared_inputs.values()):
            prepared_inputs = definition.input_contract.validate(prepared_inputs)
            if definition.task_input_validation_hook is not None:
                task_inputs_ok, task_inputs_reason = definition.task_input_validation_hook(prepared_inputs)
                if not task_inputs_ok:
                    raise ValueError(f"task_plan_capability_inputs_invalid:{task_inputs_reason or capability_id}")
        timeout_seconds = int(raw.get("timeout_seconds") or 30)
        if not 1 <= timeout_seconds <= 120:
            raise ValueError("task_plan_timeout_invalid")
        retry_limit = int(raw.get("retry_limit") or 0)
        if retry_limit < 0 or retry_limit > definition.max_task_retries:
            raise ValueError("task_plan_retry_policy_invalid")
        compensation = str(raw.get("compensation_capability_id") or "").strip().lower() or None
        if compensation != definition.compensation_capability_id:
            if compensation is not None or definition.compensation_capability_id is not None:
                raise ValueError("task_plan_compensation_invalid")
        normalized_steps.append({
            "step_id": step_id,
            "position": position,
            "capability_id": capability_id,
            "inputs": prepared_inputs,
            "depends_on": depends_on,
            "expected_evidence": _bounded_text(raw.get("expected_evidence") or definition.description, maximum=500, field="expected_evidence"),
            "verification": str(raw.get("verification") or "registry_and_task").strip().lower(),
            "mode": definition.mode.value,
            "approval_policy": definition.approval_policy.value,
            "retry_safety": definition.retry_safety,
            "retry_limit": retry_limit,
            "timeout_seconds": timeout_seconds,
            "resume_policy": definition.resume_policy,
            "compensation_capability_id": compensation,
        })
        known_ids.add(step_id)
    canonical = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "task_id": task_id,
        "version": version,
        "goal": goal,
        "success_criteria": criteria,
        "steps": normalized_steps,
        "created_by": str(proposal.get("created_by") or "deterministic_decomposition").strip()[:80],
        "planning_generation_count": planning_generations,
        "binding": {"actor_id": actor_id, "session_id": session_id, "thread_id": thread_id},
    }
    canonical["plan_hash"] = _hash(canonical)
    supplied_hash = str(proposal.get("plan_hash") or "").strip()
    if supplied_hash and supplied_hash != canonical["plan_hash"]:
        raise ValueError("task_plan_hash_mismatch")
    return ValidatedPlan(canonical)


def build_missing_capability(
    *, goal: str,
    success_criteria: list[str],
    missing_description: str,
    considered_capabilities: list[str],
    completed_evidence: list[dict[str, Any]] | None = None,
    dependencies: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": MISSING_CAPABILITY_SCHEMA_VERSION,
        "requested_outcome": str(goal)[:2_000],
        "success_criteria": [str(item)[:500] for item in success_criteria[:MAX_PLAN_STEPS]],
        "missing_user_goal_capability": str(missing_description)[:500],
        "input_shape": "bounded structured inputs determined by a future registered capability",
        "output_shape": "verified structured evidence for the requested outcome",
        "required_permissions": [],
        "required_dependencies": [str(item)[:120] for item in (dependencies or [])[:8]],
        "considered_capabilities": sorted(set(str(item) for item in considered_capabilities))[:32],
        "why_incomplete": "No currently registered capability can safely produce the remaining required evidence.",
        "completed_partial_evidence": _redacted(completed_evidence or []),
        "safe_next_step_category": "explain_or_wait_for_future_capability",
        "automatic_pack_action": False,
    }


class TaskStore:
    """Durable task records in the canonical MemoryDB connection."""

    def __init__(self, db: Any) -> None:
        self.db = db
        self._lock = threading.RLock()
        self.ensure_schema()

    @property
    def _conn(self) -> Any:
        return self.db._conn  # noqa: SLF001 - same canonical DB/transaction owner

    @staticmethod
    def _mutation_resource_keys(plan_payload: Mapping[str, Any]) -> set[str]:
        keys: set[str] = set()
        for step in plan_payload.get("steps") if isinstance(plan_payload.get("steps"), list) else []:
            if not isinstance(step, Mapping) or str(step.get("mode") or "") != CapabilityMode.MUTATING.value:
                continue
            inputs = dict(step.get("inputs")) if isinstance(step.get("inputs"), Mapping) else {}
            inputs.pop("user_id", None)
            inputs.pop("text", None)
            keys.add(_hash({"capability_id": step.get("capability_id"), "targets": inputs}))
        return keys

    def ensure_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_tasks (
                    task_id TEXT PRIMARY KEY, schema_version TEXT NOT NULL, actor_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, thread_id TEXT NOT NULL, goal TEXT NOT NULL,
                    success_criteria_json TEXT NOT NULL, plan_json TEXT NOT NULL, plan_hash TEXT NOT NULL,
                    plan_version INTEGER NOT NULL, state TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0,
                    current_step INTEGER NOT NULL DEFAULT 0, capability_calls INTEGER NOT NULL DEFAULT 0,
                    planning_generations INTEGER NOT NULL DEFAULT 0, replan_count INTEGER NOT NULL DEFAULT 0,
                    failure_json TEXT, outcome_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    terminal_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_agent_tasks_actor_thread ON agent_tasks(actor_id, thread_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS agent_task_steps (
                    task_id TEXT NOT NULL, step_id TEXT NOT NULL, position INTEGER NOT NULL,
                    capability_id TEXT NOT NULL, mode TEXT NOT NULL, approval_policy TEXT NOT NULL,
                    status TEXT NOT NULL, inputs_json TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                    invocation_id TEXT, result_json TEXT, evidence_json TEXT, verifier_status TEXT,
                    failure_json TEXT, started_at TEXT, finished_at TEXT,
                    PRIMARY KEY(task_id, step_id), FOREIGN KEY(task_id) REFERENCES agent_tasks(task_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS agent_task_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL,
                    UNIQUE(task_id, sequence), FOREIGN KEY(task_id) REFERENCES agent_tasks(task_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS agent_task_approvals (
                    approval_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, plan_version INTEGER NOT NULL,
                    plan_hash TEXT NOT NULL, binding_hash TEXT NOT NULL, actor_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, thread_id TEXT NOT NULL, mutating_steps_json TEXT NOT NULL,
                    state TEXT NOT NULL, expires_at TEXT NOT NULL, consumed_at TEXT,
                    FOREIGN KEY(task_id) REFERENCES agent_tasks(task_id) ON DELETE CASCADE
                );
                """
            )
            self._conn.commit()

    def create(self, plan: ValidatedPlan) -> dict[str, Any]:
        payload = plan.payload
        binding = payload["binding"]
        now = _now()
        with self._lock:
            active = self._conn.execute(
                "SELECT COUNT(*) AS n FROM agent_tasks WHERE actor_id=? AND state NOT IN ('succeeded','partially_completed','failed','denied','cancelled','expired','indeterminate')",
                (binding["actor_id"],),
            ).fetchone()
            if int(active["n"] or 0) >= MAX_CONCURRENT_TASKS_PER_ACTOR:
                raise RuntimeError("task_concurrency_limit_reached")
            new_resources = self._mutation_resource_keys(payload)
            if new_resources:
                for row in self._conn.execute(
                    "SELECT thread_id,plan_json FROM agent_tasks WHERE state NOT IN ('succeeded','partially_completed','failed','denied','cancelled','expired','indeterminate')"
                ).fetchall():
                    prior_plan = self._loads(row["plan_json"], {})
                    if str(row["thread_id"]) == str(binding["thread_id"]) and self._mutation_resource_keys(prior_plan):
                        raise RuntimeError("task_thread_mutation_conflict")
                    if new_resources & self._mutation_resource_keys(prior_plan):
                        raise RuntimeError("task_resource_conflict")
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT INTO agent_tasks(task_id,schema_version,actor_id,session_id,thread_id,goal,success_criteria_json,plan_json,plan_hash,plan_version,state,planning_generations,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (payload["task_id"], TASK_SCHEMA_VERSION, binding["actor_id"], binding["session_id"], binding["thread_id"], payload["goal"], _canonical_json(payload["success_criteria"]), _canonical_json(payload), payload["plan_hash"], payload["version"], TaskState.PROPOSED.value, payload["planning_generation_count"], now, now),
                )
                for step in payload["steps"]:
                    self._conn.execute(
                        "INSERT INTO agent_task_steps(task_id,step_id,position,capability_id,mode,approval_policy,status,inputs_json) VALUES(?,?,?,?,?,?,?,?)",
                        (payload["task_id"], step["step_id"], step["position"], step["capability_id"], step["mode"], step["approval_policy"], "pending", _canonical_json(step["inputs"])),
                    )
                self._append_event_locked(payload["task_id"], "task.proposed", {"plan_version": payload["version"], "plan_hash": payload["plan_hash"]})
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get(payload["task_id"], actor_id=binding["actor_id"], thread_id=binding["thread_id"]) or {}

    def _prune_terminal_locked(self) -> None:
        rows = self._conn.execute(
            "SELECT task_id FROM agent_tasks WHERE terminal_at IS NOT NULL "
            "ORDER BY terminal_at DESC LIMIT -1 OFFSET ?",
            (MAX_RETAINED_TERMINAL_TASKS,),
        ).fetchall()
        for row in rows:
            task_id = str(row["task_id"])
            self._conn.execute("DELETE FROM agent_task_approvals WHERE task_id=?", (task_id,))
            self._conn.execute("DELETE FROM agent_task_events WHERE task_id=?", (task_id,))
            self._conn.execute("DELETE FROM agent_task_steps WHERE task_id=?", (task_id,))
            self._conn.execute("DELETE FROM agent_tasks WHERE task_id=?", (task_id,))

    def _append_event_locked(self, task_id: str, event_type: str, payload: Mapping[str, Any]) -> None:
        clean = _redacted(dict(payload))
        encoded = _canonical_json(clean)
        if len(encoded.encode("utf-8")) > MAX_EVENT_BYTES:
            clean = {"truncated": True, "sha256": _hash(clean)}
            encoded = _canonical_json(clean)
        row = self._conn.execute("SELECT COALESCE(MAX(sequence),0)+1 AS seq FROM agent_task_events WHERE task_id=?", (task_id,)).fetchone()
        self._conn.execute(
            "INSERT INTO agent_task_events(task_id,sequence,event_type,payload_json,created_at) VALUES(?,?,?,?,?)",
            (task_id, int(row["seq"]), str(event_type)[:120], encoded, _now()),
        )

    def transition(self, task_id: str, target: TaskState, *, expected_revision: int | None = None, event: str | None = None, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute("SELECT state,revision FROM agent_tasks WHERE task_id=?", (task_id,)).fetchone()
                if row is None:
                    raise KeyError("task_not_found")
                current = TaskState(str(row["state"]))
                revision = int(row["revision"])
                if expected_revision is not None and revision != int(expected_revision):
                    raise RuntimeError("task_revision_conflict")
                if current in TERMINAL_STATES or target not in ALLOWED_TRANSITIONS.get(current, frozenset()):
                    raise ValueError(f"task_transition_invalid:{current.value}->{target.value}")
                terminal_at = _now() if target in TERMINAL_STATES else None
                self._conn.execute("UPDATE agent_tasks SET state=?,revision=revision+1,updated_at=?,terminal_at=COALESCE(?,terminal_at) WHERE task_id=? AND revision=?", (target.value, _now(), terminal_at, task_id, revision))
                self._append_event_locked(task_id, event or f"task.{target.value}", payload or {})
                if target in TERMINAL_STATES:
                    self._prune_terminal_locked()
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get(task_id) or {}

    def update_step(self, task_id: str, step_id: str, *, status: str, result: Any = None, evidence: Any = None, verifier_status: str | None = None, failure: Any = None, increment_attempt: bool = False) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE agent_task_steps SET status=?, attempts=attempts+?, result_json=?, evidence_json=?, verifier_status=?, failure_json=?, started_at=COALESCE(started_at,?), finished_at=? WHERE task_id=? AND step_id=?",
                (
                    status, 1 if increment_attempt else 0,
                    _canonical_json(_redacted(_audit_safe_result(result))) if result is not None else None,
                    _canonical_json(_redacted(evidence)) if evidence is not None else None,
                    verifier_status,
                    _canonical_json(_redacted(failure)) if failure is not None else None,
                    _now(), _now() if status in {"completed","failed","blocked","indeterminate","cancelled"} else None,
                    task_id, step_id,
                ),
            )
            self._append_event_locked(task_id, f"step.{status}", {"step_id": step_id, "verifier_status": verifier_status})
            self._conn.commit()

    def update_step_inputs(self, task_id: str, step_id: str, inputs: Mapping[str, Any]) -> None:
        with self._lock:
            self._conn.execute("UPDATE agent_task_steps SET inputs_json=? WHERE task_id=? AND step_id=? AND status='pending'", (_canonical_json(dict(inputs)), task_id, step_id))
            self._append_event_locked(task_id, "step.inputs_resolved", {"step_id": step_id, "inputs_hash": _hash(inputs)})
            self._conn.commit()

    def advance_step(self, task_id: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE agent_tasks SET current_step=current_step+1,revision=revision+1,updated_at=? WHERE task_id=?", (_now(), task_id))
            self._conn.commit()

    def record_capability_call(self, task_id: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE agent_tasks SET capability_calls=capability_calls+1,revision=revision+1,updated_at=? WHERE task_id=?", (_now(), task_id))
            self._conn.commit()

    def set_failure_outcome(self, task_id: str, *, failure: Mapping[str, Any] | None = None, outcome: Mapping[str, Any] | None = None) -> None:
        with self._lock:
            self._conn.execute("UPDATE agent_tasks SET failure_json=?,outcome_json=?,updated_at=? WHERE task_id=?", (_canonical_json(_redacted(dict(failure))) if failure else None, _canonical_json(_redacted(dict(outcome))) if outcome else None, _now(), task_id))
            self._conn.commit()

    def issue_approval(self, task_id: str, *, step: Mapping[str, Any], health_state: str) -> dict[str, Any]:
        task = self.get(task_id)
        if task is None or str(task.get("state") or "") != TaskState.RUNNING.value:
            raise ValueError("task_approval_state_invalid")
        expires = datetime.now(timezone.utc) + timedelta(minutes=10)
        binding_payload = {
            "schema_version": TASK_SCHEMA_VERSION,
            "task_id": task_id,
            "plan_version": int(task["plan_version"]),
            "plan_hash": str(task["plan_hash"]),
            "mutating_steps": [{
                "step_id": str(step.get("step_id") or ""),
                "capability_id": str(step.get("capability_id") or ""),
                "inputs_hash": _hash(step.get("inputs") or {}),
                "mode": str(step.get("mode") or ""),
                "approval_policy": str(step.get("approval_policy") or ""),
                "health_state": str(health_state),
            }],
            "actor_id": str(task["actor_id"]),
            "session_id": str(task["session_id"]),
            "thread_id": str(task["thread_id"]),
            "expires_at": expires.isoformat(),
        }
        approval = {
            "approval_id": f"task-approval-{uuid.uuid4().hex[:20]}",
            **binding_payload,
            "binding_hash": _hash(binding_payload),
            "state": "pending",
        }
        with self._lock:
            self._conn.execute("UPDATE agent_task_approvals SET state='superseded' WHERE task_id=? AND state='pending'", (task_id,))
            self._conn.execute(
                "INSERT INTO agent_task_approvals(approval_id,task_id,plan_version,plan_hash,binding_hash,actor_id,session_id,thread_id,mutating_steps_json,state,expires_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (approval["approval_id"], task_id, approval["plan_version"], approval["plan_hash"], approval["binding_hash"], approval["actor_id"], approval["session_id"], approval["thread_id"], _canonical_json(approval["mutating_steps"]), "pending", approval["expires_at"]),
            )
            self._append_event_locked(task_id, "task.approval_issued", {"approval_id": approval["approval_id"], "binding_hash": approval["binding_hash"], "expires_at": approval["expires_at"]})
            self._conn.commit()
        return approval

    def revise(self, plan: ValidatedPlan) -> dict[str, Any]:
        payload = plan.payload
        task_id = plan.task_id
        binding = payload["binding"]
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                old = self._conn.execute("SELECT * FROM agent_tasks WHERE task_id=?", (task_id,)).fetchone()
                if old is None:
                    raise KeyError("task_not_found")
                if str(old["state"]) not in {TaskState.AWAITING_APPROVAL.value, TaskState.AWAITING_INFORMATION.value, TaskState.PAUSED.value, TaskState.BLOCKED.value}:
                    raise ValueError("task_revision_state_invalid")
                if payload["version"] != int(old["plan_version"]) + 1 or payload["version"] > MAX_PLAN_REVISIONS:
                    raise ValueError("task_plan_revision_invalid")
                if (str(old["actor_id"]), str(old["session_id"]), str(old["thread_id"])) != (binding["actor_id"], binding["session_id"], binding["thread_id"]):
                    raise PermissionError("task_revision_binding_mismatch")
                existing = {
                    str(row["step_id"]): dict(row)
                    for row in self._conn.execute("SELECT * FROM agent_task_steps WHERE task_id=?", (task_id,)).fetchall()
                }
                retained = 0
                self._conn.execute("DELETE FROM agent_task_steps WHERE task_id=?", (task_id,))
                first_pending = len(payload["steps"])
                for step in payload["steps"]:
                    prior = existing.get(step["step_id"])
                    same_completed = bool(
                        prior
                        and str(prior.get("status") or "") == "completed"
                        and str(prior.get("capability_id") or "") == step["capability_id"]
                        and _hash(self._loads(prior.get("inputs_json"), {})) == _hash(step["inputs"])
                        and str(prior.get("verifier_status") or "") == "pass"
                    )
                    status = "completed" if same_completed else "pending"
                    if same_completed:
                        retained += 1
                    else:
                        first_pending = min(first_pending, int(step["position"]))
                    self._conn.execute(
                        "INSERT INTO agent_task_steps(task_id,step_id,position,capability_id,mode,approval_policy,status,inputs_json,attempts,invocation_id,result_json,evidence_json,verifier_status,failure_json,started_at,finished_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            task_id, step["step_id"], step["position"], step["capability_id"], step["mode"], step["approval_policy"], status, _canonical_json(step["inputs"]),
                            int(prior.get("attempts") or 0) if same_completed and prior else 0,
                            prior.get("invocation_id") if same_completed and prior else None,
                            prior.get("result_json") if same_completed and prior else None,
                            prior.get("evidence_json") if same_completed and prior else None,
                            prior.get("verifier_status") if same_completed and prior else None,
                            None, prior.get("started_at") if same_completed and prior else None,
                            prior.get("finished_at") if same_completed and prior else None,
                        ),
                    )
                self._conn.execute("UPDATE agent_task_approvals SET state='superseded' WHERE task_id=? AND state='pending'", (task_id,))
                self._conn.execute(
                    "UPDATE agent_tasks SET goal=?,success_criteria_json=?,plan_json=?,plan_hash=?,plan_version=?,state='proposed',revision=revision+1,current_step=?,failure_json=NULL,outcome_json=NULL,updated_at=? WHERE task_id=?",
                    (payload["goal"], _canonical_json(payload["success_criteria"]), _canonical_json(payload), payload["plan_hash"], payload["version"], first_pending, _now(), task_id),
                )
                self._append_event_locked(task_id, "task.plan_revised", {"plan_version": payload["version"], "plan_hash": payload["plan_hash"], "retained_completed_steps": retained, "approvals_invalidated": True})
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get(task_id) or {}

    def consume_approval(self, task_id: str, *, actor_id: str, thread_id: str, capability_id: str, inputs: Mapping[str, Any], health_state: str) -> dict[str, Any]:
        expired = False
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                task = self._conn.execute("SELECT * FROM agent_tasks WHERE task_id=?", (task_id,)).fetchone()
                row = self._conn.execute("SELECT * FROM agent_task_approvals WHERE task_id=? AND state='pending' ORDER BY expires_at DESC LIMIT 1", (task_id,)).fetchone()
                if task is None or row is None:
                    raise PermissionError("task_approval_missing")
                if str(task["state"]) != TaskState.AWAITING_APPROVAL.value:
                    raise PermissionError("task_approval_state_changed")
                if str(row["actor_id"]) != actor_id or str(row["thread_id"]) != thread_id:
                    raise PermissionError("task_approval_binding_mismatch")
                if str(row["plan_hash"]) != str(task["plan_hash"]) or int(row["plan_version"]) != int(task["plan_version"]):
                    raise PermissionError("task_approval_plan_changed")
                if datetime.fromisoformat(str(row["expires_at"])) <= datetime.now(timezone.utc):
                    self._conn.execute("UPDATE agent_task_approvals SET state='expired' WHERE approval_id=?", (row["approval_id"],))
                    now = _now()
                    self._conn.execute(
                        "UPDATE agent_tasks SET state='expired',revision=revision+1,updated_at=?,terminal_at=? WHERE task_id=? AND state='awaiting_approval'",
                        (now, now, task_id),
                    )
                    self._append_event_locked(task_id, "task.approval_expired", {"approval_id": row["approval_id"]})
                    self._prune_terminal_locked()
                    self._conn.commit()
                    expired = True
                if expired:
                    # Commit the durable expiry before returning a closed denial.
                    raise PermissionError("task_approval_expired")
                mutating_steps = self._loads(row["mutating_steps_json"], [])
                expected = mutating_steps[0] if len(mutating_steps) == 1 and isinstance(mutating_steps[0], dict) else {}
                if (
                    str(expected.get("capability_id") or "") != capability_id
                    or str(expected.get("inputs_hash") or "") != _hash(inputs)
                    or str(expected.get("mode") or "") != CapabilityMode.MUTATING.value
                    or str(expected.get("approval_policy") or "") != ApprovalPolicy.REQUIRED.value
                    or str(expected.get("health_state") or "") != health_state
                ):
                    raise PermissionError("task_approval_preconditions_changed")
                self._conn.execute("UPDATE agent_task_approvals SET state='consumed',consumed_at=? WHERE approval_id=? AND state='pending'", (_now(), row["approval_id"]))
                if int(self._conn.execute("SELECT changes() AS n").fetchone()["n"]) != 1:
                    raise PermissionError("task_approval_replayed")
                self._append_event_locked(task_id, "task.approval_consumed", {"approval_id": row["approval_id"], "binding_hash": row["binding_hash"]})
                self._conn.commit()
                return {"approval_id": str(row["approval_id"]), "binding_hash": str(row["binding_hash"])}
            except Exception:
                if not expired:
                    self._conn.rollback()
                raise

    @staticmethod
    def _loads(value: Any, fallback: Any) -> Any:
        try:
            return json.loads(str(value)) if value is not None else fallback
        except (TypeError, ValueError, json.JSONDecodeError):
            return fallback

    def get(self, task_id: str, *, actor_id: str | None = None, thread_id: str | None = None) -> dict[str, Any] | None:
        query = "SELECT * FROM agent_tasks WHERE task_id=?"
        params: list[Any] = [task_id]
        if actor_id is not None:
            query += " AND actor_id=?"
            params.append(actor_id)
        if thread_id is not None:
            query += " AND thread_id=?"
            params.append(thread_id)
        row = self._conn.execute(query, tuple(params)).fetchone()
        if row is None:
            return None
        task = dict(row)
        for key in ("success_criteria_json", "plan_json", "failure_json", "outcome_json"):
            task[key.removesuffix("_json")] = self._loads(task.pop(key), [] if key == "success_criteria_json" else None)
        steps = []
        for step_row in self._conn.execute("SELECT * FROM agent_task_steps WHERE task_id=? ORDER BY position", (task_id,)).fetchall():
            step = dict(step_row)
            for key in ("inputs_json", "result_json", "evidence_json", "failure_json"):
                step[key.removesuffix("_json")] = self._loads(step.pop(key), None)
            steps.append(step)
        task["steps"] = steps
        return task

    def list(self, *, actor_id: str, thread_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        query = "SELECT task_id FROM agent_tasks WHERE actor_id=?"
        params: list[Any] = [actor_id]
        if thread_id is not None:
            query += " AND thread_id=?"
            params.append(thread_id)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(min(50, max(1, int(limit))))
        return [task for row in self._conn.execute(query, tuple(params)).fetchall() if (task := self.get(str(row["task_id"]), actor_id=actor_id))]

    def active_for_thread(self, *, actor_id: str, thread_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT task_id FROM agent_tasks WHERE actor_id=? AND thread_id=? AND state NOT IN ('succeeded','partially_completed','failed','denied','cancelled','expired','indeterminate') ORDER BY updated_at DESC LIMIT 1",
            (actor_id, thread_id),
        ).fetchone()
        return self.get(str(row["task_id"]), actor_id=actor_id, thread_id=thread_id) if row else None

    def reconcile_startup(self) -> dict[str, int]:
        counts = {"read_only_resumable": 0, "mutations_indeterminate": 0, "approvals_retained": 0}
        with self._lock:
            rows = self._conn.execute("SELECT task_id,state FROM agent_tasks WHERE state IN ('running','verifying','awaiting_approval')").fetchall()
            for row in rows:
                task = self.get(str(row["task_id"])) or {}
                if str(row["state"]) == TaskState.AWAITING_APPROVAL.value:
                    counts["approvals_retained"] += 1
                    continue
                current = int(task.get("current_step") or 0)
                steps = task.get("steps") if isinstance(task.get("steps"), list) else []
                step = steps[current] if current < len(steps) else {}
                if str(step.get("mode") or "") == CapabilityMode.MUTATING.value and str(step.get("status") or "") in {"running", "dispatched"}:
                    self._conn.execute("UPDATE agent_tasks SET state='indeterminate',revision=revision+1,updated_at=?,terminal_at=? WHERE task_id=?", (_now(), _now(), task["task_id"]))
                    self._conn.execute("UPDATE agent_task_steps SET status='indeterminate',failure_json=? WHERE task_id=? AND step_id=?", (_canonical_json({"classification": "indeterminate", "reason": "restart_during_mutation"}), task["task_id"], step.get("step_id")))
                    counts["mutations_indeterminate"] += 1
                else:
                    self._conn.execute("UPDATE agent_tasks SET state='ready',revision=revision+1,updated_at=? WHERE task_id=?", (_now(), task["task_id"]))
                    counts["read_only_resumable"] += 1
            self._conn.commit()
        return counts


class TaskCoordinator:
    def __init__(self, *, store: TaskStore, registry: CapabilityRegistry) -> None:
        self.store = store
        self.registry = registry
        self.startup_reconciliation = store.reconcile_startup()

    @staticmethod
    def new_task_id() -> str:
        return f"task-{uuid.uuid4().hex[:20]}"

    def create(self, proposal: Mapping[str, Any], *, actor_id: str, session_id: str, thread_id: str) -> dict[str, Any]:
        plan = validate_plan(proposal, self.registry, actor_id=actor_id, session_id=session_id, thread_id=thread_id)
        return self.store.create(plan)

    def revise(self, proposal: Mapping[str, Any], *, actor_id: str, session_id: str, thread_id: str) -> dict[str, Any]:
        plan = validate_plan(proposal, self.registry, actor_id=actor_id, session_id=session_id, thread_id=thread_id)
        return self.store.revise(plan)

    @staticmethod
    def _result_payload(result: Any) -> dict[str, Any]:
        if isinstance(result, Mapping):
            text = ""
            data = dict(result)
        else:
            text = str(getattr(result, "text", "") or "").strip()
            data = getattr(result, "data", None)
        return {
            "response_text": _content_fingerprint(text),
            "data": _redacted(_audit_safe_result(data if isinstance(data, dict) else {})),
        }

    @staticmethod
    def _find_first_path(value: Any) -> str | None:
        if isinstance(value, Mapping):
            for key in ("path", "target_path", "resolved_path"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
            for nested in value.values():
                found = TaskCoordinator._find_first_path(nested)
                if found:
                    return found
        elif isinstance(value, list):
            for nested in value:
                found = TaskCoordinator._find_first_path(nested)
                if found:
                    return found
        return None

    def _resolve_step_inputs(self, task: Mapping[str, Any], step: Mapping[str, Any]) -> dict[str, Any]:
        inputs = dict(step.get("inputs")) if isinstance(step.get("inputs"), Mapping) else {}
        prior = {str(row.get("step_id") or ""): row for row in task.get("steps") if isinstance(row, Mapping)}
        for key, value in list(inputs.items()):
            if not isinstance(value, str) or not value.startswith("${"):
                continue
            match = _OUTPUT_REFERENCE_RE.fullmatch(value)
            if not match:
                raise ValueError("task_output_reference_invalid")
            source = prior.get(match.group(1)) or {}
            if str(source.get("status") or "") != "completed" or str(source.get("verifier_status") or "") != "pass":
                raise ValueError("task_output_reference_unverified")
            result = source.get("result")
            if match.group(2) == "first_path":
                resolved = self._find_first_path(result)
            else:
                resolved: Any = result
                for segment in match.group(2).split("."):
                    if isinstance(resolved, Mapping):
                        resolved = resolved.get(segment)
                    else:
                        resolved = None
                        break
            if resolved is None or isinstance(resolved, (dict, list)):
                raise ValueError("task_output_reference_unresolved")
            inputs[key] = resolved
        return inputs

    def run(self, task_id: str, *, actor_id: str, thread_id: str, defer_completion: bool = False) -> dict[str, Any]:
        task = self.store.get(task_id, actor_id=actor_id, thread_id=thread_id)
        if task is None:
            raise KeyError("task_not_found")
        state = TaskState(task["state"])
        if state is TaskState.PROPOSED:
            task = self.store.transition(task_id, TaskState.READY)
            state = TaskState.READY
        if state is TaskState.PAUSED:
            return task
        if state is not TaskState.READY:
            return task
        task = self.store.transition(task_id, TaskState.RUNNING)
        while int(task["current_step"]) < len(task["steps"]):
            created = datetime.fromisoformat(str(task["created_at"]))
            if (datetime.now(timezone.utc) - created).total_seconds() > MAX_TASK_WALL_SECONDS:
                failure = {"classification": "expired", "reason": "task_wall_time_limit"}
                self.store.set_failure_outcome(task_id, failure=failure)
                return self.store.transition(task_id, TaskState.EXPIRED, event="task.wall_time_expired", payload=failure)
            if int(task.get("capability_calls") or 0) >= MAX_CAPABILITY_CALLS:
                failure = {"classification": "policy_denial", "reason": "task_capability_call_limit"}
                self.store.set_failure_outcome(task_id, failure=failure)
                return self.store.transition(task_id, TaskState.FAILED, event="task.limit_reached", payload=failure)
            step = task["steps"][int(task["current_step"])]
            if str(step.get("status")) == "completed":
                self.store.advance_step(task_id)
                task = self.store.get(task_id) or task
                continue
            definition = self.registry.get(str(step["capability_id"]))
            if definition is None:
                failure = {"classification": "missing_capability", "reason": "planned_capability_no_longer_registered", "capability_id": str(step["capability_id"])}
                self.store.update_step(task_id, step["step_id"], status="blocked", failure=failure)
                self.store.set_failure_outcome(task_id, failure=failure, outcome={"status": "partial" if int(task["current_step"]) else "blocked"})
                target = TaskState.PARTIALLY_COMPLETED if int(task["current_step"]) else TaskState.BLOCKED
                return self.store.transition(task_id, target, event="task.capability_revoked", payload=failure)
            try:
                resolved_inputs = self._resolve_step_inputs(task, step)
                resolved_inputs = definition.input_contract.validate(resolved_inputs)
                if definition.task_input_validation_hook is not None:
                    task_inputs_ok, task_inputs_reason = definition.task_input_validation_hook(resolved_inputs)
                    if not task_inputs_ok:
                        raise ValueError(f"task_capability_inputs_invalid:{task_inputs_reason or definition.capability_id}")
                if resolved_inputs != dict(step.get("inputs") or {}):
                    self.store.update_step_inputs(task_id, step["step_id"], resolved_inputs)
                    task = self.store.get(task_id) or task
                    step = task["steps"][int(task["current_step"])]
            except ValueError as exc:
                failure = {"classification": "missing_information", "reason": str(exc), "capability_id": definition.capability_id}
                self.store.update_step(task_id, step["step_id"], status="blocked", failure=failure)
                self.store.set_failure_outcome(task_id, failure=failure)
                return self.store.transition(task_id, TaskState.BLOCKED, event="task.output_reference_blocked", payload=failure)
            health = definition.health()
            if not health.available and not definition.unavailable_invocation_safe:
                failure = {"classification": "unavailable_dependency", "reason": health.reason or "capability_unavailable", "capability_id": definition.capability_id}
                self.store.update_step(task_id, step["step_id"], status="blocked", failure=failure)
                self.store.set_failure_outcome(task_id, failure=failure, outcome={"status": "partial" if int(task["current_step"]) else "blocked"})
                target = TaskState.PARTIALLY_COMPLETED if int(task["current_step"]) else TaskState.BLOCKED
                return self.store.transition(task_id, target, event="task.dependency_blocked", payload=failure)
            if definition.mode is CapabilityMode.MUTATING:
                self.store.record_capability_call(task_id)
                preview = self.registry.preview_mutation(definition.capability_id, step["inputs"])
                preview_payload = self._result_payload(preview)
                approval = self.store.issue_approval(task_id, step=step, health_state=health.state.value)
                self.store.update_step(task_id, step["step_id"], status="awaiting_approval", result=preview_payload, verifier_status="preview_only", increment_attempt=True)
                task = self.store.transition(task_id, TaskState.AWAITING_APPROVAL, event="task.mutation_previewed", payload={"step_id": step["step_id"], "plan_hash": task["plan_hash"], "approval_binding_hash": approval["binding_hash"]})
                return task
            try:
                self.store.update_step(task_id, step["step_id"], status="running", increment_attempt=True)
                self.store.record_capability_call(task_id)
                result = self.registry.invoke(definition.capability_id, step["inputs"])
                registry_verified = bool(definition.verification_hook(result))
                independent = (
                    dict(definition.independent_verification_hook(step["inputs"], result))
                    if definition.independent_verification_hook is not None
                    else {
                        "ok": registry_verified and _declared_result_succeeded(result),
                        "kind": "declared_capability_outcome",
                        "reason": "read_only_result_success_fields_checked",
                    }
                )
                verified = registry_verified and bool(independent.get("ok"))
                evidence = {
                    "schema_version": TASK_SCHEMA_VERSION,
                    "capability_id": definition.capability_id,
                    "registry_verified": registry_verified,
                    "independent_observation": _redacted(independent),
                    "result_hash": _hash(self._result_payload(result)),
                    "recorded_at": _now(),
                }
                if not verified:
                    failure = {"classification": "verification_failure", "reason": "task_step_verifier_failed", "capability_id": definition.capability_id}
                    self.store.update_step(task_id, step["step_id"], status="failed", result=self._result_payload(result), evidence=evidence, verifier_status="fail", failure=failure)
                    self.store.set_failure_outcome(task_id, failure=failure)
                    return self.store.transition(task_id, TaskState.FAILED, event="task.verification_failed", payload=failure)
                self.store.update_step(task_id, step["step_id"], status="completed", result=self._result_payload(result), evidence=evidence, verifier_status="pass")
                self.store.advance_step(task_id)
                task = self.store.get(task_id) or task
            except Exception as exc:
                classification = (
                    "transient_failure"
                    if exc.__class__.__name__ in {"TimeoutError", "ConnectionError"}
                    else "verification_failure"
                    if str(exc) == "capability_result_verification_failed"
                    else "deterministic_failure"
                )
                failure = {"classification": classification, "reason": f"{exc.__class__.__name__}:{str(exc)[:200]}", "capability_id": definition.capability_id}
                refreshed = self.store.get(task_id) or task
                current_step = refreshed["steps"][int(refreshed["current_step"])]
                if (
                    classification == "transient_failure"
                    and definition.retry_safety in {"read_only_safe", "idempotent"}
                    and int(current_step.get("attempts") or 0) <= int((refreshed["plan"]["steps"][int(refreshed["current_step"])]).get("retry_limit") or 0)
                ):
                    self.store.update_step(task_id, step["step_id"], status="pending", failure=failure)
                    self.store.transition(task_id, TaskState.RECOVERING, event="task.transient_retry", payload={"step_id": step["step_id"], "attempt": current_step.get("attempts")})
                    task = self.store.transition(task_id, TaskState.RUNNING, event="task.retry_running")
                    continue
                self.store.update_step(task_id, step["step_id"], status="failed", verifier_status="fail", failure=failure)
                self.store.set_failure_outcome(task_id, failure=failure)
                return self.store.transition(task_id, TaskState.FAILED, event="task.execution_failed", payload=failure)
        task = self.store.transition(task_id, TaskState.VERIFYING)
        if defer_completion:
            return task
        all_verified = bool(task["steps"]) and all(str(step.get("verifier_status")) == "pass" for step in task["steps"])
        if not all_verified:
            failure = {"classification": "verification_failure", "reason": "task_level_evidence_incomplete"}
            self.store.set_failure_outcome(task_id, failure=failure)
            return self.store.transition(task_id, TaskState.FAILED, event="task.overall_verification_failed", payload=failure)
        evidence = [{"step_id": step["step_id"], "capability_id": step["capability_id"], "evidence": step.get("evidence")} for step in task["steps"]]
        outcome = {"schema_version": TASK_SCHEMA_VERSION, "status": "succeeded", "verified": True, "criteria_met": task["success_criteria"], "evidence": evidence}
        self.store.set_failure_outcome(task_id, outcome=outcome)
        return self.store.transition(task_id, TaskState.SUCCEEDED, event="task.verified_complete", payload={"evidence_count": len(evidence)})

    def control(self, task_id: str, *, action: str, actor_id: str, session_id: str, thread_id: str, expected_revision: int | None = None) -> dict[str, Any]:
        task = self.store.get(task_id, actor_id=actor_id, thread_id=thread_id)
        if task is None or str(task.get("session_id") or "") != session_id:
            raise PermissionError("task_binding_mismatch")
        action = str(action or "").strip().lower()
        revision = int(task["revision"]) if expected_revision is None else int(expected_revision)
        if action in {"cancel", "deny"}:
            for step in task["steps"]:
                if str(step.get("status")) in {"pending", "awaiting_approval"}:
                    self.store.update_step(task_id, step["step_id"], status="cancelled")
            target = TaskState.DENIED if action == "deny" and TaskState(task["state"]) is TaskState.AWAITING_APPROVAL else TaskState.CANCELLED
            return self.store.transition(task_id, target, expected_revision=revision, event=f"task.{target.value}_by_user", payload={"no_further_actions": True})
        if action == "pause" and TaskState(task["state"]) in {TaskState.READY, TaskState.RUNNING}:
            return self.store.transition(task_id, TaskState.PAUSED, expected_revision=revision, event="task.paused_by_user")
        if action == "resume" and TaskState(task["state"]) is TaskState.PAUSED:
            self.store.transition(task_id, TaskState.READY, expected_revision=revision, event="task.resumed_by_user")
            return self.run(task_id, actor_id=actor_id, thread_id=thread_id)
        raise ValueError("task_control_invalid_for_state")


def build_deterministic_plan(
    *,
    goal: str,
    capability_requests: list[tuple[str, Mapping[str, Any]]],
    actor_id: str,
) -> dict[str, Any]:
    task_id = TaskCoordinator.new_task_id()
    steps = []
    for index, (capability_id, inputs) in enumerate(capability_requests):
        steps.append({
            "step_id": f"step-{index + 1}",
            "capability_id": capability_id,
            "inputs": {**dict(inputs), "user_id": actor_id, "text": str(inputs.get("text") or goal)},
            "depends_on": [f"step-{index}"] if index else [],
            "expected_evidence": f"Verified result from {capability_id}",
            "verification": "registry_and_task",
            "retry_limit": 0,
            "timeout_seconds": 30,
        })
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "task_id": task_id,
        "version": 1,
        "goal": goal,
        "success_criteria": [f"Obtain verified evidence from {capability_id}" for capability_id, _ in capability_requests],
        "steps": steps,
        "created_by": "unified_semantic_decomposition",
        "planning_generation_count": 0,
    }


class GeneralTaskPlanner:
    """One-generation untrusted structured-plan proposer.

    Validation and execution are deliberately outside this class.
    """

    def __init__(self, route_inference_fn: Callable[..., dict[str, Any]] | None = None) -> None:
        self._route_inference = route_inference_fn or route_inference

    def propose(
        self,
        *,
        goal: str,
        actor_id: str,
        registry: CapabilityRegistry,
        llm_client: Any,
        trace_id: str,
    ) -> dict[str, Any]:
        if llm_client is None:
            return {"ok": False, "kind": "blocked", "error": "task_planner_llm_unavailable"}
        capabilities = [
            {
                "id": definition.capability_id,
                "description": definition.description,
                "inputs": definition.input_contract.public_schema(),
            }
            for definition in registry.definitions(chat_selectable_only=True)
            if definition.task_composable
        ]
        system = (
            "You are a bounded Personal Agent task proposer. Return exactly one JSON object and no prose. "
            "Model output is untrusted and never grants approval. Choose kind=plan only for a multi-step goal using two or more listed capabilities; "
            "kind=missing when the requested outcome cannot be completed with the listed capabilities; kind=clarify when material information is missing; "
            "Never emit a direct answer or completion claim; ordinary factual/casual answers use the existing chat path. "
            "Never emit shell commands, endpoints, URLs, approval, secrets, hidden reasoning, or evidence. "
            f"For plan, emit {{\"kind\":\"plan\",\"goal\":string,\"success_criteria\":[string],\"steps\":[{{\"step_id\":string,\"capability_id\":string,\"inputs\":object,\"depends_on\":[string],\"expected_evidence\":string,\"verification\":\"registry_and_task\",\"retry_limit\":0,\"timeout_seconds\":30}}]}}. "
            "For missing emit {\"kind\":\"missing\",\"missing_description\":string,\"success_criteria\":[string]}. "
            "For clarify emit {\"kind\":\"clarify\",\"question\":string}. "
            "Available capability registry: " + _canonical_json(capabilities)
        )
        result = self._route_inference(
            llm_client=llm_client,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": str(goal)[:2_000]}],
            user_text=str(goal)[:2_000],
            task_hint="bounded multi-capability task proposal",
            purpose="task_planning",
            task_type="planning",
            trace_id=trace_id,
            require_json=True,
            compute_tier="low",
            timeout_seconds=12.0,
            metadata={"component": "general_task_planner", "schema_version": PLAN_SCHEMA_VERSION},
        )
        if not bool(result.get("ok")):
            return {"ok": False, "kind": "blocked", "error": str(result.get("error_kind") or "task_planner_failed")}
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        raw = data.get("json") if isinstance(data.get("json"), dict) else None
        if raw is None:
            try:
                raw = json.loads(str(result.get("text") or ""))
            except (TypeError, ValueError, json.JSONDecodeError):
                raw = None
        if not isinstance(raw, dict):
            return {"ok": False, "kind": "blocked", "error": "task_planner_malformed_json"}
        kind = str(raw.get("kind") or "").strip().lower()
        allowed_by_kind = {
            "plan": {"kind", "goal", "success_criteria", "steps"},
            "missing": {"kind", "missing_description", "success_criteria"},
            "clarify": {"kind", "question"},
        }
        if kind not in allowed_by_kind or set(raw) - allowed_by_kind[kind]:
            return {"ok": False, "kind": "blocked", "error": "task_planner_schema_invalid"}
        if kind == "plan":
            steps = raw.get("steps") if isinstance(raw.get("steps"), list) else []
            if len(steps) < 2:
                return {"ok": False, "kind": "blocked", "error": "task_planner_not_substantial"}
            proposal = {
                "schema_version": PLAN_SCHEMA_VERSION,
                "task_id": TaskCoordinator.new_task_id(),
                "version": 1,
                "goal": str(raw.get("goal") or goal),
                "success_criteria": raw.get("success_criteria"),
                "steps": steps,
                "created_by": "configured_model_structured_proposal",
                "planning_generation_count": 1,
            }
            # Validation is repeated by coordinator creation; doing it here
            # makes the proposal boundary fail closed before persistence.
            validate_plan(proposal, registry, actor_id=actor_id, session_id="proposal", thread_id="proposal")
            return {"ok": True, "kind": "plan", "proposal": proposal, "planning_generations": 1}
        if kind == "missing":
            missing = build_missing_capability(
                goal=goal,
                success_criteria=[str(item) for item in (raw.get("success_criteria") if isinstance(raw.get("success_criteria"), list) else [goal])],
                missing_description=_bounded_text(raw.get("missing_description"), maximum=500, field="missing_capability"),
                considered_capabilities=[item.capability_id for item in registry.definitions(chat_selectable_only=True)],
            )
            return {"ok": True, "kind": "missing", "missing_capability": missing, "planning_generations": 1}
        return {
            "ok": True,
            "kind": "clarify",
            "question": _bounded_text(raw.get("question"), maximum=800, field="question"),
            "planning_generations": 1,
        }


__all__ = [
    "ALLOWED_TRANSITIONS", "MAX_CAPABILITY_CALLS", "MAX_CONCURRENT_TASKS_PER_ACTOR",
    "MAX_PLAN_REVISIONS", "MAX_PLAN_STEPS", "MAX_TASK_WALL_SECONDS", "MISSING_CAPABILITY_SCHEMA_VERSION",
    "PLAN_SCHEMA_VERSION", "TASK_SCHEMA_VERSION", "TERMINAL_STATES", "TaskCoordinator", "TaskState",
    "GeneralTaskPlanner", "TaskStore", "ValidatedPlan", "build_deterministic_plan", "build_missing_capability", "validate_plan",
]
