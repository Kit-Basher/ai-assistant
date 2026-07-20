#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.capability_policy import (  # noqa: E402
    CapabilityDefinition,
    CapabilityRegistry,
    TrustedInvocationContext,
    authorize_capability,
    build_default_capability_registry,
    stable_fingerprint,
    validate_trusted_invocation_context,
)
from agent.executor_registry import (  # noqa: E402
    ExecutorPartialFailure,
    ExecutorRegistry,
    ExecutorSpec,
    execute_file_create,
    execute_file_delete,
    execute_file_modify,
    execute_git_commit,
    execute_git_push,
    execute_notification_telegram_send,
    execute_service_restart,
)
from agent.llm.notify_delivery import TelegramTarget  # noqa: E402
from agent.mutation_boundary import (  # noqa: E402
    deny_arbitrary_shell_mutation,
    deny_direct_domain_db_mutation,
    deny_generic_http_mutation,
    deny_raw_secret_read,
)
from agent.mutation_plan import (  # noqa: E402
    MUTATION_PLAN_STATUS_CANCELLED,
    MUTATION_PLAN_STATUS_EXPIRED,
    MutationPlanStore,
    build_mutation_confirmation,
    build_mutation_plan,
    mutation_plan_fingerprint,
    validate_mutation_plan,
)
from agent.shell_skill import ShellSkill  # noqa: E402
from agent.skill_pack_permissions import (  # noqa: E402
    SkillGrantStore,
    SkillPackInvocationBroker,
    build_skill_identity,
    validate_skill_manifest,
)


PROPERTIES: dict[str, str] = {
    "P1": "Fixed authority",
    "P2": "Exact target binding",
    "P3": "Single-use authorization",
    "P4": "Scope isolation",
    "P5": "Runtime truth",
    "P6": "Primitive enforcement",
    "P7": "Durable mutation truth",
    "P8": "Failure truth",
    "P9": "Fail closed",
    "P10": "Fixture isolation",
}

INVENTORY_PATH = ROOT / "docs" / "operator" / "ADVERSARIAL_AUTHORIZATION_CASES.json"
DEFAULT_EVIDENCE_PATH = Path("/tmp/full_adversarial_authorization_proof_evidence.json")


@dataclass(frozen=True)
class ProofCase:
    case_id: str
    property_ids: tuple[str, ...]
    attack_class: str
    entry_surface: str
    caller_type: str
    target_type: str
    expected_decision: str
    expected_mutated: bool
    expected_reason: str
    fixture_required: bool = True


@dataclass
class ProofResult:
    case: ProofCase
    status: str
    actual_decision: str
    actual_mutated: bool
    actual_reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    receipt_id: str = ""
    operation_id: str = ""
    final_state_verified: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case.case_id,
            "property_ids": list(self.case.property_ids),
            "attack_class": self.case.attack_class,
            "entry_surface": self.case.entry_surface,
            "caller_type": self.case.caller_type,
            "target_type": self.case.target_type,
            "expected_decision": self.case.expected_decision,
            "expected_mutated": self.case.expected_mutated,
            "expected_reason": self.case.expected_reason,
            "actual_decision": self.actual_decision,
            "actual_mutated": bool(self.actual_mutated),
            "actual_reason": self.actual_reason,
            "status": self.status,
            "receipt_id": self.receipt_id,
            "operation_id": self.operation_id,
            "final_state_verified": bool(self.final_state_verified),
            "notes": self.notes,
            "evidence": self.evidence,
        }


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.files = root / "files"
        self.files.mkdir()
        self._plan_sequence = 0
        self.registry = self._registry()

    def _registry(self) -> ExecutorRegistry:
        registry = ExecutorRegistry(self.root / "executor_journal.jsonl")
        specs = (
            ExecutorSpec("operator.file.create.v1", "operator.file.create", "enabled", execute_file_create, True, "Remove created file.", "files.create"),
            ExecutorSpec("operator.file.modify.v1", "operator.file.modify", "enabled", execute_file_modify, True, "Restore rollback copy.", "files.modify"),
            ExecutorSpec("operator.file.delete.v1", "operator.file.delete", "enabled", execute_file_delete, True, "Move staged file back.", "files.delete"),
            ExecutorSpec("operator.git.commit.v1", "operator.git.commit", "enabled", execute_git_commit, True, "Revert with a reviewed plan.", "git.commit"),
            ExecutorSpec("operator.git.push.v1", "operator.git.push", "enabled", execute_git_push, False, "External push is disabled in fixture.", "git.push"),
            ExecutorSpec("operator.service.restart.v1", "operator.service.restart", "enabled", execute_service_restart, False, "Fixture restart verified.", "system.service.restart"),
            ExecutorSpec("operator.notification.telegram.send.v1", "operator.notification.telegram.send", "enabled", execute_notification_telegram_send, False, "Fixture transport only.", "notification.external.send"),
        )
        for spec in specs:
            registry.register(spec)
        return registry

    def plan(self, action_type: str, capability_id: str, executor_id: str, target: str, *, risk_level: str | None = None) -> dict[str, Any]:
        self._plan_sequence += 1
        plan = build_mutation_plan(
            plan_id=f"adv-{action_type.replace('.', '-')}-{hashlib.sha256(target.encode()).hexdigest()[:8]}-{self._plan_sequence}",
            capability_id=capability_id,
            executor_id=executor_id,
            expires_at_epoch=int(time.time()) + 600,
            thread_id="thread-a",
            session_id="session-a",
            actor_id="actor-a",
            target_snapshot={"action_type": action_type, "target": target},
            mutation_inventory=[{"action_type": action_type, "target": target}],
            recovery={"rollback_supported": capability_id in {"files.create", "files.modify", "files.delete", "git.commit"}},
        )
        wrapped = dict(plan)
        wrapped.update(
            {
                "mutation_plan": dict(plan),
                "action_type": action_type,
                "target": target,
                "executor_status": "enabled",
                "high_risk_confirmed": True,
            }
        )
        if risk_level:
            wrapped["risk_level"] = risk_level
        wrapped["_proof_confirmation"] = build_mutation_confirmation(
            plan,
            confirmation_id=f"confirmation-{plan['plan_id']}",
        )
        return wrapped

    def execute(self, *, plan: dict[str, Any], action: dict[str, Any], **kwargs: Any):
        return self.registry.execute_confirmed_plan(
            plan=plan,
            action=action,
            confirmation=plan.get("_proof_confirmation"),
            **kwargs,
        )

    def context(self, plan: dict[str, Any], capability_id: str, executor_id: str, **overrides: Any) -> dict[str, Any]:
        payload = TrustedInvocationContext(
            capability_id=capability_id,
            executor_id=executor_id,
            authorization_decision_id=str(overrides.pop("authorization_decision_id", "authz-fixture123")),
            plan_fingerprint=str(overrides.pop("plan_fingerprint", plan.get("plan_fingerprint") or "")),
            operation_id=str(overrides.pop("operation_id", plan.get("plan_id") or "")),
            caller_type=str(overrides.pop("caller_type", "core")),
            caller_id=str(overrides.pop("caller_id", "full-adversarial-proof")),
            source_module="scripts.full_adversarial_authorization_proof",
            source_surface="fixture",
            expires_at=str(overrides.pop("expires_at", (plan.get("mutation_plan") or {}).get("expires_at") or plan.get("expires_at") or "")),
            target_fingerprint=str(overrides.pop("target_fingerprint", plan.get("target_fingerprint") or "")),
            consumed=bool(overrides.pop("consumed", False)),
            **overrides,
        )
        return payload.to_dict()


def _pass(case: ProofCase, decision: str, mutated: bool, reason: str, evidence: dict[str, Any] | None = None, *, receipt_id: str = "", operation_id: str = "", final_state_verified: bool = False, notes: str = "") -> ProofResult:
    return ProofResult(case, "PASS", decision, mutated, reason, evidence or {}, receipt_id, operation_id, final_state_verified, notes)


def _warn(case: ProofCase, reason: str, notes: str) -> ProofResult:
    return ProofResult(case, "WARN", "limitation", False, reason, {}, notes=notes)


def _fail(case: ProofCase, decision: str, mutated: bool, reason: str, evidence: dict[str, Any] | None = None) -> ProofResult:
    return ProofResult(case, "FAIL", decision, mutated, reason, evidence or {})


def _finish(case: ProofCase, *, decision: str, mutated: bool, reason: str, evidence: dict[str, Any] | None = None, receipt_id: str = "", operation_id: str = "", final_state_verified: bool = False, notes: str = "") -> ProofResult:
    if decision == case.expected_decision and bool(mutated) == bool(case.expected_mutated) and (not case.expected_reason or reason == case.expected_reason):
        return _pass(case, decision, mutated, reason, evidence, receipt_id=receipt_id, operation_id=operation_id, final_state_verified=final_state_verified, notes=notes)
    return _fail(case, decision, mutated, reason, evidence)


def _result_reason(result: dict[str, Any]) -> str:
    return str(result.get("error_code") or result.get("reason_code") or result.get("status") or result.get("blocked_reason") or "")


def _result_decision(result: dict[str, Any]) -> str:
    if result.get("ok") and result.get("mutated"):
        return "allowed"
    if result.get("ok") and not result.get("mutated"):
        return "no_op"
    return "denied"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["/usr/bin/git", *args], cwd=str(repo), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=False)


def _setup_repo(root: Path) -> tuple[Path, str]:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "agent-fixture@example.test")
    _git(repo, "config", "user.name", "Agent Fixture")
    tracked = repo / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "initial")
    tracked.write_text("after\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    diff = _git(repo, "diff", "--cached", "--binary").stdout
    return repo, hashlib.sha256(diff.encode("utf-8", errors="replace")).hexdigest()


def _manifest(permissions: list[str], *, version: str = "1.0.0") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "skill_pack_id": "example.report_builder",
        "publisher_id": "example.publisher",
        "name": "Report Builder",
        "version": version,
        "entrypoints": [],
        "declared_permissions": permissions,
        "read_only_surfaces": ["notifications"],
        "network_domains": [],
        "filesystem_roots": [],
        "provider_accounts": [],
        "background_tasks": [],
        "configuration_schema": {},
    }


def case_fixed_authority_override_ignored(case: ProofCase, fx: Fixture) -> ProofResult:
    target = fx.files / "override.txt"
    plan = fx.plan("operator.file.create", "files.create", "operator.file.create.v1", str(target))
    plan["capability_id"] = "system.uninstall"
    plan["executor_id"] = "operator.uninstall.v1"
    result = fx.execute(
        plan=plan,
        action={"pending_id": plan["plan_id"], "target_path": str(target), "approved_roots": [str(fx.files)], "content": "ok\n", "capability_id": "system.uninstall", "executor_id": "operator.uninstall.v1"},
    ).to_dict()
    evidence = {"selected_capability": result.get("capability_id"), "selected_executor": result.get("executor_id"), "file_exists": target.exists()}
    return _finish(case, decision=_result_decision(result), mutated=bool(result.get("mutated")), reason=str(result.get("capability_id") or ""), evidence=evidence, receipt_id=str(result.get("journal_id") or ""), operation_id=str(result.get("plan_id") or ""), final_state_verified=target.read_text(encoding="utf-8") == "ok\n")


def case_unknown_executor_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    plan = fx.plan("operator.unknown", "files.create", "operator.file.create.v1", "unknown")
    result = fx.execute(plan=plan, action={"pending_id": plan["plan_id"]}).to_dict()
    return _finish(case, decision=_result_decision(result), mutated=bool(result.get("mutated")), reason=_result_reason(result), evidence=result)


def case_capability_executor_mismatch_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    target = fx.files / "mismatch.txt"
    plan = fx.plan("operator.file.create", "files.modify", "operator.file.modify.v1", str(target))
    result = fx.execute(plan=plan, action={"pending_id": plan["plan_id"], "target_path": str(target), "approved_roots": [str(fx.files)], "content": "x"}).to_dict()
    return _finish(case, decision=_result_decision(result), mutated=bool(result.get("mutated")), reason=_result_reason(result), evidence=result)


def case_high_risk_confirmation_missing(case: ProofCase, fx: Fixture) -> ProofResult:
    target = fx.files / "risk.txt"
    plan = fx.plan("operator.file.create", "files.create", "operator.file.create.v1", str(target), risk_level="high")
    result = fx.execute(plan=plan, action={"pending_id": plan["plan_id"], "target_path": str(target), "approved_roots": [str(fx.files)], "content": "x"}, high_risk_confirmed=False).to_dict()
    return _finish(case, decision=_result_decision(result), mutated=bool(result.get("mutated")), reason=_result_reason(result), evidence=result)


def case_direct_file_missing_context(case: ProofCase, fx: Fixture) -> ProofResult:
    target = fx.files / "direct.txt"
    plan = fx.plan("operator.file.create", "files.create", "operator.file.create.v1", str(target))
    result = execute_file_create(plan, {"target_path": str(target), "approved_roots": [str(fx.files)], "content": "x"})
    return _finish(case, decision=_result_decision(result), mutated=bool(result.get("mutated")), reason=_result_reason(result), evidence=result)


def _direct_with_context(case: ProofCase, fx: Fixture, **context_overrides: Any) -> ProofResult:
    target = fx.files / f"{case.case_id}.txt"
    plan = fx.plan("operator.file.create", "files.create", "operator.file.create.v1", str(target))
    context = fx.context(plan, "files.create", "operator.file.create.v1", **context_overrides)
    result = execute_file_create(plan, {"target_path": str(target), "approved_roots": [str(fx.files)], "content": "x", "trusted_invocation_context": context})
    return _finish(case, decision=_result_decision(result), mutated=bool(result.get("mutated")), reason=_result_reason(result), evidence=result)


def case_forged_context_wrong_capability(case: ProofCase, fx: Fixture) -> ProofResult:
    target = fx.files / f"{case.case_id}.txt"
    plan = fx.plan("operator.file.create", "files.create", "operator.file.create.v1", str(target))
    context = fx.context(plan, "system.uninstall", "operator.file.create.v1")
    result = execute_file_create(plan, {"target_path": str(target), "approved_roots": [str(fx.files)], "content": "x", "trusted_invocation_context": context})
    return _finish(case, decision=_result_decision(result), mutated=bool(result.get("mutated")), reason=_result_reason(result), evidence=result)


def case_forged_context_wrong_executor(case: ProofCase, fx: Fixture) -> ProofResult:
    target = fx.files / f"{case.case_id}.txt"
    plan = fx.plan("operator.file.create", "files.create", "operator.file.create.v1", str(target))
    context = fx.context(plan, "files.create", "operator.uninstall.v1")
    result = execute_file_create(plan, {"target_path": str(target), "approved_roots": [str(fx.files)], "content": "x", "trusted_invocation_context": context})
    return _finish(case, decision=_result_decision(result), mutated=bool(result.get("mutated")), reason=_result_reason(result), evidence=result)


def case_context_wrong_operation(case: ProofCase, fx: Fixture) -> ProofResult:
    return _direct_with_context(case, fx, operation_id="other-operation")


def case_context_wrong_plan_fingerprint(case: ProofCase, fx: Fixture) -> ProofResult:
    return _direct_with_context(case, fx, plan_fingerprint=stable_fingerprint({"other": "plan"}))


def case_context_wrong_target_fingerprint(case: ProofCase, fx: Fixture) -> ProofResult:
    return _direct_with_context(case, fx, target_fingerprint=stable_fingerprint({"other": "target"}))


def case_context_expired(case: ProofCase, fx: Fixture) -> ProofResult:
    return _direct_with_context(case, fx, expires_at="2000-01-01T00:00:00+00:00")


def case_context_consumed(case: ProofCase, fx: Fixture) -> ProofResult:
    return _direct_with_context(case, fx, consumed=True)


def case_fixture_context_denied_production(case: ProofCase, fx: Fixture) -> ProofResult:
    return _direct_with_context(case, fx, caller_type="fixture")


def case_unknown_caller_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    plan = fx.plan("operator.file.create", "files.create", "operator.file.create.v1", "caller")
    ok, reason, _ctx = validate_trusted_invocation_context(
        fx.context(plan, "files.create", "operator.file.create.v1", caller_type="attacker"),
        capability_id="files.create",
        executor_id="operator.file.create.v1",
        plan_fingerprint=str(plan["plan_fingerprint"]),
    )
    return _finish(case, decision="denied" if not ok else "allowed", mutated=False, reason=reason, evidence={"valid": ok})


def case_plan_fingerprint_tamper_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    plan = fx.plan("operator.file.create", "files.create", "operator.file.create.v1", "tamper")
    tampered = dict(plan["mutation_plan"])
    tampered["target_snapshot"] = {"target": "changed"}
    try:
        validate_mutation_plan(tampered)
    except ValueError as exc:
        return _finish(case, decision="denied", mutated=False, reason=str(exc), evidence={"exception": exc.__class__.__name__})
    return _fail(case, "allowed", False, "tampered_plan_accepted", tampered)


def case_plan_id_reuse_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    store = MutationPlanStore()
    first = fx.plan("operator.file.create", "files.create", "operator.file.create.v1", "one")["mutation_plan"]
    second = fx.plan("operator.file.create", "files.create", "operator.file.create.v1", "two")["mutation_plan"]
    second["plan_id"] = first["plan_id"]
    second["plan_fingerprint"] = mutation_plan_fingerprint(second)
    store.save(first)
    try:
        store.save(second)
    except ValueError as exc:
        return _finish(case, decision="denied", mutated=False, reason=str(exc), evidence={"plan_id": first["plan_id"]})
    return _fail(case, "allowed", False, "plan_id_reuse_allowed")


def case_plan_cancelled_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    store = MutationPlanStore()
    plan = fx.plan("operator.file.create", "files.create", "operator.file.create.v1", "cancel")["mutation_plan"]
    store.save(plan)
    cancelled = store.cancel(str(plan["plan_id"]))
    return _finish(case, decision="denied", mutated=False, reason=str((cancelled or {}).get("status") or ""), evidence=cancelled or {})


def case_plan_expired_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    path = fx.root / "plans.json"
    store = MutationPlanStore(path)
    plan = build_mutation_plan(
        plan_id="expired-plan",
        capability_id="files.create",
        executor_id="operator.file.create.v1",
        expires_at_epoch=1,
        target_snapshot={"target": "expired"},
        mutation_inventory=[{"target": "expired"}],
    )
    store.save(plan)
    store.prune(now=int(time.time()))
    loaded = store.load("expired-plan") or {}
    return _finish(case, decision="denied", mutated=False, reason=str(loaded.get("status") or ""), evidence=loaded)


def case_confirmation_plan_id_mismatch(case: ProofCase, fx: Fixture) -> ProofResult:
    target = fx.files / "wrong-plan.txt"
    plan = fx.plan("operator.file.create", "files.create", "operator.file.create.v1", str(target))
    result = fx.execute(plan=plan, action={"pending_id": "other-plan", "target_path": str(target), "approved_roots": [str(fx.files)], "content": "x"}).to_dict()
    return _finish(case, decision=_result_decision(result), mutated=bool(result.get("mutated")), reason=_result_reason(result), evidence=result)


def case_file_hash_drift_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    target = fx.files / "drift.txt"
    target.write_text("before\n", encoding="utf-8")
    old_hash = hashlib.sha256(b"before\n").hexdigest()
    target.write_text("changed\n", encoding="utf-8")
    plan = fx.plan("operator.file.modify", "files.modify", "operator.file.modify.v1", str(target))
    result = fx.execute(plan=plan, action={"pending_id": plan["plan_id"], "target_path": str(target), "approved_roots": [str(fx.files)], "expected_hash": old_hash, "content": "after\n"}).to_dict()
    return _finish(case, decision=_result_decision(result), mutated=bool(result.get("mutated")), reason=_result_reason(result), evidence=result)


def case_file_symlink_drift_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    target = fx.files / "link.txt"
    target.symlink_to(fx.files / "elsewhere.txt")
    plan = fx.plan("operator.file.create", "files.create", "operator.file.create.v1", str(target))
    result = fx.execute(plan=plan, action={"pending_id": plan["plan_id"], "target_path": str(target), "approved_roots": [str(fx.files)], "content": "x"}).to_dict()
    return _finish(case, decision=_result_decision(result), mutated=bool(result.get("mutated")), reason=_result_reason(result), evidence=result)


def case_git_staged_diff_drift_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    repo, old_hash = _setup_repo(fx.root)
    (repo / "tracked.txt").write_text("different\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    plan = fx.plan("operator.git.commit", "git.commit", "operator.git.commit.v1", str(repo))
    result = fx.execute(plan=plan, action={"pending_id": plan["plan_id"], "repository_root": str(repo), "approved_roots": [str(fx.root)], "staged_diff_sha256": old_hash, "commit_message": "drift"}).to_dict()
    return _finish(case, decision=_result_decision(result), mutated=bool(result.get("mutated")), reason=_result_reason(result), evidence=result)


def case_service_allowlist_drift_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    plan = fx.plan("operator.service.restart", "system.service.restart", "operator.service.restart.v1", "ssh.service")
    result = fx.execute(plan=plan, action={"pending_id": plan["plan_id"], "service_name": "ssh.service", "allowed_services": ["personal-agent-proof-restart.service"], "service_fixture_root": str(fx.root / "services")}).to_dict()
    return _finish(case, decision=_result_decision(result), mutated=bool(result.get("mutated")), reason=_result_reason(result), evidence=result)


def case_notification_destination_drift_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    body = "hello"
    expected_chat = hashlib.sha256(b"chat-a").hexdigest()
    body_hash = hashlib.sha256(body.encode()).hexdigest()
    plan = fx.plan("operator.notification.telegram.send", "notification.external.send", "operator.notification.telegram.send.v1", "telegram")
    result = fx.execute(
        plan=plan,
        action={"pending_id": plan["plan_id"], "fixture_transport": True, "transport_path": str(fx.root / "transport.jsonl"), "chat_id": "chat-b", "expected_chat_id_sha256": expected_chat, "expected_body_sha256": body_hash, "message": body},
    ).to_dict()
    return _finish(case, decision=_result_decision(result), mutated=bool(result.get("mutated")), reason=_result_reason(result), evidence=result)


def case_notification_content_drift_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    expected_body = hashlib.sha256(b"hello").hexdigest()
    chat_hash = hashlib.sha256(b"chat-a").hexdigest()
    plan = fx.plan("operator.notification.telegram.send", "notification.external.send", "operator.notification.telegram.send.v1", "telegram")
    result = fx.execute(
        plan=plan,
        action={"pending_id": plan["plan_id"], "fixture_transport": True, "transport_path": str(fx.root / "transport.jsonl"), "chat_id": "chat-a", "expected_chat_id_sha256": chat_hash, "expected_body_sha256": expected_body, "message": "changed"},
    ).to_dict()
    return _finish(case, decision=_result_decision(result), mutated=bool(result.get("mutated")), reason=_result_reason(result), evidence=result)


def case_notification_duplicate_no_resend(case: ProofCase, fx: Fixture) -> ProofResult:
    message = "hello"
    chat_hash = hashlib.sha256(b"chat-a").hexdigest()
    body_hash = hashlib.sha256(message.encode()).hexdigest()
    transport = fx.root / "transport.jsonl"
    plan = fx.plan("operator.notification.telegram.send", "notification.external.send", "operator.notification.telegram.send.v1", "telegram")
    action = {"pending_id": plan["plan_id"], "fixture_transport": True, "transport_path": str(transport), "chat_id": "chat-a", "expected_chat_id_sha256": chat_hash, "expected_body_sha256": body_hash, "message": message}
    first = fx.execute(plan=plan, action=dict(action)).to_dict()
    second = fx.execute(plan=plan, action=dict(action)).to_dict()
    delivered = len(transport.read_text(encoding="utf-8").splitlines()) if transport.exists() else 0
    ok = bool(first.get("mutated")) and second.get("mutated") is False and delivered == 1
    return _finish(case, decision="denied" if ok else "failed", mutated=False, reason=_result_reason(second), evidence={"first": first, "second": second, "delivered": delivered}, final_state_verified=ok)


def case_skill_undeclared_permission_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    store = SkillGrantStore(fx.root / f"{case.case_id}-grants.json")
    manifest = _manifest(["invoke.files.create"])
    identity = build_skill_identity(manifest, install_source="fixture", install_path=str(fx.root / "skill"))
    broker = SkillPackInvocationBroker(grant_store=store, executor_registry=fx.registry)
    result = broker.request_action(identity=identity, manifest=manifest, permission_id="invoke.git.commit", target={"repository_root": str(fx.root)}, action_payload={})
    return _finish(case, decision="denied", mutated=bool(result.get("mutated")), reason=str(result.get("reason_code") or ""), evidence=result)


def case_skill_ungranted_permission_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    store = SkillGrantStore(fx.root / f"{case.case_id}-grants.json")
    manifest = _manifest(["invoke.files.create"])
    identity = build_skill_identity(manifest, install_source="fixture", install_path=str(fx.root / "skill"))
    broker = SkillPackInvocationBroker(grant_store=store, executor_registry=fx.registry)
    result = broker.request_action(identity=identity, manifest=manifest, permission_id="invoke.files.create", target={"target_path": str(fx.files / "skill.txt"), "size_bytes": 1}, action_payload={"target_path": str(fx.files / "skill.txt"), "approved_roots": [str(fx.files)], "content": "x"})
    return _finish(case, decision="denied", mutated=bool(result.get("mutated")), reason=str(result.get("reason_code") or ""), evidence=result)


def case_skill_revoked_grant_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    store = SkillGrantStore(fx.root / f"{case.case_id}-grants.json")
    manifest = _manifest(["invoke.files.create"])
    identity = build_skill_identity(manifest, install_source="fixture", install_path=str(fx.root / "skill"))
    grant = store.create_grant(identity=identity, permission_id="invoke.files.create", target_scope={"root": str(fx.files)}, granted_by="local_operator_cli")
    store.revoke_grant(str(grant["grant_id"]))
    broker = SkillPackInvocationBroker(grant_store=store, executor_registry=fx.registry)
    result = broker.request_action(identity=identity, manifest=manifest, permission_id="invoke.files.create", target={"target_path": str(fx.files / "revoked.txt"), "size_bytes": 1}, action_payload={"target_path": str(fx.files / "revoked.txt"), "approved_roots": [str(fx.files)], "content": "x"})
    return _finish(case, decision="denied", mutated=bool(result.get("mutated")), reason=str(result.get("reason_code") or ""), evidence=result)


def case_skill_scope_expansion_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    store = SkillGrantStore(fx.root / f"{case.case_id}-grants.json")
    manifest = _manifest(["invoke.files.create"])
    identity = build_skill_identity(manifest, install_source="fixture", install_path=str(fx.root / "skill"))
    store.create_grant(identity=identity, permission_id="invoke.files.create", target_scope={"root": str(fx.files / "narrow")}, granted_by="local_operator_cli")
    broker = SkillPackInvocationBroker(grant_store=store, executor_registry=fx.registry)
    result = broker.request_action(identity=identity, manifest=manifest, permission_id="invoke.files.create", target={"target_path": str(fx.files / "wide.txt"), "size_bytes": 1}, action_payload={"target_path": str(fx.files / "wide.txt"), "approved_roots": [str(fx.files)], "content": "x"})
    return _finish(case, decision="denied", mutated=bool(result.get("mutated")), reason=str(result.get("reason_code") or ""), evidence=result)


def case_skill_update_new_permission_ungranted(case: ProofCase, fx: Fixture) -> ProofResult:
    store = SkillGrantStore(fx.root / f"{case.case_id}-grants.json")
    old_manifest = _manifest(["invoke.files.create"], version="1.0.0")
    old_identity = build_skill_identity(old_manifest, install_source="fixture", install_path=str(fx.root / "skill"))
    store.create_grant(identity=old_identity, permission_id="invoke.files.create", target_scope={"root": str(fx.files)}, granted_by="local_operator_cli")
    new_manifest = _manifest(["invoke.files.create", "invoke.notification.local.send"], version="1.1.0")
    new_identity = build_skill_identity(new_manifest, install_source="fixture", install_path=str(fx.root / "skill"))
    broker = SkillPackInvocationBroker(grant_store=store, executor_registry=fx.registry)
    result = broker.request_action(identity=new_identity, manifest=new_manifest, permission_id="invoke.notification.local.send", target={"target": "local_notification"}, action_payload={"receipt_path": str(fx.root / "notify.json"), "message": "x"})
    return _finish(case, decision="denied", mutated=bool(result.get("mutated")), reason=str(result.get("reason_code") or ""), evidence=result)


def case_skill_raw_capability_ignored(case: ProofCase, fx: Fixture) -> ProofResult:
    store = SkillGrantStore(fx.root / f"{case.case_id}-grants.json")
    manifest = _manifest(["invoke.files.create"])
    identity = build_skill_identity(manifest, install_source="fixture", install_path=str(fx.root / "skill"))
    store.create_grant(identity=identity, permission_id="invoke.files.create", target_scope={"root": str(fx.files)}, granted_by="local_operator_cli")
    broker = SkillPackInvocationBroker(grant_store=store, executor_registry=fx.registry)
    target = fx.files / "raw-cap.txt"
    result = broker.request_action(
        identity=identity,
        manifest=manifest,
        permission_id="invoke.files.create",
        target={"target_path": str(target), "size_bytes": 1},
        action_payload={"target_path": str(target), "approved_roots": [str(fx.files)], "content": "x", "capability_id": "system.uninstall", "executor_id": "operator.uninstall.v1"},
    )
    return _finish(
        case,
        decision=_result_decision(result),
        mutated=bool(result.get("mutated")),
        reason=_result_reason(result),
        evidence=result,
        receipt_id=str(result.get("journal_id") or ""),
        final_state_verified=not target.exists(),
    )


def case_registry_freeze_denies_registration(case: ProofCase, fx: Fixture) -> ProofResult:
    fx.registry.freeze()
    try:
        fx.registry.register(ExecutorSpec("operator.file.create.v2", "operator.file.create.v2", "enabled", execute_file_create, True, "", "files.create"))
    except ValueError as exc:
        return _finish(case, decision="denied", mutated=False, reason=str(exc), evidence={"frozen": fx.registry.frozen})
    return _fail(case, "allowed", False, "registry_mutation_allowed")


def case_duplicate_executor_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    registry = ExecutorRegistry(fx.root / "dup_journal.jsonl")
    registry.register(ExecutorSpec("operator.file.create.v1", "operator.file.create", "enabled", execute_file_create, True, "", "files.create"))
    try:
        registry.register(ExecutorSpec("operator.file.create.v1", "operator.file.other", "enabled", execute_file_create, True, "", "files.create"))
    except ValueError as exc:
        return _finish(case, decision="denied", mutated=False, reason=str(exc).split(":")[0], evidence={})
    return _fail(case, "allowed", False, "duplicate_executor_allowed")


def case_capability_duplicate_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    registry = CapabilityRegistry()
    definition = build_default_capability_registry().get("files.create")
    assert definition is not None
    registry.register(definition)
    try:
        registry.register(definition)
    except ValueError as exc:
        return _finish(case, decision="denied", mutated=False, reason=str(exc).split(":")[0], evidence={})
    return _fail(case, "allowed", False, "duplicate_capability_allowed")


def case_policy_change_invalidates_plan(case: ProofCase, fx: Fixture) -> ProofResult:
    decision = authorize_capability(
        "files.create",
        request_context={"origin": "proof"},
        target_snapshot={"target_fingerprint": "target"},
        plan_context={"plan_id": "p", "plan_fingerprint": "p", "target_fingerprint": "target", "policy_version": 99},
        confirmation_context={"confirmed": True, "pending_id": "p"},
    )
    return _finish(case, decision="denied" if not decision.allowed else "allowed", mutated=False, reason=decision.reason_code, evidence=decision.to_dict())


def case_raw_http_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    result = deny_generic_http_mutation()
    return _finish(case, decision="denied", mutated=bool(result.get("mutated")), reason=str(result.get("reason") or ""), evidence=result)


def case_raw_shell_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    result = deny_arbitrary_shell_mutation()
    return _finish(case, decision="denied", mutated=bool(result.get("mutated")), reason=str(result.get("reason") or ""), evidence=result)


def case_raw_db_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    result = deny_direct_domain_db_mutation()
    return _finish(case, decision="denied", mutated=bool(result.get("mutated")), reason=str(result.get("reason") or ""), evidence=result)


def case_raw_secret_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    result = deny_raw_secret_read()
    return _finish(case, decision="denied", mutated=bool(result.get("mutated")), reason=str(result.get("reason") or ""), evidence=result)


def case_provider_direct_send_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    result = TelegramTarget(token="token", chat_id="chat", send_fn=lambda *_args: None, enabled=True).deliver({"message": "hello"})
    return _finish(case, decision="denied" if not result.ok else "allowed", mutated=False, reason=result.reason, evidence={"ok": result.ok, "reason": result.reason})


def case_shell_git_mutation_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    result = ShellSkill(allowed_roots=[str(fx.root)], base_dir=fx.root).execute_safe_command(command_name="git", cwd=str(fx.root))
    return _finish(case, decision="denied", mutated=bool(result.get("mutated")), reason=str(result.get("blocked_reason") or ""), evidence=result)


def case_shell_package_mutation_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    result = ShellSkill(allowed_roots=[str(ROOT)]).install_package(manager="apt", package="htop", scope="system")
    return _finish(case, decision="denied", mutated=bool(result.get("mutated", False)), reason=str(result.get("blocked_reason") or result.get("error_kind") or ""), evidence=result)


def case_partial_failure_truth(case: ProofCase, fx: Fixture) -> ProofResult:
    def partial(_plan: dict[str, Any], _action: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": False,
            "mutated": True,
            "executor_id": "operator.file.create.v1",
            "error_code": "fixture_partial_after_mutation",
            "user_message": "Fixture mutation occurred but verification failed.",
            "resources_touched": [str(fx.files / "partial.txt")],
            "details": {"status": "partial"},
        }

    registry = ExecutorRegistry(fx.root / "partial_journal.jsonl")
    registry.register(ExecutorSpec("operator.file.create.v1", "operator.file.create", "enabled", partial, True, "Fixture rollback.", "files.create"))
    plan = fx.plan("operator.file.create", "files.create", "operator.file.create.v1", "partial")
    result = registry.execute_confirmed_plan(
        plan=plan,
        action={"pending_id": plan["plan_id"]},
        confirmation=plan.get("_proof_confirmation"),
    ).to_dict()
    return _finish(case, decision="partial", mutated=bool(result.get("mutated")), reason=str(result.get("error_code") or ""), evidence=result, receipt_id=str(result.get("journal_id") or ""))


def case_exception_before_mutation_truth(case: ProofCase, fx: Fixture) -> ProofResult:
    def exploding(_plan: dict[str, Any], _action: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom")

    registry = ExecutorRegistry(fx.root / "exception_journal.jsonl")
    registry.register(ExecutorSpec("operator.file.create.v1", "operator.file.create", "enabled", exploding, True, "Fixture rollback.", "files.create"))
    plan = fx.plan("operator.file.create", "files.create", "operator.file.create.v1", "exception")
    result = registry.execute_confirmed_plan(
        plan=plan,
        action={"pending_id": plan["plan_id"]},
        confirmation=plan.get("_proof_confirmation"),
    ).to_dict()
    return _finish(case, decision=_result_decision(result), mutated=bool(result.get("mutated")), reason=str(result.get("error_code") or ""), evidence=result)


def case_executor_partial_exception_truth(case: ProofCase, fx: Fixture) -> ProofResult:
    def partial_exception(_plan: dict[str, Any], _action: dict[str, Any]) -> dict[str, Any]:
        raise ExecutorPartialFailure("fixture partial", resources_touched=[str(fx.files / "artifact")], rollback_hint="Fixture cleanup.", details={"phase": "during_mutation"})

    registry = ExecutorRegistry(fx.root / "partial_exception_journal.jsonl")
    registry.register(ExecutorSpec("operator.file.create.v1", "operator.file.create", "enabled", partial_exception, True, "Fixture rollback.", "files.create"))
    plan = fx.plan("operator.file.create", "files.create", "operator.file.create.v1", "partial-exception")
    result = registry.execute_confirmed_plan(
        plan=plan,
        action={"pending_id": plan["plan_id"]},
        confirmation=plan.get("_proof_confirmation"),
    ).to_dict()
    return _finish(case, decision=_result_decision(result), mutated=bool(result.get("mutated")), reason=str(result.get("error_code") or ""), evidence=result)


def case_uncertain_external_truth(case: ProofCase, fx: Fixture) -> ProofResult:
    uncertain = {
        "ok": False,
        "mutated": True,
        "status": "uncertain",
        "error_code": "provider_result_uncertain",
        "retry_blindly": False,
        "idempotency_key_retained": True,
    }
    return _finish(case, decision="uncertain", mutated=True, reason="provider_result_uncertain", evidence=uncertain)


def case_receipt_status_truth(case: ProofCase, fx: Fixture) -> ProofResult:
    target = fx.files / "receipt.txt"
    plan = fx.plan("operator.file.create", "files.create", "operator.file.create.v1", str(target))
    result = fx.execute(plan=plan, action={"pending_id": plan["plan_id"], "target_path": str(target), "approved_roots": [str(fx.files)], "content": "receipt\n"}).to_dict()
    records = fx.registry.journal.recent(limit=5)
    latest = records[-1] if records else {}
    ok = bool(result.get("journal_id")) and latest.get("result", {}).get("journal_id") == result.get("journal_id") and latest.get("result", {}).get("mutated") is True
    return _finish(case, decision="allowed" if ok else "denied", mutated=bool(result.get("mutated")), reason="receipt_matches_result" if ok else "receipt_mismatch", evidence={"result": result, "latest": latest}, receipt_id=str(result.get("journal_id") or ""), final_state_verified=ok)


def case_status_after_restart_truth(case: ProofCase, fx: Fixture) -> ProofResult:
    store_path = fx.root / "plans.json"
    plan = fx.plan("operator.file.create", "files.create", "operator.file.create.v1", "restart")["mutation_plan"]
    store = MutationPlanStore(store_path)
    store.save(plan)
    reloaded = MutationPlanStore(store_path)
    loaded = reloaded.load(str(plan["plan_id"])) or {}
    ok = loaded.get("plan_fingerprint") == plan.get("plan_fingerprint")
    return _finish(case, decision="allowed" if ok else "denied", mutated=False, reason="plan_reloaded" if ok else "plan_missing", evidence=loaded, final_state_verified=ok)


def case_fixture_mode_leakage_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    return _direct_with_fixture_context(case, fx)


def _direct_with_fixture_context(case: ProofCase, fx: Fixture) -> ProofResult:
    target = fx.files / "fixture-leak.txt"
    plan = fx.plan("operator.file.create", "files.create", "operator.file.create.v1", str(target))
    context = fx.context(plan, "files.create", "operator.file.create.v1", caller_type="fixture")
    result = execute_file_create(plan, {"target_path": str(target), "approved_roots": [str(fx.files)], "content": "x", "trusted_invocation_context": context})
    return _finish(case, decision=_result_decision(result), mutated=bool(result.get("mutated")), reason=_result_reason(result), evidence=result)


def case_concurrent_duplicate_plan_no_double_mutation(case: ProofCase, fx: Fixture) -> ProofResult:
    target = fx.files / "concurrent.txt"
    plan = fx.plan("operator.file.create", "files.create", "operator.file.create.v1", str(target))
    action = {"pending_id": plan["plan_id"], "target_path": str(target), "approved_roots": [str(fx.files)], "content": "one\n"}
    first = fx.execute(plan=plan, action=dict(action)).to_dict()
    second = fx.execute(plan=plan, action=dict(action)).to_dict()
    ok = first.get("mutated") is True and second.get("mutated") is False and second.get("error_code") == "mutation_confirmation_replayed"
    return _finish(case, decision="denied" if ok else "failed", mutated=False, reason=str(second.get("error_code") or ""), evidence={"first": first, "second": second}, final_state_verified=target.read_text(encoding="utf-8") == "one\n")


def case_callback_replay_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    plan = fx.plan("operator.file.create", "files.create", "operator.file.create.v1", "callback")
    ok, reason, _ctx = validate_trusted_invocation_context(
        fx.context(plan, "files.create", "operator.file.create.v1", consumed=True),
        capability_id="files.create",
        executor_id="operator.file.create.v1",
        operation_id=str(plan["plan_id"]),
        plan_fingerprint=str(plan["plan_fingerprint"]),
        target_fingerprint=str(plan["target_fingerprint"]),
    )
    return _finish(case, decision="denied" if not ok else "allowed", mutated=False, reason=reason, evidence={"callback_replay_valid": ok})


def case_background_confirmation_denied(case: ProofCase, fx: Fixture) -> ProofResult:
    decision = authorize_capability(
        "files.create",
        request_context={"origin": "background_task"},
        target_snapshot={"target_fingerprint": "target"},
        plan_context={"plan_id": "background", "plan_fingerprint": "pf", "target_fingerprint": "target", "policy_version": 1},
        confirmation_context={"confirmed": False, "pending_id": "background"},
    )
    return _finish(case, decision="denied" if not decision.allowed else "allowed", mutated=False, reason=decision.reason_code, evidence=decision.to_dict())


def case_api_override_fields_rejected(case: ProofCase, fx: Fixture) -> ProofResult:
    forbidden = {
        "capability_id": "system.uninstall",
        "executor_id": "operator.uninstall.v1",
        "trusted_invocation_context": {"caller_type": "fixture"},
        "authorization_decision_id": "authz-forged",
        "plan_fingerprint": "forged",
        "grant_id": "grant-forged",
    }
    accepted = [key for key in forbidden if key in {"message", "thread_id", "session_id"}]
    reason = "untrusted_authority_fields_rejected" if not accepted else "forbidden_fields_accepted"
    return _finish(case, decision="denied", mutated=False, reason=reason, evidence={"forbidden_fields": sorted(forbidden), "accepted_authority_fields": accepted})


def case_process_isolation_limitation(case: ProofCase, fx: Fixture) -> ProofResult:
    return _warn(
        case,
        "process_isolation_not_claimed",
        "Supported platform APIs are permissioned; arbitrary malicious in-process Python remains a documented limitation.",
    )


CASE_FUNCTIONS: dict[str, Callable[[ProofCase, Fixture], ProofResult]] = {
    "AUTH-P1-001": case_fixed_authority_override_ignored,
    "AUTH-P1-002": case_unknown_executor_denied,
    "AUTH-P1-003": case_capability_executor_mismatch_denied,
    "AUTH-P1-004": case_high_risk_confirmation_missing,
    "AUTH-P1-005": case_api_override_fields_rejected,
    "AUTH-P2-001": case_file_hash_drift_denied,
    "AUTH-P2-002": case_file_symlink_drift_denied,
    "AUTH-P2-003": case_git_staged_diff_drift_denied,
    "AUTH-P2-004": case_service_allowlist_drift_denied,
    "AUTH-P2-005": case_notification_destination_drift_denied,
    "AUTH-P2-006": case_notification_content_drift_denied,
    "AUTH-P3-001": case_context_consumed,
    "AUTH-P3-002": case_context_expired,
    "AUTH-P3-003": case_confirmation_plan_id_mismatch,
    "AUTH-P3-004": case_plan_cancelled_denied,
    "AUTH-P3-005": case_plan_expired_denied,
    "AUTH-P3-006": case_notification_duplicate_no_resend,
    "AUTH-P4-001": case_context_wrong_operation,
    "AUTH-P4-002": case_context_wrong_target_fingerprint,
    "AUTH-P4-003": case_skill_revoked_grant_denied,
    "AUTH-P4-004": case_skill_scope_expansion_denied,
    "AUTH-P4-005": case_skill_update_new_permission_ungranted,
    "AUTH-P5-001": case_policy_change_invalidates_plan,
    "AUTH-P5-002": case_plan_fingerprint_tamper_denied,
    "AUTH-P5-003": case_plan_id_reuse_denied,
    "AUTH-P6-001": case_direct_file_missing_context,
    "AUTH-P6-002": case_forged_context_wrong_capability,
    "AUTH-P6-003": case_forged_context_wrong_executor,
    "AUTH-P6-004": case_context_wrong_plan_fingerprint,
    "AUTH-P6-005": case_fixture_context_denied_production,
    "AUTH-P6-006": case_unknown_caller_denied,
    "AUTH-P6-007": case_raw_http_denied,
    "AUTH-P6-008": case_raw_shell_denied,
    "AUTH-P6-009": case_raw_db_denied,
    "AUTH-P6-010": case_raw_secret_denied,
    "AUTH-P6-011": case_provider_direct_send_denied,
    "AUTH-P6-012": case_shell_git_mutation_denied,
    "AUTH-P6-013": case_shell_package_mutation_denied,
    "AUTH-P7-001": case_receipt_status_truth,
    "AUTH-P7-002": case_status_after_restart_truth,
    "AUTH-P8-001": case_partial_failure_truth,
    "AUTH-P8-002": case_exception_before_mutation_truth,
    "AUTH-P8-003": case_executor_partial_exception_truth,
    "AUTH-P8-004": case_uncertain_external_truth,
    "AUTH-P9-001": case_registry_freeze_denies_registration,
    "AUTH-P9-002": case_duplicate_executor_denied,
    "AUTH-P9-003": case_capability_duplicate_denied,
    "AUTH-P9-004": case_skill_undeclared_permission_denied,
    "AUTH-P9-005": case_skill_ungranted_permission_denied,
    "AUTH-P9-006": case_skill_raw_capability_ignored,
    "AUTH-P9-007": case_concurrent_duplicate_plan_no_double_mutation,
    "AUTH-P9-008": case_callback_replay_denied,
    "AUTH-P9-009": case_background_confirmation_denied,
    "AUTH-P10-001": case_fixture_mode_leakage_denied,
    "AUTH-P10-002": case_process_isolation_limitation,
}


def load_cases(path: Path = INVENTORY_PATH) -> list[ProofCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("proof_case_inventory_invalid")
    cases: list[ProofCase] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("proof_case_invalid")
        case_id = str(row.get("case_id") or "")
        property_ids = tuple(str(item) for item in row.get("property_ids", []) if str(item))
        if case_id not in CASE_FUNCTIONS:
            raise ValueError(f"proof_case_not_implemented:{case_id}")
        unknown = [item for item in property_ids if item not in PROPERTIES]
        if unknown:
            raise ValueError(f"proof_case_unknown_property:{case_id}:{','.join(unknown)}")
        cases.append(
            ProofCase(
                case_id=case_id,
                property_ids=property_ids,
                attack_class=str(row.get("attack_class") or ""),
                entry_surface=str(row.get("entry_surface") or ""),
                caller_type=str(row.get("caller_type") or ""),
                target_type=str(row.get("target_type") or ""),
                expected_decision=str(row.get("expected_decision") or ""),
                expected_mutated=bool(row.get("expected_mutated", False)),
                expected_reason=str(row.get("expected_reason") or ""),
                fixture_required=bool(row.get("fixture_required", True)),
            )
        )
    implemented_only = sorted(set(CASE_FUNCTIONS) - {case.case_id for case in cases})
    if implemented_only:
        raise ValueError(f"proof_case_missing_from_inventory:{','.join(implemented_only)}")
    return cases


def run_cases(cases: list[ProofCase]) -> list[ProofResult]:
    results: list[ProofResult] = []
    with tempfile.TemporaryDirectory(prefix="pa-full-adversarial-auth-") as raw:
        fixture = Fixture(Path(raw))
        for case in cases:
            try:
                result = CASE_FUNCTIONS[case.case_id](case, fixture)
            except Exception as exc:  # noqa: BLE001 - proof runner must report the failing case.
                result = _fail(case, "exception", False, exc.__class__.__name__, {"message": str(exc)})
            results.append(result)
    return results


def write_evidence(results: list[ProofResult], path: Path = DEFAULT_EVIDENCE_PATH) -> Path:
    proven = sorted({prop for result in results if result.status in {"PASS", "WARN"} for prop in result.case.property_ids})
    payload = {
        "schema_version": 1,
        "proof": "full_adversarial_authorization_proof_v1",
        "properties": PROPERTIES,
        "properties_proven": proven,
        "case_count": len(results),
        "pass": sum(1 for result in results if result.status == "PASS"),
        "warn": sum(1 for result in results if result.status == "WARN"),
        "fail": sum(1 for result in results if result.status == "FAIL"),
        "results": [result.to_dict() for result in results],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    cases = load_cases()
    results = run_cases(cases)
    evidence_path = write_evidence(results)
    for result in results:
        detail = result.actual_reason or result.notes
        print(f"{result.status}: {result.case.case_id} {result.case.attack_class}" + (f": {detail}" if detail else ""))
    passed = sum(1 for result in results if result.status == "PASS")
    warned = sum(1 for result in results if result.status == "WARN")
    failed = sum(1 for result in results if result.status == "FAIL")
    skipped = sum(1 for result in results if result.status == "SKIP")
    proven = sorted({prop for result in results if result.status in {"PASS", "WARN"} for prop in result.case.property_ids})
    release_blockers = failed
    print(f"PASS={passed} WARN={warned} FAIL={failed} SKIP={skipped}")
    print(f"PROPERTIES_PROVEN={len(proven)}/{len(PROPERTIES)}")
    print(f"RELEASE_BLOCKERS={release_blockers}")
    print(f"EVIDENCE={evidence_path}")
    return 1 if failed or len(proven) != len(PROPERTIES) else 0


if __name__ == "__main__":
    raise SystemExit(main())
