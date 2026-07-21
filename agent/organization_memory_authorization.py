from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import shlex
import sqlite3
import stat
import time
from typing import Any, Callable
import uuid

from agent.capability_policy import stable_fingerprint
from agent.assistant_ux import classify_memory_request
from agent.executor_registry import ExecutorRegistry, ExecutorSpec, redact_executor_value
from agent.mutation_boundary import assert_authorized_mutation
from agent.mutation_plan import build_mutation_plan, target_fingerprint_for_snapshot, validate_mutation_plan
from agent.secret_store import SecretStore


PRIVATE_FIELDS = {
    "text", "content", "message", "title", "note", "notes", "body", "description",
    "private_content", "document_text", "notification_content", "memory_content", "path",
}


@dataclass(frozen=True)
class OrganizationMutationSpec:
    operation: str
    capability_id: str
    executor_id: str
    action_type: str
    rollback_available: bool
    rollback_hint: str
    destructive: bool = False


@dataclass(frozen=True)
class AssistantSubOperationSpec:
    command: str
    capability_id: str
    executor_id: str
    action_type: str
    argument_schema: str
    resource_types: tuple[str, ...]
    target_tables: tuple[str, ...]
    rollback_available: bool
    rollback_hint: str
    audit_description: str
    create_operation: bool = False


def _assistant_spec(
    command: str,
    capability: str,
    schema: str,
    resources: tuple[str, ...],
    tables: tuple[str, ...],
    description: str,
    *,
    create: bool = False,
) -> AssistantSubOperationSpec:
    return AssistantSubOperationSpec(
        command=command,
        capability_id=capability,
        executor_id=f"assistant.{command}.v1",
        action_type=f"assistant.{command}",
        argument_schema=schema,
        resource_types=resources,
        target_tables=tables,
        rollback_available=False,
        rollback_hint="No automatic rollback is available; any supported inverse requires a new scoped Plan.",
        audit_description=description,
        create_operation=create,
    )


ASSISTANT_SUBOPERATIONS: dict[str, AssistantSubOperationSpec] = {
    "remember": _assistant_spec("remember", "memory.store", "nonempty_private_text", ("note",), ("notes",), "Store one explicit user note.", create=True),
    "project_new": _assistant_spec("project_new", "organization.project.manage", "project_name_optional_pitch", ("project",), ("projects",), "Create one project.", create=True),
    "task_add": _assistant_spec("task_add", "organization.task.manage", "task_title_or_project_pipe", ("task", "project"), ("tasks", "projects"), "Create one task, optionally linked to one project.", create=True),
    "remind": _assistant_spec("remind", "organization.reminder.manage", "reminder_time_pipe_text", ("reminder",), ("reminders",), "Create one reminder.", create=True),
    "done": _assistant_spec("done", "organization.task.manage", "positive_task_id", ("task",), ("tasks",), "Complete one exact task version."),
    "anchor": _assistant_spec("anchor", "organization.thread.manage", "nonempty_private_text", ("thread_anchor",), ("thread_anchors",), "Create one thread anchor.", create=True),
    "checkpoint": _assistant_spec("checkpoint", "organization.thread.manage", "nonempty_private_text", ("thread_anchor",), ("thread_anchors",), "Create one thread checkpoint.", create=True),
    "node": _assistant_spec("node", "organization.graph.manage", "at_least_two_shell_tokens", ("graph_node",), ("graph_nodes",), "Create one graph node.", create=True),
    "link": _assistant_spec("link", "organization.graph.manage", "exactly_three_shell_tokens", ("graph_edge", "graph_node"), ("graph_nodes", "graph_edges"), "Create one graph edge.", create=True),
    "relation_add": _assistant_spec("relation_add", "organization.graph.manage", "one_nonempty_argument", ("graph_relation",), ("graph_relation_types",), "Add one graph relation type.", create=True),
    "relation_remove": _assistant_spec("relation_remove", "organization.graph.manage", "one_nonempty_argument", ("graph_relation",), ("graph_relation_types", "graph_edges"), "Remove one graph relation type."),
    "relation_mode": _assistant_spec("relation_mode", "organization.graph.manage", "strict_or_open", ("graph_policy",), ("graph_relation_mode",), "Set graph relation policy for one thread."),
    "relation_constraint_add": _assistant_spec("relation_constraint_add", "organization.graph.manage", "relation_acyclic_pair", ("graph_constraint",), ("graph_relation_constraints",), "Add one acyclic graph constraint.", create=True),
    "relation_constraint_remove": _assistant_spec("relation_constraint_remove", "organization.graph.manage", "relation_acyclic_pair", ("graph_constraint",), ("graph_relation_constraints",), "Remove one acyclic graph constraint."),
    "graph_import": _assistant_spec("graph_import", "organization.graph.manage", "bounded_graph_json", ("graph",), ("graph_nodes", "graph_edges", "graph_aliases", "graph_relation_types", "graph_relation_mode", "graph_relation_constraints", "thread_focus"), "Import one bounded graph payload."),
    "graph_pack_import": _assistant_spec("graph_pack_import", "organization.graph.manage", "bounded_graph_pack_json", ("graph",), ("graph_nodes", "graph_edges", "graph_aliases", "graph_relation_types", "graph_relation_mode", "graph_relation_constraints", "thread_focus"), "Import one bounded graph pack payload."),
    "graph_clone": _assistant_spec("graph_clone", "organization.graph.manage", "thread_id_optional_merge", ("graph", "thread"), ("graph_nodes", "graph_edges", "graph_aliases", "graph_relation_types", "graph_relation_mode", "graph_relation_constraints", "thread_focus"), "Clone one graph into the active thread."),
    "node_rename": _assistant_spec("node_rename", "organization.graph.manage", "at_least_two_shell_tokens", ("graph_node",), ("graph_nodes",), "Rename one graph node."),
    "node_alias": _assistant_spec("node_alias", "organization.graph.manage", "exactly_two_shell_tokens", ("graph_node_alias",), ("graph_aliases",), "Add one graph-node alias.", create=True),
    "node_unalias": _assistant_spec("node_unalias", "organization.graph.manage", "one_nonempty_argument", ("graph_node_alias",), ("graph_aliases",), "Remove one graph-node alias."),
    "node_delete": _assistant_spec("node_delete", "organization.graph.manage", "one_nonempty_argument", ("graph_node", "graph_edge"), ("graph_nodes", "graph_edges"), "Delete one graph node and bounded incident edges."),
    "focus_node": _assistant_spec("focus_node", "organization.graph.manage", "one_nonempty_argument", ("thread_focus", "graph_node"), ("thread_focus", "graph_nodes"), "Set one thread focus node."),
    "focus_node_clear": _assistant_spec("focus_node_clear", "organization.graph.manage", "no_arguments", ("thread_focus",), ("thread_focus",), "Clear one thread focus node."),
    "graph_clear": _assistant_spec("graph_clear", "organization.graph.manage", "no_arguments", ("graph",), ("graph_nodes", "graph_edges"), "Clear the active thread graph."),
    "thread_new": _assistant_spec("thread_new", "organization.thread.manage", "nonempty_private_text", ("thread", "thread_preferences", "thread_anchor"), ("thread_labels", "thread_prefs", "thread_anchors"), "Create one thread with bounded initial metadata.", create=True),
    "thread_label": _assistant_spec("thread_label", "organization.thread.manage", "nonempty_private_text", ("thread_label",), ("thread_labels",), "Set one active-thread label."),
    "thread_unlabel": _assistant_spec("thread_unlabel", "organization.thread.manage", "no_arguments", ("thread_label",), ("thread_labels",), "Clear one active-thread label."),
    "anchors_reset": _assistant_spec("anchors_reset", "organization.thread.manage", "no_arguments", ("thread_anchor",), ("thread_anchors",), "Clear anchors for one active thread."),
    "prefs_set": _assistant_spec("prefs_set", "organization.preference.manage", "preference_toggle", ("preference",), ("preferences",), "Set one allowlisted global preference."),
    "prefs_reset": _assistant_spec("prefs_reset", "organization.preference.manage", "no_arguments", ("preference",), ("preferences",), "Reset allowlisted global preferences."),
    "prefs_thread_set": _assistant_spec("prefs_thread_set", "organization.preference.manage", "preference_toggle", ("thread_preference",), ("thread_prefs",), "Set one allowlisted thread preference."),
    "project_mode": _assistant_spec("project_mode", "organization.preference.manage", "on_or_off", ("thread_preference",), ("thread_prefs",), "Set project mode for one thread."),
    "prefs_thread_reset": _assistant_spec("prefs_thread_reset", "organization.preference.manage", "no_arguments", ("thread_preference",), ("thread_prefs",), "Reset allowlisted preferences for one thread."),
    "memory_policy_store": _assistant_spec("memory_policy_store", "memory.store", "inferred_memory_store", ("saved_memory_policy",), ("user_prefs",), "Store one explicitly classified memory policy value."),
    "memory_policy_forget": _assistant_spec("memory_policy_forget", "memory.store", "inferred_memory_forget", ("saved_memory_policy",), ("user_prefs",), "Forget one explicitly classified memory policy value."),
    "open_loop_add": _assistant_spec("open_loop_add", "organization.task.manage", "inferred_open_loop_add", ("open_loop",), ("open_loops",), "Create one open loop.", create=True),
    "open_loop_complete": _assistant_spec("open_loop_complete", "organization.task.manage", "inferred_open_loop_complete", ("open_loop",), ("open_loops",), "Complete one exact open loop."),
}


SPECS: dict[str, OrganizationMutationSpec] = {
    "memory.reset": OrganizationMutationSpec("memory.reset", "memory.forget", "memory.reset.v1", "memory.reset", False, "A reset is destructive; restore only from a separately reviewed backup.", True),
    "semantic.ingest": OrganizationMutationSpec("semantic.ingest", "memory.semantic.manage", "semantic.ingest.v1", "semantic.ingest", True, "Remove the exact ingested source through a new confirmed Plan."),
    "semantic.rebuild": OrganizationMutationSpec("semantic.rebuild", "memory.semantic.manage", "semantic.rebuild.v1", "semantic.rebuild", False, "Reconcile the semantic store before authorizing another rebuild."),
    "semantic.repair": OrganizationMutationSpec("semantic.repair", "memory.semantic.manage", "semantic.repair.v1", "semantic.repair", False, "Reconcile the semantic store before authorizing another repair."),
    "notification.test": OrganizationMutationSpec("notification.test", "notification.external.send", "notification.test.v1", "notification.test", False, "External delivery cannot be rolled back."),
    "notification.mark_read": OrganizationMutationSpec("notification.mark_read", "notification.mark_read", "notification.mark_read.v1", "notification.mark_read", True, "Restore read state through a new confirmed Plan."),
    "notification.prune": OrganizationMutationSpec("notification.prune", "notification.prune", "notification.prune.v1", "notification.prune", False, "Pruned history is not automatically restored."),
}


def _opaque_key() -> bytes:
    seed = SecretStore._machine_secret()
    return hashlib.sha256(seed + b":organization-memory-plan:v1").digest()


def opaque_content_fingerprint(namespace: str, value: Any) -> str:
    raw = str(value or "").encode("utf-8")
    return "opaque-v1:" + hmac.new(_opaque_key(), str(namespace).encode("utf-8") + b"\0" + raw, hashlib.sha256).hexdigest()


def _redacted_request(payload: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in payload.items():
        normalized = str(key).strip().lower()
        if normalized in {"operation", "mutation_plan", "confirmation", "confirm", "confirmed", "actor_id", "thread_id", "session_id"}:
            continue
        if normalized in PRIVATE_FIELDS or any(marker in normalized for marker in ("content", "message", "note", "description")):
            output[f"{normalized}_opaque_fingerprint"] = opaque_content_fingerprint(normalized, value)
            output[f"{normalized}_bytes"] = len(str(value or "").encode("utf-8"))
        elif normalized == "metadata" and isinstance(value, dict):
            output[normalized] = {str(k): "[UNTRUSTED]" for k in sorted(value)[:32]}
        else:
            output[str(key)] = value
    return output


class OrganizationMemoryAuthorizationService:
    """Canonical durable boundary for v2E user-directed mutations."""

    def __init__(self, runtime: Any, *, state_root: str | Path) -> None:
        self.runtime = runtime
        root = Path(state_root).expanduser().resolve()
        self._transaction_path = root / "confirmation_transactions.sqlite3"
        self.registry = ExecutorRegistry(
            root / "executor_registry_journal.jsonl",
            confirmation_store_path=self._transaction_path,
        )
        for spec in SPECS.values():
            self.registry.register(ExecutorSpec(
                executor_id=spec.executor_id,
                action_type=spec.action_type,
                status="enabled",
                run=self._executor(spec),
                rollback_available=spec.rollback_available,
                rollback_hint=spec.rollback_hint,
                capability_id=spec.capability_id,
            ))
        for suboperation in ASSISTANT_SUBOPERATIONS.values():
            spec = self._organization_spec(suboperation)
            self.registry.register(ExecutorSpec(
                executor_id=spec.executor_id,
                action_type=spec.action_type,
                status="enabled",
                run=self._executor(spec),
                rollback_available=spec.rollback_available,
                rollback_hint=spec.rollback_hint,
                capability_id=spec.capability_id,
            ))
        self.registry.freeze()

    @staticmethod
    def _organization_spec(suboperation: AssistantSubOperationSpec) -> OrganizationMutationSpec:
        return OrganizationMutationSpec(
            operation=f"assistant.{suboperation.command}",
            capability_id=suboperation.capability_id,
            executor_id=suboperation.executor_id,
            action_type=suboperation.action_type,
            rollback_available=suboperation.rollback_available,
            rollback_hint=suboperation.rollback_hint,
            destructive=suboperation.command in {
                "done", "relation_remove", "relation_constraint_remove", "node_unalias",
                "node_delete", "focus_node_clear", "graph_clear", "thread_unlabel",
                "anchors_reset", "prefs_reset", "prefs_thread_reset", "memory_policy_forget",
                "open_loop_complete",
            },
        )

    @staticmethod
    def _resolve_spec(operation: str, payload: dict[str, Any]) -> tuple[str, OrganizationMutationSpec | None]:
        if operation != "assistant.mutate":
            return operation, SPECS.get(operation)
        command = str(payload.get("command") or "").strip().lower()
        suboperation = ASSISTANT_SUBOPERATIONS.get(command)
        if suboperation is None:
            return f"assistant.{command or 'unknown'}", None
        canonical = f"assistant.{command}"
        return canonical, OrganizationMemoryAuthorizationService._organization_spec(suboperation)

    @staticmethod
    def _scope(payload: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(payload.get("actor_id") or "loopback_operator"),
            str(payload.get("thread_id") or "operator"),
            str(payload.get("session_id") or "local"),
        )

    def _runtime_snapshot(self) -> dict[str, Any]:
        truth = self.runtime.runtime_truth_service()
        target = dict(truth.current_chat_target_status())
        target.pop("truth_timing_ms", None)
        return {
            "mode": "safe" if bool(self.runtime._safe_mode_enabled()) else str(self.runtime.llm_control_mode_status().get("mode") or "controlled").lower(),
            "chat_target_fingerprint": stable_fingerprint(target),
            "telegram_enabled": bool(getattr(self.runtime.config, "telegram_enabled", False)),
        }

    def _semantic_roots(self) -> list[Path]:
        roots = [
            Path.cwd().resolve(),
            Path.home().joinpath("Documents").resolve(),
            Path(self.runtime.config.db_path).expanduser().resolve().parent,
        ]
        configured = getattr(self.runtime.config, "semantic_memory_allowed_roots", None)
        if isinstance(configured, (list, tuple)):
            roots.extend(Path(str(item)).expanduser().resolve() for item in configured if str(item).strip())
        return list(dict.fromkeys(roots))

    def _read_semantic_source(self, path_value: str) -> tuple[dict[str, Any], str]:
        candidate = Path(path_value).expanduser()
        absolute = Path(os.path.abspath(str(candidate)))
        selected_root: Path | None = None
        relative: Path | None = None
        for root in self._semantic_roots():
            try:
                relative = absolute.relative_to(root)
                selected_root = root
                break
            except ValueError:
                continue
        if selected_root is None or relative is None or not relative.parts:
            raise ValueError("semantic_source_outside_allowed_roots")
        if absolute.suffix.lower() not in {".txt", ".md", ".markdown", ".json", ".csv"}:
            raise ValueError("semantic_source_type_unsupported")

        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        directory_fd = os.open(selected_root, directory_flags)
        file_fd = -1
        try:
            for component in relative.parts[:-1]:
                if component in {"", ".", ".."}:
                    raise ValueError("semantic_source_outside_allowed_roots")
                next_fd = os.open(component, directory_flags | nofollow, dir_fd=directory_fd)
                os.close(directory_fd)
                directory_fd = next_fd
            final_name = relative.parts[-1]
            if final_name in {"", ".", ".."}:
                raise ValueError("semantic_source_outside_allowed_roots")
            file_fd = os.open(final_name, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow, dir_fd=directory_fd)
            info_before = os.fstat(file_fd)
            if not stat.S_ISREG(info_before.st_mode):
                raise ValueError("semantic_source_not_regular_file")
            if info_before.st_size > 8 * 1024 * 1024:
                raise ValueError("semantic_source_too_large")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(file_fd, min(1024 * 1024, 8 * 1024 * 1024 + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > 8 * 1024 * 1024:
                    raise ValueError("semantic_source_too_large")
            info_after = os.fstat(file_fd)
            if (info_before.st_dev, info_before.st_ino, info_before.st_size, info_before.st_mtime_ns) != (
                info_after.st_dev, info_after.st_ino, info_after.st_size, info_after.st_mtime_ns
            ):
                raise ValueError("semantic_source_changed_during_read")
            raw = b"".join(chunks)
            identity = {
                "allowed_root_index": self._semantic_roots().index(selected_root),
                "path_opaque_fingerprint": opaque_content_fingerprint("semantic_source_path", str(absolute)),
                "device": int(info_after.st_dev),
                "inode": int(info_after.st_ino),
                "size": int(info_after.st_size),
                "mtime_ns": int(info_after.st_mtime_ns),
                "content_sha256": hashlib.sha256(raw).hexdigest(),
                "file_type": absolute.suffix.lower(),
                "parser": "bounded_utf8_text_v1",
                "document_count": 1,
                "trust": "untrusted_document",
            }
            return identity, raw.decode("utf-8", errors="replace")
        except OSError as exc:
            if exc.errno in {getattr(os, "ELOOP", 40), getattr(os, "ENOTDIR", 20)}:
                raise ValueError("semantic_source_symlink_forbidden") from exc
            raise ValueError("semantic_source_unavailable") from exc
        finally:
            if file_fd >= 0:
                os.close(file_fd)
            os.close(directory_fd)

    def _idempotency_identity(self, command: str, payload: dict[str, Any]) -> str:
        actor, thread, session = self._scope(payload)
        material = "\0".join((
            command, actor, thread, session, str(payload.get("user_id") or actor),
            opaque_content_fingerprint(f"assistant:{command}:arguments", payload.get("private_content")),
        ))
        return opaque_content_fingerprint(f"assistant:{command}:idempotency", material)

    def _claim_create_identity(self, identity: str, plan_id: str) -> bool:
        conn = sqlite3.connect(self._transaction_path, timeout=30, isolation_level=None)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS organization_create_idempotency ("
                "identity TEXT PRIMARY KEY, plan_id TEXT NOT NULL, state TEXT NOT NULL, updated_at INTEGER NOT NULL)"
            )
            row = conn.execute(
                "SELECT state FROM organization_create_idempotency WHERE identity = ?", (identity,)
            ).fetchone()
            if row is not None and str(row[0]) in {"executing", "succeeded", "indeterminate"}:
                conn.rollback()
                return False
            conn.execute(
                "INSERT INTO organization_create_idempotency(identity, plan_id, state, updated_at) VALUES (?, ?, 'executing', ?) "
                "ON CONFLICT(identity) DO UPDATE SET plan_id=excluded.plan_id, state='executing', updated_at=excluded.updated_at",
                (identity, plan_id, int(time.time())),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def _finish_create_identity(self, identity: str, *, succeeded: bool) -> None:
        conn = sqlite3.connect(self._transaction_path, timeout=30)
        try:
            conn.execute(
                "UPDATE organization_create_idempotency SET state = ?, updated_at = ? WHERE identity = ?",
                ("succeeded" if succeeded else "failed", int(time.time()), identity),
            )
            conn.commit()
        finally:
            conn.close()

    def _target_snapshot(
        self,
        operation: str,
        payload: dict[str, Any],
        *,
        semantic_source: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = _redacted_request(payload)
        state: dict[str, Any]
        if operation.startswith("notification."):
            store = self.runtime._notification_store
            state = {
                "store_hash": store.state_hash(),
                "stored_count": int(store.status().get("stored_count") or 0),
                "target_hash": str(payload.get("hash") or ""),
            }
        elif operation.startswith("semantic."):
            status = self.runtime.semantic_memory_status()
            state = {
                "semantic_store_version": stable_fingerprint({
                    "target": status.get("target"), "index_state": status.get("index_state"), "counts": status.get("counts")
                }),
                "scope": str(payload.get("scope") or payload.get("semantic_scope") or "global"),
            }
            path_value = str(payload.get("path") or "").strip()
            if path_value:
                try:
                    identity = semantic_source or self._read_semantic_source(path_value)[0]
                    state["source"] = identity
                except ValueError as exc:
                    state["source"] = {"unavailable": True, "reason": str(exc)}
        elif operation == "memory.reset":
            _ok, status = self.runtime.memory_status()
            state = {
                "components": sorted(payload.get("components") if isinstance(payload.get("components"), list) else [str(payload.get("components") or "all")]),
                "state_fingerprint": stable_fingerprint({
                    "continuity": status.get("continuity"), "memory_v2": status.get("memory_v2"), "semantic": status.get("semantic")
                }),
            }
        else:
            command = str(payload.get("command") or "").strip().lower()
            private_text = str(payload.get("private_content") or "")
            suboperation = ASSISTANT_SUBOPERATIONS.get(command)
            organization_state: dict[str, Any] = {}
            try:
                db = self.runtime._ensure_memory_db()
                if command == "done":
                    match = re.match(r"^/done\s+(\d+)\s*$", private_text, re.IGNORECASE)
                    task_id = int(match.group(1)) if match else 0
                    task = db.get_task(task_id) if task_id else None
                    organization_state = {
                        "record_type": "task",
                        "record_id": task_id,
                        "record_status": str((task or {}).get("status") or "missing"),
                        "record_fingerprint": stable_fingerprint(task or {"missing": task_id}),
                    }
                elif command == "open_loop_complete":
                    fragment_match = re.match(r"^mark (.+) done$", private_text.strip(), re.IGNORECASE)
                    fragment = fragment_match.group(1).strip().lower() if fragment_match else ""
                    candidates = [row for row in db.list_open_loops(status="open", limit=100, order="created") if fragment and fragment in str(row.get("title") or "").lower()]
                    selected = candidates[0] if candidates else None
                    organization_state = {
                        "record_type": "open_loop",
                        "record_id": (selected or {}).get("id"),
                        "record_fingerprint": stable_fingerprint(selected or {"missing": opaque_content_fingerprint("open_loop_fragment", fragment)}),
                    }
                else:
                    if command in {"memory_policy_store", "memory_policy_forget"}:
                        decision = classify_memory_request(private_text)
                        key = str(getattr(decision, "key", "") or "")
                        keys = [key, f"{key}:{payload.get('user_id')}"] if key else []
                        placeholders = ",".join("?" for _item in keys)
                        rows = (
                            [dict(row) for row in db._conn.execute(
                                f"SELECT key, value, updated_at FROM user_prefs WHERE key IN ({placeholders}) ORDER BY key",
                                keys,
                            ).fetchall()]
                            if keys
                            else []
                        )
                        organization_state = {"preference_keys": keys, "preference_fingerprint": stable_fingerprint(rows)}
                        tables: tuple[str, ...] = tuple()
                    else:
                        tables = suboperation.target_tables if suboperation is not None else tuple()
                    table_hashes: dict[str, str] = {}
                    for table in tables:
                        exists = db._conn.execute(
                            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
                        ).fetchone()
                        if exists is None:
                            table_hashes[table] = "table_absent"
                            continue
                        rows = [dict(row) for row in db._conn.execute(f"SELECT * FROM {table} ORDER BY rowid LIMIT 10000").fetchall()]
                        table_hashes[table] = stable_fingerprint(rows)
                    if not organization_state:
                        organization_state = {"table_fingerprints": table_hashes}
            except Exception as exc:  # noqa: BLE001 - snapshot failure is fail-closed at apply.
                organization_state = {"snapshot_error": exc.__class__.__name__}
            state = {
                "command": command,
                "record_id": str(payload.get("record_id") or payload.get("target_id") or ""),
                "target_version": str(payload.get("target_version") or "create_namespace"),
                "project_scope": str(payload.get("project_id") or ""),
                "organization_state": organization_state,
                "suboperation": {
                    "command": command,
                    "argument_schema": suboperation.argument_schema if suboperation else "denied",
                    "resource_types": list(suboperation.resource_types) if suboperation else [],
                    "rollback_available": bool(suboperation.rollback_available) if suboperation else False,
                    "rollback_hint": suboperation.rollback_hint if suboperation else "Unknown operations are denied.",
                    "audit_description": suboperation.audit_description if suboperation else "Unknown operation denied.",
                },
                "create_idempotency_identity": (
                    self._idempotency_identity(command, payload)
                    if suboperation is not None and suboperation.create_operation
                    else None
                ),
                "canonical_batch_target_set": (
                    {
                        "opaque_fingerprint": opaque_content_fingerprint(
                            f"assistant:{command}:canonical_target_set", private_text
                        ),
                        "immutable": True,
                        "maximum_records": 1000,
                    }
                    if command in {"graph_import", "graph_pack_import", "graph_clone"}
                    else None
                ),
            }
        return {**self._runtime_snapshot(), "operation": operation, "request": request, "state": state}

    @staticmethod
    def _resources(operation: str, request: dict[str, Any]) -> list[str]:
        resources = [f"operation:{operation}"]
        for key in ("user_id", "thread_id", "project_id", "record_id", "target_id", "scope", "source_ref", "path", "hash"):
            value = str(request.get(key) or "").strip()
            if value:
                resources.append(f"{key}:{value}")
        return resources

    @staticmethod
    def _shell_tokens(value: str) -> list[str] | None:
        try:
            return shlex.split(value)
        except ValueError:
            return None

    def _validate_assistant_arguments(self, command: str, payload: dict[str, Any]) -> str | None:
        suboperation = ASSISTANT_SUBOPERATIONS.get(command)
        if suboperation is None:
            return "assistant_mutation_command_denied"
        allowed_fields = {
            "command", "private_content", "user_id", "actor_id", "thread_id", "session_id",
            "mutation_plan", "confirmation", "confirm", "confirmed", "operation",
        }
        unknown_fields = sorted(str(key) for key in payload if str(key) not in allowed_fields)
        if unknown_fields:
            return "assistant_mutation_extra_fields_denied"
        for key, value in payload.items():
            if key in {"mutation_plan", "confirmation"}:
                continue
            if isinstance(value, (dict, list, tuple, set)):
                return "assistant_mutation_nested_or_batch_payload_denied"
        text = str(payload.get("private_content") or "")
        if not text or len(text.encode("utf-8")) > 64 * 1024 or "\0" in text:
            return "assistant_mutation_arguments_invalid"
        trailing_lines = text[text.find("\n") + 1 :] if "\n" in text else ""
        if trailing_lines and re.search(r"(?m)^\s*/[a-z][a-z0-9_]*\b", trailing_lines):
            return "assistant_mutation_nested_operation_denied"

        inferred = {
            "memory_policy_store", "memory_policy_forget", "open_loop_add", "open_loop_complete",
        }
        if command in inferred:
            if command == "open_loop_complete" and re.fullmatch(r"mark\s+.+\s+done", text.strip(), re.IGNORECASE) is None:
                return "assistant_mutation_arguments_invalid"
            if command == "open_loop_add" and not text.strip():
                return "assistant_mutation_arguments_invalid"
            return None

        parsed = self.runtime.parse_assistant_command(text) if callable(getattr(self.runtime, "parse_assistant_command", None)) else None
        if parsed is None:
            from agent.commands import parse_command  # local import avoids a routing dependency at module import.
            parsed = parse_command(text)
        if parsed is None or str(parsed.name or "").strip().lower() != command:
            return "assistant_mutation_command_mismatch"
        args = str(parsed.args or "").strip()
        schema = suboperation.argument_schema
        tokens = self._shell_tokens(args)
        if schema == "no_arguments":
            valid = not args
        elif schema in {"nonempty_private_text", "one_nonempty_argument"}:
            valid = bool(args)
        elif schema == "positive_task_id":
            valid = bool(re.fullmatch(r"[1-9]\d*", args))
        elif schema == "reminder_time_pipe_text":
            parts = [part.strip() for part in args.split("|")]
            valid = len(parts) == 2 and all(parts)
        elif schema == "project_name_optional_pitch":
            parts = [part.strip() for part in args.split("|")]
            valid = len(parts) <= 2 and bool(parts[0])
        elif schema == "task_title_or_project_pipe":
            parts = [part.strip() for part in args.split("|")]
            valid = bool(args) and len(parts) <= 4 and (len(parts) == 1 or bool(parts[1]))
            valid = valid and (len(parts) < 3 or not parts[2] or parts[2].isdigit())
            valid = valid and (len(parts) < 4 or not parts[3] or (parts[3].isdigit() and 1 <= int(parts[3]) <= 5))
        elif schema == "at_least_two_shell_tokens":
            valid = tokens is not None and len(tokens) >= 2
        elif schema == "exactly_three_shell_tokens":
            valid = tokens is not None and len(tokens) == 3
        elif schema == "exactly_two_shell_tokens":
            valid = tokens is not None and len(tokens) == 2
        elif schema == "relation_acyclic_pair":
            valid = tokens is not None and len(tokens) == 2 and tokens[1].lower() == "acyclic"
        elif schema == "strict_or_open":
            valid = args.lower() in {"strict", "open"}
        elif schema == "on_or_off":
            valid = args.lower() in {"on", "off"}
        elif schema == "preference_toggle":
            valid = tokens is not None and len(tokens) == 2 and tokens[1].lower() in {"on", "off"}
        elif schema == "thread_id_optional_merge":
            valid = tokens is not None and len(tokens) in {1, 2} and (len(tokens) == 1 or tokens[1] == "--merge")
        elif schema in {"bounded_graph_json", "bounded_graph_pack_json"}:
            payload_text = args
            if payload_text.startswith("--merge") and (
                len(payload_text) == len("--merge") or payload_text[len("--merge")].isspace()
            ):
                payload_text = payload_text[len("--merge") :].strip()
            try:
                decoded = json.loads(payload_text)
            except (json.JSONDecodeError, ValueError):
                decoded = None
            valid = isinstance(decoded, dict)
            if valid and schema == "bounded_graph_json":
                valid = (
                    isinstance(decoded.get("nodes", []), list)
                    and isinstance(decoded.get("edges", []), list)
                    and isinstance(decoded.get("aliases", []), list)
                    and len(decoded.get("nodes", [])) <= 200
                    and len(decoded.get("edges", [])) <= 500
                    and len(decoded.get("aliases", [])) <= 300
                )
            elif valid:
                threads = decoded.get("threads", [])
                valid = isinstance(threads, list) and 0 < len(threads) <= 10
        else:
            valid = False
        return None if valid else "assistant_mutation_arguments_invalid"

    def _validate(self, operation: str, payload: dict[str, Any]) -> str | None:
        canonical_operation, spec = self._resolve_spec(operation, payload)
        if spec is None:
            return "mutation_operation_unknown"
        if len(str(payload)) > 1024 * 1024:
            return "mutation_request_too_large"
        if operation == "notification.mark_read" and not str(payload.get("hash") or "").strip():
            return "notification_hash_required"
        if operation == "memory.reset":
            valid, error_or_components = self.runtime._normalize_memory_reset_components(payload)
            if not valid:
                return str(error_or_components)
        if operation == "semantic.ingest":
            path_value = str(payload.get("path") or "").strip()
            text = str(payload.get("text") or payload.get("content") or "").strip()
            if not path_value and not text:
                return "document_text_or_path_required"
            if len(text.encode("utf-8")) > 8 * 1024 * 1024:
                return "semantic_source_too_large"
            if path_value:
                try:
                    self._read_semantic_source(path_value)
                except ValueError as exc:
                    return str(exc)
        if operation == "assistant.mutate":
            command = str(payload.get("command") or "").strip().lower()
            invalid = self._validate_assistant_arguments(command, payload)
            if invalid:
                return invalid
        return None

    def preview(self, operation: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        invalid = self._validate(operation, payload)
        if invalid:
            return False, {"ok": False, "error": invalid, "error_kind": "bad_request", "mutated": False}
        canonical_operation, spec = self._resolve_spec(operation, payload)
        if spec is None:
            return False, {"ok": False, "error": "mutation_operation_unknown", "mutated": False}
        actor, thread, session = self._scope(payload)
        snapshot = self._target_snapshot(canonical_operation, payload)
        request = _redacted_request(payload)
        plan = build_mutation_plan(
            plan_id=f"mutation-{uuid.uuid4().hex}", capability_id=spec.capability_id,
            executor_id=spec.executor_id, expires_at_epoch=int(time.time()) + 600,
            actor_id=actor, thread_id=thread, session_id=session, target_snapshot=snapshot,
            mutation_inventory=[{"operation": canonical_operation, "resources": self._resources(canonical_operation, request)}],
            preserved_resources=["unrelated user records", "private plaintext", "recovery artifacts"],
            expected_side_effects=[canonical_operation],
            recovery={"rollback_available": spec.rollback_available, "scope": spec.rollback_hint},
            activation_fingerprint=stable_fingerprint(self._runtime_snapshot()),
        )
        plan.update({"action_type": spec.action_type, "executor_status": "enabled", "target": canonical_operation})
        return True, {
            "ok": True, "requires_confirmation": True, "operation": canonical_operation,
            "plan": plan, "mutated": False,
            "privacy": "Private content is represented only by keyed opaque fingerprints.",
        }

    def apply(self, operation: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        canonical_operation, spec = self._resolve_spec(operation, payload)
        plan = payload.get("mutation_plan") if isinstance(payload.get("mutation_plan"), dict) else None
        confirmation = payload.get("confirmation") if isinstance(payload.get("confirmation"), dict) else None
        if spec is None or plan is None or confirmation is None:
            return False, {"ok": False, "error": "scoped_mutation_plan_and_confirmation_required", "mutated": False}
        invalid = self._validate(operation, payload)
        if invalid:
            return False, {"ok": False, "error": invalid, "error_kind": "bad_request", "mutated": False}
        try:
            validate_mutation_plan(plan)
        except ValueError as exc:
            return False, {"ok": False, "error": str(exc), "mutated": False}
        if str(plan.get("capability_id")) != spec.capability_id or str(plan.get("executor_id")) != spec.executor_id:
            return False, {"ok": False, "error": "mutation_operation_scope_mismatch", "mutated": False}
        semantic_source: dict[str, Any] | None = None
        authorized_document_text: str | None = None
        if canonical_operation == "semantic.ingest" and str(payload.get("path") or "").strip():
            try:
                semantic_source, authorized_document_text = self._read_semantic_source(str(payload.get("path") or ""))
            except ValueError as exc:
                return False, {"ok": False, "error": str(exc), "mutated": False}
        current_snapshot = self._target_snapshot(canonical_operation, payload, semantic_source=semantic_source)
        if target_fingerprint_for_snapshot(current_snapshot) != str(plan.get("target_fingerprint") or ""):
            return False, {"ok": False, "error": "mutation_plan_target_changed", "mutated": False}
        private_payload = {
            str(k): v for k, v in payload.items()
            if k not in {"operation", "mutation_plan", "confirmation", "confirm", "confirmed", "actor_id", "thread_id", "session_id"}
        }
        if authorized_document_text is not None:
            private_payload["text"] = authorized_document_text
            private_payload.pop("path", None)
            private_payload.setdefault("source_ref", str(payload.get("source_ref") or payload.get("path") or "authorized-local-document"))
        action = {
            "type": spec.action_type, "origin": "organization_memory_authorization",
            "pending_id": str(plan.get("plan_id") or ""), "target_snapshot": current_snapshot,
            "private_content": private_payload, "runtime_mode": "production",
            "organization_create_idempotency": (
                ((current_snapshot.get("state") or {}).get("create_idempotency_identity"))
                if isinstance(current_snapshot.get("state"), dict)
                else None
            ),
        }
        execution_plan = {
            "plan_id": str(plan.get("plan_id") or ""), "action_type": spec.action_type,
            "executor_status": "enabled", "target": canonical_operation,
            "risk_level": str(plan.get("risk_level") or ("high" if spec.destructive else "medium")),
            "capability_id": spec.capability_id, "policy_schema_version": int(plan.get("policy_version") or 1),
            "plan_fingerprint": str(plan.get("plan_fingerprint") or ""),
            "target_fingerprint": str(plan.get("target_fingerprint") or ""), "mutation_plan": plan,
        }
        result = self.registry.execute_confirmed_plan(plan=execution_plan, action=action, confirmation=confirmation, high_risk_confirmed=True)
        receipt = redact_executor_value(result.to_dict())
        details = result.details if isinstance(result.details, dict) else {}
        legacy = details.get("result") if isinstance(details.get("result"), dict) else {}
        response = {**redact_executor_value(legacy), "authorization_receipt": receipt}
        response.update({"ok": bool(result.ok), "mutated": bool(result.mutated), "capability_id": result.capability_id, "executor_id": result.executor_id, "error_code": result.error_code})
        return bool(result.ok), response

    def route(self, operation: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        if (payload.get("confirm") is True or payload.get("confirmed") is True) and not isinstance(payload.get("confirmation"), dict):
            return False, {"ok": False, "error": "boolean_confirmation_not_authorization", "mutated": False}
        if "mutation_plan" in payload or "confirmation" in payload:
            return self.apply(operation, payload)
        return self.preview(operation, payload)

    def _executor(self, spec: OrganizationMutationSpec) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
        def run(plan: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
            valid, reason, _ = assert_authorized_mutation(
                action.get("trusted_invocation_context"), expected_capability=spec.capability_id,
                expected_executor=spec.executor_id, expected_operation=str(plan.get("plan_id") or ""),
                expected_plan_fingerprint=str(plan.get("plan_fingerprint") or ""),
                expected_target_fingerprint=str(plan.get("target_fingerprint") or ""), runtime_mode="production",
            )
            if not valid:
                return {"ok": False, "mutated": False, "error_code": reason or "generic_bypass_blocked", "user_message": "Direct mutation execution was blocked."}
            params = action.get("private_content") if isinstance(action.get("private_content"), dict) else {}
            idempotency_identity = str(action.get("organization_create_idempotency") or "").strip()
            if idempotency_identity and not self._claim_create_identity(idempotency_identity, str(plan.get("plan_id") or "")):
                return {
                    "ok": True,
                    "mutated": False,
                    "executor_id": spec.executor_id,
                    "error_code": None,
                    "user_message": "This exact create request was already applied or is awaiting reconciliation.",
                    "resources_touched": [],
                    "rollback_available": False,
                    "rollback_hint": spec.rollback_hint,
                    "details": {"operation": spec.operation, "duplicate_suppressed": True},
                }
            try:
                ok, body = self._dispatch(spec.operation, params)
            except Exception:
                # An executing idempotency row deliberately remains fail-closed; an
                # operator must reconcile whether a create crossed its write boundary.
                raise
            if idempotency_identity:
                self._finish_create_identity(idempotency_identity, succeeded=bool(ok))
            safe_body = self._safe_executor_result(spec.operation, body)
            return {
                "ok": bool(ok), "mutated": bool(ok), "executor_id": spec.executor_id,
                "error_code": None if ok else str(safe_body.get("error") or "mutation_failed"),
                "user_message": str(safe_body.get("message") or ("Mutation completed." if ok else "Mutation failed.")),
                "resources_touched": self._resources(spec.operation, _redacted_request(params)),
                "rollback_available": spec.rollback_available, "rollback_hint": spec.rollback_hint,
                "details": {"operation": spec.operation, "result": redact_executor_value(safe_body)},
            }
        return run

    @staticmethod
    def _safe_executor_result(operation: str, body: dict[str, Any]) -> dict[str, Any]:
        if operation.startswith("assistant."):
            return {
                "ok": bool(body.get("ok", False)),
                "message": "Organization mutation completed." if bool(body.get("ok", False)) else "Organization mutation failed.",
                "error": str(body.get("error") or "") or None,
            }
        safe = dict(body)
        if operation == "semantic.ingest":
            safe.pop("source_ref", None)
            safe.pop("metadata", None)
            journal = safe.get("journal")
            if isinstance(journal, dict):
                safe["journal"] = redact_executor_value(journal)
        return safe

    def _dispatch(self, operation: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        runtime = self.runtime
        fixed = {
            "memory.reset": lambda: runtime.memory_reset(
                {**payload, "confirm": True},
                changed_by=str(payload.get("actor") or "authorized_operator"),
            ),
            "semantic.ingest": lambda: runtime.semantic_memory_ingest(payload),
            "semantic.rebuild": lambda: runtime.semantic_memory_rebuild(payload),
            "semantic.repair": lambda: runtime.semantic_memory_repair({**payload, "confirm": True}),
            "notification.test": lambda: runtime.llm_notifications_test({**payload, "confirm": True}),
            "notification.mark_read": lambda: runtime.llm_notifications_mark_read(payload),
            "notification.prune": lambda: runtime.llm_notifications_prune({**payload, "confirm": True}),
        }
        if operation.startswith("assistant."):
            expected_command = operation.removeprefix("assistant.")
            if str(payload.get("command") or "").strip().lower() != expected_command:
                return False, {"ok": False, "error": "assistant_mutation_command_mismatch", "mutated": False}
            return runtime.execute_authorized_assistant_mutation(payload)
        return fixed[operation]()


MUTATING_ASSISTANT_COMMANDS = frozenset(ASSISTANT_SUBOPERATIONS)


__all__ = [
    "ASSISTANT_SUBOPERATIONS", "MUTATING_ASSISTANT_COMMANDS",
    "OrganizationMemoryAuthorizationService", "OrganizationMutationSpec", "SPECS",
    "opaque_content_fingerprint",
]
