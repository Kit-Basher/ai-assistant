from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import ipaddress
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
from typing import Any
import urllib.parse

from agent.config import Config, code_root_path, load_config


MAX_JSON_REQUEST_BYTES = 1024 * 1024
ALLOWED_TASK_STATUSES = ("READY", "CLAIMED", "IMPLEMENTED", "VERIFIED", "BLOCKED", "DONE")
ALLOWED_TASK_OWNERS = ("manager", "codex", "kimi", "unassigned")
TASK_FENCE_RE = re.compile(r"```json\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
TASK_DOC_TITLE = "# Development Tasks"
MASTER_PLAN_BOOTSTRAP = """# Master Plan

This file is the canonical high-level plan for the local control plane.

- Update this file through `PUT /control/master_plan`.
- Keep it short, current, and decisive.
"""
TASKS_BOOTSTRAP = """# Development Tasks

This file is the canonical machine-readable task list for the local control plane.

Rules:
- The task list lives in the fenced JSON block below.
- Keep `task_id` values stable.
- Use uppercase task `status` values from the allowed set.
- Use `owner` values from the allowed set.

```json
[]
```
"""


class ControlPlaneError(ValueError):
    pass


class ControlPlaneJSONLError(ControlPlaneError):
    def __init__(self, path: Path, line_number: int, message: str) -> None:
        self.path = Path(path)
        self.line_number = int(line_number)
        super().__init__(f"{self.path} line {self.line_number}: {message}")


class ControlPlaneTaskError(ControlPlaneError):
    pass


class FileStore:
    @staticmethod
    def ensure_parent_dirs(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def read_text(path: Path) -> str:
        if not path.exists():
            return ""
        if not path.is_file():
            raise ControlPlaneError(f"{path} is not a regular file")
        return path.read_text(encoding="utf-8")

    @staticmethod
    def mtime_iso(path: Path) -> str | None:
        if not path.exists():
            return None
        if not path.is_file():
            raise ControlPlaneError(f"{path} is not a regular file")
        stat_result = path.stat()
        return datetime.fromtimestamp(stat_result.st_mtime, tz=timezone.utc).isoformat()

    @staticmethod
    def atomic_write_text(path: Path, content: str) -> int:
        FileStore.ensure_parent_dirs(path)
        text = str(content)
        payload = text.encode("utf-8")
        fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
        return len(payload)

    @staticmethod
    def append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
        FileStore.ensure_parent_dirs(path)
        line = json.dumps(record, ensure_ascii=True, sort_keys=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        if not path.is_file():
            raise ControlPlaneError(f"{path} is not a regular file")
        rows: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            text = raw_line.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ControlPlaneJSONLError(path, line_number, f"invalid JSON: {exc.msg}") from exc
            if not isinstance(parsed, dict):
                raise ControlPlaneJSONLError(path, line_number, "expected a JSON object")
            rows.append(parsed)
        return rows


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _require_string(value: Any, *, field_name: str, allow_blank: bool = False) -> str:
    if not isinstance(value, str):
        raise ControlPlaneTaskError(f"{field_name} must be a string")
    text = value.strip()
    if not text and not allow_blank:
        raise ControlPlaneTaskError(f"{field_name} is required")
    return text


def _normalize_choice(value: Any, *, allowed: tuple[str, ...], field_name: str, default: str | None = None) -> str:
    if value is None:
        if default is not None:
            return default
        raise ControlPlaneTaskError(f"{field_name} is required")
    text = _require_string(value, field_name=field_name)
    if not text:
        if default is not None:
            return default
        raise ControlPlaneTaskError(f"{field_name} is required")
    normalized = text.lower() if allowed and allowed[0].islower() else text.upper()
    if normalized not in allowed:
        raise ControlPlaneTaskError(f"{field_name} must be one of: {', '.join(allowed)}")
    return normalized


def _normalize_status(value: Any) -> str:
    return _normalize_choice(value, allowed=ALLOWED_TASK_STATUSES, field_name="status")


def _normalize_owner(value: Any) -> str:
    return _normalize_choice(value, allowed=ALLOWED_TASK_OWNERS, field_name="owner")


def _normalize_optional_string_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ControlPlaneTaskError(f"{field_name} must be a list of strings")
    rows: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _require_string(item, field_name=field_name, allow_blank=False)
        if not text or text in seen:
            continue
        seen.add(text)
        rows.append(text)
    return rows


def _normalize_priority(value: Any) -> int:
    try:
        priority = int(value)
    except (TypeError, ValueError) as exc:
        raise ControlPlaneTaskError("priority must be an integer") from exc
    if priority < 1:
        raise ControlPlaneTaskError("priority must be >= 1")
    return priority


def _normalize_task_record(raw: dict[str, Any]) -> dict[str, Any]:
    task_id = _require_string(raw.get("task_id"), field_name="task_id")
    title = _require_string(raw.get("title"), field_name=f"task {task_id}: title")
    return {
        "task_id": task_id,
        "title": title,
        "owner": _normalize_owner(raw.get("owner", "unassigned")),
        "status": _normalize_status(raw.get("status", "READY")),
        "kind": _require_string(raw.get("kind"), field_name="kind", allow_blank=True) or "task",
        "priority": _normalize_priority(raw.get("priority", 100)),
        "depends_on": _normalize_optional_string_list(raw.get("depends_on"), field_name="depends_on"),
        "summary": _require_string(raw.get("summary"), field_name="summary", allow_blank=True),
        "files_expected": _normalize_optional_string_list(raw.get("files_expected"), field_name="files_expected"),
        "acceptance_criteria": _normalize_optional_string_list(
            raw.get("acceptance_criteria"),
            field_name="acceptance_criteria",
        ),
    }


def _render_tasks_document(tasks: list[dict[str, Any]]) -> str:
    payload = json.dumps(tasks, ensure_ascii=True, sort_keys=True, indent=2)
    return f"{TASK_DOC_TITLE}\n\n" \
        "This file is the canonical machine-readable task list for the local control plane.\n\n" \
        "Rules:\n" \
        "- The task list lives in the fenced JSON block below.\n" \
        "- Keep `task_id` values stable.\n" \
        "- Use uppercase task `status` values from the allowed set.\n" \
        "- Use `owner` values from the allowed set.\n\n" \
        f"```json\n{payload}\n```\n"


def _extract_json_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped.startswith("["):
        return stripped
    match = TASK_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    raise ControlPlaneTaskError("DEVELOPMENT_TASKS.md must contain a fenced JSON task block")


def _parse_tasks_document(text: str) -> list[dict[str, Any]]:
    raw = _extract_json_text(text)
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ControlPlaneTaskError(f"invalid task JSON: {exc.msg}") from exc
    if not isinstance(parsed, list):
        raise ControlPlaneTaskError("task document must contain a JSON array")
    tasks: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    for idx, row in enumerate(parsed, start=1):
        if not isinstance(row, dict):
            raise ControlPlaneTaskError(f"task #{idx} must be a JSON object")
        task = _normalize_task_record(row)
        if task["task_id"] in seen_task_ids:
            raise ControlPlaneTaskError(f"duplicate task_id: {task['task_id']}")
        seen_task_ids.add(task["task_id"])
        tasks.append(task)
    tasks.sort(key=lambda row: (int(row.get("priority", 100)), str(row.get("task_id") or "")))
    return tasks


def _task_fields_for_sort(task: dict[str, Any]) -> tuple[int, str]:
    return int(task.get("priority", 100)), str(task.get("task_id") or "")


class ControlPlaneStore:
    master_plan_path: Path
    tasks_path: Path
    events_path: Path

    def __init__(
        self,
        master_plan_path: str | Path,
        tasks_path: str | Path,
        events_path: str | Path,
        *,
        control_dir: str | Path | None = None,
    ) -> None:
        self.master_plan_path = Path(master_plan_path).expanduser().resolve()
        self.tasks_path = Path(tasks_path).expanduser().resolve()
        self.events_path = Path(events_path).expanduser().resolve()
        self.control_dir = (
            Path(control_dir).expanduser().resolve()
            if control_dir is not None
            else self.master_plan_path.parent
        )
        self._lock_path = self.control_dir / ".control.lock"
        self._mutation_lock = threading.Lock()

    @classmethod
    def from_config(cls, config: Config) -> "ControlPlaneStore":
        control_dir = Path(config.control_dir or (code_root_path() / "control")).expanduser().resolve()
        master_plan_path = Path(
            config.control_master_plan_path or (control_dir / "master_plan.md")
        ).expanduser().resolve()
        tasks_path = Path(config.control_tasks_path or (control_dir / "DEVELOPMENT_TASKS.md")).expanduser().resolve()
        events_path = Path(config.control_events_path or (control_dir / "agent_events.jsonl")).expanduser().resolve()
        return cls(
            master_plan_path=master_plan_path,
            tasks_path=tasks_path,
            events_path=events_path,
            control_dir=control_dir,
        )

    @contextmanager
    def _exclusive_mutation(self):
        FileStore.ensure_parent_dirs(self._lock_path)
        with self._mutation_lock:
            with open(self._lock_path, "a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def bootstrap(self) -> dict[str, Any]:
        created: list[str] = []
        with self._exclusive_mutation():
            FileStore.ensure_parent_dirs(self.control_dir)
            if not self.master_plan_path.exists():
                FileStore.atomic_write_text(self.master_plan_path, MASTER_PLAN_BOOTSTRAP)
                created.append(str(self.master_plan_path))
            if not self.tasks_path.exists():
                FileStore.atomic_write_text(self.tasks_path, TASKS_BOOTSTRAP)
                created.append(str(self.tasks_path))
            if not self.events_path.exists():
                FileStore.atomic_write_text(self.events_path, "")
                created.append(str(self.events_path))
        return {
            "ok": True,
            "path": str(self.control_dir),
            "created": created,
            "existing": [str(self.master_plan_path), str(self.tasks_path), str(self.events_path)],
        }

    @staticmethod
    def _markdown_payload(path: Path) -> dict[str, Any]:
        return {
            "ok": True,
            "path": str(path),
            "content": FileStore.read_text(path),
            "mtime": FileStore.mtime_iso(path),
        }

    @staticmethod
    def _markdown_write_payload(path: Path, content: str) -> dict[str, Any]:
        bytes_written = FileStore.atomic_write_text(path, content)
        return {
            "ok": True,
            "path": str(path),
            "bytes_written": bytes_written,
            "mtime": FileStore.mtime_iso(path),
        }

    def read_master_plan(self) -> dict[str, Any]:
        return self._markdown_payload(self.master_plan_path)

    def write_master_plan(self, content: str) -> dict[str, Any]:
        if not isinstance(content, str):
            raise ControlPlaneError("content must be a string")
        with self._exclusive_mutation():
            return self._markdown_write_payload(self.master_plan_path, content)

    def read_tasks(self) -> dict[str, Any]:
        return self._markdown_payload(self.tasks_path)

    def write_tasks(self, content: str) -> dict[str, Any]:
        if not isinstance(content, str):
            raise ControlPlaneError("content must be a string")
        tasks = _parse_tasks_document(content)
        rendered = _render_tasks_document(tasks)
        with self._exclusive_mutation():
            FileStore.atomic_write_text(self.tasks_path, rendered)
        return {
            "ok": True,
            "path": str(self.tasks_path),
            "bytes_written": len(rendered.encode("utf-8")),
            "mtime": FileStore.mtime_iso(self.tasks_path),
        }

    def read_tasks_index(self) -> dict[str, Any]:
        tasks = self._load_tasks()
        return {
            "ok": True,
            "path": str(self.tasks_path),
            "mtime": FileStore.mtime_iso(self.tasks_path),
            "tasks": tasks,
        }

    def read_events(self) -> dict[str, Any]:
        return {
            "ok": True,
            "path": str(self.events_path),
            "events": FileStore.read_jsonl_records(self.events_path),
        }

    def _load_tasks(self) -> list[dict[str, Any]]:
        raw = FileStore.read_text(self.tasks_path)
        if not raw.strip():
            return []
        return _parse_tasks_document(raw)

    def _save_tasks(self, tasks: list[dict[str, Any]]) -> None:
        FileStore.atomic_write_text(self.tasks_path, _render_tasks_document(tasks))

    @staticmethod
    def _find_task(tasks: list[dict[str, Any]], task_id: str) -> dict[str, Any]:
        for task in tasks:
            if str(task.get("task_id") or "") == task_id:
                return task
        raise ControlPlaneTaskError(f"task not found: {task_id}")

    @staticmethod
    def _task_dependencies_satisfied(task: dict[str, Any], tasks: list[dict[str, Any]]) -> bool:
        depends_on = [str(item) for item in (task.get("depends_on") if isinstance(task.get("depends_on"), list) else [])]
        if not depends_on:
            return True
        by_id = {str(row.get("task_id") or ""): row for row in tasks}
        for dependency_id in depends_on:
            dependency = by_id.get(dependency_id)
            if dependency is None or str(dependency.get("status") or "") != "DONE":
                return False
        return True

    @staticmethod
    def _allowed_transitions(task: dict[str, Any], owner: str) -> tuple[str, ...]:
        status = str(task.get("status") or "")
        task_owner = str(task.get("owner") or "")
        if owner == "manager":
            return ("READY", "BLOCKED", "DONE")
        if status == "READY":
            return ("CLAIMED",)
        if status == "CLAIMED" and task_owner == owner:
            return ("IMPLEMENTED", "BLOCKED")
        if status == "IMPLEMENTED" and owner == "kimi":
            return ("VERIFIED", "BLOCKED")
        return ()

    @staticmethod
    def _task_editable_without_transition(task: dict[str, Any], owner: str) -> bool:
        status = str(task.get("status") or "")
        task_owner = str(task.get("owner") or "")
        if owner == "manager":
            return True
        if status in {"CLAIMED", "IMPLEMENTED"} and task_owner == owner:
            return True
        if status == "READY":
            return False
        return False

    @staticmethod
    def _event_payload(
        *,
        actor: str,
        type: str,
        task_id: str,
        status_before: str,
        status_after: str,
        message: str,
        extra: dict[str, Any],
        ts: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "ts": ts or _now_iso(),
            "actor": _require_string(actor, field_name="actor"),
            "type": _require_string(type, field_name="type"),
            "task_id": _require_string(task_id, field_name="task_id"),
            "status_before": _normalize_status(status_before),
            "status_after": _normalize_status(status_after),
            "message": _require_string(message, field_name="message"),
            "extra": extra,
        }
        if not isinstance(extra, dict):
            raise ControlPlaneError("extra must be an object")
        return payload

    def _validate_event_object(self, event: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(event, dict):
            raise ControlPlaneError("event body must be a JSON object")
        payload = {
            "ts": _require_string(event.get("ts"), field_name="ts") if event.get("ts") is not None else _now_iso(),
            "actor": _require_string(event.get("actor"), field_name="actor"),
            "type": _require_string(event.get("type"), field_name="type"),
            "task_id": _require_string(event.get("task_id"), field_name="task_id"),
            "status_before": _normalize_status(event.get("status_before")),
            "status_after": _normalize_status(event.get("status_after")),
            "message": _require_string(event.get("message"), field_name="message"),
            "extra": event.get("extra"),
        }
        if not isinstance(payload["extra"], dict):
            raise ControlPlaneError("extra must be an object")
        return payload

    def _append_event_locked(self, event: dict[str, Any]) -> None:
        FileStore.append_jsonl_record(self.events_path, event)

    def append_event(self, event: dict[str, Any]) -> dict[str, Any]:
        payload = self._validate_event_object(event)
        with self._exclusive_mutation():
            self._append_event_locked(payload)
        return payload

    def claim_task(self, *, task_id: str, owner: str) -> dict[str, Any]:
        actor = _normalize_owner(owner)
        if actor == "unassigned":
            raise ControlPlaneTaskError("owner must not be unassigned")
        with self._exclusive_mutation():
            tasks = self._load_tasks()
            task = self._find_task(tasks, task_id)
            before = str(task.get("status") or "")
            if before != "READY":
                raise ControlPlaneTaskError(f"task {task_id} must be READY to claim")
            task_owner = str(task.get("owner") or "unassigned")
            if task_owner not in {"unassigned", actor}:
                raise ControlPlaneTaskError(f"task {task_id} is already owned by {task_owner}")
            task["owner"] = actor
            task["status"] = "CLAIMED"
            event = self._event_payload(
                actor=actor,
                type="claim",
                task_id=task_id,
                status_before=before,
                status_after="CLAIMED",
                message=f"{actor} claimed {task_id}",
                extra={"owner": actor},
            )
            self._save_tasks(tasks)
            self._append_event_locked(event)
            return {"ok": True, "path": str(self.tasks_path), "task": task, "event": event}

    def update_task(
        self,
        *,
        task_id: str,
        owner: str,
        status: str,
        summary: str | None = None,
        files_expected: list[str] | str | None = None,
        acceptance_criteria: list[str] | str | None = None,
        depends_on: list[str] | str | None = None,
    ) -> dict[str, Any]:
        actor = _normalize_owner(owner)
        target_status = _normalize_status(status)
        with self._exclusive_mutation():
            tasks = self._load_tasks()
            task = self._find_task(tasks, task_id)
            before = str(task.get("status") or "")
            if target_status == "CLAIMED":
                raise ControlPlaneTaskError("use /control/tasks/claim to claim READY tasks")
            if target_status == before:
                if not self._task_editable_without_transition(task, actor):
                    raise ControlPlaneTaskError(
                        f"task {task_id} may only be edited by its owner or manager at the current status"
                    )
            else:
                allowed = self._allowed_transitions(task, actor)
                if target_status not in allowed:
                    raise ControlPlaneTaskError(
                        f"invalid transition {before} -> {target_status} for owner {actor}"
                    )
            if summary is not None:
                task["summary"] = _require_string(summary, field_name="summary", allow_blank=True)
            if files_expected is not None:
                task["files_expected"] = _normalize_optional_string_list(files_expected, field_name="files_expected")
            if acceptance_criteria is not None:
                task["acceptance_criteria"] = _normalize_optional_string_list(
                    acceptance_criteria,
                    field_name="acceptance_criteria",
                )
            if depends_on is not None:
                task["depends_on"] = _normalize_optional_string_list(depends_on, field_name="depends_on")
            if target_status == "READY":
                task["owner"] = "unassigned"
            task["status"] = target_status
            self._save_tasks(tasks)
            event = self._event_payload(
                actor=actor,
                type="update",
                task_id=task_id,
                status_before=before,
                status_after=target_status,
                message=f"{actor} set {task_id} to {target_status}",
                extra={
                    "updated_fields": [
                        field
                        for field, value in {
                            "summary": summary,
                            "files_expected": files_expected,
                            "acceptance_criteria": acceptance_criteria,
                            "depends_on": depends_on,
                        }.items()
                        if value is not None
                    ],
                },
            )
            self._append_event_locked(event)
            return {"ok": True, "path": str(self.tasks_path), "task": task, "event": event}

    def comment_task(self, *, task_id: str, owner: str, comment: str) -> dict[str, Any]:
        actor = _normalize_owner(owner)
        comment_text = _require_string(comment, field_name="comment")
        with self._exclusive_mutation():
            tasks = self._load_tasks()
            task = self._find_task(tasks, task_id)
            current_status = str(task.get("status") or "")
            event = self._event_payload(
                actor=actor,
                type="comment",
                task_id=task_id,
                status_before=current_status,
                status_after=current_status,
                message=comment_text,
                extra={"comment": comment_text},
            )
            self._append_event_locked(event)
            return {"ok": True, "path": str(self.events_path), "event": event}

    def next_task(self, *, owner: str) -> dict[str, Any]:
        actor = _normalize_owner(owner)
        tasks = self._load_tasks()
        eligible = [
            task
            for task in tasks
            if str(task.get("status") or "") == "READY"
            and self._task_dependencies_satisfied(task, tasks)
            and (actor == "manager" or str(task.get("owner") or "unassigned") in {"unassigned", actor})
        ]
        eligible.sort(key=_task_fields_for_sort)
        task = eligible[0] if eligible else None
        return {
            "ok": True,
            "path": str(self.tasks_path),
            "owner": actor,
            "task": task,
            "reason": "no_eligible_tasks" if task is None else "next_task_selected",
        }


class ControlPlaneHandler(BaseHTTPRequestHandler):
    store: ControlPlaneStore

    def log_message(self, format: str, *args) -> None:  # pragma: no cover - keep local service quiet in tests
        _ = format
        _ = args

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.close_connection = True
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PUT,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        try:
            self.end_headers()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
            return
        try:
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
            return

    def _read_json(self) -> dict[str, Any]:
        self._last_json_error = None
        raw_length = str(self.headers.get("Content-Length") or "").strip()
        try:
            length = int(raw_length or 0)
        except (TypeError, ValueError):
            self._last_json_error = "invalid_content_length"
            return {}
        if length < 0:
            self._last_json_error = "invalid_content_length"
            return {}
        if length > MAX_JSON_REQUEST_BYTES:
            self._last_json_error = "request_too_large"
            return {}
        if length <= 0:
            return {}
        content_type = str(self.headers.get("Content-Type") or "").strip().lower()
        if "application/json" not in content_type:
            self._last_json_error = "content_type_not_json"
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._last_json_error = "invalid_json_body"
            return {}
        if isinstance(parsed, dict):
            return parsed
        self._last_json_error = "invalid_json_body"
        return {}

    def _read_json_if_present(self) -> dict[str, Any]:
        raw_length = str(self.headers.get("Content-Length") or "").strip()
        try:
            length = int(raw_length or 0)
        except (TypeError, ValueError):
            return {}
        if length <= 0:
            return {}
        return self._read_json()

    def _json_request_error_response(self, *, path: str, json_error: str) -> tuple[int, dict[str, Any]]:
        if json_error == "content_type_not_json":
            return 400, {
                "ok": False,
                "error": "bad_request",
                "message": "Request body must use Content-Type: application/json.",
                "path": path,
            }
        if json_error == "invalid_json_body":
            return 400, {
                "ok": False,
                "error": "bad_request",
                "message": "Request body is not valid JSON.",
                "path": path,
            }
        if json_error == "request_too_large":
            return 413, {
                "ok": False,
                "error": "bad_request",
                "message": "Request body is too large.",
                "path": path,
            }
        return 400, {
            "ok": False,
            "error": "bad_request",
            "message": "Request body is not valid JSON.",
            "path": path,
        }

    def _path(self) -> str:
        parsed = urllib.parse.urlparse(self.path)
        return parsed.path or "/"

    def _request_client_host(self) -> str:
        try:
            return str((self.client_address or ("", 0))[0]).strip()
        except Exception:
            return ""

    @staticmethod
    def _host_is_loopback(host: str | None) -> bool:
        text = str(host or "").strip()
        if not text:
            return False
        try:
            return ipaddress.ip_address(text).is_loopback
        except ValueError:
            return text.lower() == "localhost"

    def _request_is_loopback(self) -> bool:
        return self._host_is_loopback(self._request_client_host())

    def _reject_non_loopback(self, *, path: str) -> bool:
        if self._request_is_loopback():
            return False
        self._send_json(
            403,
            {
                "ok": False,
                "error": "forbidden",
                "error_kind": "forbidden",
                "message": f"{path} is loopback-only.",
                "operator_only": True,
            },
        )
        return True

    def _path_and_query(self) -> tuple[str, dict[str, list[str]]]:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path or "/"
        query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        return path, query

    def do_GET(self) -> None:  # noqa: N802
        try:
            path, query = self._path_and_query()
            if path == "/control/master_plan":
                if self._reject_non_loopback(path=path):
                    return
                self._send_json(200, self.store.read_master_plan())
                return
            if path == "/control/tasks/index":
                if self._reject_non_loopback(path=path):
                    return
                self._send_json(200, self.store.read_tasks_index())
                return
            if path == "/control/tasks":
                if self._reject_non_loopback(path=path):
                    return
                self._send_json(200, self.store.read_tasks())
                return
            if path == "/control/tasks/next":
                if self._reject_non_loopback(path=path):
                    return
                owner = str((query.get("owner") or [""])[0]).strip()
                if not owner:
                    payload = self._read_json_if_present()
                    json_error = str(getattr(self, "_last_json_error", "") or "").strip().lower() or None
                    if json_error is not None:
                        status, response = self._json_request_error_response(path=path, json_error=json_error)
                        self._send_json(status, response)
                        return
                    owner = str(payload.get("owner") or "").strip()
                if not owner:
                    self._send_json(
                        400,
                        {
                            "ok": False,
                            "error": "bad_request",
                            "message": "owner is required.",
                            "path": path,
                        },
                    )
                    return
                self._send_json(200, self.store.next_task(owner=owner))
                return
            if path == "/control/events":
                if self._reject_non_loopback(path=path):
                    return
                self._send_json(200, self.store.read_events())
                return
            self._send_json(404, {"ok": False, "error": "not_found", "path": path})
        except ControlPlaneJSONLError as exc:
            self._send_json(
                500,
                {
                    "ok": False,
                    "error": "invalid_jsonl",
                    "path": str(exc.path),
                    "line": exc.line_number,
                    "message": str(exc),
                },
            )
        except ControlPlaneError as exc:
            self._send_json(500, {"ok": False, "error": "control_plane_error", "message": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._send_json(500, {"ok": False, "error": "internal_error", "message": exc.__class__.__name__})

    def do_PUT(self) -> None:  # noqa: N802
        try:
            path, _query = self._path_and_query()
            payload = self._read_json()
            json_error = str(getattr(self, "_last_json_error", "") or "").strip().lower() or None
            if json_error is not None:
                status, response = self._json_request_error_response(path=path, json_error=json_error)
                self._send_json(status, response)
                return
            if not isinstance(payload, dict):
                self._send_json(
                    400,
                    {
                        "ok": False,
                        "error": "bad_request",
                        "message": "Request body must be a JSON object.",
                        "path": path,
                    },
                )
                return
            content = payload.get("content")
            if not isinstance(content, str):
                self._send_json(
                    400,
                    {
                        "ok": False,
                        "error": "bad_request",
                        "message": "content must be a string.",
                        "path": path,
                    },
                )
                return
            if path == "/control/master_plan":
                if self._reject_non_loopback(path=path):
                    return
                self._send_json(200, self.store.write_master_plan(content))
                return
            if path == "/control/tasks":
                if self._reject_non_loopback(path=path):
                    return
                self._send_json(200, self.store.write_tasks(content))
                return
            self._send_json(404, {"ok": False, "error": "not_found", "path": path})
        except ControlPlaneError as exc:
            self._send_json(400, {"ok": False, "error": "bad_request", "message": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._send_json(500, {"ok": False, "error": "internal_error", "message": exc.__class__.__name__})

    def do_POST(self) -> None:  # noqa: N802
        try:
            path, _query = self._path_and_query()
            payload = self._read_json()
            json_error = str(getattr(self, "_last_json_error", "") or "").strip().lower() or None
            if json_error is not None:
                status, response = self._json_request_error_response(path=path, json_error=json_error)
                self._send_json(status, response)
                return
            if not isinstance(payload, dict):
                self._send_json(
                    400,
                    {
                        "ok": False,
                        "error": "bad_request",
                        "message": "Request body must be a JSON object.",
                        "path": path,
                    },
                )
                return
            if path == "/control/events":
                if self._reject_non_loopback(path=path):
                    return
                event = self.store.append_event(payload)
                self._send_json(200, {"ok": True, "path": str(self.store.events_path), "event": event})
                return
            if path == "/control/tasks/claim":
                if self._reject_non_loopback(path=path):
                    return
                task_id = str(payload.get("task_id") or "").strip()
                owner = str(payload.get("owner") or "").strip()
                if not task_id or not owner:
                    self._send_json(
                        400,
                        {
                            "ok": False,
                            "error": "bad_request",
                            "message": "task_id and owner are required.",
                            "path": path,
                        },
                    )
                    return
                self._send_json(200, self.store.claim_task(task_id=task_id, owner=owner))
                return
            if path == "/control/tasks/update":
                if self._reject_non_loopback(path=path):
                    return
                task_id = str(payload.get("task_id") or "").strip()
                owner = str(payload.get("owner") or "").strip()
                status = str(payload.get("status") or "").strip()
                if not task_id or not owner or not status:
                    self._send_json(
                        400,
                        {
                            "ok": False,
                            "error": "bad_request",
                            "message": "task_id, owner, and status are required.",
                            "path": path,
                        },
                    )
                    return
                response = self.store.update_task(
                    task_id=task_id,
                    owner=owner,
                    status=status,
                    summary=payload.get("summary") if isinstance(payload.get("summary"), str) else None,
                    files_expected=payload.get("files_expected"),
                    acceptance_criteria=payload.get("acceptance_criteria"),
                    depends_on=payload.get("depends_on"),
                )
                self._send_json(200, response)
                return
            if path == "/control/tasks/comment":
                if self._reject_non_loopback(path=path):
                    return
                task_id = str(payload.get("task_id") or "").strip()
                owner = str(payload.get("owner") or "").strip()
                comment = str(payload.get("comment") or "").strip()
                if not task_id or not owner or not comment:
                    self._send_json(
                        400,
                        {
                            "ok": False,
                            "error": "bad_request",
                            "message": "task_id, owner, and comment are required.",
                            "path": path,
                        },
                    )
                    return
                self._send_json(200, self.store.comment_task(task_id=task_id, owner=owner, comment=comment))
                return
            if path != "/control/events":
                self._send_json(404, {"ok": False, "error": "not_found", "path": path})
                return
            event = self.store.append_event(payload)
            self._send_json(200, {"ok": True, "path": str(self.store.events_path), "event": event})
        except ControlPlaneError as exc:
            self._send_json(400, {"ok": False, "error": "bad_request", "message": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._send_json(500, {"ok": False, "error": "internal_error", "message": exc.__class__.__name__})


def run_server(*, host: str = "127.0.0.1", port: int = 18888, config: Config | None = None) -> None:
    resolved_host = str(host or "").strip() or "127.0.0.1"
    if resolved_host != "127.0.0.1":
        raise SystemExit("Control plane binds only to 127.0.0.1.")
    loaded = config or load_config(require_telegram_token=False)
    store = ControlPlaneStore.from_config(loaded)

    class _Handler(ControlPlaneHandler):
        pass

    _Handler.store = store
    server = ThreadingHTTPServer((resolved_host, int(port)), _Handler)
    print(
        f"Control plane started listening=http://127.0.0.1:{int(port)} "
        f"master_plan={store.master_plan_path} tasks={store.tasks_path} events={store.events_path}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Personal Agent control plane")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.getenv("AGENT_CONTROL_PORT", "18888")))
    parser.add_argument("--init", action="store_true", help="Initialize the control directory and exit")
    args = parser.parse_args()
    loaded = load_config(require_telegram_token=False)
    store = ControlPlaneStore.from_config(loaded)
    if args.init:
        result = store.bootstrap()
        print(json.dumps(result, ensure_ascii=True, sort_keys=True), flush=True)
        return
    run_server(host=args.host, port=args.port, config=loaded)


if __name__ == "__main__":
    main()
