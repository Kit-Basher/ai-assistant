from __future__ import annotations

from typing import Any
from agent.time_utils import parse_local_datetime, to_utc_iso


def _parse_tags_and_project(text: str) -> tuple[str, list[str], str | None]:
    words = text.split()
    tags: list[str] = []
    project: str | None = None
    content_words: list[str] = []
    for word in words:
        if word.startswith("#") and len(word) > 1:
            tags.append(word[1:])
        elif word.startswith("@") and len(word) > 1 and project is None:
            project = word[1:]
        else:
            content_words.append(word)
    return " ".join(content_words).strip(), tags, project


def remember_note(context: dict[str, Any], text: str) -> dict[str, Any]:
    db = context["db"]
    semantic_memory = context.get("semantic_memory")
    cleaned, tags, project_name = _parse_tags_and_project(text)
    project_id = None
    if project_name:
        project = db.find_project_by_name(project_name)
        if project:
            project_id = project.id
    tag_str = ",".join(tags) if tags else None
    note_id = db.add_note(cleaned or text, project_id, tag_str)
    if semantic_memory is not None:
        try:
            scope = f"project:{project_name}" if project_name else "global"
            ingest = getattr(semantic_memory, "ingest_note_text", None)
            if callable(ingest):
                ingest(
                    source_ref=f"note:{note_id}",
                    text=cleaned or text,
                    scope=scope,
                    project_id=str(project_id) if project_id is not None else None,
                    pinned=True,
                    metadata={
                        "tags": tags,
                        "project": project_name,
                        "note_id": note_id,
                    },
                )
        except Exception:
            pass
    return {
        "status": "ok",
        "note_id": note_id,
        "project": project_name,
        "tags": tags,
    }


def list_projects(context: dict[str, Any]) -> dict[str, Any]:
    db = context["db"]
    projects = db.list_projects()
    return {
        "status": "ok",
        "projects": [
            {
                "id": proj.id,
                "name": proj.name,
                "pitch": proj.pitch,
                "status": proj.status,
                "priority": proj.priority,
            }
            for proj in projects
        ],
    }


def add_project(context: dict[str, Any], name: str, pitch: str | None = None) -> dict[str, Any]:
    db = context["db"]
    project_id = db.add_project(name, pitch)
    return {"status": "ok", "project_id": project_id}


def add_task(
    context: dict[str, Any],
    title: str,
    project: str | None = None,
    effort_mins: int | None = None,
    impact_1to5: int | None = None,
) -> dict[str, Any]:
    db = context["db"]
    project_id = None
    if project:
        match = db.find_project_by_name(project)
        if match:
            project_id = match.id
    task_id = db.add_task(project_id, title, effort_mins, impact_1to5)
    return {"status": "ok", "task_id": task_id}


def add_reminder(context: dict[str, Any], when_local: str, text: str) -> dict[str, Any]:
    db = context["db"]
    tz_name = context["timezone"]
    local_dt = parse_local_datetime(when_local, tz_name)
    when_utc = to_utc_iso(local_dt)
    reminder_id = db.add_reminder(when_utc, text)
    return {"status": "ok", "reminder_id": reminder_id, "when_utc": when_utc}


def set_reminder(context: dict[str, Any], when_ts: str, text: str) -> dict[str, Any]:
    return add_reminder(context, when_local=when_ts, text=text)


def next_best_task(context: dict[str, Any], minutes: int, energy: str) -> dict[str, Any]:
    return {
        "status": "ok",
        "minutes": minutes,
        "energy": energy,
        "suggestions": [],
    }


def daily_plan(context: dict[str, Any], minutes: int, energy: str) -> dict[str, Any]:
    return {
        "status": "ok",
        "minutes": minutes,
        "energy": energy,
        "plan": [],
    }


def weekly_review(context: dict[str, Any]) -> dict[str, Any]:
    return {"status": "ok", "summary": ""}
