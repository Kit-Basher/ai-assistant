from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any


_MAX_BULLETS = 3
_MAX_LINE_LEN = 140


@dataclass(frozen=True)
class Anchor:
    thread_id: str
    id: int
    created_at: str
    title: str
    bullets: tuple[str, ...]
    open_line: str


def _collapse_whitespace(value: str) -> str:
    return " ".join((value or "").replace("\n", " ").split())


def _strip_questions(value: str) -> str:
    return (value or "").replace("?", "")


def _truncate(value: str, limit: int = _MAX_LINE_LEN) -> str:
    if len(value) <= limit:
        return value
    return value[:limit].rstrip()


def _normalize_title(title: str) -> str:
    value = _collapse_whitespace(_strip_questions(title)).strip()
    value = _truncate(value)
    if value:
        return value
    return "Checkpoint"


def _normalize_bullet(value: str) -> str:
    cleaned = _collapse_whitespace(_strip_questions(value)).strip()
    cleaned = re.sub(r"^[-*]\s*", "", cleaned).strip()
    cleaned = _truncate(cleaned)
    return cleaned


def _normalize_open_line(open_line: str) -> str:
    cleaned = _collapse_whitespace(_strip_questions(open_line)).strip()
    if not cleaned:
        return ""
    if cleaned.lower().startswith("open:"):
        content = cleaned[5:].strip()
    else:
        content = cleaned
    if not content:
        return ""
    return _truncate(f"Open: {content}")


def _split_paragraph_into_bullets(text: str) -> list[str]:
    chunks = [part.strip() for part in re.split(r"(?:[.;]\s+|\n+)", text) if part.strip()]
    return chunks


def parse_anchor_input(args: str) -> tuple[str, list[str], str] | None:
    raw = (args or "").strip()
    if not raw:
        return None
    lines = [line.rstrip() for line in raw.splitlines()]
    if not lines:
        return None
    title = lines[0].strip()
    body_lines = [line for line in lines[1:] if line.strip()]
    open_line = ""
    bullet_lines: list[str] = []
    explicit_bullets = False
    paragraph_lines: list[str] = []
    for line in body_lines:
        stripped = line.strip()
        if stripped.lower().startswith("open:") and not open_line:
            open_line = stripped
            continue
        if stripped.startswith("-") or stripped.startswith("*"):
            explicit_bullets = True
        paragraph_lines.append(stripped)
        bullet_lines.append(stripped)
    if explicit_bullets:
        bullets = bullet_lines
    elif paragraph_lines:
        bullets = _split_paragraph_into_bullets("\n".join(paragraph_lines))
    else:
        bullets = []
    return title, bullets, open_line


def create_anchor(db: Any, thread_id: str, title: str, bullets: list[str], open_line: str) -> int:
    normalized_title = _normalize_title(title)
    normalized_bullets = [_normalize_bullet(item) for item in bullets]
    normalized_bullets = [item for item in normalized_bullets if item][:_MAX_BULLETS]
    if not normalized_bullets:
        fallback = normalized_title if normalized_title else "Checkpoint saved."
        normalized_bullets = [_normalize_bullet(fallback) or "Checkpoint saved."]
    normalized_open_line = _normalize_open_line(open_line)
    bullets_json = json.dumps(normalized_bullets, ensure_ascii=True, separators=(",", ":"))
    return int(db.add_thread_anchor(thread_id, normalized_title, bullets_json, normalized_open_line))


def list_anchors(db: Any, thread_id: str, limit: int = 10) -> list[Anchor]:
    rows = db.list_thread_anchors(thread_id, limit=int(limit))
    anchors: list[Anchor] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_bullets = row.get("bullets")
        bullets_list: list[str] = []
        if isinstance(raw_bullets, str):
            try:
                decoded = json.loads(raw_bullets)
            except Exception:
                decoded = []
            if isinstance(decoded, list):
                for item in decoded:
                    if isinstance(item, str):
                        normalized = _normalize_bullet(item)
                        if normalized:
                            bullets_list.append(normalized)
        bullets = tuple(bullets_list[:_MAX_BULLETS]) if bullets_list else ("Checkpoint saved.",)
        anchors.append(
            Anchor(
                thread_id=str(row.get("thread_id") or thread_id),
                id=int(row.get("id") or 0),
                created_at=str(row.get("created_at") or ""),
                title=_normalize_title(str(row.get("title") or "")),
                bullets=bullets,
                open_line=_normalize_open_line(str(row.get("open_line") or "")),
            )
        )
    return anchors


def reset_anchors(db: Any, thread_id: str) -> None:
    db.clear_thread_anchors(thread_id)
