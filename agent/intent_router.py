from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo


_REMEMBER_PREFIX = re.compile(
    r"^\s*(remember|note|save this|save a note|make a note|jot this down|don't forget|dont forget)\b[:\-]?\s*",
    re.IGNORECASE,
)
_REMEMBER_ANY = re.compile(
    r"\b(remember|note|save this|save a note|make a note|jot this down|don't forget|dont forget)\b[:\-]?\s*",
    re.IGNORECASE,
)
_REMINDER_TS = re.compile(r"\b(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\b")
_TIME_ONLY = re.compile(r"^\s*(\d{1,2})(?::(\d{2}))\s*([ap]m)?\s*$", re.IGNORECASE)
_REMIND_PHRASES = (
    "remind me",
    "set a reminder",
    "set reminder",
    "reminder for",
)

_NEXT_PHRASES = (
    "what should i do next",
    "what next",
    "next task",
    "what's next",
    "whats next",
    "what should i do now",
)
_PLAN_PHRASES = (
    "plan my day",
    "plan my evening",
    "make me a plan",
    "make a plan",
    "plan my afternoon",
    "plan my morning",
)
_WEEKLY_PHRASES = (
    "weekly review",
    "how did i do this week",
    "review this week",
    "weekly check-in",
    "weekly check in",
)
_PROJECTS_PHRASES = (
    "what am i working on",
    "show my projects",
    "list my projects",
    "my projects",
)

_ENERGY_KEYWORDS = {
    "low": {"tired", "exhausted", "low"},
    "med": {"ok", "okay", "medium", "normal"},
    "high": {"focused", "energized", "energised", "high"},
}


def _now_dt(context: dict[str, Any] | None = None) -> datetime:
    if context and context.get("now_ts"):
        candidate = datetime.fromisoformat(str(context["now_ts"]))
        if candidate.tzinfo is None:
            return candidate.replace(tzinfo=timezone.utc)
        return candidate
    return datetime.now(timezone.utc)


def _now_iso(now_dt: datetime | None = None) -> str:
    base = now_dt or datetime.now(timezone.utc)
    return base.astimezone(timezone.utc).isoformat()


def _expires_iso(now_dt: datetime, minutes: int = 10) -> str:
    return (now_dt + timedelta(minutes=minutes)).astimezone(timezone.utc).isoformat()


def _normalize(text: str) -> str:
    return text.strip()


def _contains_any(haystack: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in haystack for phrase in phrases)


def _extract_remember_content(text: str) -> str:
    match = _REMEMBER_PREFIX.match(text)
    if match:
        return text[match.end() :].strip()
    match = _REMEMBER_ANY.search(text)
    if match:
        return text[match.end() :].strip()
    return text.strip()


def _parse_minutes(text: str) -> int | None:
    lower = text.lower()
    candidates: list[int] = []

    if "half hour" in lower:
        candidates.append(30)

    for match in re.finditer(r"\b(\d+)\s*(h|hr|hrs|hour|hours)\b", lower):
        candidates.append(int(match.group(1)) * 60)

    for match in re.finditer(r"\b(\d+)\s*(m|min|mins|minute|minutes)\b", lower):
        candidates.append(int(match.group(1)))

    if not candidates:
        for match in re.finditer(r"\b(\d{1,4})\b", lower):
            candidates.append(int(match.group(1)))

    if not candidates:
        return None

    unique = list(dict.fromkeys(candidates))
    if len(unique) == 1:
        return unique[0]
    return None


def _parse_energy(text: str) -> str | None:
    lower = text.lower()
    found: set[str] = set()
    for level, keywords in _ENERGY_KEYWORDS.items():
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", lower):
                found.add(level)
                break
    if len(found) == 1:
        return next(iter(found))
    return None


def _extract_reminder(text: str) -> tuple[str | None, str | None]:
    match = _REMINDER_TS.search(text)
    if not match:
        return None, None

    when_ts = match.group(1)
    after = text[match.end() :].strip()
    if after.lower().startswith("to "):
        after = after[3:].strip()

    reminder_text = after
    if not reminder_text:
        before = text[: match.start()].strip()
        lowered = before.lower()
        if lowered.startswith("remind me"):
            before = before[len("remind me") :].strip()
        if before.lower().startswith("to "):
            before = before[3:].strip()
        reminder_text = before.strip()

    reminder_text = reminder_text.strip() or None
    return when_ts, reminder_text


def _extract_reminder_text_only(text: str) -> str | None:
    lowered = text.lower()
    if "remind me" not in lowered:
        return None
    idx = lowered.find("remind me")
    remainder = text[idx + len("remind me") :].strip()
    if remainder.lower().startswith("to "):
        remainder = remainder[3:].strip()
    return remainder.strip() or None


def _skill_call(
    skill: str, function: str, args: dict[str, Any], confidence: float, explanation: str
) -> dict[str, Any]:
    return {
        "type": "skill_call",
        "skill": skill,
        "function": function,
        "args": args,
        "confidence": confidence,
        "explanation": explanation,
    }


def _clarify(
    question: str, options: list[str], confidence: float, intent_type: str | None
) -> dict[str, Any]:
    return {
        "type": "clarify",
        "question": question,
        "options": options,
        "confidence": confidence,
        "intent": intent_type,
    }


def _noop(text: str, confidence: float) -> dict[str, Any]:
    return {
        "type": "noop",
        "text": text,
        "confidence": confidence,
    }


def _store_pending(
    db: Any,
    user_id: str,
    chat_id: str,
    intent_type: str,
    partial_args: dict[str, Any],
    question: str,
    options: list[str],
    now_dt: datetime,
) -> None:
    payload_json = json.dumps(partial_args, ensure_ascii=True)
    options_json = json.dumps(options, ensure_ascii=True)
    pending_id = str(uuid.uuid4())
    now = _now_iso(now_dt)
    expires_at = _expires_iso(now_dt)
    db.replace_pending_clarification(
        pending_id,
        user_id,
        chat_id,
        intent_type,
        payload_json,
        question,
        options_json,
        expires_at,
        now,
    )


def _suggest_reminder_times(
    now_dt: datetime, context: dict[str, Any] | None
) -> list[str]:
    tz_name = (context or {}).get("timezone") or "UTC"
    tzinfo = ZoneInfo(tz_name)
    local_now = now_dt.astimezone(tzinfo)
    next_day = local_now.date() + timedelta(days=1)
    candidates = [
        datetime(next_day.year, next_day.month, next_day.day, 9, 0, tzinfo=tzinfo),
        datetime(next_day.year, next_day.month, next_day.day, 17, 0, tzinfo=tzinfo),
        datetime(next_day.year, next_day.month, next_day.day, 20, 0, tzinfo=tzinfo),
    ]
    future = [dt for dt in candidates if dt > local_now]
    return [dt.strftime("%Y-%m-%d %H:%M") for dt in future]


def _next_day_date_str(now_dt: datetime, context: dict[str, Any] | None) -> str:
    tz_name = (context or {}).get("timezone") or "UTC"
    tzinfo = ZoneInfo(tz_name)
    local_now = now_dt.astimezone(tzinfo)
    next_day = local_now.date() + timedelta(days=1)
    return next_day.strftime("%Y-%m-%d")


def _parse_time_only(text: str) -> tuple[int, int, str | None] | None:
    match = _TIME_ONLY.match(text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    ampm = match.group(3).lower() if match.group(3) else None
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return hour, minute, ampm


def _time_only_options(
    hour: int, minute: int, ampm: str | None, date_str: str
) -> list[str] | None:
    if ampm:
        if hour < 1 or hour > 12:
            return None
        if ampm == "pm" and hour != 12:
            hour = hour + 12
        if ampm == "am" and hour == 12:
            hour = 0
        return [f"{date_str} {hour:02d}:{minute:02d}"]
    if hour >= 13:
        return [f"{date_str} {hour:02d}:{minute:02d}"]
    return [
        f"{date_str} {hour:02d}:{minute:02d}",
        f"{date_str} {(hour + 12):02d}:{minute:02d}",
    ]


def _parse_local_timestamp(ts: str, context: dict[str, Any] | None) -> datetime | None:
    try:
        local = datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M")
    except ValueError:
        return None
    tz_name = (context or {}).get("timezone") or "UTC"
    return local.replace(tzinfo=ZoneInfo(tz_name))


def _is_past_timestamp(ts: str, now_dt: datetime, context: dict[str, Any] | None) -> bool:
    when_dt = _parse_local_timestamp(ts, context)
    if when_dt is None:
        return False
    now_local = now_dt.astimezone(when_dt.tzinfo or timezone.utc)
    return when_dt <= now_local


def _handle_pending(
    db: Any,
    pending: Any,
    user_id: str,
    chat_id: str,
    text: str,
    now_dt: datetime,
    context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    intent_type = pending.intent_type
    partial_args = json.loads(pending.partial_args_json)

    if intent_type in {"next_best_task", "daily_plan"}:
        minutes = partial_args.get("minutes") or _parse_minutes(text)
        energy = partial_args.get("energy") or _parse_energy(text)
        if minutes and energy:
            db.delete_pending_clarification(pending.id)
            return _skill_call(
                "core",
                intent_type,
                {"minutes": minutes, "energy": energy},
                0.85,
                "Resolved clarification for time and energy.",
            )
        question = "Quick check: how much time do you have (minutes) and energy (low/med/high)?"
        options = ["30 low", "60 med", "120 high"]
        _store_pending(
            db,
            user_id,
            chat_id,
            intent_type,
            {"minutes": minutes, "energy": energy},
            question,
            options,
            now_dt,
        )
        return _clarify(question, options, 0.60, intent_type)

    if intent_type == "set_reminder":
        when_ts = partial_args.get("when_ts")
        reminder_text = partial_args.get("text")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", text):
            partial_args["when_ts"] = text
            if _is_past_timestamp(text, now_dt, context):
                question = "That time is in the past. Please pick a future time (YYYY-MM-DD HH:MM)."
                options = _suggest_reminder_times(now_dt, context)
                _store_pending(
                    db,
                    user_id,
                    chat_id,
                    intent_type,
                    {"when_ts": None, "text": reminder_text},
                    question,
                    options,
                    now_dt,
                )
                return _clarify(question, options, 0.60, intent_type)
            if not partial_args.get("text"):
                question = "What should I remind you about?"
                options = ["Call someone", "Pay a bill", "Do something"]
                _store_pending(
                    db,
                    user_id,
                    chat_id,
                    intent_type,
                    {"when_ts": text, "text": None},
                    question,
                    options,
                    now_dt,
                )
                return _clarify(question, options, 0.60, intent_type)
            db.delete_pending_clarification(pending.id)
            return _skill_call(
                "core",
                "set_reminder",
                {"when_ts": partial_args.get("when_ts"), "text": partial_args.get("text")},
                0.85,
                "Resolved clarification with explicit timestamp.",
            )
        time_only = _parse_time_only(text)
        if not when_ts and time_only:
            date_str = _next_day_date_str(now_dt, context)
            options = _time_only_options(time_only[0], time_only[1], time_only[2], date_str)
            if options:
                question = "Please pick an explicit timestamp (YYYY-MM-DD HH:MM)."
                _store_pending(
                    db,
                    user_id,
                    chat_id,
                    intent_type,
                    {"when_ts": None, "text": reminder_text},
                    question,
                    options,
                    now_dt,
                )
                return _clarify(question, options, 0.60, intent_type)
        parsed_when, parsed_text = _extract_reminder(text)
        when_ts = when_ts or parsed_when
        reminder_text = reminder_text or parsed_text or _extract_reminder_text_only(text)
        if when_ts and reminder_text:
            if _is_past_timestamp(when_ts, now_dt, context):
                question = "That time is in the past. Please pick a future time (YYYY-MM-DD HH:MM)."
                options = _suggest_reminder_times(now_dt, context)
                _store_pending(
                    db,
                    user_id,
                    chat_id,
                    intent_type,
                    {"when_ts": None, "text": reminder_text},
                    question,
                    options,
                    now_dt,
                )
                return _clarify(question, options, 0.60, intent_type)
            db.delete_pending_clarification(pending.id)
            return _skill_call(
                "core",
                "set_reminder",
                {"when_ts": when_ts, "text": reminder_text},
                0.85,
                "Resolved clarification for reminder.",
            )
        if when_ts and not reminder_text:
            question = "What should I remind you about?"
            options = ["Call someone", "Pay a bill", "Do something"]
            _store_pending(
                db,
                user_id,
                chat_id,
                intent_type,
                {"when_ts": when_ts, "text": None},
                question,
                options,
                now_dt,
            )
            return _clarify(question, options, 0.60, intent_type)
        if not when_ts:
            question = "When should I remind you? Please use YYYY-MM-DD HH:MM."
            options = _suggest_reminder_times(now_dt, context)
            _store_pending(
                db,
                user_id,
                chat_id,
                intent_type,
                {"when_ts": when_ts, "text": reminder_text},
                question,
                options,
                now_dt,
            )
            return _clarify(question, options, 0.60, intent_type)
        question = "What should I remind you about?"
        options = ["Call someone", "Pay a bill", "Do something"]
        _store_pending(
            db,
            user_id,
            chat_id,
            intent_type,
            {"when_ts": when_ts, "text": reminder_text},
            question,
            options,
            now_dt,
        )
        return _clarify(question, options, 0.60, intent_type)

    if intent_type == "remember_note":
        content = text.strip()
        if content:
            db.delete_pending_clarification(pending.id)
            return _skill_call(
                "core",
                "remember_note",
                {"text": content},
                0.85,
                "Resolved clarification for note content.",
            )
        question = "What should I remember?"
        options = ["Buy milk", "Call someone", "Pay a bill"]
        _store_pending(
            db,
            user_id,
            chat_id,
            intent_type,
            {"text": None},
            question,
            options,
            now_dt,
        )
        return _clarify(question, options, 0.60, intent_type)

    return None


def route_message(user_id: str, text: str, context: dict | None) -> dict[str, Any]:
    cleaned = _normalize(text or "")
    if not cleaned:
        return _noop("Try /help", 0.20)

    if cleaned.startswith("/"):
        return _noop("Try /help", 0.20)

    now_dt = _now_dt(context)
    db = (context or {}).get("db")
    chat_id = str((context or {}).get("chat_id") or user_id)
    if db:
        pending = db.get_pending_clarification(user_id, chat_id, _now_iso(now_dt))
        if pending:
            pending_decision = _handle_pending(
                db, pending, user_id, chat_id, cleaned, now_dt, context
            )
            if pending_decision:
                return pending_decision

    lowered = cleaned.lower()
    if _contains_any(lowered, _REMIND_PHRASES):
        when_ts, reminder_text = _extract_reminder(cleaned)
        if not reminder_text:
            reminder_text = _extract_reminder_text_only(cleaned)
        if not when_ts:
            question = "When should I remind you? Please use YYYY-MM-DD HH:MM."
            options = _suggest_reminder_times(now_dt, context)
            if db:
                _store_pending(
                    db,
                    user_id,
                    chat_id,
                    "set_reminder",
                    {"when_ts": None, "text": reminder_text},
                    question,
                    options,
                    now_dt,
                )
            return _clarify(question, options, 0.60, "set_reminder")
        if not reminder_text:
            question = "What should I remind you about?"
            options = ["Call someone", "Pay a bill", "Do something"]
            if db:
                _store_pending(
                    db,
                    user_id,
                    chat_id,
                    "set_reminder",
                    {"when_ts": when_ts, "text": None},
                    question,
                    options,
                    now_dt,
                )
            return _clarify(question, options, 0.60, "set_reminder")
        if _is_past_timestamp(when_ts, now_dt, context):
            question = "That time is in the past. Please pick a future time (YYYY-MM-DD HH:MM)."
            options = _suggest_reminder_times(now_dt, context)
            if db:
                _store_pending(
                    db,
                    user_id,
                    chat_id,
                    "set_reminder",
                    {"when_ts": None, "text": reminder_text},
                    question,
                    options,
                    now_dt,
                )
            return _clarify(question, options, 0.60, "set_reminder")
        return _skill_call(
            "core",
            "set_reminder",
            {"when_ts": when_ts, "text": reminder_text},
            0.85,
            "Matched reminder intent with explicit timestamp.",
        )

    if "next week" not in lowered and _contains_any(lowered, _NEXT_PHRASES):
        minutes = _parse_minutes(cleaned)
        energy = _parse_energy(cleaned)
        if minutes and energy:
            return _skill_call(
                "core",
                "next_best_task",
                {"minutes": minutes, "energy": energy},
                0.85,
                "Matched next-best-task intent.",
            )
        question = "Quick check: how much time do you have (minutes) and energy (low/med/high)?"
        options = ["30 low", "60 med", "120 high"]
        if db:
            _store_pending(
                db,
                user_id,
                chat_id,
                "next_best_task",
                {"minutes": minutes, "energy": energy},
                question,
                options,
                now_dt,
            )
        return _clarify(question, options, 0.60, "next_best_task")

    if _contains_any(lowered, _PLAN_PHRASES):
        minutes = _parse_minutes(cleaned)
        energy = _parse_energy(cleaned)
        if minutes and energy:
            return _skill_call(
                "core",
                "daily_plan",
                {"minutes": minutes, "energy": energy},
                0.85,
                "Matched daily-plan intent.",
            )
        question = "Quick check: how much time do you have (minutes) and energy (low/med/high)?"
        options = ["30 low", "60 med", "120 high"]
        if db:
            _store_pending(
                db,
                user_id,
                chat_id,
                "daily_plan",
                {"minutes": minutes, "energy": energy},
                question,
                options,
                now_dt,
            )
        return _clarify(question, options, 0.60, "daily_plan")

    if _contains_any(lowered, _WEEKLY_PHRASES):
        return _skill_call(
            "core",
            "weekly_review",
            {},
            0.85,
            "Matched weekly-review intent.",
        )

    if re.search(r"\bprojects?\b", lowered) or _contains_any(lowered, _PROJECTS_PHRASES):
        return _skill_call(
            "core",
            "list_projects",
            {},
            0.85,
            "Matched list-projects intent.",
        )

    if _REMEMBER_ANY.search(cleaned):
        content = _extract_remember_content(cleaned)
        if content:
            return _skill_call(
                "core",
                "remember_note",
                {"text": content},
                0.85,
                "Matched remember intent.",
            )
        question = "What should I remember?"
        options = ["Buy milk", "Call someone", "Pay a bill"]
        if db:
            _store_pending(
                db,
                user_id,
                chat_id,
                "remember_note",
                {"text": None},
                question,
                options,
                now_dt,
            )
        return _clarify(question, options, 0.60, "remember_note")

    return _noop("Try /help", 0.30)
