from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from agent.ask_timeframe import parse_timeframe

_REMEMBER_PREFIX = re.compile(
    r"^\s*(remember|note|save this|save a note|make a note|jot this down|don't forget|dont forget)\b[:\-]?\s*",
    re.IGNORECASE,
)
_REMEMBER_ANY = re.compile(
    r"\b(remember|note|save this|save a note|make a note|jot this down|don't forget|dont forget)\b[:\-]?\s*",
    re.IGNORECASE,
)
_REMINDER_TS = re.compile(r"\b(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\b")
_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_ONLY = re.compile(r"^\s*(\d{1,2})(?::(\d{2}))?\s*([ap]m)?\s*$", re.IGNORECASE)
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

_DISK_REGEX = re.compile(
    r"\b(disk|ssd|storage|space|cleanup|largest folders|storage report|disk usage|ssd full|safe to clean)\b",
    re.IGNORECASE,
)
_DISK_CHANGES = re.compile(r"\b(disk changes|what changed on my disk)\b", re.IGNORECASE)
_DISK_BASELINE = re.compile(r"\bset disk baseline\b", re.IGNORECASE)
_DISK_GROW = re.compile(r"\b(what grew under|show growth under)\b", re.IGNORECASE)
_CLEAN_APT = re.compile(r"^\s*(clean|clear)\s+apt\s+cache\s*$", re.IGNORECASE)
_CLEAN_HOME_CACHE = re.compile(r"^\s*(clear|clean)\s+cache\s*$", re.IGNORECASE)
_CLEAN_JOURNALD = re.compile(r"^\s*(clean\s+logs|vacuum\s+journal)\s*$", re.IGNORECASE)
_CLEAN_GENERIC = re.compile(r"\b(clean|clear|vacuum)\b", re.IGNORECASE)
_ASK_PHRASES = (
    "ask ",
    "question ",
    "what happened",
    "what changed",
    "summarize",
)
_ADVICE_PHRASES = (
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

_OPINION_PHRASES = (
    "what do you think",
    "is this unusual",
    "does this look stable",
    "does this seem stable",
    "is this normal for me",
    "outside my baseline",
)

_OPINION_FOLLOWUP_PHRASES = (
    "opinion",
    "give me your opinion",
    "what should i watch out for",
    "any concerns",
)

_KNOWLEDGE_PHRASES = (
    "what changed",
    "what happened",
    "any anomalies",
    "disk usage",
    "storage growth",
    "over the last",
    "this week",
    "last week",
    "last 30 days",
)

_KNOWLEDGE_ACTION_VERBS = (
    "delete",
    "clean",
    "clear",
    "move",
    "fix",
    "run",
    "stop",
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


def _contains_advice(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _ADVICE_PHRASES)


def _contains_opinion_trigger(text: str) -> str | None:
    lowered = text.lower()
    for phrase in _OPINION_PHRASES:
        if phrase in lowered:
            return phrase
    return None


def _has_time_window_hint(text: str) -> bool:
    lowered = text.lower()
    if any(phrase in lowered for phrase in ("today", "yesterday", "this week", "last week")):
        return True
    if re.search(r"last\s+\d{1,2}\s+days", lowered):
        return True
    if "over the last" in lowered:
        return True
    if re.search(r"\d{4}-\d{2}-\d{2}\s*(?:to|-|–)\s*\d{4}-\d{2}-\d{2}", lowered):
        return True
    return False


def _matches_knowledge_query(text: str) -> bool:
    lowered = text.lower()
    if not _contains_any(lowered, _KNOWLEDGE_PHRASES):
        return False
    if any(verb in lowered for verb in _KNOWLEDGE_ACTION_VERBS):
        return False
    if _contains_advice(lowered):
        return False
    if _contains_any(lowered, _ASK_PHRASES) and not _has_time_window_hint(lowered):
        return False
    return True


def _matches_opinion_followup(text: str) -> bool:
    lowered = text.lower().strip()
    if lowered == "opinion" or lowered == "yes":
        return True
    return _contains_any(lowered, _OPINION_FOLLOWUP_PHRASES)


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
    skill: str,
    function: str,
    args: dict[str, Any],
    confidence: float,
    explanation: str,
    scopes: list[str] | None = None,
) -> dict[str, Any]:
    payload = {
        "type": "skill_call",
        "skill": skill,
        "function": function,
        "args": args,
        "confidence": confidence,
        "explanation": explanation,
    }
    if scopes:
        payload["scopes"] = scopes
    return payload


def _action_proposal(action_type: str, action_id: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "action_proposal",
        "action_type": action_type,
        "action_id": action_id,
        "details": details,
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


def _resolve_time_only(parsed: tuple[int, int, str | None]) -> tuple[int, int] | None:
    hour, minute, ampm = parsed
    if ampm:
        if hour < 1 or hour > 12:
            return None
        if ampm == "pm" and hour != 12:
            hour = hour + 12
        if ampm == "am" and hour == 12:
            hour = 0
        return hour, minute
    if hour >= 0 and hour <= 23:
        return hour, minute
    return None


def resolve_datetime(
    text: str, now_dt: datetime, context: dict[str, Any] | None
) -> datetime | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    tz_name = (context or {}).get("timezone") or "UTC"
    tzinfo = ZoneInfo(tz_name)
    lowered = cleaned.lower()

    if _REMINDER_TS.fullmatch(cleaned):
        try:
            parsed = datetime.strptime(cleaned, "%Y-%m-%d %H:%M")
        except ValueError:
            return None
        return parsed.replace(tzinfo=tzinfo)

    if _DATE_ONLY.fullmatch(cleaned):
        try:
            parsed = datetime.strptime(cleaned, "%Y-%m-%d")
        except ValueError:
            return None
        return parsed.replace(tzinfo=tzinfo, hour=9, minute=0)

    match = re.match(r"^(today|tomorrow)(?:\s+(.+))?$", lowered)
    if match:
        base = now_dt.astimezone(tzinfo).date()
        if match.group(1) == "tomorrow":
            base = base + timedelta(days=1)
        time_part = match.group(2)
        if time_part:
            parsed_time = _parse_time_only(time_part)
            if not parsed_time:
                return None
            resolved = _resolve_time_only(parsed_time)
            if not resolved:
                return None
            hour, minute = resolved
        else:
            hour, minute = 9, 0
        return datetime(base.year, base.month, base.day, hour, minute, tzinfo=tzinfo)

    return None


def _format_local_timestamp(when_dt: datetime, context: dict[str, Any] | None) -> str:
    tz_name = (context or {}).get("timezone") or "UTC"
    tzinfo = ZoneInfo(tz_name)
    return when_dt.astimezone(tzinfo).strftime("%Y-%m-%d %H:%M")


def _is_past_datetime(when_dt: datetime, now_dt: datetime, context: dict[str, Any] | None) -> bool:
    tz_name = (context or {}).get("timezone") or "UTC"
    tzinfo = ZoneInfo(tz_name)
    now_local = now_dt.astimezone(tzinfo)
    return when_dt <= now_local


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
        options = json.loads(pending.options_json) if pending.options_json else []
        candidate = text.strip()
        if candidate.isdigit() and options:
            idx = int(candidate)
            if 1 <= idx <= len(options):
                candidate = options[idx - 1]

        if not when_ts:
            time_only = _parse_time_only(candidate)
            if time_only:
                date_str = _next_day_date_str(now_dt, context)
                time_options = _time_only_options(time_only[0], time_only[1], time_only[2], date_str)
                if time_options:
                    question = "Please pick an explicit timestamp (YYYY-MM-DD HH:MM)."
                    _store_pending(
                        db,
                        user_id,
                        chat_id,
                        intent_type,
                        {"when_ts": None, "text": reminder_text},
                        question,
                        time_options,
                        now_dt,
                    )
                    return _clarify(question, time_options, 0.60, intent_type)

        resolved_dt = resolve_datetime(candidate, now_dt, context)
        if resolved_dt is None and not when_ts:
            if partial_args.get("retry"):
                db.delete_pending_clarification(pending.id)
                return _noop(
                    "Sorry, I still couldn't parse that. Try 'tomorrow 2pm' or '2026-02-05 14:00'.",
                    0.20,
                )
            question = "I couldn't parse that. Try 'tomorrow 2pm' or '2026-02-05 14:00'."
            options = _suggest_reminder_times(now_dt, context)
            _store_pending(
                db,
                user_id,
                chat_id,
                intent_type,
                {"when_ts": None, "text": reminder_text, "retry": True},
                question,
                options,
                now_dt,
            )
            return _clarify(question, options, 0.60, intent_type)

        if resolved_dt and _is_past_datetime(resolved_dt, now_dt, context):
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

        if resolved_dt and not reminder_text:
            resolved_ts = _format_local_timestamp(resolved_dt, context)
            question = "What should I remind you about?"
            options = ["Call someone", "Pay a bill", "Do something"]
            _store_pending(
                db,
                user_id,
                chat_id,
                intent_type,
                {"when_ts": resolved_ts, "text": None},
                question,
                options,
                now_dt,
            )
            return _clarify(question, options, 0.60, intent_type)

        if resolved_dt and reminder_text:
            resolved_ts = _format_local_timestamp(resolved_dt, context)
            db.delete_pending_clarification(pending.id)
            return _skill_call(
                "core",
                "set_reminder",
                {"when_ts": resolved_ts, "text": reminder_text},
                0.85,
                "Resolved clarification for reminder.",
            )

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

    if intent_type == "ask_query":
        candidate = text.strip()
        parsed = parse_timeframe(candidate, db, (context or {}).get("timezone") or "UTC")
        db.delete_pending_clarification(pending.id)
        if not parsed.ok or parsed.clarify:
            return _noop(
                "Please re-run /ask with a specific timeframe (last 7 days, last 72 hours, or last week).",
                0.20,
            )
        timeframe = {
            "label": parsed.label,
            "start_date": parsed.start_date,
            "end_date": parsed.end_date,
            "start_ts": parsed.start_ts,
            "end_ts": parsed.end_ts,
            "user_id": user_id,
            "clarification_required": True,
        }
        return _skill_call(
            "recall",
            "ask_query",
            {"question": partial_args.get("question") or "", "timeframe": timeframe},
            0.85,
            "Resolved clarification for ask_query.",
            scopes=["db:read"],
        )

    if intent_type == "ask_opinion":
        candidate = text.strip()
        parsed = parse_timeframe(candidate, db, (context or {}).get("timezone") or "UTC")
        db.delete_pending_clarification(pending.id)
        if not parsed.ok or parsed.clarify:
            return _noop(
                "Please re-run with a specific timeframe (last 7 days, last 72 hours, or last week).",
                0.20,
            )
        timeframe = {
            "label": parsed.label,
            "start_date": parsed.start_date,
            "end_date": parsed.end_date,
            "start_ts": parsed.start_ts,
            "end_ts": parsed.end_ts,
            "user_id": user_id,
            "clarification_required": True,
        }
        return _skill_call(
            "opinion",
            "ask_opinion",
            {
                "question": partial_args.get("question") or "",
                "timeframe": timeframe,
                "trigger": partial_args.get("trigger") or "",
            },
            0.85,
            "Resolved clarification for ask_opinion.",
            scopes=["db:read"],
        )

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
    knowledge_cache = (context or {}).get("knowledge_cache")
    if db:
        pending = db.get_pending_clarification(user_id, chat_id, _now_iso(now_dt))
        if pending:
            pending_decision = _handle_pending(
                db, pending, user_id, chat_id, cleaned, now_dt, context
            )
            if pending_decision:
                return pending_decision

    if _matches_opinion_followup(cleaned):
        if knowledge_cache:
            entry = knowledge_cache.get_recent(user_id, now_dt)
            if entry:
                return _skill_call(
                    "opinion_on_report",
                    "opinion_on_report",
                    {
                        "facts": entry.facts,
                        "context_note": f"from knowledge_query: {entry.query}",
                    },
                    0.85,
                    "Matched opinion follow-up for knowledge query.",
                    scopes=[],
                )
        return _noop(
            "I can, but I need a report first — ask a knowledge question like “what changed this week?”",
            0.60,
        )

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

    if _contains_any(lowered, _ASK_PHRASES) or cleaned.endswith("?"):
        if _contains_advice(cleaned):
            return _noop(
                "I can only provide factual recall from existing snapshots. Please ask for observations, not advice or actions.",
                0.20,
            )
        trigger = _contains_opinion_trigger(cleaned)
        if trigger:
            if not db:
                return _noop("Database unavailable for recall.", 0.20)
            parsed = parse_timeframe(cleaned, db, (context or {}).get("timezone") or "UTC")
            if parsed.clarify:
                question = "What timeframe should I use? (last 7 days, last 72 hours, or last week)"
                options = ["last 7 days", "last 72 hours", "last week"]
                _store_pending(
                    db,
                    user_id,
                    chat_id,
                    "ask_opinion",
                    {"question": cleaned, "trigger": trigger},
                    question,
                    options,
                    now_dt,
                )
                return _clarify(question, options, 0.60, "ask_opinion")
            if not parsed.ok:
                return _noop("No snapshots found yet.", 0.20)
            timeframe = {
                "label": parsed.label,
                "start_date": parsed.start_date,
                "end_date": parsed.end_date,
                "start_ts": parsed.start_ts,
                "end_ts": parsed.end_ts,
                "user_id": user_id,
                "clarification_required": False,
            }
            return _skill_call(
                "opinion",
                "ask_opinion",
                {"question": cleaned, "timeframe": timeframe, "trigger": trigger},
                0.85,
                "Matched ask_opinion intent.",
                scopes=["db:read"],
            )
        if _matches_knowledge_query(cleaned):
            return _skill_call(
                "knowledge_query",
                "knowledge_query",
                {"query": cleaned},
                0.85,
                "Matched knowledge query intent.",
                scopes=["db:read"],
            )
        if not db:
            return _noop("Database unavailable for recall.", 0.20)
        parsed = parse_timeframe(cleaned, db, (context or {}).get("timezone") or "UTC")
        if parsed.clarify:
            question = "What timeframe should I use? (last 7 days, last 72 hours, or last week)"
            options = ["last 7 days", "last 72 hours", "last week"]
            _store_pending(
                db,
                user_id,
                chat_id,
                "ask_query",
                {"question": cleaned},
                question,
                options,
                now_dt,
            )
            return _clarify(question, options, 0.60, "ask_query")
        if not parsed.ok:
            return _noop("No snapshots found yet.", 0.20)
        timeframe = {
            "label": parsed.label,
            "start_date": parsed.start_date,
            "end_date": parsed.end_date,
            "start_ts": parsed.start_ts,
            "end_ts": parsed.end_ts,
            "user_id": user_id,
            "clarification_required": False,
        }
        return _skill_call(
            "recall",
            "ask_query",
            {"question": cleaned, "timeframe": timeframe},
            0.85,
            "Matched ask_query intent.",
            scopes=["db:read"],
        )

    if _matches_knowledge_query(cleaned):
        if _DISK_REGEX.search(lowered) or _DISK_CHANGES.search(lowered) or _DISK_GROW.search(lowered):
            question = (
                "Do you want a summary of recent changes or a specific disk report?\n"
                "Examples: what changed this week; latest disk report; any anomalies lately?"
            )
            options = ["what changed this week", "latest disk report", "any anomalies lately?"]
            return _clarify(question, options, 0.60, "knowledge_query")
        return _skill_call(
            "knowledge_query",
            "knowledge_query",
            {"query": cleaned},
            0.85,
            "Matched knowledge query intent.",
            scopes=["db:read"],
        )

    if _contains_any(lowered, _WEEKLY_PHRASES):
        return _skill_call(
            "core",
            "weekly_review",
            {},
            0.85,
            "Matched weekly-review intent.",
        )

    if _DISK_CHANGES.search(lowered):
        return {"type": "disk_changes"}

    if _DISK_BASELINE.search(lowered):
        return {"type": "disk_baseline"}

    if _DISK_GROW.search(lowered):
        match = re.search(r"under\\s+(.+)$", cleaned, re.IGNORECASE)
        if match:
            return {"type": "disk_grow", "path": match.group(1).strip()}

    if _DISK_REGEX.search(lowered) or "why is my ssd full" in lowered:
        return _skill_call(
            "disk_report",
            "disk_report",
            {},
            0.75,
            "Matched disk report intent.",
        )

    if _CLEAN_APT.search(cleaned):
        return _action_proposal(
            "disk_cleanup",
            "apt_cache",
            {"path": "/var/cache/apt/archives"},
        )

    if _CLEAN_HOME_CACHE.search(cleaned):
        return _action_proposal(
            "disk_cleanup",
            "home_cache",
            {"path": "~/.cache"},
        )

    if _CLEAN_JOURNALD.search(cleaned):
        return _action_proposal(
            "disk_cleanup",
            "journald_vacuum",
            {"path": "/var/log/journal"},
        )

    if _CLEAN_GENERIC.search(cleaned):
        return {
            "type": "respond",
            "text": "I can only propose: 'clean apt cache', 'clean logs', 'vacuum journal', or 'clear cache'. Try /disk_report for advice.",
        }

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
