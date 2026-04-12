from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque


@dataclass
class DebugTriggerState:
    last_trigger_at: datetime | None = None


class DebugProtocol:
    def __init__(
        self,
        threshold: int = 3,
        window_seconds: int = 300,
        cooldown_seconds: int = 600,
    ) -> None:
        self.threshold = threshold
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self._reminder_events: dict[tuple[str, str], Deque[datetime]] = {}
        self._audit_events: dict[tuple[str, str, str], Deque[datetime]] = {}
        self.state = DebugTriggerState()

    def record_reminder(
        self,
        chat_id: str,
        text: str,
        now: datetime | None = None,
    ) -> bool:
        now_dt = now or datetime.now(timezone.utc)
        key = (str(chat_id), text)
        events = self._reminder_events.setdefault(key, deque())
        self._prune(events, now_dt)
        events.append(now_dt)
        return self._should_trigger(events, now_dt)

    def record_audit_event(
        self,
        action_type: str,
        action_id: str,
        status: str,
        now: datetime | None = None,
    ) -> bool:
        now_dt = now or datetime.now(timezone.utc)
        key = (action_type, action_id, status)
        events = self._audit_events.setdefault(key, deque())
        self._prune(events, now_dt)
        events.append(now_dt)
        return self._should_trigger(events, now_dt)

    def _prune(self, events: Deque[datetime], now_dt: datetime) -> None:
        cutoff = now_dt.timestamp() - self.window_seconds
        while events and events[0].timestamp() < cutoff:
            events.popleft()

    def _should_trigger(self, events: Deque[datetime], now_dt: datetime) -> bool:
        if len(events) <= self.threshold:
            return False
        if self.state.last_trigger_at is None:
            self.state.last_trigger_at = now_dt
            return True
        elapsed = (now_dt - self.state.last_trigger_at).total_seconds()
        if elapsed >= self.cooldown_seconds:
            self.state.last_trigger_at = now_dt
            return True
        return False
