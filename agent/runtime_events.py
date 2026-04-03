from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
import threading
from typing import Any


_LOGGER = logging.getLogger("agent.runtime_events")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class RuntimeEventHistory:
    def __init__(self, *, max_events: int = 100) -> None:
        self._events: deque[dict[str, Any]] = deque(maxlen=max(1, int(max_events)))
        self._lock = threading.Lock()

    def append(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(deepcopy(event))

    def snapshot(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(self._events)
        if limit is not None:
            rows = rows[-max(1, int(limit)) :]
        return [deepcopy(row) for row in rows]


def log_runtime_event(
    event_name: str,
    *,
    runtime_id: str | None = None,
    history: RuntimeEventHistory | None = None,
    logger: logging.Logger | None = None,
    **fields: Any,
) -> dict[str, Any]:
    payload = {
        "event": str(event_name or "").strip() or "runtime_event",
        "timestamp": _timestamp(),
        "runtime_id": str(runtime_id or "").strip() or None,
        **fields,
    }
    active_logger = logger or _LOGGER
    try:
        active_logger.info(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    except Exception:
        pass
    if history is not None:
        history.append(payload)
    return payload


def log_runtime_phase_change(
    old_phase: str | None,
    new_phase: str | None,
    *,
    runtime_id: str | None = None,
    history: RuntimeEventHistory | None = None,
    logger: logging.Logger | None = None,
    **fields: Any,
) -> dict[str, Any]:
    return log_runtime_event(
        "runtime_phase_change",
        runtime_id=runtime_id,
        history=history,
        logger=logger,
        phase_from=str(old_phase or "").strip().lower() or None,
        phase_to=str(new_phase or "").strip().lower() or None,
        **fields,
    )


def log_provider_switch(
    old_provider: str | None,
    new_provider: str | None,
    *,
    runtime_id: str | None = None,
    history: RuntimeEventHistory | None = None,
    logger: logging.Logger | None = None,
    **fields: Any,
) -> dict[str, Any]:
    return log_runtime_event(
        "provider_switch",
        runtime_id=runtime_id,
        history=history,
        logger=logger,
        old_provider=str(old_provider or "").strip().lower() or None,
        new_provider=str(new_provider or "").strip().lower() or None,
        **fields,
    )


def log_default_model_change(
    old_model: str | None,
    new_model: str | None,
    *,
    runtime_id: str | None = None,
    history: RuntimeEventHistory | None = None,
    logger: logging.Logger | None = None,
    **fields: Any,
) -> dict[str, Any]:
    return log_runtime_event(
        "default_model_change",
        runtime_id=runtime_id,
        history=history,
        logger=logger,
        old_model=str(old_model or "").strip() or None,
        new_model=str(new_model or "").strip() or None,
        **fields,
    )


def log_health_transition(
    old_status: str | None,
    new_status: str | None,
    *,
    runtime_id: str | None = None,
    history: RuntimeEventHistory | None = None,
    logger: logging.Logger | None = None,
    **fields: Any,
) -> dict[str, Any]:
    return log_runtime_event(
        "health_transition",
        runtime_id=runtime_id,
        history=history,
        logger=logger,
        old_status=str(old_status or "").strip().lower() or None,
        new_status=str(new_status or "").strip().lower() or None,
        **fields,
    )


def log_provider_health_transition(
    provider: str | None,
    old_status: str | None,
    new_status: str | None,
    *,
    runtime_id: str | None = None,
    history: RuntimeEventHistory | None = None,
    logger: logging.Logger | None = None,
    **fields: Any,
) -> dict[str, Any]:
    return log_runtime_event(
        "provider_health_transition",
        runtime_id=runtime_id,
        history=history,
        logger=logger,
        provider=str(provider or "").strip().lower() or None,
        old_status=str(old_status or "").strip().lower() or None,
        new_status=str(new_status or "").strip().lower() or None,
        **fields,
    )


def log_chat_request_start(
    request_id: str,
    source: str,
    *,
    runtime_id: str | None = None,
    history: RuntimeEventHistory | None = None,
    logger: logging.Logger | None = None,
    **fields: Any,
) -> dict[str, Any]:
    return log_runtime_event(
        "chat_request_start",
        runtime_id=runtime_id,
        history=history,
        logger=logger,
        request_id=str(request_id or "").strip() or None,
        source=str(source or "").strip().lower() or None,
        **fields,
    )


def log_chat_request_end(
    request_id: str,
    duration_ms: int,
    result: str,
    *,
    runtime_id: str | None = None,
    history: RuntimeEventHistory | None = None,
    logger: logging.Logger | None = None,
    **fields: Any,
) -> dict[str, Any]:
    return log_runtime_event(
        "chat_request_end",
        runtime_id=runtime_id,
        history=history,
        logger=logger,
        request_id=str(request_id or "").strip() or None,
        duration_ms=max(0, int(duration_ms)),
        result=str(result or "").strip().lower() or None,
        **fields,
    )


class RuntimeEventRecorder:
    def __init__(self, *, runtime_id: str, max_events: int = 100, logger: logging.Logger | None = None) -> None:
        self.runtime_id = str(runtime_id or "").strip() or "runtime"
        self.history = RuntimeEventHistory(max_events=max_events)
        self.logger = logger or _LOGGER

    def snapshot(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        return self.history.snapshot(limit=limit)

    def log_runtime_event(self, event_name: str, **fields: Any) -> dict[str, Any]:
        return log_runtime_event(
            event_name,
            runtime_id=self.runtime_id,
            history=self.history,
            logger=self.logger,
            **fields,
        )

    def log_runtime_phase_change(self, old_phase: str | None, new_phase: str | None, **fields: Any) -> dict[str, Any]:
        return log_runtime_phase_change(
            old_phase,
            new_phase,
            runtime_id=self.runtime_id,
            history=self.history,
            logger=self.logger,
            **fields,
        )

    def log_provider_switch(self, old_provider: str | None, new_provider: str | None, **fields: Any) -> dict[str, Any]:
        return log_provider_switch(
            old_provider,
            new_provider,
            runtime_id=self.runtime_id,
            history=self.history,
            logger=self.logger,
            **fields,
        )

    def log_default_model_change(self, old_model: str | None, new_model: str | None, **fields: Any) -> dict[str, Any]:
        return log_default_model_change(
            old_model,
            new_model,
            runtime_id=self.runtime_id,
            history=self.history,
            logger=self.logger,
            **fields,
        )

    def log_health_transition(self, old_status: str | None, new_status: str | None, **fields: Any) -> dict[str, Any]:
        return log_health_transition(
            old_status,
            new_status,
            runtime_id=self.runtime_id,
            history=self.history,
            logger=self.logger,
            **fields,
        )

    def log_provider_health_transition(
        self,
        provider: str | None,
        old_status: str | None,
        new_status: str | None,
        **fields: Any,
    ) -> dict[str, Any]:
        return log_provider_health_transition(
            provider,
            old_status,
            new_status,
            runtime_id=self.runtime_id,
            history=self.history,
            logger=self.logger,
            **fields,
        )

    def log_chat_request_start(self, request_id: str, source: str, **fields: Any) -> dict[str, Any]:
        return log_chat_request_start(
            request_id,
            source,
            runtime_id=self.runtime_id,
            history=self.history,
            logger=self.logger,
            **fields,
        )

    def log_chat_request_end(self, request_id: str, duration_ms: int, result: str, **fields: Any) -> dict[str, Any]:
        return log_chat_request_end(
            request_id,
            duration_ms,
            result,
            runtime_id=self.runtime_id,
            history=self.history,
            logger=self.logger,
            **fields,
        )
