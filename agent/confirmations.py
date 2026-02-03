from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PendingAction:
    user_id: str
    action: dict[str, Any]
    message: str


class ConfirmationStore:
    def __init__(self) -> None:
        self._pending: dict[str, PendingAction] = {}

    def set(self, pending: PendingAction) -> None:
        self._pending[pending.user_id] = pending

    def pop(self, user_id: str) -> PendingAction | None:
        return self._pending.pop(user_id, None)

    def has(self, user_id: str) -> bool:
        return user_id in self._pending
