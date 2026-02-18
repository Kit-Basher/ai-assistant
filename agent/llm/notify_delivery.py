from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class DeliveryTarget:
    id: str
    kind: str
    enabled: bool
    configured: bool


@dataclass(frozen=True)
class DeliveryResult:
    ok: bool
    delivered_to: str
    error_kind: str | None
    reason: str


class TelegramTarget:
    def __init__(
        self,
        *,
        token: str | None,
        chat_id: str | None,
        send_fn: Callable[[str, str, str], None],
        enabled: bool = True,
    ) -> None:
        self._token = str(token or "").strip()
        self._chat_id = str(chat_id or "").strip()
        self._send_fn = send_fn
        self._enabled = bool(enabled)

    @property
    def target(self) -> DeliveryTarget:
        configured = bool(self._token and self._chat_id)
        return DeliveryTarget(id="telegram", kind="telegram", enabled=self._enabled, configured=configured)

    def deliver(self, notification: dict[str, Any]) -> DeliveryResult:
        descriptor = self.target
        if not descriptor.enabled:
            return DeliveryResult(ok=False, delivered_to="none", error_kind="delivery_disabled", reason="delivery_disabled")
        if not descriptor.configured:
            return DeliveryResult(
                ok=False,
                delivered_to="none",
                error_kind="telegram_not_configured_or_no_chat",
                reason="telegram_not_configured_or_no_chat",
            )
        try:
            text = str(notification.get("message") or "").strip()
            self._send_fn(self._token, self._chat_id, text)
        except Exception as exc:
            kind = exc.__class__.__name__
            return DeliveryResult(
                ok=False,
                delivered_to="none",
                error_kind=f"telegram_send_failed:{kind}",
                reason=f"telegram_send_failed:{kind}",
            )
        return DeliveryResult(ok=True, delivered_to="telegram", error_kind=None, reason="sent")


class LocalTarget:
    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = bool(enabled)

    @property
    def target(self) -> DeliveryTarget:
        return DeliveryTarget(id="local", kind="local", enabled=self._enabled, configured=True)

    def deliver(self, notification: dict[str, Any]) -> DeliveryResult:
        _ = notification
        if not self._enabled:
            return DeliveryResult(ok=False, delivered_to="none", error_kind="delivery_disabled", reason="delivery_disabled")
        return DeliveryResult(ok=True, delivered_to="local", error_kind=None, reason="sent_local")
