from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import urllib.error
import urllib.parse
import urllib.request

from agent.config import load_config
from agent.llm.router import LLMRouter
from agent.model_scout import build_model_scout
from agent.secret_store import SecretStore
from memory.db import MemoryDB


_TELEGRAM_BOT_TOKEN_SECRET_KEY = "telegram:bot_token"


def _log(message: str) -> None:
    print(f"[scheduled_model_scout] {message}", flush=True)


def _resolve_telegram_bot_token(secret_store: SecretStore) -> str | None:
    secret_token = (secret_store.get_secret(_TELEGRAM_BOT_TOKEN_SECRET_KEY) or "").strip()
    if secret_token:
        return secret_token
    env_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    return env_token or None


def _send_telegram_message(token: str, chat_id: str, text: str) -> None:
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"telegram_send_failed: {exc}") from exc

    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("telegram_send_failed: invalid_response") from exc

    if not isinstance(parsed, dict) or not bool(parsed.get("ok")):
        description = str((parsed or {}).get("description") or "unknown_error")
        raise RuntimeError(f"telegram_send_failed: {description}")


def run_once() -> int:
    config = load_config(require_telegram_token=False)
    scout = build_model_scout(config)
    db: MemoryDB | None = None

    try:
        secret_store = SecretStore(path=os.getenv("AGENT_SECRET_STORE_PATH", "").strip() or None)
        router = LLMRouter(config, log_path=config.log_path, secret_store=secret_store)

        db = MemoryDB(config.db_path)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        db.init_schema(os.path.join(repo_root, "memory", "schema.sql"))

        token = _resolve_telegram_bot_token(secret_store)
        chat_id = (db.get_preference("telegram_chat_id") or "").strip()

        def _notify_sender(message: str, _batch: list[dict[str, object]]) -> None:
            if not token or not chat_id:
                raise RuntimeError("telegram_not_configured")
            _send_telegram_message(token, chat_id, message)

        notify_sender = _notify_sender if token and chat_id else None
        if notify_sender is None:
            _log("notify_skip reason=telegram_not_configured")

        result = scout.run(
            registry_document=router.registry.to_document(),
            router_snapshot=router.doctor_snapshot(),
            usage_stats_snapshot=router.usage_stats_snapshot(),
            notify_sender=notify_sender,
        )

        if result.get("error"):
            _log(f"completed_with_error error={result['error']}")
        _log(
            "completed"
            f" ok={bool(result.get('ok'))}"
            f" fetched_trending={int(result.get('fetched_trending') or 0)}"
            f" suggestions={len(result.get('suggestions') or [])}"
            f" new={len(result.get('new_suggestions') or [])}"
            f" notified={int(result.get('notified') or 0)}"
            f" ts={datetime.now(timezone.utc).isoformat()}"
        )
        return 0
    finally:
        if db is not None:
            db.close()
        scout.close()


if __name__ == "__main__":
    raise SystemExit(run_once())
