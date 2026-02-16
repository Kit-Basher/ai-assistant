from __future__ import annotations

from datetime import datetime, timezone
import os

from agent.config import load_config
from agent.llm.router import LLMRouter
from agent.model_scout import build_model_scout
from agent.secret_store import SecretStore


def _log(message: str) -> None:
    print(f"[scheduled_model_scout] {message}", flush=True)


def run_once() -> int:
    config = load_config(require_telegram_token=False)
    scout = build_model_scout(config)

    try:
        secret_store = SecretStore(path=os.getenv("AGENT_SECRET_STORE_PATH", "").strip() or None)
        router = LLMRouter(config, log_path=config.log_path, secret_store=secret_store)

        result = scout.run(
            registry_document=router.registry.to_document(),
            router_snapshot=router.doctor_snapshot(),
            usage_stats_snapshot=router.usage_stats_snapshot(),
            notify_sender=None,
        )

        if result.get("error"):
            _log(f"completed_with_error error={result['error']}")
        _log(
            "completed"
            f" ok={bool(result.get('ok'))}"
            f" fetched_trending={int(result.get('fetched_trending') or 0)}"
            f" suggestions={len(result.get('suggestions') or [])}"
            f" new={len(result.get('new_suggestions') or [])}"
            f" ts={datetime.now(timezone.utc).isoformat()}"
        )
        return 0
    finally:
        scout.close()


if __name__ == "__main__":
    raise SystemExit(run_once())
