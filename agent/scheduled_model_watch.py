from __future__ import annotations

from datetime import datetime, timezone

from agent.config import load_config
from agent.model_watch_skill import run_watch_once_for_config, top_pick_plan_payload


def _log(message: str) -> None:
    print(f"[scheduled_model_watch] {message}", flush=True)


def run_once() -> int:
    config = load_config(require_telegram_token=False)
    result = run_watch_once_for_config(config, trigger="scheduled")
    _log(
        "completed"
        f" ok={bool(result.get('ok'))}"
        f" found={bool(result.get('found'))}"
        f" fetched_candidates={int(result.get('fetched_candidates') or 0)}"
        f" new_batch_created={bool(result.get('new_batch_created'))}"
        f" ts={datetime.now(timezone.utc).isoformat()}"
    )
    return 0 if bool(result.get("ok")) else 1


__all__ = [
    "run_once",
    "run_watch_once_for_config",
    "top_pick_plan_payload",
]


if __name__ == "__main__":
    raise SystemExit(run_once())
