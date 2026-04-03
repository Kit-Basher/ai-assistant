from __future__ import annotations

import os

from agent.config import load_observe_config
from agent.skills_loader import SkillLoader
from memory.db import MemoryDB


def run_once() -> None:
    config = load_observe_config()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    db = MemoryDB(config.db_path)
    schema_path = os.path.join(repo_root, "memory", "schema.sql")
    skills_path = os.path.join(repo_root, "skills")
    timezone_name = os.getenv("AGENT_TIMEZONE", "UTC")
    db.init_schema(schema_path)
    user_id = db.get_preference("telegram_chat_id") or "system"
    audit_id = db.audit_log_create(
        user_id=str(user_id),
        action_type="observe_now_scheduled",
        action_id="observe_now_scheduled",
        status="started",
        details={"event_type": "observe_now_scheduled"},
    )
    try:
        skills = SkillLoader(skills_path).load_all()
        skill = skills.get("observe_now")
        if not skill or "observe_now" not in skill.functions:
            raise RuntimeError("observe_now skill not available")
        fn = skill.functions["observe_now"].handler
        fn({"db": db, "timezone": timezone_name, "user_id": str(user_id)}, user_id=str(user_id))
        db.audit_log_update_status(
            audit_id,
            "executed",
            details={"event_type": "observe_now_scheduled", "status": "executed"},
        )
    except Exception as exc:
        db.audit_log_update_status(audit_id, "failed", str(exc))
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run_once()
