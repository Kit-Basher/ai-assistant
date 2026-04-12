from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

REPO_ROOT = os.path.abspath(os.path.expanduser("~/personal-agent"))


@dataclass
class GitResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    error: str | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(db: Any, payload: dict[str, Any]) -> None:
    if not db:
        return
    db.log_activity("git_invocation", payload)


def _ensure_repo_root() -> GitResult | None:
    if os.path.abspath(REPO_ROOT) != REPO_ROOT:
        return GitResult(
            ok=False,
            returncode=1,
            stdout="",
            stderr="",
            error="invalid_repo_root",
        )
    if not os.path.isdir(REPO_ROOT):
        return GitResult(
            ok=False,
            returncode=1,
            stdout="",
            stderr="",
            error="repo_not_found",
        )
    if not os.path.isdir(os.path.join(REPO_ROOT, ".git")):
        return GitResult(
            ok=False,
            returncode=1,
            stdout="",
            stderr="",
            error="repo_not_git",
        )
    return None


def _run(db: Any, argv: list[str]) -> GitResult:
    guard = _ensure_repo_root()
    if guard:
        _log(
            db,
            {
                "ts": _now_iso(),
                "argv": argv,
                "stdout": "",
                "stderr": "",
                "returncode": guard.returncode,
                "error": guard.error,
            },
        )
        return guard

    result = subprocess.run(
        argv,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    payload = {
        "ts": _now_iso(),
        "argv": argv,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": result.returncode,
        "error": None,
    }
    _log(db, payload)
    return GitResult(
        ok=result.returncode == 0,
        returncode=result.returncode,
        stdout=stdout,
        stderr=stderr,
        error=None if result.returncode == 0 else "git_failed",
    )


def git_status(db: Any) -> GitResult:
    return _run(db, ["git", "status", "-sb"])


def git_add_all(db: Any) -> GitResult:
    return _run(db, ["git", "add", "-A"])


def git_commit(db: Any, message: str) -> GitResult:
    return _run(db, ["git", "commit", "-m", message])


def git_tag(db: Any, name: str, message: str) -> GitResult:
    return _run(db, ["git", "tag", "-a", name, "-m", message])


def git_push(db: Any) -> GitResult:
    return _run(db, ["git", "push"])
