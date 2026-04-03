from __future__ import annotations

from typing import Any

from skills.git import safe_git

_APPROVAL_COMMIT = "APPROVE:git_commit"
_APPROVAL_TAG = "APPROVE:git_tag"
_APPROVAL_PUSH = "APPROVE:git_push"


def _blocked(message: str) -> dict[str, Any]:
    return {"status": "blocked", "message": message}


def _result_payload(result: safe_git.GitResult) -> dict[str, Any]:
    return {
        "status": "ok" if result.ok else "error",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error": result.error,
    }


def git_status(context: dict[str, Any]) -> dict[str, Any]:
    result = safe_git.git_status(context.get("db"))
    return _result_payload(result)


def git_add_all(context: dict[str, Any]) -> dict[str, Any]:
    result = safe_git.git_add_all(context.get("db"))
    return _result_payload(result)


def git_commit(context: dict[str, Any], message: str, approval_token: str) -> dict[str, Any]:
    if approval_token != _APPROVAL_COMMIT:
        return _blocked("Approval token required for git_commit.")
    result = safe_git.git_commit(context.get("db"), message)
    return _result_payload(result)


def git_tag(context: dict[str, Any], name: str, message: str, approval_token: str) -> dict[str, Any]:
    if approval_token != _APPROVAL_TAG:
        return _blocked("Approval token required for git_tag.")
    result = safe_git.git_tag(context.get("db"), name, message)
    return _result_payload(result)


def git_push(context: dict[str, Any], approval_token: str) -> dict[str, Any]:
    if approval_token != _APPROVAL_PUSH:
        return _blocked("Approval token required for git_push.")
    result = safe_git.git_push(context.get("db"))
    return _result_payload(result)
