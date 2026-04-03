from __future__ import annotations

import re

from agent.epistemics.types import CandidateContract, ContextPack


_ALLOWED_VERBS = {"Run", "Open", "Paste", "Create", "Commit", "Tag", "Test", "Add", "Update", "Check"}
_COMMAND_STARTERS = {
    "pytest",
    "python",
    "python3",
    "git",
    "npm",
    "pnpm",
    "yarn",
    "make",
    "cargo",
    "go",
    "uv",
    "pip",
    "poetry",
    "bash",
    "sh",
    "node",
    "npx",
    "ruff",
    "mypy",
}
_ARTIFACT_INTENT_TOKENS = ("write", "create", "produce", "generate", "draft", "implement", "update", "add")


def _sanitize_step(step: str) -> str | None:
    value = " ".join((step or "").replace("\n", " ").split())
    if not value:
        return None
    if "?" in value:
        value = value.replace("?", "").strip()
    if not value:
        return None
    verb = value.split(" ", 1)[0]
    if verb not in _ALLOWED_VERBS:
        return None
    if len(value) > 120:
        return None
    return value


def _has_supported_evidence(candidate: CandidateContract, ctx: ContextPack) -> bool:
    for claim in candidate.claims:
        if claim.support == "user" and claim.user_turn_id and claim.user_turn_id in ctx.recent_turn_ids:
            return True
        if claim.support == "memory" and claim.memory_id is not None and str(claim.memory_id) in ctx.in_scope_memory_ids:
            return True
        if claim.support == "tool" and claim.tool_event_id and claim.tool_event_id in ctx.tool_event_ids:
            return True
    return False


def _is_command_like(command: str) -> bool:
    value = " ".join((command or "").split())
    if not value:
        return False
    first = value.split(" ", 1)[0].strip()
    lowered = first.lower()
    if lowered in _COMMAND_STARTERS:
        return True
    if first.startswith("./") or first.startswith("/") or first.endswith(".sh") or first.endswith(".py"):
        return True
    return False


def _clean_command(command: str) -> str:
    value = command.strip()
    if value.startswith("$ "):
        value = value[2:]
    value = value.strip().rstrip(".,;")
    return " ".join(value.split())


def _extract_command(answer_text: str) -> str | None:
    lines = [line.strip() for line in answer_text.splitlines()]

    # Explicit shell line.
    for line in lines:
        if line.startswith("$ "):
            command = _clean_command(line)
            if _is_command_like(command):
                return command

    # Command inside inline backticks.
    for match in re.finditer(r"`([^`\n]+)`", answer_text):
        command = _clean_command(match.group(1))
        if _is_command_like(command):
            return command

    # Bare command line.
    for line in lines:
        if not line or " " not in line:
            continue
        command = _clean_command(line)
        if _is_command_like(command):
            return command

    # Entire answer is a command.
    full = _clean_command(answer_text)
    if _is_command_like(full):
        return full
    return None


def _extract_target_file(answer_text: str) -> str | None:
    for match in re.finditer(r"`([A-Za-z0-9._/\-]+)`", answer_text):
        target = match.group(1).strip()
        if "/" in target or "." in target:
            return target
    for match in re.finditer(r"\b([A-Za-z0-9._-]+/[A-Za-z0-9._/\-]+)\b", answer_text):
        return match.group(1).strip()
    return None


def _artifact_intent(user_text: str) -> bool:
    lowered = user_text.lower()
    return any(token in lowered for token in _ARTIFACT_INTENT_TOKENS)


def compute_next_action(user_text: str, ctx: ContextPack, candidate: CandidateContract) -> str | None:
    if candidate.kind != "answer":
        return None
    if not _has_supported_evidence(candidate, ctx):
        return None

    answer_text = candidate.final_answer or ""
    if not answer_text.strip():
        return None

    has_tool_support = any(
        claim.support == "tool" and claim.tool_event_id and claim.tool_event_id in ctx.tool_event_ids
        for claim in candidate.claims
    )
    has_user_support = any(
        claim.support == "user" and claim.user_turn_id and claim.user_turn_id in ctx.recent_turn_ids
        for claim in candidate.claims
    )

    command = _extract_command(answer_text)
    if has_tool_support and command:
        return _sanitize_step(f"Run {command}")

    if command and has_user_support:
        return _sanitize_step(f"Run {command}")

    if _artifact_intent(user_text):
        target = _extract_target_file(answer_text)
        if target:
            return _sanitize_step(f"Paste this into {target}")

    return None

