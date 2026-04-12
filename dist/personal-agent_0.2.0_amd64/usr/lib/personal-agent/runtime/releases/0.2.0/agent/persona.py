from __future__ import annotations

import re


_STATUS_READY_RE = re.compile(r"(?i)^agent is ready\.\s+using\s+")
_STATUS_READY_ONLY_RE = re.compile(r"(?i)^agent is ready\.$")
_APPROVAL_RE = re.compile(r"(?i)\breply yes to proceed or no to cancel\.")
_HELP_COMMAND_RE = re.compile(r"(?i)send 'help' for commands\.")
_LOCAL_PATH_ERROR_RE = re.compile(r"(?i)^i can't do that yet\.\s*it needs a valid local path\.$")
_LOCAL_PATH_ERROR_ALT_RE = re.compile(r"(?i)^i cannot do that yet\.\s*it needs a valid local path\.$")
_RUNTIME_STATUS_UNAVAILABLE_RE = re.compile(
    r"(?i)^i can't read a clean runtime status from the current state yet\.$"
)


def normalize_persona_text(text: str | None) -> str:
    cleaned = str(text or "")
    if not cleaned.strip():
        return ""
    normalized = cleaned
    normalized = _STATUS_READY_RE.sub("Ready. Using ", normalized, count=1)
    normalized = _STATUS_READY_ONLY_RE.sub("Ready.", normalized, count=1)
    normalized = _APPROVAL_RE.sub("Say yes to continue, or no to cancel.", normalized)
    normalized = _HELP_COMMAND_RE.sub("Say help for commands.", normalized)
    if _LOCAL_PATH_ERROR_RE.match(normalized.strip()) or _LOCAL_PATH_ERROR_ALT_RE.match(normalized.strip()):
        normalized = "That won't work yet - it needs a valid local path."
    if _RUNTIME_STATUS_UNAVAILABLE_RE.match(normalized.strip()):
        normalized = "I can't read a clean runtime status yet."
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


__all__ = ["normalize_persona_text"]
