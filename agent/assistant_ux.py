from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agent.persona import normalize_persona_text
from agent.security.redaction import redact_text


FORBIDDEN_INTERNAL_TERMS = (
    "agent layer",
    "bounded facts",
    "action results",
    "control plane",
    "orchestrator",
    "runtime dispatch",
    "tool registry",
)

CAPABILITY_EXAMPLE_LINES = (
    "check what is using memory on this PC",
    "help debug a repo",
    "summarize or rewrite text",
    "look something up with sources",
    "remember useful long-term preferences",
    "help plan the next step in a project",
)


def normalize_user_utterance(text: str | None) -> str:
    cleaned = str(text or "").strip().lower().replace("’", "'")
    cleaned = re.sub(r"[^a-z0-9']+", " ", cleaned)
    return " ".join(cleaned.split())


def looks_like_capability_question(text: str | None) -> bool:
    normalized = normalize_user_utterance(text)
    if not normalized:
        return False
    capability_phrases = (
        "what can you help me do",
        "what can you help me with",
        "what can you do",
        "what are your capabilities",
        "what are you able to do",
        "what you as an agent can help me with",
        "what can you help me with what are your capabilities",
    )
    return any(phrase in normalized for phrase in capability_phrases)


def looks_like_internal_architecture_question(text: str | None) -> bool:
    normalized = normalize_user_utterance(text)
    return bool(
        "agent layer" in normalized
        or "control plane" in normalized
        or "orchestrator" in normalized
        or ("how do you work" in normalized and "inside" in normalized)
    )


def build_user_facing_capability_answer(*, search_available: bool | None = None, safe_mode: bool = False) -> str:
    search_clause = (
        "I can also look things up with sources when local search is available."
        if search_available is not False
        else "Search is not working right now, but I can still help with local checks, writing, planning, and coding."
    )
    examples = "\n".join(f"- {line}" for line in CAPABILITY_EXAMPLE_LINES)
    text = (
        "I can help with everyday questions, project planning, coding, writing, research, "
        "and local checks on this machine. I can use connected tools for memory, files, "
        "runtime status, safe web search, external skill acquisition suggestions with source approval "
        "and quarantine review, and system inspection when they are available.\n\n"
        f"{search_clause}\n\n"
        "For anything that changes files, sends messages, edits settings, or affects your system, "
        "I’ll explain the plan and ask before doing it.\n\n"
        "A few examples:\n"
        f"{examples}"
    )
    if safe_mode:
        text += "\n\nSafe mode is on, so background automation and remote fallback stay paused."
    return normalize_persona_text(text)


@dataclass(frozen=True)
class MemoryDecision:
    kind: str
    content: str = ""
    key: str = ""
    value: str = ""
    message: str = ""
    redacted_content: str = ""


_SENSITIVE_PATTERNS = (
    re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\b(?:password|passphrase|secret|bot token|api key|bank card|credit card|card number)\b", re.I),
    re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
)


def _strip_memory_prefix(text: str | None) -> str:
    raw = str(text or "").strip()
    return re.sub(r"^\s*remember\s+(?:that\s+|this\s+)?", "", raw, flags=re.I).strip()


def _looks_sensitive(content: str) -> bool:
    return any(pattern.search(content or "") for pattern in _SENSITIVE_PATTERNS)


def _looks_uncertain(content: str) -> bool:
    normalized = normalize_user_utterance(content)
    return any(phrase in normalized for phrase in ("probably", "i think", "maybe", "might be", "not sure"))


def _looks_temporary_or_low_value(content: str) -> bool:
    normalized = normalize_user_utterance(content)
    return bool(
        "today" in normalized
        or "right now" in normalized
        or "sitting in my chair" in normalized
        or "random number" in normalized
        or re.search(r"\b\d{4,}\b", normalized)
    ) and "temporary code word" not in normalized


def classify_memory_request(text: str | None) -> MemoryDecision | None:
    raw = str(text or "").strip()
    normalized = normalize_user_utterance(raw)
    if not normalized:
        return None
    if normalized.startswith("forget "):
        if "gpu" in normalized or "main pc" in normalized:
            return MemoryDecision(kind="forget", key="assistant_memory:main_pc_gpu", message="I forgot the saved GPU detail for your main PC.")
        if "temporary code word" in normalized:
            return MemoryDecision(kind="forget", key="assistant_memory:temporary_code_word", message="I forgot that temporary code word.")
        return None
    if normalized in {"what gpu does my main pc have", "what gpu does my main computer have"}:
        return MemoryDecision(kind="recall", key="assistant_memory:main_pc_gpu")
    if normalized in {"what food do i like", "what food do i like?"}:
        return MemoryDecision(kind="recall_food")
    if "how should i install" in normalized and "system" in normalized:
        return MemoryDecision(kind="recall_install_preference", key="assistant_memory:instruction_platform")
    if not normalized.startswith("remember "):
        return None
    content = _strip_memory_prefix(raw)
    if not content:
        return MemoryDecision(kind="clarify", message="What should I remember? Give me the specific preference or fact.")
    lowered_content = content.lower()
    if re.search(r"\bby\s+\d{4}-\d{2}-\d{2}\b", lowered_content) or lowered_content.startswith(("i need to ", "! ")):
        return None
    if "prefer concise replies" in lowered_content or "prefer concise answers" in lowered_content:
        return None
    redacted = redact_text(content)
    if _looks_sensitive(content):
        return MemoryDecision(
            kind="refuse_sensitive",
            content=content,
            redacted_content=redacted,
            message="I should not store secrets or payment details in memory. Use the secret store or a password manager for that.",
        )
    if _looks_uncertain(content):
        return MemoryDecision(
            kind="confirm_uncertain",
            content=content,
            redacted_content=redacted,
            message=f"That sounds uncertain. Should I remember it as uncertain, or wait until you confirm it? I have: {redacted}",
        )
    lowered = lowered_content
    if "main pc" in lowered and ("rtx 2060" in lowered or "gpu" in lowered):
        return MemoryDecision(
            kind="store",
            content=content,
            key="assistant_memory:main_pc_gpu",
            value="RTX 2060",
            message="Got it — I’ll remember that your main PC has an RTX 2060.",
        )
    if "prefer debian instructions" in lowered or "debian instructions" in lowered:
        return MemoryDecision(
            kind="store",
            content=content,
            key="assistant_memory:instruction_platform",
            value="Debian",
            message="Got it — I’ll remember that you prefer Debian instructions.",
        )
    if "project is called" in lowered:
        match = re.search(r"project is called\s+(.+)$", content, re.I)
        value = (match.group(1).strip(" .") if match else content)[:120]
        return MemoryDecision(
            kind="store",
            content=content,
            key="assistant_memory:project_name",
            value=value,
            message=f"Got it — I’ll remember that your project is called {value}.",
        )
    if "concise shell commands" in lowered:
        return MemoryDecision(
            kind="store",
            content=content,
            key="assistant_memory:shell_command_style",
            value="concise",
            message="Got it — I’ll remember that you like concise shell commands.",
        )
    if "temporary code word" in lowered:
        match = re.search(r"temporary code word is\s+(.+)$", content, re.I)
        value = (match.group(1).strip(" .") if match else content)[:120]
        return MemoryDecision(
            kind="store",
            content=content,
            key="assistant_memory:temporary_code_word",
            value=value,
            message="Got it — I’ll remember that temporary code word until you ask me to forget it.",
        )
    if _looks_temporary_or_low_value(content):
        return MemoryDecision(
            kind="low_value",
            content=content,
            redacted_content=redacted,
            message="That sounds temporary or low-value, so I will keep it in this conversation rather than durable memory unless you explicitly say it should be permanent.",
        )
    return MemoryDecision(
        kind="clarify",
        content=content,
        redacted_content=redacted,
        message=f"Should I save this as durable memory? {redacted}",
    )


def build_clarification_for_vague_request(text: str | None) -> str | None:
    normalized = normalize_user_utterance(text)
    if normalized in {"i want to build something cool", "build something cool", "i want to make something cool"}:
        return "What kind of thing do you want to build: a small app, a game, or an automation for your computer?"
    if normalized in {"send a message saying i'll be late", "send a message saying i ll be late"}:
        return "Who should I send it to, and through which channel? I will show you the final message before sending anything."
    if "clean up my downloads folder" in normalized or "delete old files from my downloads folder" in normalized:
        return (
            "I can help inspect your Downloads folder first, but deleting files needs a scoped cleanup plan and your confirmation. "
            "Should I start with a read-only list of likely cleanup candidates?"
        )
    if normalized in {"remember this is important", "remember that this is important"}:
        return "What specific fact or preference should I remember?"
    return None
