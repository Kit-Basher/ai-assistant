from __future__ import annotations

import re

from agent.epistemics.types import CandidateContract, Claim


_TRIGGER_PHRASES = (
    "which should i",
    "should i use",
    "a or b",
    "compare",
    "options",
)
_APPROACH_VERBS = {
    "Add",
    "Check",
    "Commit",
    "Create",
    "Define",
    "Implement",
    "Inspect",
    "Open",
    "Refactor",
    "Run",
    "Set",
    "Tag",
    "Test",
    "Update",
    "Use",
    "Verify",
    "Write",
}
_MAX_OPTIONS = 3
_MAX_OPTION_LEN = 100


def _normalize_text(value: str) -> str:
    lowered = (value or "").lower()
    lowered = re.sub(r"[^a-z0-9 ./_\\-]+", " ", lowered)
    return " ".join(lowered.split())


def _claim_has_supported_provenance(claim: Claim) -> bool:
    if claim.support == "tool":
        return bool(claim.tool_event_id)
    if claim.support == "memory":
        return claim.memory_id is not None
    if claim.support == "user":
        return bool(claim.user_turn_id)
    return False


def _supported_claim_texts(candidate: CandidateContract) -> tuple[str, ...]:
    out: list[str] = []
    for claim in candidate.claims:
        if claim.support == "none":
            continue
        if not _claim_has_supported_provenance(claim):
            continue
        text = " ".join((claim.text or "").replace("\n", " ").split())
        if text:
            out.append(text)
    return tuple(out)


def _iter_sentence_chunks(text: str) -> tuple[str, ...]:
    chunks: list[str] = []
    for line in (text or "").splitlines():
        for piece in line.split(". "):
            value = piece.strip()
            if value:
                chunks.append(value)
    return tuple(chunks)


def _sanitize_option(value: str) -> str | None:
    text = " ".join((value or "").replace("\n", " ").split())
    if not text:
        return None
    text = re.sub(r"^(?:[-*]\s+|\d+\.\s+)", "", text).strip()
    text = text.strip("`").strip()
    text = text.replace("?", "").strip()
    text = text.rstrip(" .!;:")
    text = " ".join(text.split())
    if not text:
        return None
    if len(text) > _MAX_OPTION_LEN:
        text = text[:_MAX_OPTION_LEN].rstrip()
    if not text:
        return None
    return text


def _extract_imperative_options(text: str) -> tuple[tuple[str, str], ...]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for chunk in _iter_sentence_chunks(text):
        step = _sanitize_option(chunk)
        if not step:
            continue
        match = re.match(r"^([A-Za-z]+)\b", step)
        if not match:
            continue
        verb = match.group(1).capitalize()
        if verb not in _APPROACH_VERBS:
            continue
        key = step.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((verb, step))
    return tuple(out)


def _extract_structured_options(text: str) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for line in (text or "").splitlines():
        match = re.match(r"^\s*(?:-\s+|\d+\.\s+)(.+)$", line)
        if not match:
            continue
        option = _sanitize_option(match.group(1))
        if not option:
            continue
        key = option.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(option)
    return tuple(out)


def _is_supported(option: str, supported_claims: tuple[str, ...]) -> bool:
    option_norm = _normalize_text(option)
    if not option_norm:
        return False
    for claim_text in supported_claims:
        claim_norm = _normalize_text(claim_text)
        if not claim_norm:
            continue
        if option_norm in claim_norm or claim_norm in option_norm:
            return True
    return False


def _user_trigger(user_text: str) -> bool:
    lowered = (user_text or "").lower()
    return any(token in lowered for token in _TRIGGER_PHRASES)


def _choose_options(
    imperative_supported: tuple[tuple[str, str], ...],
    structured_supported: tuple[str, ...],
) -> list[str]:
    options: list[str] = []
    seen_text: set[str] = set()
    seen_verb: set[str] = set()

    for verb, option in imperative_supported:
        key = option.lower()
        if key in seen_text:
            continue
        if verb in seen_verb:
            continue
        seen_text.add(key)
        seen_verb.add(verb)
        options.append(option)
        if len(options) >= _MAX_OPTIONS:
            return options

    for option in structured_supported:
        key = option.lower()
        if key in seen_text:
            continue
        seen_text.add(key)
        options.append(option)
        if len(options) >= _MAX_OPTIONS:
            return options

    for _, option in imperative_supported:
        key = option.lower()
        if key in seen_text:
            continue
        seen_text.add(key)
        options.append(option)
        if len(options) >= _MAX_OPTIONS:
            return options

    return options


def compute_options(user_text: str, candidate: CandidateContract, rendered_answer: str) -> list[str] | None:
    if candidate.kind != "answer":
        return None

    supported_claims = _supported_claim_texts(candidate)
    if not supported_claims:
        return None

    imperative = _extract_imperative_options(rendered_answer)
    imperative_supported = tuple((verb, text) for verb, text in imperative if _is_supported(text, supported_claims))
    structured = _extract_structured_options(rendered_answer)
    structured_supported = tuple(option for option in structured if _is_supported(option, supported_claims))

    distinct_verbs = {verb for verb, _ in imperative_supported}
    trigger_a = len(distinct_verbs) >= 2 and len(imperative_supported) >= 2
    trigger_b = len(structured_supported) >= 2
    trigger_c = _user_trigger(user_text)
    if not (trigger_a or trigger_b or trigger_c):
        return None

    options = _choose_options(imperative_supported, structured_supported)
    filtered: list[str] = []
    for option in options:
        clean = _sanitize_option(option)
        if not clean:
            continue
        if "?" in clean:
            continue
        filtered.append(clean)
        if len(filtered) >= _MAX_OPTIONS:
            break
    if len(filtered) < 2:
        return None
    return filtered
