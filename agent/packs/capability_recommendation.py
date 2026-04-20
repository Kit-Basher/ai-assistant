from __future__ import annotations

import re
from typing import Any

from agent.persona import normalize_persona_text
from agent.packs.state_truth import normalize_available_pack_truth, normalize_installed_pack_truth


_CAPABILITY_RULES: dict[str, dict[str, Any]] = {
    "dev_tools": {
        "label": "coding tools",
        "phrases": (
            "coding tools",
            "developer tools",
            "dev tools",
            "code tools",
            "coding",
            "code",
            "developer",
            "development",
            "programming",
            "debugging",
        ),
        "search_terms": ("coding", "developer", "programming", "terminal", "git", "test"),
        "installed_blocker": "it isn't enabled as a live capability yet",
        "blocked_blocker": "the pack was blocked during import",
    },
    "system_tools": {
        "label": "system tools",
        "phrases": (
            "system tools",
            "system / pc tasks",
            "system pc tasks",
            "system tasks",
            "pc tasks",
            "computer tasks",
            "diagnostics",
            "maintenance",
            "troubleshoot",
        ),
        "search_terms": ("system", "pc", "computer", "diagnostic", "maintenance", "setup", "files"),
        "installed_blocker": "it isn't enabled as a live capability yet",
        "blocked_blocker": "the pack was blocked during import",
    },
    "creative_tools": {
        "label": "creative tools",
        "phrases": (
            "creative tools",
            "creative writing",
            "writing tools",
            "creative",
            "writing",
            "draft",
            "content",
            "notes",
            "brainstorm",
        ),
        "search_terms": ("writing", "creative", "draft", "notes", "content"),
        "installed_blocker": "it isn't enabled as a live capability yet",
        "blocked_blocker": "the pack was blocked during import",
    },
    "voice_output": {
        "label": "voice output",
        "phrases": (
            "talk to me out loud",
            "talk out loud",
            "speak to me",
            "speak out loud",
            "read aloud",
            "read this aloud",
            "read this page back to me",
            "read this page",
            "read this back to me",
            "read it back to me",
            "read to me",
            "say it aloud",
            "voice output",
            "speech output",
            "text to speech",
            "tts",
        ),
        "search_terms": ("voice", "speech", "audio", "tts"),
        "installed_blocker": "it isn't enabled as a live capability yet",
        "blocked_blocker": "the pack was blocked during import",
    },
    "voice_input": {
        "label": "voice input",
        "phrases": (
            "voice input",
            "speech input",
            "listen to me",
            "listen for me",
            "dictation",
            "speech recognition",
            "transcribe my voice",
            "voice transcription",
        ),
        "search_terms": ("voice", "speech", "listen", "dictation", "transcribe"),
        "installed_blocker": "it isn't enabled as a live capability yet",
        "blocked_blocker": "the pack was blocked during import",
    },
    "avatar_visual": {
        "label": "visual avatar",
        "phrases": (
            "use the avatar",
            "show the avatar",
            "avatar support",
            "visual avatar",
            "visual persona",
            "animated avatar",
            "robot avatar",
            "persona avatar",
        ),
        "search_terms": ("avatar", "persona", "visual"),
        "installed_blocker": "it isn't enabled as a live capability yet",
        "blocked_blocker": "the pack was blocked during import",
    },
    "camera_feed": {
        "label": "camera feed",
        "phrases": (
            "robot camera feed",
            "camera feed",
            "robot camera",
            "camera support",
            "camera stream",
            "see the camera",
            "watch the camera",
            "open the camera",
            "video feed",
        ),
        "search_terms": ("camera", "vision", "robot", "feed", "stream"),
        "installed_blocker": "it isn't enabled as a live capability yet",
        "blocked_blocker": "the pack was blocked during import",
    },
}

_CAPABILITY_HELPER_NAMES = {
    "dev_tools": "coding helper",
    "system_tools": "system helper",
    "creative_tools": "writing helper",
    "voice_output": "voice helper",
    "voice_input": "speech input helper",
    "avatar_visual": "visual avatar helper",
    "camera_feed": "camera helper",
}

_CAPABILITY_PROPOSAL_SUMMARIES = {
    "dev_tools": "helps with coding and terminal work",
    "system_tools": "helps with system tasks and troubleshooting",
    "creative_tools": "helps with drafting, brainstorming, and editing",
    "voice_output": "reads text aloud",
    "voice_input": "listens and transcribes speech",
    "avatar_visual": "shows a visual avatar",
    "camera_feed": "looks through a camera feed",
}

_PARTIAL_HELP_CUES = (
    "help me",
    "can you help",
    "could you help",
    "show me how",
    "walk me through",
    "what is the best way",
    "what's the best way",
    "what is the easiest way",
    "what's the easiest way",
    "recommend",
    "suggest",
    "plan",
    "sketch",
    "guide me",
    "build me",
    "make me",
    "set up",
)

_CAPABILITY_ACTION_CUES = (
    "read aloud",
    "read this to me",
    "read my screen",
    "say it out loud",
    "speak to me",
    "talk to me",
    "narrate",
    "listen",
    "hear me",
    "dictate",
    "transcribe",
    "camera",
    "webcam",
    "watch",
    "look at",
    "scan",
    "avatar",
    "voice",
    "audio",
    "code",
    "script",
    "debug",
    "terminal",
    "install",
    "configure",
    "set up",
    "troubleshoot",
    "monitor",
    "track",
    "alert",
    "notify",
)

_KNOWLEDGE_QUESTION_PREFIXES = (
    "what is ",
    "what are ",
    "what does ",
    "what do ",
    "what's ",
    "whats ",
    "who is ",
    "who are ",
    "how does ",
    "how do ",
    "why is ",
    "why does ",
    "why do ",
    "tell me about ",
    "explain ",
    "define ",
    "describe ",
    "what happened ",
)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "how",
    "i",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "should",
    "the",
    "to",
    "use",
    "what",
    "when",
    "with",
    "you",
}


def _normalize_text(text: str | None) -> str:
    normalized = " ".join(str(text or "").strip().lower().replace("/", " ").split())
    return normalized


def _tokenize(text: str | None) -> list[str]:
    normalized = _normalize_text(text)
    return [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9._-]*", normalized)
        if token and token not in _STOPWORDS
    ]


def _matches_rule(normalized: str, rule: dict[str, Any]) -> bool:
    phrases = tuple(str(item).strip().lower() for item in (rule.get("phrases") if isinstance(rule.get("phrases"), tuple) else ()))
    if any(phrase and phrase in normalized for phrase in phrases):
        return True
    label = str(rule.get("label") or "").strip().lower()
    if label and label in normalized:
        return True
    if label == "camera feed" and "camera" in normalized and any(token in normalized for token in ("feed", "stream", "robot", "vision")):
        return True
    if label == "voice output" and (
        any(token in normalized for token in ("out loud", "speak", "talk", "aloud", "audio", "tts"))
        or any(
            phrase in normalized
            for phrase in (
                "read aloud",
                "read this to me",
                "read this page back to me",
                "read this page",
                "read this back to me",
                "read it back to me",
                "read to me",
                "read my screen",
            )
        )
    ):
        return True
    if label == "voice input" and any(token in normalized for token in ("listen", "dictation", "transcribe", "speech input", "speech to text")):
        return True
    if label == "visual avatar" and "avatar" in normalized:
        return True
    return False


def detect_pack_capability_need(text: str | None) -> dict[str, Any] | None:
    normalized = _normalize_text(text)
    if not normalized:
        return None
    if _classify_request_kind(normalized) == "knowledge":
        return None
    for capability, rule in _CAPABILITY_RULES.items():
        if _matches_rule(normalized, rule):
            return {
                "capability": capability,
                "label": str(rule.get("label") or capability).strip(),
                "detected_from": str(text or "").strip(),
                "confidence": 0.9,
                "fallback": "text_only",
            }
    return None


def _helper_name_for_capability(capability: str | None, label: str | None = None) -> str:
    capability_key = str(capability or "").strip().lower()
    if capability_key in _CAPABILITY_HELPER_NAMES:
        return _CAPABILITY_HELPER_NAMES[capability_key]
    label_text = str(label or "").strip().lower()
    if "voice" in label_text:
        return "voice helper"
    if "camera" in label_text:
        return "camera helper"
    if "avatar" in label_text:
        return "visual avatar helper"
    if "code" in label_text or "dev" in label_text:
        return "coding helper"
    if "system" in label_text or "pc" in label_text:
        return "system helper"
    if "writing" in label_text or "creative" in label_text:
        return "writing helper"
    return "small helper"


def _proposal_summary_for_capability(capability: str | None, label: str | None = None) -> str:
    capability_key = str(capability or "").strip().lower()
    if capability_key in _CAPABILITY_PROPOSAL_SUMMARIES:
        return _CAPABILITY_PROPOSAL_SUMMARIES[capability_key]
    label_text = str(label or "").strip().lower()
    if "voice" in label_text:
        return "reads text aloud"
    if "camera" in label_text:
        return "looks through a camera feed"
    if "avatar" in label_text:
        return "shows a visual avatar"
    if "code" in label_text or "dev" in label_text:
        return "helps with coding and terminal work"
    if "system" in label_text or "pc" in label_text:
        return "helps with system tasks and troubleshooting"
    if "writing" in label_text or "creative" in label_text:
        return "helps with drafting, brainstorming, and editing"
    return "handles this task"


def _compact_request_summary(text: str | None) -> str:
    cleaned = _normalize_text(text)
    if not cleaned:
        return "handles this task"
    prefixes = (
        "can you ",
        "could you ",
        "would you ",
        "please ",
        "help me ",
        "help me to ",
        "help me with ",
        "i need ",
        "i want ",
        "i'd like ",
        "i would like ",
        "build me ",
        "make me ",
        "set up ",
        "set me up to ",
        "create ",
        "add ",
    )
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    lead_ins = (
        "an assistant that ",
        "a assistant that ",
        "assistant that ",
        "an assistant to ",
        "assistant to ",
        "a helper that ",
        "helper that ",
        "a helper to ",
        "helper to ",
        "a tool that ",
        "tool that ",
        "a thing that ",
        "thing that ",
    )
    for lead_in in lead_ins:
        if cleaned.startswith(lead_in):
            cleaned = cleaned[len(lead_in):]
            break
    cleaned = cleaned.strip(" ?.!;:")
    return cleaned or "handles this task"


def _has_partial_help_cues(normalized: str) -> bool:
    return any(cue in normalized for cue in _PARTIAL_HELP_CUES)


def _classify_request_kind(normalized: str) -> str:
    if not normalized:
        return "action"
    if any(normalized.startswith(prefix) for prefix in _KNOWLEDGE_QUESTION_PREFIXES):
        return "knowledge"
    if normalized.endswith("?") and any(
        re.search(rf"\b{token}\b", normalized) for token in ("what", "who", "why", "how", "when", "where")
    ):
        return "knowledge"
    if any(
        cue in normalized
        for cue in (
            "help me",
            "can you",
            "could you",
            "would you",
            "please ",
            "build me",
            "make me",
            "set up",
            "install",
            "configure",
            "troubleshoot",
            "debug",
            "run ",
            "check ",
            "inspect",
            "monitor",
            "kill ",
            "terminate",
            "stop ",
            "close ",
            "quit ",
            "end ",
        )
    ):
        return "action"
    return "action"


def _behavioral_capability_match(normalized: str) -> dict[str, Any] | None:
    if not normalized:
        return None
    if any(phrase in normalized for phrase in ("read aloud", "read this to me", "read my screen", "say it out loud", "speak to me", "talk to me", "narrate", "voice output", "speech output", "text to speech", "tts")):
        return {
            "capability": "voice_output",
            "label": "voice output",
            "confidence": 0.82,
        }
    if any(phrase in normalized for phrase in ("read this page back to me", "read this page", "read this back to me", "read it back to me", "read to me")):
        return {
            "capability": "voice_output",
            "label": "voice output",
            "confidence": 0.82,
        }
    if any(phrase in normalized for phrase in ("listen to me", "hear me", "dictate", "transcribe", "speech to text", "voice input", "speech input", "voice transcription")):
        return {
            "capability": "voice_input",
            "label": "voice input",
            "confidence": 0.82,
        }
    if any(phrase in normalized for phrase in ("camera feed", "camera stream", "webcam", "keep an eye on", "look at the camera", "watch the camera", "see what", "look at what", "video feed", "live feed")):
        return {
            "capability": "camera_feed",
            "label": "camera feed",
            "confidence": 0.8,
        }
    if any(phrase in normalized for phrase in ("avatar", "visual avatar", "visual persona", "animated avatar", "robot avatar")):
        return {
            "capability": "avatar_visual",
            "label": "visual avatar",
            "confidence": 0.8,
        }
    if any(phrase in normalized for phrase in ("write code", "write a script", "code this", "debug this", "terminal", "git ", "run tests", "programming", "developer tools", "coding")):
        return {
            "capability": "dev_tools",
            "label": "coding tools",
            "confidence": 0.8,
        }
    if any(phrase in normalized for phrase in ("install", "configure", "set up", "set up", "troubleshoot", "repair", "fix my pc", "manage files", "monitor the system", "watch my machine", "keep an eye on", "alert me when", "notify me when")):
        return {
            "capability": "system_tools",
            "label": "system tools",
            "confidence": 0.78,
        }
    return None


def _custom_capability_match(normalized: str) -> dict[str, Any] | None:
    if not normalized:
        return None
    request_cues = any(cue in normalized for cue in _PARTIAL_HELP_CUES) or any(
        cue in normalized
        for cue in (
            "can you ",
            "could you ",
            "would you ",
            "please ",
            "i need ",
            "i want ",
            "i'd like ",
            "i would like ",
            "make a ",
            "make an ",
            "make me ",
            "build a ",
            "build an ",
            "build me ",
            "create a ",
            "create an ",
            "add a ",
            "add an ",
            "need a ",
            "want a ",
            "want an ",
            "assistant that ",
            "helper that ",
        )
    )
    if not request_cues:
        return None
    task_cues = any(
        cue in normalized
        for cue in (
            "assistant",
            "helper",
            "tool",
            "something that",
            "something to",
            "thing that",
            "thing to",
            "ability to",
            "coordinate",
            "orchestrate",
            "manage",
            "monitor",
            "track",
            "automate",
            "sync",
            "control",
            "alert",
            "notify",
            "watch",
            "handle",
        )
    )
    if not task_cues:
        return None
    summary = _compact_request_summary(normalized)
    if not summary or summary == "handles this task":
        return None
    return {
        "capability": None,
        "label": "helper",
        "confidence": 0.7,
        "proposal_summary": summary,
        "helper_name": "helper",
    }


def classify_capability_gap_request(text: str | None) -> dict[str, Any]:
    normalized = _normalize_text(text)
    if not normalized:
        return {
            "classification": "can_answer_locally",
            "request_kind": "action",
            "capability": None,
            "label": None,
            "detected_from": str(text or "").strip() or None,
            "confidence": 0.0,
            "reason": "empty",
        }

    request_kind = _classify_request_kind(normalized)
    if request_kind == "knowledge":
        return {
            "classification": "can_answer_locally",
            "request_kind": request_kind,
            "capability": None,
            "label": None,
            "detected_from": str(text or "").strip() or None,
            "confidence": 0.0,
            "reason": "knowledge_question",
        }

    exact_need = detect_pack_capability_need(text)
    behavioral_need = _behavioral_capability_match(normalized)
    need = exact_need or behavioral_need or _custom_capability_match(normalized)
    if need is None:
        return {
            "classification": "can_answer_locally",
            "request_kind": request_kind,
            "capability": None,
            "label": None,
            "detected_from": str(text or "").strip() or None,
            "confidence": 0.0,
            "reason": "no_gap_detected",
        }

    capability = str(need.get("capability") or "").strip().lower() or None
    label = str(need.get("label") or capability or "").strip() or None
    confidence = float(need.get("confidence") or 0.0)
    help_cues = _has_partial_help_cues(normalized)
    classification = "can_partially_answer_but_capability_would_help" if help_cues else "cannot_answer_without_new_capability"
    proposal_summary = str(need.get("proposal_summary") or "").strip() or _proposal_summary_for_capability(capability, label)
    helper_name = str(need.get("helper_name") or "").strip() or _helper_name_for_capability(capability, label)

    if exact_need is not None:
        reason = "exact_capability_phrase"
    elif behavioral_need is not None:
        reason = "behavioral_capability_match"
    else:
        reason = "custom_capability_gap"

    return {
        "classification": classification,
        "request_kind": "capability",
        "capability": capability,
        "label": label,
        "detected_from": str(text or "").strip() or None,
        "confidence": confidence,
        "reason": reason,
        "help_cues": help_cues,
        "proposal_summary": proposal_summary,
        "helper_name": helper_name,
    }


def _pack_name(row: dict[str, Any] | None) -> str:
    if not isinstance(row, dict):
        return "Imported pack"
    review = row.get("review_envelope") if isinstance(row.get("review_envelope"), dict) else {}
    canonical = row.get("canonical_pack") if isinstance(row.get("canonical_pack"), dict) else {}
    source = canonical.get("source") if isinstance(canonical.get("source"), dict) else {}
    for value in (
        review.get("pack_name"),
        canonical.get("display_name"),
        canonical.get("name"),
        source.get("display_name"),
        source.get("name"),
        row.get("name"),
    ):
        cleaned = " ".join(str(value or "").strip().replace("_", "-").split())
        if cleaned:
            return cleaned
    return "Imported pack"


def _pack_corpus(row: dict[str, Any] | None) -> str:
    if not isinstance(row, dict):
        return ""
    review = row.get("review_envelope") if isinstance(row.get("review_envelope"), dict) else {}
    canonical = row.get("canonical_pack") if isinstance(row.get("canonical_pack"), dict) else {}
    capabilities = canonical.get("capabilities") if isinstance(canonical.get("capabilities"), dict) else {}
    source = canonical.get("source") if isinstance(canonical.get("source"), dict) else {}
    source_history = row.get("source_history") if isinstance(row.get("source_history"), list) else []
    parts = [
        _pack_name(row),
        str(row.get("classification") or ""),
        str(row.get("status") or ""),
        str(row.get("pack_type") or ""),
        str(row.get("summary") or ""),
        str(row.get("review_summary") or ""),
        str(review.get("summary") or ""),
        str(capabilities.get("summary") or ""),
        " ".join(str(item) for item in (capabilities.get("declared") if isinstance(capabilities.get("declared"), list) else [])),
        " ".join(str(item) for item in (capabilities.get("inferred") if isinstance(capabilities.get("inferred"), list) else [])),
        str(source.get("origin") or ""),
        str(source.get("url") or ""),
        str(source.get("ref") or ""),
        str(source.get("commit_hash") or ""),
    ]
    for entry in source_history:
        if not isinstance(entry, dict):
            continue
        parts.extend(
            [
                str(entry.get("origin") or ""),
                str(entry.get("url") or ""),
                str(entry.get("ref") or ""),
                str(entry.get("commit_hash") or ""),
            ]
        )
    return " ".join(part for part in parts if part).lower()


def _search_terms(rule: dict[str, Any]) -> list[str]:
    terms = [
        str(item).strip().lower()
        for item in (rule.get("search_terms") if isinstance(rule.get("search_terms"), tuple) else ())
        if str(item).strip()
    ]
    if not terms:
        terms = [str(rule.get("label") or "").strip().lower()]
    return terms


def _score_installed_pack(row: dict[str, Any], rule: dict[str, Any], query_tokens: list[str]) -> float:
    corpus = _pack_corpus(row)
    if not corpus:
        return 0.0
    name = _pack_name(row).lower()
    score = 0.0
    label = str(rule.get("label") or "").strip().lower()
    if label and label in name:
        score += 1.0
    if label and label in corpus:
        score += 0.5
    search_terms = _search_terms(rule)
    score += min(0.5, 0.1 * sum(1 for token in query_tokens if token in corpus))
    score += min(0.4, 0.1 * sum(1 for token in search_terms if token in corpus))
    if str(row.get("status") or "").strip().lower() == "blocked":
        score -= 0.25
    return score


def _score_listing(listing: dict[str, Any], rule: dict[str, Any], query_tokens: list[str]) -> float:
    corpus = " ".join(
        part
        for part in (
            _pack_name(listing),
            str(listing.get("summary") or ""),
            str(listing.get("author") or ""),
            str(listing.get("artifact_type_hint") or ""),
            " ".join(str(item) for item in (listing.get("tags") if isinstance(listing.get("tags"), list) else [])),
            " ".join(str(item) for item in (listing.get("badges") if isinstance(listing.get("badges"), list) else [])),
            str(listing.get("source_kind_hint") or ""),
            str(listing.get("latest_ref_hint") or ""),
        )
        if part
    ).lower()
    if not corpus:
        return 0.0
    score = 0.0
    label = str(rule.get("label") or "").strip().lower()
    if label and label in corpus:
        score += 0.8
    score += min(0.6, 0.12 * sum(1 for token in query_tokens if token in corpus))
    if bool(listing.get("installable_by_current_policy", False)):
        score += 0.45
    else:
        score -= 0.45
    if bool(listing.get("review_required", False)):
        score -= 0.05
    elif listing.get("review_required") is False:
        score += 0.05
    artifact_type = str(listing.get("artifact_type_hint") or "").strip().lower()
    if artifact_type == "portable_text_skill":
        score += 0.3
    elif artifact_type in {"native_code_pack", "experience_pack"}:
        score -= 0.2
    if "voice" in label and "speech" in corpus:
        score += 0.1
    if "avatar" in label and "avatar" in corpus:
        score += 0.1
    if "camera" in label and any(token in corpus for token in ("camera", "vision", "feed", "stream")):
        score += 0.1
    return score


def _resource_note_for_listing(listing: dict[str, Any]) -> str | None:
    artifact_type = str(listing.get("artifact_type_hint") or "").strip().lower()
    summary = " ".join(
        part
        for part in (
            _pack_name(listing),
            str(listing.get("summary") or ""),
            " ".join(str(item) for item in (listing.get("tags") if isinstance(listing.get("tags"), list) else [])),
            " ".join(str(item) for item in (listing.get("badges") if isinstance(listing.get("badges"), list) else [])),
        )
        if part
    ).lower()
    if artifact_type == "portable_text_skill":
        return "lighter"
    if artifact_type in {"native_code_pack", "experience_pack"}:
        return "may need more resources"
    if any(token in summary for token in ("lightweight", "minimal", "small", "compact", "lean")):
        return "lighter"
    if any(token in summary for token in ("full", "complete", "studio", "pro", "heavy", "heavier")):
        return "may need more resources"
    return None


def _capability_width_for_listing(listing: dict[str, Any]) -> int:
    tags = listing.get("tags") if isinstance(listing.get("tags"), list) else []
    badges = listing.get("badges") if isinstance(listing.get("badges"), list) else []
    return len({str(item).strip().lower() for item in list(tags) + list(badges) if str(item).strip()})


def _tradeoff_notes(primary_listing: dict[str, Any], alternate_listing: dict[str, Any]) -> tuple[str | None, str | None]:
    primary_note = _resource_note_for_listing(primary_listing)
    alternate_note = _resource_note_for_listing(alternate_listing)
    if primary_note and alternate_note and primary_note != alternate_note:
        return primary_note, alternate_note
    primary_width = _capability_width_for_listing(primary_listing)
    alternate_width = _capability_width_for_listing(alternate_listing)
    if alternate_width >= primary_width + 2 and alternate_width > 1:
        return "narrower", "broader capability set"
    if primary_width >= alternate_width + 2 and primary_width > 1:
        return "broader capability set", "narrower"
    if primary_note and not alternate_note:
        return primary_note, None
    if alternate_note and not primary_note:
        return None, alternate_note
    return None, None


def _pack_projection(
    *,
    listing: dict[str, Any],
    source: dict[str, Any],
    normalized_state: dict[str, Any],
    score: float,
    tradeoff_note: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "name": _pack_name(listing),
        "source_id": str(source.get("id") or "").strip() or None,
        "source_name": str(source.get("name") or "").strip() or None,
        "source_kind": str(source.get("kind") or "").strip() or None,
        "remote_id": str(listing.get("remote_id") or "").strip() or None,
        "source_url": str(listing.get("source_url") or "").strip() or None,
        "latest_ref_hint": str(listing.get("latest_ref_hint") or "").strip() or None,
        "artifact_type_hint": str(listing.get("artifact_type_hint") or "").strip() or None,
        "installable": bool(normalized_state.get("installable", False)),
        "usable": bool(normalized_state.get("task_usable", False)),
        "reason": reason or ("best_fit_for_machine" if float(score or 0.0) >= 0.8 else "compatibility_unconfirmed"),
        "summary": str(listing.get("summary") or "").strip() or None,
        "preview_handoff": {
            "source": str(listing.get("source_url") or "").strip() or None,
            "source_kind": str(listing.get("source_kind_hint") or source.get("kind") or "").strip() or None,
            "ref": str(listing.get("latest_ref_hint") or "").strip() or None,
        },
        "normalized_state": normalized_state,
        "tradeoff_note": tradeoff_note,
    }


def recommend_packs_for_capability(
    text: str | None,
    *,
    pack_store: Any,
    pack_registry_discovery: Any,
    capability: str | None = None,
) -> dict[str, Any] | None:
    need = detect_pack_capability_need(text)
    if need is None and capability:
        capability_key = str(capability).strip().lower()
        rule = _CAPABILITY_RULES.get(capability_key)
        if rule is not None:
            need = {
                "capability": capability_key,
                "label": str(rule.get("label") or capability_key).strip(),
                "detected_from": str(text or "").strip(),
                "confidence": 0.75,
                "fallback": "text_only",
            }
    if need is None:
        return None

    capability_key = str(need.get("capability") or "").strip().lower()
    rule = _CAPABILITY_RULES.get(capability_key)
    if rule is None:
        return None
    label = str(need.get("label") or rule.get("label") or capability_key).strip() or capability_key
    query_tokens = _tokenize(text)
    search_terms = _search_terms(rule)
    source_errors: list[dict[str, Any]] = []
    installed_candidates: list[dict[str, Any]] = []
    for row in pack_store.list_external_packs() if callable(getattr(pack_store, "list_external_packs", None)) else []:
        if not isinstance(row, dict):
            continue
        score = _score_installed_pack(row, rule, query_tokens)
        if score <= 0.0:
            continue
        installed_candidates.append({"row": row, "score": score})
    installed_candidates.sort(key=lambda item: (-float(item.get("score") or 0.0), _pack_name(item.get("row")).lower()))

    source_candidates: list[dict[str, Any]] = []
    seen_source_candidates: set[tuple[str, str]] = set()
    list_sources = getattr(pack_registry_discovery, "list_sources", None)
    search = getattr(pack_registry_discovery, "search", None)
    sources: list[dict[str, Any]] = []
    if callable(list_sources):
        try:
            maybe_sources = list_sources()
            if isinstance(maybe_sources, list):
                sources = maybe_sources
        except Exception as exc:  # pragma: no cover - defensive discovery fallback
            source_errors.append({"source_id": "discovery", "query": "*", "error": str(exc)})
    for source in sources if isinstance(sources, list) else []:
        if not isinstance(source, dict):
            continue
        if not bool(source.get("enabled", True)):
            continue
        source_id = str(source.get("id") or "").strip()
        if not source_id:
            continue
        for term in search_terms:
            try:
                payload = search(source_id, term) if callable(search) else None
            except Exception as exc:  # pragma: no cover - best effort source read
                source_errors.append({"source_id": source_id, "query": term, "error": str(exc)})
                continue
            if not isinstance(payload, dict):
                continue
            search_rows = payload.get("search") if isinstance(payload.get("search"), dict) else {}
            listings = search_rows.get("results") if isinstance(search_rows.get("results"), list) else []
            for listing in listings:
                if not isinstance(listing, dict):
                    continue
                candidate_key = (
                    source_id,
                    str(listing.get("remote_id") or _pack_name(listing)).strip().lower(),
                )
                if candidate_key in seen_source_candidates:
                    continue
                seen_source_candidates.add(candidate_key)
                score = _score_listing(listing, rule, query_tokens)
                if score <= 0.0:
                    continue
                source_candidates.append(
                    {
                        "source": source,
                        "listing": listing,
                        "score": score,
                        "from_cache": bool(payload.get("from_cache", False)),
                        "stale": bool(payload.get("stale", False)),
                    }
                )

    source_candidates.sort(
        key=lambda item: (
            -float(item.get("score") or 0.0),
            _pack_name(item.get("listing")).lower(),
            str((item.get("source") or {}).get("id") or ""),
        )
    )

    installed_top = installed_candidates[0] if installed_candidates else None
    installable_candidates = [
        item for item in source_candidates if bool((item.get("listing") or {}).get("installable_by_current_policy", False))
    ]
    blocked_candidates = [
        item for item in source_candidates if not bool((item.get("listing") or {}).get("installable_by_current_policy", False))
    ]
    recommended_top = installable_candidates[0] if installable_candidates else None
    blocked_top = blocked_candidates[0] if blocked_candidates else None
    alternate_top = None
    alternate_notes: tuple[str | None, str | None] | None = None
    if recommended_top is not None:
        primary_listing = recommended_top["listing"] if isinstance(recommended_top.get("listing"), dict) else {}
        for candidate in installable_candidates[1:]:
            listing = candidate.get("listing") if isinstance(candidate.get("listing"), dict) else None
            if listing is None:
                continue
            notes = _tradeoff_notes(primary_listing, listing)
            if not any(notes):
                continue
            if float(candidate.get("score") or 0.0) < 0.45:
                continue
            alternate_top = candidate
            alternate_notes = notes
            break

    installed_pack = None
    if installed_top is not None:
        row = installed_top["row"]
        normalized_state = normalize_installed_pack_truth(row)
        installed_pack = {
            "pack_id": str(row.get("pack_id") or row.get("canonical_id") or "").strip() or None,
            "name": _pack_name(row),
            "status": str(row.get("status") or "").strip().lower() or None,
            "installed": True,
            "enabled": normalized_state.get("enabled"),
            "usable": bool(normalized_state.get("task_usable", False)),
            "state": str(normalized_state.get("state_key") or "installed_unknown"),
            "normalized_state": normalized_state,
            "blocker": str(normalized_state.get("blocker") or "").strip() or None,
            "source": row.get("source") if isinstance(row.get("source"), dict) else {},
            "state_label": normalized_state.get("state_label"),
            "status_note": normalized_state.get("status_note"),
            "next_action": normalized_state.get("next_action"),
        }
    recommendation_pack = None
    if recommended_top is not None:
        listing = recommended_top["listing"]
        source = recommended_top["source"] if isinstance(recommended_top.get("source"), dict) else {}
        normalized_state = normalize_available_pack_truth(source, listing)
        recommendation_pack = _pack_projection(
            listing=listing,
            source=source,
            normalized_state=normalized_state,
            score=float(recommended_top.get("score") or 0.0),
            tradeoff_note=_tradeoff_notes(listing, alternate_top["listing"])[0] if alternate_top is not None else _resource_note_for_listing(listing),
        )
    blocked_pack = None
    if blocked_top is not None:
        listing = blocked_top["listing"]
        source = blocked_top["source"] if isinstance(blocked_top.get("source"), dict) else {}
        normalized_state = normalize_available_pack_truth(source, listing)
        blocked_pack = _pack_projection(
            listing=listing,
            source=source,
            normalized_state=normalized_state,
            score=float(blocked_top.get("score") or 0.0),
            tradeoff_note=None,
            reason="blocked_by_policy",
        )
        blocked_pack["blocker"] = str(normalized_state.get("blocker") or "").strip() or "the current policy blocks it"

    if installed_pack is not None:
        state = str(installed_pack.get("state") or "installed_unknown")
    elif recommendation_pack is not None:
        state = "missing"
    elif blocked_pack is not None:
        state = "blocked"
    else:
        state = "missing"

    if recommendation_pack is not None:
        fallback = "install_preview"
        next_step = "Say yes and I'll show the install preview."
    elif installed_pack is not None and str((installed_pack.get("normalized_state") or {}).get("state_key") or "").startswith("installed"):
        fallback = "text_only"
        next_step = "I can keep responding in text, or I can show what would need to change."
    elif blocked_pack is not None:
        fallback = "text_only"
        next_step = "I can still help you set it up or keep this in text."
    else:
        fallback = "propose_new_capability"
        next_step = f"I can help sketch a small {_helper_name_for_capability(capability_key, label)}."

    comparison_mode = "recommended_plus_alternate" if alternate_top is not None and recommendation_pack is not None else "single_recommendation" if recommendation_pack is not None else "installed_only" if installed_pack is not None else "blocked_only" if blocked_pack is not None else "missing"
    alternate_pack = None
    if alternate_top is not None and recommendation_pack is not None:
        listing = alternate_top["listing"]
        source = alternate_top["source"] if isinstance(alternate_top.get("source"), dict) else {}
        normalized_state = normalize_available_pack_truth(source, listing)
        alternate_note = alternate_notes[1] if alternate_notes is not None else None
        alternate_pack = _pack_projection(
            listing=listing,
            source=source,
            normalized_state=normalized_state,
            score=float(alternate_top.get("score") or 0.0),
            tradeoff_note=alternate_note,
            reason="alternate_" + (alternate_note or "relevant").replace(" ", "_"),
        )

    return {
        "ok": True,
        "capability_required": capability_key,
        "capability_label": label,
        "detected_from": str(need.get("detected_from") or "").strip() or None,
        "confidence": float(need.get("confidence") or 0.0),
        "status": state,
        "installed_pack": installed_pack,
        "recommended_pack": recommendation_pack,
        "alternate_pack": alternate_pack,
        "alternates": [alternate_pack] if alternate_pack is not None else [],
        "blocked_pack": blocked_pack,
        "comparison_mode": comparison_mode,
        "fallback": fallback,
        "next_step": next_step,
        "proposal_summary": _proposal_summary_for_capability(capability_key, label) if fallback == "propose_new_capability" else None,
        "helper_name": _helper_name_for_capability(capability_key, label) if fallback == "propose_new_capability" else None,
        "warnings": list(source_errors),
        "source_errors": list(source_errors),
        "queries": list(search_terms),
    }


def render_pack_capability_response(result: dict[str, Any] | None) -> str:
    if not isinstance(result, dict):
        return normalize_persona_text("I can keep responding in text.")
    label = str(result.get("capability_label") or result.get("capability_required") or "That capability").strip()
    installed_pack = result.get("installed_pack") if isinstance(result.get("installed_pack"), dict) else None
    recommended_pack = result.get("recommended_pack") if isinstance(result.get("recommended_pack"), dict) else None
    alternate_pack = result.get("alternate_pack") if isinstance(result.get("alternate_pack"), dict) else None
    blocked_pack = result.get("blocked_pack") if isinstance(result.get("blocked_pack"), dict) else None
    comparison_mode = str(result.get("comparison_mode") or "").strip().lower()

    def _render_tradeoff(name: str, note: str | None) -> str | None:
        cleaned = str(note or "").strip().lower()
        if not cleaned:
            return None
        if cleaned == "lighter":
            return f"{name} looks lighter."
        if cleaned == "may need more resources":
            return f"{name} may need more resources."
        if cleaned == "broader capability set":
            return f"{name} looks more complete."
        if cleaned == "narrower":
            return f"{name} is a narrower fit."
        return f"{name} looks like a relevant option."

    lines: list[str] = []
    if installed_pack is not None:
        normalized_state = installed_pack.get("normalized_state") if isinstance(installed_pack.get("normalized_state"), dict) else {}
        state_key = str(normalized_state.get("state_key") or installed_pack.get("state") or "").strip().lower()
        if state_key == "installed_disabled":
            lines.append(f"{label.capitalize()} is installed, but it is disabled.")
            blocker = str(normalized_state.get("blocker") or installed_pack.get("blocker") or "").strip()
            if blocker:
                lines.append(f"The blocker is {blocker}.")
        elif state_key == "installed_healthy":
            lines.append(f"{label.capitalize()} is installed and healthy, but I can't confirm it's usable for this task yet.")
            blocker = str(normalized_state.get("blocker") or installed_pack.get("blocker") or "").strip()
            if blocker:
                lines.append(f"The blocker is {blocker}.")
        elif state_key == "installed_limited":
            lines.append(f"{label.capitalize()} is installed, but compatibility is not fully confirmed yet.")
            blocker = str(normalized_state.get("blocker") or installed_pack.get("blocker") or "").strip()
            if blocker:
                lines.append(f"The blocker is {blocker}.")
        elif state_key == "installed_blocked":
            lines.append(f"{label.capitalize()} is installed, but it was blocked during import.")
            blocker = str(normalized_state.get("blocker") or installed_pack.get("blocker") or "").strip()
            if blocker:
                lines.append(f"The blocker is {blocker}.")
        else:
            lines.append(f"{label.capitalize()} is installed, but I can't confirm it's usable for this task yet.")
            blocker = str(normalized_state.get("blocker") or installed_pack.get("blocker") or "").strip()
            if blocker:
                lines.append(f"The blocker is {blocker}.")
    else:
        lines.append(f"{label.capitalize()} isn't installed.")

    if recommended_pack is not None:
        pack_name = str(recommended_pack.get("name") or "That pack").strip()
        if alternate_pack is not None and comparison_mode == "recommended_plus_alternate":
            lines.append("I found 2 packs that fit this machine.")
        elif str(recommended_pack.get("reason") or "").strip() == "best_fit_for_machine":
            lines.append(f"I found {pack_name}, which looks like the best fit for this machine.")
        else:
            lines.append(f"I found {pack_name}, which looks like a relevant match.")
        normalized_state = recommended_pack.get("normalized_state") if isinstance(recommended_pack.get("normalized_state"), dict) else {}
        if bool(normalized_state.get("installable", recommended_pack.get("installable", False))):
            if alternate_pack is not None and comparison_mode == "recommended_plus_alternate":
                primary_tradeoff = _render_tradeoff(pack_name, recommended_pack.get("tradeoff_note"))
                alternate_name = str(alternate_pack.get("name") or "The alternate").strip()
                alternate_tradeoff = _render_tradeoff(alternate_name, alternate_pack.get("tradeoff_note"))
                if primary_tradeoff:
                    lines.append(primary_tradeoff)
                if alternate_tradeoff:
                    lines.append(alternate_tradeoff)
                lines.append(f"I'd start with {pack_name}.")
                lines.append(f"Say yes and I'll show the install preview for {pack_name}.")
            else:
                tradeoff = _render_tradeoff(pack_name, recommended_pack.get("tradeoff_note"))
                if tradeoff:
                    lines.append(tradeoff)
                lines.append("It is installable, but I can't confirm it is usable until I fetch and inspect it.")
                lines.append("Say yes and I'll show the install preview.")
        else:
            lines.append("I can't confirm it will work here yet.")
            lines.append("I can still help you set it up or keep this in text.")
    elif blocked_pack is not None:
        pack_name = str(blocked_pack.get("name") or "That pack").strip()
        normalized_state = blocked_pack.get("normalized_state") if isinstance(blocked_pack.get("normalized_state"), dict) else {}
        lines.append(f"I found {pack_name}, but it isn't usable on this machine yet.")
        blocker = str(blocked_pack.get("blocker") or "").strip()
        if blocker:
            lines.append(f"The blocker is {blocker}.")
        lines.append("I can still help you set it up or keep this in text.")
    elif str(result.get("fallback") or "").strip().lower() == "propose_new_capability":
        helper_name = str(result.get("helper_name") or _helper_name_for_capability(result.get("capability_required"), label)).strip()
        proposal_summary = str(result.get("proposal_summary") or "").strip()
        classification = str(result.get("classification") or "").strip().lower()
        discovery_failed = bool(result.get("source_errors"))
        if discovery_failed:
            lines.append("I couldn't check for an existing match cleanly just now.")
        if classification == "can_partially_answer_but_capability_would_help":
            if helper_name:
                lines.append(f"I can help with the text side, but this would be better as a small {helper_name}.")
            else:
                lines.append("I can help with the text side, but this would be better as a small helper.")
        else:
            if helper_name:
                lines.append(f"I couldn't find a ready-made {helper_name} for this yet.")
            else:
                lines.append("I couldn't find a ready-made helper for this yet.")
        if proposal_summary:
            lines.append(f"The narrowest thing to add would be a small helper that {proposal_summary}.")
        else:
            lines.append("The narrowest thing to add would be a small helper for this task.")
        lines.append("If you want, I can help sketch it.")
    else:
        fallback = str(result.get("fallback") or "").strip().lower()
        if fallback == "text_only":
            lines.append("I can still help you set it up or keep this in text.")
        else:
            helper_name = str(result.get("helper_name") or "").strip()
            if helper_name:
                lines.append(f"I can help sketch a small {helper_name}.")
            else:
                lines.append("I can help sketch a small helper.")
    return normalize_persona_text(" ".join(lines).strip())


def build_capability_gap_response(
    text: str | None,
    *,
    pack_store: Any,
    pack_registry_discovery: Any,
) -> dict[str, Any] | None:
    assessment = classify_capability_gap_request(text)
    classification = str(assessment.get("classification") or "").strip().lower()
    request_kind = str(assessment.get("request_kind") or "").strip().lower()
    if classification == "can_answer_locally" or request_kind != "capability":
        return None

    capability = str(assessment.get("capability") or "").strip().lower() or None
    label = str(assessment.get("label") or capability or "that task").strip()
    proposal_summary = str(assessment.get("proposal_summary") or "").strip()
    helper_name = str(assessment.get("helper_name") or "").strip()
    source_errors: list[dict[str, Any]] = []
    recommendation = None

    if capability:
        recommendation = recommend_packs_for_capability(
            text,
            pack_store=pack_store,
            pack_registry_discovery=pack_registry_discovery,
            capability=capability,
        )
        if isinstance(recommendation, dict):
            source_errors = [dict(row) for row in recommendation.get("source_errors") if isinstance(row, dict)] if isinstance(recommendation.get("source_errors"), list) else []

    has_existing_option = bool(
        isinstance(recommendation, dict)
        and (
            recommendation.get("installed_pack")
            or recommendation.get("recommended_pack")
            or recommendation.get("blocked_pack")
        )
    )

    if has_existing_option and isinstance(recommendation, dict):
        rendered_result = dict(recommendation)
        rendered_result["classification"] = classification
        rendered_result["proposal_summary"] = proposal_summary
        rendered_result["helper_name"] = helper_name or _helper_name_for_capability(capability, label)
        rendered = render_pack_capability_response(rendered_result)
        if classification == "can_partially_answer_but_capability_would_help":
            intro = f"I can help with the text side, but this would be better as a small {rendered_result['helper_name']}."
            rendered = normalize_persona_text(f"{intro} {rendered}")
        return {
            "ok": True,
            "type": "capability_gap_plan",
            "summary": rendered,
            "classification": classification,
            "capability_required": capability,
            "capability_label": label,
            "detected_from": str(assessment.get("detected_from") or "").strip() or None,
            "confidence": float(assessment.get("confidence") or 0.0),
            "fallback": str(recommendation.get("fallback") or "install_preview").strip() or "install_preview",
            "next_step": str(recommendation.get("next_step") or "").strip() or None,
            "recommendation": recommendation,
            "source_errors": source_errors,
            "proposal_summary": proposal_summary,
            "helper_name": helper_name or _helper_name_for_capability(capability, label),
        }

    proposal_result = {
        "ok": True,
        "capability_required": capability,
        "capability_label": label,
        "detected_from": str(assessment.get("detected_from") or "").strip() or None,
        "confidence": float(assessment.get("confidence") or 0.0),
        "classification": classification,
        "fallback": "propose_new_capability",
        "proposal_summary": proposal_summary or _proposal_summary_for_capability(capability, label),
        "helper_name": helper_name or _helper_name_for_capability(capability, label),
        "source_errors": source_errors,
        "next_step": "If you want, I can help sketch it.",
        "warnings": source_errors,
        "queries": [],
    }
    rendered = render_pack_capability_response(proposal_result)
    return {
        "ok": True,
        "type": "capability_gap_plan",
        "summary": rendered,
        "classification": classification,
        "capability_required": capability,
        "capability_label": label,
        "detected_from": proposal_result["detected_from"],
        "confidence": proposal_result["confidence"],
        "fallback": proposal_result["fallback"],
        "next_step": proposal_result["next_step"],
        "recommendation": None,
        "source_errors": source_errors,
        "proposal_summary": proposal_result["proposal_summary"],
        "helper_name": proposal_result["helper_name"],
    }
