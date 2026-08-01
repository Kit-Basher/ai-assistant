from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
import math
import re
import unicodedata
from typing import Any, Mapping

from agent.capability_registry import ApprovalPolicy, CapabilityMode, CapabilityRegistry


class AmbiguityStatus(str, Enum):
    CLEAR = "clear"
    AMBIGUOUS = "ambiguous"
    NO_MATCH = "no_match"


class FallbackCategory(str, Enum):
    NONE = "none"
    CASUAL = "grounded_casual"
    GENERIC_CHAT = "grounded_generic_chat"
    UNAVAILABLE = "unavailable_capability"
    CLARIFY = "clarification"


@dataclass(frozen=True)
class CapabilityCandidate:
    capability_id: str
    score: float
    available: bool
    material_group: str


@dataclass(frozen=True)
class RequestUnderstanding:
    original_text: str
    normalized_meaning: str
    context_used: tuple[str, ...] = ()
    candidates: tuple[CapabilityCandidate, ...] = ()
    selected_capability_id: str | None = None
    confidence: float = 0.0
    ambiguity: AmbiguityStatus = AmbiguityStatus.NO_MATCH
    read_only: bool = True
    approval_required: bool = False
    approval_state: str = "not_applicable"
    structured_inputs: Mapping[str, Any] = field(default_factory=dict)
    clarification: str | None = None
    fallback_category: FallbackCategory = FallbackCategory.GENERIC_CHAT
    audit: Mapping[str, Any] = field(default_factory=dict)

    def public_audit(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("original_text", None)
        payload["ambiguity"] = self.ambiguity.value
        payload["fallback_category"] = self.fallback_category.value
        return payload


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)?|(?:/|~)[^\s,;!?]+", re.IGNORECASE)
_CASUAL_FEATURES = {
    "hello", "hi", "hey", "thanks", "thank", "morning", "afternoon", "evening",
    "joke",
}
_ACTION_WORDS = {
    "find", "search", "read", "open", "list", "show", "check", "inspect", "switch",
    "change", "run", "use", "install", "delete", "create", "write", "history", "model",
    "file", "files", "pack", "skill", "memory", "system", "runtime",
}
_LOW_INFORMATION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "could", "do", "for", "from",
    "have", "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "should",
    "about", "something", "that", "the", "this", "to", "what", "which", "with", "would", "you", "your", "tell",
}


def normalize_user_meaning(text: str | None) -> str:
    """Normalize for matching while preserving the caller's original text."""
    value = unicodedata.normalize("NFKC", str(text or "")).casefold().replace("’", "'")
    tokens = _TOKEN_RE.findall(value)
    expansions = {
        "u": "you",
        "ur": "your",
        "r": "are",
        "pls": "please",
        "plz": "please",
        "wanna": "want to",
        "gonna": "going to",
    }
    expanded: list[str] = []
    for token in tokens:
        expanded.extend(expansions.get(token, token).split())
    return " ".join(expanded)


def _features(text: str) -> Counter[str]:
    normalized = normalize_user_meaning(text)
    compact = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
    tokens = compact.split()
    vector: Counter[str] = Counter()
    for token in tokens:
        if token in _LOW_INFORMATION_WORDS:
            continue
        vector[f"w:{token}"] += 3.0
        bounded = f"^{token}$"
        for size in (3, 4, 5):
            for index in range(max(0, len(bounded) - size + 1)):
                vector[f"c:{bounded[index:index + size]}"] += 0.18
    meaningful = [token for token in tokens if token not in _LOW_INFORMATION_WORDS]
    for index in range(len(meaningful) - 1):
        vector[f"b:{meaningful[index]}_{meaningful[index + 1]}"] += 1.5
    return vector


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(weight * right.get(key, 0.0) for key, weight in left.items())
    left_norm = math.sqrt(sum(weight * weight for weight in left.values()))
    right_norm = math.sqrt(sum(weight * weight for weight in right.values()))
    return float(dot / (left_norm * right_norm)) if left_norm and right_norm else 0.0


def _is_single_edit_or_transposition(left: str, right: str) -> bool:
    if left == right:
        return True
    if not left or not right or left[0] != right[0] or abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        differences = [index for index, pair in enumerate(zip(left, right, strict=True)) if pair[0] != pair[1]]
        if len(differences) <= 1:
            return True
        return bool(
            len(differences) == 2
            and differences[1] == differences[0] + 1
            and left[differences[0]] == right[differences[1]]
            and left[differences[1]] == right[differences[0]]
        )
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    return any(longer[:index] + longer[index + 1 :] == shorter for index in range(len(longer)))


def _semantic_domain_boost(capability_id: str, tokens: set[str]) -> float:
    """Small concept taxonomy; examples remain the language surface."""
    def has(*concepts: str) -> bool:
        return bool(tokens & set(concepts))

    def has_fuzzy(*concepts: str) -> bool:
        for token in tokens:
            bare = token.strip("/~.,:;!?")
            for concept in concepts:
                if len(bare) >= 3 and len(concept) >= 3 and _is_single_edit_or_transposition(bare, concept):
                    return True
        return False

    score = 0.0
    presence_words = has_fuzzy("here", "there", "online", "present", "responding", "ping", "around")
    domain_words = tokens & {
        "model", "models", "ollama", "provider", "telegram", "file", "files",
        "pack", "packs", "skill", "skills", "system", "runtime", "search",
        "install", "switch", "upgrade",
    }
    if capability_id == "assistant.presence" and presence_words and len(tokens) <= 7 and not domain_words:
        score += 0.42
    if capability_id == "assistant.capabilities" and has("tool", "tools", "capability", "capabilities", "abilities", "functions"):
        score += 0.28
    filesystem_domain = has("file", "files", "folder", "directory", "document", "drive", "download", "downloaded") or any(token.startswith(("/", "~")) for token in tokens)
    if capability_id.startswith("filesystem.") and filesystem_domain:
        score += 0.14
        if capability_id == "filesystem.read" and has("read", "open", "preview", "contents", "text"):
            score += 0.18
        if capability_id == "filesystem.search" and has("find", "locate", "search", "where"):
            score += 0.18
        if capability_id == "filesystem.list" and has("list", "inside", "under", "beneath", "lives"):
            score += 0.18
    if capability_id == "system.status" and has("runtime", "system", "computer", "service", "cpu", "ram", "memory", "resources", "disk", "running", "working", "healthy", "status", "alive", "slow"):
        score += 0.20
    model_domain = has("model", "models", "ollama", "gemma", "qwen", "provider", "engine")
    if model_domain and capability_id == "models.inventory" and has("active", "current", "using", "configured", "installed", "available", "ready", "answering", "now", "name"):
        score += 0.24
    if model_domain and capability_id == "models.scout" and has("scout", "best", "better", "stronger", "worth", "recommend", "recommendation", "should", "upgrade", "candidate"):
        score += 0.26
    if (
        model_domain
        and capability_id == "models.switch"
        and has("switch", "change", "default", "temporary", "temporarily", "session", "instead")
        and not has("should", "would", "which", "what", "recommend")
    ):
        score += 0.28
    if capability_id == "packs.use" and has("pack", "packs", "skill", "skills", "bundle", "bundles", "guidance", "addon"):
        score += 0.26
    if capability_id == "conversation.history" and has("history", "previous", "earlier", "remember", "recall", "conversation", "continue", "again", "last", "carry", "recap", "underway"):
        score += 0.24
    return score


class RequestUnderstandingService:
    """Fast offline semantic matcher plus deterministic state classification."""

    def __init__(self, registry: CapabilityRegistry) -> None:
        self.registry = registry
        self._vectors: dict[str, tuple[Counter[str], ...]] = {}
        for definition in registry.definitions():
            corpus = (definition.description, *definition.example_goals)
            self._vectors[definition.capability_id] = tuple(_features(item) for item in corpus)

    def understand(
        self,
        text: str | None,
        *,
        context: Mapping[str, Any] | None = None,
        extracted_inputs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> RequestUnderstanding:
        original = str(text or "")
        normalized = normalize_user_meaning(original)
        context = dict(context or {})
        context_used = tuple(str(item) for item in context.get("context_used", ()) if str(item).strip())
        if not normalized:
            return RequestUnderstanding(
                original_text=original,
                normalized_meaning=normalized,
                context_used=context_used,
                clarification="What would you like me to do?",
                fallback_category=FallbackCategory.CLARIFY,
                audit={"matcher": "offline_feature_vector_v1", "reason": "empty"},
            )

        # Requests to produce or transform text are ordinary chat, even when the
        # requested text happens to contain a routing word such as "ping",
        # "model", or "thread".  This is a semantic boundary, not a trigger:
        # the user is asking for language output rather than invoking the named
        # runtime concept.
        if re.search(r"\b(?:answer|reply|respond)\s+(?:with\s+)?exactly\b", normalized) or re.search(
            r"\b(?:write|draft|rewrite|rephrase|summarize)\b.*\b(?:saying|text|note|paragraph|sentence)\b",
            normalized,
        ):
            return RequestUnderstanding(
                original_text=original,
                normalized_meaning=normalized,
                context_used=context_used,
                fallback_category=FallbackCategory.GENERIC_CHAT,
                audit={"matcher": "offline_feature_vector_v1", "reason": "language_generation_request"},
            )

        # Value-policy escalation is applied by the existing deterministic
        # preflight and is not a request to inspect or mutate model settings.
        if (
            ("high stakes" in normalized and {"issue", "analysis", "problem"} & set(normalized.split()))
            or (
                "premium model" in normalized
                and {"analysis", "legal", "security", "reasoning"} & set(normalized.split())
                and not {"recommend", "compare", "switch", "change"} & set(normalized.split())
            )
        ):
            return RequestUnderstanding(
                original_text=original,
                normalized_meaning=normalized,
                context_used=context_used,
                fallback_category=FallbackCategory.GENERIC_CHAT,
                audit={"matcher": "offline_feature_vector_v1", "reason": "value_policy_chat_request"},
            )

        if normalized in {"model switch", "switch model"}:
            return RequestUnderstanding(
                original_text=original,
                normalized_meaning=normalized,
                context_used=context_used,
                fallback_category=FallbackCategory.GENERIC_CHAT,
                audit={"matcher": "offline_feature_vector_v1", "reason": "bare_model_topic"},
            )

        query = _features(normalized)
        semantic_tokens = set(normalized.split())
        ranked: list[CapabilityCandidate] = []
        for definition in self.registry.definitions():
            vectors = self._vectors[definition.capability_id]
            score = max((_cosine(query, vector) for vector in vectors), default=0.0)
            boost = _semantic_domain_boost(definition.capability_id, semantic_tokens)
            word_keys = {key for key in query if key.startswith("w:")}
            overlap_count = max(
                (len(word_keys & {key for key in vector if key.startswith("w:")}) for vector in vectors),
                default=0,
            )
            if definition.capability_id == "assistant.presence" and boost == 0.0:
                score *= 0.15
            if definition.capability_id == "packs.use" and boost == 0.0:
                score *= 0.15
            if boost == 0.0 and overlap_count < 2:
                score *= 0.55
            score = min(1.0, score + boost)
            available, _ = definition.availability()
            ranked.append(CapabilityCandidate(definition.capability_id, round(score, 4), available, definition.material_group))
        ranked.sort(key=lambda item: (-item.score, item.capability_id))
        top = ranked[0] if ranked else None
        second = ranked[1] if len(ranked) > 1 else None
        top_score = float(top.score if top else 0.0)
        margin = top_score - float(second.score if second else 0.0)

        # References are allowed only when the same thread supplied a bounded,
        # registry-valid target. They never synthesize approval.
        referenced_id = str(context.get("referenced_capability_id") or "").strip().lower()
        reference_words = {"again", "that", "it", "its", "those", "same", "second", "one"}
        reference_threshold = 0.30 if len(normalized.split()) <= 4 else 0.24
        if (
            referenced_id
            and self.registry.get(referenced_id)
            and set(normalized.split()) & reference_words
            and top_score < reference_threshold
        ):
            top = CapabilityCandidate(referenced_id, max(top_score, 0.86), self.registry.require(referenced_id).availability()[0], self.registry.require(referenced_id).material_group)
            top_score = top.score
            margin = max(margin, 0.25)
            context_used = (*context_used, "same_thread_capability_reference")

        casual = not ({token for token in normalized.split()} & _ACTION_WORDS)
        threshold = 0.30 if len(normalized.split()) <= 4 else 0.24
        if top is None or top_score < threshold:
            casual_tokens = {token for token in normalized.split()}
            casual_signal = bool(
                casual_tokens & _CASUAL_FEATURES
                or any(token.startswith(("hi", "hey", "thank")) for token in casual_tokens)
                or any(
                    _is_single_edit_or_transposition(token, greeting)
                    for token in casual_tokens
                    for greeting in ("hello", "thanks")
                )
            )
            fallback = (
                FallbackCategory.CASUAL
                if casual and casual_signal and len(normalized.split()) <= 2
                else FallbackCategory.GENERIC_CHAT
            )
            return RequestUnderstanding(
                original_text=original,
                normalized_meaning=normalized,
                context_used=context_used,
                candidates=tuple(ranked[:3]),
                confidence=top_score,
                fallback_category=fallback,
                audit={"matcher": "offline_feature_vector_v1", "threshold": threshold, "margin": round(margin, 4)},
            )

        explicit_alternative = " or " in f" {normalized} "
        materially_ambiguous = bool(
            second
            and second.score >= threshold
            and (margin < 0.03 or (explicit_alternative and margin < 0.12))
            and second.material_group != top.material_group
        )
        if materially_ambiguous:
            first_def = self.registry.require(top.capability_id)
            second_def = self.registry.require(second.capability_id)
            question = f"Do you want me to {first_def.description.rstrip('.').lower()}, or {second_def.description.rstrip('.').lower()}?"
            return RequestUnderstanding(
                original_text=original,
                normalized_meaning=normalized,
                context_used=context_used,
                candidates=tuple(ranked[:3]),
                confidence=top_score,
                ambiguity=AmbiguityStatus.AMBIGUOUS,
                clarification=question,
                fallback_category=FallbackCategory.CLARIFY,
                audit={"matcher": "offline_feature_vector_v1", "threshold": threshold, "margin": round(margin, 4)},
            )

        definition = self.registry.require(top.capability_id)
        available, unavailable_reason = definition.availability()
        inputs = dict((extracted_inputs or {}).get(top.capability_id) or {})
        try:
            validated_inputs = definition.input_contract.validate(inputs)
        except ValueError as exc:
            return RequestUnderstanding(
                original_text=original,
                normalized_meaning=normalized,
                context_used=context_used,
                candidates=tuple(ranked[:3]),
                selected_capability_id=top.capability_id,
                confidence=top_score,
                ambiguity=AmbiguityStatus.CLEAR,
                clarification="What specific target should I use?",
                fallback_category=FallbackCategory.CLARIFY,
                audit={"matcher": "offline_feature_vector_v1", "validation": str(exc)},
            )
        fallback = FallbackCategory.NONE if available else FallbackCategory.UNAVAILABLE
        return RequestUnderstanding(
            original_text=original,
            normalized_meaning=normalized,
            context_used=context_used,
            candidates=tuple(ranked[:3]),
            selected_capability_id=top.capability_id,
            confidence=top_score,
            ambiguity=AmbiguityStatus.CLEAR,
            read_only=definition.mode is CapabilityMode.READ_ONLY,
            approval_required=definition.approval_policy is ApprovalPolicy.REQUIRED,
            approval_state="required" if definition.approval_policy is ApprovalPolicy.REQUIRED else "not_required",
            structured_inputs=validated_inputs,
            fallback_category=fallback,
            audit={
                "matcher": "offline_feature_vector_v1",
                "threshold": threshold,
                "margin": round(margin, 4),
                "availability_reason": unavailable_reason,
            },
        )
