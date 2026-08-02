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

_LOCAL_PATH_RE = re.compile(r"(?<!\w)(?P<path>(?:~|/|\./|\.\./)[^\s,;!?]+)")
_QUOTED_VALUE_RE = re.compile(r"(?P<quote>['\"`])(?P<value>[^'\"`]+)(?P=quote)")


def normalize_user_meaning(text: str | None) -> str:
    """Normalize for matching while preserving the caller's original text."""
    value = unicodedata.normalize("NFKC", str(text or "")).casefold().replace("’", "'")
    value = re.sub(r"(?<=\w)/(?=\w)", " ", value)
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
    # Retain light grammatical shape even for function words.  This separates
    # conversational questions such as subject/predicate checks from noun-only
    # capability inventories without teaching literal trigger phrases.
    for index in range(len(tokens) - 1):
        vector[f"s:{tokens[index]}_{tokens[index + 1]}"] += 0.8
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
    presence_words = has_fuzzy("here", "there", "online", "present", "responding", "available", "ping", "around")
    domain_words = tokens & {
        "model", "models", "ollama", "provider", "telegram", "file", "files",
        "engine", "engines", "pack", "packs", "skill", "skills", "system", "runtime", "search",
        "install", "switch", "upgrade",
    }
    if capability_id == "assistant.presence" and presence_words and len(tokens) <= 7 and not domain_words:
        score += 0.42
    if capability_id == "assistant.capabilities" and has("tool", "tools", "capability", "capabilities", "abilities", "functions"):
        score += 0.28
    if capability_id == "assistant.capabilities" and (
        (has("can") and has("you") and has("do", "handle"))
        or (has("what") and has("help") and has("with"))
    ):
        score += 0.42
    filesystem_domain = has(
        "file", "files", "filename", "folder", "directory", "document", "drive", "download", "downloaded",
        "repo", "repository",
    ) or any(token.startswith(("/", "~")) for token in tokens)
    if capability_id.startswith("filesystem.") and filesystem_domain:
        score += 0.14
        if capability_id == "filesystem.read" and (
            has("read", "open", "preview", "text")
            or has_fuzzy("read", "open", "preview", "text")
            or (has("show", "contents") and has("file", "document"))
        ):
            score += 0.26
        if capability_id == "filesystem.search" and (
            has("find", "locate", "search", "where")
            or has_fuzzy("find", "locate", "search", "where")
        ):
            score += 0.26
        if capability_id == "filesystem.list" and (has("list") or has_fuzzy("list")):
            score += 0.26
        elif capability_id == "filesystem.list" and (
            (has("inside", "under", "beneath", "lives") and not has("find", "search", "locate"))
            or (has("folder", "directory") and has("show", "contents", "inside", "in"))
        ):
            score += 0.26
    if capability_id == "system.status" and (
        has("runtime", "system", "computer", "service", "cpu", "ram", "memory", "resources", "disk", "storage", "running", "working", "healthy", "health", "status", "alive", "slow")
        or has_fuzzy("runtime", "system", "healthy", "health", "status", "storage")
    ):
        score += 0.24
    if capability_id == "system.status" and has("doctor"):
        score += 0.42
    if capability_id == "system.status" and has("agent") and has("running", "working", "healthy", "health", "status", "alive", "doctor"):
        score += 0.16
    model_domain = has("model", "models", "ollama", "openrouter", "gemma", "qwen", "provider", "engine") or has_fuzzy(
        "model", "models", "ollama", "openrouter", "gemma", "qwen", "provider", "engine"
    ) or (has("remote") and has("policy", "cap", "cheap", "free", "switch", "choose"))
    model_inventory_signal = has("active", "current", "using", "configured", "setup", "installed", "available", "ready", "answering", "now", "name", "inventory", "status", "health", "policy", "cap", "choose") or has_fuzzy(
        "active", "current", "using", "configured", "setup", "installed", "available", "ready", "answering", "inventory", "status", "health"
    ) or (has("why") and has("switch"))
    if model_domain and capability_id == "models.inventory" and (model_inventory_signal or has("local", "cloud") or {"set", "up"}.issubset(tokens)):
        score += 0.40 if (has("provider", "providers", "openrouter", "ollama") or has_fuzzy("provider", "providers", "openrouter", "ollama")) and (has("status", "health") or has_fuzzy("status", "health")) else 0.24
    if model_domain and capability_id == "models.scout" and (has("scout") or has_fuzzy("scout")):
        score += 0.40
    elif model_domain and capability_id == "models.scout" and has("discover", "discovery", "catalog", "hugging", "huggingface"):
        score += 0.34
    elif model_domain and capability_id == "models.scout" and has("best", "better", "stronger", "worth", "recommend", "recommendation", "should", "upgrade", "candidate"):
        score += 0.26
    if (
        model_domain
        and capability_id == "models.switch"
        and (
            has("switch", "change", "default", "temporary", "temporarily", "session", "instead")
            or has_fuzzy("switch", "change", "default", "temporary", "temporarily", "session", "instead")
        )
        and not has("should", "would", "which", "what", "why", "recommend")
    ):
        score += 0.28
    if capability_id == "packs.use" and not has("support") and (
        has("pack", "packs", "skill", "skills", "bundle", "bundles", "guidance", "addon")
        or has_fuzzy("pack", "packs", "skill", "skills", "bundle", "bundles", "guidance", "addon")
    ):
        score += 0.26
    if capability_id == "conversation.history" and has("history", "previous", "earlier", "remember", "recall", "conversation", "continue", "again", "last", "carry", "recap", "underway"):
        score += 0.24
    if capability_id == "conversation.history" and has("memory") and not has("ram", "system", "computer", "machine", "resources", "using", "usage", "much", "eating", "consuming"):
        score += 0.36
    if capability_id == "conversation.history" and (
        (has("we") and (has("next", "plan") or has_fuzzy("doing", "should", "back")))
        or (has("plan") and has_fuzzy("back"))
    ):
        score += 0.34
    return score


def _structured_capability_inputs(
    capability_id: str,
    normalized: str,
    supplied: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Derive bounded arguments after capability selection, never select intent.

    These extractors operate only inside the already-selected capability's
    schema.  They may identify a path, target, scope, or view, but cannot change
    the capability ID or fall back to another router.
    """
    result = dict(supplied or {})
    original = str(result.get("text") or normalized)
    tokens = set(normalized.split())

    if capability_id.startswith("filesystem."):
        path_match = _LOCAL_PATH_RE.search(original)
        path_hint = str(result.get("path_hint") or "").strip()
        if not path_hint and path_match:
            path_hint = str(path_match.group("path") or "").strip().rstrip(".\"'`")
        relative_file_match = re.search(
            r"(?<![/\\\w.:-])(?P<name>[A-Za-z0-9][A-Za-z0-9_-]*\.[A-Za-z0-9][A-Za-z0-9_-]*)\b",
            original,
        )
        if not path_hint and capability_id == "filesystem.read" and relative_file_match:
            path_hint = str(relative_file_match.group("name") or "").strip()
        if not path_hint and capability_id == "filesystem.list" and tokens & {"this", "current", "here"}:
            path_hint = "."
        if path_hint:
            result["path_hint"] = path_hint
        if capability_id == "filesystem.search":
            if "filesystem_view" not in result:
                if tokens & {"recent", "just", "downloaded", "download"} and not path_hint:
                    result["filesystem_view"] = "recent_download"
                elif {"can", "search"}.issubset(tokens) and tokens & {"file", "files", "drive"} and not path_hint:
                    result["filesystem_view"] = "capability_status"
                else:
                    result["filesystem_view"] = "search"
            quoted = [str(match.group("value") or "").strip() for match in _QUOTED_VALUE_RE.finditer(original)]
            quoted = [item for item in quoted if item and item != path_hint]
            query = quoted[0] if quoted else ""
            if not query:
                without_path = original.replace(path_hint, " ") if path_hint else original
                patterns = (
                    r"\b(?:named|called|resembling|resembles|matching|matches)\s+(?P<query>[\w.-]+)",
                    r"\b(?:containing|contains|saying)\s+(?P<query>[\w.-]+)",
                )
                for pattern in patterns:
                    match = re.search(pattern, without_path, re.IGNORECASE)
                    if match:
                        query = str(match.group("query") or "").strip()
                        break
            if not query:
                ignored = {
                    "find", "search", "locate", "look", "through", "for", "file", "files", "folder",
                    "directory", "document", "documents", "drive", "under", "inside", "beneath", "within",
                    "whose", "name", "text", "content", "please", "could", "would", "you", "me", "the",
                    "a", "an", "my", "local", "that", "this", "where", "did", "go", "recent", "download",
                    "repo", "repository", "project",
                }
                query_source = original.replace(path_hint, " ") if path_hint else original
                candidates = [
                    token
                    for token in normalize_user_meaning(query_source).split()
                    if token not in ignored
                    and not any(
                        _is_single_edit_or_transposition(token, action)
                        for action in ("find", "search", "locate", "look")
                    )
                ]
                query = " ".join(candidates[-3:]).strip()
            if query:
                result["query"] = query
                if not path_hint:
                    # A named relative search uses the runtime's already
                    # configured filesystem base. The native filesystem skill
                    # still resolves it and enforces every allowed-root rule.
                    result["path_hint"] = "."
            result["search_mode"] = "text" if tokens & {
                "text", "content", "contents", "phrase", "saying", "contains", "containing", "repo", "repository",
            } else "filename"

    elif capability_id == "system.status":
        if tokens & {"fix", "yourself"} or ("why" in tokens and "working" in tokens):
            result["status_scope"] = "self_diagnostics"
        elif "doctor" in tokens:
            result["status_scope"] = "doctor"
        elif "slow" in tokens or "storage" in tokens or (
            tokens & {"memory", "cpu", "resources"} and tokens & {"using", "usage", "much", "eating", "consuming"}
        ):
            result["status_scope"] = "observe"
        else:
            result["status_scope"] = "system" if tokens & {"cpu", "ram", "memory", "disk", "storage", "computer", "machine", "resources"} else "runtime"

    elif capability_id == "models.inventory":
        model_tokens = re.findall(
            r"\b(?:ollama|openrouter|openai):[A-Za-z0-9][A-Za-z0-9._:/-]*|\b[A-Za-z][A-Za-z0-9._-]*:[A-Za-z0-9][A-Za-z0-9._:-]*",
            original,
        )
        if model_tokens:
            result["model_target"] = model_tokens[-1].rstrip(".?!,;")
        provider_match = next(
            (
                provider
                for provider in ("ollama", "openrouter")
                if any(_is_single_edit_or_transposition(token, provider) for token in tokens)
            ),
            None,
        )
        provider_state_question = bool(
            tokens & {"status", "health", "setup", "configured", "current", "now"}
            or {"set", "up"}.issubset(tokens)
        )
        provider_guidance_question = bool(
            tokens & {"gpu", "vram", "debian", "hardware"}
            and tokens & {"setup", "support", "use", "recommend", "provider", "model"}
        )
        if model_tokens and tokens & {"status", "state", "installed", "installing", "download", "downloading", "ready"}:
            result["model_view"] = "lifecycle"
        elif provider_guidance_question:
            result["model_view"] = "provider_guidance"
        elif "policy" in tokens:
            result["model_view"] = "policy"
        elif "cap" in tokens:
            result["model_view"] = "cost_cap"
        elif "why" in tokens and "switch" in tokens and provider_match:
            result["model_view"] = "provider_explanation"
        elif "remote" in tokens and tokens & {"free", "cheap"} and tokens & {"choose", "chosen", "candidate"}:
            result["model_view"] = "tier_candidate"
            result["model_tier"] = "free_remote" if "free" in tokens else "cheap_remote"
        elif "switch" in tokens and tokens & {"what", "which", "would", "should"}:
            result["model_view"] = "switch_candidate"
        elif (tokens & {"provider", "providers"} or provider_match) and provider_state_question:
            result["model_view"] = "providers"
        elif "why" in tokens and tokens & {"using", "configured", "chosen", "selected"}:
            result["model_view"] = "explanation"
        else:
            result["model_view"] = "current" if tokens & {"active", "current", "answering", "using", "configured", "now", "name"} else "inventory"
        result["local_only"] = bool(tokens & {"local", "ollama", "installed", "downloaded"})
        result["remote_only"] = bool(tokens & {"cloud", "remote"})
        if provider_match:
            result["provider_id"] = provider_match

    elif capability_id == "models.switch":
        result["promote_default"] = "default" in tokens
        if "best" in tokens and "local" in tokens:
            result["model_target"] = "__best_local__"
        model_tokens = re.findall(r"\b(?:ollama|openrouter|openai):[A-Za-z0-9][A-Za-z0-9._:/-]*|\b[A-Za-z][A-Za-z0-9._-]*:[A-Za-z0-9][A-Za-z0-9._:-]*", original)
        if model_tokens:
            result["model_target"] = model_tokens[-1].rstrip(".?!,;")

    elif capability_id == "models.scout":
        ordered_tokens = normalized.split()
        generic_model_modifiers = {
            "a", "an", "any", "best", "better", "budget", "chat", "cheap", "cloud", "coding",
            "cost", "different", "free", "good", "language", "local", "low", "model", "models",
            "new", "premium", "remote", "research", "run", "stronger", "the", "those", "use", "what", "which",
        }
        custom_focus: list[str] = []
        for index, token in enumerate(ordered_tokens):
            if token not in {"model", "models"} or index == 0:
                continue
            candidate = ordered_tokens[index - 1]
            if candidate not in generic_model_modifiers and candidate not in custom_focus:
                custom_focus.append(candidate)
        recommendation_question = bool(
            tokens & {"recommend", "recommendation", "should", "best", "better"}
            and tokens & {"model", "models"}
        )
        remote_role = None
        if recommendation_question and tokens & {"cloud", "remote", "cheap", "budget", "premium"}:
            if "premium" in tokens and "coding" in tokens:
                remote_role = "premium_coding"
            elif "premium" in tokens and tokens & {"research", "reasoning", "analysis"}:
                remote_role = "premium_research"
            elif "premium" in tokens:
                remote_role = "premium_general"
            else:
                remote_role = "cheap_cloud"
            result["scout_role"] = remote_role
        if tokens & {"discover", "discovery", "catalog", "download", "hugging", "huggingface"}:
            result["scout_view"] = "discovery"
        elif remote_role or tokens & {"strategy", "policy", "approach", "method"} or (
            "scout" in tokens and tokens & {"what", "sees", "status", "show"}
        ) or (
            tokens & {"what", "which", "why"} and tokens & {"should", "use", "recommend"}
        ):
            result["scout_view"] = "strategy"
        elif custom_focus:
            result["scout_view"] = "inventory"
        else:
            result["scout_view"] = "recommendations"
        scout_domains = ("coding", "research", "chat", "reasoning", "vision", "embedding")
        result["scout_focus"] = [*custom_focus, *(domain for domain in scout_domains if domain in tokens)]
        requested_domain = next((domain for domain in scout_domains if domain in tokens), "chat")
        result["scout_task"] = "reasoning" if requested_domain == "research" else requested_domain

    elif capability_id == "packs.use":
        result["pack_operation"] = "use" if tokens & {"use", "apply", "run"} else "list"
        result["pack_query"] = normalized

    elif capability_id == "conversation.history":
        result["history_focus"] = "recent" if tokens & {"recent", "last", "previous", "earlier", "recap"} else "current"
    return result


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

        normalized_tokens = set(normalized.split())
        referenced_context_id = str(context.get("referenced_capability_id") or "").strip().lower()
        if bool(context.get("setup_repair_followup")):
            return RequestUnderstanding(
                original_text=original,
                normalized_meaning=normalized,
                context_used=(*context_used, "same_thread_setup_repair_followup"),
                fallback_category=FallbackCategory.GENERIC_CHAT,
                audit={"matcher": "offline_feature_vector_v1", "reason": "setup_repair_followup"},
            )
        if (
            referenced_context_id
            and normalized_tokens & {"that", "there", "it", "result", "report"}
            and normalized_tokens & {"concerned", "concerning", "mean", "means", "explain", "why", "anything"}
        ):
            # This is a question about the meaning of the preceding grounded
            # result, not a new presence check or capability invocation. The
            # existing interpretation renderer consumes the bounded result.
            return RequestUnderstanding(
                original_text=original,
                normalized_meaning=normalized,
                context_used=(*context_used, "same_thread_interpretation_followup"),
                fallback_category=FallbackCategory.GENERIC_CHAT,
                audit={"matcher": "offline_feature_vector_v1", "reason": "interpretation_followup"},
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

        short_social_tokens = set(normalized.split())
        if len(normalized.split()) <= 2 and short_social_tokens & {"hello", "hi", "hey", "hiya"}:
            return RequestUnderstanding(
                original_text=original,
                normalized_meaning=normalized,
                context_used=context_used,
                fallback_category=FallbackCategory.CASUAL,
                audit={"matcher": "offline_feature_vector_v1", "reason": "short_social_greeting"},
            )

        # The WP1 conversation capability is read-only. A request to create a
        # durable preference/memory must remain with the existing deterministic
        # mutation flow and cannot be silently reinterpreted as history recall.
        memory_tokens = set(normalized.split())
        if memory_tokens & {"remember", "save", "store", "record"} and not memory_tokens & {
            "what", "which", "show", "recall", "history", "previous", "earlier"
        }:
            return RequestUnderstanding(
                original_text=original,
                normalized_meaning=normalized,
                context_used=context_used,
                fallback_category=FallbackCategory.GENERIC_CHAT,
                audit={"matcher": "offline_feature_vector_v1", "reason": "read_only_memory_boundary"},
            )

        provider_setup_action = bool(
            "configure" in memory_tokens
            or (
                "setup" in memory_tokens
                and not memory_tokens & {"what", "which", "is", "are", "do", "does"}
            )
            or (
                {"set", "up"}.issubset(memory_tokens)
                and not memory_tokens & {"what", "which", "is", "are", "do", "does"}
            )
        )
        if provider_setup_action and memory_tokens & {"provider", "providers", "ollama", "openrouter"}:
            return RequestUnderstanding(
                original_text=original,
                normalized_meaning=normalized,
                context_used=context_used,
                fallback_category=FallbackCategory.GENERIC_CHAT,
                audit={"matcher": "offline_feature_vector_v1", "reason": "provider_setup_boundary"},
            )

        explicit_web_lookup = bool(
            (
                "look up" in normalized
                or memory_tokens & {"lookup", "browse"}
                or memory_tokens & {"web", "internet", "online"} and memory_tokens & {"find", "search", "check"}
                or "search" in memory_tokens and memory_tokens & {"available", "health", "status", "working"}
            )
            and not memory_tokens & {"file", "files", "filename", "folder", "directory", "drive", "repo", "repository"}
            and not re.search(r"\b(?:do not|dont|don't|without|no)\s+(?:web\s+|internet\s+)?(?:search|browse|look)\b", normalized)
        )
        if explicit_web_lookup:
            return RequestUnderstanding(
                original_text=original,
                normalized_meaning=normalized,
                context_used=context_used,
                fallback_category=FallbackCategory.GENERIC_CHAT,
                audit={"matcher": "offline_feature_vector_v1", "reason": "safe_web_lookup_boundary"},
            )

        public_entity_question = bool(
            re.search(r"\b[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)+\b", original)
            and memory_tokens & {"what", "is", "are", "good", "useful", "worth"}
            and not memory_tokens & {"find", "install", "list", "open", "read", "search", "switch"}
            and not memory_tokens & {
                "model", "models", "provider", "providers", "runtime", "system", "file", "files",
                "folder", "directory",
            }
        )
        if public_entity_question:
            return RequestUnderstanding(
                original_text=original,
                normalized_meaning=normalized,
                context_used=context_used,
                fallback_category=FallbackCategory.GENERIC_CHAT,
                audit={"matcher": "offline_feature_vector_v1", "reason": "public_entity_question"},
            )

        if (
            re.search(r"https?://", original, re.IGNORECASE)
            and not memory_tokens & {"file", "files", "folder", "list", "open", "path", "read"}
        ):
            return RequestUnderstanding(
                original_text=original,
                normalized_meaning=normalized,
                context_used=context_used,
                fallback_category=FallbackCategory.GENERIC_CHAT,
                audit={"matcher": "offline_feature_vector_v1", "reason": "external_url_context"},
            )

        governance_question = bool(
            "governance" in memory_tokens
            or {"execution", "mode"}.issubset(memory_tokens)
            or ({"background", "task"}.issubset(memory_tokens) or {"background", "tasks"}.issubset(memory_tokens))
            or ({"managed", "adapter"}.issubset(memory_tokens) or {"managed", "adapters"}.issubset(memory_tokens))
            or memory_tokens & {"skill", "skills"} and memory_tokens & {"approval", "blocked", "waiting"}
        )
        if governance_question:
            return RequestUnderstanding(
                original_text=original,
                normalized_meaning=normalized,
                context_used=context_used,
                fallback_category=FallbackCategory.GENERIC_CHAT,
                audit={"matcher": "offline_feature_vector_v1", "reason": "governance_question"},
            )

        pack_acquisition_action = bool(
            memory_tokens & {"install", "acquire", "create"}
            or ("add" in memory_tokens and "on" not in memory_tokens)
        )
        if pack_acquisition_action and memory_tokens & {
            "pack", "packs", "skill", "skills", "capability", "capabilities"
        }:
            return RequestUnderstanding(
                original_text=original,
                normalized_meaning=normalized,
                context_used=context_used,
                fallback_category=FallbackCategory.GENERIC_CHAT,
                audit={"matcher": "offline_feature_vector_v1", "reason": "pack_acquisition_boundary"},
            )

        # The WP1 filesystem capabilities are explicitly read-only. Creation,
        # editing, moving, and deletion stay with the deterministic mutation
        # controller, which owns preview, approval, and cancellation.
        filesystem_mutation_action = bool(
            memory_tokens & {"create", "make", "write", "edit", "modify", "rename", "move", "delete", "remove"}
            and memory_tokens & {"file", "files", "folder", "directory", "path", "repo", "repository", "project"}
        )
        if filesystem_mutation_action:
            return RequestUnderstanding(
                original_text=original,
                normalized_meaning=normalized,
                context_used=context_used,
                fallback_category=FallbackCategory.GENERIC_CHAT,
                audit={"matcher": "offline_feature_vector_v1", "reason": "read_only_filesystem_boundary"},
            )

        query = _features(normalized)
        semantic_tokens = set(normalized.split())
        if (
            re.search(r"(?<![/\\\w.:-])[A-Za-z0-9][A-Za-z0-9_-]*\.[A-Za-z0-9][A-Za-z0-9_-]*\b", original)
            and semantic_tokens & {"find", "list", "locate", "open", "preview", "read", "search", "show"}
            and not re.search(r"\b(?:do not|dont|don't|without|no)\s+(?:web\s+|internet\s+)?(?:search|browse|look)\b", normalized)
        ):
            # A relative filename is a filesystem-domain signal even though
            # punctuation normalization deliberately separates its suffix.
            semantic_tokens.add("file")
        model_domain_present = any(
            _is_single_edit_or_transposition(token, concept)
            for token in semantic_tokens
            for concept in ("model", "models", "ollama", "openrouter", "gemma", "qwen", "llama", "provider", "engine")
        ) or bool("remote" in semantic_tokens and semantic_tokens & {"policy", "cap", "cheap", "free", "switch", "choose"})
        model_switch_action_present = any(
            _is_single_edit_or_transposition(token, concept)
            for token in semantic_tokens
            for concept in ("switch", "change", "default", "temporary", "temporarily", "try", "use", "another", "different")
        )
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
            query_words = {key[2:] for key in word_keys}
            sequence_keys = {key for key in query if key.startswith("s:")}
            sequence_overlap_count = max(
                (len(sequence_keys & {key for key in vector if key.startswith("s:")}) for vector in vectors),
                default=0,
            )
            fuzzy_overlap_count = max(
                (
                    sum(
                        1
                        for word in query_words
                        if any(
                            _is_single_edit_or_transposition(word, candidate[2:])
                            for candidate in vector
                            if candidate.startswith("w:")
                        )
                    )
                    for vector in vectors
                ),
                default=0,
            )
            if fuzzy_overlap_count > overlap_count:
                score += min(0.18, 0.09 * (fuzzy_overlap_count - overlap_count))
                if sequence_overlap_count:
                    score += min(0.22, 0.11 * sequence_overlap_count)
            if (
                definition.capability_id == "assistant.presence"
                and boost == 0.0
            ):
                score *= 0.15
            if definition.capability_id == "packs.use" and boost == 0.0:
                score *= 0.15
            if definition.capability_id == "assistant.capabilities" and boost == 0.0:
                score *= 0.15
            if definition.capability_id.startswith("filesystem.") and boost == 0.0:
                score *= 0.20
            if definition.capability_id == "system.status" and boost == 0.0:
                score *= 0.15
            if definition.capability_id == "conversation.history" and boost == 0.0:
                score *= 0.15
            if semantic_tokens & {"backup", "restore", "update", "clean", "uninstall", "repair"}:
                if definition.capability_id in {
                    "system.status", "filesystem.list", "filesystem.search", "filesystem.read", "packs.use"
                }:
                    score *= 0.08
            if definition.capability_id == "models.scout" and boost == 0.0:
                score *= 0.20
            if definition.capability_id.startswith("models.") and not model_domain_present:
                # Words such as "change", "current", and "which" occur in
                # ordinary questions too. A model action needs an actual model
                # domain concept before example-vector similarity can select it.
                score *= 0.12
            if definition.capability_id == "models.switch" and not model_switch_action_present:
                # Merely asking which model is active must not become a
                # mutation ambiguity. Switching requires action language; the
                # deterministic controller still owns approval afterward.
                score *= 0.20
            if (
                boost == 0.0
                and overlap_count < 2
                and not (fuzzy_overlap_count > overlap_count and sequence_overlap_count)
            ):
                score *= 0.55
            score = min(1.0, score + boost)
            if definition.capability_id == "system.status" and model_domain_present:
                score *= 0.30
            if (
                definition.capability_id == "system.status"
                and "memory" in semantic_tokens
                and not semantic_tokens & {"ram", "system", "computer", "machine", "resources", "usage", "using"}
            ):
                score *= 0.25
            if semantic_tokens & {"backup", "restore", "update", "clean", "uninstall", "repair"}:
                if definition.capability_id in {
                    "system.status", "filesystem.list", "filesystem.search", "filesystem.read", "packs.use"
                }:
                    score *= 0.08
            available, _ = definition.availability()
            ranked.append(CapabilityCandidate(definition.capability_id, round(score, 4), available, definition.material_group))
        ranked.sort(key=lambda item: (-item.score, item.capability_id))
        top = ranked[0] if ranked else None
        second = ranked[1] if len(ranked) > 1 else None
        top_score = float(top.score if top else 0.0)
        margin = top_score - float(second.score if second else 0.0)
        if (
            top
            and top.capability_id == "models.switch"
            and {"best", "local"}.issubset(semantic_tokens)
            and not semantic_tokens & {"should", "recommend", "recommendation", "would"}
        ):
            margin = max(margin, 0.20)

        # References are allowed only when the same thread supplied a bounded,
        # registry-valid target. They never synthesize approval.
        referenced_id = str(context.get("referenced_capability_id") or "").strip().lower()
        continuation_id = str(context.get("continuation_capability_id") or "").strip().lower()
        reference_words = {"again", "that", "it", "its", "those", "same", "second", "one"}
        explicit_reference_actions = {
            "repair", "fix", "restart", "install", "remove", "delete", "change", "switch",
            "read", "open", "list", "search", "find", "use", "run", "test", "probe",
        }
        reference_threshold = 0.30 if len(normalized.split()) <= 4 else 0.24
        if continuation_id and self.registry.get(continuation_id):
            continuation = self.registry.require(continuation_id)
            top = CapabilityCandidate(
                continuation_id,
                max(top_score, 0.9),
                continuation.availability()[0],
                continuation.material_group,
            )
            top_score = top.score
            margin = max(margin, 0.3)
            context_used = (*context_used, "same_thread_structured_continuation")
        elif (
            referenced_id
            and self.registry.get(referenced_id)
            and set(normalized.split()) & {"that", "it", "those", "second", "one", "other"}
            and not set(normalized.split()) & {"again", "same"}
            and not set(normalized.split()) & explicit_reference_actions
            and len(normalized.split()) <= 8
        ):
            if referenced_id.startswith("filesystem."):
                clarification = "Which file or folder do you mean, and should I list, search, or read it?"
            elif referenced_id.startswith("models."):
                clarification = "Do you want details about that model, or do you want to switch to it?"
            else:
                clarification = "Do you want me to repeat the previous action, or do something different with that result?"
            return RequestUnderstanding(
                original_text=original,
                normalized_meaning=normalized,
                context_used=(*context_used, "same_thread_unresolved_reference"),
                candidates=tuple(ranked[:3]),
                confidence=top_score,
                ambiguity=AmbiguityStatus.AMBIGUOUS,
                clarification=clarification,
                fallback_category=FallbackCategory.CLARIFY,
                audit={"matcher": "offline_feature_vector_v1", "reason": "unresolved_result_reference"},
            )
        elif (
            referenced_id
            and self.registry.get(referenced_id)
            and set(normalized.split()) & reference_words
            and not set(normalized.split()) & explicit_reference_actions
            and top_score < reference_threshold
        ):
            top = CapabilityCandidate(referenced_id, max(top_score, 0.86), self.registry.require(referenced_id).availability()[0], self.registry.require(referenced_id).material_group)
            top_score = top.score
            margin = max(margin, 0.25)
            context_used = (*context_used, "same_thread_capability_reference")

        casual = not ({token for token in normalized.split()} & _ACTION_WORDS)
        threshold = (
            0.24
            if len(normalized.split()) > 4
            else 0.16
            if top_score >= 0.16 and margin >= 0.15
            else 0.30
        )
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
        inputs = _structured_capability_inputs(
            top.capability_id,
            normalized,
            (extracted_inputs or {}).get(top.capability_id),
        )
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
