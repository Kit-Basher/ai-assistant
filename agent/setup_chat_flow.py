from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import re

from agent.packs.capability_recommendation import classify_capability_gap_request
from agent.nl_router import classify_free_text


_OPENROUTER_KEY_RE = re.compile(r"\b(sk-or-v1-[A-Za-z0-9_-]{16,}|sk-or-[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9_-]{20,})\b")

_OPENROUTER_SETUP_PHRASES = (
    "help me set up openrouter",
    "setup openrouter",
    "set up openrouter",
    "configure openrouter",
    "connect openrouter",
    "repair openrouter",
    "fix openrouter",
    "add openrouter",
)
_OPENROUTER_USE_PHRASES = (
    "use openrouter",
    "switch to openrouter",
    "use openrouter for chat",
    "make openrouter the default",
)
_OLLAMA_SETUP_PHRASES = (
    "help me set up ollama",
    "configure ollama",
    "setup ollama",
    "set up ollama",
    "repair ollama",
    "fix ollama",
)
_OLLAMA_USE_PHRASES = (
    "use ollama",
    "switch to ollama",
    "use ollama for chat",
    "make ollama the default",
)
_OPENROUTER_SETUP_KEYWORDS = ("configure", "connect", "setup", "set up", "add", "repair", "fix")
_OPENROUTER_SWITCH_KEYWORDS = ("use", "switch", "default")
_OLLAMA_SETUP_KEYWORDS = ("configure", "setup", "set up", "repair", "fix")
_OLLAMA_SWITCH_KEYWORDS = ("use", "switch", "default")
_BETTER_LOCAL_MODEL_PHRASES = (
    "switch to a better local model",
    "use a better local model",
    "use the best local model",
    "switch to the best local model",
    "upgrade the local model",
)
_SETUP_EXPLANATION_PHRASES = (
    "check setup",
    "explain what s wrong with setup",
    "explain whats wrong with setup",
    "check setup and explain",
    "diagnose setup",
)
_FIND_OLLAMA_MODELS_PHRASES = (
    "find ollama models",
    "show ollama models",
    "list ollama models",
    "what ollama models do i have",
    "which ollama models do i have",
)
_LOCAL_MODEL_INVENTORY_PHRASES = (
    "what ollama models do we have downloaded",
    "what ollama models do i have downloaded",
    "do we have any other models downloaded",
    "do we have any other local models",
    "what models do we have downloaded",
    "which models do we have downloaded",
    "what local models do we have",
    "which local models do we have",
    "what local models are available",
    "which local models are available",
    "what local models u got",
    "wut local models u got",
    "what local models do you got",
    "what models u got",
    "which models u got",
    "show me local models",
    "list local models",
    "show downloaded models",
    "list downloaded models",
    "what models are installed",
    "which models are installed",
    "what models are downloaded",
    "which models are downloaded",
)
_LOCAL_MODEL_RECOMMENDATION_PHRASES = (
    "recommend a local model",
    "recommend a local chat model",
    "what local model should i use",
    "which local model should i use",
    "what local model do you recommend",
    "which local model do you recommend",
)
_LOCAL_PROVIDER_GUIDANCE_PHRASES = (
    "what local model/provider setup should i use",
    "what local model provider setup should i use",
    "which local model/provider setup should i use",
    "which local model provider setup should i use",
    "what provider setup should i use",
    "which provider setup should i use",
    "local provider setup",
    "local model provider setup",
)
_FILESYSTEM_LIST_PHRASES = (
    "list files in",
    "list files under",
    "list the files in",
    "list the files under",
    "list the files inside",
    "show me what s in",
    "show me whats in",
    "show me what is in",
    "show the files in",
    "show the files under",
    "show files in",
    "show files under",
    "what files are in",
    "what s in",
    "whats in",
    "what is in",
    "list directory",
    "list folder",
    "show directory",
    "show folder",
)
_FILESYSTEM_READ_PHRASES = (
    "read this file",
    "open this text file",
    "open this file",
    "read this text file",
    "inspect this file",
    "inspect the file",
    "read the file",
    "open the file",
)
_FILESYSTEM_STAT_PHRASES = (
    "what is this file",
    "what is this path",
    "how big is this file",
    "how big is this path",
)
_FILESYSTEM_CURRENT_DIRECTORY_PHRASES = (
    "this folder",
    "this directory",
    "this repo",
    "current folder",
    "current directory",
    "workspace root",
    "repo root",
)
_FILESYSTEM_QUOTED_PATH_RE = re.compile(r"(?P<quote>['\"`])(?P<path>[^'\"`]+)(?P=quote)")
_FILESYSTEM_PATH_TOKEN_RE = re.compile(r"(?P<path>(?:~|/|\./|\.\./)[^\s'\"`]+)")
_FILESYSTEM_BARE_FILENAME_RE = re.compile(r"\b(?P<path>[A-Za-z0-9._-]+\.[A-Za-z0-9._-]+)\b")
_SHELL_INLINE_PATH_TOKEN_RE = re.compile(
    r"(?P<path>(?:~|/|\./|\.\./)[^\s'\"`]+|[A-Za-z0-9._-]+)"
)
_RUNTIME_STATUS_PHRASES = (
    "runtime check",
    "give me a runtime check",
    "are you ready",
    "are you working",
    "are you alive",
    "is the agent ready",
    "is the agent healthy right now",
    "is the agent healthy",
    "are you healthy right now",
    "are you healthy",
    "r u healthy right now",
    "r u healthy",
    "is personal agent ready",
    "what is the runtime status",
    "runtime status",
    "what needs attention",
    "system health",
    "health status",
    "check health",
    "what is the agent status",
    "what is agent status",
    "can you read the runtime now",
)
_GOVERNANCE_ADAPTERS_PHRASES = (
    "what managed adapters exist",
    "which managed adapters exist",
    "list managed adapters",
    "what adapters exist",
    "which adapters exist",
)
_GOVERNANCE_BACKGROUND_TASKS_PHRASES = (
    "what background tasks are active",
    "which background tasks are active",
    "what background tasks exist",
    "which background tasks exist",
    "list background tasks",
    "what managed background tasks exist",
)
_GOVERNANCE_BLOCKS_PHRASES = (
    "what got blocked by skill governance",
    "what got blocked by governance",
    "which skills were blocked by governance",
    "what skills were blocked by governance",
    "show blocked skills",
)
_GOVERNANCE_PENDING_PHRASES = (
    "is any skill waiting for approval",
    "is anything waiting for approval",
    "what is waiting for approval",
    "which skills are waiting for approval",
    "which skills need approval",
)
_GOVERNANCE_OVERVIEW_PHRASES = (
    "what persistent background components are currently allowed",
    "what persistent components are currently allowed",
    "what background components are currently allowed",
    "what persistent components are allowed",
)
_TELEGRAM_STATUS_PHRASES = (
    "telegram status",
    "check telegram",
    "no check telegram",
    "no, check telegram",
    "is telegram configured",
    "is telegram running",
    "is telegram working",
    "check if telegram is set up",
    "check if telegram is setup",
    "is telegram set up",
    "is telegram setup",
    "how is telegram",
    "why is telegram not responding",
    "why isn t telegram responding",
    "why isnt telegram responding",
    "telegram not responding",
    "telegram is not responding",
    "send me a telegram test",
    "send a telegram test",
    "test telegram",
)
_TELEGRAM_ACTION_RE = re.compile(r"\b(start|restart|stop)\s+(?:the\s+)?telegram\b|\btelegram\s+(start|restart|stop)\b")
_SAFETY_BYPASS_RE = re.compile(
    r"\b(ignore|bypass|skip|override)\s+(?:the\s+)?(?:safety|policy|approval|confirmation|plan mode|plan)\b"
    r"|(?:\bjust\s+run\s+it\b|\brun\s+it\s+anyway\b|\bdo\s+it\s+anyway\b)"
)
_SECRET_REVEAL_RE = re.compile(
    r"\b(show|tell|print|display|reveal|what(?:'s| is)?)\b.*\b(secret|token|api key|apikey|password|credential)\b"
    r"|\b(telegram|openai|openrouter|provider)\b.*\b(secret|token|api key|apikey|password|credential)\b"
)
_AMBIGUOUS_RESTART_RE = re.compile(r"^(?:please\s+)?restart\s+(?:it|that|this|the service|the thing)\??$")
_PLAN_DAY_PHRASES = (
    "plan my day",
    "today plan",
    "help me plan my day",
    "help me plan today",
    "what should i do today",
    "what should i work on today",
    "show quick wins",
    "show top 3 priorities",
)
_OPERATOR_HEALTH_PHRASES = (
    "is the assistant healthy",
    "is personal agent healthy",
    "is the assistant ok",
    "is personal agent ok",
    "what is broken",
    "what's broken",
    "whats broken",
    "what is broken right now",
)
_OPERATOR_STORAGE_PHRASES = (
    "how much space is this using",
    "how much disk space is this using",
    "how much space does personal agent use",
    "how much disk space does personal agent use",
    "storage usage",
    "personal agent storage usage",
)
_OPERATOR_REPAIR_PHRASES = (
    "repair the assistant",
    "repair personal agent",
    "fix the assistant",
    "fix personal agent",
)
_OPERATOR_BACKUP_PHRASES = (
    "back up the assistant",
    "backup the assistant",
    "back up personal agent",
    "backup personal agent",
    "show my backups",
    "list my backups",
    "show backups",
    "list backups",
)
_OPERATOR_RESTORE_PHRASES = (
    "restore from backup",
    "restore personal agent from backup",
    "restore the assistant from backup",
    "validate this backup",
    "check this backup before restore",
    "inspect backup",
)
_OPERATOR_UPDATE_PHRASES = (
    "update the assistant",
    "update personal agent",
    "upgrade the assistant",
    "upgrade personal agent",
)
_OPERATOR_CLEANUP_PHRASES = (
    "clean old runtime files",
    "cleanup old runtime files",
    "clean up old runtime files",
    "clean old backup files",
    "cleanup old backup files",
    "clean up old backup files",
    "clean old backups",
    "cleanup old backups",
    "clean personal agent old files",
    "clean old personal agent files",
)
_OPERATOR_UNINSTALL_PHRASES = (
    "uninstall the assistant",
    "uninstall personal agent",
    "remove personal agent",
)
_OPERATOR_SUPPORT_BUNDLE_PHRASES = (
    "make a support bundle",
    "create a support bundle",
    "generate a support bundle",
    "support bundle",
)
_CURRENT_MODEL_PHRASES = (
    "what model are you using",
    "which model are you using",
    "what model are you on",
    "what model u on",
    "what model are you on right now",
    "what model u using",
    "wut model u on",
    "wut model are you using",
    "what provider are you using",
    "which provider are you using",
    "what provider are you on",
    "what provider u on",
    "and provider",
    "what model are we using",
    "which model are we using",
    "current model",
)
_MODEL_POLICY_STATUS_PHRASES = (
    "what is my model selection policy",
    "what s my model selection policy",
    "what is the model selection policy",
    "what is my model policy",
    "model selection policy",
)
_MODEL_CONTROLLER_POLICY_PHRASES = (
    "what mode am i in",
    "what mode are you in",
    "which mode am i in",
    "which mode are you in",
    "what does this mode allow",
    "what does your mode allow",
    "why can t you switch that here",
    "why can't you switch that here",
    "why cant you switch that here",
    "why can t you install that here",
    "why can't you install that here",
    "why cant you install that here",
    "what would you need my approval for",
    "what do you need my approval for",
    "what requires my approval",
)
_MODEL_POLICY_CAP_PHRASES = (
    "what is my cheap remote cap",
    "what s my cheap remote cap",
    "what is the cheap remote cap",
    "cheap remote cap",
)
_MODEL_POLICY_CURRENT_REASON_PHRASES = (
    "why are you using this model",
    "why are we using this model",
    "why this model",
)
_MODEL_POLICY_SWITCH_CANDIDATE_PHRASES = (
    "what model would you switch to right now",
    "which model would you switch to right now",
    "what would you switch to right now",
)
_MODEL_POLICY_FREE_REMOTE_PHRASES = (
    "what free remote model would you choose",
    "which free remote model would you choose",
)
_MODEL_POLICY_CHEAP_REMOTE_PHRASES = (
    "what cheap remote model would you choose",
    "which cheap remote model would you choose",
)
_MODEL_SCOUT_RECOMMENDATION_CLOUD_ROLE_PHRASES = (
    "what cheap cloud model should i use",
    "what low cost cloud model should i use",
    "what budget cloud model should i use",
    "what cheap remote model should i use",
    "what low cost remote model should i use",
    "what budget remote model should i use",
)
_MODEL_SCOUT_RECOMMENDATION_PREMIUM_ROLE_PHRASES = (
    "what premium model should i use",
    "what premium model should i use for coding",
    "what premium model should i use for research",
    "what premium coding model should i use",
    "what premium research model should i use",
)
_PROVIDER_STATUS_PHRASES = (
    "is the openrouter setup",
    "what model do we have set up for openrouter",
    "what openrouter model is configured",
    "is openrouter configured",
    "are we using openrouter",
    "is the ollama setup",
    "what model do we have set up for ollama",
    "what ollama model is configured",
    "is ollama configured",
    "are we using ollama",
    "is the openai setup",
    "what model do we have set up for openai",
    "what openai model is configured",
    "is openai configured",
    "are we using openai",
)
_PROVIDERS_STATUS_PHRASES = (
    "what providers are configured",
    "which providers are configured",
    "what providers do we have configured",
    "which providers do we have configured",
    "what providers are set up",
    "which providers are set up",
    "what providers do we have set up",
    "which providers do we have set up",
    "what providers are setup",
    "which providers are setup",
    "what providers do we have setup",
    "which providers do we have setup",
    "what chat providers are configured",
    "what chat providers are set up",
    "which chat providers are configured",
    "which chat providers are set up",
)
_PROVIDER_STATUS_KEYWORDS = (
    "configured",
    "using",
    "model",
    "status",
    "provider",
    "ready",
    "set up",
    "setup",
    "health",
    "healthy",
    "working",
    "work",
)
_SET_DEFAULT_MODEL_PATTERNS = (
    "make this model the default",
    "make that model the default",
    "make this the default",
    "make that the default",
)
_EXPLICIT_MODEL_ACQUISITION_PREFIXES = (
    "install ",
    "download ",
    "pull ",
    "acquire ",
    "import ",
)
_EXPLICIT_MODEL_ACQUISITION_CONTEXT_PHRASES = (
    "install this model",
    "install that model",
    "install it",
    "install it first",
    "download this model",
    "download that model",
    "download it",
    "download it first",
    "pull this model",
    "pull that model",
    "pull it",
    "pull it first",
    "acquire this model",
    "acquire that model",
    "acquire it",
    "acquire it first",
    "import this model",
    "import that model",
    "import it",
    "import it first",
)
_DIRECT_MODEL_SWITCH_RE = re.compile(
    r"\b(?:switch(?:\s+chat)?\s+to|use|change(?:\s+chat)?\s+to)\s+"
    r"([a-z0-9][a-z0-9./_-]*:[a-z0-9][a-z0-9./_-]*)\b"
)
_EXPLICIT_MODEL_TARGET_RE = re.compile(
    r"\b(?:[a-z0-9._-]+:)?[a-z0-9][a-z0-9./_-]*:[a-z0-9][a-z0-9./_-]*\b"
)
_CONFIRM_TEXT = {"yes", "y", "ok", "okay", "confirm", "approve", "do it", "switch"}
_CANCEL_TEXT = {"no", "n", "cancel", "stop", "keep current", "not now"}
_KNOWN_PROVIDER_IDS = ("openrouter", "ollama", "openai")
_PRODUCT_GUARD_PHRASES = (
    "openrouter",
    "ollama",
    "provider",
    "default model",
    "runtime",
    "telegram",
)
_PRODUCT_GUARD_WORDS = ("configured", "ready", "agent", "model")
_GENERIC_MODEL_REQUEST_PHRASES = (
    "premium model",
    "use premium",
    "best model",
    "upgrade model",
    "stronger model",
)
_MODEL_DISCOVERY_HINT_PHRASES = (
    "hugging face",
    "huggingface",
    "smol model",
    "smol models",
    "small model",
    "small models",
    "tiny model",
    "tiny models",
    "lightweight model",
    "lightweight models",
    "find new models",
    "find new local models",
    "look for new models",
    "search for new models",
    "discover new models",
)

_CURRENT_MODEL_QUERY_HEADS = (
    "what model",
    "which model",
    "check what model",
    "check which model",
    "tell me what model",
)
_CURRENT_MODEL_QUERY_KEYWORDS = ("using", "enabled", "current", "currently", "active", "selected")
_RUNTIME_STATUS_KEYWORDS = ("healthy", "health", "ready", "attention", "status", "working", "work", "ok", "okay")
_KNOWN_ADAPTER_IDS = ("telegram",)
_OPERATIONAL_DOCTOR_PHRASES = (
    "agent doctor",
    "run doctor",
    "doctor",
    "check the agent",
)
_OPERATIONAL_STATUS_PHRASES = (
    "agent status",
    "bot status",
    "service status",
    "service health",
)
_OPERATIONAL_OBSERVE_PHRASES = (
    "what are my specs",
    "what are my pc specs",
    "system info",
    "hardware info",
    "computer info",
    "what cpu do i have",
    "what gpu do i have",
    "how much ram do i have",
    "how much disk do i have",
    "what is my cpu",
    "what is my gpu",
    "what is my ram",
    "what is my disk",
    "is something eating resources",
    "what is using resources",
    "what is eating memory",
    "what is eating cpu",
    "what is eating my memory",
    "what is eating my cpu",
    "what s eating memory",
    "what s eating cpu",
    "whats eating memory",
    "whats eating cpu",
)
_ASSISTANT_CAPABILITY_PHRASES = (
    "what can you do",
    "what can you do with the agent",
    "what can you help me with",
    "how you can help",
    "how can you help",
    "say what you do",
    "what you do in one sentence",
    "help me think this through",
    "help me think through something messy",
    "help me think through a messy task",
    "help me think through a messy idea",
    "help me think through a messy problem",
    "help me think through a problem",
    "help me think through an idea",
    "help me think this through without overcomplicating it",
    "help me think through this without overcomplicating it",
    "i need help thinking through something messy",
    "i need help thinking through a messy task",
    "i need help thinking through a messy idea",
    "i need help thinking through a messy problem",
    "what skills do you have",
    "what skills do you have access to",
    "what skill packs can you use",
    "what skill packs do you have",
    "what packs can you use",
    "what packs do you have",
    "what abilities do you have",
    "what abilities do you have access to",
    "what agentic abilities do you have",
    "what can you access",
    "what tools do you have",
    "what are you able to do",
    "what capabilities do you have",
    "what are you",
    "who are you",
    "what is the agent layer",
    "what does the agent layer do",
    "what is the assistant layer",
    "what does the assistant layer do",
    "what is the agent supposed to do",
    "what is the assistant supposed to do",
    "agent layer supposed to do",
    "assistant layer supposed to do",
    "how do you use the agent layer",
    "how do you work with the agent layer",
)
_MODEL_AVAILABILITY_PHRASES = (
    "are there others available to switch to easily",
    "what other models are available",
    "what other models are ready to switch to",
    "what models can you switch to",
    "what providers models can you use right now",
    "what providers and models can you use right now",
    "what can you switch to",
    "what models do you actually have available",
)
_MODEL_READY_NOW_PHRASES = (
    "what models are ready now",
    "which models are ready now",
    "what models are usable right now",
    "which models are usable right now",
)
_MODEL_LIFECYCLE_LIST_PHRASES = (
    "what models are downloading",
    "which models are downloading",
    "what installs are downloading",
    "which installs are downloading",
    "what model installs failed",
    "which model installs failed",
    "what installs failed",
    "which installs failed",
    "what models failed to install",
    "which models failed to install",
)
_AGENT_MEMORY_PHRASES = (
    "is memory on",
    "is your memory on",
    "is agent memory on",
    "is memory enabled",
    "is your memory enabled",
    "is agent memory enabled",
    "what is in your memory files",
    "what is currently in your memory files",
    "what is currently in your memory",
    "what do you remember",
    "what do you have saved about me",
    "what is in agent memory",
    "what is in the memory database",
    "show memory status",
    "memory status",
)
_MEMORY_THREAD_DISABLE_PHRASES = (
    "disable memory for this thread",
    "turn off memory for this thread",
    "stop using memory in this thread",
)
_MEMORY_THREAD_ENABLE_PHRASES = (
    "enable memory for this thread",
    "turn on memory for this thread",
    "use memory in this thread again",
)
_MEMORY_GLOBAL_DISABLE_PHRASES = (
    "disable memory globally",
    "turn off memory globally",
    "disable all memory",
)
_MEMORY_GLOBAL_ENABLE_PHRASES = (
    "enable memory globally",
    "turn on memory globally",
    "enable all memory",
)
_MEMORY_FORGET_TOPIC_RE = re.compile(r"\bforget (?:what you remember about|everything about|all memory about) .+")
_MEMORY_DELETE_ALL_PHRASES = (
    "delete all memory about me",
    "delete everything you remember about me",
    "forget everything about me",
)
_MEMORY_EXPORT_PHRASES = (
    "export my memory",
    "export memory",
    "download my memory",
)
_MEMORY_REDACT_PHRASES = (
    "redact sensitive memory",
    "redact my memory",
    "remove sensitive memory",
)
_MEMORY_CLEANUP_PHRASES = (
    "clean up duplicate memories",
    "cleanup duplicate memories",
    "dedupe memory",
    "deduplicate memory",
)
_AGENT_MEMORY_WORKING_CONTEXT_PHRASES = (
    "do you remember what we were doing",
    "what are we working on",
    "what were we working on",
    "what were we doing before",
    "what are we doing",
    "what are we doing right now",
    "go back and explain the larger task",
    "explain the larger task",
    "what is the larger task",
    "continue from here",
    "what would you do next",
    "what should we do next",
    "what should i do next",
    "go back to the day plan",
    "return to the day plan",
    "back to the day plan",
    "go back to the plan",
    "can you help with the thing we were doing before",
)
_AGENT_MEMORY_SYSTEM_CONTEXT_PHRASES = (
    "what do you know about my system",
    "what do you know about this system",
    "what do you know about my machine",
    "what do you know about this machine",
)
_AGENT_PREFERENCES_PHRASES = (
    "show my preferences",
    "show my current preferences",
)
_AGENT_OPEN_LOOPS_PHRASES = (
    "show my open loops",
    "show open loops",
    "what are my open loops",
)
_CRITICAL_INTENT_TYPO_MAP = {
    "waht": "what",
    "wat": "what",
    "wht": "what",
    "availble": "available",
    "avaiable": "available",
    "avaialble": "available",
    "modle": "model",
    "modles": "models",
    "stauts": "status",
    "provder": "provider",
    "heathy": "healthy",
    "runtim": "runtime",
    "doin": "doing",
    "bak": "back",
    "shoud": "should",
    "uding": "using",
    "useing": "using",
    "olama": "ollama",
    "ollma": "ollama",
}


def normalize_setup_text(text: str | None) -> str:
    lowered = str(text or "").strip().lower()
    cleaned = re.sub(r"[^a-z0-9:/._-]+", " ", lowered)
    tokens = [
        _CRITICAL_INTENT_TYPO_MAP.get(token, token)
        for token in cleaned.split()
    ]
    return " ".join(tokens)


def extract_openrouter_api_key(text: str | None) -> str | None:
    value = str(text or "").strip()
    if not value:
        return None
    match = _OPENROUTER_KEY_RE.search(value)
    if match is None:
        return None
    return str(match.group(1)).strip() or None


def _looks_like_current_model_query(normalized: str) -> bool:
    if any(phrase in normalized for phrase in _CURRENT_MODEL_PHRASES):
        return True
    if any(head in normalized for head in _CURRENT_MODEL_QUERY_HEADS) and any(
        token in normalized for token in _CURRENT_MODEL_QUERY_KEYWORDS
    ):
        return True
    if "provider" in normalized and any(token in normalized for token in ("using", "current", "active")):
        return True
    return False


def _looks_like_runtime_status_query(normalized: str) -> bool:
    normalized_space = normalized.replace("/", " ")
    words = {piece for piece in normalized_space.split(" ") if piece}
    if normalized in {"status", "runtime"}:
        return True
    if any(phrase in normalized for phrase in _RUNTIME_STATUS_PHRASES):
        return True
    if re.search(r"\bi said status\b", normalized):
        return True
    if any(
        phrase in normalized_space
        for phrase in (
            "what is happening",
            "whats happening",
            "what is going on",
            "whats going on",
            "give me a system report",
        )
    ):
        return True
    if "everything" in normalized and "agent" in normalized and any(
        token in normalized for token in ("work", "working", "ok", "okay")
    ):
        return True
    if "read the runtime" in normalized:
        return True
    if words & {"system", "runtime", "agent"} and words & {"status", "report", "health"}:
        return True
    if "agent" in normalized and any(token in normalized for token in _RUNTIME_STATUS_KEYWORDS):
        return True
    if "runtime" in normalized and any(token in normalized for token in _RUNTIME_STATUS_KEYWORDS):
        return True
    return False


def _looks_like_assistant_capabilities_query(normalized: str) -> bool:
    normalized_space = normalized.replace("/", " ")
    if any(phrase in normalized_space for phrase in _ASSISTANT_CAPABILITY_PHRASES):
        return True
    if (
        any(phrase in normalized_space for phrase in ("help me think this through", "help me think through", "thinking through"))
        and any(phrase in normalized_space for phrase in ("messy", "keep it simple", "without overcomplicating it"))
        and not any(token in normalized_space for token in ("about ", "whether ", "should i ", "should we ", "for ", "between "))
    ):
        return True
    if "skill pack" in normalized_space or "skill packs" in normalized_space:
        if any(
            token in normalized_space
            for token in ("what", "which", "use", "abilities", "skills", "tools", "capabilities", "access")
        ):
            return True
    if (
        "what" in normalized_space
        and "you" in normalized_space
        and any(token in normalized_space for token in ("skills", "abilities", "tools", "capabilities", "access"))
        and any(token in normalized_space for token in ("have", "help", "able", "access"))
    ):
        return True
    if (
        "what can you" in normalized_space
        and any(token in normalized_space for token in ("do", "help me with", "access"))
        and any(token in normalized_space for token in ("agent", "skills", "abilities", "tools", "capabilities"))
    ):
        return True
    return False


def _looks_like_model_availability_query(normalized: str) -> bool:
    normalized_space = normalized.replace("/", " ")
    if any(phrase in normalized_space for phrase in _MODEL_AVAILABILITY_PHRASES):
        return True
    if (
        any(token in normalized_space for token in ("cloud", "remote"))
        and any(token in normalized_space for token in ("model", "models"))
        and any(token in normalized_space for token in ("what", "which", "show", "show me", "list"))
    ):
        return True
    if (
        any(token in normalized_space for token in ("model", "models", "provider", "providers"))
        and any(token in normalized_space for token in ("available", "switch to", "use right now", "usable", "can use"))
        and any(token in normalized_space for token in ("what", "which", "other", "actually", "right now"))
    ):
        return True
    if "switch to" in normalized_space and any(token in normalized_space for token in ("what can you", "what could you", "what else")):
        return True
    return False


def _looks_like_model_ready_now_query(normalized: str) -> bool:
    normalized_space = normalized.replace("/", " ")
    if any(phrase in normalized_space for phrase in _MODEL_READY_NOW_PHRASES):
        return True
    return bool(
        any(token in normalized_space for token in ("model", "models"))
        and "ready" in normalized_space
        and "right now" in normalized_space
    )


def _looks_like_model_lifecycle_query(normalized: str) -> bool:
    normalized_space = normalized.replace("/", " ")
    if any(phrase in normalized_space for phrase in _MODEL_LIFECYCLE_LIST_PHRASES):
        return True
    has_explicit_target = _EXPLICIT_MODEL_TARGET_RE.search(str(normalized or "")) is not None
    if not has_explicit_target:
        return False
    lifecycle_terms = (
        "installed",
        "install",
        "installing",
        "download",
        "downloading",
        "status",
        "failed",
        "successfully",
    )
    query_heads = (
        "is ",
        "did ",
        "what is the status of",
        "what s the status of",
        "check status of",
        "show status of",
    )
    request_heads = (
        "can you install",
        "could you install",
        "please install",
        "can you download",
        "could you download",
        "please download",
        "can you import",
        "could you import",
        "please import",
        "can you pull",
        "could you pull",
        "please pull",
    )
    return bool(
        any(term in normalized_space for term in lifecycle_terms)
        and (
            any(head in normalized_space for head in query_heads)
            or any(head in normalized_space for head in request_heads)
        )
    )


def _looks_like_local_model_inventory_query(normalized: str) -> bool:
    normalized_space = normalized.replace("/", " ")
    if any(token in normalized_space for token in ("try", "better", "should use", "switch")):
        if "local" in normalized_space and "model" in normalized_space:
            return False
    if _EXPLICIT_MODEL_TARGET_RE.search(str(normalized or "")) is not None and any(
        token in normalized_space for token in ("installed", "install", "status", "download", "downloading", "failed")
    ):
        return False
    if any(phrase in normalized_space for phrase in _LOCAL_MODEL_INVENTORY_PHRASES):
        return True
    if (
        any(token in normalized_space for token in ("model", "models"))
        and any(token in normalized_space for token in ("local", "downloaded", "installed"))
        and (
            any(token in normalized_space for token in ("available", "downloaded", "installed", "ready", "existing"))
            or "do we have" in normalized_space
            or "any other" in normalized_space
        )
    ):
        return True
    if (
        any(token in normalized_space for token in ("local model", "local models"))
        and any(token in normalized_space for token in ("available", "downloaded", "installed"))
        and any(token in normalized_space for token in ("what", "which", "show", "show me", "list"))
    ):
        return True
    if "ollama" in normalized_space and any(token in normalized_space for token in ("downloaded", "installed", "local models")):
        return True
    return False


def _looks_like_local_model_recommendation_query(normalized: str) -> bool:
    normalized_space = normalized.replace("/", " ")
    if any(phrase in normalized_space for phrase in _LOCAL_MODEL_RECOMMENDATION_PHRASES):
        return True
    recommendation_hints = (
        "recommend",
        "do you recommend",
        "should i use",
        "should we use",
        "would you choose",
    )
    if (
        any(token in normalized_space for token in ("local model", "local models"))
        and any(token in normalized_space for token in recommendation_hints)
    ):
        return True
    return False


def _looks_like_local_provider_guidance_query(normalized: str) -> bool:
    normalized_space = str(normalized or "").replace("/", " ")
    if any(phrase in normalized_space for phrase in _LOCAL_PROVIDER_GUIDANCE_PHRASES):
        return True
    hardware_hint = any(
        token in normalized_space
        for token in ("rtx", "vram", "gb ram", "debian", "linux", "2060", "64gb", "64 gb")
    )
    provider_hint = any(
        token in normalized_space
        for token in ("provider", "ollama", "llama.cpp", "llama cpp", "lm studio", "vllm", "openai-compatible", "openai compatible")
    )
    model_setup_hint = any(
        phrase in normalized_space
        for phrase in ("should i use", "setup should i use", "do i have direct", "direct support", "local endpoint")
    )
    return bool(hardware_hint and provider_hint and model_setup_hint)


def _looks_like_setup_explanation_query(normalized: str) -> bool:
    normalized_space = normalized.replace("/", " ")
    if any(phrase in normalized_space for phrase in _SETUP_EXPLANATION_PHRASES):
        return True
    if (
        "setup" in normalized_space
        and any(token in normalized_space for token in ("check", "explain", "wrong", "diagnose", "problem"))
    ):
        return True
    if (
        any(phrase in normalized_space for phrase in ("what s wrong", "whats wrong", "explain what s wrong"))
        and any(token in normalized_space for token in ("setup", "ollama", "openrouter", "provider", "model", "chat"))
    ):
        return True
    return False


def _looks_like_direct_model_switch_request(normalized: str) -> bool:
    return _DIRECT_MODEL_SWITCH_RE.search(str(normalized or "")) is not None


def _looks_like_explicit_model_controller_request(normalized: str) -> bool:
    normalized_space = str(normalized or "").replace("/", " ")
    has_explicit_target = _EXPLICIT_MODEL_TARGET_RE.search(str(normalized or "")) is not None
    if not has_explicit_target:
        return False
    if "test" in normalized_space and any(
        token in normalized_space for token in ("without adopting", "without switching", "without using")
    ):
        return True
    if "temporar" in normalized_space and any(token in normalized_space for token in ("switch", "use", "try")):
        return True
    if normalized_space.startswith("make ") and " default" in normalized_space:
        return True
    return False


def _classify_agent_memory_route(normalized: str) -> dict[str, Any] | None:
    normalized_space = normalized.replace("/", " ")
    lifecycle_checks: tuple[tuple[tuple[str, ...], str], ...] = (
        (_MEMORY_THREAD_DISABLE_PHRASES, "memory_thread_disable_preview"),
        (_MEMORY_THREAD_ENABLE_PHRASES, "memory_thread_enable_preview"),
        (_MEMORY_GLOBAL_DISABLE_PHRASES, "memory_global_disable_preview"),
        (_MEMORY_GLOBAL_ENABLE_PHRASES, "memory_global_enable_preview"),
        (_MEMORY_DELETE_ALL_PHRASES, "memory_delete_all_preview"),
        (_MEMORY_EXPORT_PHRASES, "memory_export_preview"),
        (_MEMORY_REDACT_PHRASES, "memory_redact_preview"),
        (_MEMORY_CLEANUP_PHRASES, "memory_cleanup_preview"),
    )
    for phrases, kind in lifecycle_checks:
        if any(phrase in normalized_space for phrase in phrases):
            return {
                "route": "memory_lifecycle",
                "kind": kind,
                "generic_allowed": False,
                "fallback_reason": "memory_lifecycle",
            }
    if _MEMORY_FORGET_TOPIC_RE.search(normalized_space):
        return {
            "route": "memory_lifecycle",
            "kind": "memory_forget_topic_preview",
            "generic_allowed": False,
            "fallback_reason": "memory_lifecycle",
        }
    if any(phrase in normalized_space for phrase in _AGENT_PREFERENCES_PHRASES):
        return {
            "route": "agent_memory",
            "kind": "agent_memory_preferences",
            "generic_allowed": False,
            "fallback_reason": "agent_memory",
        }
    if any(phrase in normalized_space for phrase in _AGENT_OPEN_LOOPS_PHRASES):
        return {
            "route": "agent_memory",
            "kind": "agent_memory_open_loops",
            "generic_allowed": False,
            "fallback_reason": "agent_memory",
        }
    if any(phrase in normalized_space for phrase in _AGENT_MEMORY_PHRASES):
        return {
            "route": "agent_memory",
            "kind": "agent_memory_inspect",
            "generic_allowed": False,
            "fallback_reason": "agent_memory",
        }
    if any(phrase in normalized_space for phrase in _AGENT_MEMORY_WORKING_CONTEXT_PHRASES):
        return {
            "route": "agent_memory",
            "kind": "agent_memory_inspect",
            "generic_allowed": False,
            "fallback_reason": "agent_memory",
        }
    if any(phrase in normalized_space for phrase in _AGENT_MEMORY_SYSTEM_CONTEXT_PHRASES):
        return {
            "route": "agent_memory",
            "kind": "agent_memory_inspect",
            "generic_allowed": False,
            "fallback_reason": "agent_memory",
        }
    if (
        "memory" in normalized_space
        and not any(token in normalized_space for token in ("ram", "system memory", "using memory", "memory am i using"))
        and any(token in normalized_space for token in ("your memory", "agent memory", "memory files", "memory database", "remember", "saved about me"))
    ):
        return {
            "route": "agent_memory",
            "kind": "agent_memory_inspect",
            "generic_allowed": False,
            "fallback_reason": "agent_memory",
        }
    return None


def _classify_operational_route(text: str | None, normalized: str) -> dict[str, Any] | None:
    if any(phrase in normalized for phrase in _OPERATOR_HEALTH_PHRASES):
        return {
            "route": "operational_status",
            "kind": "operational_doctor",
            "generic_allowed": False,
            "fallback_reason": "operational_status",
        }
    if any(phrase in normalized for phrase in _OPERATIONAL_DOCTOR_PHRASES):
        return {
            "route": "operational_status",
            "kind": "operational_doctor",
            "generic_allowed": False,
            "fallback_reason": "operational_status",
        }
    if any(phrase in normalized for phrase in _OPERATIONAL_OBSERVE_PHRASES):
        return {
            "route": "operational_status",
            "kind": "operational_observe",
            "generic_allowed": False,
            "fallback_reason": "operational_status",
        }
    if any(phrase in normalized for phrase in _OPERATIONAL_STATUS_PHRASES):
        return {
            "route": "operational_status",
            "kind": "operational_agent_status",
            "generic_allowed": False,
            "fallback_reason": "operational_status",
        }
    nl_intent = classify_free_text(str(text or ""))
    if nl_intent in {"OBSERVE_PC", "EXPLAIN_PREVIOUS"}:
        return {
            "route": "operational_status",
            "kind": "operational_observe",
            "intent": nl_intent,
            "generic_allowed": False,
            "fallback_reason": "operational_status",
        }
    return None


def _classify_operator_lifecycle_route(normalized: str) -> dict[str, Any] | None:
    if any(phrase in normalized for phrase in ("show my backups", "list my backups", "show backups", "list backups")):
        return {
            "route": "operator_lifecycle",
            "kind": "operator_backup_list",
            "generic_allowed": False,
            "fallback_reason": "operator_lifecycle",
        }
    if any(phrase in normalized for phrase in ("validate this backup", "check this backup before restore", "inspect backup")):
        return {
            "route": "operator_lifecycle",
            "kind": "operator_restore_validate",
            "generic_allowed": False,
            "fallback_reason": "operator_lifecycle",
        }
    checks: tuple[tuple[tuple[str, ...], str], ...] = (
        (_OPERATOR_STORAGE_PHRASES, "operator_storage_status"),
        (_OPERATOR_REPAIR_PHRASES, "operator_repair_preview"),
        (_OPERATOR_BACKUP_PHRASES, "operator_backup_preview"),
        (_OPERATOR_RESTORE_PHRASES, "operator_restore_preview"),
        (_OPERATOR_UPDATE_PHRASES, "operator_update_preview"),
        (_OPERATOR_CLEANUP_PHRASES, "operator_cleanup_preview"),
        (_OPERATOR_UNINSTALL_PHRASES, "operator_uninstall_preview"),
        (_OPERATOR_SUPPORT_BUNDLE_PHRASES, "operator_support_bundle_preview"),
    )
    for phrases, kind in checks:
        if any(phrase in normalized for phrase in phrases):
            return {
                "route": "operator_lifecycle",
                "kind": kind,
                "generic_allowed": False,
                "fallback_reason": "operator_lifecycle",
            }
    return None


def _extract_governance_skill_id(normalized: str) -> str | None:
    patterns = (
        r"\bwhat execution mode does skill (?P<skill>[a-z0-9_-]+) use\b",
        r"\bwhat execution mode does (?P<skill>[a-z0-9_-]+) use\b",
        r"\bexecution mode for skill (?P<skill>[a-z0-9_-]+)\b",
        r"\bexecution mode for (?P<skill>[a-z0-9_-]+)\b",
        r"\bhow is skill (?P<skill>[a-z0-9_-]+) governed\b",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match is None:
            continue
        skill = str(match.group("skill") or "").strip().lower()
        if skill:
            return skill
    return None


def _extract_governance_execution_target(normalized: str) -> str | None:
    patterns = (
        r"\bwhat execution mode does (?P<target>[a-z0-9_-]+) use\b",
        r"\bwhich execution mode does (?P<target>[a-z0-9_-]+) use\b",
        r"\bwhat execution mode does skill (?P<target>[a-z0-9_-]+) use\b",
        r"\bwhich execution mode does skill (?P<target>[a-z0-9_-]+) use\b",
        r"\bexecution mode for (?P<target>[a-z0-9_-]+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match is None:
            continue
        target = str(match.group("target") or "").strip().lower()
        if target and target not in {"this", "skill"}:
            return target
    return None


def _extract_governance_adapter_id(normalized: str) -> str | None:
    for adapter_id in _KNOWN_ADAPTER_IDS:
        if adapter_id in normalized:
            return adapter_id
    return None


def _looks_like_governance_adapter_detail_query(normalized: str) -> bool:
    return (
        ("why does" in normalized and "exist" in normalized)
        or "what skill requested" in normalized
        or "why it exists" in normalized
    )


def _extract_model_policy_provider_id(normalized: str) -> str | None:
    if "why" not in normalized or "switch to" not in normalized:
        return None
    for provider_id in _KNOWN_PROVIDER_IDS:
        if f"switch to {provider_id}" in normalized:
            return provider_id
    return None


def _classify_model_policy_route(normalized: str) -> dict[str, Any] | None:
    if any(phrase in normalized for phrase in _MODEL_CONTROLLER_POLICY_PHRASES):
        return {
            "route": "model_policy_status",
            "kind": "model_controller_policy",
            "generic_allowed": False,
            "fallback_reason": "model_policy_status",
        }
    if any(phrase in normalized for phrase in _MODEL_POLICY_STATUS_PHRASES):
        return {
            "route": "model_policy_status",
            "kind": "model_policy_status",
            "generic_allowed": False,
            "fallback_reason": "model_policy_status",
        }
    if any(phrase in normalized for phrase in _MODEL_POLICY_CAP_PHRASES):
        return {
            "route": "model_policy_status",
            "kind": "model_policy_cap",
            "generic_allowed": False,
            "fallback_reason": "model_policy_status",
        }
    if any(phrase in normalized for phrase in _MODEL_POLICY_CURRENT_REASON_PHRASES):
        return {
            "route": "model_policy_status",
            "kind": "model_policy_current_choice",
            "generic_allowed": False,
            "fallback_reason": "model_policy_status",
        }
    if any(phrase in normalized for phrase in _MODEL_POLICY_SWITCH_CANDIDATE_PHRASES):
        return {
            "route": "model_policy_status",
            "kind": "model_policy_switch_candidate",
            "generic_allowed": False,
            "fallback_reason": "model_policy_status",
        }
    if any(phrase in normalized for phrase in _MODEL_POLICY_FREE_REMOTE_PHRASES):
        return {
            "route": "model_policy_status",
            "kind": "model_policy_tier_candidate",
            "tier": "free_remote",
            "generic_allowed": False,
            "fallback_reason": "model_policy_status",
        }
    if any(phrase in normalized for phrase in _MODEL_POLICY_CHEAP_REMOTE_PHRASES):
        return {
            "route": "model_policy_status",
            "kind": "model_policy_tier_candidate",
            "tier": "cheap_remote",
            "generic_allowed": False,
            "fallback_reason": "model_policy_status",
        }
    policy_provider_id = _extract_model_policy_provider_id(normalized)
    if policy_provider_id:
        return {
            "route": "model_policy_status",
            "kind": "model_policy_provider_explanation",
            "provider_id": policy_provider_id,
            "generic_allowed": False,
            "fallback_reason": "model_policy_status",
        }
    return None


def _looks_like_explicit_model_acquisition_request(normalized: str) -> bool:
    working = str(normalized or "").strip()
    if not working:
        return False
    if working.startswith("please "):
        working = working[len("please "):].strip()
    if any(working == phrase or working.startswith(f"{phrase} ") for phrase in _EXPLICIT_MODEL_ACQUISITION_CONTEXT_PHRASES):
        return True
    if not any(working.startswith(prefix) for prefix in _EXPLICIT_MODEL_ACQUISITION_PREFIXES):
        return False
    if _EXPLICIT_MODEL_TARGET_RE.search(working):
        return True
    return any(phrase in working for phrase in _EXPLICIT_MODEL_ACQUISITION_CONTEXT_PHRASES)


def _looks_like_model_scout_recommendation_query(normalized: str) -> bool:
    working = str(normalized or "").strip().replace("-", " ")
    if not working:
        return False
    if any(phrase in working for phrase in _MODEL_SCOUT_RECOMMENDATION_CLOUD_ROLE_PHRASES):
        return True
    if any(phrase in working for phrase in _MODEL_SCOUT_RECOMMENDATION_PREMIUM_ROLE_PHRASES):
        return True
    if any(phrase in working for phrase in _GENERIC_MODEL_REQUEST_PHRASES):
        return True
    if re.search(r"\bmodels?\b", working) and any(
        token in working for token in ("tiny", "small", "lightweight", "local", "ollama", "fast")
    ) and any(
        token in working for token in ("coding", "code", "coder", "vision", "image", "chat", "reasoning", "research")
    ):
        return True
    if re.search(r"\bmodels?\b", working) and any(
        token in working for token in ("newer than", "more recent than", "better than", "latest", "newest", "recent")
    ) and any(token in working for token in ("chat", "coding", "code", "vision", "reasoning", "local")):
        return True
    recommendation_hints = (
        "should i use",
        "should we use",
        "would you use",
        "would you choose",
        "do you recommend",
        "recommend",
    )
    cloud_role_hints = (
        "cheap cloud",
        "low cost cloud",
        "budget cloud",
        "cheap remote",
        "low cost remote",
        "budget remote",
        "premium model",
        "premium coding model",
        "premium research model",
    )
    return bool(
        re.search(r"\bmodels?\b", working)
        and any(phrase in working for phrase in recommendation_hints)
        and (
            any(phrase in working for phrase in cloud_role_hints)
            or any(token in working for token in ("coding", "code", "debug", "refactor", "review"))
            or any(token in working for token in ("research", "reasoning", "analysis", "analyze"))
        )
    )


def _looks_like_model_scout_discovery_query(normalized: str) -> bool:
    working = str(normalized or "").strip().replace("-", " ")
    if not working:
        return False
    if any(phrase in working for phrase in _MODEL_DISCOVERY_HINT_PHRASES):
        return True
    if any(phrase in working for phrase in ("newer than", "more recent than", "newest than", "better than", "latest than")) and any(
        token in working for token in ("chat", "coding", "code", "vision", "reasoning", "local", "model", "models", "qwen", "gemma", "llama")
    ):
        return True
    if ("hugging face" in working or "huggingface" in working) and re.search(r"\bmodels?\b", working):
        return True
    if any(token in working for token in ("smol", "small", "tiny", "lightweight")) and re.search(
        r"\bmodels?\b",
        working,
    ):
        return True
    return bool(
        any(token in working for token in ("find", "look", "search", "discover", "show"))
        and any(phrase in working for phrase in ("new model", "new models", "models to download", "promising models"))
    )


def _looks_like_safe_web_search_request(normalized: str) -> bool:
    working = " ".join(str(normalized or "").strip().lower().split())
    if not working:
        return False
    if _contains_safe_web_search_suppression(working):
        return False
    explicit_prefixes = (
        "search the web for ",
        "search web for ",
        "web search for ",
        "search online for ",
        "look up online ",
        "look this up online ",
        "look that up online ",
        "find online ",
    )
    if any(working.startswith(prefix) and len(working) > len(prefix) for prefix in explicit_prefixes):
        return True
    if working.startswith("look up ") and " online" in working:
        return True
    explicit_lookup = re.match(r"^(?:can you|could you|please)?\s*look up\s+(.+)$", working)
    if explicit_lookup is not None:
        lookup_target = explicit_lookup.group(1).strip()
        if re.search(r"\b[a-z0-9][a-z0-9_.-]*\.[a-z0-9_.-]+\b", lookup_target):
            return True
        if re.search(r"\b(model|tool|library|project|package|repo|company|site|website|channel|creator)\b", lookup_target):
            return True
    if working.startswith("find ") and (" on the web" in working or " online" in working):
        return True
    if re.search(r"\b(?:tell me about|what is|who is|who are|summarize|explain)\b", working) and re.search(
        r"\b(?:youtube channel|youtube creator|youtube account|youtube video|twitch channel|twitter account|x account|instagram account|tiktok account|website|online)\b",
        working,
    ):
        return True
    return bool(
        re.search(r"\b(search|look up|find)\b", working)
        and re.search(r"\b(web|internet|online)\b", working)
        and len(working.split()) >= 4
    )


def _contains_safe_web_search_suppression(working: str) -> bool:
    return any(
        phrase in working
        for phrase in (
            "do not search",
            "don't search",
            "don t search",
            "dont search",
            "without searching",
            "without web search",
            "no web search",
            "no internet search",
            "don't look it up",
            "don t look it up",
            "dont look it up",
        )
    )


def _looks_like_provided_text_transform(working: str) -> bool:
    if re.search(r"\bmake\s+this\s+sound\s+(?:nicer|better|more)\b", working):
        return True
    if re.search(r"\b(rewrite|reword|edit|proofread|translate|summarize|condense|shorten)\s+(this|the following|my)\b", working):
        return True
    return bool(re.search(r"\b(rewrite|reword|edit|proofread|translate|summarize|condense|shorten)\b", working) and ":" in working)


def _looks_like_simple_timeless_explanation(working: str) -> bool:
    timeless = (
        "why the sky is blue",
        "why is the sky blue",
        "what is photosynthesis",
        "what is gravity",
        "what is a hyperlink",
        "what is hyperlink",
        "explain gravity",
        "explain recursion",
    )
    return any(phrase in working for phrase in timeless)


def _looks_like_safe_web_search_fallback_request(working: str) -> bool:
    if _looks_like_provided_text_transform(working) or _looks_like_simple_timeless_explanation(working):
        return False
    if re.search(
        r"\b(local api|web ui|runtime|current task|this task|my task|this repo|local repo|repo root|workspace root|this project|my project|my system|my machine|this machine|controlled mode|control mode)\b",
        working,
    ):
        return False
    if re.search(r"\b(next step|next steps|what should we do next|keep making progress)\b", working):
        return False
    if re.search(r"\bwhat is (?:using|eating) (?:resources|memory|cpu|ram)\b", working):
        return False
    if re.search(r"\b(still active|active anymore|latest|currently|recent|newest|release|released|changelog|price|pricing|compatibility|compatible|recommendations?)\b", working):
        return True
    if re.search(r"\bstill around\b", working):
        return True
    if re.search(r"\b(current|latest|public)\s+status\b|\bstatus\s+(?:of|for)\s+\S", working):
        return True
    if re.search(r"\b(docs|documentation)\s+(?:for|of)\s+\S", working) or re.search(r"\S\s+(?:docs|documentation)\b", working):
        return True
    if re.fullmatch(r"[a-z0-9][a-z0-9_.-]*\.[a-z0-9_.-]+\??", working.strip()):
        return True
    if re.search(r"\b[a-z0-9][a-z0-9_-]{1,40}(?:\s+|-)(?:tts|agi|llm|mcp|ai|dev)\b", working) and (
        "?" in working
        or re.search(r"\b(any good|useful|worth|should|model|tool|library|project|people are talking about)\b", working)
    ):
        return True
    if re.search(r"\b(?:tts|agi|llm|mcp|ai|dev)\b", working) and 2 <= len(working.split()) <= 6:
        return True
    if re.search(
        r"\b(company|startup|project|library|package|tool|model|api|site|website|repo|github|channel|creator|streamer|plugin|sdk|app|service)\b",
        working,
    ) and re.search(r"\b(what is|who is|who are|tell me about|should i use|useful|worth|does\s+\S|is\s+\S)\b", working):
        return True
    if re.search(r"\b(?:what is|tell me about|is|does|can|should i use)\s+[a-z0-9][a-z0-9_.-]*\.[a-z0-9_.-]+\b", working):
        return True
    if re.search(r"\b(?:what is|tell me about)\s+[a-z0-9][a-z0-9_.-]*(?:tts|llm|mcp|ai|dev|js|py|rs|gguf)\b", working):
        return True
    if re.fullmatch(r"[a-z][a-z0-9_.-]{2,40}\?", working.strip()):
        return True
    if re.search(r"\b(?:what is|who is|tell me about)\s+[a-z][a-z0-9_.-]{2,40}\s+[a-z][a-z0-9_.-]{2,40}\b", working):
        return True
    return False


def _looks_like_bare_public_lookup_question(raw_text: str | None, normalized: str) -> bool:
    raw = str(raw_text or "").strip()
    working = " ".join(str(normalized or "").strip().lower().split())
    if not raw.endswith("?") or not re.fullmatch(r"[a-z][a-z0-9_.-]{3,40}", working):
        return False
    common_direct = {
        "hello",
        "thanks",
        "what",
        "status",
        "runtime",
        "recursion",
        "photosynthesis",
        "hyperlink",
        "wat",
    }
    return working not in common_direct


def _looks_like_raw_niche_public_lookup_question(raw_text: str | None, normalized: str) -> bool:
    raw = str(raw_text or "").strip()
    working = " ".join(str(normalized or "").strip().lower().split())
    if not raw.endswith("?"):
        return False
    words = working.split()
    return bool(2 <= len(words) <= 6 and re.search(r"\b(?:tts|agi|llm|mcp|ai|dev)\b", working))


def _looks_like_safe_web_search_clarification_request(normalized: str) -> bool:
    working = " ".join(str(normalized or "").strip().lower().split())
    if not working or _contains_safe_web_search_suppression(working):
        return False
    bare_referents = {
        "what is it",
        "what is it?",
        "what's it",
        "whats it",
        "what is that",
        "what is that?",
        "what's that",
        "whats that",
        "that thing",
        "the thing",
        "the tts thing",
        "what thing",
    }
    if working in bare_referents:
        return True
    return bool(
        re.search(r"\b(?:that|the)\s+(?:tts|llm|agi|ai|mcp)?\s*thing\b", working)
        and re.search(r"\b(?:people are talking about|everyone is talking about|trending|going around|you mentioned)\b", working)
    )


def _looks_like_safe_web_search_suppressed_request(normalized: str) -> bool:
    working = " ".join(str(normalized or "").strip().lower().split())
    if not working or not _contains_safe_web_search_suppression(working):
        return False
    stripped = re.sub(
        r"\b(?:do not search|don't search|don t search|dont search|without searching|without web search|no web search|no internet search|don't look it up|don t look it up|dont look it up)\b",
        "",
        working,
    )
    stripped = " ".join(stripped.split())
    if _looks_like_safe_web_search_fallback_request(stripped):
        return True
    return bool(re.search(r"\b(?:what do you know about|what is|who is|tell me about)\b", stripped))


def _looks_like_safe_web_search_status_request(normalized: str) -> bool:
    working = " ".join(str(normalized or "").strip().lower().split())
    if not working:
        return False
    if "searxng" in working and any(phrase in working for phrase in ("set up", "setup", "configure", "status", "enabled")):
        return True
    if re.search(r"\b(?:127\.0\.0\.1|localhost):(?:8080|8888)\b", working) and re.search(
        r"\b(?:cannot be reached|can't be reached|cant be reached|connection refused|refused connection|unreachable|not reachable|stopped working)\b",
        working,
    ):
        return True
    explicit = (
        "is web search enabled",
        "is search working",
        "is web search working",
        "is internet search working",
        "is search configured",
        "how do i set up web search",
        "how do i setup web search",
        "how do i configure web search",
        "set up web search",
        "setup web search",
        "enable web search",
        "check web search",
        "how do i set up searxng",
        "how do i setup searxng",
        "why can't you search the internet",
        "why cant you search the internet",
        "why can t you search the internet",
        "can you search online",
        "what is your search status",
        "what's your search status",
        "web search status",
        "search status",
        "i meant search",
        "no i meant search",
        "no, i meant search",
    )
    if any(phrase in working for phrase in explicit):
        return True
    return bool(
        re.search(r"\b(web search|search|internet search)\b", working)
        and any(token in working for token in ("enable", "enabled", "configured", "setup", "status", "unavailable", "disabled", "check"))
    )


def _model_inventory_scope(normalized: str) -> str | None:
    normalized_space = str(normalized or "").replace("/", " ")
    if (
        any(token in normalized_space for token in ("cloud", "remote"))
        and any(token in normalized_space for token in ("model", "models"))
    ):
        return "remote"
    if (
        any(token in normalized_space for token in ("local", "downloaded", "installed", "ollama"))
        and any(token in normalized_space for token in ("model", "models"))
    ):
        return "local"
    return None


def _looks_like_model_switch_advisory_query(normalized: str) -> bool:
    normalized_space = str(normalized or "").replace("/", " ")
    if re.search(r"\bshould (?:i|we) switch models?\b", normalized_space):
        return True
    return bool(
        any(token in normalized_space for token in ("switch", "change"))
        and any(token in normalized_space for token in ("model", "models"))
        and any(token in normalized_space for token in ("should", "would", "recommend"))
    )


def _clean_filesystem_path_hint(raw_path: str | None) -> str | None:
    candidate = str(raw_path or "").strip()
    if not candidate:
        return None
    candidate = candidate.rstrip(".,!?;:")
    candidate = candidate.strip()
    return candidate or None


def _clean_filesystem_search_query(raw_query: str | None) -> str | None:
    candidate = str(raw_query or "").strip()
    if not candidate:
        return None
    candidate = candidate.strip("'\"` ")
    candidate = candidate.rstrip(".,!?;:")
    candidate = candidate.strip()
    return candidate or None


def _extract_filesystem_path_hint(text: str | None, normalized: str) -> str | None:
    raw_text = str(text or "").strip()
    normalized_space = str(normalized or "").replace("/", " ")
    if any(phrase in normalized_space for phrase in _FILESYSTEM_CURRENT_DIRECTORY_PHRASES):
        return "."
    for match in _FILESYSTEM_QUOTED_PATH_RE.finditer(raw_text):
        candidate = _clean_filesystem_path_hint(match.group("path"))
        if not candidate:
            continue
        if candidate.startswith(("~", "/", "./", "../")) or "/" in candidate or "\\" in candidate:
            return candidate
        if _FILESYSTEM_BARE_FILENAME_RE.fullmatch(candidate):
            return candidate
    token_match = _FILESYSTEM_PATH_TOKEN_RE.search(raw_text)
    if token_match is not None:
        return _clean_filesystem_path_hint(token_match.group("path"))
    if any(token in normalized_space for token in ("file", "text file", "path", "read ", "open ", "how big", "stat ")):
        bare_match = _FILESYSTEM_BARE_FILENAME_RE.search(raw_text)
        if bare_match is not None:
            return _clean_filesystem_path_hint(bare_match.group("path"))
    return None


def _extract_filesystem_search_query(text: str | None, normalized: str) -> tuple[str | None, str | None]:
    raw_text = str(text or "").strip()
    normalized_space = str(normalized or "").replace("/", " ")
    filename_patterns = (
        r"\bfind files? named (?P<query>.+)$",
        r"\bfind directories? named (?P<query>.+)$",
        r"\bfind folders? named (?P<query>.+)$",
        r"\bsearch for (?P<query>.+?) in (?:this repo|this folder|this directory|the repo|the folder|the directory|workspace root|repo root|~[^\s]+|/[^\s]+|\./[^\s]+|\.\./[^\s]+)$",
        r"\bfind (?P<query>.+?) in (?:this repo|this folder|this directory|the repo|the folder|the directory|workspace root|repo root|~[^\s]+|/[^\s]+|\./[^\s]+|\.\./[^\s]+)$",
    )
    for pattern in filename_patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match is None:
            continue
        query = _clean_filesystem_search_query(match.group("query"))
        if query:
            return "search_filenames", query
    text_patterns = (
        r"\bsearch (?:this repo|this folder|this directory|the repo|the folder|the directory|workspace root|repo root|~[^\s]+|/[^\s]+|\./[^\s]+|\.\./[^\s]+) for (?P<query>.+)$",
        r"\bfind .+ mentioning (?P<query>.+)$",
        r"\bfind .+ containing (?P<query>.+)$",
    )
    for pattern in text_patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match is None:
            continue
        query = _clean_filesystem_search_query(match.group("query"))
        if query:
            return "search_text", query
    if (
        normalized_space.startswith("search for ")
        and any(phrase in normalized_space for phrase in _FILESYSTEM_CURRENT_DIRECTORY_PHRASES)
    ):
        query = _clean_filesystem_search_query(raw_text[11:])
        if query:
            for phrase in _FILESYSTEM_CURRENT_DIRECTORY_PHRASES:
                if phrase in query.lower():
                    query = _clean_filesystem_search_query(query[: query.lower().rfind(phrase)])
                    break
        if query:
            return "search_filenames", query
    return None, None


def _classify_filesystem_route(text: str | None, normalized: str) -> dict[str, Any] | None:
    normalized_space = str(normalized or "").replace("/", " ")
    path_hint = _extract_filesystem_path_hint(text, normalized)
    search_kind, search_query = _extract_filesystem_search_query(text, normalized)
    if search_kind and search_query:
        normalized_root_hint = path_hint or "."
        if path_hint and search_query and str(path_hint).strip().lower() == str(search_query).strip().lower():
            normalized_root_hint = "."
        return {
            "route": "action_tool",
            "kind": "filesystem_search_text" if search_kind == "search_text" else "filesystem_search_filenames",
            "path_hint": normalized_root_hint,
            "query": search_query,
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }
    has_path_language = bool(path_hint) or any(
        token in normalized_space
        for token in (" file", " files", " folder", " directory", " path", " text file")
    )
    if not has_path_language:
        return None
    if any(phrase in normalized_space for phrase in _FILESYSTEM_LIST_PHRASES):
        return {
            "route": "action_tool",
            "kind": "filesystem_list_directory",
            "path_hint": path_hint,
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }
    if any(phrase in normalized_space for phrase in _FILESYSTEM_READ_PHRASES):
        return {
            "route": "action_tool",
            "kind": "filesystem_read_text_file",
            "path_hint": path_hint,
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }
    if any(phrase in normalized_space for phrase in _FILESYSTEM_STAT_PHRASES):
        return {
            "route": "action_tool",
            "kind": "filesystem_stat_path",
            "path_hint": path_hint,
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }
    if path_hint and any(token in normalized_space for token in ("read ", "open ", "show file", "show text file")):
        return {
            "route": "action_tool",
            "kind": "filesystem_read_text_file",
            "path_hint": path_hint,
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }
    if path_hint and any(token in normalized_space for token in ("how big", "stat ", "file info", "path info", "what is")):
        return {
            "route": "action_tool",
            "kind": "filesystem_stat_path",
            "path_hint": path_hint,
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }
    return None


def _extract_shell_create_directory_path(text: str | None, normalized: str) -> str | None:
    raw_text = str(text or "").strip()
    if not raw_text:
        return None
    match = re.search(
        r"\bcreate (?:a |the )?(?:folder|directory)(?: called| named)? (?P<path>[^,!?]+)",
        raw_text,
        re.IGNORECASE,
    )
    if match is None:
        return None
    candidate = str(match.group("path") or "").strip()
    for suffix in (
        " in this repo",
        " in this folder",
        " in this directory",
        " in the repo",
        " in the folder",
        " in the directory",
        " in workspace root",
        " in repo root",
    ):
        if candidate.lower().endswith(suffix):
            candidate = candidate[: -len(suffix)].strip()
            break
    candidate = candidate.strip("'\"` ").rstrip(".,!?;:").strip()
    if not candidate:
        return None
    token_match = _SHELL_INLINE_PATH_TOKEN_RE.fullmatch(candidate)
    if token_match is None:
        return None
    return str(token_match.group("path") or "").strip() or None


def _classify_shell_route(text: str | None, normalized: str) -> dict[str, Any] | None:
    raw_text = str(text or "").strip()
    normalized_space = str(normalized or "").replace("/", " ")
    if not raw_text:
        return None

    if any(token in raw_text for token in ("&&", "||", ";", "|", "$(")):
        return {
            "route": "action_tool",
            "kind": "shell_blocked_request",
            "blocked_reason": "shell_interpolation_blocked",
            "request_text": raw_text,
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }

    if re.search(r"^\s*(?:run|execute|bash|sh)\b", raw_text, re.IGNORECASE):
        return {
            "route": "action_tool",
            "kind": "shell_blocked_request",
            "blocked_reason": "unsupported_command",
            "request_text": raw_text,
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }

    if re.search(r"^\s*(?:rm|sudo|chmod|chown|curl|wget|systemctl|podman|docker)\b", raw_text, re.IGNORECASE):
        return {
            "route": "action_tool",
            "kind": "shell_blocked_request",
            "blocked_reason": "destructive_operation_blocked",
            "request_text": raw_text,
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }

    if re.search(r"\b(?:podman|docker|systemctl)\b", normalized_space) and re.search(
        r"\b(?:run|start|restart|stop|rm|remove|exec|compose|pull|build|ps)\b",
        normalized_space,
    ):
        return {
            "route": "action_tool",
            "kind": "shell_blocked_request",
            "blocked_reason": "unsupported_container_or_service_command",
            "request_text": raw_text,
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }

    if re.search(r"\b(?:delete|remove|uninstall|move)\b", normalized_space) and any(
        token in normalized_space for token in ("file", "folder", "directory", "package", "path")
    ):
        return {
            "route": "action_tool",
            "kind": "shell_blocked_request",
            "blocked_reason": "operation_not_supported",
            "request_text": raw_text,
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }

    if re.search(r"\b(?:what version of python(?: do i have)?|python version|python --version)\b", normalized_space):
        return {
            "route": "action_tool",
            "kind": "shell_safe_command",
            "command_name": "python_version",
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }
    if re.search(r"\b(?:what version of pip(?: do i have)?|pip version|pip --version)\b", normalized_space):
        return {
            "route": "action_tool",
            "kind": "shell_safe_command",
            "command_name": "pip_version",
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }
    if re.search(r"\b(?:where is pip(?: installed)?|which pip)\b", normalized_space):
        return {
            "route": "action_tool",
            "kind": "shell_safe_command",
            "command_name": "which",
            "subject": "pip",
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }
    if re.search(r"\b(?:list installed ollama models|ollama list|list ollama models|show ollama models)\b", normalized_space):
        return {
            "route": "action_tool",
            "kind": "shell_safe_command",
            "command_name": "ollama_list",
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }
    if re.search(r"\bollama ps\b", normalized_space):
        return {
            "route": "action_tool",
            "kind": "shell_safe_command",
            "command_name": "ollama_ps",
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }

    apt_search_match = re.search(
        r"\b(?:search apt for|apt search|apt-cache search)\s+(?P<query>[A-Za-z0-9][A-Za-z0-9+._-]{0,127})\b",
        raw_text,
        re.IGNORECASE,
    )
    if apt_search_match is not None:
        return {
            "route": "action_tool",
            "kind": "shell_safe_command",
            "command_name": "apt_search",
            "query": str(apt_search_match.group("query") or "").strip() or None,
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }

    apt_policy_match = re.search(
        r"\b(?:apt(?:-cache)? policy|show apt policy for)\s+(?P<package>[A-Za-z0-9][A-Za-z0-9+._-]{0,127})\b",
        raw_text,
        re.IGNORECASE,
    )
    if apt_policy_match is not None:
        return {
            "route": "action_tool",
            "kind": "shell_safe_command",
            "command_name": "apt_cache_policy",
            "subject": str(apt_policy_match.group("package") or "").strip() or None,
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }

    install_match = re.search(
        r"\b(?:(?:sudo\s+)?apt(?:-get)?\s+install|(?:can|could|would)\s+you\s+install|please\s+install|install\s+(?:a\s+)?(?:linux\s+|debian\s+|system\s+)?package|install\s+(?:a\s+)?(?:python\s+|pip\s+)?package|pip\s+install|install\s+(?=[A-Za-z0-9][A-Za-z0-9+._-]{0,127}(?:\s|$|[?.!,])))\s+(?:-y\s+)?(?P<package>[A-Za-z0-9][A-Za-z0-9+._-]{0,127})\b",
        raw_text,
        re.IGNORECASE,
    )
    if install_match is None:
        install_match = re.search(
            r"^\s*install\s+(?:-y\s+)?(?P<package>[A-Za-z0-9][A-Za-z0-9+._-]{0,127})\s*$",
            raw_text,
            re.IGNORECASE,
        )
    if install_match is None and re.search(r"\binstall\b", normalized_space):
        install_match = re.search(
            r"\binstall\s+(?:-y\s+)?(?P<package>[A-Za-z0-9][A-Za-z0-9+._-]{0,127})\b",
            raw_text,
            re.IGNORECASE,
        )
    if install_match is not None and not _looks_like_agent_pack_language(normalized_space):
        package = str(install_match.group("package") or "").strip() or None
        manager = "pip" if re.search(r"\b(?:pip\s+install|python\s+package|pip\s+package)\b", normalized_space) else "apt"
        return {
            "route": "action_tool",
            "kind": "shell_install_package",
            "manager": manager,
            "package": package,
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }

    create_directory_path = _extract_shell_create_directory_path(text, normalized)
    if create_directory_path:
        return {
            "route": "action_tool",
            "kind": "shell_create_directory",
            "path_hint": create_directory_path,
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }

    return None


def _looks_like_agent_pack_language(normalized: str | None) -> bool:
    text = normalize_setup_text(normalized)
    if not text:
        return False
    if re.search(r"\b(?:install|add|get|find)\b.{0,80}\b(?:skill|skills|guidance pack|guidance packs)\b", text):
        return True
    if re.search(r"\b(?:install|add|get|find|show|list|what)\b.{0,100}\b(?:capability|capabilities|pack|packs)\b", text):
        return True
    if any(term in text for term in ("browser capability", "browser capabilities", "web research", "web browsing", "reading webpages", "reading web pages")):
        return True
    phrases = (
        "skill pack",
        "skill packs",
        "guidance pack",
        "guidance packs",
        "capability pack",
        "capability packs",
        "agent skill",
        "agent skills",
        "whatever skill",
        "portable text skill",
        "portable text skills",
        "mcp server",
        "mcp adapter",
        "plugin pack",
        "native code pack",
    )
    return any(phrase in text for phrase in phrases)


def classify_setup_intent(
    text: str | None,
    *,
    awaiting_secret: bool = False,
    awaiting_confirmation: bool = False,
) -> dict[str, Any]:
    normalized = normalize_setup_text(text)
    if not normalized:
        return {"kind": "none"}

    if awaiting_secret:
        api_key = extract_openrouter_api_key(text)
        if api_key:
            return {"kind": "provide_openrouter_key", "api_key": api_key}

    if awaiting_confirmation:
        if normalized in _CONFIRM_TEXT:
            return {"kind": "confirm_pending_setup"}
        if normalized in _CANCEL_TEXT:
            return {"kind": "cancel_pending_setup"}

    if any(phrase in normalized for phrase in _PROVIDERS_STATUS_PHRASES):
        return {"kind": "providers_status"}
    if any(phrase in normalized for phrase in _GOVERNANCE_ADAPTERS_PHRASES):
        return {"kind": "governance_adapters"}
    if any(phrase in normalized for phrase in _GOVERNANCE_BACKGROUND_TASKS_PHRASES):
        return {"kind": "governance_background_tasks"}
    if any(phrase in normalized for phrase in _GOVERNANCE_BLOCKS_PHRASES):
        return {"kind": "governance_blocks"}
    if any(phrase in normalized for phrase in _GOVERNANCE_PENDING_PHRASES):
        return {"kind": "governance_pending"}
    if any(phrase in normalized for phrase in _GOVERNANCE_OVERVIEW_PHRASES):
        return {"kind": "governance_overview"}
    if "what execution mode does this skill use" in normalized:
        return {
            "kind": "governance_skill_status",
            "skill_id": None,
        }
    execution_target = _extract_governance_execution_target(normalized)
    if execution_target:
        return {
            "kind": "model_controller_policy",
            "target_id": execution_target,
        }
    if "execution mode" in normalized and "skill" in normalized:
        return {
            "kind": "model_controller_policy",
            "skill_id": _extract_governance_skill_id(normalized),
        }
    adapter_id = _extract_governance_adapter_id(normalized)
    if adapter_id and _looks_like_governance_adapter_detail_query(normalized):
        return {"kind": "governance_adapter_detail", "adapter_id": adapter_id}
    if any(phrase in normalized for phrase in _PROVIDER_STATUS_PHRASES):
        for provider_id in _KNOWN_PROVIDER_IDS:
            if provider_id in normalized:
                return {"kind": "provider_status", "provider_id": provider_id}
    if _looks_like_setup_explanation_query(normalized):
        return {"kind": "setup_explanation"}
    if _looks_like_direct_model_switch_request(normalized):
        return {"kind": "set_default_model"}
    if _looks_like_local_model_inventory_query(normalized):
        return {
            "kind": "local_model_inventory",
            "provider_id": "ollama" if "ollama" in normalized else None,
        }
    if _looks_like_local_provider_guidance_query(normalized):
        return {"kind": "local_provider_guidance"}
    if _looks_like_explicit_model_acquisition_request(normalized):
        return {"kind": "model_acquisition_request"}
    if _looks_like_model_lifecycle_query(normalized):
        return {"kind": "model_lifecycle_status"}
    explicit_controller_request = _looks_like_explicit_model_controller_request(normalized)
    if "ollama" in normalized and not explicit_controller_request:
        if any(phrase in normalized for phrase in _OLLAMA_SETUP_PHRASES):
            return {"kind": "configure_ollama", "make_default": False}
        if any(phrase in normalized for phrase in _OLLAMA_USE_PHRASES):
            return {"kind": "configure_ollama", "make_default": True}
        if any(token in normalized for token in _OLLAMA_SETUP_KEYWORDS):
            return {"kind": "configure_ollama", "make_default": False}
        if any(token in normalized for token in _OLLAMA_SWITCH_KEYWORDS):
            return {"kind": "configure_ollama", "make_default": True}
    if "openrouter" in normalized and not explicit_controller_request:
        if any(phrase in normalized for phrase in _OPENROUTER_SETUP_PHRASES):
            return {"kind": "configure_openrouter", "make_default": False}
        if any(phrase in normalized for phrase in _OPENROUTER_USE_PHRASES):
            return {"kind": "configure_openrouter", "make_default": True}
        if any(token in normalized for token in _OPENROUTER_SETUP_KEYWORDS):
            return {"kind": "configure_openrouter", "make_default": False}
        if any(token in normalized for token in _OPENROUTER_SWITCH_KEYWORDS):
            return {"kind": "configure_openrouter", "make_default": True}
        if any(token in normalized for token in _PROVIDER_STATUS_KEYWORDS):
            return {"kind": "provider_status", "provider_id": "openrouter"}
    for provider_id in ("ollama", "openai"):
        if provider_id in normalized and any(token in normalized for token in _PROVIDER_STATUS_KEYWORDS):
            return {"kind": "provider_status", "provider_id": provider_id}
    if _looks_like_current_model_query(normalized):
        return {"kind": "describe_current_model"}
    if _looks_like_model_switch_advisory_query(normalized):
        return {"kind": "model_switch_advisory"}
    if any(phrase in normalized for phrase in _FIND_OLLAMA_MODELS_PHRASES):
        return {"kind": "find_ollama_models", "provider_id": "ollama"}
    if any(phrase in normalized for phrase in _BETTER_LOCAL_MODEL_PHRASES):
        return {"kind": "switch_better_local_model"}
    if _looks_like_local_model_recommendation_query(normalized):
        return {"kind": "recommend_local_model"}
    if any(phrase in normalized for phrase in _OPENROUTER_USE_PHRASES):
        return {"kind": "configure_openrouter", "make_default": True}
    if any(phrase in normalized for phrase in _OPENROUTER_SETUP_PHRASES):
        return {"kind": "configure_openrouter", "make_default": False}
    if any(phrase in normalized for phrase in _SET_DEFAULT_MODEL_PATTERNS):
        return {"kind": "set_default_model"}

    if normalized.startswith("make ") and " default" in normalized:
        return {"kind": "set_default_model"}

    api_key = extract_openrouter_api_key(text)
    if api_key and "openrouter" in normalized:
        return {"kind": "provide_openrouter_key", "api_key": api_key}

    return {"kind": "none"}


@dataclass(frozen=True)
class SemanticChatIntent:
    intent: str
    route: str
    kind: str
    confidence: float
    evidence: tuple[str, ...] = ()


def _semantic_intent_for_route(route_decision: dict[str, Any]) -> SemanticChatIntent:
    route = str(route_decision.get("route") or "generic_chat").strip().lower() or "generic_chat"
    kind = str(route_decision.get("kind") or "generic_chat").strip().lower() or "generic_chat"
    fallback_reason = str(route_decision.get("fallback_reason") or "").strip().lower()
    evidence = tuple(
        item
        for item in (
            kind,
            fallback_reason,
        )
        if item
    )
    if kind in {"safe_web_search", "safe_web_search_status", "safe_web_search_suppressed"}:
        intent = "web_search" if kind == "safe_web_search" else "status_check" if kind == "safe_web_search_status" else "answer_directly"
        return SemanticChatIntent(intent=intent, route=route, kind=kind, confidence=0.9, evidence=evidence)
    if kind == "safe_web_search_clarify" or route == "assistant_clarification":
        return SemanticChatIntent(intent="ask_clarifying_question", route=route, kind=kind, confidence=0.85, evidence=evidence)
    if kind == "telegram_service_action":
        return SemanticChatIntent(intent="managed_service_action", route=route, kind=kind, confidence=0.9, evidence=evidence)
    if kind == "safety_bypass_refusal":
        return SemanticChatIntent(intent="refusal_or_safe_redirect", route=route, kind=kind, confidence=0.95, evidence=evidence)
    if kind in {"telegram_status", "runtime_status", "provider_status", "providers_status"} or route in {
        "runtime_status",
        "provider_status",
        "model_status",
        "governance_status",
        "model_policy_status",
    }:
        return SemanticChatIntent(intent="status_check", route=route, kind=kind, confidence=0.9, evidence=evidence)
    if route == "operator_lifecycle":
        intent = "status_check" if kind == "operator_storage_status" else "managed_service_action"
        return SemanticChatIntent(intent=intent, route=route, kind=kind, confidence=0.9, evidence=evidence)
    if route == "memory_lifecycle":
        return SemanticChatIntent(intent="managed_service_action", route=route, kind=kind, confidence=0.9, evidence=evidence)
    if kind in {"shell_install_package", "shell_create_directory", "shell_safe_command", "shell_blocked_request"}:
        intent = "package_or_system_mutation_preview" if kind in {"shell_install_package", "shell_create_directory"} else "status_check"
        if kind == "shell_blocked_request":
            intent = "refusal_or_safe_redirect"
        return SemanticChatIntent(intent=intent, route=route, kind=kind, confidence=0.9, evidence=evidence)
    if kind in {"pack_capability_recommendation", "capability_gap_plan"}:
        return SemanticChatIntent(intent="pack_guidance", route=route, kind=kind, confidence=0.8, evidence=evidence)
    if route == "setup_flow" or kind in {"configure_openrouter", "configure_ollama", "provide_openrouter_key"}:
        return SemanticChatIntent(intent="managed_service_action", route=route, kind=kind, confidence=0.85, evidence=evidence)
    if route == "runtime_guard" or "blocked" in kind:
        return SemanticChatIntent(intent="refusal_or_safe_redirect", route=route, kind=kind, confidence=0.85, evidence=evidence)
    if route == "generic_chat" or kind in {"generic_chat", "none"}:
        return SemanticChatIntent(intent="answer_directly", route=route, kind=kind, confidence=0.65, evidence=evidence)
    return SemanticChatIntent(intent="answer_directly", route=route, kind=kind, confidence=0.6, evidence=evidence)


def _with_semantic_intent(route_decision: dict[str, Any]) -> dict[str, Any]:
    semantic = _semantic_intent_for_route(route_decision)
    enriched = dict(route_decision)
    enriched["semantic_intent"] = semantic.intent
    enriched["confidence"] = semantic.confidence
    enriched["evidence"] = list(semantic.evidence)
    enriched.setdefault("stale_context_cleared", False)
    enriched["semantic"] = asdict(semantic)
    return enriched


def _classify_runtime_chat_route_raw(
    text: str | None,
    *,
    awaiting_secret: bool = False,
    awaiting_confirmation: bool = False,
) -> dict[str, Any]:
    normalized = normalize_setup_text(text)
    if not normalized:
        return {
            "route": "generic_chat",
            "kind": "none",
            "generic_allowed": True,
            "fallback_reason": "empty_message",
        }

    model_policy_route = _classify_model_policy_route(normalized)
    if model_policy_route is not None:
        return model_policy_route

    setup_intent = classify_setup_intent(
        text,
        awaiting_secret=awaiting_secret,
        awaiting_confirmation=awaiting_confirmation,
    )
    setup_kind = str(setup_intent.get("kind") or "none").strip().lower()
    if setup_kind in {"configure_openrouter", "configure_ollama", "provide_openrouter_key", "confirm_pending_setup", "cancel_pending_setup"}:
        return {
            **setup_intent,
            "route": "setup_flow",
            "generic_allowed": False,
            "fallback_reason": "setup_flow",
        }
    if setup_kind in {"provider_status", "providers_status"}:
        return {
            **setup_intent,
            "route": "provider_status",
            "generic_allowed": False,
            "fallback_reason": "provider_status",
        }
    if setup_kind == "setup_explanation":
        return {
            **setup_intent,
            "route": "setup_flow",
            "generic_allowed": False,
            "fallback_reason": "setup_flow",
        }
    if setup_kind in {
        "governance_adapters",
        "governance_background_tasks",
        "governance_blocks",
        "governance_pending",
        "governance_overview",
        "governance_adapter_detail",
    }:
        return {
            **setup_intent,
            "route": "governance_status",
            "generic_allowed": False,
            "fallback_reason": "governance_status",
        }
    if setup_kind == "model_controller_policy":
        return {
            **setup_intent,
            "route": "model_policy_status",
            "generic_allowed": False,
            "fallback_reason": "model_policy_status",
        }
    if setup_kind in {
        "describe_current_model",
        "local_model_inventory",
        "model_lifecycle_status",
        "find_ollama_models",
        "switch_better_local_model",
        "set_default_model",
    }:
        return {
            **setup_intent,
            "route": "model_status",
            "generic_allowed": False,
            "fallback_reason": "model_status",
        }
    if setup_kind in {"model_acquisition_request", "model_switch_advisory"}:
        return {
            **setup_intent,
            "route": "action_tool",
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }
    if setup_kind in {"recommend_local_model", "local_provider_guidance"}:
        return {
            **setup_intent,
            "route": "action_tool",
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }
    if _looks_like_runtime_status_query(normalized):
        return {
            "route": "runtime_status",
            "kind": "runtime_status",
            "generic_allowed": False,
            "fallback_reason": "runtime_status",
        }
    if _SAFETY_BYPASS_RE.search(normalized):
        return {
            "route": "action_tool",
            "kind": "safety_bypass_refusal",
            "generic_allowed": False,
            "fallback_reason": "safety_bypass_refusal",
        }
    if _SECRET_REVEAL_RE.search(normalized):
        return {
            "route": "runtime_status",
            "kind": "secret_store_status",
            "generic_allowed": False,
            "fallback_reason": "secret_store_status",
        }
    if _TELEGRAM_ACTION_RE.search(normalized):
        action_match = _TELEGRAM_ACTION_RE.search(normalized)
        action = str((action_match.group(1) or action_match.group(2) or "") if action_match else "").strip().lower()
        return {
            "route": "action_tool",
            "kind": "telegram_service_action",
            "telegram_action": action,
            "generic_allowed": False,
            "fallback_reason": "telegram_service_action",
        }
    if _AMBIGUOUS_RESTART_RE.search(normalized):
        return {
            "route": "assistant_clarification",
            "kind": "ambiguous_restart_target",
            "generic_allowed": False,
            "fallback_reason": "ambiguous_restart_target",
        }
    if _looks_like_safe_web_search_status_request(normalized):
        return {
            "route": "action_tool",
            "kind": "safe_web_search_status",
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }
    if _looks_like_provided_text_transform(normalized):
        return {
            "route": "generic_chat",
            "kind": "generic_chat",
            "generic_allowed": True,
            "fallback_reason": "provided_text_transform",
        }
    if _looks_like_safe_web_search_request(normalized):
        return {
            "route": "action_tool",
            "kind": "safe_web_search",
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }
    if _looks_like_model_scout_discovery_query(normalized):
        return {
            "route": "action_tool",
            "kind": "model_scout_discovery",
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }
    if _looks_like_model_scout_recommendation_query(normalized):
        return {
            "route": "action_tool",
            "kind": "model_scout_strategy",
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }
    if any(phrase in normalized for phrase in _TELEGRAM_STATUS_PHRASES):
        return {
            "route": "runtime_status",
            "kind": "telegram_status",
            "generic_allowed": False,
            "fallback_reason": "telegram_status",
        }
    if any(phrase in normalized for phrase in _PLAN_DAY_PHRASES):
        return {
            "route": "plan_day",
            "kind": "plan_day",
            "generic_allowed": False,
            "fallback_reason": "plan_day",
        }
    if _looks_like_assistant_capabilities_query(normalized):
        return {
            "route": "assistant_capabilities",
            "kind": "assistant_capabilities",
            "generic_allowed": False,
            "fallback_reason": "assistant_capabilities",
        }
    shell_route = _classify_shell_route(text, normalized)
    if shell_route is not None:
        return shell_route
    if _looks_like_safe_web_search_suppressed_request(normalized):
        return {
            "route": "action_tool",
            "kind": "safe_web_search_suppressed",
            "generic_allowed": False,
            "fallback_reason": "user_declined_search",
        }
    if _looks_like_safe_web_search_clarification_request(normalized):
        return {
            "route": "assistant_clarification",
            "kind": "safe_web_search_clarify",
            "generic_allowed": False,
            "fallback_reason": "ambiguous_search_target",
        }
    if (
        _looks_like_bare_public_lookup_question(text, normalized)
        or _looks_like_raw_niche_public_lookup_question(text, normalized)
        or _looks_like_safe_web_search_fallback_request(normalized)
    ):
        return {
            "route": "action_tool",
            "kind": "safe_web_search",
            "generic_allowed": False,
            "fallback_reason": "search_fallback",
        }
    capability_gap = classify_capability_gap_request(normalized)
    if str(capability_gap.get("request_kind") or "").strip().lower() == "capability" and str(
        capability_gap.get("classification") or ""
    ).strip().lower() != "can_answer_locally":
        capability_key = str(capability_gap.get("capability") or "").strip().lower() or None
        if capability_key is not None:
            return {
                "route": "action_tool",
                "kind": "pack_capability_recommendation",
                "capability": capability_key,
                "capability_label": capability_gap.get("label"),
                "generic_allowed": False,
                "fallback_reason": "action_tool",
            }
        return {
            "route": "action_tool",
            "kind": "capability_gap_plan",
            "capability_label": capability_gap.get("label"),
            "capability_classification": capability_gap.get("classification"),
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }
    if _looks_like_model_ready_now_query(normalized):
        return {
            "route": "model_status",
            "kind": "model_ready_now",
            "generic_allowed": False,
            "fallback_reason": "model_status",
        }
    if _looks_like_model_availability_query(normalized):
        return {
            "route": "model_status",
            "kind": "model_availability",
            "inventory_scope": _model_inventory_scope(normalized),
            "generic_allowed": False,
            "fallback_reason": "model_status",
        }
    agent_memory_route = _classify_agent_memory_route(normalized)
    if agent_memory_route is not None:
        return agent_memory_route
    operator_lifecycle_route = _classify_operator_lifecycle_route(normalized)
    if operator_lifecycle_route is not None:
        return operator_lifecycle_route
    filesystem_route = _classify_filesystem_route(text, normalized)
    if filesystem_route is not None:
        return filesystem_route
    operational_route = _classify_operational_route(text, normalized)
    if operational_route is not None:
        return operational_route
    if any(phrase in normalized for phrase in _GENERIC_MODEL_REQUEST_PHRASES):
        return {
            "route": "action_tool",
            "kind": "model_scout_strategy",
            "generic_allowed": False,
            "fallback_reason": "action_tool",
        }

    mentions_guard_phrase = any(phrase in normalized for phrase in _PRODUCT_GUARD_PHRASES)
    guard_words = {word for word in _PRODUCT_GUARD_WORDS if re.search(rf"\b{re.escape(word)}\b", normalized)}
    if mentions_guard_phrase or guard_words:
        return {
            "route": "runtime_guard",
            "kind": "product_specific_guard",
            "generic_allowed": False,
            "fallback_reason": "product_specific_guard",
        }
    return {
        "route": "generic_chat",
        "kind": "generic_chat",
        "generic_allowed": True,
        "fallback_reason": "ordinary_open_chat",
    }


def classify_runtime_chat_route(
    text: str | None,
    *,
    awaiting_secret: bool = False,
    awaiting_confirmation: bool = False,
) -> dict[str, Any]:
    return _with_semantic_intent(
        _classify_runtime_chat_route_raw(
            text,
            awaiting_secret=awaiting_secret,
            awaiting_confirmation=awaiting_confirmation,
        )
    )


def is_setup_related_text(text: str | None) -> bool:
    intent = classify_setup_intent(text)
    return str(intent.get("kind") or "none") != "none"
