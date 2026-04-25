from __future__ import annotations

import re
from typing import Any

from agent.cards import normalize_card

_INTENTS = {
    "OBSERVE_PC",
    "EXPLAIN_PREVIOUS",
    "DIAGNOSTICS_CAPTURE_REQUEST",
    "DIAGNOSTICS_CAPTURE_GENERIC_DEVICE_FALLBACK_REQUEST",
    "DIAGNOSTICS_CAPTURE_BLUETOOTH_AUDIO_REQUEST",
    "DIAGNOSTICS_CAPTURE_STORAGE_DISK_REQUEST",
    "DIAGNOSTICS_CAPTURE_PRINTER_CUPS_REQUEST",
    "MEMORY_WRITE_REQUEST",
    "PLAN_DAY",
    "SHOW_PREFERENCES",
    "MEMORY_INSPECT",
    "OPEN_LOOP_ADD",
    "OPEN_LOOP_DONE",
    "OPEN_LOOPS_LIST",
    "DAILY_BRIEF_STATUS",
    "CHITCHAT",
    "UNKNOWN",
}

_OBSERVE_DISK = re.compile(r"\b(disk|storage|ssd|space|filesystem|mount|directory|folder)\b", re.IGNORECASE)
_OBSERVE_RESOURCE = re.compile(r"\b(cpu|load|memory|ram|process|proc|performance)\b", re.IGNORECASE)
_OBSERVE_MACHINE = re.compile(
    r"\b(gpu|graphics|video card|hardware|pc stats|pc specs|pc info|computer stats|computer specs|computer info|machine stats|machine specs|machine info|system stats|system info|system details)\b",
    re.IGNORECASE,
)
_OBSERVE_PERFORMANCE = re.compile(
    r"\b(slow|slowly|laggy|lagging|sluggish|stutter|stuttering|throttle|throttling|unresponsive|frozen|stuck|dragging)\b",
    re.IGNORECASE,
)
_OBSERVE_PERFORMANCE_CONTEXT = re.compile(
    r"\b(pc|computer|system|machine|download|downloading|upload|uploading|network|internet|wifi|connection|browser|app|game|disk|storage|drive|file|transfer|latency|ping)\b",
    re.IGNORECASE,
)
_DIAGNOSTICS_CONTEXT = re.compile(
    r"\b(linux|laptop|computer|pc|system|machine|wifi|wi-fi|network|driver|kernel|suspend|resume|sleep|wake|bluetooth|audio|display|gpu|graphics)\b",
    re.IGNORECASE,
)
_DIAGNOSTICS_ISSUE = re.compile(
    r"\b(problem|issue|bug|broken|not working|won't|can't|fails?|error|crash(?:es|ed)?|freeze(?:s|d)?|stuck|disconnect(?:s|ed)?|drop(?:s|ping)?|intermittent|after suspend|after sleep)\b",
    re.IGNORECASE,
)
_DIAGNOSTICS_HELP = re.compile(r"\b(help me debug|debug|diagnose|troubleshoot|fix this|figure out why)\b", re.IGNORECASE)
_BLUETOOTH_AUDIO_CONTEXT = re.compile(
    r"\b(bluetooth|headphones?|headset|earbuds?|earbud|buds?|speaker(?:s)?|audio|sound|microphone|mic)\b",
    re.IGNORECASE,
)
_BLUETOOTH_AUDIO_ISSUE = re.compile(
    r"\b(problem|issue|bug|broken|not working|won't|can't|fails?|error|crash(?:es|ed)?|freeze(?:s|d)?|stuck|"
    r"disconnect(?:s|ed)?|drop(?:s|ping)?|intermittent|after suspend|after sleep|no sound|sound stops|audio stops|"
    r"won't pair|won't connect|cuts out)\b",
    re.IGNORECASE,
)
_STORAGE_DISK_CONTEXT = re.compile(r"\b(disk|storage|filesystem|filesystems|space|drive|volume|partition)\b", re.IGNORECASE)
_STORAGE_DISK_ISSUE = re.compile(
    r"\b(full|out of space|almost full|no space left on device|can't save|cannot save|can't write|cannot write|write failed|save failed|read-only file system)\b",
    re.IGNORECASE,
)
_PRINTER_CUPS_CONTEXT = re.compile(r"\b(printer|printers|cups|print queue|print job|print jobs|print spooler)\b", re.IGNORECASE)
_PRINTER_CUPS_ISSUE = re.compile(
    r"\b(not printing|offline|stuck|stalled|queue stuck|can't print|cannot print|won't print|failed to print|print(?:ing)? fails?)\b",
    re.IGNORECASE,
)
_GENERIC_DEVICE_CONTEXT = re.compile(
    r"\b(camera|webcam|usb|hardware|device|peripheral|display|monitor|screen|keyboard|mouse|touchpad|trackpad|scanner|microphone|mic|dock|adapter|sensor|controller|laptop|desktop|pc)\b",
    re.IGNORECASE,
)
_GENERIC_DEVICE_ISSUE = re.compile(
    r"\b(not detected|not recognized|not working|broken|fails?|error|crash(?:es|ed)?|freeze(?:s|d)?|stuck|disconnect(?:s|ed)?|drop(?:s|ping)?|intermittent|after suspend|after sleep|won't turn on|won't start|no response|not found|stopped working|can't use|cannot use)\b",
    re.IGNORECASE,
)
_GENERIC_DEVICE_HELP = re.compile(r"\b(help me debug|debug|diagnose|troubleshoot|fix this|figure out why|what's wrong|what is wrong)\b", re.IGNORECASE)
_PROCESS_ROLE_QUESTION = re.compile(
    r"\b(why (?:are|is) there|why do i have|how many|what are these|what is this|which one is the|"
    r"double check|recheck|check again|look again|confirm it)\b.*\b(ollama|qemu|qemu-system|virtualbox|vmware|browser|chrome|chromium|firefox|instance|instances|process|processes)\b",
    re.IGNORECASE,
)
_OBSERVE_DEEPER = re.compile(
    r"\b(show me more system info|tell me more about (?:this )?(?:machine|computer|system)|"
    r"inspect the system more deeply|inspect the machine more deeply|inspect the computer more deeply|"
    r"what else can you find|dig deeper|dig deeper into (?:my|the) system|"
    r"run a check and see if you can learn more|run a check and see what else you can find|"
    r"run a system check|check the system|scan the system|inspect the machine more|learn more)\b",
    re.IGNORECASE,
)
_EXPLAIN = re.compile(r"\b(explain|why|what changed|change[d]?|previous|last time|since)\b", re.IGNORECASE)
_MEMORY_WRITE = re.compile(
    r"^(remember that|from now on|remember this|set max cards to|turn confidence (on|off)|default compare (on|off)|daily brief (on|off)|set disk delta threshold to|only send if service unhealthy|set open loops due window to)\b",
    re.IGNORECASE,
)
_PLAN_DAY = re.compile(
    r"\b(plan my day|today plan|help me plan my day|help me plan today|what should i do today|what should i work on today|show quick wins|show top 3 priorities)\b",
    re.IGNORECASE,
)
_SHOW_PREFS = re.compile(r"\b(show my preferences|what are my preferences|preferences)\b", re.IGNORECASE)
_MEMORY_INSPECT = re.compile(
    r"\b("
    r"what do you remember about me|"
    r"what do you remember|"
    r"what are we working on|"
    r"what were we working on|"
    r"what were we doing before|"
    r"can you help with the thing we were doing before|"
    r"what do you know about my system|"
    r"what do you know about this system|"
    r"what do you know about my machine|"
    r"what do you know about this machine|"
    r"what is currently in your memory files|"
    r"what is in your memory files|"
    r"what s in your memory files"
    r")\b",
    re.IGNORECASE,
)
_OPEN_LOOP_ADD = re.compile(r"^remember that (?:i need to )?.+ by .+$", re.IGNORECASE)
_OPEN_LOOP_DONE = re.compile(r"^mark .+ done$", re.IGNORECASE)
_OPEN_LOOPS_LIST = re.compile(r"\b(open loops|show open loops|list open loops)\b", re.IGNORECASE)
_DAILY_BRIEF_STATUS = re.compile(
    r"\b(daily brief status|why didn.?t .*daily brief|why did(n't| not) .*daily brief)\b",
    re.IGNORECASE,
)
_CHITCHAT = re.compile(r"^(hi|hello|hey|thanks|thank you)\b", re.IGNORECASE)
_HARDWARE_INVENTORY_PHRASES = (
    "what do i have for ram and vram right now",
    "how much ram and vram do i have",
    "what cpu do i have",
    "what gpu do i have",
    "what cpu and gpu do i have",
    "can you tell what cpu and gpu i have",
    "can you see the gpu",
    "what hardware am i running",
    "what are my pc specs",
    "what are my computer specs",
    "what specs do i have",
    "what hardware do i have",
)
_RAM_VRAM_REQUEST_RE = re.compile(r"\bram\b.*\bvram\b|\bvram\b.*\bram\b", re.IGNORECASE)
_MACHINE_BROAD_PHRASES = (
    "what other pc stats can you find",
    "other pc stats",
    "show me more system info",
    "tell me more about this machine",
    "tell me more about this computer",
    "tell me more about this system",
    "inspect the system more deeply",
    "inspect the machine more deeply",
    "inspect the computer more deeply",
    "what else can you find",
    "dig deeper",
    "can you dig deeper into my system",
    "dig deeper into my system",
    "run a check and see if you can learn more",
    "run a check and see what else you can find",
    "run a system check",
    "check the system",
    "scan the system",
    "learn more about my computer",
    "learn more about this machine",
    "learn more about this system",
)


def _contains_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _hardware_inventory_requested(text: str) -> bool:
    return _contains_any_phrase(text, _HARDWARE_INVENTORY_PHRASES) or bool(_RAM_VRAM_REQUEST_RE.search(text))


def _machine_broad_requested(text: str) -> bool:
    return _contains_any_phrase(text, _MACHINE_BROAD_PHRASES) or bool(_OBSERVE_DEEPER.search(text))


def _diagnostics_capture_requested(text: str) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return False
    return bool(
        (
            _DIAGNOSTICS_HELP.search(lowered)
            and (_DIAGNOSTICS_CONTEXT.search(lowered) or _DIAGNOSTICS_ISSUE.search(lowered))
        )
        or (_DIAGNOSTICS_ISSUE.search(lowered) and _DIAGNOSTICS_CONTEXT.search(lowered))
    )


def _bluetooth_audio_diagnostics_requested(text: str) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return False
    return bool(
        (_DIAGNOSTICS_HELP.search(lowered) and _BLUETOOTH_AUDIO_CONTEXT.search(lowered))
        or (_BLUETOOTH_AUDIO_CONTEXT.search(lowered) and _BLUETOOTH_AUDIO_ISSUE.search(lowered))
    )


def _storage_disk_diagnostics_requested(text: str) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return False
    return bool(_STORAGE_DISK_CONTEXT.search(lowered) and _STORAGE_DISK_ISSUE.search(lowered))


def _printer_cups_diagnostics_requested(text: str) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return False
    return bool(
        (_DIAGNOSTICS_HELP.search(lowered) and _PRINTER_CUPS_CONTEXT.search(lowered))
        or (_PRINTER_CUPS_CONTEXT.search(lowered) and _PRINTER_CUPS_ISSUE.search(lowered))
    )


def _generic_device_fallback_diagnostics_requested(text: str) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return False
    return bool(
        (_GENERIC_DEVICE_HELP.search(lowered) and _GENERIC_DEVICE_CONTEXT.search(lowered))
        or (_GENERIC_DEVICE_CONTEXT.search(lowered) and _GENERIC_DEVICE_ISSUE.search(lowered))
    )


def _process_role_requested(text: str) -> bool:
    lowered = (text or "").lower()
    return bool(
        _PROCESS_ROLE_QUESTION.search(lowered)
        or (("ollama" in lowered or "qemu" in lowered or "browser" in lowered) and ("instance" in lowered or "process" in lowered))
    )


def _looks_like_coding_request(text: str) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return False
    if not any(token in lowered for token in ("code", "coding", "script", "program", "function", "refactor", "implement", "debug", "review")):
        return False
    return any(token in lowered for token in ("write", "build", "create", "make", "help me", "can you", "could you", "convert", "rewrite"))


def looks_like_system_performance_question(text: str) -> bool:
    lowered = (text or "").lower()
    return bool(_OBSERVE_PERFORMANCE.search(lowered) and _OBSERVE_PERFORMANCE_CONTEXT.search(lowered))


def _looks_like_network_observation(text: str) -> bool:
    lowered = (text or "").lower()
    return bool(
        re.search(r"\b(network|dns|gateway|latency|ping|bandwidth|throughput|internet|wifi|connection|connections)\b", lowered)
        or (
            re.search(r"\b(download|downloading|upload|uploading)\b", lowered)
            and bool(_OBSERVE_PERFORMANCE.search(lowered))
        )
    )


def _append_skill(selected: list[dict[str, str]], skill: str, function: str) -> None:
    entry = {"skill": skill, "function": function}
    if entry not in selected:
        selected.append(entry)


def classify_free_text(text: str) -> str:
    cleaned = (text or "").strip()
    lowered = cleaned.lower()
    if not cleaned:
        return "UNKNOWN"
    if _OPEN_LOOP_ADD.search(lowered):
        return "OPEN_LOOP_ADD"
    if _OPEN_LOOP_DONE.search(lowered):
        return "OPEN_LOOP_DONE"
    if _OPEN_LOOPS_LIST.search(lowered):
        return "OPEN_LOOPS_LIST"
    if _DAILY_BRIEF_STATUS.search(lowered):
        return "DAILY_BRIEF_STATUS"
    if _printer_cups_diagnostics_requested(lowered):
        return "DIAGNOSTICS_CAPTURE_PRINTER_CUPS_REQUEST"
    if _storage_disk_diagnostics_requested(lowered):
        return "DIAGNOSTICS_CAPTURE_STORAGE_DISK_REQUEST"
    if _bluetooth_audio_diagnostics_requested(lowered):
        return "DIAGNOSTICS_CAPTURE_BLUETOOTH_AUDIO_REQUEST"
    if _generic_device_fallback_diagnostics_requested(lowered):
        return "DIAGNOSTICS_CAPTURE_GENERIC_DEVICE_FALLBACK_REQUEST"
    if _diagnostics_capture_requested(lowered):
        return "DIAGNOSTICS_CAPTURE_REQUEST"
    if _MEMORY_WRITE.search(lowered):
        return "MEMORY_WRITE_REQUEST"
    if _PLAN_DAY.search(lowered):
        return "PLAN_DAY"
    if _SHOW_PREFS.search(lowered):
        return "SHOW_PREFERENCES"
    if _MEMORY_INSPECT.search(lowered):
        return "MEMORY_INSPECT"
    if _CHITCHAT.search(lowered):
        return "CHITCHAT"
    if _process_role_requested(lowered):
        return "EXPLAIN_PREVIOUS"
    if _looks_like_coding_request(lowered):
        return "UNKNOWN"
    if _EXPLAIN.search(lowered) and (
        _OBSERVE_DISK.search(lowered)
        or _OBSERVE_RESOURCE.search(lowered)
        or _OBSERVE_MACHINE.search(lowered)
        or looks_like_system_performance_question(lowered)
        or _hardware_inventory_requested(lowered)
        or _machine_broad_requested(lowered)
        or _process_role_requested(lowered)
    ):
        return "EXPLAIN_PREVIOUS"
    if (
        _OBSERVE_DISK.search(lowered)
        or _OBSERVE_RESOURCE.search(lowered)
        or _OBSERVE_MACHINE.search(lowered)
        or looks_like_system_performance_question(lowered)
        or _OBSERVE_DEEPER.search(lowered)
        or _hardware_inventory_requested(lowered)
        or _machine_broad_requested(lowered)
        or _process_role_requested(lowered)
        or "my pc" in lowered
        or "my computer" in lowered
    ):
        return "OBSERVE_PC"
    return "UNKNOWN"


def select_observe_skills(text: str) -> list[dict[str, str]]:
    lowered = (text or "").lower()
    selected: list[dict[str, str]] = []
    hardware_requested = _hardware_inventory_requested(lowered)
    machine_broad = _machine_broad_requested(lowered)
    performance_requested = looks_like_system_performance_question(lowered)
    process_role_requested = _process_role_requested(lowered)
    machine_requested = bool(
        _OBSERVE_MACHINE.search(lowered)
        or hardware_requested
        or machine_broad
        or process_role_requested
        or "my pc" in lowered
        or "my computer" in lowered
        or "this machine" in lowered
        or "this system" in lowered
    )
    if machine_broad:
        _append_skill(selected, "hardware_report", "hardware_report")
        _append_skill(selected, "resource_governor", "resource_report")
        _append_skill(selected, "storage_governor", "storage_report")
    elif hardware_requested:
        _append_skill(selected, "hardware_report", "hardware_report")
        # A direct RAM/VRAM inventory question should stay on the lightweight
        # hardware path; broader memory/performance wording still gets the
        # heavier resource summary below.
        if bool(_OBSERVE_RESOURCE.search(lowered)) and not _RAM_VRAM_REQUEST_RE.search(lowered):
            _append_skill(selected, "resource_governor", "resource_report")
    else:
        if machine_requested:
            _append_skill(selected, "hardware_report", "hardware_report")
        if process_role_requested:
            _append_skill(selected, "resource_governor", "resource_report")
        if _OBSERVE_DISK.search(lowered):
            _append_skill(selected, "storage_governor", "storage_report")
            if "pressure" in lowered or "largest files" in lowered:
                selected = [{"skill": "disk_pressure_report", "function": "disk_pressure_report"}]
        if _OBSERVE_RESOURCE.search(lowered) or performance_requested:
            _append_skill(selected, "resource_governor", "resource_report")
    if _looks_like_network_observation(lowered):
        _append_skill(selected, "network_governor", "network_report")
    if re.search(r"\bservice health|service status|agent service|personal-agent service\b", lowered):
        _append_skill(selected, "service_health_report", "service_health_report")
    if not selected:
        if machine_requested:
            if machine_broad:
                _append_skill(selected, "hardware_report", "hardware_report")
                _append_skill(selected, "resource_governor", "resource_report")
                _append_skill(selected, "storage_governor", "storage_report")
            else:
                _append_skill(selected, "hardware_report", "hardware_report")
                if process_role_requested:
                    _append_skill(selected, "resource_governor", "resource_report")
        else:
            _append_skill(selected, "storage_governor", "storage_report")
            _append_skill(selected, "resource_governor", "resource_report")
    # Deterministic order.
    selected.sort(key=lambda item: (item["skill"], item["function"]))
    return selected


def build_cards_payload(
    cards: list[dict[str, Any]],
    raw_available: bool,
    summary: str,
    confidence: float,
    next_questions: list[str],
) -> dict[str, Any]:
    normalized = [normalize_card(card, idx) for idx, card in enumerate(cards)]
    return {
        "cards": normalized,
        "raw_available": bool(raw_available),
        "summary": summary,
        "confidence": max(0.0, min(1.0, float(confidence))),
        "next_questions": [str(item) for item in next_questions],
    }


def nl_route(text: str) -> dict[str, Any]:
    intent = classify_free_text(text)
    if intent not in _INTENTS:
        intent = "UNKNOWN"
    response: dict[str, Any] = {"intent": intent, "skills": [], "cards_payload": None}

    if intent in {"OBSERVE_PC", "EXPLAIN_PREVIOUS"}:
        response["skills"] = select_observe_skills(text)
    return response
