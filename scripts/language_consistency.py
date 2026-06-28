from __future__ import annotations

import re
from dataclasses import dataclass


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_OTHER_LANGUAGE_REQUEST_RE = re.compile(
    r"\b(answer|reply|respond|write|explain|translate|summarize)\s+(?:this\s+)?(?:in|to)\s+"
    r"(?:chinese|mandarin|cantonese|japanese|korean|spanish|french|german|italian|portuguese|russian|arabic|hindi)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LanguageConsistencyResult:
    ok: bool
    reason: str
    cjk_count: int
    checked_chars: int
    excerpt: str


def _strip_ignored_text(text: str) -> str:
    value = _FENCED_CODE_RE.sub(" ", str(text or ""))
    value = _INLINE_CODE_RE.sub(" ", value)
    value = _URL_RE.sub(" ", value)
    return value


def prompt_allows_non_english(prompt: str) -> bool:
    return bool(_OTHER_LANGUAGE_REQUEST_RE.search(str(prompt or "")))


def check_english_language_consistency(prompt: str, response: str) -> LanguageConsistencyResult:
    """Lightweight smoke-test heuristic for obvious mixed-language output.

    This intentionally avoids full language detection. It catches substantial
    CJK leakage in responses to English prompts while allowing URLs, code, and
    short foreign terms.
    """

    if prompt_allows_non_english(prompt):
        return LanguageConsistencyResult(True, "prompt_requested_other_language", 0, 0, "")
    cleaned = _strip_ignored_text(response)
    cjk_count = len(_CJK_RE.findall(cleaned))
    checked_chars = len([char for char in cleaned if not char.isspace()])
    if cjk_count >= 4 and checked_chars > 0 and (cjk_count / checked_chars) >= 0.02:
        compact = " ".join(str(response or "").split())
        return LanguageConsistencyResult(
            False,
            "language_consistency_failed: English prompt received substantial CJK text",
            cjk_count,
            checked_chars,
            compact[:240],
        )
    return LanguageConsistencyResult(True, "ok", cjk_count, checked_chars, "")

