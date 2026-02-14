from __future__ import annotations

from agent.epistemics.types import CandidateContract, Claim


_PREFIX = "In short: "
_MAX_TOTAL_LEN = 140
_SUPPORT_PRIORITY = ("tool", "memory", "user")


def _claim_is_supported(claim: Claim) -> bool:
    if claim.support == "tool":
        return bool(claim.tool_event_id)
    if claim.support == "memory":
        return claim.memory_id is not None
    if claim.support == "user":
        return bool(claim.user_turn_id)
    return False


def _pick_claim(candidate: CandidateContract) -> Claim | None:
    for support in _SUPPORT_PRIORITY:
        for claim in candidate.claims:
            if claim.support == support and _claim_is_supported(claim):
                return claim
    return None


def compute_summary(candidate: CandidateContract, rendered_answer: str) -> str | None:
    line_count = len((rendered_answer or "").splitlines())
    if line_count <= 8:
        return None

    claim = _pick_claim(candidate)
    if claim is None:
        return None

    summary = " ".join((claim.text or "").replace("\n", " ").split())
    summary = summary.replace("?", "").strip()
    if not summary:
        return None

    max_content = _MAX_TOTAL_LEN - len(_PREFIX)
    if max_content <= 0:
        return None
    if len(summary) > max_content:
        summary = summary[:max_content].rstrip()
    if not summary:
        return None

    line = f"{_PREFIX}{summary}"
    if "?" in line or "\n" in line or len(line) > _MAX_TOTAL_LEN:
        return None
    return line

