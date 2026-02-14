from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


CandidateKind = Literal["answer", "clarify"]
ClaimSupport = Literal["memory", "tool", "user", "none"]


@dataclass(frozen=True)
class Claim:
    text: str
    support: ClaimSupport
    ref: str | None = None


@dataclass(frozen=True)
class ThreadRef:
    target_thread_id: str
    needs_confirmation: bool


@dataclass(frozen=True)
class CandidateContract:
    kind: CandidateKind
    final_answer: str
    clarifying_question: str | None
    claims: tuple[Claim, ...] = field(default_factory=tuple)
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    unresolved_refs: tuple[str, ...] = field(default_factory=tuple)
    thread_refs: tuple[ThreadRef, ...] = field(default_factory=tuple)
    raw_json: str | None = None


@dataclass(frozen=True)
class MessageTurn:
    role: Literal["user", "assistant"]
    text: str


@dataclass(frozen=True)
class ContextPack:
    user_id: str
    active_thread_id: str | None
    recent_messages: tuple[MessageTurn, ...] = field(default_factory=tuple)
    memory_hits: tuple[str, ...] = field(default_factory=tuple)
    memory_ambiguous: tuple[str, ...] = field(default_factory=tuple)
    memory_miss: bool = False
    tools_available: tuple[str, ...] = field(default_factory=tuple)
    tool_failures: tuple[str, ...] = field(default_factory=tuple)
    referents: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DetectorReason:
    code: str
    detail: str
    evidence: str | None = None
    hard: bool = False


@dataclass(frozen=True)
class DetectorResult:
    score: float
    reasons: tuple[DetectorReason, ...]
    hard_reasons: tuple[str, ...]


@dataclass(frozen=True)
class GateDecision:
    intercepted: bool
    user_text: str
    reasons: tuple[str, ...]
    hard_reasons: tuple[str, ...]
    score: float
    question: str | None = None
    contract_errors: tuple[str, ...] = field(default_factory=tuple)
    candidate_kind: str = "answer"
