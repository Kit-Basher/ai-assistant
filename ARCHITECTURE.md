# Architecture (v0.8)

This document explains the stable architecture spine and how new features must fit into it.

## Core Spine (Plain Terms)
Telegram input (untrusted) flows through a strict sequence:

Telegram input (untrusted)
  -> intent / commands
  -> orchestrator
  -> skill
  -> DB + audit
  -> report text (facts)
  -> optional narration (rephrase only)

The orchestrator is the single entry point for behavior. Skills are the only execution surface. Reports are deterministic and factual. Narration is optional and may never alter or add facts.

## Three-Layer Model
1) Governors produce facts
- Collect snapshots, diffs, and raw measurements.
- Observe-only by default.

2) Reports format facts deterministically
- Convert facts into a stable, neutral report text.
- No advice or recommendations.

3) Narration optionally rephrases
- Takes report text as input.
- Produces a short, neutral summary.
- If narration is disabled or fails, raw report is returned unchanged.

## ASCII Diagram

[Telegram (untrusted)]
        |
        v
[Intent / Commands]
        |
        v
[Orchestrator]
        |
        v
[Skill / Governor]
        |
        v
[DB + Audit] ---> [Report Text (facts)] ---> [Optional Narration]

## Invariants
See `STABILITY.md` for the authoritative list of guarantees. This document does not redefine them.

## Extension Rules
Any new jurisdiction or governor must:
- Remain observe-only unless explicitly approved.
- Persist snapshots and compute diffs against previous snapshots.
- Enforce audit hard-fail (abort if audit logging fails).
- Be covered by tests comparable to storage/resource governors.

Narration rules:
- Must be optional and failure-safe.
- Must not introduce new facts or recommendations.
