# Runtime Gap Analysis

This document describes the current-to-intended runtime architecture gap for v1.
Canonical target is defined in [PRODUCT_RUNTIME_SPEC.md](/home/c/personal-agent/PRODUCT_RUNTIME_SPEC.md).

## Summary
- Current implementation has a strong shared core, and Telegram setup/status/help flows now route through shared runtime contracts.
- Intended model is one core runtime brain with thin surfaces.
- Main remaining gap is trimming residual adapter-owned UX helpers and keeping all decisions on shared runtime contracts.

## Current vs Intended

| Area | Current | Intended | Action |
|---|---|---|---|
| Telegram runtime | `telegram_adapter/bot.py` is mostly transport, with limited pre-routing/fix-it interception | Telegram is a thin transport adapter | Keep setup/status/help/doctor/memory decision text behind runtime-facing interfaces; keep bot focused on transport and message safety |
| API runtime | API runtime is primary core owner and already hosts major contracts | Keep API runtime as core owner | Continue consolidating shared behavior here; avoid reintroducing logic to adapters |
| Onboarding/setup | Shared contracts own setup truth; Telegram routes operators to canonical setup surfaces | One setup contract with native UI primary | Keep setup state source in runtime contracts; Telegram renders only |
| Recovery UX | Mostly unified contracts, some surface-specific wording remains | One recovery decision path, one next action | Replace residual surface-specific fallback wording with shared contract outputs |
| Status/help/doctor | Semantics mostly aligned but formatting and routing differ by surface | Same semantics across CLI/UI/Telegram | Preserve surface-specific formatting only; keep decision logic centralized |
| Memory/continuity | Central runtime exists; historical recursion bugs show adapter interactions can pollute state | Continuity state updated only by core runtime rules | Keep meta-action filtering in runtime; never let transport output rewrite meaningful continuity |
| LLM routing | Orchestrator owns main routing; some transport pre-routing still exists | Core runtime decides behavior | Keep only safe deterministic transport preprocessing; route decisions to runtime |
| Skill pack ownership | Pack policy/runtime gates exist in core | Skill packs extend, never own core runtime | Keep pack enforcement in core runtime execution gates |

## What Must Move Out Of Telegram Adapter
- Any business-rule branching that decides product state independent of runtime contracts.
- Any setup/recovery truth derivation that does not come from runtime state.
- Any continuity/state mutation policy.

## What Must Remain In Telegram Adapter
- Telegram transport IO.
- Input normalization and basic routing tokens.
- Safe-send behavior (truncate/retry/plain fallback).
- Transport-specific logging/audit wrappers.

## What Stays In Core Runtime
- Runtime mode and readiness truth.
- Onboarding/recovery contracts.
- Tool permissions/execution.
- LLM selection/routing.
- Continuity/memory policy.
- Skill pack policy enforcement.

## What Does Not Need To Change Yet
- No broad runtime rewrite.
- No immediate process-model change if embedded Telegram remains operational.
- No replacement of existing endpoint contracts during this pass.

## Near-Term Refactor Direction
1. Keep adapter transport-safe and deterministic.
2. Reduce adapter-owned product logic behind shared runtime interfaces.
3. Keep native UI setup/recovery as primary onboarding path.
4. Preserve one-brain guarantees across all surfaces.
