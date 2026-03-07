# Product Runtime Spec (v1 Canonical)

This document is the canonical product/runtime source of truth for v1.
If implementation or other docs conflict with this file, this file wins.

## 1) Product Purpose
- Personal Agent is a local-first assistant.
- Primary v1 purpose: manage the user's PC health.
- Phase 1 is read-only diagnostics, reporting, and guidance.
- Natural language is the primary interaction model.
- Skill packs extend capability later under explicit approval.

## 2) Core Principles
- One brain.
- One source of truth.
- Local-first.
- Read-only first.
- Deterministic recovery.
- Truthful identity.
- Minimal background processes.
- Transport surfaces are not separate assistants.

## 3) Runtime Model
- One main runtime/service owns core behavior:
  - orchestrator
  - runtime contract
  - onboarding/recovery
  - tool execution
  - memory/continuity
  - LLM routing
  - skill pack loading
- Surfaces (native UI, CLI, optional Telegram) call into the same runtime.
- Telegram is an optional adapter surface, not an independent decision-making layer.

## 4) Process Model
- Default install runs one main service.
- Optional adapters may run separately only as thin transport shells.
- Telegram is opt-in (`TELEGRAM_ENABLED=1`), and disabled by default.
- Business logic must not be duplicated across processes.
- Split-brain behavior is not allowed.

## 5) Bootstrap / Setup Model
- User manually installs Ollama.
- User launches Personal Agent.
- Agent detects readiness/setup state.
- Native UI is the primary setup/recovery guidance path.
- If LLM is unavailable, agent returns one deterministic next action.
- Telegram setup guidance is a mirror of runtime state, not a separate bootstrap brain.

## 6) Recovery Model
- Degraded/down states return one exact next action.
- Recovery logic comes from shared contracts, not transport-specific branches.
- Read-only tools remain available when safe.

## 7) Capability Growth Model
- Core runtime remains focused on PC health in v1.
- The core runtime also owns the LLM control plane for approved/local models:
  - inventory
  - capability classification
  - health probing
  - model selection
  - approved install planning
  - approved local model profiles used for deterministic Ollama recommendations
  - explicit approved local Ollama install execution behind operator approval
- Skill packs add additional capabilities later.
- Skill packs require explicit approval/registry policy.
- Future self-generated skills must still follow approval/registry controls.
- Skill packs cannot take ownership of core runtime behavior.

## 8) Surface Model
- Native UI: primary end-user experience.
- CLI: operator/developer surface.
- Telegram: optional transport surface.
- Future surfaces must remain transport-only and reuse core runtime contracts.

## 9) Out Of Scope For v1
- Competing with frontier general chat assistants.
- Multi-service sprawl by default.
- Uncontrolled autonomy.
- Arbitrary code generation/execution without approval.

## 10) Migration Plan
- Phase A: declare and adopt this canonical product/runtime model.
- Phase B: thin transport adapters (Telegram/UI/CLI call shared runtime interfaces).
- Phase C: unify native UI bootstrap/recovery as primary onboarding path.
- Phase D: expand capability via approved skill-pack extension flow.
