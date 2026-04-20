# PM_MEMORY

Canonical project-management memory for parallel AI contributors.

## Mission Lock

- Ship a general personal AI assistant that is useful in daily workflows.
- Preserve local-first privacy, explicit user control, and approval gating.
- Prioritize usefulness, reliability, and clarity before capability sprawl.

## Active Priorities

1. P0 usefulness stabilization on ordinary chat paths.
2. Memory continuity reliability across turns and sessions.
3. Model arbitration clarity for quality-cost tradeoffs.

## Delegation Protocol

- Every delegated task must return a compact handoff with:
  - scope
  - changed files
  - tests run
  - risks
  - blockers
- Do not write to this file from a delegated run until tests for that scope pass.
- If a delegated attempt fails, record the failure in `WORKLOG.md` only.
- Merge only after:
  - local tests for touched scope pass
  - no obvious regressions in routing, memory, or policy surfaces

## Decision Ledger

### 2026-04-19: Recovery strategy

- Choose targeted fixes first, not architecture replacement.
- Deep rebuild is allowed only when an issue is shown to be structural.
- First hard milestone is useful everyday conversation quality.

### 2026-04-19: P0 guardrails

- Do not hide provider-health issues behind optimistic status overrides.
- Keep degraded providers eligible with penalty rather than hard blocking.
- Preserve safe partial helpful output instead of collapsing to terse generic replies.

## Risk Register

- Split-brain behavior between deterministic and semantic memory paths.
- Drift between model-routing controls and effective selector behavior.
- Retry/fallback behavior can silently drop constraints if not logged.

## Exit Criteria Snapshot

- Fewer generic fallback replies on basic prompts.
- Lower routing contradiction rate.
- Memory context appears predictably on follow-up turns.
- Selection traces explain why a model was or was not chosen.
