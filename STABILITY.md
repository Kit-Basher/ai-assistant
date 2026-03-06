# Stability Guarantees (Current Branch)

This file captures practical guarantees that should remain stable unless intentionally changed and documented.

## Core Guarantees

- Local-first operation: core runtime works with local DB and local config paths.
- Telegram and API input are treated as untrusted.
- ModelOps execution is constrained to explicit, whitelisted model-management actions.
- Permission defaults are restrictive (`manual_confirm` mode with allow/deny action map).
- Sensitive operational paths are audited with append-only style records.
- Systemd-specific doctor checks are environment-aware:
  - local/dev default: missing units are skipped
  - enforce mode: `AGENT_DOCTOR_REQUIRE_SYSTEMD_UNITS=1`
- Schema/version compatibility is validated by doctor checks (`VERSION` vs `schema_meta`).
- Deterministic doctor output ordering and stable JSON/text field names.
- No silent token leakage in logs/doctor/CLI output (redaction by default).
- One canonical help text per user surface (Telegram and CLI paths).
- Truthful identity responses via centralized identity helper.
- Telegram safe-send fallback: truncate, retry plain text on BadRequest, and emit `telegram.out` after successful delivery.
- Unified runtime contract across user surfaces (`agent/runtime_contract.py`) with stable mode names:
  - `READY`
  - `BOOTSTRAP_REQUIRED`
  - `DEGRADED`
  - `FAILED`
- One deterministic next action for degraded/failed user-facing states.
- One deterministic LLM tool-use schema (`agent/tool_contract.py`) for agent actions.
- One execution gate (`agent/tool_executor.py`) and one permission decision helper (`agent/permission_contract.py`) for tool execution semantics.
- Stable structured tool logs:
  - `tool.request`
  - `tool.decision`
  - `tool.execute`
  - `tool.result`
- Continuity guarantees (`agent/memory_contract.py` + `agent/memory_runtime.py`):
  - deterministic thread-state and pending-item normalization
  - pending expiry is explicit (never silently assumed active)
  - follow-up binding requires exactly one valid pending item in-thread
  - ambiguous follow-ups are refused deterministically (no thread mixing)

## Compatibility Expectations

- Existing API endpoints should remain backward-compatible unless versioned or explicitly documented.
- Existing Telegram command names should remain stable unless migration notes are added.
- DB schema changes should be additive/migrated and covered by tests.
- Operator workflows should remain discoverable through `python -m agent` subcommands.

## Explicit Non-Goals

- No unrestricted arbitrary shell execution from chat input.
- No hidden autonomous permission expansion.
- No silent remote dependency requirement for local baseline operation.
