# Stability Guarantees (v0.6.x)

This document lists the current, shipped invariants for v0.6.x. It is factual and intentionally conservative.

## Invariants / Guarantees
- Read-only by default.
- Writes require `ENABLE_WRITES=true` and a double confirmation flow.
- Audit hard-fail: if audit logging fails, the operation aborts.
- Execution modes are limited to: `off`, `sandbox`, `live`.
- No sudo usage.
- Telegram is treated as untrusted input.
- Storage governor is observe-only.

## Non-Goals / Out of Scope (v0.6.x)
- No autonomous actions.
- No cleanup or deletion actions.
- No advice or recommendation language.
- No scope expansion of permissions or capabilities.
- No autonomy unlocks or background decision-making.
