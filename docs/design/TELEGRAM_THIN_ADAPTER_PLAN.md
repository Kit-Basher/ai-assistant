# Telegram Thin Adapter Plan

Canonical target is defined in [PRODUCT_RUNTIME_SPEC.md](/home/c/personal-agent/PRODUCT_RUNTIME_SPEC.md).

## Objective
Keep Telegram as a transport adapter, not a second brain.

## What Stays In Telegram Adapter
- Telegram SDK integration and polling/webhook mechanics.
- Message receive/send transport code.
- Safe-send protections (length cap, parse fallback, retry).
- Transport-scoped logging/audit wrappers.
- Minimal deterministic input normalization.

## What Must Move To Core Runtime
- Setup/onboarding state decisions.
- Recovery decision and next-action selection.
- Runtime status truth construction.
- Doctor/status/help/memory semantic decisions.
- Continuity mutation policy.
- Tool-execution and permission decisions.

## Canonical Interfaces Telegram Should Call
- Runtime status/readiness summary interface.
- Setup/onboarding summary interface.
- Recovery summary interface.
- Doctor summary interface.
- Memory/continuity summary interface.
- Shared deterministic error envelope formatter.

## Current Extraction Step
- Implemented bridge: `agent/telegram_bridge.py`.
- Canonical text/command UX (`help/setup/status/health/doctor/brief/memory`) now routes through the bridge.
- Transport send/retry/lock/polling remains in `telegram_adapter/bot.py`.

## Runtime Unavailable Behavior
- Telegram should return a deterministic transport error block:
  - short failure statement
  - trace id
  - component
  - one next action (`python -m agent doctor` or restart command)
- No speculative setup logic in adapter when runtime is unavailable.

## Process Isolation Note
- A separate Telegram process can be acceptable for transport isolation.
- It is not acceptable as a second business-logic owner.
- If separate process is used, it must call canonical runtime APIs/contracts only.

## Why Duplication Is Not Acceptable
- Divergent setup/recovery answers confuse users.
- Split-brain behavior creates false readiness and contradictory guidance.
- Contract drift increases operational risk and support load.

## Not Now (Explicitly Deferred)
- No large Telegram runtime rewrite in this pass.
- No protocol redesign in this pass.
- No endpoint deprecations in this pass.
- No broad behavior refactor beyond contract alignment work.
