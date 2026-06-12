# Core Workflow Proof

Date: 2026-06-11
Checkpoint before proof pass: `019ac0c` Persist journals for pack lifecycle

This proof is an operator acceptance check for the user-facing promise of
Personal Agent. It is intentionally stricter than the normal unit and safety
gates: a workflow may be `BLOCKED` or `NOT_PROVEN` without being called a
failure, but it must not be reported as `PASS` unless the script observed a real
runtime path and state change.

Run:

```bash
python scripts/prove_core_workflows.py
```

The script prints a plain-English report with the command or API path used, the
state or artifact that changed, intentionally blocked behavior, remaining
unproven claims, and the smallest next fix for each blocker.

## Latest Result

Status at this checkpoint:

| Workflow | Result | Meaning |
| --- | --- | --- |
| External skill pack lifecycle | PASS | A deterministic approved local source was created, searched, previewed, imported through quarantine/normalization, reviewed without exposing hostile fixture text, approved, enabled, granted one metadata-only adapter permission, invoked through a harmless core-owned adapter dry-run, disabled, removed, tombstoned, and cleaned up. |
| Missing capability flow | PASS | Approved-source search, preview availability, and the public chat response were proven. The assistant names the candidate, proposes the preview/install/review path, asks for confirmation before mutation, and does not claim installation or use. |
| Internet/search status | BLOCKED | No trusted SearXNG/search backend was configured in the proof environment. The script reports search disabled instead of claiming search works. |
| Model scout/provider behavior | PASS | The provider boundary documentation and public chat guidance were checked. The assistant distinguishes Ollama, OpenAI-compatible local endpoints, llama.cpp server/LM Studio/vLLM through user-run OpenAI-compatible endpoints, and direct llama.cpp management as absent. |
| Release gates still pass | NOT_PROVEN inside script | `external_pack_safety_smoke`, docs tests, `git diff --check`, and `git status --short` run from the script. The behavior/release pytest group must be run directly after the proof and treated as authoritative. |

## What Was Proven

- Safe local external pack discovery from an approved local catalog source.
- Candidate preview before mutation.
- Local pack import through the existing quarantine/normalization path.
- Review metadata exists and hostile fixture text is not exposed in review/status
  payloads checked by the proof.
- Review approval and enablement are separate state changes.
- Permission grant is separate from enablement.
- A harmless imported pack action can run only through a core-owned managed
  adapter dry-run after the metadata-only permission grant.
- Disable/remove leaves a tombstone/removal record and removes the active pack
  record without exposing hostile fixture text.
- Missing-capability public chat checks approved pack sources and offers the
  confirmation-gated preview/install/review path without claiming install or
  use.
- Model/provider public chat guidance is grounded in the current provider
  boundary: Ollama optional, OpenAI-compatible local servers supported, direct
  llama.cpp binary/library management absent, and llama.cpp server/LM Studio/vLLM
  usable only through user-run OpenAI-compatible endpoints.

## What Remains Blocked Or Unproven

- Real internet/search is not proven until `SEARXNG_BASE_URL` points at a
  trusted reachable SearXNG instance.
- The behavior/release pytest group is intentionally run directly outside the
  proof harness:

```bash
python -m pytest -q tests/test_chat_behavior_audit.py tests/test_live_user_barrage.py tests/test_assistant_behavior_release_gate.py
```

## Smallest Next Fixes

1. Configure `SEARXNG_BASE_URL` for a trusted SearXNG instance, then rerun the
   proof to turn internet/search from `BLOCKED` into an observed real query.
2. Keep running the behavior/release pytest group directly after the proof until
   the proof harness can run it without introducing test-environment drift.

## Guardrails

- The proof must not use arbitrary remote packs.
- The proof must not execute foreign code.
- The proof must not auto-enable arbitrary packs or auto-grant permissions.
- The proof must not persist hostile pack text, raw pack contents, secrets,
  tokens, private text, prompts, or provider response bodies.
- The proof must not claim direct llama.cpp support unless a direct runtime
  adapter exists. Current support is documented in
  `docs/operator/LOCAL_MODEL_PROVIDER_SUPPORT.md`.
