# Release Ledger

Date: 2026-06-15

This ledger is the compact release-history index for the current Personal Agent
line. It does not replace detailed design, operator, or checkpoint documents.

## Current Stable Checkpoint

Current stable checkpoint: `v0.2.1`

Commit: `f900954` Close v0.2.1 lifecycle and release hardening

Summary:

- Host Lifecycle Runner v1, primary update, verified rollback, preserve-data
  uninstall wiring, strict local uninstall activation, backup/restore/cleanup,
  runtime truth, receipts, and full sequential installed-product proof are
  complete.
- Primary uninstall remains disabled unless a local operator creates a valid
  activation marker. Purge remains unsupported.
- Release warnings are zero at the v0.2.1 checkpoint.
- The next phase is Tool Authorization and Plan Mode Maturity. Capability
  Policy Schema and Central Authorization Gate v1, Universal Plan Mode
  Enforcement v1, Executor Authorization Migration v1, Files/Git/Service
  Migration v1, Communications Migration v1, and Skill-Pack Permission
  Boundary v1, and Generic Mutation Bypass Hardening v1 are complete. Full
  Adversarial Authorization Proof v1 is the active checkpoint.

## Recent Release Tags

| Tag | Date | Commit | What it records |
| --- | --- | --- | --- |
| `v0.2.1` | 2026-07-10 | `f900954` | Lifecycle and release-hardening closure. |
| `v0.2.2-capability-policy-schema-v1` | 2026-07-10 | `514aef3` | Capability Policy Schema v1 and central authorization gate foundation. |
| `v0.2.2-universal-plan-mode-v1` | 2026-07-10 | `401b6e5` | Universal Mutation Plan v1 and shared confirmation/receipt contract. |
| `v0.2.2-executor-authorization-migration-v1` | 2026-07-10 | `e5e097b` | Backup v1, Restore v1, support-bundle, and memory lifecycle authorization migration. |
| `v0.2.2-files-git-service-migration-v1` | 2026-07-11 | `240bcd1` | Bounded file, Git, and service-control authorization migration. |
| `v0.2.2-communications-migration-v1` | 2026-07-11 | `24bfca5` | Implemented notification communications migration. |
| `v0.2.2-skill-pack-permission-boundary-v1` | 2026-07-12 | `71e25f3` | Skill-pack platform API permission boundary. |
| `v0.2.2-generic-mutation-bypass-hardening-v1` | 2026-07-12 | `b56a449` | Repository-wide generic mutation bypass hardening. |
| `v0.2.2-full-adversarial-authorization-proof-v1` | not cut | pending | Recommended checkpoint for end-to-end adversarial authorization proof. |
| `v0.2.0-managed-searxng` | 2026-06-14 | `f26ba6f` | Managed local SearXNG safe web search. |
| `v0.2.0-plan-mode-policy` | 2026-06-14 | `e88281b` | Central Plan Mode policy layer. |
| `v0.2.0-plan-mode-pack-lifecycle` | 2026-06-14 | `7096852` | Plan Mode enforcement for external pack lifecycle writes. |
| `v0.2.0-plan-mode-user-confirmation-ux` | 2026-06-15 | `d699ef1` | User-facing Plan Mode confirmation UX for managed SearXNG and external pack lifecycle. |
| `v0.2.0-release-proof-surfaces` | 2026-06-14 | `48ce105` | Release proof surface consistency for `/ready`, doctor, optional Telegram, and dev pack-state interpretation. |
| `v0.2.0-live-usefulness-proof` | 2026-06-15 | `e5dc9f8` | Live managed SearXNG search and external text-pack usefulness proof. |

## Full Tag List

Dates are from `git for-each-ref --sort=creatordate`.

| Tag | Date |
| --- | --- |
| `v0.1.0` | 2026-02-03 |
| `v0.2-safe-disk-hygiene` | 2026-02-04 |
| `v0.3-safe-home-cache` | 2026-02-04 |
| `v0.4-observability` | 2026-02-04 |
| `v0.5-sandbox-runner` | 2026-02-04 |
| `v0.5-sandbox-runner-audited` | 2026-02-04 |
| `v0.6.1` | 2026-02-04 |
| `v0.8.0` | 2026-02-04 |
| `v0.9.0` | 2026-02-04 |
| `v1.0.0` | 2026-02-04 |
| `v1.2.0` | 2026-02-04 |
| `v1.2.1` | 2026-02-04 |
| `canonical-tracking-2026-02-04` | 2026-02-04 |
| `conversation-continuity-v1` | 2026-02-05 |
| `v0.2.0` | 2026-02-12 |
| `v0.2.1-epistemics-phase1` | 2026-02-14 |
| `v0.2.1-good-assistant-foundation` | 2026-02-14 |
| `v0.2.2-pathB-continuity` | 2026-02-14 |
| `v0.2.3-thread-navigation` | 2026-02-14 |
| `v0.2.4-thread-workflow` | 2026-02-14 |
| `v0.2.5-project-mode` | 2026-02-14 |
| `v0.2.6-memory-graph` | 2026-02-14 |
| `v0.2.7-graph-packs` | 2026-02-14 |
| `v0.2.8-graph-queries` | 2026-02-14 |
| `v0.2.9-typed-relations` | 2026-02-14 |
| `v0.3.0-memory-constraints` | 2026-02-14 |
| `v0.3.0-autopilot` | 2026-02-18 |
| `v0.3.1-hardening` | 2026-02-18 |
| `v0.4.1-polish` | 2026-02-19 |
| `v0.4.2-docs-polish` | 2026-02-19 |
| `v0.4.3-ops-polish` | 2026-02-19 |
| `v0.5.0-bounded-autonomy` | 2026-02-19 |
| `v0.3.0` | 2026-04-08 |
| `v1.0.0-rc1` | 2026-04-11 |
| `v0.4.0` | 2026-04-12 |
| `v0.2.0-managed-searxng` | 2026-06-14 |
| `v0.2.0-plan-mode-policy` | 2026-06-14 |
| `v0.2.0-plan-mode-pack-lifecycle` | 2026-06-14 |
| `v0.2.0-plan-mode-user-confirmation-ux` | 2026-06-15 |
| `v0.2.0-release-proof-surfaces` | 2026-06-14 |
| `v0.2.0-live-usefulness-proof` | 2026-06-15 |

## What Was Proven

### `v0.2.0-managed-searxng`

Proven:

- Rootless Podman is the preferred managed-service engine on Linux.
- Managed SearXNG binds to loopback only.
- The approved image is `docker.io/searxng/searxng:latest`.
- The managed config is seeded before mount, enables JSON metadata output, and
  uses a generated or preserved non-default `server.secret_key`.
- Search remains metadata-only: no page fetching, browser automation,
  downloads, or pack install/import from search results.
- Setup verifies health and provider-style JSON search before enabling search.
- Setup rolls back only owned SearXNG resources on failure.
- Approved existing containers can be reused or repaired only after image, bind,
  and mount checks.
- Config ownership failures produce a bounded operator handoff rather than
  hidden sudo from the API service.

### `v0.2.0-plan-mode-policy`

Proven:

- `agent/policy.py` centrally classifies read-only versus mutating operations.
- Read/list/search/status/preview operations can run without confirmation.
- Known mutators require plan, preview, confirmation token, expiry, and apply
  validation.
- Unknown operations default to mutating.
- Managed SearXNG setup embeds `mutation_plan` in `/search/setup/plan` and
  validates it in `/search/setup/apply`.

### `v0.2.0-plan-mode-pack-lifecycle`

Proven:

- External pack lifecycle writes expose Plan Mode plan/apply flows:
  install, approve, enable, grant, remove/tombstone.
- Direct lifecycle write endpoints return confirmation-required responses
  instead of mutating.
- Apply rejects missing mutation plans, expired plans, tampered action types,
  tampered resources, tampered tokens, and plan/apply mismatches.
- Preview/list/search/status pack surfaces remain read-only.
- Existing external pack safety gates still pass.

### `v0.2.0-plan-mode-user-confirmation-ux`

Proven:

- Managed SearXNG setup chat UX presents the existing Plan Mode setup plan
  naturally: resources that may be created/changed/deleted, rollback scope,
  rollback support, expiry, and explicit yes/no confirmation.
- External pack lifecycle user-facing paths use the existing Plan Mode
  plan/apply payloads for install/import-for-review, approve, enable, grant
  metadata, and remove/tombstone where routed through chat/API.
- The assistant preserves `plan_id`, `confirmation_token`, and `mutation_plan`
  through confirmation and applies only the matching pending plan.
- Declined, expired, consumed, missing, mismatched, or tampered confirmations
  block cleanly without pack use, service start, image pull, permission grant,
  or search enablement.
- Telegram rendering remains concise and does not dump raw JSON,
  confirmation tokens, raw hostile pack text, raw `SKILL.md`, or raw `AGENTS.md`
  content.

### `v0.2.0-release-proof-surfaces`

Proven:

- `/ready` reports no recovery mode when the runtime is `READY` with no
  failure, blocker, or reason.
- Optional inactive Telegram is not treated as required recovery on `/ready`.
- `python -m agent doctor` exits OK when optional inactive Telegram is the only
  inactive surface.
- Required/enabled Telegram failures remain visible when Telegram is expected to
  run.
- Development `/packs/state` may include old blocked smoke-test packs; this is
  local dev state, not a release proof failure. Cleanup must use confirmed
  remove/tombstone or a fresh state directory for final proof.

### `v0.2.0-live-usefulness-proof`

Proven:

- Live `/search/status` started disabled/unconfigured, then managed SearXNG was
  previewed and applied through the existing Plan Mode setup path.
- After confirmation, live `/search/status` reported `enabled=true`,
  `provider=searxng`, `endpoint_configured=true`, `available=true`, and a
  loopback base URL.
- Live `/search/query` returned untrusted metadata-only results and did not
  fetch pages, run JavaScript, download files, or import packs.
- Live `/chat` used `safe_web_search` for an explicit web-search request and
  returned metadata result titles with the same safety limits.
- The starter `Linux Troubleshooting Workflow` text pack was previewed,
  imported through Plan Mode, approved, enabled, and reported by `/packs/state`
  as installed, enabled, healthy, and usable.
- Live `/chat` used the installed text-only pack in a normal Linux
  troubleshooting request and stated that it did not run commands, execute pack
  code, read files, use the network, or change system state.

## Deferred

- Fresh Debian VM install test is intentionally deferred until the very end
  because it is high-cost and time-consuming.
- Do not claim broad Green release readiness before the final clean install,
  first-run setup, launch, proof, rollback/uninstall, and live verification pass.
- Startup auto-recovery for persistent managed-action journals remains deferred.
- Semantic memory remains release-gated and off by default.
- Package install/directory creation shell flows must not become normal
  assistant actions without a dedicated Plan Mode and managed-action design.

## Verification Command Groups

Focused docs and policy:

```bash
python -m py_compile agent/api_server.py agent/policy.py
python -m pytest -q tests/test_plan_policy.py tests/test_api_packs_endpoints.py tests/test_project_intent_docs.py
```

Managed local services and safe search:

```bash
python -m pytest -q tests/test_managed_local_services.py tests/test_safe_web_search.py
```

External pack safety:

```bash
python scripts/external_pack_safety_smoke.py
```

Core workflow proof:

```bash
python scripts/prove_core_workflows.py
```

Behavior/release gates:

```bash
python -m pytest -q tests/test_chat_behavior_audit.py tests/test_live_user_barrage.py tests/test_assistant_behavior_release_gate.py
```

Recent checkpoint verification:

```bash
python -m py_compile agent/api_server.py agent/policy.py agent/bot.py
# PASS

python -m pytest -q tests/test_plan_policy.py tests/test_chat_behavior_audit.py tests/test_live_user_barrage.py tests/test_assistant_behavior_release_gate.py
# 58 passed

python -m pytest -q tests/test_managed_local_services.py tests/test_safe_web_search.py tests/test_api_packs_endpoints.py
# 95 passed

python scripts/external_pack_safety_smoke.py
# PASS external_pack_safety_smoke (39 gates)

python scripts/prove_core_workflows.py
# PASS: external skill pack lifecycle, missing capability flow, model scout/provider behavior
# BLOCKED: internet/search in isolated proof environment when no trusted SearXNG backend is configured
# FAIL: none

python scripts/release_smoke.py
# PASS after updating stale smoke assertions to current Plan Mode and model-router behavior

git diff --check
# PASS
```

Release proof surface checkpoint verification:

```bash
python -m py_compile agent/api_server.py agent/policy.py agent/bot.py
# PASS

python -m py_compile agent/runtime_truth_service.py agent/doctor.py
# PASS

python -m pytest -q tests/test_ready_endpoint.py tests/test_doctor_cli.py
# 42 passed

python -m pytest -q tests/test_chat_behavior_audit.py tests/test_assistant_behavior_release_gate.py tests/test_first_run_release_smoke.py tests/test_publishability_smoke.py
# 27 passed

python scripts/release_smoke.py
# 52 passed

python -m agent doctor
# OK

git diff --check
# PASS
```

Repository hygiene:

```bash
git diff --check
git status
```

Final high-cost release proof, intentionally deferred:

```bash
# Run only at the end of the release track.
# Fresh Debian VM install, first launch, setup completion, proof, rollback, and uninstall.
```

## Search Proof Note

`scripts/prove_core_workflows.py` may report internet/search as `BLOCKED` in an
isolated proof environment when no trusted SearXNG backend is configured. That
is expected and must not be converted into a fake PASS.

For the live workstation, `/search/status` is the authoritative runtime check
for managed SearXNG. A live PASS requires `enabled=true`, `provider=searxng`,
`endpoint_configured=true`, `available=true`, a loopback redacted base URL, and
successful metadata-only `/search/query` behavior.
