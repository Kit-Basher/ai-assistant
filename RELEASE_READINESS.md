# RELEASE_READINESS.md

## Purpose

This file is the canonical release-readiness hardening plan for the product.

The system is now treated as functionally core-complete enough for productization.  
This phase is about making it reliable, understandable, recoverable, testable, and shippable.

This file is a work order, progress tracker, and release gate reference.

---

## Status

- [x] 1. Install / first-run / upgrade hardening
- [x] 2. Packaging cleanup
- [x] 3. Service / runtime hardening
- [x] 4. Memory hardening
- [x] 5. Error envelope / wording / idiotproofing pass
- [x] 6. Diagnostics / recovery / backup
- [x] 7. Release-gate smoke suite
- [x] 8. Docs / publishability pass

---

## Execution Rules

Codex must treat this file as the single source of truth for the release-readiness pass.

### Required rules

- Work one section at a time.
- Do not skip ahead unless the current section is blocked and the blocker is documented here.
- Never mark a task complete without implementing the code and adding or updating tests where applicable.
- Keep changes minimal, scoped, and deterministic.
- Do not add new major subsystems or broaden trust/execution boundaries.
- Prefer one canonical path for each common task.
- Prefer explicit states/enums over vague booleans.
- Prefer removing confusing options over documenting them.
- Prefer safe refusal over partial undefined behavior.
- Prefer boring, predictable behavior over clever but fragile behavior.
- Docs must match actual behavior exactly.
- If something is unsupported, say so explicitly in code paths and docs.
- If uncertain, choose fail-closed behavior and document the reason.

### For each section, Codex must

1. Implement the required changes
2. Add or update tests
3. Update docs if behavior or operator flow changed
4. Update the checklist in this file
5. Add a short completion note under that section summarizing:
   - files changed
   - exact behavior changed
   - tests added/updated
   - any follow-up intentionally deferred

---

## Design Invariants

These invariants must remain true throughout the release pass:

1. There is one canonical path for each common user/operator task.
2. Unsafe, invalid, unknown, or ambiguous states fail closed.
3. Degraded, blocked, and partial states are explicit and surfaced clearly.
4. User-facing wording is plain, direct, and actionable.
5. Operator controls remain explicit and auditable.
6. Version, config, and runtime truth each have clear canonical sources.
7. Fresh install and upgrade are both first-class supported paths.
8. Memory and pack behavior are understandable, inspectable, and recoverable.
9. Docs describe the actual shipping product, not aspirations.

---

## Non-Goals

The following are explicitly out of scope for this release-readiness pass:

- New major feature families
- Broader third-party execution support
- Expansion of trust boundaries
- Speculative plugin/runtime architecture work
- UI redesign unless required to remove a release blocker
- Product-scope expansion disguised as “cleanup”

---

# 1. Install / First-Run / Upgrade Hardening

## Goal

Make installation, first-run, recovery, reinstall, and upgrade predictable and easy to follow without guesswork.

## Required outcomes

- One canonical install path
- One canonical upgrade path
- One canonical reset/recovery path
- One canonical config location
- One canonical data location
- One canonical service-management story
- Safe first-run defaults
- No required manual file edits for common setup

## Tasks

- [x] Audit the current install flow end to end
- [x] Ensure there is one documented canonical install path
- [x] Ensure there is one documented canonical upgrade path
- [x] Ensure first-run setup is linear and obvious
- [x] Validate behavior for missing directories
- [x] Validate behavior for missing config
- [x] Validate behavior for invalid config
- [x] Validate behavior for corrupted config
- [x] Validate behavior for missing secrets
- [x] Validate behavior for partially initialized state
- [x] Add safe reset/recovery handling where missing
- [x] Ensure reinstall over an existing install behaves predictably
- [x] Ensure uninstall behavior is documented and predictable
- [x] Ensure restart survives reboot cleanly

## Definition of done

- A fresh machine can be brought to a working state from docs alone
- An existing install can be upgraded without manual surgery
- Broken config/state produces actionable errors and recovery paths
- A user/operator can identify install, upgrade, reset, and service-management paths without ambiguity

## Completion notes

- Canonical install story is now the user-level systemd path from `~/personal-agent`
  with mutable state under `~/.local/share/personal-agent` and operator config
  under `~/.config/personal-agent`.
- `python -m agent setup` and `python -m agent doctor --fix` are the canonical
  first-run and recovery path; doctor now repairs missing local dirs and copies
  legacy repo-local db/log state into the canonical state dir.
- Legacy root/system-service wrapper scripts now fail closed and point to the
  supported docs instead of presenting a second install path.

---

# 2. Packaging Cleanup

## Goal

Make build, install, CLI entry points, version metadata, and distribution story production-ready and unambiguous.

## Required outcomes

- Packaging metadata is complete and accurate
- Build path is canonical
- CLI install path is stable
- Release artifacts build cleanly
- Version/build metadata has one source of truth
- Debian/system packaging support is either real and tested or explicitly out of scope

## Tasks

- [x] Audit current packaging metadata
- [x] Ensure `pyproject.toml` is complete and production-ready
- [x] Ensure build backend and build path are canonical
- [x] Ensure console entry points / CLI install path are stable
- [x] Ensure wheel/sdist build cleanly
- [x] Ensure release artifact naming/versioning are deterministic
- [x] Ensure version/build info is surfaced consistently in CLI/API/UI
- [x] Decide whether Debian/system packaging is supported for this release
- [x] If supported, make it explicit and testable
- [x] If not supported, document that clearly and remove misleading wording
- [x] Ensure service unit files are shipped, generated, or documented cleanly
- [x] Ensure default directories/permissions are predictable
- [x] Ensure changelog / release notes flow exists or is explicitly deferred with rationale

## Definition of done

- Building the release artifact is deterministic and documented
- Installing the artifact yields the expected CLI/service behavior
- Version/build truth is consistent everywhere it appears
- Packaging claims in docs exactly match reality

## Completion notes

- Files changed:
  - `pyproject.toml`
  - `build_backend.py`
  - `scripts/build_dist.py`
  - `requirements.txt`
  - `agent/bootstrap/routes.py`
  - `agent/bootstrap/snapshot.py`
  - `agent/version.py`
  - `agent/__init__.py`
  - `agent/cli.py`
  - `agent/api_server.py`
  - `telegram_adapter/bot.py`
  - `tools/dump_routes.py`
  - `tests/test_packaging_build.py`
  - `tests/test_agent_cli.py`
  - `tests/test_api_server.py`
  - `README.md`
  - `docs/operator/SETUP.md`
  - `PRODUCT_RUNTIME_SPEC.md`
  - `PROJECT_STATUS.md`
- Exact behavior changed:
  - packaging metadata is now canonicalized in `pyproject.toml`
  - `VERSION` is now the canonical version source, consumed through `agent/version.py`
  - CLI, API, and the served web UI HTML shell now surface the same version/build truth
  - repo install/update now uses `pip install -e .`
  - release artifact build now uses `python scripts/build_dist.py --outdir dist --clean`
  - deterministic release artifacts now build as `personal_agent-<version>-py3-none-any.whl` and `personal_agent-<version>.tar.gz`
  - packaged console entry points are now `personal-agent`, `personal-agent-api`, and `personal-agent-telegram`
  - packaged runtime bootstrap no longer depends on repo-only `tools/` imports; route extraction now lives under the shipped `agent.bootstrap` package
  - `personal-agent-api --help` and `personal-agent-telegram --help` now start cleanly from a fresh wheel install
  - Debian/system packaging is explicitly unsupported for this release; legacy `packaging/` artifacts are not the shipping path
- Tests added/updated:
  - added `tests/test_packaging_build.py`
  - updated `tests/test_agent_cli.py`
  - updated `tests/test_api_server.py`
  - updated packaging validation to install a fresh wheel in a clean venv and run the packaged entry points directly
- Intentionally deferred:
  - Debian/system packaging remains out of scope because the supported runtime deployment is still the repo checkout plus user-systemd flow
  - a formal changelog/release-notes workflow remains deferred; release notes stay manual for this release rather than pretending there is an automated flow

---

# 3. Service / Runtime Hardening

## Goal

Ensure runtime behavior is predictable under startup, shutdown, restart, reboot, degraded dependency, and partial failure conditions.

## Required outcomes

- No hanging endpoints
- No silent degraded states
- Health/readiness/status are fast and consistent
- Startup/warmup/degraded/blocked states are explicit
- Partial subsystem failures degrade clearly
- Safe-mode / blocked-by-policy outcomes are first-class, not edge cases

## Tasks

- [x] Audit service startup/shutdown/restart behavior
- [x] Audit endpoint behavior for hangs/timeouts in degraded conditions
- [x] Ensure health/readiness/status surfaces are fast and consistent
- [x] Ensure startup/warmup transitions are explicit
- [x] Ensure degraded and blocked states are explicit
- [x] Verify clean behavior across process restart
- [x] Verify clean behavior across machine reboot
- [x] Verify clean behavior during temporary dependency absence
- [x] Verify clean behavior during partial subsystem failure
- [x] Tighten service defaults that create ambiguity or confusion
- [x] Ensure logs are useful without leaking secrets
- [x] Ensure blocked-by-policy results are surfaced clearly and consistently

## Definition of done

- Restart and reboot behavior are predictable
- No common endpoint silently hangs in normal degraded scenarios
- Degraded/blocked states are visible and understandable

## Completion notes

- Files changed:
  - `agent/api_server.py`
  - `tests/test_api_server.py`
  - `tests/test_runtime_lifecycle.py`
  - `README.md`
  - `PRODUCT_RUNTIME_SPEC.md`
  - `PROJECT_STATUS.md`
  - `docs/operator/LOOKHERE.md`
- Exact behavior changed:
  - `/health` now surfaces the same explicit lifecycle contract as `/ready` and `/runtime`, including `phase`, `startup_phase`, `runtime_mode`, `runtime_status`, `warmup_remaining`, `message`, compact Telegram state, safe-mode target state, control-mode policy state, and blocked-state summary
  - `/runtime` now includes the same explicit runtime status/control-mode/blocked-state contract instead of only the lighter aggregate snapshot
  - pre-router deferred-startup runtimes now return explicit warmup/degraded status from `/health` and `/runtime` instead of asserting when the router has not been warmed yet
  - startup/degraded policy blocking is now surfaced explicitly through a shared blocked-state payload rather than only indirectly through separate status fields
- Tests added/updated:
  - updated `tests/test_api_server.py`
  - updated `tests/test_runtime_lifecycle.py`
  - validated existing `tests/test_ready_endpoint.py` lifecycle/degraded overlays against the new shared status contract
- Intentionally deferred:
  - no service-manager redesign or shutdown-sequencing refactor was added in this pass
  - no new health probe families were added; the hardening is limited to making existing lifecycle truth explicit and non-crashing
- Runtime/operator truth surfaces remain aligned

---

# 4. Memory Hardening

## Goal

Make memory behavior explicit, inspectable, recoverable, and safe under enable/disable/reset/corruption/migration scenarios.

## Required outcomes

- Memory can be enabled/disabled cleanly
- Memory state is inspectable
- Memory erase/reset flow is explicit and safe
- Corrupt/missing/migration cases are tested
- User-facing explanation of memory is simple and accurate
- No hidden persistence behavior exists outside the documented model

## Tasks

- [x] Audit memory enable/disable behavior
- [x] Ensure memory state is inspectable through canonical surfaces
- [x] Ensure erase/reset flow is explicit and safe
- [x] Test corrupt memory state behavior
- [x] Test missing memory state behavior
- [x] Test migration path behavior
- [x] Ensure broken memory state does not poison the rest of the product
- [x] Audit user-facing wording about memory
- [x] Ensure documented memory behavior matches actual persistence behavior exactly
- [x] Ensure no spooky/hidden persistence behavior exists outside the documented model

## Definition of done

- A user can understand, inspect, and clear memory without ambiguity
- Memory failure does not cascade into whole-product failure
- Memory behavior is documented accurately and minimally

## Completion notes

- Files changed:
  - `memory/db.py`
  - `agent/memory_runtime.py`
  - `agent/memory_v2/storage.py`
  - `agent/semantic_memory/storage.py`
  - `agent/api_server.py`
  - `agent/conversation_memory.py`
  - `agent/memory_ingest.py`
  - `agent/orchestrator.py`
  - `tests/test_memory_hardening.py`
  - `tests/test_orchestrator.py`
  - `README.md`
  - `PROJECT_STATUS.md`
  - `PRODUCT_RUNTIME_SPEC.md`
  - `docs/operator/LOOKHERE.md`
- Exact behavior changed:
  - memory enable/disable state is now explicit through the canonical loopback-only `GET /memory/status` surface
  - memory erase/reset is now explicit through loopback-only `POST /memory/reset`, with preview first and reset only after `confirm=true`
  - continuity memory corruption is inspectable instead of silently masked
  - `/memory` now reports degraded continuity state plainly and avoids hidden repair when the continuity blob is corrupt
  - optional `memory_v2` and semantic memory failures now degrade clearly and are reflected in status/debug state instead of only being logged and skipped
  - memory reset is transactional and scoped to `continuity`, `memory_v2`, `semantic`, or `all`
- Tests added/updated:
  - added `tests/test_memory_hardening.py`
  - updated `tests/test_orchestrator.py`
  - validated focused memory runtime/orchestrator/api hardening with `20 passed`
- Intentionally deferred:
  - no new memory subsystem, embedding path, or semantic-memory feature expansion was added
  - no new user-facing chat commands were added beyond hardening the existing memory behavior and operator surfaces

---

# 5. Error Envelope / Wording / Idiotproofing Pass

## Goal

Make normal failures understandable and recoverable without requiring code-reading or log-diving.

## Required outcomes

- Error surfaces answer what happened, why, and what to do next
- Internal jargon is removed from normal user-facing paths
- Blocked/degraded/partial/unavailable states are explained simply
- Operator-only actions are clearly separated from user flows
- Common tasks have one obvious path

## Tasks

- [x] Audit top user-facing flows for confusing wording
- [x] Audit top operator-facing flows for confusing wording
- [x] Standardize errors to answer:
  - [x] what happened
  - [x] why it happened
  - [x] what to do next
- [x] Remove unnecessary internal jargon from normal user-facing responses
- [x] Ensure blocked-by-policy states are explained simply
- [x] Ensure partial-safe-import states are explained simply
- [x] Ensure degraded/unavailable states are explained simply
- [x] Ensure dangerous actions require explicit confirmation
- [x] Ensure operator-only actions are clearly separated from user actions
- [x] Ensure “it didn’t work” paths are intentionally designed, not accidental

## Definition of done

- A normal user can recover from common mistakes without reading code
- An operator can identify the next action quickly from surfaced state
- The product feels understandable instead of fragile

## Completion notes

Completed.

- Files changed:
  `agent/error_response_ux.py`, `agent/api_server.py`, `agent/runtime_truth_service.py`, `agent/packs/external_ingestion.py`, `agent/orchestrator.py`, `agent/llm/model_manager.py`, `tests/test_api_server.py`, `tests/test_api_pack_sources_endpoints.py`, `tests/test_api_packs_endpoints.py`, `tests/test_external_pack_ingestion.py`, `tests/test_orchestrator.py`
- Exact behavior changes:
  standardized existing deterministic error payloads around plain `message` + `why` + `next_action`; tightened operator-only and confirm-required responses; made blocked SAFE MODE install/switch flows explain what was blocked and what to do next; simplified blocked filesystem/shell wording; simplified pack blocked/partial-safe-import review summaries; kept dangerous actions on the existing explicit confirmation path.
- Tests added/updated:
  updated focused API, pack-ingestion, and orchestrator tests to assert actionable wording for operator-only, confirm-required, blocked-by-policy, partial-safe-import, blocked native-pack import, SAFE MODE install/switch blocks, and blocked filesystem/shell paths.
- Intentionally deferred:
  no new error framework, no new operator/user flows, no docs changes because this pass tightened wording and envelopes without changing the supported runtime/operator flow.

---

# 6. Diagnostics / Recovery / Backup

## Goal

Make support/debug/recovery flows product-grade rather than dev-only.

## Required outcomes

- `doctor` / self-check output is human-usable
- There is one canonical diagnostics collection path
- Backup/export/import story exists and is documented
- Failed upgrade/corrupted state recovery is explicit
- Logs remain useful without leaking secrets

## Tasks

- [x] Harden doctor/self-check flows
- [x] Add or improve one canonical “collect diagnostics” path
- [x] Ensure backup/export story is documented and testable
- [x] Ensure import/recovery story is documented and testable where supported
- [x] Document or implement recovery for failed upgrades
- [x] Document or implement recovery for corrupted state
- [x] Add tests for supportability/recovery flows where feasible
- [x] Add no-secrets-in-logs assertions where practical
- [x] Ensure diagnostics output is actionable for operators

## Definition of done

- Support/debug flows are usable without intimate repo knowledge
- Recovery paths are explicit and tested
- Logs and diagnostics help without exposing secrets

## Completion notes

Completed.

- Files changed:
  `agent/doctor.py`, `tests/test_doctor_cli.py`, `docs/operator/doctor.md`, `docs/operator/SETUP.md`, `docs/operator/LOOKHERE.md`
- Exact behavior changes:
  `python -m agent doctor --collect-diagnostics` is now the one explicit redacted support-bundle path; doctor bundles now include the doctor report, plain-text summary, startup/self-check snapshots, redacted local API snapshots when available, canonical paths, and backup/restore recovery guidance; text doctor output now includes per-check next steps for WARN/FAIL rows; `--fix` still produces a bundle after safe local fixes.
- Tests added/updated:
  added doctor CLI coverage for diagnostics bundle creation, redaction, recovery manifest content, fix-mode bundle generation, and more actionable text rendering; existing doctor/startup tests were rerun to keep self-check and recovery behavior stable.
- Intentionally deferred:
  no broad backup subsystem, no remote upload/share flow for diagnostics bundles, and no new recovery API surface beyond the existing canonical doctor/setup paths.

---

# 7. Release-Gate Smoke Suite

## Goal

Create one obvious pre-release validation path that checks the product is releasable.

## Required outcomes

- One canonical smoke command/script exists
- It is deterministic
- It is fast enough to run before release
- It covers the core product path and the main safety/recovery gates

## Tasks

- [x] Create one canonical release-readiness smoke command/script
- [x] Ensure it covers fresh install or clean-environment validation where feasible
- [x] Ensure it covers first run
- [x] Ensure it covers health/readiness/status
- [x] Ensure it covers chat/tool basic path
- [x] Ensure it covers memory basic path
- [x] Ensure it covers safe pack discovery/preview/install blocked-path behavior
- [x] Ensure it covers corrupt config handling
- [x] Ensure it covers no-secrets-in-logs sanity checks
- [x] Ensure it covers restart/restart persistence where feasible
- [x] If reboot validation is too heavy, add a clearly named secondary release validation suite
- [x] Document exactly when to run the smoke suite and what a passing result means

## Definition of done

- There is one obvious command to run before calling a build releasable
- Passing the smoke suite has a clear meaning
- Heavy validations are separated cleanly if needed

## Completion notes

- Files changed:
  - `scripts/release_smoke.py`
  - `scripts/release_validation_extended.py`
  - `tests/test_release_smoke.py`
  - `tests/test_golden_path_smoke.py`
  - `README.md`
  - `PROJECT_STATUS.md`
  - `PRODUCT_RUNTIME_SPEC.md`
  - `RELEASE_READINESS.md`
- Exact behavior changed:
  - the canonical pre-release smoke command is now `python scripts/release_smoke.py`
  - the main smoke gate now runs one fixed deterministic set of fast pytest nodes covering core product flow, first-run/config safety, memory basic/degraded behavior, external pack safety, redacted diagnostics, and restart truth
  - a separate heavier follow-up path now exists at `python scripts/release_validation_extended.py` for slower pre-release checks without bloating the main gate
  - docs now state when to run the main smoke command and what a passing result means
- Tests added/updated:
  - added `tests/test_release_smoke.py`
  - updated `tests/test_golden_path_smoke.py` so the smoke assertions match the
    current CLI status contract and Telegram local-API bridge contract
- Intentionally deferred:
  - real reboot automation remains outside the main smoke gate; the extended suite stays test-driven instead of trying to script a machine reboot

---

# 8. Docs / Publishability Pass

## Goal

Make product docs describe the actual shipping product exactly, with no stale roadmap language or implied capabilities.

## Required outcomes

- README reflects the real product
- PROJECT_STATUS reflects current truth
- PRODUCT_RUNTIME_SPEC reflects current runtime truth and boundaries
- Install/operator docs match actual supported flows
- Out-of-scope items are stated clearly

## Tasks

- [x] Update `README.md`
- [x] Update `PROJECT_STATUS.md`
- [x] Update `PRODUCT_RUNTIME_SPEC.md`
- [x] Update install/operator docs as needed
- [x] Ensure docs include supported install path
- [x] Ensure docs include supported upgrade path
- [x] Ensure docs include service-management path
- [x] Ensure docs include memory behavior
- [x] Ensure docs include pack safety model
- [x] Ensure docs include diagnostics/recovery path
- [x] Ensure docs include what is explicitly out of scope for this release
- [x] Remove stale or aspirational wording
- [x] Remove wording that implies unsupported capabilities
- [x] Ensure docs describe the shipping product, not the roadmap

## Definition of done

- A new user can understand how to install and run the product from docs alone
- An operator can understand how to inspect, recover, and upgrade the product from docs alone
- The docs no longer oversell, drift, or imply unsupported flows

## Completion notes

- Files changed:
  - `README.md`
  - `PROJECT_STATUS.md`
  - `PRODUCT_RUNTIME_SPEC.md`
  - `docs/operator/SETUP.md`
  - `docs/operator/LOOKHERE.md`
  - `docs/operator/doctor.md`
  - `RELEASE_READINESS.md`
- Exact doc changes:
  - tightened the README around the supported install, upgrade, diagnostics,
    recovery, release-smoke, and out-of-scope story
  - clarified the runtime spec around the canonical diagnostics path, release
    gate usage, and explicit out-of-scope boundaries
  - removed roadmap-style wording from `PROJECT_STATUS.md` and replaced it with
    current boundaries that describe the shipping product as it exists today
  - updated operator docs so the canonical service-management, recovery,
    diagnostics, release-smoke, memory, and external-pack safety paths are easy
    to find from one place
- Intentionally deferred:
  - no new behavior or new operator APIs were added in this section
  - archived design notes were not rewritten; this pass only tightened the
    canonical product/operator docs

---

# Release Readiness Checklist

This checklist is the release gate summary.

## Product baseline

- [ ] One canonical install path
- [ ] One canonical upgrade path
- [ ] One canonical reset/recovery path
- [ ] One canonical config location
- [ ] One canonical data location
- [ ] One canonical service-management story
- [ ] Version/build info exposed clearly in CLI/API/UI
- [ ] Safe defaults on first boot
- [ ] No required manual file edits for common setup

## Fresh install

- [ ] Fresh machine install works from docs exactly
- [ ] First-run setup is linear and obvious
- [ ] Missing dependency errors are actionable
- [ ] Service starts cleanly after install
- [ ] Restart survives reboot
- [ ] Uninstall leaves system in a predictable state
- [ ] Reinstall over previous version is clean

## Configuration safety

- [ ] Invalid config fails closed with clear errors
- [ ] Corrupt config can be repaired or reset safely
- [ ] Missing files/directories are recreated safely
- [ ] Secret/config separation is explicit
- [ ] Sample config and real config do not drift
- [ ] One source of truth for version and config schema

## Runtime reliability

- [ ] No hanging endpoints
- [ ] No silent degraded states
- [ ] Health/readiness/status are fast and consistent
- [ ] Startup/warmup transitions are explicit
- [ ] Partial subsystem failures degrade clearly
- [ ] Safe-mode / blocked-by-policy outcomes are first-class
- [ ] Logs are useful without leaking secrets

## Memory readiness

- [ ] Memory can be enabled/disabled cleanly
- [ ] Memory state is inspectable
- [ ] Memory corruption/migration path is tested
- [ ] Clear user-facing explanation of memory
- [ ] Clear erase/reset flow
- [ ] No hidden persistence behavior

## Pack ecosystem safety

- [ ] Discovery is read-only
- [ ] Preview never installs
- [ ] Fetch always goes to quarantine
- [ ] Normalize/identity/diff/history work on release builds
- [ ] Unknown/ambiguous packs fail closed
- [ ] Policy/admin surfaces are stable
- [ ] Safe import vs partial import vs blocked is obvious to users

## Idiotproofing

- [ ] Error messages say what happened, why, and what to do next
- [ ] Common tasks have one obvious path
- [ ] Dangerous actions require explicit confirmation
- [ ] Operator-only actions are clearly separated
- [ ] User-facing wording avoids internal jargon
- [ ] Docs reflect current behavior exactly
- [ ] “It didn’t work” paths are intentionally designed

## Packaging

- [ ] `pyproject.toml` is complete and accurate
- [ ] CLI entry points are stable
- [ ] Wheel/sdist build cleanly
- [ ] Debian/system package support is either real or explicitly out of scope
- [ ] Service unit files are shipped/generated/documented cleanly
- [ ] Default directories/permissions are predictable
- [ ] Release artifact naming/versioning are deterministic
- [ ] Changelog/release notes flow exists

## Recovery / supportability

- [ ] `doctor` / self-check output is human-usable
- [ ] “Collect diagnostics” path exists
- [ ] Backup/export/import story exists for user data
- [ ] Upgrade rollback or recovery path is documented
- [ ] Failed upgrades do not strand the system half-broken

## Release gates

- [ ] Fresh install test
- [ ] Upgrade test
- [ ] Corrupt-config test
- [ ] No-network / degraded-network test
- [ ] Safe-mode test
- [ ] Pack quarantine/blocked-path test
- [ ] Memory migration/reset test
- [ ] Service restart/reboot persistence test
- [ ] No-secrets-in-logs test
- [ ] Smoke suite runs in one command

---

# Suggested Deliverables

- [ ] Hardening fixes
- [ ] Packaging cleanup
- [ ] Canonical install/upgrade/reset docs
- [ ] Release smoke suite
- [ ] Updated doctor/diagnostics path
- [ ] Docs pass
- [ ] Finalized `RELEASE_READINESS.md`

---

# Suggested Test Additions

- [ ] Fresh install from clean environment
- [ ] Upgrade from prior known-good version
- [ ] Invalid config
- [ ] Corrupted config
- [ ] Missing state directories
- [ ] Memory reset/migration
- [ ] Safe pack blocked-path
- [ ] No-network/degraded-network behavior
- [ ] Restart persistence
- [ ] No-secrets-in-logs assertions

---

# Definition of Done

This release-readiness pass is complete when all of the following are true:

- A new user can install and run the product using the docs alone
- An operator can inspect status, manage policy, recover from common failures, and upgrade safely
- Memory and pack flows are understandable and resilient
- Release artifacts are buildable and installable via one canonical path
- A single release smoke suite can be run before shipping
- Docs match actual product behavior
- The product is boring in the good way: predictable, explicit, recoverable, and hard to misuse
