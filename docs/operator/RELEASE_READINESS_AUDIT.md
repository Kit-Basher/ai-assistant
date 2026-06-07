# Release Readiness Audit

Date: 2026-06-06
Checkpoint: `295b578` Harden semantic memory indexing reliability
Updated after `4e794d2` Add release readiness audit for scoped bulk preference reset/clear reliability.

This is a release-readiness audit, not a claim that every planned capability is
finished. It records what is safe to put in front of a public user and what must
remain operator-only, optional, or future work.

## Rating

Yellow.

The core local assistant path, first-run path, external-pack safety gates,
assistant/agent boundary, and supported install stories are coherent enough for
a controlled public trial. This is not Green because several reliability
controls are still in-memory or operator-grade, semantic memory remains
release-gated/off-by-default, and the live barrage is a smoke check rather than
a full behavioral certification.

Do not call this broadly release-ready until the remaining release blockers
below are resolved or explicitly scoped out of the public build.

## Evidence Reviewed

- Baseline checks reported before this audit:
  - `tests/test_project_intent_docs.py`: 12 passed
  - `scripts/external_pack_safety_smoke.py`: PASS 39 gates
  - behavior/release tests: 42 passed
  - strict live barrage with Telegram bridge: PASS, quality warnings=0
  - git status clean
- Release-facing docs:
  - `README.md`
  - `docs/operator/SETUP.md`
  - `docs/product/PROJECT_INTENT.md`
  - `docs/operator/CURRENT_CHECKPOINT.md`
  - `docs/operator/MANAGED_ACTION_RELIABILITY_AUDIT.md`
  - `docs/operator/OPERATIONS.md`
  - `docs/operator/doctor.md`
  - `docs/operator/BACKUP_RESTORE.md`
  - `docs/operator/RELEASE.md`
  - `docs/operator/SAFE_WEB_SEARCH.md`
  - `docs/operator/KNOWN_LIMITS.md`
- Release and safety tests:
  - first-run/install hardening
  - release bundle packaging
  - Debian package packaging
  - assistant behavior release gate
  - chat behavior audit
  - live barrage classifier
  - external-pack safety smoke

## Top Release Blockers

These block broad public release unless explicitly scoped out of the public
surface:

1. Persistent managed-action journal storage is still missing. Many mutating
   flows have in-memory journals and readback verification, but post-crash
   recovery is not yet a uniform product guarantee.
2. Package install and directory creation shell flows are not covered as
   runtime managed actions. They must stay out of normal assistant actions.
3. Future filesystem writes have no transaction design. Current native
   filesystem behavior is read-only and should remain that way for public use.
4. Semantic memory must remain off by default and release-gated. Optional
   ingest/rebuild/repair has improved, but it needs more restart/provider-failure
   soak before default-on promotion.
5. Pack quarantine file artifact cleanup is still only partially covered. Pack
   metadata rollback is stronger than artifact cleanup.
6. Model acquisition/import rollback remains conservative. It does not delete
   Ollama cache/model data without ownership proof, so failed acquisitions may
   require operator cleanup.
7. Provider setup transaction coverage does not yet span every later
   model/default mutation after provider verification.
8. Scoped bulk preference reset/clear now has in-memory journal, verification,
   redaction, and scoped rollback coverage, but it still lacks persistent
   post-crash journal storage.
9. Live behavior barrage is good smoke coverage only. It catches boundary,
   quality, and stale-context regressions, but it is not enough by itself for a
   release gate.
10. Public install/update/rollback needs one clean end-to-end pass on a clean
    user-local environment or VM before any unassisted public trial.

## Non-Blocking Polish Items

These should not block a controlled trial if the blockers above are scoped:

1. Make first-run copy even shorter when model/provider setup is missing.
2. Add more examples for local model/provider setup in `SETUP.md`.
3. Add a one-command public preflight checklist that wraps existing read-only
   diagnostics.
4. Promote live web UI and reference pack smokes only after they are boring
   across repeated runs.
5. Add more memory-specific prompt-injection regressions for semantic/document/
   working-memory recall.
6. Expand release-gate assertions around assistant/agent boundary wording
   without making them brittle phrase snapshots.
7. Add targeted tests for registry maintenance unrelated-field drift.
8. Add richer ownership inspection for managed-service container reuse.
9. Add clearer operator wording for search-disabled versus SearXNG-not-running.
10. Add a concise rollback decision tree to `RELEASE.md`.

## Area Findings

### First-Run Setup

Status: mostly ready for controlled public trial.

The docs distinguish stable bundle install, checkout/dev install, and Debian
package install. The first-run command is `python -m agent setup`, and setup is
complete when onboarding is `READY` and runtime status is healthy. Telegram is
optional/off by default. Web search is optional and requires explicit SearXNG
configuration.

Remaining risk: a public user still needs a clean-machine install pass before
trial, especially for the bundled installer and desktop launcher.

### Normal User Experience

Status: ready for controlled public trial.

The assistant/agent boundary is documented in `PROJECT_INTENT.md`, README, and
release tests. Behavior tests cover no fake action claims, no internal routing
leaks, no hidden core presentation rewrites, and failure copy with a safe next
step.

Remaining risk: live barrage is a smoke. Keep deterministic release-gate tests
as the actual guard.

### Reliability

Status: Yellow.

SearXNG setup/cleanup is the reference managed-action implementation. Many
other write paths now have journals, verification, and scoped rollback where
ownership is proven. Remaining gaps are clearly tracked in
`MANAGED_ACTION_RELIABILITY_AUDIT.md`.

Remaining risk: persistent journal storage is the next reliability target.

### Safety / Security

Status: strongest area.

External pack ingestion has explicit source/content trust separation,
quarantine, normalization, strict scans, denied permissions by default, and no
foreign code execution. Secrets and support payloads are redacted. The normal
assistant does not expose arbitrary shell, Docker, systemctl, or filesystem
writes.

Remaining risk: keep package install, directory creation, Docker/container
control, and future filesystem writes outside normal assistant actions unless
they use a managed transaction path.

### Docs

Status: mostly ready.

The release-facing docs now align on install paths, optional Telegram/search,
assistant/agent boundary, managed-action gaps, backup/restore, and operator
diagnostics.

Remaining risk: keep docs synchronized with code whenever setup, recovery, or
pack lifecycle behavior changes.

### Tests

Status: adequate for controlled trial, not enough for Green.

The release gate covers deterministic behavior, first-run/install hardening,
assistant boundary, packaging, diagnostics, and external-pack safety. The live
barrage is valuable as a smoke and should stay separate from the canonical gate
until it is stable enough for every release environment.

Remaining risk: add focused tests for the listed reliability gaps rather than
brittle whole-answer wording snapshots.

### Packaging / Update

Status: ready for controlled trial after a clean end-to-end install pass.

`install_local.sh` is the developer checkout path. The release bundle installer
and uninstaller are tested for idempotence and state-preserving/state-removing
uninstall. The Debian package path is present and tested as an optional install
path.

Remaining risk: rollback is documented but still manual; no formal database
migration story exists.

## What Must Happen Before Any Public User Tries It

1. Run the verification suite listed below on a clean working tree.
2. Build and install the release bundle on a clean user-local environment or VM.
3. Launch from the desktop launcher and browser URL.
4. Complete or skip first-run onboarding and confirm it does not re-prompt.
5. Confirm `python -m agent doctor`, `/ready`, `/state`, and `/packs/state`.
6. Confirm Telegram remains off unless explicitly enabled.
7. Confirm web search reports disabled/missing SearXNG without claiming search
   works.
8. Confirm semantic memory is disabled by default.
9. Confirm no assistant surface exposes arbitrary shell, package install,
   filesystem write, Docker, or systemctl actions as normal chat behavior.
10. Record the release gate output, smoke output, git commit, and known limits
    in release notes.

## What Can Wait

- Persistent managed-action journal storage if the public trial excludes
  crash-recovery claims for mutating flows.
- Semantic memory default-on promotion.
- Future filesystem write operations.
- Additional managed adapters beyond the current bounded core-owned contracts.
- More polished provider/model setup copy.
- More live smoke promotion once repeated runs are stable.

## Exact Next Recommended Work Item

Add persistent managed-action journal storage before making crash/restart
recovery claims for mutating managed actions. Scoped preference reset/clear now
has in-memory journal, verification, redaction, and rollback coverage.

## Required Verification

Run:

```bash
python -m pytest -q tests/test_project_intent_docs.py
python scripts/external_pack_safety_smoke.py
python -m pytest -q tests/test_chat_behavior_audit.py tests/test_live_user_barrage.py tests/test_assistant_behavior_release_gate.py
git diff --check
git status
```
