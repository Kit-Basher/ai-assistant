# Reality-based release and recovery (WP6)

Status: implementation design for the v0.2.24 pre-reinstall gate.

WP6 does not add a second assistant, capability registry, approval system, or
installer.  It makes the assembled WP1-WP5 product prove itself through the
same production boundaries used by a normal user.  The physical Ubuntu 24.04
fresh-host recovery remains a deliberate human boundary and is not performed
by this work package.

## Baseline and authority

The protected starting point is commit
`71a9b54bd9ecbf0b7dca03fec9b5daac40224c12`, runtime v0.2.23.  The live
`CapabilityRegistry`, unified chat understanding, WP3 task coordinator, pack
lifecycle, and central confirmation controller remain authoritative.  The
single user service and versioned runtime promotion mechanism remain the only
installed product.

The release gate reconciles live registry output with the protected native
manifest by identity, not by a hard-coded count.  Dynamic pack capabilities
are reconciled to exact pack/version/digest records.  A missing, unexpected,
or unproved entry is a failure.

## Layered evidence model

The canonical gate consumes these independent layers:

1. unit, schema, and contract tests;
2. registry-generated native and dynamic capability proof;
3. generated language transformations and property invariants;
4. multi-turn state, approval, cancellation, and recovery scenarios;
5. real installed local-model production-path probes;
6. real WP4/WP5 worker, broker, acquisition, and isolation proof;
7. complete normal-user journeys through chat/API/Web UI;
8. portable backup/isolated restore and application lifecycle proof.

Every report records its schema version, UTC observation time, full candidate
commit, dirty-diff fingerprint, command, category totals, and redaction
result.  Reports from another commit or a changed worktree cannot satisfy the
gate.  Implementation code never imports the blind corpus.  The corpus lives
under `tests/held_out/`, and a repository check rejects held-out text or IDs
found in production source/configuration.

Final held-out thresholds are fixed before execution: every required category
must pass, overall success must be at least 95%, and unsafe unintended
mutation, fabricated completion, false runtime/access/capability claims, and
developer-only recovery for an ordinary supported path must each be zero.

## User journeys and evidence

The WP6 report links each required journey to its production-path proof owner
and records the bounded user input/goal class, route or capability family,
policy/approval outcome, invocation and verifier outcome, mutation summary,
latency evidence when the owner exposes it, and the pass/fail reason.  The
exact candidate runner retains the richer per-turn API evidence.  Sensitive
inputs and result bodies are replaced by digests or bounded summaries.  The
candidate runner uses temporary state/config and ports and must pass three
consecutive complete runs without order leakage.

## Diagnostics boundary

The existing model-only support snapshot is extended into one normal-user
diagnostics export.  The API and UI use the same bounded, versioned document.
It contains build/runtime identity, OS class, service/readiness, capability and
dependency health, model/provider identity without credentials, policy mode,
pack provenance/status summaries, and bounded error-class counts.  It excludes
environment values, raw conversations, prompts/reasoning, raw pack documents,
private file contents, headers, cookies, tokens, keys, and secret-store data.
Recursive key/value redaction, size ceilings, and an adversarial leak scan are
release requirements.

## First-run state machine

First-run setup is surfaced by the normal Web UI and the durable onboarding
completion/intent record.  It explains goals/privacy, shows configured local
roots in Files, exposes model health/choice, explains Safe/Controlled behavior,
and identifies optional packs/network access.  An interrupted conversational
prompt is safely offered again after restart rather than being silently marked
complete.
Public next actions refer to UI controls and ordinary chat, never required
shell, JSON, Git, or systemd commands.  Operator commands remain available in
advanced documentation.  Cancellation preserves completed choices; restart
reconstructs the next incomplete step; an already configured upgrade is not
forced through setup again.

## Portable backup v2 and restore

Portable backup is an extension of the canonical operator lifecycle, not a
second product.  `personal-agent.backup.v2` is a deterministic archive and
manifest of an explicit allowlist:

- the canonical state database via SQLite online backup;
- model registry and non-secret Personal Agent configuration;
- a declared secret re-entry model (machine-bound secret-store ciphertext is
  deliberately excluded from the portable archive);
- external-pack records/artifacts and pack-private data needed for recovery;
- Personal Agent service unit/drop-ins and release-independent user settings.

Runtime releases, the `current` link, caches, logs, transient workers,
quarantine downloads, pending temporary files, model artifacts, arbitrary
home data, and unrelated systemd units are excluded.  Each member has mode,
size, SHA-256, logical class, and restore policy.  Archive paths are canonical,
bounded, link-free, and verified before any extraction.  Restore is staged,
integrity/version checked, and atomically merged only into an explicit empty or
approved temporary target.  Repeated restore is content-idempotent.  A corrupt,
truncated, future-major, or policy-incompatible backup fails before mutation.

The proof creates a backup from a controlled copy of current-shaped state and
restores it into a clean temporary home.  The broader isolated candidate and
lifecycle suites then start the exact candidate against temporary state,
exercise representative native and pack paths, restart, and prove persistence.
Live state is never the restore target.

## Installation and recovery

The existing staged release/promotion, host lifecycle runner, rollback
checkpoint, and one user service remain canonical.  WP6 exercises clean
application installation, first start, upgrade, failed/interrupted upgrade,
rollback, restart, identity verification, and release retention in isolated
roots.  No permanent parallel service is created.

Ubuntu 24.04 preparation consists of an idempotent preflight, dependency
matrix, backup compatibility check, and post-install verifier.  Container/host
checks prove scripts and application boundaries but are explicitly not
evidence of a physical fresh-host journey.

## Gate sensitivity

Sensitivity runs the actual WP6 canonical proof command with one signed,
test-only defect overlay at a time.  The command must reject missing,
unexpected, or unproved capabilities; verifier/authorization/false-success
bypass; stale fingerprints; isolation/broker/network regressions; preservation
failure; stuck approvals; orphan workers; unhandled errors; restart failure;
and held-out threshold failure.  The overlay cannot be accepted by a normal
release invocation.  Source is restored and the complete clean gate is rerun.

## Security review scope

The independent register covers authorization and exact approval binding,
Safe/Controlled mode, task verification and recovery, pack lifecycle and
sandboxing, broker/SSRF and filesystem boundaries, secrets and diagnostics,
backup/restore, update/rollback provenance, UI/backend trust, and containment
of model false-success claims.  An unresolved high/critical finding or a
release-relevant medium finding blocks promotion.

## Explicit boundary

WP6 may finish the safe current-host work and promote its exact candidate, but
it cannot be called fully complete until the documented physical Ubuntu 24.04
install -> restore -> product journey -> upgrade -> rollback test is performed
with explicit user approval.  The only valid intermediate status is:

`WP6 PRE-REINSTALL GATE COMPLETE - READY FOR UBUNTU RECOVERY TEST`.
