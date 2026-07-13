# v0.2.2 Final Release Audit and Version Decision

Checkpoint under audit:

- Tag: `v0.2.2-runtime-latency-closure-v1`
- Commit: `34632188bcd90ad41e74ba7e188db905dfa710dc`

Final release tag has not been created.

## Version Decision

Recommended final version: `v0.2.2`.

`v0.2.2` is the honest semantic-versioning choice for this release line. The
work is substantial, but it completes the intended v0.2 authorization,
proof, and operational-hardening foundation while preserving normal user
workflows, API routes, state layout, installation paths, Backup v1, Restore v1,
and operator lifecycle behavior.

The rejected option is `v0.3.0`. The authorization architecture is a meaningful
platform phase, and skill-pack permissions now default-deny. However, this
repository currently derives the expected core DB schema from the product minor
version; a `0.3.0` product version would imply schema `3`, while the current
compatible state line remains schema `2`. Choosing `v0.3.0` would therefore
overstate migration burden and conflict with current runtime truth.

## Compatibility Findings

| Area | Classification | Finding |
| --- | --- | --- |
| SQLite schema | compatible | Product `0.2.2` remains on schema `2`; no destructive migration is required. |
| Preferences and memory | compatible | Existing preference and memory summary rows remain readable. |
| Anchors/tasks/notification history | compatible | Stores remain durable and are not reset by upgrade. |
| Plan and receipt stores | compatible | Universal Plan and receipt metadata are additive. |
| Skill grants | compatible with safe default | New grant store initializes empty; old or ungranted skill-pack mutations fail closed. |
| Backup v1 | compatible | Backup manifests/checksums remain Backup v1. |
| Restore v1 | compatible | Restore remains limited to supported allowlisted categories. |
| API routes | compatible | Existing readiness, state, chat, search, Telegram, pack, and version routes remain. |
| Confirmation UX | user-visible hardening | Mutations require exact Plan/confirmation binding. |
| Git push | intentionally denied | Capability is classified, but execution remains disabled. |
| Primary uninstall | intentionally disabled | Requires valid local activation marker. |
| Purge uninstall | unsupported | No purge path is enabled. |

## User Action Required

No ordinary user data migration is required.

Operators or skill-pack developers may need to add or approve skill-pack
permission manifests and grants. This is an intentional safety boundary, not an
automatic migration.

## Rollback Truth

Code rollback to `v0.2.1` is not the same as full state rollback. The supported
rollback boundary is:

- code/runtime rollback is supported through the existing lifecycle runner when
  a previous release checkpoint exists;
- state rollback after `v0.2.2` writes new Plan, receipt, grant, or permission
  records should use a Backup v1 artifact made before upgrade;
- new `v0.2.2` authorization records may be ignored by older code, but older
  code is not promised to understand every new receipt/grant field;
- destructive rollback is not attempted automatically.

## Security and Privacy Truth

- Capability policy audit: expected `WARN=0 FAIL=0`.
- Universal Plan audit: expected `WARN=0 FAIL=0`.
- Generic mutation bypass audit: expected `WARN=0 FAIL=0`.
- Full adversarial authorization proof: expected `FAIL=0`,
  `RELEASE_BLOCKERS=0`, `PROPERTIES_PROVEN=10/10`.
- Process isolation for arbitrary malicious in-process Python is not claimed.
- Raw secrets, token-bearing URLs, arbitrary shell, arbitrary HTTP mutation,
  and direct provider mutation remain denied through supported platform APIs.

## Accepted Warnings

Accepted warning classes are checked in at
`docs/operator/RUNTIME_LATENCY_ACCEPTANCE_V1.json`.

Expected accepted items:

- evidence-backed runtime latency variance;
- documented process-isolation limitation in security proof/smoke output.

Unresolved warnings must remain zero for final release.

## Artifact Boundary

Supported artifacts for this audit:

- final Git tag after commit;
- source checkout/archive from Git;
- repository release bundle from `scripts/build_release_bundle.sh`;
- wheel/sdist from the repository build backend.

Artifacts must not include local databases, raw secrets, `.env`, cache
directories, `/tmp` evidence, personal absolute source paths, or generated
bytecode.

## Manual Release Commands

After committing final-audit changes and reviewing the final status:

```bash
git status --short
python scripts/final_release_audit.py
python scripts/prove_ready.py
git tag -a v0.2.2 -m "Release v0.2.2"
git push origin HEAD
git push origin v0.2.2
```

Do not run these commands from the audit automatically.
