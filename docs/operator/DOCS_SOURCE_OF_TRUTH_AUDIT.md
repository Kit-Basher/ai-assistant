# Docs Source-of-Truth Audit

Date: 2026-06-20

This audit is intentionally compact. It identifies the docs to trust first and
the contradictions to fix before a public release, without deleting historical
or supporting docs.

## Canonical Sources

- Product promise and assistant/agent boundary:
  `docs/product/PROJECT_INTENT.md`
- Current state and proof commands:
  `docs/operator/CURRENT_CHECKPOINT.md`
- Release ledger:
  `docs/operator/RELEASE_LEDGER.md`
- Roadmap/doc routing:
  `docs/operator/ROADMAP_INDEX.md`
- Operator health and incident response:
  `docs/operator/OPERATIONS.md`
- Setup/install paths:
  `docs/operator/SETUP.md`
- Doctor behavior:
  `docs/operator/doctor.md`
- Backup/restore:
  `docs/operator/BACKUP_RESTORE.md`
- Search/SearXNG:
  `docs/operator/SAFE_WEB_SEARCH.md`
- Security boundary:
  `docs/operator/SECURITY_AUDIT.md`

## Outdated Or Contradictory Items

- `docs/operator/CURRENT_CHECKPOINT.md` still leads with
  `v0.2.0-live-usefulness-proof` on 2026-06-15 even though the current work has
  moved through chat reliability harness and release-readiness hardening. Treat
  it as historical until refreshed with the next checkpoint.
- `docs/operator/ROADMAP_INDEX.md` lists `v0.2.0-live-usefulness-proof` as the
  current checkpoint. It should be updated when the next clean checkpoint is
  promoted.
- `README.md` now frames package installs and directory creation as narrow,
  confirmation-gated bounded paths, not arbitrary assistant powers.
- `docs/operator/SETUP.md` now points at `docs/product/PROJECT_INTENT.md`
  instead of the missing `PRODUCT_RUNTIME_SPEC.md`.
- Some historical/archive docs predate Plan Mode, managed SearXNG, and the
  chat reliability harness. Keep them for history; do not use them as release
  authority.

## Missing Or Newly Added Docs

- Added `docs/operator/SECURITY_AUDIT.md` for the current security boundary.
- `scripts/prove_ready.py` is now the compact readiness command. Use it for
  pre-VM readiness and keep `release_gate.py` as a heavier packaging/release
  command.
- `scripts/llm_behavior_eval.py` is now the deterministic second-tier
  orchestrator behavior eval. It should become part of the operator checklist
  when it stays stable.

## Delete/Merge Recommendation

Do not delete docs in this pass. After fresh Debian VM proof, merge the current
checkpoint, roadmap index, release ledger, and release-readiness audit into a
shorter operator path:

1. `README.md`
2. `docs/operator/SETUP.md`
3. `docs/operator/OPERATIONS.md`
4. `docs/operator/RELEASE_LEDGER.md`
5. supporting design docs by link only

## Current Next Doc Work

After the next clean checkpoint, refresh:

- `docs/operator/CURRENT_CHECKPOINT.md`
- `docs/operator/ROADMAP_INDEX.md`
- `docs/operator/RELEASE_LEDGER.md`
- checkpoint docs that still name older tags as current
- broader consolidation of overlapping release/checkpoint docs
