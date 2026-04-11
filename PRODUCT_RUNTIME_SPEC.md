# Product Runtime Spec

This document captures the canonical product/runtime scope for the current
Personal Agent. README is the product-facing overview; this file is the deeper
runtime/contract companion. If older notes conflict with this file, this file
wins.

## 1. Product Purpose
- Personal Agent is a local-first assistant for grounded runtime/model control
  and bounded native local operations.
- It is designed to answer from real runtime truth, not generic assistant
  improvisation, when deterministic data is available.
- It supports explicit, reviewable local mutation only through guarded
  controller/native-skill paths.

## 2. Core Principles
- One assistant execution path.
- One runtime truth source.
- One selector/scout path.
- One canonical recommendation/advisory contract.
- Local-first by default.
- Explicit approval for mutation.
- No hidden switching, installing, or adoption.

## 3. Runtime Model
- `RuntimeTruthService` is the canonical runtime truth source.
- The orchestrator owns assistant routing and grounded response rendering.
- The selector/scout owns canonical advisory computation.
- The controller owns explicit model actions.
- The canonical model manager owns install/acquire execution.
- Native skills are controller-backed and bounded.
- The API server also serves the browser/web UI from `/` when the built web
  assets are present; Telegram remains an optional transport adapter.
- Canonical operator/runtime path:
  - repo checkout: `~/personal-agent`
  - user service: `personal-agent-api.service`
  - mutable state: `~/.local/share/personal-agent`
  - operator config/policy: `~/.config/personal-agent`
  - repo install/update path: `pip install -e .`
  - release artifact build path: `python scripts/build_dist.py --outdir dist --clean`
  - first-run/recovery commands: `python -m agent setup`, `python -m agent doctor --fix`
  - diagnostics path: `python -m agent doctor`, `python -m agent doctor --collect-diagnostics`
- Canonical packaging truth:
  - `pyproject.toml` is the packaging metadata source
  - `VERSION` is the version source
  - the release bundle and Debian package are supported optional install
    paths; legacy root/system packaging is out of scope
- Legacy root/system-service wrapper scripts are retired and fail closed.

## 4. Mode Model
- SAFE MODE is the baseline unless explicitly overridden.
- Controlled Mode is explicit override only.
- Mode changes happen through the loopback-only, confirm-gated
  `/llm/control_mode` surface.
- Neither mode allows silent switching or silent installs.

## 4A. Runtime Status Surfaces
- `GET /health`, `GET /ready`, and `GET /runtime` are the canonical fast
  lifecycle/status surfaces.
- They are read-only and deterministic.
- They surface explicit:
  - `phase`
  - `startup_phase`
  - `runtime_mode`
  - `warmup_remaining`
  - degraded/blocked state
- `/ready` remains the richest operator/user readiness surface.
- Startup before router warmup completes must degrade explicitly; it must not
  crash or hang the status surface.

## 5. Recommendation Model
- `recommendation_roles` is the canonical recommendation/advisory truth.
- `POST /llm/models/check` and `POST /llm/models/recommend` consume that same
  truth.
- Compatibility fields are derived summaries only.

## 6. Native Capability Model
- Runtime/model inspection is deterministic and grounded.
- Memory behavior is explicit and inspectable:
  - deterministic continuity memory is the canonical thread/resume layer
  - continuity persistence is still full-record replace, but writes are now
    revision-aware compare-and-swap updates
  - per-key optimistic concurrency control is enforced at the storage write
    boundary
  - stale cross-runtime continuity writes are rejected explicitly instead of
    blindly overwriting newer state
  - merge-on-write is still out of scope
  - cross-key atomic snapshots are still out of scope
  - stale runtimes must reload before retrying a rejected save
  - optional `memory_v2` and optional semantic memory are additive helper
    stores only
  - `GET /memory/status` is the canonical loopback-only inspect surface
  - `/memory/status` exposes current continuity revisions, last attempted
    write outcome, last successful write outcome, and last stale-write
    conflict metadata when present
  - conflict metadata is observable and explicit; it is not auto-resolved
  - `POST /memory/reset` is the canonical loopback-only preview + confirm erase
    surface
  - corrupt or unavailable memory must degrade clearly without taking down
    unrelated runtime/chat paths
- Filesystem capability is read-only plus bounded search:
  - list
  - stat
  - read text
  - filename search
  - bounded text search
- Shell capability is bounded:
  - safe read-only commands
  - bounded install path
  - bounded directory creation
- External pack ingestion is bounded:
  - quarantined first
  - remote fetch is allowed only for explicit supported `https` archive sources
  - static-scanned before any normalization
  - portable text skills only in this pass
  - canonical content identity is authoritative for normalized packs
  - imported content remains non-executable and gets no granted permissions by
    default
- Unsupported by design:
  - arbitrary shell
  - unrestricted disk access
  - delete/remove flows
  - foreign code/plugin pack execution

## 7. Mutation / Approval Model
- Read-only actions execute immediately.
- Mutating actions preview first.
- Execution happens only after explicit confirmation.
- This applies across controller-backed model changes and current mutating
  native shell actions.

## 8. Discovery / Proposal / Policy Model
- Discovery is separate from canonical selector truth.
- `ModelDiscoveryManager` is the thin provider-agnostic fan-out layer over
  Hugging Face, OpenRouter, Ollama, and external snapshots.
- Proposals are non-canonical, review-required, and not auto-adopted.
- Curated policy is the reviewed operator layer.
- Reviewed policy may describe:
  - `known_good`
  - `known_stale`
  - `avoid`

## 9. External Pack Ingestion Model
- Discovery is read-only:
  - `GET /pack_sources`
  - `GET /pack_sources/catalog`
  - `POST /pack_sources/catalog`
  - `GET /pack_sources/catalog/<source_id>`
  - `PUT /pack_sources/catalog/<source_id>`
  - `DELETE /pack_sources/catalog/<source_id>`
  - `GET /pack_sources/<source_id>/packs`
  - `GET /pack_sources/<source_id>/search?q=...`
  - `GET /pack_sources/<source_id>/packs/<remote_id>/preview`
- Discovery source catalog is manageable through loopback/operator-only API
  surfaces, but configured source still does not imply trust.
- Discovery sources are policy-gated by local source policy before list/search/
  preview runs.
- Discovery source policy is now manageable through loopback/operator-only API
  surfaces:
  - `GET /pack_sources/policy`
  - `PUT /pack_sources/policy`
  - `GET /pack_sources/<source_id>/policy`
  - `PUT /pack_sources/<source_id>/policy`
- Catalog and policy are separate controls. Deleting a source also removes its
  per-source policy override so recreated sources fall back to current
  defaults.
- Allowlisted discovery source does not imply trust, approval, or executability.
- Registry/listing metadata is untrusted input and never becomes authoritative
  pack identity.
- Discovery cache is performance-only and remains untrusted metadata.
- Preview is not install. It may generate a safe install handoff, but pack
  contents are not fetched or made usable until explicit `/packs/install`.
- `POST /packs/install` treats downloaded third-party packs as hostile input by
  default.
- Ingestion order is:
  - optional safe remote archive fetch
  - quarantine
  - classify
  - static risk scan
  - normalize
  - plain-language review output
- Supported today:
  - `SKILL.md`-centered portable text skills
  - optional `references/`, `assets/`, `AGENTS.md`, and metadata files
- Supported remote ingress today:
  - `github_repo`
  - `github_archive`
  - `generic_archive_url`
  - `https` only, with provenance capture and archive validation
- Normalized external packs are inspectable through read-only surfaces:
  - `GET /packs/<canonical_id>`
  - `GET /packs/<canonical_id>/history`
  - `GET /packs/compare?from=<canonical_id>&to=<canonical_id>`
- Same normalized content from different sources collapses to one canonical pack
  identity; upstream content changes are treated as new versions and compared
  explicitly.
- Discovery may surface likely portable text skills, experience packs, or
  likely native/plugin packs, but only portable text skills are currently
  compatible with safe import.
- Unsupported/native/plugin packs are blocked or reduced to safe text/assets
  only when possible.
- No imported pack gets executable runtime privileges in this pass.

## 10. Out Of Scope
- Arbitrary autonomous shell behavior.
- Unrestricted filesystem mutation.
- Foreign code or plugin-pack execution.
- Automatic model switching or installing.
- Automatic proposal adoption.
- Automatic external-pack trust, approval, or background sync.
- Background full-disk indexing or unrestricted scanning.
- Legacy root/system packaging.
- Duplicate recommendation or controller paths.

## 11. Release Confidence
- The canonical release gate is `python scripts/release_gate.py`.
- The fast pre-check inside that gate is `python scripts/release_smoke.py`.
- Run it before calling a build releasable and after risky install/upgrade work.
- It is intended to prove the coherent product path plus the main
  safety/recovery gates, not to exhaustively test every internal subsystem.
- A heavier follow-up validation path exists at
  `python scripts/release_validation_extended.py`.
- Release, rollback, backup, defaults, and support boundaries are documented
  in `docs/operator/RELEASE.md`.
