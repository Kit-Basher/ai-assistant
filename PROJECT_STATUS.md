# PROJECT_STATUS.md

This file is the canonical current-state and handover document for the live
Personal Agent architecture. If older design notes or archived audits disagree
with this file, trust this file and the implementation. Use `README.md` for the
product-facing overview; use this file for current-state and handover truth.

## 1. Big Picture
- Assistant path:
  - user `/chat` flows through `api_server -> orchestrator -> deterministic assistant handlers`
  - assistant wording is grounded in canonical runtime truth and controller
    results; it must not hallucinate state or actions
  - the same API server also serves the browser/web UI from `/` when the built
    web assets are present; there is no separate desktop app in this repo
- Runtime truth:
  - `RuntimeTruthService` is the single runtime truth source for inventory,
    provider usability, readiness, lifecycle, mode policy, scout output, and
    canonical advisory state
- Selector / scout:
  - one selector path feeds assistant recommendations and operator advisory
    surfaces
  - selector input quality now depends on preserved canonical metadata:
    `task_types`, context, pricing, modality, and exact-model enrichment where
    available
- Controller:
  - explicit `test`, `switch temporarily`, `make default`, `switch back`, and
    control-mode changes execute only through the controller/runtime path
- Model manager:
  - `CanonicalModelManager` remains the only install/download/import execution
    path
- Discovery / proposal / policy:
  - discovery is separate from selector truth
  - `ModelDiscoveryManager` is the thin provider-agnostic fan-out layer over
    Hugging Face, OpenRouter, Ollama, and external snapshots
  - proposals are non-canonical review objects
  - curated policy is a small reviewed operator layer, not automatic adoption
- External pack ingestion:
  - downloaded third-party packs are quarantined first
  - supported remote archive fetch now exists, but remote content still lands in
    quarantine only
  - only portable text skills are normalized into locally usable text/reference
    content
  - foreign code/plugin packs are blocked from execution by default
  - Current invariants:
    - one execution path
    - one runtime truth source
    - one selector path
    - one canonical recommendation/advisory contract
    - one canonical packaging/build story:
      - `pyproject.toml` is the packaging metadata source
      - `VERSION` is the version source
      - repo installs/updates use `pip install -e .`
      - release artifacts build through `python scripts/build_dist.py --outdir dist --clean`
      - the release bundle and Debian package are supported optional install
        paths; legacy root/system packaging remains unsupported
    - one canonical install/service story:
      - repo checkout at `~/personal-agent`
      - user service `personal-agent-api.service`
      - mutable state under `~/.local/share/personal-agent`
    - operator config/policy under `~/.config/personal-agent`
    - `python -m agent setup` / `python -m agent doctor --fix` as the
      first-run and recovery path
    - `python -m agent doctor --collect-diagnostics` as the canonical redacted
      diagnostics bundle path
    - `python scripts/hardware_observe_smoke.py` as a non-blocking live
      answer-shape smoke for RAM/VRAM observation
    - `tests/test_webui_conversation_smoke.py` as the deterministic shipped-path
      two-turn assistant proof
    - `tests/test_assistant_behavior_release_gate.py` as the explicit
      assistant-behavior release gate for greeting, vague, nonsense, mixed,
      and no-LLM chat cases
    - `tests/test_clean_context_validation.py` as the isolated temp-home
      release-bundle clean-context proof for install, launcher, chat, relaunch,
      and uninstall/remove-state
    - legacy root/system-service scripts retired and fail closed

## 2. Mode Contract
- SAFE MODE
  - config baseline unless explicitly overridden
  - local-first
  - remote switching blocked at the shared low-level switch boundary
  - install/download/import blocked at the canonical model-manager boundary
  - scout remains advisory-only
- Controlled Mode
  - explicit override only
  - entered and exited through the loopback-only, confirm-gated
    `/llm/control_mode` surface
  - allows remote recommendations, explicit remote switching, and explicit
    acquire/install actions when policy and provider/model usability allow them
- mode-intent parsing
  - flexible phrasing is reduced to a tiny bounded control intent schema in the
    orchestrator
  - action execution still happens only through the canonical controller/runtime
    control path
- In both modes:
  - nothing switches automatically
  - nothing installs automatically
  - all test/switch/default/acquire actions remain explicit and approval-gated

## 3. Canonical Recommendation Contract
- `recommendation_roles` is the canonical advisory truth
- It resolves per-role outcomes for:
  - `best_local`
  - `cheap_cloud`
  - `premium_coding`
  - `premium_research`
  - `best_task_chat`
  - `best_task_coding`
  - `best_task_research`
- Each role can carry:
  - `state`
    - `selected`
    - `blocked_by_mode`
    - `no_qualifying_candidate`
    - `unavailable`
  - `model_id`
  - `recommendation_basis`
  - `comparison`
  - `advisory_actions`
  - `reason_code`
  - `explanation`
- Contract rules:
  - premium coding and premium research do not silently degrade into
    cheap-cloud or generic task-best answers
  - blocked-by-mode stays distinct from no qualifying candidate
  - comparison and action availability are computed canonically once and reused
  - assistant wording consumes this structure; it does not invent separate
    recommendation logic
- Canonical operator surfaces aligned on this truth:
  - `POST /llm/models/check`
  - `POST /llm/models/recommend`
- Compatibility summaries only:
  - `/llm/models/recommend`
    - `selected`
    - `top_for_purpose`
  - `/llm/models/check`
    - `recommendations_by_purpose`
  - these are thin derived views from `recommendation_roles`, not independent
    recommendation logic

## 4. Canonical And Operator Surfaces
- Canonical mode/status
  - `GET /llm/control_mode`
  - `POST /llm/control_mode`
  - `GET /ready` -> `llm.policy`
  - `GET /health`, `GET /ready`, and `GET /runtime` now align on the same
    explicit lifecycle truth:
    - `phase`
    - `startup_phase`
    - `runtime_mode`
    - `warmup_remaining`
    - safe-mode / blocked-state visibility
    - `/ready` remains the richest readiness surface
- Canonical recommendation/advisory surfaces
  - `GET/POST` model recommendation flows derive from
    `RuntimeTruthService.model_scout_v2_status()`
  - `POST /llm/models/check`
  - `POST /llm/models/recommend`
- Discovery / proposal / policy surfaces
  - `POST /llm/models/proposals`
    - operator-only discovery queue
    - proposals remain `proposed`, `non_canonical`, `review_required`,
      `not_adopted`
    - includes `review_suggestion` to help operators prepare explicit policy
      writes
  - `POST /llm/models/policy`
    - explicit curated policy write/update/remove surface
  - `GET /llm/models/policy`
    - curated policy read/list surface over the same store
- External pack surfaces
  - `GET /pack_sources`
    - lists configured discovery-only external pack sources
    - now includes policy-aware metadata such as allowed-by-policy state and
      cache TTL
  - `GET /pack_sources/catalog`
    - loopback/operator-only read surface for the persisted discovery source
      catalog
  - `POST /pack_sources/catalog`
    - loopback/operator-only create surface for discovery sources
  - `GET /pack_sources/catalog/<source_id>`
    - loopback/operator-only detail read surface for one configured discovery
      source
  - `GET /pack_sources/policy`
    - loopback/operator-only read surface for persisted discovery source
      policy defaults plus effective normalized policy
  - `GET /pack_sources/<source_id>/packs`
    - lists normalized external registry metadata for one source
  - `GET /pack_sources/<source_id>/search?q=...`
    - read-only search over untrusted registry metadata
  - `GET /pack_sources/<source_id>/packs/<remote_id>/preview`
    - preview-only surface
    - generates policy hints, related-local-pack hints, and install handoff
      data
    - does not fetch pack contents
  - `PUT /pack_sources/catalog/<source_id>`
    - loopback/operator-only update surface for one configured discovery source
  - `GET /pack_sources/<source_id>/policy`
    - loopback/operator-only read surface for one source override plus its
      effective policy
  - `DELETE /pack_sources/catalog/<source_id>`
    - loopback/operator-only delete surface for one configured discovery source
    - also removes any matching per-source policy override so recreated sources
      fall back to current defaults
  - `PUT /pack_sources/policy`
    - loopback/operator-only write surface for discovery source policy defaults
    - strictly validated and auditable
  - `PUT /pack_sources/<source_id>/policy`
    - loopback/operator-only write surface for per-source policy overrides
    - strictly validated and auditable
  - `POST /packs/install`
    - quarantines a downloaded local pack snapshot or safely fetched remote
      archive snapshot
    - classifies, static-scans, normalizes, and returns a plain-language review
      envelope
    - supports portable text skills only; foreign code/plugin packs are blocked
      or partially stripped by policy
  - `GET /packs`
    - lists runtime/native pack records plus external pack ingestion records
  - `GET /packs/<canonical_id>`
    - reads one normalized external pack by canonical content id
  - `GET /packs/<canonical_id>/history`
    - returns merged source history and version chain for one normalized
      external pack
  - `GET /packs/compare?from=<canonical_id>&to=<canonical_id>`
    - returns a read-only structured diff plus plain-language change summary
      between normalized pack versions
- Curated policy statuses
  - `known_good`
  - `known_stale`
  - `avoid`
- Remaining compatibility/operator-only surfaces
  - `/llm/status`, `/model`, `/llm/model`
    - compatibility wrappers over canonical truth
  - `/llm/fixit`
    - loopback/operator-only compatibility recovery plane
  - `/defaults/rollback`
    - operator persisted-default recovery; intentionally separate from
      assistant `switch back`

## 5. Native Skills And Approval Contract
- Runtime/model/controller
  - current model and effective target inspection
  - local/cloud inventory inspection
  - canonical recommendations via the scout
  - explicit controller actions for test / temporary switch / make default /
    switch back / acquire
- Filesystem skill
  - bounded list/stat/read/search only
  - allowed-root enforcement
  - sensitive-path blocking
  - bounded reads and searches
- Shell skill
  - bounded read-only inspection commands
  - bounded install path
  - bounded directory creation path
  - no arbitrary shell public surface
- External pack ingestion
  - discovery layer is read-only and separate from install
  - operators can now manage the discovery source catalog through loopback-only
    API surfaces instead of editing local files directly
  - discovery sources are policy-gated before query execution
  - operators can now manage discovery source policy through loopback-only API
    surfaces instead of editing local files directly
  - configured source does not mean trusted source
  - catalog and policy are separate controls
  - allowlisted source does not mean trusted source
  - discovery cache is performance-only and remains untrusted
  - registry metadata is untrusted and advisory only
  - preview is not install; only explicit `/packs/install` fetches content
  - preview may hint:
    - likely portable text skill
    - likely experience pack
    - likely native/plugin pack
    - whether a related local canonical pack already exists
  - supported today: portable text skills centered on `SKILL.md`
  - supported ingress today:
    - local downloaded snapshots
    - remote `https` archive sources with explicit provenance capture
  - optional `references/`, `assets/`, `AGENTS.md`, and metadata files are
    preserved as non-executable content where safe
  - all external packs are quarantined, scanned, normalized, and granted no
    permissions by default
  - canonical content identity is authoritative:
    - same normalized safe content => same canonical pack id
    - same source with changed content => new pack/version with higher review
      risk
  - normalized pack history and compare are now inspectable through read-only
    surfaces
  - remote fetch records source url, ref, resolved commit when available,
    archive hash, fetched time, and quarantine path
  - shell/python/js/plugin/native bundles are not executed
- Approval contract
  - read-only actions remain immediate
  - mutating native/controller actions preview first
  - execution happens only after explicit confirmation
  - current mutating covered paths include:
    - model switch/default/acquire flows
    - package install
    - create directory

## 6. Memory Architecture
- Deterministic SQLite-backed memory remains the canonical source of truth for
  thread state, prefs, anchors, labels, open loops, reminders, and snapshots
- `memory_v2` remains an optional deterministic helper store behind
  `AGENT_MEMORY_V2_ENABLED`
- Optional semantic memory is now a separate helper layer behind
  `AGENT_SEMANTIC_MEMORY_ENABLED`
- Chat now uses a working-memory compactor ahead of prompt assembly:
  - hot recent turns stay verbatim
  - older unpinned turns compact into structured warm summaries
  - oldest distilled state becomes compact cold blocks
  - semantic memory is fed before detailed history is evicted
  - summary provenance is now explicit and bounded:
    - raw vs merged-summary origin is tracked
    - generation count is tracked
    - compression stays capped at level 3
  - the recency shield is now explicit and ephemeral:
    - the last few user turns stay verbatim during chunk selection
    - shielded turns are not converted into permanent pins
  - semantic durable-memory extraction now uses duplicate-ingest suppression for
    unchanged compacted material
  - panic trim now follows an explicit low-value-first order and keeps pinned
    and recency-shielded task context intact
  - `/memory/status` exposes hot/warm/cold token usage plus last compaction
    action, emergency trim count, and compact compaction debug metadata
- Continuity persistence is now revision-aware:
  - managed continuity writes remain full-record replace
  - per-key optimistic concurrency control is enforced at the storage write
    boundary
  - successful writes increment a stored revision
  - stale cross-runtime writes are rejected explicitly instead of silently
    overwriting newer state
  - merge-on-write remains out of scope
  - there is no cross-key atomic snapshot
  - stale runtimes must reload before retrying a rejected save
  - `/memory/status` exposes current continuity revisions plus last
    attempted write outcome, last successful write outcome, and last
    stale-write conflict metadata when present
  - conflict metadata is observable; it is not auto-resolved
- Canonical operator memory surfaces now exist:
  - `python -m agent memory` for a plain current-thread summary
  - loopback-only `GET /memory/status` for inspection
  - loopback-only `POST /memory/reset` for preview + confirm erase
- Broken memory state is explicit, not silent:
  - corrupt continuity entries are reported as degraded
  - optional memory helpers can fail without taking down unrelated chat/runtime
    flows
  - reset is explicit and scoped; no memory component is erased without
    confirmation
- Semantic recall is additive only:
  - it can suggest relevant context for conversations, notes, and documents
  - it must not overwrite deterministic facts or thread state
  - it fails closed when the embedding target, index state, or provider is not
    healthy
- Phase 2 now adds:
  - explicit document/file ingestion entry points
  - operator-facing rebuild/reindex workflows for stale, partial, missing, and
    model-changed indices
  - loopback-restricted semantic status surfaces that expose recovery state,
    counts, and operator next actions
- Cleanup pass:
  - semantic compatibility shims now route only through canonical ingest
    helpers
  - semantic status and rebuild responses now include a compact operator
    summary string
- Deterministic memory still outranks semantic recall in prompt assembly
- Compatibility shims remain thin wrappers around the canonical semantic
  ingestion helpers instead of a separate ingestion system

## 7. Metadata And Catalog Reality
- Selector quality now depends heavily on trustworthy metadata
- Exact-model enrichment exists only for a small curated set of known models
- Catalog/provider ingestion now preserves more trustworthy source metadata:
  - context length
  - pricing
  - modality
  - explicit task metadata when present in provider/catalog payloads
- Unknown remote models remain intentionally untyped unless trustworthy
  structured metadata exists
- Discovery/proposal work exists to surface promising or stale models without
  letting raw discovery output become canonical selector truth

## 8. Definitively Fixed
- explicit runtime mode control exists and is verified
- SAFE MODE remains the baseline unless explicitly overridden
- Controlled Mode is explicit override only
- bounded mode-intent parsing routes explicit mode changes through the
  controller/runtime control path
- first-chat startup race was fixed through readiness/bootstrap gating
- unmatched / ambiguous SAFE MODE prompts stay grounded or bounded
- provider/model eligibility is default-deny and canonical
- premium-role fallback leakage is removed
- recommendation explanations are canonical via `recommendation_basis`
- current-vs-recommended comparison is canonical via `comparison`
- next-action availability is canonical via `advisory_actions`
- `/llm/models/check` and `/llm/models/recommend` are aligned on the same
  canonical `recommendation_roles`
- compatibility summaries are derived-only views, not separate recommendation
  logic
- curated policy review/write/read flow exists:
  - `POST /llm/models/policy`
  - `GET /llm/models/policy`
- discovery proposals are separated from canonical selector truth
- proposal review UX exists via proposal `review_suggestion`
- native filesystem skill exists with bounded read/search behavior
- native shell skill exists with bounded read-only commands plus guarded
  install/create-directory paths
- secure external pack ingestion exists for downloaded third-party packs:
  - remote fetch -> quarantine -> classify -> static risk scan -> normalize ->
    review
  - portable text skills only
  - foreign code/plugin packs blocked by default
- secure external pack discovery/preview exists:
  - read-only source listing and search
  - untrusted preview metadata normalized into one local shape
  - explicit handoff into the existing quarantine-based install path only
- external pack identity and trust are now content-anchored:
  - canonical id derives from normalized content
  - trust attaches to content hash, not name or URL
  - upstream changes are treated as new versions and can be compared safely
- mutating native actions use unified preview/confirm gating
- confirmed model-switch responses now preserve truthful action-tool metadata
- compact release smoke coverage exists:
  - `python scripts/release_smoke.py`
- public legacy `/model_scout/*` operator HTTP endpoints are gone

## 9. Release Smoke Suite
- `python scripts/release_gate.py` is the canonical release gate
- `python scripts/release_smoke.py` is the fast pre-check inside that gate
- `python scripts/release_validation_extended.py` is the clearly named heavier
  follow-up path
- The main suite covers:
  - mode control
  - runtime/model status
  - recommendations
  - native filesystem skill
  - native shell skill
  - preview/confirm mutation flow
  - discovery/proposal/policy workflow
- It also pulls in targeted fast checks for:
  - fresh-install and first-run path safety
  - corrupt-config failure handling
  - memory inspect/degrade behavior
  - redacted diagnostics
  - restart persistence truth
- It intentionally does not try to exhaustively test every internal subsystem,
  do live network/provider smoke, or automate full reboot validation

## 10. Current Boundaries
- compatibility summary fields still exist for external consumers:
  - `/llm/models/recommend.selected`
  - `/llm/models/recommend.top_for_purpose`
  - `/llm/models/check.recommendations_by_purpose`
- discovery remains read-only and non-canonical
- explicit `/packs/install` still supports portable text skills only
- native/plugin packs remain blocked from execution
- the canonical release gate is `python scripts/release_gate.py`; it stays
  compact and does not attempt live network/provider smoke coverage or real
  reboot automation

## 11. Do Not Regress
- do not add duplicate selector paths
- do not add duplicate recommendation/advisory computation paths
- do not weaken SAFE MODE or Controlled Mode boundaries
- do not add automatic switching, automatic installs, or hidden fallbacks
- do not add automatic proposal adoption
- do not let external packs bypass quarantine, scanning, normalization, or
  default-deny permissions
- do not let remote fetch become a direct execution or trust path
- do not blur scout, controller, and model-manager responsibilities
- do not let premium-role prompts silently degrade into cheap/general answers
- do not let compatibility summaries become independent sources of truth
- keep proposals non-canonical unless future deliberate adoption work explicitly
  changes that contract
- keep `recommendation_roles` canonical
- keep mutating native actions behind explicit confirmation
- keep filesystem and shell capabilities bounded and privacy-protective
- keep foreign code/plugin packs non-executable unless future deliberate review
  work explicitly adds a different trust model
