# Personal Agent

Personal Agent is a local-first AI assistant that answers from real runtime
state, uses bounded native skills, and previews any mutating action before it
executes it.

It is exposed through the HTTP API, the browser/web UI served by the API
server, the CLI, and the optional Telegram adapter.

It can inspect the current model/runtime, recommend better model choices, read
and search safe parts of the filesystem, run a small set of bounded shell
operations, safely ingest downloaded text-based skill packs, and carry out
explicit controller actions such as testing or switching models.

It is not a guessy autonomous agent. It does not invent state, does not expose
arbitrary shell execution, and does not mutate local or system state without an
explicit confirmation step.

## What It Does
- Local-first assistant: local models and local state are preferred by default.
- Grounded: assistant answers for runtime, model, filesystem, shell, and
  controller actions come from deterministic runtime truth or native skill
  results, not generic prose.
- Controller-backed: testing models, switching temporarily, making a default,
  acquiring models, package installs, and directory creation all go through the
  canonical controller/native-skill path.
- Approval-gated: mutating actions preview first and execute only after explicit
  confirmation.
- Safe external pack ingestion: downloaded third-party packs are quarantined,
  scanned, normalized, and denied permissions by default.

## What It Won't Do
- It will not run arbitrary shell commands.
- It will not read your whole filesystem indiscriminately.
- It will not mutate local or system state without explicit confirmation.
- It will not auto-install or auto-switch models.
- It will not let discovery proposals change canonical recommendations on their
  own.
- It will not execute foreign code or plugin packs.
- It will not install dependencies from imported packs.

## Core Concepts

### SAFE MODE
- Baseline mode unless explicitly overridden.
- Local-first and advisory-first.
- Remote switching is blocked.
- Install/download/import actions are blocked.
- Recommendations can still be shown, but they do not execute anything.

### Controlled Mode
- Explicit override only.
- Entered or exited through `GET/POST /llm/control_mode`.
- Enables explicit remote switching and explicit acquisition/install flows when
  provider/model policy allows them.
- Nothing switches or installs automatically here either.

### Local Control Plane
- A tiny loopback-only file-backed service lives at the repo-local `control/`
  directory.
- Canonical files:
  - `control/master_plan.md`
  - `control/DEVELOPMENT_TASKS.md`
  - `control/agent_events.jsonl`
- ChatGPT writes plan/task markdown through the HTTP control plane.
- Local agents watch those files locally and append progress/failure events to
  `agent_events.jsonl`.
- See `docs/control_plane.md` for the exact endpoints, task workflow, and run
  commands.

### Canonical Recommendation Truth
- `recommendation_roles` is the one canonical advisory contract.
- `POST /llm/models/check` and `POST /llm/models/recommend` are aligned on that
  truth.
- Compatibility fields are derived-only summaries, not a second advisory path.

### Discovery / Proposals / Policy
- Discovery output is non-canonical and review-only.
- Discovery now flows through a thin provider-agnostic `ModelDiscoveryManager`
  over Hugging Face, OpenRouter, Ollama, and external snapshot sources; the
  richer per-source metadata is preserved.
- `POST /llm/models/proposals` lists proposals and supports filtering.
- Proposals remain `proposed`, `non_canonical`, `review_required`, and
  `not_adopted`.
- `review_suggestion` helps operators prepare explicit policy writes.
- `POST /llm/models/policy` writes reviewed policy entries.
- `GET /llm/models/policy` lists current reviewed policy entries.
- Curated policy statuses:
  - `known_good`
  - `known_stale`
  - `avoid`

## Native Capabilities

### Runtime / Model / Controller
- Current model and effective runtime target inspection.
- Local and cloud model inventory queries.
- Local, coding, research, and cheap-cloud recommendations.
- Deterministic controller actions:
  - test a model
  - switch temporarily
  - make default
  - switch back
  - acquire/install model through the canonical model manager

### Filesystem
- List directories.
- Stat paths.
- Read bounded text files.
- Search by filename.
- Search bounded text content.
- Safety boundary:
  - allowed roots only
  - sensitive-path blocking
  - symlink/path normalization checks
  - bounded reads and searches

### Shell
- Safe read-only environment inspection commands.
- Bounded package install path.
- Bounded directory creation path.
- No public arbitrary shell execution surface.

### External Packs
- Read-only discovery is available through pack sources:
  - `GET /pack_sources`
  - `GET /pack_sources/catalog`
  - `POST /pack_sources/catalog`
  - `GET /pack_sources/catalog/<source_id>`
  - `PUT /pack_sources/catalog/<source_id>`
  - `DELETE /pack_sources/catalog/<source_id>`
  - `GET /pack_sources/<source_id>/packs`
  - `GET /pack_sources/<source_id>/search?q=...`
  - `GET /pack_sources/<source_id>/packs/<remote_id>/preview`
- Operators can manage the configured discovery source catalog through the API.
- Configuring a source does not imply trust. Catalog and policy are separate
  controls.
- Discovery sources are policy-gated per source id, source kind, and cache TTL.
- Allowlisted discovery sources are still untrusted. Allowlisted means
  queryable, not trusted or executable.
- Operators can inspect and update discovery source policy through the API:
  - `GET /pack_sources/policy`
  - `PUT /pack_sources/policy`
  - `GET /pack_sources/<source_id>/policy`
  - `PUT /pack_sources/<source_id>/policy`
  - these surfaces are loopback/operator-only
- Discovery metadata is untrusted and advisory only.
- Discovery cache is performance-only and remains untrusted metadata.
- Preview is not install. Nothing becomes locally usable until an explicit
  fetch/install request goes through quarantine and review.
- `POST /packs/install` ingests either:
  - a local downloaded pack snapshot
  - a supported remote archive source over `https`
- Remote fetch fills quarantine only, then goes through:
  - quarantine
  - safe remote fetch / archive validation
  - classification
  - static risk scan
  - normalization
  - plain-language review output
- Supported today:
  - portable text skills centered on `SKILL.md`
  - optional `references/`, `assets/`, `AGENTS.md`, and metadata files
- Foreign code or plugin packs are discovered and audited, but they are not
  executed.
- GitHub/archive sources are provenance-stamped and pinned where possible.
- Imported external packs get no granted permissions by default.
- Canonical pack identity is content-derived:
  - same safe normalized content from different URLs collapses to the same
    canonical pack id
  - same source with changed content is treated as a new pack/version
- Registry listings may surface likely portable text skills, experience packs,
  or likely native/plugin packs, but those type hints do not grant trust.
- Native/plugin packages may be discoverable, but current policy still blocks
  them from execution and safe import.
- Read-only inspection surfaces exist for normalized external packs:
  - `GET /packs/<canonical_id>`
  - `GET /packs/<canonical_id>/history`
  - `GET /packs/compare?from=<canonical_id>&to=<canonical_id>`

### Intentionally Not Supported
- `rm`/delete/remove flows.
- Unrestricted disk access.
- Automatic installs or switches.
- Automatic adoption of discovery proposals.

## Confirmation / Approval Rules
- Read-only actions run immediately:
  - runtime/model status
  - recommendations
  - filesystem list/stat/read/search
  - safe read-only shell queries
- Mutating actions preview first:
  - model switch / make default / acquire
  - package install
  - create directory
- Execution happens only after explicit confirmation such as `yes`.

## Example Flows
Mutating action preview/confirm:

```text
User: install ripgrep
Assistant: I will install ripgrep. This mutates the system. Reply yes to proceed or no to cancel.
User: yes
Assistant: [runs the bounded install path and returns the real result]
```

Blocked sensitive access:

```text
User: read ~/.ssh/config
Assistant: Blocked. sensitive_path_blocked.
```

The same preview-first rule applies to `create a folder called logs in this
repo`, `switch temporarily to ...`, `make this the default`, and explicit model
acquisition requests.

## Quick Start
Daily-driver install path:
1. Download or build a stable bundle, or install the packaged release.
2. Run its bundled `install.sh`.
3. Open Personal Agent from the desktop menu or browse to `http://127.0.0.1:8765/`.

That stable install:
- copies the runtime into `~/.local/share/personal-agent/runtime`
- keeps mutable state in `~/.local/share/personal-agent`
- installs the desktop launcher and user service
- is the path to use for normal daily use

Developer checkout install:
1. Clone the repo to `~/personal-agent`.
2. Run `bash scripts/install_local.sh --desktop-launcher`.
3. Use the checkout-backed dev service and launcher for repo work only.

That checkout install:
- creates or refreshes the repo `.venv`
- installs the package in editable mode
- installs the dev user service
- optionally installs a dev-named desktop launcher

If you need manual install, recovery, or rollback steps, use
`docs/operator/SETUP.md`.

If you downloaded a release bundle instead of cloning the repo:
1. Extract the bundle.
2. Run the bundled `install.sh`.
3. Open Personal Agent from the desktop menu or browse to `http://127.0.0.1:8765/`.

If you downloaded the Debian package instead:
1. Install the `.deb` with your normal Debian/Ubuntu package tools.
2. Launch Personal Agent from the desktop menu or run `personal-agent-webui`.
3. On first launch, the user service registers itself if needed and then opens the UI.

Useful local commands:
- `python -m agent status`
- `python -m agent doctor`
- `python -m agent health`
- `python -m agent version`
- `python -m agent packs`
- `python -m agent llm_inventory`
- `python -m agent memory`

## Release Artifacts
- Canonical packaging metadata lives in `pyproject.toml`.
- Canonical version truth lives in `VERSION`.
- Canonical release build command:
  - `python scripts/build_dist.py --outdir dist --clean`
- Debian package build command:
  - `bash scripts/build_deb.sh --clean`
- Expected artifacts:
  - `dist/personal_agent-<version>-py3-none-any.whl`
  - `dist/personal_agent-<version>.tar.gz`
- Debian package artifact:
  - `dist/personal-agent_<version>_amd64.deb`
- Canonical packaged CLI entry points:
  - `personal-agent`
  - `personal-agent-api`
  - `personal-agent-telegram`
- Debian packaging is supported through the release `.deb` and user-service
  activation model described in `docs/operator/SETUP.md`.
- Legacy `packaging/` service/env artifacts are not the shipping install path.

## Product-Relevant Operator Surfaces
- `POST /chat`
  - assistant front door
- `GET /health`
  - fast service/runtime health
  - explicit `phase`, `startup_phase`, `runtime_mode`, `warmup_remaining`,
    safe-mode/policy blocking state
- `GET /ready`
  - richest readiness surface
  - includes runtime readiness, warmup/degraded state, next action, and policy state
- `GET /runtime`
  - operator runtime snapshot
  - includes explicit runtime status plus provider/router summary
- `GET /version`
  - version/build metadata for support and release verification
- `GET /llm/control_mode`
  - current SAFE/Controlled mode state
- `POST /llm/control_mode`
  - explicit mode override
- `POST /llm/models/check`
  - canonical operator recommendation/status view
- `POST /llm/models/recommend`
  - canonical assistant/operator recommendation view
- `POST /llm/models/proposals`
  - non-canonical discovery proposal queue
- `GET /llm/models/policy`
  - curated policy list/read surface
- `POST /llm/models/policy`
  - curated policy write/update/remove surface
- `GET /packs`
  - runtime/native packs plus external pack ingestion records
- `GET /pack_sources`
  - configured discovery-only external pack sources
- `GET /pack_sources/catalog`
  - operator-only discovery source catalog read surface
- `POST /pack_sources/catalog`
  - operator-only discovery source catalog create surface
- `GET /pack_sources/policy`
  - operator-only discovery source policy read surface
- `GET /pack_sources/catalog/<source_id>`
  - operator-only discovery source detail read surface
- `GET /pack_sources/<source_id>/packs`
  - normalized listing view for one discovery source
- `GET /pack_sources/<source_id>/search?q=...`
  - read-only search over untrusted registry metadata
- `GET /pack_sources/<source_id>/packs/<remote_id>/preview`
  - preview one external listing and generate a safe install handoff
- `PUT /pack_sources/catalog/<source_id>`
  - operator-only discovery source catalog update surface
- `GET /pack_sources/<source_id>/policy`
  - operator-only per-source policy read surface
- `DELETE /pack_sources/catalog/<source_id>`
  - operator-only discovery source catalog delete surface
- `PUT /pack_sources/policy`
  - operator-only discovery source policy update surface
- `PUT /pack_sources/<source_id>/policy`
  - operator-only per-source policy override update surface
- `GET /packs/<canonical_id>`
  - inspect one normalized external pack by canonical content id
- `GET /packs/<canonical_id>/history`
  - inspect source history and version chain for one external pack
- `GET /packs/compare?from=<canonical_id>&to=<canonical_id>`
  - read-only structured diff and plain-language change summary between two
    normalized pack versions
- `POST /packs/install`
  - quarantined external pack ingestion for downloaded snapshots or supported
    remote archives

These are the product-facing surfaces worth learning first. The repo contains
additional internal/operator endpoints, but they are not the core publishable
surface.

## Release Smoke Suite
Run this before calling a build releasable:
- `python scripts/release_gate.py`

It is the canonical release gate. The fast pre-check inside it is
`python scripts/release_smoke.py`. It runs a fixed deterministic set of checks
covering:
- fresh-install and first-run path checks
- health/readiness/runtime restart truth
- the shipped two-turn web UI/API conversation proof
- chat/tool golden-path behavior
- the assistant-behavior release gate
- basic memory inspect/degrade behavior
- safe external pack discovery/preview/install blocked-path behavior
- redacted diagnostics/no-secrets sanity
- syntax validation, diff sanity, and the heavier release validation suite

A passing run means the core supported product path is still coherent and the
main safety/recovery gates are intact.

If you want the heavier follow-up validation path, run:
- `python scripts/release_validation_extended.py`

For live, non-blocking operator smoke checks on a running system, use:
- `python scripts/hardware_observe_smoke.py`
- `python scripts/live_product_smoke.py`
- `python scripts/telegram_smoke.py`
- `python scripts/hardware_discovery_smoke.py`
- `python scripts/discovery_quality_smoke.py`
- `python scripts/pack_route_smoke.py`
- `python scripts/restart_memory_smoke.py`
- `python scripts/provider_matrix_smoke.py`
- `python scripts/reference_pack_workflow_smoke.py`
- `python scripts/webui_smoke.py`

The Advanced drawer now includes read-only `State` and `Packs` sections backed by `GET /state` and `GET /packs/state`.

For a small brief transcript check, use:
- `python scripts/brief_smoke.py`

`python scripts/pack_route_smoke.py` also supports a remote-download mode when `PACK_ROUTE_SMOKE_REMOTE_URL` is set.

`python scripts/release_validation_extended.py --with-live-smokes` can also run the optional live smoke follow-ups, including restart-safe memory churn and provider matrix checks, after the fast release gate passes.

That extended suite adds slower checks such as fresh wheel-install validation
and extra deferred-startup/restart coverage without bloating the main smoke
gate.

## Current Boundaries
- Remote recommendation quality still depends on trustworthy metadata quality.
- Discovery is proposal-only; it does not automatically change canonical
  recommendations.
- External pack discovery metadata is advisory only; safe import still supports
  portable text skills only.
- There is no unrestricted shell surface and no unrestricted filesystem
  mutation.
- The release bundle and Debian package are supported optional install paths;
  legacy root/system packaging is out of scope for this release.
- Foreign code/plugin packs are not executable.
- Automatic switching, automatic installing, and automatic proposal adoption are
  out of scope for this release.
- The release smoke suite is intentionally compact and does not attempt live
  network/provider smoke coverage or real reboot automation.

## Source Of Truth
Use docs in this order when context conflicts:
1. `README.md`
2. `PRODUCT_RUNTIME_SPEC.md`
3. `PROJECT_STATUS.md`
4. `docs/operator/*`
5. `docs/design/*`

## More Context
- Product/runtime scope: `PRODUCT_RUNTIME_SPEC.md`
- Current architecture/handover: `PROJECT_STATUS.md`
- Operator guides: `docs/operator/`
