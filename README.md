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
Canonical install path:
- code checkout: `~/personal-agent`
- mutable state: `~/.local/share/personal-agent`
- operator config/policy: `~/.config/personal-agent`
- user service: `~/.config/systemd/user/personal-agent-api.service`

1. Clone or move the repo to `~/personal-agent`, then:
   - `cd ~/personal-agent`
2. Create a virtualenv and install the package:
   - `python3 -m venv .venv`
   - `. .venv/bin/activate`
   - `pip install -e .`
3. Install the user service:
   - `mkdir -p ~/.config/systemd/user`
   - `ln -sf ~/personal-agent/systemd/personal-agent-api.service ~/.config/systemd/user/personal-agent-api.service`
   - `systemctl --user daemon-reload`
4. Ensure the user service can survive reboot:
   - `loginctl enable-linger "$USER"`
5. Enable and start the runtime:
   - `systemctl --user enable --now personal-agent-api.service`
6. Run first-run setup and diagnostics:
   - `python -m agent setup`
   - `python -m agent doctor`
7. Run the release smoke suite:
   - if you are validating this install or preparing a release:
     - `python scripts/release_smoke.py`
   - for a live answer-shape check of RAM/VRAM observation:
     - `python scripts/hardware_observe_smoke.py`

Upgrade path:
- `cd ~/personal-agent`
- `git pull --ff-only`
- `. .venv/bin/activate`
- `pip install -e .`
- `python -m agent doctor --fix`
- `systemctl --user daemon-reload`
- `systemctl --user restart personal-agent-api.service`
- `python -m agent status`

Recovery/reset path:
- `python -m agent setup`
- `python -m agent doctor --fix`
- `systemctl --user restart personal-agent-api.service`
- `python -m agent status`

Uninstall path:
- `systemctl --user disable --now personal-agent-api.service`
- `rm -f ~/.config/systemd/user/personal-agent-api.service`
- `systemctl --user daemon-reload`
- optionally remove `~/.config/personal-agent` and `~/.local/share/personal-agent`

Legacy `install.sh`, `uninstall.sh`, and `doctor.sh` are intentionally retired
and fail closed. Use the user-service path above instead.

If you are upgrading an older repo-local install, `python -m agent doctor --fix`
copies legacy `memory/agent.db` and `logs/agent.jsonl` into the canonical
state directory before restart.

Useful local commands:
- `python -m agent status`
- `python -m agent doctor`
- `python -m agent health`
- `python -m agent llm_inventory`
- `python -m agent memory`

Diagnostics / recovery path:
- `python -m agent setup`
  - canonical first-run and guided recovery surface
- `python -m agent doctor`
  - deterministic local diagnostics
- `python -m agent doctor --collect-diagnostics`
  - one redacted local diagnostics bundle for support/debugging
- `python -m agent doctor --fix`
  - safe local repair for missing dirs, drop-ins, and legacy state migration

Memory operator surfaces:
- `python -m agent memory`
  - plain resumable-state summary for the current thread
  - if continuity memory is degraded, the summary says so explicitly instead of silently repairing it
- continuity persistence remains full-record replace, but it is now revision-aware:
  - per-key optimistic concurrency control is enforced at the storage write boundary
  - successful writes increment a stored revision
  - stale cross-runtime writes are rejected instead of silently overwriting newer state
  - there is no merge-on-write path
  - there is no cross-key atomic snapshot or cross-key merge behavior
  - a stale runtime must reload before retrying a rejected save
- `GET /memory/status` (loopback only)
  - canonical inspect surface for deterministic continuity memory, optional `memory_v2`, and optional semantic memory
  - includes current continuity revisions, last attempted write outcome, last successful write outcome, and last stale-write conflict metadata
  - conflict metadata is observable, not auto-resolved
- `POST /memory/reset` (loopback only)
  - explicit preview + confirm reset surface
  - supported components: `continuity`, `memory_v2`, `semantic`, or `all`
  - no memory component is erased until `confirm=true`

## Release Artifacts
- Canonical packaging metadata lives in `pyproject.toml`.
- Canonical version truth lives in `VERSION`.
- Canonical release build command:
  - `python scripts/build_dist.py --outdir dist --clean`
- Expected artifacts:
  - `dist/personal_agent-<version>-py3-none-any.whl`
  - `dist/personal_agent-<version>.tar.gz`
- Canonical packaged CLI entry points:
  - `personal-agent`
  - `personal-agent-api`
  - `personal-agent-telegram`
- The canonical long-running service install for this release is still the repo
  checkout plus user-systemd path above.
- Debian/system packaging is not supported for this release.
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
- `python scripts/release_smoke.py`

It is the canonical fast release gate. It runs a fixed deterministic set of
tests covering:
- fresh-install and first-run path checks
- health/readiness/runtime restart truth
- chat/tool golden-path behavior
- basic memory inspect/degrade behavior
- safe external pack discovery/preview/install blocked-path behavior
- redacted diagnostics/no-secrets sanity

A passing run means the core supported product path is still coherent and the
main safety/recovery gates are intact.

If you want the heavier follow-up validation path, run:
- `python scripts/release_validation_extended.py`

For live, non-blocking operator smoke checks on a running system, use:
- `python scripts/hardware_observe_smoke.py`
- `python scripts/live_product_smoke.py`
- `python scripts/webui_smoke.py`

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
- Debian/system packaging is out of scope for this release.
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
