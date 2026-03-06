# Personal Agent (Telegram-first)

Local-first personal assistant with SQLite memory, deterministic routing, and systemd-managed scheduling.

## Documentation Order (Source of Truth)
Use docs in this order when context conflicts:
1. `README.md`
2. `ARCHITECTURE.md`
3. `PROJECT_STATUS.md`
4. `docs/operator/*`
5. `docs/design/*`
6. `docs/history/*`

## Getting Started In 60 Seconds
1. Create venv:
   - `python3 -m venv .venv`
   - `. .venv/bin/activate`
2. Install deps:
   - `pip install -r requirements.txt`
3. Start API service:
   - `systemctl --user restart personal-agent-api.service`
4. Verify runtime:
   - `python -m agent status`
   - `python -m agent doctor`

Unified CLI:
- `python -m agent doctor`
- `python -m agent status`
- `python -m agent health`
- `python -m agent brief`
- `python -m agent logs`
- `python -m agent version`

## Golden Path
1. Start/restart the API service.
2. Verify with `python -m agent status`.
3. Talk to Telegram in plain English.
4. If anything looks wrong, run `python -m agent doctor` and follow the single `Next action`.

Runtime contract (all surfaces use the same mode names):
- `READY`: normal operation.
- `BOOTSTRAP_REQUIRED`: setup guidance only.
- `DEGRADED`: partial operation, read-only checks still work.
- `FAILED`: deterministic error block with trace id + one next step.

## If You Only Learn 3 Commands
- `python -m agent status`
- `python -m agent doctor`
- `python -m agent logs`

## First 5 Minutes
1. Set Telegram token: `python -m agent.secrets set telegram:bot_token`
2. Restart Telegram service: `systemctl --user restart personal-agent-telegram.service`
3. Verify: `python -m agent status`
4. Diagnose if needed: `python -m agent doctor`
5. Send Telegram message: `help`

## Troubleshooting
- First step for any issue: `python -m agent doctor`
- Detailed machine-readable output: `python -m agent doctor --json`
- Safe local remediation only: `python -m agent doctor --fix`

## Local API + Web UI
Run local HTTP API (no Telegram token required):
- `. .venv/bin/activate`
- `.venv/bin/python -m agent.api_server --host 127.0.0.1 --port 8765`
- Open browser: `http://127.0.0.1:8765`

Endpoints:
- `GET /health`
- `GET /version`
- `GET /telegram/status`
- `GET /models`
- `POST /chat`
- `GET /config`
- `PUT /config`
- `GET /providers`
- `POST /providers`
- `PUT /providers/{id}`
- `DELETE /providers/{id}`
- `POST /providers/{id}/secret`
- `POST /providers/{id}/test`
- `POST /providers/{id}/models`
- `POST /providers/{id}/models/refresh`
- `GET /defaults`
- `PUT /defaults`
- `POST /models/refresh`
- `POST /telegram/secret`
- `POST /telegram/test`
- `GET /model_scout/status`
- `GET /model_scout/suggestions`
- `GET /model_scout/sources`
- `POST /model_scout/run`
- `POST /model_scout/suggestions/{id}/dismiss`
- `POST /model_scout/suggestions/{id}/mark_installed`
- `GET /llm/health`
- `POST /llm/health/run`
- `GET /llm/catalog`
- `GET /llm/catalog/status`
- `POST /llm/catalog/run`
- `POST /llm/capabilities/reconcile/plan`
- `POST /llm/capabilities/reconcile/apply`
- `POST /llm/autoconfig/plan`
- `POST /llm/autoconfig/apply`
- `POST /llm/hygiene/plan`
- `POST /llm/hygiene/apply`
- `POST /llm/cleanup/plan`
- `POST /llm/cleanup/apply`
- `POST /llm/self_heal/plan`
- `POST /llm/self_heal/apply`
- `GET /llm/autopilot/ledger?limit=N`
- `GET /llm/autopilot/ledger/{id}`
- `GET /llm/autopilot/explain_last`
- `POST /llm/autopilot/undo`
- `POST /llm/autopilot/bootstrap`
- `GET /llm/registry/snapshots?limit=N`
- `POST /llm/registry/rollback`
- `GET /llm/notifications?limit=N`
- `GET /llm/notifications/status`
- `GET /llm/notifications/last_change`
- `GET /llm/notifications/policy`
- `POST /llm/notifications/test`
- `POST /llm/notifications/mark_read`
- `POST /llm/notifications/prune`
- `GET /llm/support/bundle`
- `GET /llm/support/diagnose?id=<provider_or_model_id>`
- `POST /llm/support/remediate/plan`
- `GET /permissions`
- `PUT /permissions`
- `GET /audit`
- `POST /modelops/plan`
- `POST /modelops/execute`

UI behavior:
- `GET /` serves the local web UI from `agent/webui/dist`.
- Static assets (`/assets/*`) are served by the same API process.
- Runtime installs do not require Node or Rust.
- `Model Scout` tab shows recommendation-only model suggestions and lets you dismiss/mark installed.
- `Permissions` tab controls constrained ModelOps autonomy and shows recent audit entries.

Web UI build (dev-time only):
1. `./scripts/build_webui.sh`
2. This runs:
   - `npm ci`
   - `npm run build`
3. Build source lives in `desktop/`; output is written to `agent/webui/dist/`.

Optional dev mode:
- `WEBUI_DEV_PROXY=1 .venv/bin/python -m agent.api_server`
- Then open the dev server URL shown on `/` (default `http://127.0.0.1:1420`).
- For hot reload: `cd desktop && npm run dev` (requests use same-origin paths and Vite proxies API routes to `127.0.0.1:8765`).

Check running API identity:
- `curl -s http://127.0.0.1:8765/health`
- `curl -s http://127.0.0.1:8765/version`
- `curl -s http://127.0.0.1:8765/telegram/status`

## API User Service (systemd)
Install as a user service:
- `mkdir -p ~/.config/systemd/user`
- `cp systemd/personal-agent-api.service ~/.config/systemd/user/`
- `systemctl --user daemon-reload`
- `systemctl --user enable --now personal-agent-api.service`

Or use helper script:
- `./scripts/install_user_service.sh`

Check status/logs:
- `systemctl --user status personal-agent-api.service`
- `journalctl --user -u personal-agent-api.service -f`

Telegram token setup (UI/API):
- Open `http://127.0.0.1:8765` and use the `Telegram` tab, or:
- `curl -X POST http://127.0.0.1:8765/telegram/secret -H 'Content-Type: application/json' -d '{"bot_token":"<token>"}'`
- `curl -X POST http://127.0.0.1:8765/telegram/test`

Secret storage:
- Primary: OS keychain via `keyring` (if available in runtime environment)
- Fallback: encrypted local file at `~/.local/share/personal-agent/secrets.enc.json`
- Telegram bot token key: `telegram:bot_token`

Provider/model registry:
- File: `llm_registry.json` (schema v2, JSON on disk)
- Backward compatibility: v1 files are loaded via migration/compat logic at runtime.

Add any OpenAI-compatible provider:
1. Open `http://127.0.0.1:8765` and go to `Providers`.
2. Use `Add Provider (OpenAI-Compatible)`:
   - `id` (e.g. `openrouter`, `myrouter`)
   - `base_url` (e.g. `https://openrouter.ai/api/v1`)
   - `chat_path` (default `/v1/chat/completions`; OpenRouter preset uses `/chat/completions`)
   - optional default headers/query params as JSON objects
   - optional API key (stored in secret store as `provider:<id>:api_key`)
3. Save and test. Then fetch models or add a model manually if listing is unsupported.

Examples:
- OpenRouter:
  - `base_url=https://openrouter.ai/api/v1`
  - `chat_path=/chat/completions`
- Generic router/OpenAI-compatible:
  - `base_url=https://api.example.com`
  - `chat_path=/v1/chat/completions`

Routing (high-level):
- `auto`: balanced quality/cost ordering.
- `prefer_cheap`: prioritize lower-cost models.
- `prefer_best`: prioritize higher-quality models.
- `prefer_local_lowest_cost_capable`: local-capable models first; otherwise lowest expected token-cost among capable remote models.
  - Expected cost uses rolling usage averages per `(task_type, provider, model)` from `llm_usage_stats.json` (next to DB by default).
  - Local models with unknown pricing are treated as zero-cost for ranking.

Model Scout v1 (recommend-only):
- Scans Hugging Face trending models (`/api/trending?type=model`) and proposes local GGUF/Ollama candidates first.
- Uses local backends for discovery:
  - Ollama: reads local `/api/tags` for installed models.
  - OpenRouter: reads `/models` when key source is configured.
- Adds remote suggestions only for enabled/tested remote providers, based on expected cost + health.
- Never auto-installs models and never auto-changes defaults.
- Uses deterministic scoring, dedupe, and cooldown to avoid spam.
- Storage: SQLite tables in `agent.db` when available; JSON fallback at `~/.local/share/personal-agent/model_scout_state.json`.

LLM catalog/health/autoconfig/hygiene/cleanup:
- Catalog store persists provider model metadata (capabilities, context window, pricing when available) at `LLM_CATALOG_PATH` (default `~/.local/share/personal-agent/llm_catalog.json`).
- Catalog endpoints:
  - `GET /llm/catalog` returns normalized catalog rows (optional provider filter + limit).
  - `GET /llm/catalog/status` returns last refresh/error per provider.
  - `POST /llm/catalog/run` triggers a refresh and syncs catalog metadata into registry model rows.
- Health monitor persists provider/model status at `LLM_HEALTH_STATE_PATH` (default `~/.local/share/personal-agent/llm_health_state.json`).
- Router skip-list avoids candidates marked down/degraded during active cooldown windows.
- Capabilities reconcile endpoints:
  - `POST /llm/capabilities/reconcile/plan` computes deterministic capability mismatch fixes from catalog inference.
  - `POST /llm/capabilities/reconcile/apply` applies capability/default_for fixes via transactional write + ledger + audit.
- Autoconfig endpoints:
  - `POST /llm/autoconfig/plan` returns deterministic proposed defaults/provider toggles.
  - `POST /llm/autoconfig/apply` applies plan via permission gate + audit.
- Hygiene endpoints:
  - `POST /llm/hygiene/plan` returns deterministic registry cleanup diff.
  - `POST /llm/hygiene/apply` applies cleanup via permission gate + audit.
- Cleanup endpoints:
  - `POST /llm/cleanup/plan` returns deterministic prune/disable plan from usage + health + catalog.
  - `POST /llm/cleanup/apply` applies cleanup via `llm.registry.prune` permission gate and scheduler policy.

LLM Autopilot loop (API process, deterministic):
- Sequence per scheduler cycle:
  1. `POST /models/refresh` logic (provider inventory refresh; Ollama uses `/api/tags` as source of truth)
  2. `POST /llm/catalog/run` logic (authoritative provider catalogs + deterministic metadata sync)
  3. `POST /llm/capabilities/reconcile/plan` + `POST /llm/capabilities/reconcile/apply` (if permissions/policy allow)
  4. `POST /llm/health/run` (provider-specific probes + persisted cooldown/backoff)
  5. `POST /llm/hygiene/plan` + `POST /llm/hygiene/apply` (if permissions allow)
  6. `POST /llm/cleanup/plan` + `POST /llm/cleanup/apply` (if permissions allow/policy allows)
  7. `POST /llm/self_heal/plan` + `POST /llm/self_heal/apply` (if drift exists; if allowed)
  8. `POST /llm/autoconfig/plan` + `POST /llm/autoconfig/apply` (if permissions allow)
- Conservative default behavior:
  - keep current healthy/routable defaults unchanged
  - detect and report defaults drift in `GET /llm/health` under `health.drift`
  - self-heal defaults when current default provider/model drifts (missing, unavailable, unroutable, non-chat, unhealthy)
  - repair missing/invalid defaults deterministically (prefer local chat-capable model, then remote if allowed/available)
  - normalize `default_model` to fully-qualified `provider:model`
- Safety model:
  - apply actions are permission-gated (`llm.autoconfig.apply`, `llm.hygiene.apply`, `llm.self_heal.apply`, `llm.registry.prune`, `llm.capabilities.reconcile.apply`)
  - scheduler capability-reconcile auto-apply is only allowed by default on loopback bindings when `LLM_CAPABILITIES_RECONCILE_ALLOW_APPLY` is unset
  - scheduler self-heal auto-apply is only allowed by default on loopback bindings when `LLM_SELF_HEAL_ALLOW_APPLY` is unset
  - scheduler cleanup auto-apply is only allowed by default on loopback bindings when `LLM_REGISTRY_PRUNE_ALLOW_APPLY` is unset
  - every catalog/health/autoconfig/hygiene/cleanup/self-heal run emits an audit record with decision, outcome, reason, duration, and modified ids
  - cleanup/hygiene never remove secrets; they only update registry entries

Autopilot safety + rollback:
- Apply actions (`llm.autoconfig.apply`, `llm.hygiene.apply`, `llm.cleanup.apply`, `llm.self_heal.apply`, `llm.capabilities.reconcile.apply`) use transactional registry writes:
  - snapshot before apply
  - atomic write/replace
  - post-write invariant verification
  - automatic restore if verification fails
- Every successful transactional apply now records:
  - `snapshot_id_before`
  - `snapshot_id_after` (post-apply snapshot, best-effort)
  - `resulting_registry_hash`
  - stable-sorted `changed_ids`
- Snapshot API:
  - `GET /llm/registry/snapshots?limit=N`
  - `POST /llm/registry/rollback` with `{ "snapshot_id": "..." }`
- Rollback policy:
  - loopback binding + `LLM_REGISTRY_ROLLBACK_ALLOW` unset: auto-allow
  - non-loopback: requires `llm.registry.rollback` permission
- One-click undo API:
  - `POST /llm/autopilot/undo`
  - resolves latest successful autopilot apply action and rolls back to its `snapshot_id_before`
  - uses existing rollback policy/permission checks and audits `llm.autopilot.undo`
- Safe mode (`LLM_AUTOPILOT_SAFE_MODE=1` default):
  - allows local repairs (mark unroutable, fix local defaults)
  - blocks remote activation changes:
    - enabling remote providers/models
    - switching defaults to remote provider/model
    - enabling remote fallback from false->true
  - blocked actions are reported in health under `health.autopilot.last_blocked_reason`
- Churn escalation:
  - runtime tracks recent autopilot applies and detects churn (`LLM_AUTOPILOT_CHURN_MIN_APPLIES` within `LLM_AUTOPILOT_CHURN_WINDOW_SECONDS`)
  - on churn, runtime enters a persisted safe-mode override (`LLM_AUTOPILOT_STATE_PATH`) and pauses apply phases until operator intervention
  - emits audit action `llm.autopilot.safe_mode.enter` and notification context
- Bootstrap defaults:
  - `POST /llm/autopilot/bootstrap` deterministically selects a local chat-capable healthy model when defaults are unset/unroutable
  - loopback + `LLM_AUTOPILOT_BOOTSTRAP_ALLOW_APPLY` unset: auto-allow
  - non-loopback: requires `llm.autopilot.bootstrap.apply` permission
- Explain endpoint:
  - `GET /llm/autopilot/explain_last`
  - returns latest successful autopilot apply with rationale lines, stable `changed_ids`, snapshot/hash metadata, and redacted evidence subset
- Action ledger API (UI-friendly):
  - `GET /llm/autopilot/ledger?limit=N`
  - `GET /llm/autopilot/ledger/{id}`
  - entries include action, decision/outcome/reason, `snapshot_id_before`, `snapshot_id_after`, `resulting_registry_hash`, and stable-sorted `changed_ids`

Autopilot notifications:
- When scheduler cycles mutate defaults/provider/model state, it builds one combined natural-language notification for that cycle.
- Diff source is deterministic and limited to:
  - defaults: `routing_mode`, `default_provider`, `default_model`, `allow_remote_fallback`
  - providers: `enabled`, `available`, `health.status`, `health.cooldown_until`, `health.down_since`, `health.failure_streak`
  - models: `enabled`, `available`, `routable`, `health.status`, `health.cooldown_until`, `health.down_since`, `health.failure_streak`
- Delivery targets (deterministic order):
  - `telegram` target first when configured and remote send policy permits.
  - `local` target fallback (always configured) so UI still shows delivered notifications when Telegram is absent/unavailable.
- Send policy:
  - loopback binding + `LLM_NOTIFICATIONS_ALLOW_SEND` unset: auto-allow (developer default)
  - non-loopback: remote send requires `llm.notifications.send` permission (local target still records delivery)
- Anti-spam:
  - rate limit (`AUTOPILOT_NOTIFY_RATE_LIMIT_SECONDS`, default `1800`)
  - dedupe by stable change hash (`AUTOPILOT_NOTIFY_DEDUPE_WINDOW_SECONDS`, default `86400`)
  - quiet hours defer Telegram send but still record (`AUTOPILOT_NOTIFY_QUIET_START_HOUR` / `AUTOPILOT_NOTIFY_QUIET_END_HOUR`)
  - retention/compaction in store:
    - `LLM_NOTIFICATIONS_MAX_ITEMS` (default `200`)
    - `LLM_NOTIFICATIONS_MAX_AGE_DAYS` (default `30`)
    - `LLM_NOTIFICATIONS_COMPACT` (default `1`; keeps only recent repeated `no_changes`/same-diff groups)
- API:
  - `GET /llm/notifications?limit=N`
  - `GET /llm/notifications/status`
    - includes `last_read_hash` and `unread_count`
  - `GET /llm/notifications/last_change`
    - returns the latest actionable change summary (skips no-op/rate-limit/dedupe/quiet-hour rows)
  - `GET /llm/notifications/policy` (shows effective test policy from loopback/knob logic)
  - `POST /llm/notifications/test`
  - `POST /llm/notifications/mark_read` with `{ "hash": "<dedupe_hash>" }`
  - `POST /llm/notifications/prune` (permission-gated by `llm.notifications.prune`)
    - Requires explicit policy allow for `llm.notifications.prune` (no loopback auto-allow path).
  - Web UI shows this policy as a badge in the Autopilot Notifications card.
- `GET /llm/health` includes last scheduler notify outcome/hash in `health.notifications`.
- `GET /llm/health` includes deterministic defaults drift report in `health.drift`.

Chat response meta now includes an ops summary:
- `meta.autopilot.last_notification`:
  - `hash`, `ts_iso`, `title`, `outcome`, `reason`, `delivered_to`
- `meta.autopilot.since_last_user_message`:
  - count of notifications with `ts > chat_request_start_ts`

Support diagnostics (local-only, deterministic):
- `GET /llm/support/bundle`
  - exports a redacted local support bundle (defaults/providers/models/health/audit/ledger/notifications/policies)
  - no network calls; no secret values
- `GET /llm/support/diagnose?id=<provider_id_or_model_id>`
  - explains provider/model failure state from stored validation + health + catalog evidence
  - includes stable `root_causes` and safe `recommended_actions`
- `POST /llm/support/remediate/plan`
  - returns a plan-only remediation sequence for `fix_routing`, `reduce_churn`, or `bootstrap`
  - never applies changes and never writes registry state

Background automation (API process):
- Enabled by default when `LLM_AUTOMATION_ENABLED=1`.
- Jobs:
  - provider/model refresh every `LLM_HEALTH_INTERVAL_SECONDS` (default `900`)
  - bootstrap check every `LLM_SELF_HEAL_INTERVAL_S` (default `86400`; first run shortly after startup)
  - catalog refresh every `LLM_CATALOG_REFRESH_INTERVAL_S` (default `21600`)
  - capability reconcile every `LLM_HEALTH_INTERVAL_SECONDS` (default `900`)
  - health probe every `LLM_HEALTH_INTERVAL_SECONDS` (default `900`)
  - hygiene every `LLM_HYGIENE_INTERVAL_SECONDS` (default `86400`)
  - cleanup every `LLM_HYGIENE_INTERVAL_SECONDS` (default `86400`)
  - model scout every `LLM_MODEL_SCOUT_INTERVAL_SECONDS` (default `86400`)
  - autoconfig every `LLM_AUTOCONFIG_INTERVAL_SECONDS` (default `604800`)
- Optional startup autoconfig: `LLM_AUTOCONFIG_RUN_ON_STARTUP=1`.
- Disable all background jobs with `LLM_AUTOMATION_ENABLED=0`.

Constrained autonomy (ModelOps only):
- Autonomy scope is limited to model-management actions:
  - `modelops.install_ollama`
  - `modelops.pull_ollama_model`
  - `modelops.import_gguf_to_ollama`
  - `modelops.set_default_model`
  - `modelops.enable_disable_provider_or_model`
  - `llm.autoconfig.apply`
  - `llm.hygiene.apply`
  - `llm.registry.prune`
  - `llm.registry.rollback`
  - `llm.self_heal.apply`
  - `llm.capabilities.reconcile.apply`
  - `llm.autopilot.bootstrap.apply`
  - `llm.notifications.test`
  - `llm.notifications.send`
  - `llm.notifications.prune`
- Default policy is deny for all actions.
- Exception: `llm.notifications.test` is auto-allowed on loopback-only API bindings when `LLM_NOTIFICATIONS_ALLOW_TEST` is unset (developer ergonomics).
- Exception: `llm.notifications.send` is auto-allowed for loopback-only API bindings when `LLM_NOTIFICATIONS_ALLOW_SEND` is unset.
- API flow is plan-first:
  - `POST /modelops/plan` returns deterministic steps + allow/deny decision.
  - `POST /modelops/execute` executes only if policy allows (and confirmation is provided in `manual_confirm` mode).
- No arbitrary shell execution: only whitelisted command paths are used by ModelOps executor.
- Audit is append-only and redacted.

Side-by-side staging run:
1. Use a different API port:
   - `.venv/bin/python -m agent.api_server --port 8876`
2. Use separate data/config paths:
   - `AGENT_DB_PATH=/tmp/personal-agent-staging/agent.db`
   - `LLM_REGISTRY_PATH=/tmp/personal-agent-staging/llm_registry.json`
   - `AGENT_SECRET_STORE_PATH=/tmp/personal-agent-staging/secrets.enc.json`
   - `LLM_USAGE_STATS_PATH=/tmp/personal-agent-staging/llm_usage_stats.json`

## Environment Variables
Telegram token source:
- Preferred: secret store key `telegram:bot_token` (configured via web UI/API)
- Backward-compatible fallback: `TELEGRAM_BOT_TOKEN`

Common optional:
- `AGENT_TIMEZONE` (default `America/Regina`)
- `AGENT_DB_PATH` (default `memory/agent.db`)
- `AGENT_LOG_PATH` (default `logs/agent.jsonl`)
- `AGENT_SKILLS_PATH` (default `skills/`)
- `AGENT_DOCTOR_REQUIRE_SYSTEMD_UNITS` (`1` makes `scripts/doctor.py` fail when required systemd units are missing; default skips these checks when units are not installed)
  - Doctor guide: `docs/operator/doctor.md` (`python -m agent doctor`, `--json`, `--fix`)
- `PERCEPTION_ENABLED` (default `1`)
- `PERCEPTION_ROOTS` (comma-separated allowlist roots for perception top-dir sizing; default `/home,/data/projects`)
- `PERCEPTION_INTERVAL_SECONDS` (default `5`; reserved for future background scheduling)
- `ENABLE_SCHEDULED_SNAPSHOTS` (`1` to enable periodic snapshots)
- `ENABLE_WRITES` (default off)

LLM/provider optional:
- `OPENAI_API_KEY`, `OPENAI_MODEL`
- `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`
- `OLLAMA_HOST`, `OLLAMA_MODEL`
- `LLM_REGISTRY_PATH` (defaults to `llm_registry.json` if present)
- `LLM_ROUTING_MODE` (`auto`, `prefer_cheap`, `prefer_best`, `prefer_local_lowest_cost_capable`)
- `LLM_RETRY_ATTEMPTS`
- `LLM_RETRY_BASE_DELAY_MS`
- `LLM_CIRCUIT_BREAKER_FAILURES`
- `LLM_CIRCUIT_BREAKER_WINDOW_SECONDS`
- `LLM_CIRCUIT_BREAKER_COOLDOWN_SECONDS`
- `LLM_USAGE_STATS_PATH` (default: `llm_usage_stats.json` next to DB)
- `LLM_CATALOG_PATH` (default `~/.local/share/personal-agent/llm_catalog.json`)
- `MODEL_SCOUT_ENABLED` (default `1`)
- `MODEL_SCOUT_NOTIFY_DELTA` (default `15`)
- `MODEL_SCOUT_ABSOLUTE_THRESHOLD` (default `80`)
- `MODEL_SCOUT_MAX_SUGGESTIONS_PER_NOTIFY` (default `2`)
- `MODEL_SCOUT_LICENSE_ALLOWLIST` (default `apache-2.0,mit,bsd-3-clause`)
- `MODEL_SCOUT_SIZE_MAX_B` (default `12`)
- `AGENT_MODEL_SCOUT_STATE_PATH` (JSON fallback path when SQLite is unavailable)
- `LLM_AUTOMATION_ENABLED` (default `1`)
- `LLM_HEALTH_INTERVAL_SECONDS` (default `900`)
- `LLM_HEALTH_MAX_PROBES_PER_RUN` (default `6`)
- `LLM_HEALTH_PROBE_TIMEOUT_SECONDS` (default `6`)
- `LLM_HEALTH_STATE_PATH` (default `~/.local/share/personal-agent/llm_health_state.json`)
- `LLM_CATALOG_REFRESH_INTERVAL_S` (default `21600`)
- `LLM_MODEL_SCOUT_INTERVAL_SECONDS` (default `86400`)
- `LLM_AUTOCONFIG_INTERVAL_SECONDS` (default `604800`)
- `LLM_AUTOCONFIG_RUN_ON_STARTUP` (default `0`)
- `LLM_HYGIENE_INTERVAL_SECONDS` (default `86400`)
- `LLM_HYGIENE_UNAVAILABLE_DAYS` (default `7`)
- `LLM_HYGIENE_REMOVE_EMPTY_DISABLED_PROVIDERS` (default `1`)
- `LLM_HYGIENE_DISABLE_REPEATEDLY_FAILING_PROVIDERS` (default `0`)
- `LLM_HYGIENE_PROVIDER_FAILURE_STREAK` (default `8`)
- `LLM_REGISTRY_PRUNE_ALLOW_APPLY` (optional: `true`/`false`)
  - unset: auto policy (`true` for loopback-only API binding, `false` otherwise)
  - `true`: always allow scheduler cleanup applies without `llm.registry.prune`
  - `false`: always require `llm.registry.prune`
- `LLM_REGISTRY_PRUNE_UNUSED_DAYS` (default `30`)
- `LLM_REGISTRY_PRUNE_DISABLE_FAILING_PROVIDER` (default `0`)
- `LLM_REGISTRY_SNAPSHOTS_DIR` (default `~/.local/share/personal-agent/registry_snapshots`)
- `LLM_REGISTRY_SNAPSHOT_MAX_ITEMS` (default `40`)
- `LLM_REGISTRY_ROLLBACK_ALLOW` (optional: `true`/`false`)
  - unset: auto policy (`true` for loopback-only API binding, `false` otherwise)
  - `true`: allow rollback without `llm.registry.rollback`
  - `false`: always require `llm.registry.rollback`
- `LLM_AUTOPILOT_SAFE_MODE` (default `1`)
- `LLM_AUTOPILOT_STATE_PATH` (default `~/.local/share/personal-agent/autopilot_state.json`)
- `LLM_AUTOPILOT_CHURN_WINDOW_SECONDS` (default `1800`)
- `LLM_AUTOPILOT_CHURN_MIN_APPLIES` (default `4`)
- `LLM_AUTOPILOT_CHURN_RECENT_LIMIT` (default `80`)
- `LLM_AUTOPILOT_BOOTSTRAP_ALLOW_APPLY` (optional: `true`/`false`)
  - unset: auto policy (`true` for loopback-only API binding, `false` otherwise)
  - `true`: allow `/llm/autopilot/bootstrap` without `llm.autopilot.bootstrap.apply`
  - `false`: always require `llm.autopilot.bootstrap.apply`
- `LLM_AUTOPILOT_LEDGER_PATH` (default `~/.local/share/personal-agent/autopilot_action_ledger.json`)
- `LLM_AUTOPILOT_LEDGER_MAX_ITEMS` (default `400`)
- `LLM_SELF_HEAL_INTERVAL_S` (default `86400`)
- `LLM_SELF_HEAL_ALLOW_APPLY` (optional: `true`/`false`)
  - unset: auto policy (`true` for loopback-only API binding, `false` otherwise)
  - `true`: always allow scheduler self-heal applies without `llm.self_heal.apply` permission action
  - `false`: always require `llm.self_heal.apply` permission action
- `LLM_CAPABILITIES_RECONCILE_ALLOW_APPLY` (optional: `true`/`false`)
  - unset: auto policy (`true` for loopback-only API binding, `false` otherwise)
  - `true`: always allow scheduler capability-reconcile applies without `llm.capabilities.reconcile.apply`
  - `false`: always require `llm.capabilities.reconcile.apply`
- `AUTOPILOT_NOTIFY_ENABLED` (default `1`)
- `AUTOPILOT_NOTIFY_RATE_LIMIT_SECONDS` (default `1800`)
- `AUTOPILOT_NOTIFY_DEDUPE_WINDOW_SECONDS` (default `86400`)
- `AUTOPILOT_NOTIFY_STORE_PATH` (default `~/.local/share/personal-agent/llm_notifications.json`)
- `AUTOPILOT_NOTIFY_QUIET_START_HOUR` (optional, 0-23)
- `AUTOPILOT_NOTIFY_QUIET_END_HOUR` (optional, 0-23)
- `LLM_NOTIFICATIONS_MAX_ITEMS` (default `200`)
- `LLM_NOTIFICATIONS_MAX_AGE_DAYS` (default `30`; `0` disables age-based pruning)
- `LLM_NOTIFICATIONS_COMPACT` (default `1`; `0` disables repeated-diff compaction)
- `LLM_NOTIFICATIONS_ALLOW_TEST` (optional: `true`/`false`)
  - unset: auto policy (`true` for loopback-only API binding, `false` otherwise)
  - `true`: always allow `/llm/notifications/test` without permission action toggle
  - `false`: always require `llm.notifications.test` permission action
- `LLM_NOTIFICATIONS_ALLOW_SEND` (optional: `true`/`false`)
  - unset: auto policy (`true` for loopback-only API binding, `false` otherwise)
  - `true`: always allow scheduler autopilot sends without `llm.notifications.send`
  - `false`: always require `llm.notifications.send` permission action

Desktop/API optional:
- `AGENT_API_HOST` (default `127.0.0.1`)
- `AGENT_API_PORT` (default `8765`)
- `AGENT_SECRET_STORE_PATH` (encrypted file fallback location)
- `AGENT_WEBUI_DIST_PATH` (default `agent/webui/dist`)
- `WEBUI_DEV_PROXY` (`1` to show dev server landing page on `/`)
- `WEBUI_DEV_URL` (default `http://127.0.0.1:1420`)
- `AGENT_PERMISSIONS_PATH` (default `~/.config/personal-agent/permissions.json`)
- `AGENT_AUDIT_LOG_PATH` (default `~/.local/share/personal-agent/audit.jsonl`)

## Legacy/Optional
- Legacy Tauri scaffold files remain under `desktop/src-tauri/`, but they are not used in the default install or runtime path.

## Install (systemd)
Recommended user install:
- `bash ops/install.sh --user`

System install:
- `bash ops/install.sh`

Uninstall:
- `bash ops/install.sh uninstall`

## Telegram Commands (Registered in `telegram_adapter/bot.py`)
- `/remind <YYYY-MM-DD HH:MM> | <text>`
- `/status`
- `/runtime_status`
- `/disk_grow [path]`
- `/audit`
- `/storage_snapshot`
- `/storage_report`
- `/resource_report`
- `/brief`
- `/network_report`
- `/weekly_reflection`
- `/today`
- `/task_add <title>`
- `/done <id>`
- `/open_loops [all|due|important]`
- `/health`
- `/daily_brief_status`
- `/ask <question>`
- `/ask_opinion <question>`
- `/scout`
- `/scout_dismiss <suggestion_id>`
- `/scout_installed <suggestion_id>`
- `/permissions` (ModelOps permissions summary)
- `/audit` (last 5 redacted ModelOps audit events)

Notes:
- `/task_add` also accepts advanced pipe syntax:
  - `/task_add <project> | <title> | <effort_mins> | <impact_1to5>`
- `/done` requires numeric task id.

## Daily Brief Scheduling
Daily brief scheduling is systemd-driven via:
- `ops/systemd/personal-agent-daily-brief.service`
- `ops/systemd/personal-agent-daily-brief.timer`

Entrypoint:
- `.venv/bin/python -m agent.scheduled_daily_brief`

User timer status:
- `systemctl --user status personal-agent-daily-brief.timer`

## Testing
- Full suite: `pytest -q`
- Current local result (2026-02-18): `486 passed`

## Architecture References
- `ARCHITECTURE.md`
- `STABILITY.md`
