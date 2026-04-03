# UI Surface Report

This document is a fast handoff for another GPT or engineer. It describes what the current web UI contains, how it is organized, and which backend operations it exposes.

## High-Level Shape

- The UI is a single-page tabbed dashboard, not a multi-route app.
- The top-level shell lives in `desktop/src/App.jsx`.
- `App.jsx` owns most runtime state, fetches API data, and passes state/handlers into extracted tab components.
- Tab components live in `desktop/src/components/`.
- Shared presentational helpers live in `desktop/src/components/DetailRow.jsx` and `desktop/src/components/ModelStatusRow.jsx`.
- Shared pure helpers live in `desktop/src/lib/uiHelpers.js` and `desktop/src/lib/providerModelHelpers.js`.

## Visual System

- Warm paper-like background with teal accent color.
- Rounded cards inside a bordered main panel.
- `Space Grotesk` typography.
- Badge-based status language for health, provider state, capabilities, and chat metadata.
- Primary, secondary, and danger button treatments are visually distinct.

Main styling file: `desktop/src/styles.css`

## Top-Level Navigation

The UI has 8 tabs:

1. `Defaults`
2. `Operations`
3. `Providers`
4. `Telegram`
5. `Permissions`
6. `Model Scout`
7. `Chat`
8. `Logs/Debug`

There is no client-side router. Tab switching is driven by `activeTab` in `App.jsx`.

## Runtime Data Loaded On Startup

On initial refresh, the UI loads:

- `/providers`
- `/models`
- `/defaults`
- `/telegram/status`
- `/model_scout/status`
- `/model_scout/suggestions`
- `/llm/health`
- `/llm/catalog?limit=50`
- `/llm/catalog/status`
- `/llm/notifications?limit=20`
- `/llm/notifications/status`
- `/llm/notifications/policy`
- `/llm/autopilot/ledger?limit=10`
- `/llm/registry/snapshots?limit=20`
- `/permissions`
- `/audit?limit=20`

This means the UI is not a mock shell. It is built around live operational state.

## Tab Inventory

### 1. Defaults

Component: `desktop/src/components/SetupTab.jsx`

Contains:

- Routing mode selector
- Default provider selector
- Default model selector
- `Allow remote fallback when local candidates fail` toggle
- `Save Defaults` action
- `Refresh Models` action
- Provider recommendation notes
- `Model Quick View` list showing model health/status

Purpose:

- Set the default routing behavior
- Show the current model inventory at a glance

### 2. Operations

Component: `desktop/src/components/OperationsTab.jsx`

This is the operational control center. It has two large sections: `LLM Health + Automation` and `Support`.

#### LLM Health + Automation

Contains:

- Last health run details
- Health counts: `ok`, `degraded`, `down`
- Scheduler timing for health, catalog, refresh, bootstrap, reconcile, hygiene, cleanup, self-heal, autoconfig, scout
- `Run Health Check`
- `Run Catalog Refresh`
- Recent autopilot actions
- Recent notifications feed
- Notification policy badge
- `Send Test Notification`
- `Explain last autopilot change`
- `Mark Latest Read`
- Last autopilot change summary
- Autoconfig plan/apply controls
- Hygiene plan/apply controls
- Catalog preview rows
- Cleanup plan/apply controls
- Capabilities reconcile plan/apply controls
- Safety state, blocked reasons, churn info
- Rollback and bootstrap policy display
- `Undo Last Autopilot Change`
- `Bootstrap Defaults`
- Autopilot ledger entries
- Registry snapshot list with per-snapshot rollback actions

#### Support

Contains:

- Deterministic support bundle export
- Diagnose target selector
- Remediation intent selector
- `Export Support Bundle`
- `Run Diagnosis`
- `Fix LLM setup`
- `Execute safe steps`
- Diagnosis result view
- Remediation plan view
- Remediation execution result view

Purpose:

- Surface health/automation state
- Let operators plan and apply safe remediation flows
- Expose rollback/bootstrap/autopilot actions from the UI

### 3. Providers

Component: `desktop/src/components/ProvidersTab.jsx`

Contains three main surfaces.

#### Add Provider

- Preset selector for `Custom OpenAI-Compatible`, `OpenRouter`, and `Ollama (Local)`
- Provider id
- Base URL
- Chat path
- API key input
- Initial model
- `Local provider` toggle
- `Enabled` toggle
- Default headers JSON
- Default query params JSON
- `Save Provider`
- `Save + Test`

#### Models

- Provider selector
- `Fetch Models`
- `Refresh All`
- Model list for the selected provider
- Manual model add form:
  - model name
  - capabilities
  - `Add Model Manually`

#### Existing Provider Cards

For each provider, the card exposes:

- Editable base URL and chat path
- `Enabled` and `Local` toggles
- Secure API key input
- `Save`
- `Set Key`
- `Test`
- `Delete`
- Provider health summary and last error
- API key source info
- Compact provider model summary badges:
  - total models
  - available models
  - routable models
  - issue count
- Capability summary badges
- Expand/collapse model inspector
- Per-provider model search
- Per-provider model status filter:
  - all
  - available
  - routable
  - needs attention
- Expanded model list rendered as rows instead of one long comma string

Purpose:

- Configure providers
- Test provider connectivity
- Manage provider model inventory at a usable scale

### 4. Telegram

Component: `desktop/src/components/TelegramTab.jsx`

Contains:

- Telegram configured/not-configured status
- Secure bot token input
- `Save Token`
- `Test Telegram`

Purpose:

- Configure and validate Telegram delivery for notifications

### 5. Permissions

Component: `desktop/src/components/PermissionsTab.jsx`

Contains:

- Confirmation mode selector:
  - require confirmation
  - auto-apply allowed actions
- Friendly permission entries with:
  - human-readable action label
  - short description
  - raw action id for audit/debug
- Download budget constraint
- `Allow installing Ollama automatically`
- `Allow remote models and providers`
- Allowed providers list
- Allowed model patterns list
- `Save Permissions`
- `Recent Audit Events` list

Current action metadata comes from `MODELOPS_ACTIONS` in `desktop/src/App.jsx`.

Purpose:

- Control what automation is allowed to change
- Keep raw ids visible without forcing operators to read only internal jargon

### 6. Model Scout

Component: `desktop/src/components/ModelScoutTab.jsx`

Contains:

- Scout readiness/status
- Suggestion counts
- Source health for Hugging Face, Ollama, and OpenRouter
- `Run Scout`
- Suggestion list with:
  - kind badge
  - score
  - status
  - rationale
  - optional install command
- Per-suggestion actions:
  - `Try This Model`
  - `Confirm Execute`
  - `Dismiss`
  - `Mark Installed`
- ModelOps decision preview after planning

Purpose:

- Recommend local/remote models
- Let the operator plan and then explicitly execute model operations

### 7. Chat

Component: `desktop/src/components/ChatTab.jsx`

Contains:

- Scrollable chat transcript
- Message bubbles for user and assistant
- Assistant metadata badges:
  - provider/model
  - fallback used
  - autopilot notification outcome
  - count of new ops since last user message
- Draft input
- `Send`
- `Reset`
- `Export`
- Provider override selector
- Model override selector

Purpose:

- Chat with the agent
- Inspect which provider/model handled the response
- Override routing for a conversation

### 8. Logs/Debug

Component: `desktop/src/components/DebugTab.jsx`

Contains:

- `Recent Requests` list
- Each row shows:
  - timestamp
  - endpoint
  - success/error
  - detail text

Purpose:

- Lightweight in-UI request log for troubleshooting

## Shared UI Patterns

- `DetailRow` is the generic bordered row shell with title, badge, and meta lines.
- `ModelStatusRow` is a model-focused wrapper around `DetailRow`.
- `model-list` containers are scrollable bounded panels used across tabs.
- Status is communicated through badge colors rather than large charts or tables.

## Current Responsive Behavior

The UI has explicit responsive improvements already in place:

- Tabs wrap under narrower widths
- Two-column grids collapse to one column under `900px`
- Action rows wrap under `900px`
- Provider model filter controls collapse to one column under `900px`
- Chat input/button stack under `640px`
- Model row headers can wrap cleanly

## Main Backend Operations Exposed By The UI

The UI can drive these operation groups:

- defaults save and model refresh
- provider create/update/delete/test/secret/model refresh/manual model add
- chat send
- Telegram save/test
- model scout run/dismiss/mark installed
- LLM health run
- catalog refresh
- autoconfig plan/apply
- hygiene plan/apply
- cleanup plan/apply
- capabilities reconcile plan/apply
- registry rollback
- notifications test/mark read
- autopilot explain/undo/bootstrap
- support bundle export/diagnose/remediate
- permissions update
- modelops plan/execute

## Architecture Notes

- `App.jsx` is still the orchestration layer, but the large view blocks have been extracted into per-tab components.
- The UI is intentionally operational and admin-oriented, not consumer-facing.
- There is no auth UI, wizard flow, modal system, or multi-user concept visible in the current frontend.
- The strongest complexity centers are `Operations` and `Providers`.

## File Map

- `desktop/src/App.jsx`
- `desktop/src/components/SetupTab.jsx`
- `desktop/src/components/OperationsTab.jsx`
- `desktop/src/components/ProvidersTab.jsx`
- `desktop/src/components/TelegramTab.jsx`
- `desktop/src/components/PermissionsTab.jsx`
- `desktop/src/components/ModelScoutTab.jsx`
- `desktop/src/components/ChatTab.jsx`
- `desktop/src/components/DebugTab.jsx`
- `desktop/src/components/DetailRow.jsx`
- `desktop/src/components/ModelStatusRow.jsx`
- `desktop/src/lib/uiHelpers.js`
- `desktop/src/lib/providerModelHelpers.js`
- `desktop/src/styles.css`

## Short Summary

If another GPT needs the shortest possible description: this UI is a single-page admin console for managing LLM providers, routing defaults, model inventory, autopilot/health automation, Telegram notifications, permissions, model scouting, chat, and request logs against a live local API.
