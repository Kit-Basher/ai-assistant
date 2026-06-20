# Security Audit

Date: 2026-06-20
Checkpoint scope: `v0.2.0-chat-reliability-harness` plus release-readiness hardening pass.

This is an operator-focused security boundary audit. It records what is enforced
now, what remains release-blocking, and what must stay out of normal assistant
behavior.

## Current Rating

Yellow for controlled local trial. No new public release claim is made until the
fresh Debian VM install proof passes.

## Enforced Boundaries

- Secrets/tokens: doctor, status, support bundle, Telegram setup, provider/API
  key setup, and managed-action journals are expected to redact raw tokens,
  API keys, private paths where applicable, hostile pack text, and SearXNG
  `server.secret_key`.
- Plan Mode: known mutators require plan, preview, confirmation token, expiry,
  apply validation, journaling, verification, and scoped rollback where
  supported. Unknown operations default to mutating unless explicitly
  classified read-only.
- Search: SearXNG search is metadata-only. The assistant must not fetch pages,
  run JavaScript, download files, import packs, or treat result metadata as
  trusted fact. Search remains disabled until a trusted endpoint or managed
  loopback SearXNG setup is explicitly configured.
- External packs: text packs are quarantined, normalized, reviewed, approved,
  enabled, and permission-gated. External pack code execution is not supported.
  Search results cannot install/import packs directly.
- Managed SearXNG: the approved container image is
  `docker.io/searxng/searxng:latest`; bind is loopback-only; config mount is
  owned and seeded; empty config mounts and arbitrary volume paths are rejected;
  Docker is fallback-only; rootless Podman is preferred. Verified setup may
  persist only a loopback SearXNG runtime config; non-loopback persisted search
  config is rejected as untrusted and does not enable search.
- Shell/package/system mutations: normal chat may preview bounded package or
  managed-service actions only. It must not expose arbitrary shell, arbitrary
  Podman/Docker, host networking, broad filesystem writes, or hidden sudo.
- Telegram: optional by default. Inactive optional Telegram must not fail core
  readiness. Raw tokens must not appear in status/chat/doctor output.
- Backup/restore/support: backup paths include sensitive local state. Support
  bundles are redacted artifacts, not raw state exports.
- Local exposure: runtime and web UI are intended for loopback/local user
  operation. Public network exposure is outside the current release boundary.

## Cheap Safety Checks Now Covered

- `python scripts/external_pack_safety_smoke.py` covers hostile pack intake,
  lifecycle boundaries, managed-service tamper rejection, rollback scope, and
  support redaction.
- `python scripts/chat_eval.py` covers deterministic semantic routing,
  mutation-preview boundaries, stale-context escape, public lookup/search
  routing, no-search suppression, malformed input, and mixed prompt invariants.
- `python scripts/llm_behavior_eval.py` covers full orchestrator response
  invariants with mocked tools/status/search: no mutation without confirmation,
  no token/secret markers, no raw Podman/shell advice for managed services, no
  irrelevant pack hijack, no stale clarification loop, and no page
  fetch/browser/download/import claims.
- `python scripts/prove_ready.py` runs the current single-command readiness
  gate and distinguishes release-blocking failures from optional runtime
  warnings such as isolated proof search being disabled.

## Release Blockers

- Fresh Debian VM install, first launch, setup completion, managed search
  setup, external pack usefulness proof, rollback/uninstall, and doctor proof
  have not been completed in a clean machine environment.
- Public network hardening is not claimed. Do not expose the API/web UI or
  SearXNG beyond loopback for this release track.
- Real local-LLM behavior fuzzing is not part of the release gate yet; the new
  second-tier eval is deterministic/mocked by default.

## Non-Blocking Gaps To Track

- Broader managed-action journal rollout is intentionally paused.
- Startup auto-recovery that mutates state is intentionally absent.
- Direct llama.cpp binary/library management is absent.
- MCP/tool runtime execution is absent.
- Semantic memory remains off by default and release-gated.

## Operator Rule

If any chat/status/doc output appears to imply arbitrary shell, pack code
execution, full browser/page fetching, public web exposure, or silent install,
treat it as a security bug and add a regression before fixing it.
