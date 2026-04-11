# Release

This is the canonical release and operational handoff for Personal Agent.
If another note disagrees, trust the code plus this document.

## Release Gate

Canonical ship gate:

- `python scripts/release_gate.py`

Fast pre-check:

- `python scripts/release_smoke.py`

Heavier follow-up validation:

- `python scripts/release_validation_extended.py`

The release gate intentionally combines:

- syntax validation with `py_compile`
- core release-stability pytest coverage
- diff sanity checks
- the shipped-path web UI/API conversation proof at
  `tests/test_webui_conversation_smoke.py`
- the explicit assistant-behavior gate at `tests/test_assistant_behavior_release_gate.py`

The live web UI smoke and live pack workflow smoke remain available as separate
operator smokes in the README/runbook, but they are not required in the
canonical ship gate because they depend on live service state that is not
stable enough for every release environment.

The assistant-behavior gate is the narrow release truth for chat quality:

- greeting
- vague request
- nonsense input
- mixed/confusing request
- no-LLM setup/help reply

Passing it means the public assistant boundary is still intact and the shipped
chat surface is not leaking routing, tool, pack, or internal setup details in
ordinary replies.

For clean-context validation, the extended release suite now also includes
`tests/test_clean_context_validation.py`, which installs the release bundle
into a temporary install root, launches the installed web UI through the
shipped launcher, exercises a no-LLM two-turn chat, relaunches, and verifies
the uninstall/remove-state policy.

For day-to-day operations, incidents, support bundles, and release-note
discipline, use `docs/operator/OPERATIONS.md`.

Manual/operator checks before calling a build releasable:

- verify the release bundle builds:
  - `bash scripts/build_release_bundle.sh --clean`
- verify the Debian package builds:
  - `bash scripts/build_deb.sh --clean`
- install the release bundle on a clean user-local root:
  - `bash install.sh`
- install the Debian package on a clean machine or clean local state:
  - `sudo apt install ./dist/personal-agent_<version>_amd64.deb`
- reinstall the same bundle once to confirm idempotence
- reinstall the same Debian package once to confirm idempotence
- uninstall once with state preserved and once with `--remove-state`
- remove the Debian package and, if desired, run `personal-agent-uninstall --remove-state`
- install on a clean machine or clean local state with:
  - `bash scripts/install_local.sh --desktop-launcher`
- fresh startup on a clean runtime state
- first web-ui launch shows onboarding only on a fresh state, and second launch
  stays quiet after onboarding is completed or skipped
- desktop launcher appears and opens the local UI
- `python -m agent version`
- `curl -sS http://127.0.0.1:8765/version`
- `python -m agent doctor`
- `python -m agent status`
- `curl -sS http://127.0.0.1:8765/ready`
- `curl -sS http://127.0.0.1:8765/state`
- `curl -sS http://127.0.0.1:8765/packs/state`
- `dpkg-deb -I dist/personal-agent_<version>_amd64.deb`
- `dpkg-deb -c dist/personal-agent_<version>_amd64.deb`
- end-to-end pack workflow still works
- confirm/approval flow still works if the build exposes it
- if the installer or launcher is part of the change, reinstall it once and
  confirm the idempotent path still works

Release blockers:

- contradictory `/ready`, `/state`, or `/packs/state` truth
- failed smoke path
- release gate failure
- stale or missing backup/restore guidance
- docs that describe behavior the code no longer has
- operator-facing errors with no blocker or next step
- unresolved blocking incident that is not represented cleanly in the state
  surfaces

## Install, Update, Roll Back

Canonical install/update path:

- repo checkout at `~/personal-agent`
- virtualenv under the checkout
- `pip install -e .`
- user service `personal-agent-api.service`

Update:

- pull or otherwise update the checkout
- run `pip install -e .`
- restart the user service
- re-run the release gate before shipping

Rollback:

- restore the previous checkout or wheel
- reinstall it in the same environment
- restart the user service
- re-run the release gate

There is no formal database migration story in this repo today.
If the persisted state looks incompatible, back it up first, then restore the
previous version or repair the state with `python -m agent doctor --fix`.

## Defaults

Important shipped defaults are intentionally conservative:

| Default | Why | Change when |
| --- | --- | --- |
| `PREFER_LOCAL=true` | Keep ordinary chat local-first. | You want a different model preference policy. |
| `ALLOW_CLOUD=true` | Allow cloud fallback when policy permits it. | You need a strict local-only machine. |
| `LLM_PROVIDER=none` | Do not assume a provider before setup. | You have explicitly configured a provider. |
| `TELEGRAM_ENABLED=false` | Keep optional transport off by default. | You want the Telegram adapter active. |
| `LLM_AUTOMATION_ENABLED=true` | Keep background maintenance available. | You want fully manual operation or a safe-mode test. |
| `AGENT_SAFE_MODE=false` | Ship the normal runtime, not read-only mode. | You need a conservative read-only session. |
| `MODEL_SCOUT_ENABLED=true` | Keep model advisory available. | You need a quieter or more manual setup. |
| `AGENT_MODEL_WATCH_HF_ENABLED=false` | Keep external model discovery opt-in. | You want Hugging Face discovery enabled. |
| `AGENT_MODEL_WATCH_BUZZ_ENABLED=false` | Keep buzz-based discovery opt-in. | You want trend-based discovery enabled. |

## Backup and Restore

See `docs/operator/BACKUP_RESTORE.md` for the canonical backup set and restore
steps.

## Known Limits

See `docs/operator/KNOWN_LIMITS.md` for the honest list of unsupported or
still-limited behavior.

## Versioning and Release Process

- canonical version source: `VERSION`
- build metadata source: `python -m agent version`
- release artifacts: `python scripts/build_dist.py --outdir dist --clean`
- ship notes should record:
  - release version
  - git commit
  - release gate output
  - smoke outputs
  - any intentional follow-up

Patch releases should be compatibility, stability, and correctness fixes.
Minor releases should add supported user-facing behavior or new canonical
surfaces.
Anything larger should be called out explicitly in release notes.

Release notes should follow `docs/operator/RELEASE_NOTES_TEMPLATE.md`.

## Support Boundaries

Normal diagnosis should come from:

- `/ready`
- `/state`
- `/packs/state`
- `python -m agent doctor`
- `python -m agent version`

If normal diagnosis requires logs, that is a support gap that should be fixed
or called out as an exceptional case.
