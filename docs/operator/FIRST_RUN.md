# First-Run / Fresh-State Proof

Current checkpoint truth lives in `docs/operator/PROJECT_STATE.md`.

`scripts/first_run_smoke.py` proves that Personal Agent can start from an
isolated empty user state and explain itself honestly. It is not the fresh
Debian VM install proof.

## What It Does

The smoke starts a temporary API process on `127.0.0.1` with an isolated:

- `HOME`
- `XDG_DATA_HOME`
- `XDG_CONFIG_HOME`
- `XDG_CACHE_HOME`
- `XDG_RUNTIME_DIR`
- temp directory
- agent database, logs, audit log, permissions, secret store, and registry
  paths

It does not start or overwrite the promoted stable runtime. It does not use the
real user state under `~/.local/share/personal-agent`.

## Proven Behavior

The first-run smoke verifies:

- required state/config directories are created under the isolated home
- `/ready` and `/state` return coherent JSON
- `/version` reports `runtime_instance` and `git_commit`
- the web UI root responds
- missing Telegram configuration is optional and not fatal
- missing search configuration is reported honestly with safe setup guidance
- memory starts empty or says there is no useful saved memory yet
- package install remains Plan Mode gated
- support bundle, backup, restore, and cleanup prompts stay safe in fresh state
- no root-level smoke artifacts are left behind
- git status is unchanged by the smoke

The fresh state may be degraded or bootstrap-required. That is acceptable when
the response says what is missing and keeps chat/mutation behavior safe.

## What It Does Not Prove

- It does not install onto a clean Debian VM.
- It does not exercise a full packaged installer from an empty OS account.
- It does not require internet, Telegram, SearXNG, Ollama, Podman, or Docker.
- It does not prove real Telegram delivery or live web search.
- It does not run destructive restore, cleanup, memory delete, update, or
  uninstall actions.

## Command

```bash
python scripts/first_run_smoke.py
```

Expected result:

```text
FIRST_RUN_SMOKE: pass
```

If it fails, inspect the named command/API path in the report and the isolated
API log tail printed by the smoke.
