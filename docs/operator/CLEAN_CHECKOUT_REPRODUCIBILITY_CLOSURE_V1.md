# Clean Checkout Reproducibility Closure v1

## Failed Final Verification

Final release verification at checkpoint
`9f74f7af3ee0b823de520da71e644bbab93a34ec` was stopped because a detached
clean worktree did not match the primary checkout result.

Observed clean-worktree result before fixes:

```text
10 failed, 2438 passed, 22 skipped
```

After building desktop assets from `desktop/`, the remaining affected cluster
was:

```text
4 failed, 8 passed
```

## Root Causes

- `tests/test_config.py` hard-coded `/home/c/personal-agent/control` instead of
  deriving the current repository root.
- `scripts/build_deb.sh` required and packaged ignored local
  `llm_registry.json` state.
- `pyproject.toml` did not declare `test` or `release` extras, so clean
  verification depended on manually installed tools.
- Web UI assets are generated from tracked `desktop/` sources and must be built
  in clean verification before package/artifact tests.

## LLM Registry Decision

`llm_registry.json` is classified as mutable local runtime state.

It is not a release input and must not be copied from the operator's checkout
into Debian packages. Runtime code already has deterministic built-in registry
defaults and installed services point `LLM_REGISTRY_PATH` at user state. First
run or runtime configuration may create local registry state, but package build
must not require the ignored file.

## Package Reproducibility

Debian package build now:

- succeeds from a clean checkout after Web UI assets are built;
- omits `llm_registry.json`;
- includes tracked runtime source, Web UI dist assets, release notes, launcher,
  icon, and systemd user-service template;
- fails only for tracked or generated release inputs that are actually required.

Regression proof:

```bash
python scripts/clean_checkout_debian_package_smoke.py
```

## Dependency Extras

Clean verification now uses declared extras:

```bash
python -m pip install -e '.[test,release]'
```

The `test` extra provides pytest. The `release` extra provides the Python build
frontend used by release artifact checks.

## Desktop Asset Policy

Web UI assets are built during release verification from tracked frontend
sources:

```bash
cd desktop
npm ci
npm run build
cd ..
```

Generated `agent/webui/dist` output is not copied from the primary checkout.

## Frontend Audit

Production dependency audit is clean:

```text
npm audit --omit=dev
found 0 vulnerabilities
```

Full audit still reports Vite/esbuild dev-server advisories. These affect the
development/build toolchain, not shipped production dependencies. They are
recorded in `docs/operator/FRONTEND_DEPENDENCY_AUDIT_V0_2_2.md`.

## Clean Verification Command

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,release]'
cd desktop
npm ci
npm run build
cd ..
python -m compileall agent tests scripts
python -m pytest -q
python -m pytest -q
python scripts/clean_checkout_reproducibility_smoke.py
python scripts/clean_checkout_debian_package_smoke.py
python -m build
python scripts/final_release_audit.py
python scripts/prove_ready.py
```

## Release Impact

Final `v0.2.2` tagging remains blocked until a fresh detached worktree can run
the clean verification sequence with zero failures, zero unresolved warnings,
and `READY_TO_TAG=true` / `READY_TO_RELEASE=true`.
