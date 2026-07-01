# Fresh Debian VM Proof Plan

Current checkpoint truth lives in `docs/operator/PROJECT_STATE.md`.

This is the clean-host install lifecycle proof. It is separate from
`scripts/first_run_smoke.py`, which proves only an isolated fresh user state
inside the existing checkout.

## VM Assumptions

- OS: Debian 12 stable or Debian 13 testing/stable-equivalent when Debian 13 is
  the current target. Record the exact `/etc/os-release` output in the proof.
- User: non-root desktop or login user with `sudo` privileges for install-time
  system packages only.
- Python: system Python 3.11 or newer available as `python3`.
- Init: systemd with user services available.
- Session: user login session can run `systemctl --user`; if needed, enable
  linger with `loginctl enable-linger "$USER"` and record that action.
- Network: required for initial clone/archive download and Python/package
  dependency installation. Runtime smoke must not require live internet,
  Telegram, SearXNG, Ollama, Podman, or Docker.
- Desktop: browser access to `http://127.0.0.1:8765/` is expected for UI
  inspection, but the scripted API smoke does not require browser automation.

## What To Prove

From a clean VM:

1. Clone the repo or unpack the release archive.
2. Run the supported install path.
3. Confirm the user service starts.
4. Confirm `GET /ready` responds.
5. Confirm `GET /state` responds.
6. Confirm `GET /version` reports the expected commit and runtime instance.
7. Confirm the web UI root responds.
8. Confirm missing Telegram is optional and not fatal.
9. Confirm missing/unconfigured search gives safe setup guidance.
10. Confirm Plan Mode gates mutating actions.
11. Confirm support bundle preview works.
12. Confirm backup remains confirmation-gated.
13. Confirm restore remains preview-only.

Do not configure a real Telegram token for this proof. Do not require live web
search. Do not delete host data. Uninstall and cleanup guidance applies only to
the disposable VM.

## Command Sequence

On the VM:

```bash
cat /etc/os-release
python3 --version
systemctl --user status
```

Acquire the code using one of these paths:

```bash
git clone <repo-url> ~/personal-agent
cd ~/personal-agent
git checkout da6c71e
```

or unpack the release archive and `cd` into it.

Install:

```bash
bash scripts/install_local.sh --desktop-launcher
```

For a release archive, use the archive's bundled `install.sh` instead.

Verify the service:

```bash
systemctl --user status personal-agent-api.service
curl -fsS http://127.0.0.1:8765/version
curl -fsS http://127.0.0.1:8765/ready
curl -fsS http://127.0.0.1:8765/state
curl -fsS http://127.0.0.1:8765/
```

Run the VM proof smoke:

```bash
python scripts/vm_proof_smoke.py --expected-commit da6c71e
```

Then run the local product gates that do not require optional services:

```bash
python scripts/first_run_smoke.py
python scripts/docs_truth_smoke.py
python scripts/release_smoke.py
python scripts/prove_ready.py
```

If managed search is not configured, `prove_ready.py` may report expected
runtime-state warnings. The VM proof should treat missing search as acceptable
only when `/search/status` and chat guidance explain setup safely.

## Manual Checks

Open `http://127.0.0.1:8765/` in the VM browser and verify:

- page loads
- normal chat can send and receive a response
- search setup prompt is understandable when search is missing
- Plan Mode preview is readable for `install htop`
- browser refresh does not show stale endpoint errors

These browser checks are manual until a VM browser automation lane is added.

## Disposable VM Cleanup

Only on the disposable VM, after proof:

```bash
systemctl --user stop personal-agent-api.service
systemctl --user disable personal-agent-api.service
```

If the goal is to test uninstall, use only the documented uninstall path from
the installed release bundle. Do not run cleanup or uninstall commands on the
developer host as part of this VM proof.

## Pass Criteria

- `scripts/vm_proof_smoke.py` prints `VM_PROOF_SMOKE: pass`.
- `/version` commit matches the intended commit or release archive.
- Telegram is missing/unconfigured and optional.
- Search is missing/unconfigured and gives safe setup guidance.
- Mutating prompts produce Plan Mode previews and do not execute.
- No real Telegram token, live internet search, or destructive cleanup is
  required.

## Remaining Gaps After This Plan

Until the VM proof is actually run, Personal Agent remains “ready for VM proof,”
not release-candidate complete. The plan also leaves browser automation and full
installer failure-recovery testing as future hardening lanes.
