# Release Gate Matrix

Current checkpoint truth lives in `docs/operator/PROJECT_STATE.md`.

This file separates deterministic CI-safe checks from local-runtime and optional
integration checks. The split prevents GitHub Actions from requiring a personal
machine's services while keeping the local release proof strict.

## CI-Safe Gates

These can run from a clean checkout without Personal Agent already installed as a
local service:

- `python scripts/release_gate.py --py-compile-only`
- `python scripts/release_smoke.py`
- `python scripts/chat_eval.py`
- `python scripts/llm_behavior_eval.py`
- `python scripts/docs_truth_smoke.py`
- `python scripts/external_pack_safety_smoke.py`
- `python scripts/backup_restore_proof.py`
- `python scripts/release_gate_matrix_smoke.py`
- `python -m pytest -q tests/test_release_gate.py tests/test_release_smoke.py`
- `python -m pytest -q tests/test_backup_restore_proof.py tests/test_pre_vm_complete_gate.py`

The current GitHub Actions workflow intentionally stays small and CI-safe. It
does not require SearXNG, Telegram, Ollama, a desktop session, systemd user
services, Podman, or Docker.

## Live-Runtime Gates

These are the live-runtime gates. They require the local API/runtime or
machine-specific state and should run on the release machine before the fresh VM
proof.

Commands:

- `python scripts/prove_ready.py`
- `python scripts/prove_pre_vm_complete.py`
- `python scripts/prove_daily_driver_product.py`
- `python scripts/installed_product_abuse.py`
- `python scripts/perf_smoke.py`
- `python scripts/daily_driver_smoke.py --timeout 90`
- `python scripts/prove_core_workflows.py`
- `python scripts/webui_robustness_smoke.py`
- `python -m agent doctor`

Runtime-state warnings are acceptable only when the command clearly labels them,
for example search disabled because no trusted SearXNG backend is configured.
`installed_product_abuse.py` is stricter than the ordinary smoke: it talks only
to the installed API surface, verifies promoted runtime freshness, checks
documented endpoint availability, and abuses search/Telegram/memory/Plan Mode
flows like a confused web UI user.

At `v0.2.1-search-recovery-product-pass`, the installed-product gate is the
stronger daily-driver truth gate because an earlier real UI search request found
failures that internal/mock gates had missed.

## Optional Integration Gates

These require optional services or hardware and are not CI requirements:

- managed SearXNG live setup/query proof
- Telegram bridge smoke
- local model/Ollama live model smoke
- local model/provider matrix on the target hardware
- future browser/GUI automation proof for the desktop web UI

Optional gates should report `BLOCKED` when the required service is not
configured, not `PASS`.

The fresh Debian VM proof is intentionally not listed as completed here. It is
the next expensive confirmation after local product-facing gates pass.

## GitHub Actions Direction

Broader GitHub Actions coverage should be added later, after the local
PRE_VM_COMPLETE gate and the fresh Debian VM proof are stable. The next likely
CI expansion is to add `chat_eval.py`, `llm_behavior_eval.py`,
`backup_restore_proof.py`, and `external_pack_safety_smoke.py` to the existing
workflow. Do not add local-runtime gates to CI unless they are explicitly
mocked or converted to CI-safe mode.
