# WP6 independent architecture and security review

The review was performed against the assembled WP1-WP5 production boundaries,
not against feature descriptions. Findings are release-blocking until their
listed regression proof passes.

| ID | Severity | Boundary | Evidence / failure scenario | Disposition and regression proof |
|---|---|---|---|---|
| WP6-01 | High | backup/restore | The legacy standalone archive recursively included runtime releases, unrelated user units, and machine-bound encrypted secrets. | Fixed by portable backup v2 allowlists, link/size/hash validation, staged conflict-safe restore, explicit secret re-entry, corruption/version/idempotency tests. |
| WP6-02 | Medium | onboarding/recovery | Public onboarding required Python and systemd commands for ordinary recovery. | Fixed with UI/chat next actions; every non-ready state is tested against shell/JSON/Git language. |
| WP6-03 | Medium | diagnostics/secrets | The UI export described itself as general support but represented mainly model state. | Fixed with one versioned runtime/capability/model/pack/policy diagnostic document and adversarial recursive redaction test. |
| WP6-04 | Medium | release evidence | Canonical proof enumerated large static test tuples but had no WP6 layer/journey/fingerprint contract. | Fixed with commit/diff-bound WP6 proof requirements and canonical gate integration; 16 actual gate sensitivity cases. |
| WP6-05 | Medium | Web build tooling | Vite 5/esbuild advisories affected the development server, not shipped static files, but remained in the release toolchain. | Fixed with compatible Vite/plugin upgrade; audit, locked build, JS and browser checks are release gates. |
| WP6-06 | Low | documentation truth | Native manifest product version and pack/runtime claims lagged released behavior. | Reconciled current docs and version metadata; documentation/manifest checks remain gated. |
| WP6-07 | Info | physical recovery | Current-host/container proof cannot prove firmware, storage, OS installer, user-session systemd, GPU driver, or desktop integration on a physically fresh Ubuntu host. | Accepted only as an explicit unfinished WP6 boundary. The human runbook and post-install verifier must be completed before WP6 closes. |

No unresolved release-blocking finding is knowingly accepted by the
pre-reinstall gate. The physical Ubuntu journey remains unproved rather than
being downgraded.
