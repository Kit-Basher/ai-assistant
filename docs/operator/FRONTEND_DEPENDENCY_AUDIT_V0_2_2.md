# Frontend Dependency Audit v0.2.2

Audit date: 2026-07-13

## Commands

```bash
cd desktop
npm ci
npm audit
npm audit --omit=dev
```

## Production Audit

```text
npm audit --omit=dev
found 0 vulnerabilities
```

Release classification: pass.

The shipped Web UI bundle is static output from Vite. Runtime production
dependencies in the frontend dependency graph have no reported npm advisories.

## Full Audit

After `npm audit fix` and a safe Vite patch update to `5.4.21`, full audit
still reports the Vite/esbuild dev-server advisory:

```text
esbuild <=0.24.2
vite <=6.4.2 depends on vulnerable versions of esbuild
fix available via npm audit fix --force
```

Npm reports that the remaining automatic fix requires a breaking change to
Vite `8.x`.

## Release Decision

Accepted for `v0.2.2` as a non-production build-tool warning:

- affected path: local development server / build toolchain;
- production audit: clean;
- shipped artifact: static built assets, not the Vite dev server;
- mitigation: do not expose the Vite dev server as production;
- revisit trigger: Vite major-upgrade planning or any production audit finding.

No production-critical or externally exploitable frontend dependency issue is
accepted for this release.
