# Web UI dependency maintenance

At the WP6 audit, Vite 5.4.21 was a direct development dependency with a high
advisory and its transitive esbuild carried a moderate advisory. The affected
Vite development server is not shipped in the versioned runtime, which serves
locked static files, but the tools are part of the trusted build path. Node
22 on the release host supports Vite 8, so WP6 upgraded Vite and the React
plugin together. `npm audit` reports zero known vulnerabilities after the
change; locked build and browser/unit regression gates decide compatibility.

For each release candidate, run `npm audit --json` and record every advisory's
package, severity, dependency type, reachable surface, fix, regression risk,
and disposition. Never use `--audit-level`, ignore files, or lockfile deletion
to hide a result. Review direct dependencies monthly and before every release;
prefer focused compatible updates and retain the locked build output as proof.
