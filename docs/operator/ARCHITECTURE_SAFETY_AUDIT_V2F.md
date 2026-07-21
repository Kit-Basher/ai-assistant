# Architecture and Safety Audit v2F

Status: working-tree audit, unreleased. Starting checkpoint:
`724de3cbbbd25b2396d0f660fb0062b84d339944`.

## Disposition

The deterministic v2F inventory records thirteen centrally authorized public
operations, one bounded scheduled-notification writer, and two explicitly
unimplemented/denied operations. The global mutation inventory has no
`legacy_unmigrated`, `plan_confirm_gated`, `plan_gated_legacy`, or unclassified
mutation-bearing file.

Pack-source catalog/policy, permission policy, local text-pack lifecycle, and
managed-search setup now use one path: Universal Mutation Plan, durable scoped
confirmation, Capability Policy, Executor Registry, bounded executor, durable
transaction result, and redacted receipt. Compatibility plan/apply routes are
serializers/adapters over that path; old tokens and boolean approvals fail.

## Safety boundaries

- Source allowlisting makes metadata queryable; it grants neither trust nor
  permissions.
- Approval, enablement, and permission grant are independent operations.
- External packs start with zero grants. Foreign code and dependency execution
  remain denied.
- Combined remote fetch/install is unavailable. A future implementation must
  authorize fetch-to-quarantine separately, validate the connection target,
  and bind approval to the resulting digest. This audit does not claim DNS
  rebinding protection for an unimplemented arbitrary-host fetcher. Final
  reachability review also removed the stale assistant approval/fetch
  continuation: the controller and runtime install helper now deny before any
  URL is opened.
- Search setup accepts only the bounded SearXNG service definition and loopback
  exposure. SAFE MODE blocks setup/prerequisite mutation. The fixed image
  allowlist remains tag-based; registry digest resolution is a release warning,
  not a claim of immutable image binding.
- Notification delivery intent is durable before I/O. An interrupted executing
  delivery becomes `indeterminate` and is never resent automatically. External
  transports do not provide a general exactly-once guarantee. Backup v1 takes
  an online SQLite snapshot of the delivery ledger; restore preserves terminal
  state and converts an interrupted executing delivery to `indeterminate`.

## Limitations

Universal authorization can be claimed for the inventoried public interfaces
and registered internal writers. It is not process/OS isolation: malicious
Python already executing inside the trusted process can invoke implementation
primitives. Remote pack acquisition, foreign plugin execution, automatic
notification-indeterminate resolution, and immutable search-image resolution
remain unavailable or incomplete.

Machine evidence:
`PACK_SEARCH_AUTHORIZATION_INVENTORY_V2F.json`,
`MUTATION_SURFACE_INVENTORY_V2.json`, and
`MUTATION_FILE_CLASSIFICATIONS_V2B.json`.
