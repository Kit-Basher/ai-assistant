# Internal Writer Authority v1

Status: Audit v2C implementation checkpoint, 2026-07-20. This contract is not
public mutation authorization and does not make universal authorization complete.

## Boundary

`INTERNAL_WRITER_REGISTRY_V1.json` is the machine-readable allowlist.
`InternalWriterFactory` is the only constructor of live authority. Authority is
nonce-backed, single-use, and bound to writer, capability, trigger, mode, and
durable operation ID. It cannot be reconstructed from serialized data. API,
control-plane, and skill-pack boundaries reject reserved internal identity keys.
Chat, Telegram, files, memory, web results, and pack text remain untrusted
content and cannot create the in-process nonce.

Every invocation checks the registered operation, symbolic resource type,
symbolic target scope, trigger, runtime mode, argument limits, and callback
module. A SQLite journal reserves a unique operation identity before the
callback and records a redacted receipt. No contract accepts caller-selected
executors, commands, URLs, host paths, capabilities, or trusted context.

## Reviewed set of 24

| Writer | Disposition | v2C enforcement |
|---|---|---|
| `audit_log` | trusted bookkeeping | enforced internal boundary |
| `control_plane` | operator legacy | public; not internal-authorized |
| `llm_action_ledger` | trusted bookkeeping | enforced internal boundary |
| `llm_autopilot_safety` | scheduled maintenance | enforced internal boundary |
| `llm_catalog` | trusted cache | enforced internal boundary |
| `llm_health` | trusted health bookkeeping | enforced internal boundary |
| `llm_model_discovery_policy` | mixed | authority denied pending internal/public separation |
| `llm_notifications` | mixed | authority denied pending internal/public separation |
| `llm_registry` | operator legacy | public; not internal-authorized |
| `llm_registry_txn` | operator legacy | public; not internal-authorized |
| `llm_usage_stats` | trusted telemetry | enforced internal boundary |
| `logging_utils` | trusted redacted logging | enforced internal boundary |
| `model_scout` | mixed | scheduled entry authorized; user status mutation pending |
| `model_watch` | mixed | authority denied pending internal/public separation |
| `model_watch_catalog` | trusted cache | enforced internal boundary |
| `model_watch_hf` | mixed | authority denied pending internal/public separation |
| `modelops_seen_state` | trusted cache | enforced internal boundary |
| `packs_managed_adapters` | operator legacy | public; not internal-authorized |
| `permissions` | operator legacy | public; not internal-authorized |
| `primary_uninstall_policy` | operator central | existing central authorization only |
| `scheduled_daily_brief` | scheduled maintenance | enforced, durable daily identity |
| `scheduled_model_scout` | scheduled maintenance | enforced, durable interval identity |
| `skill_governance` | read-only | no writer authority issued |
| `telegram_runtime_state` | operator legacy | public; not internal-authorized |

Only eleven files are wholly trusted bookkeeping/scheduled writers. Five are
mixed, six are operator-triggered legacy mutation, one is already central, and
one is read-only. Treating all 24 as “internal” would create a public bypass.

## Retry and failure semantics

The journal begins in `executing` before the callback. Success and clean
failure receive terminal redacted receipts. An existing operation identity is
never automatically run again. A crash with an `executing` row is an uncertain
outcome requiring reconciliation, not a retry invitation. Scheduled jobs use
stable operation identities: daily brief uses the date and model scout its
interval slot.

## Storage and backup

All stable, developer, release-bundle, and Debian API configurations resolve
`AGENT_DB_PATH` to `~/.local/share/personal-agent/agent.db`. Runtime processes
therefore share `~/.local/share/personal-agent/confirmation_transactions.sqlite3`.
The former journal-derived sidecar is imported idempotently at startup and is
never treated as a second authority.

Transaction and internal-writer SQLite files, including WAL/SHM side files,
are mode `0600`. Backup v1 uses SQLite's online backup API, so committed state
present in WAL is copied into standalone snapshot databases; correctness does
not depend on copying a live WAL/SHM pair. Restore validates hashes, integrity,
tables, and bounded target paths, then merges rows without replacing the live
confirmation database. Restored reserved rows become failed and restored
executing rows become indeterminate.

## Threat-model limit

This prevents public serialized inputs and ordinary extension content from
claiming authority. It does not sandbox malicious Python already executing in
the trusted process. Foreign executable/plugin packs remain denied. Protection
from arbitrary in-process code requires a process/OS isolation boundary.

## Audit v2D mixed-writer split

Manual watch, HF scan, proposal, policy, autopilot, and maintenance entry points
are public mutations and use central authorization. Read-only discovery remains
immediate. Scheduled catalog/health/scout/watch bookkeeping retains only bounded
internal identities and cannot invoke the new public mutation executors.
