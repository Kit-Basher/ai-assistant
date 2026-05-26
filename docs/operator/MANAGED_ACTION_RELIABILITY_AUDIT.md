# Managed Action Reliability Audit

This audit is a checkpoint, not a claim that every flow already meets the full standard. A flow is marked safe only where code and tests prove the relevant property. Unknown or partially tested areas are marked as needs follow-up.

## Summary

Highest-confidence flow today: managed SearXNG setup/cleanup. It has preflight, preview, confirmation, journal, verification, owned rollback, and tests.

Main gaps: provider/API key writes, registry/autoconfig/self-heal, and several pack lifecycle mutations have confirmation and validation but do not yet share the managed-action journal and rollback contract. Model downloads/imports now have in-memory managed-action journals, post-action verification, and owned temp/Modelfile cleanup where ownership is proven, but still do not delete unproven Ollama cache/model data.

## Audit Table

| Flow | Mutates what? | Current confirmation gate? | Preflight? | Journal? | Verification? | Rollback? | Rollback scope | Failure quality | Risk | Required next fix |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| managed local services / SearXNG setup | Docker/Podman image pull, `personal-agent-searxng` container, managed volume | yes | yes | yes | health check | yes | owned setup container only; pre-existing container untouched | good; reports cleanup and remaining issue | low | keep smoke coverage; add persistent journal storage if needed |
| managed service stop/remove | `personal-agent-searxng` container only | yes | yes | yes | command result | partial | approved container name only | good; reports scoped target | low | add richer container ownership inspection before future restart/reuse |
| model downloads / Ollama pulls | Ollama model inventory, model manager state | yes through model acquisition preview | partial; provider/backend delegated to existing guard, disk space recorded as not checked when no helper exists | yes, in-memory result journal | yes; inventory refresh and verification row | partial | owned temp files only; Ollama cache/model data is not deleted unless ownership is proven | improved; names failed step, cleanup, possible remaining cache/model data, and one next step | medium | add persistent journal storage, disk-space preflight, and explicit ownership markers before any safe `ollama rm` cleanup |
| GGUF import / local model import | downloaded files, generated Modelfile, Ollama model create, model manager state | yes for acquisition/import request | partial; backend delegated to existing guard, disk space recorded as not checked when no helper exists | yes, in-memory result journal | yes; Ollama inventory refresh and verification row | partial | generated Personal-Agent Modelfile/temp files only; user-provided Modelfile and Ollama cache/model data are not deleted | improved; verification/import failures report cleanup and what may remain | medium | add persistent journal storage, disk-space preflight, and model ownership markers for safe future cleanup of failed Ollama creates |
| default model changes | routing/default model config | yes for chat default changes | yes, target resolution | no shared journal | partial post-switch response/status | trial switch has remembered previous target; default rollback exists separately | limited previous-target/default state | medium | medium | journal config write, verify active target, and expose explicit rollback for failed default changes |
| provider/API key config | secret store and provider config/status | setup flow confirmation/secret entry varies by path | partial | no shared journal | provider test/status exists | no transactional rollback proven | none proven | medium | high | add preview/confirm for writes, journal previous secret/config metadata, verify provider, rollback failed writes |
| Telegram token/service setup | secret store, enablement drop-in/env, service state | mostly operator/setup guided; confirmation varies | partial | no shared journal | doctor/startup checks | no transactional rollback proven | none proven | medium | medium | add managed setup flow with journal, token write verification, service restart verification, and rollback of owned enablement writes |
| pack source approval | catalog source/policy JSON | yes: source approval preview then confirmation | yes | no shared journal | policy/source record returned | no rollback; write is atomic-ish | source/policy record only | good; says no content fetched/imported | medium | add journal and rollback/delete of source policy if paired write partially fails |
| pack quarantine fetch/import | quarantine files, normalized pack files, pack store record | yes: fetch preview then confirmation | yes | no shared journal | ingestion/normalization status and lifecycle | quarantine hardening blocks unsafe content; full rollback of partial record/files not proven | none proven beyond ingestion safeguards | medium-good; reports review-only or blocked | medium | add transaction journal across fetch, quarantine, normalization, and store record; cleanup owned partial artifacts |
| pack review approval | pack approval metadata | yes: review state, approval preview, confirmation | yes | no shared journal | lifecycle result/status | no rollback needed beyond metadata update, but not journaled | none | good; one gate only | low-medium | add journal for approval metadata writes and verify lifecycle state |
| pack enablement | pack enabled metadata | yes: enablement preview then confirmation | yes | no shared journal | lifecycle result/status | no rollback/journal | none | good; one gate only | low-medium | add journal and verification for enabled flag writes |
| pack permission/config grant | adapter grant metadata/config | yes: permission/config preview, scoped grant preview, confirmation | yes | no shared journal | grant/lifecycle status | no rollback/journal | none | good; says grant is metadata only | low-medium | add journal and delete/undo path for failed or mistaken grant metadata |
| managed adapter invocation | core-owned adapter operation result | yes: invocation preview then confirmation | yes | operation result only, no shared journal | yes for validate/dry-run | not needed for no-content dry-run; future read/write ops need rollback | no mutation in current local file dry-run | good | low now; higher for future ops | require journal before adding content read/write/index operations |
| file operations | native filesystem skill is read-only today; future writes unknown | destructive skill calls require policy confirmation | read-only path preflight exists | no shared journal | read-only result only | no write rollback implemented | not applicable for read-only | good for read-only | low now; high for future writes | before write/move/delete features, add preview, journal, backup/rollback, and path ownership rules |
| notification/autopilot mutating actions | scheduler/autopilot may change registry/provider defaults depending policy | policies exist; automatic apply is gated by config | partial | no shared journal | partial status/audit | no full rollback contract | none proven | medium | high | keep auto-apply off unless explicitly enabled; add journal, verification, and rollback for every automatic mutation |
| registry prune/rollback/hygiene/autoconfig/self-heal | provider/model registry documents and defaults | loopback/operator and policy gates | partial | audit records/snapshots exist, not managed-action journal | partial | rollback/defaults exist in some paths, but not uniform | snapshot/default restore where present | medium | high | unify under managed-action journal; require preview/confirm for manual apply and policy proof for scheduled apply |

## Findings

- SearXNG is the reference implementation for the standard.
- External pack lifecycle gates are strong on safety and one-step confirmation, but most pack metadata writes are not yet journaled as managed actions.
- Model acquisition now records managed-action journals and verifies inventory after pulls/imports. It still needs persistent journals, disk-space preflight, and provable model/cache ownership before any automated cache/model deletion.
- Provider and Telegram setup affect secrets and services. They need transactional metadata and rollback/repair messaging.
- Registry/autoconfig/self-heal are operator-grade today. They should not be treated as normal-user managed actions until journal and rollback coverage exists.

## Required Follow-Up Order

1. Add provider/API key config transaction records and rollback/verify behavior.
2. Add Telegram setup transaction records and service verification/rollback.
3. Add persistent managed-action journals and disk-space preflight for model downloads/imports.
4. Add pack source/fetch/import journal coverage and partial artifact cleanup.
5. Move registry/autoconfig/self-heal mutations under the same standard before expanding automatic apply behavior.
