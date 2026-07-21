# End-User Behavior and UX Audit 3

Status: verification-complete checkpoint; not released.

Baseline: `a324ff65d7614bcd7955306f35252302b4c6ece3`.

## Scope and method

This audit evaluates the public assistant, Web UI, CLI wording, mocked Telegram
transport, and user-facing operator material. It does not redesign mutation
authorization, contact Telegram, alter a service, or run fresh-VM acceptance.
Runtime claims were checked against `RuntimeTruthService`, the v2F surface
inventory, canonical mutation Plans, transaction states, and pack/search/
notification executors rather than accepted from documentation alone.

The deterministic evidence file is
`docs/operator/END_USER_UX_INVENTORY_AUDIT_3.json`; regenerate or verify it with
`python scripts/end_user_ux_audit.py [--check]`.

## Confirmed contradictions and corrections

1. README and assistant discovery implied that an approved remote archive could
   be fetched into quarantine. The product actually denies arbitrary remote
   acquisition before network access. README, assistant capability summaries,
   Web rescue copy, source-approval copy, registry previews, API recovery text,
   and install handoffs now say metadata-only discovery. Registry preview no
   longer emits a remote install handoff.
2. README classified all remove flows as unsupported. It now distinguishes
   denied arbitrary filesystem deletion from centrally authorized removal of an
   exact registered pack/version.
3. Capability-policy answers still described a representative migration and
   exposed internal capability identifiers. They now explain immediate reads,
   previewed changes, local activation, SAFE MODE blocks, and Controlled Mode in
   ordinary language consistent with completed v2F inventory truth.
4. Generic confirmation cards and Plan text emphasized machinery and internal
   resource names. They now lead with what changes, creates, deletes, external
   effect, risk, rollback limits, expiry, approval, and cancellation. Diagnostic
   identifiers remain in structured payloads/support detail, not the primary
   explanation.
5. Failure messages called approval artifacts tokens and advised retrying a
   downstream failure. They now use user language and require status checking;
   indeterminate outcomes explicitly prohibit blind or automatic retry.

## Assistant-first behavior

Capability discovery leads with useful outcomes and a small set of examples.
It is derived from runtime/search/mode status and does not advertise remote pack
download. Commands remain optional operator shortcuts. The same public
serializer and orchestration response feed Web and mocked Telegram formatting;
no parallel router or inference path was introduced.

Natural follow-ups retain the existing durable, scoped pending-Plan semantics:
one clearly active proposal may accept `yes`; cancellation aborts it; a new plan
replaces the old one; unrelated conversation does not execute it; actor,
thread, session, target, expiry, and fingerprint mismatches fail closed. Audit 3
adds semantic assertions around these distinctions rather than sentence-wide
snapshots.

## Confirmation and modes

The canonical Plan remains the source of confirmation truth. Web cards present
plain language and keep approval/cancellation commands identical to natural
chat. Destructive deletion receives a stronger title and action label.

SAFE MODE is described as the normal default. It allows reads and policy-allowed
local behavior, while blocking named higher-risk operations. Controlled Mode
only changes which proposals are eligible; it is neither unrestricted nor
automatic, and explicit confirmation remains mandatory.

## Organization, memory, and continuity

The journey contracts cover one-target `/done`, ambiguity clarification, stale
task/reminder rejection, retry idempotency, restart continuity, saved-memory
versus conversation-context wording, correction/forget behavior, and destructive
memory reset. Private memory is not copied into Plans, receipts, ledgers, logs,
or support output. Unsupported archive/merge/restore variants remain unavailable
rather than being invented for a smoother demo.

## Pack, permissions, search, and notifications

Pack discovery, local ingestion, review approval, enablement, grant/revoke, and
registered removal remain separate. Installed external packs start with zero
permissions; source allowlisting is query access, not trust; text remains
untrusted; foreign executable/plugin packs remain denied.

Managed search status is immediate. SAFE MODE blocks setup/repair; Controlled
Mode still requires preview and approval. SearXNG uses a configured image tag
bound with local runtime state, not an immutable digest. Upstream tag movement
remains release hardening.

Notification intent is durable before transport I/O. An uncertain restored send
is indeterminate and never automatically resent. Public resolve/resend/abandon
is unavailable, and exactly-once delivery is not claimed. Telegram remains
optional and disabled; tests use mocks only.

## Web and accessibility

Chat remains primary and Advanced remains secondary. The transcript is one
polite ARIA live log; progress and new messages are announced through that
single owner rather than nested live regions. Approval is a labelled group,
approval/cancel buttons have explicit accessible names, and the disabled
attachment control explains its state. Busy state continues to block duplicate
submission. Responsive CSS retains single-column behavior at narrow widths.

The UI now labels assistant message operation state as waiting for confirmation,
complete, or failed. Indeterminate truth remains driven by the backend receipt/
status response and is never optimistically converted to success.

## Remaining UX limitations

- Several advanced panels remain operator-dense and warrant observation on a
  real clean install; this audit deliberately avoided a visual rebrand.
- Public notification reconciliation actions are unavailable.
- Remote pack fetch-to-quarantine is unavailable; only metadata discovery and
  reviewed local text-pack ingestion are supported.
- SearXNG immutable digest pinning is unresolved.
- Some organization CRUD variants remain explicitly unsupported.
- Accessibility checks here are deterministic markup/interaction tests, not a
  substitute for assistive-technology testing on the final packaged build.
- Authorization does not provide OS/process isolation from malicious Python
  already executing in the trusted runtime.

## Verification evidence

- Focused assistant, runtime-truth, confirmation, recovery, transport, and UX
  contracts: 480 passed, 3 skipped.
- Realistic journey and cross-transport contracts: 253 passed, 1 skipped.
- Full Python suite: 2,575 passed, 22 skipped.
- Canonical release smoke: 74 passed; release-artifact smoke: 14 passed,
  0 warnings, 0 failures.
- Web interaction/accessibility helper suites: 4 passed; the canonical Web
  build completed and wrote the staged build manifest.
- Capability Policy: 94 passed; Universal Plan: 55 passed; internal-writer
  authority: 52 passed; generic mutation-bypass audit: 169 passed, 0 warnings,
  0 failures.
- The architecture inventory reports 98 surfaces: 65 centrally authorized,
  2 bounded internal writers, 27 read/preview surfaces, 2 removed, and 2
  explicitly denied. No legacy or unclassified surface remains.
- The UX inventory regenerates byte-for-byte with 20 product claims and 15
  journeys. Production dependency audit reports zero vulnerabilities. The full
  audit reports one moderate `esbuild` advisory and one high aggregate `vite`
  advisory, both confined to development tooling; they remain scheduled for a
  compatible Vite upgrade without a forced major rewrite.

## Release disposition

Audit 3 verification is complete and the checkpoint is ready for source-control
finalization. It is not a release or fresh-VM acceptance claim. The next audit
should be fresh Debian VM acceptance covering first launch, packaged Web assets,
keyboard and screen-reader basics, local model readiness, restart continuity,
and uninstall/restore on a clean host.
