# Communications Migration v1

Checkpoint input:

- Tag: `v0.2.2-files-git-service-migration-v1`
- Commit: `240bcd134081af79397ffeda7549d2c71674f54b`

Communications Mutation Migration v1 migrates the communications surfaces that
currently exist in the repository. The implemented surface is notification
delivery/history:

- local notification record creation;
- configured Telegram notification delivery;
- notification history mark-read;
- notification history prune.

No email, calendar, Slack, Discord, SMS, webhook, or issue-comment provider is
implemented as a core communications provider in this checkpoint. Those actions
remain unsupported, not routed through a generic HTTP or shell fallback.

## Capabilities

Read-only:

- `notification.inspect`

Mutating:

- `notification.local.send`
- `notification.external.send`
- `notification.mark_read`
- `notification.prune`

Every mutating notification capability is registered in the central capability
registry, requires a Universal Mutation Plan, requires confirmation, records
receipt metadata, and forbids generic bypass.

## Provider and Account Binding

The external notification provider currently supported by runtime code is the
configured Telegram notification target. Plans bind:

- provider: `telegram`;
- account/configured bot identity by hashed account/configuration summary;
- destination chat by hash, never raw credential;
- message content by hash and length;
- fixture transport path for non-destructive proof.

The local notification target is treated as a local filesystem/history mutation.

## Recipient and Content Binding

Plans and receipts use safe summaries:

- destination hash;
- recipient count;
- body SHA-256;
- body length;
- short redacted preview;
- empty attachment inventory.

Full credentials, bot tokens, authorization headers, raw secret values, and
private attachment contents are not stored in Plans, receipts, or logs.

## Active-Channel Exception

Normal assistant responses over the active chat/Telegram transport remain the
transport response path, not a separate communications mutation, as long as the
response goes only to the current authenticated conversation and does not select
a new recipient, channel, group, webhook, attachment, or side-channel delivery.

Any separate notification delivery to a configured external target must use the
communications capability policy and Universal Plan path.

## Bypass Protection

`TelegramTarget.deliver` and `LocalTarget.deliver` require trusted invocation
context for mutation. Direct provider-client calls without context return
`generic_bypass_blocked` and do not call the send function.

Generic shell HTTP tools such as `curl` and `wget` remain blocked as destructive
command classes. The project does not expose an arbitrary `http.post`,
`network.post`, or webhook mutation capability.

## Runtime Revalidation

At execution time the notification executors revalidate:

- trusted invocation context;
- capability id;
- executor id;
- Plan fingerprint;
- destination hash when supplied;
- body hash when supplied;
- fixture transport boundary for proof sends;
- sensitive-content block before delivery.

Changed destination or content requires a new Plan.

## Idempotency

External notification proof delivery records the operation id in the fixture
transport log. Duplicate confirmation for the same Plan returns the existing
accepted delivery status and does not append another outbound fixture message.

If a real provider response is uncertain in a future provider migration, the
operation must report uncertainty and reconcile provider truth rather than
retrying blindly.

## Receipts and Status

Notification receipts include:

- provider;
- hashed account identity;
- capability id;
- executor id;
- Universal Plan schema and fingerprint metadata;
- recipient/audience summary;
- body hash and length;
- provider/fixture message id;
- delivery status.

Status UX should use notification store state, fixture/provider operation state,
and receipts. It must not infer delivery from conversational memory.

## Unsupported Providers

Email and calendar examples in the roadmap are policy requirements for future
provider migrations. They are not implemented in this checkpoint, and tests
prove there is no fallback path that sends email, calendar invitations, or
webhooks through arbitrary shell or HTTP mutation.

## Superseded Legacy Area

At the time of this checkpoint, the remaining expected authorization warning was
broader skill-pack mutation paths. Skill-Pack Permission Boundary v1 supersedes
that warning for Personal Agent platform APIs while documenting that arbitrary
malicious in-process Python skill code is not isolated.

## Proof

Run:

```bash
python scripts/communications_migration_smoke.py
python scripts/capability_policy_audit.py
python scripts/universal_plan_mode_audit.py
```

The communications smoke uses fake providers, local fixtures, and temporary
notification stores. It does not send real email, messages, invitations,
notifications, comments, or webhooks.
