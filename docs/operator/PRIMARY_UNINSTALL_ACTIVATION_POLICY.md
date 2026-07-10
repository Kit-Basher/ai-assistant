# Primary Uninstall Activation Policy

Checkpoint truth:

- Tag: `v0.2.1-primary-uninstall-guard-wiring-v1`
- Commit: `e64a2f7f4708f43d9f58b619ea3144da99f583bb`

Primary uninstall is wired only for preserve-data uninstall. Purge is unsupported.

The assistant cannot enable its own primary uninstall capability from chat. Local operator activation is out-of-band and must be done with:

```bash
python scripts/primary_uninstall_policy.py enable \
  --acknowledge-primary-uninstall-capability \
  --expires-in-days 30
```

Disable:

```bash
python scripts/primary_uninstall_policy.py disable
```

Status:

```bash
python scripts/primary_uninstall_policy.py status
```

Marker path:

```text
~/.local/share/personal-agent/host_lifecycle/primary_uninstall_enabled.json
```

The v1 marker schema is strict and versioned. It contains `schema_version=1`, capability `primary_preserve_data_uninstall`, `enabled=true`, installation id, canonical repository path, primary service `personal-agent-api.service`, UTC `created_at` and `expires_at`, `created_by=local_operator_cli`, preserve-data policy, nonce, current uid, host id, and a SHA-256 payload integrity record.

Validation fails closed for malformed JSON, duplicate JSON keys, unknown schema versions, unknown fields, missing nonce, integrity mismatch, expired marker, future-created marker, wrong installation id, wrong repository path, wrong service, wrong uid, wrong host, purge-enabled policy, symlinks, non-regular files, hard-link ambiguity where detectable, wrong owner, oversized marker, marker mode broader than `0600`, or host-lifecycle directory mode broader than `0700`.

Default expiry is 30 days. Maximum expiry is 90 days. Permanent enablement is not supported.

Preview may show preserve-data uninstall behavior while disabled, but confirmation revalidates the marker before lock acquisition, final backup, operation record creation, or runner handoff. Marker races after preview fail closed with `mutated=false`, no backup, no operation record, and no runner handoff.

Accepted primary uninstall records only the marker fingerprint and policy summary, not raw marker JSON. The marker is consumed before runner handoff so preserved user data does not keep an active marker. Reinstall defaults back to disabled.

Normal updates do not create, rewrite, or extend the marker. The marker survives update-shaped runtime replacement because it lives under the host-lifecycle state root outside runtime releases, provided installation id, repository path, service, uid, host, permissions, integrity, and expiry remain valid. Rollback does not alter the marker.

No destructive primary uninstall proof was run for this policy batch. The activation-policy smoke uses isolated roots for mutation and reads the actual host marker status only.
