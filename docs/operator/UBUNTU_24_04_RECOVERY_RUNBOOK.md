# Ubuntu 24.04 recovery acceptance runbook

Status: pre-reinstall runbook. The physical journey has not yet been run.

This is the only deliberately destructive boundary in WP6. Do not begin it
until a verified `personal-agent.portable-backup.v2` archive exists on media
that will survive the OS install and the user has explicitly approved the
fresh Ubuntu installation.

## Before reinstalling

1. In Personal Agent, open Diagnostics and export the redacted diagnostic
   report. Record the displayed version/commit, selected model, Safe Mode,
   remote-fallback state, filesystem roots, and pack count.
2. Ask the assistant to create a portable backup. Verify its manifest and
   archive digest using the built-in validation action. Copy the archive and
   its displayed SHA-256 to separate removable or otherwise persistent media.
3. Keep provider/Telegram credentials in the user's password manager. Portable
   backup deliberately excludes the machine-bound encrypted secret file;
   secrets are re-entered in Setup after restore.
4. Run `python scripts/ubuntu_recovery_preflight.py --json` only as the final
   operator check. Confirm `ok: true`. This command is not a replacement for
   the normal UI backup and validation actions.

## Human physical actions

The user performs the Ubuntu 24.04 installation, disk/partition choices,
account creation, Wi-Fi/network entry, encryption passphrase handling, and
restores access to the persistent backup medium. Personal Agent does not
automate or approve these OS-level decisions.

## Install, restore, and verify

1. Install the documented base dependencies, obtain the trusted Personal
   Agent repository, and check out the exact release commit recorded in the
   backup evidence. Run the single supported `scripts/install_local.sh` path.
2. Stop at the recovery checkpoint printed by the installer. The current
   pre-reinstall candidate proves archive validation and restoration into an
   isolated empty state root, but it does **not** yet claim that the installed
   Web UI can replace a running state database from an uploaded archive. Use
   the physical acceptance procedure recorded with the final WP6 candidate;
   do not extract or copy archive members by hand. The fresh-host journey must
   fail until that checkpoint validates the exact manifest/digest and restores
   into the new empty state root without conflicts.
3. Re-enter optional provider or Telegram secrets in Setup. Select an installed
   local model if model artifacts were not preserved separately; model files
   are deliberately outside the Personal Agent backup.
4. Confirm readiness, selected/effective model truth, Safe Mode, remote
   fallback, roots, conversation/history counts, memory status, task status,
   packs, grants, and capability health.
5. Run the post-install verification workflow from Diagnostics. Exercise:
   presence and ordinary conversation; filesystem list/search/read; system and
   model status; a multi-step read-only task; supported pack discovery,
   quarantine, separate review/grant/enable/use; and restart persistence.
6. Upgrade through the normal exact-preview lifecycle, verify exact runtime
   identity and preserved state, then roll back to the recorded prior runtime
   and verify again. Return to the intended current release only after both
   checks pass.

## Stop conditions

Stop without improvising if archive validation fails, the backup contract is
newer than the installer, the exact release cannot be verified, restore reports
a target conflict, database integrity is not `ok`, a secret appears in the
diagnostics export, the model changes silently, Safe Mode/fallback differs, a
pack is unexpectedly usable, upgrade lacks a rollback checkpoint, or any
assistant success claim lacks verifier evidence.

Successful completion of this runbook, including a normal-user recovery
handoff that is verified on the physical fresh host, with the resulting
machine-readable journey report, is the missing evidence required to change
WP6 from `PRE-REINSTALL GATE COMPLETE` to `COMPLETE`. The absence of a proven
running-Web-UI portable-restore action is intentionally recorded here rather
than disguised as completed behavior.
