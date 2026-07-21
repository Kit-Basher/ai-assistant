# Notification Delivery UX

Public send/test/read/prune actions use the canonical preview and approval
path. A preview identifies the transport and recipient scope without exposing
private message content in Plans, receipts, logs, or transaction ledgers.

Delivery intent and redacted target/content fingerprints are durable before
transport I/O. If the process stops after delivery may have started but before
the result is durable, the operation restores as `indeterminate`. That means
the assistant genuinely does not know whether delivery occurred: it must not
call the operation successful or failed and must never automatically retry or
resend it.

Public resolve, resend, and abandon operations for indeterminate deliveries are
currently unavailable. Provider idempotency may reduce duplicates but does not
justify an exactly-once claim. Internal delivery authority may record outcomes
only; it cannot select new content or recipients or initiate a send. Telegram
is optional and disabled at the Audit 3 checkpoint.
