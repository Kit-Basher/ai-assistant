# Model-led assistant loop (WP6.1)

> Status: **incomplete checkpoint**. This document records implemented
> boundaries and evaluation evidence only; WP6.1 is not released, promoted,
> or qualified until the complete acceptance and release gates pass.

`personal-agent.assistant-turn.v1` is the ordinary `/chat` contract. When the configured local model is available, it receives the unchanged user message, bounded conversation/pending context, and a catalog generated directly from `CapabilityRegistry`. It returns exactly one of `respond`, `clarify`, `invoke`, `propose_task`, `control_pending`, or `unsupported`.

The model is an untrusted planner and narrator. It has no actor, approval, policy, verifier, status, shell, endpoint, or filesystem authority. The kernel validates a closed document, injects runtime-bound `user_id`/message fields, validates inputs against the live registry contract, invokes only the registry, and treats every result as bounded untrusted data. Mutations invoke only the existing preview boundary; existing confirmation and WP3 durable-task authority remain unchanged.

Ceilings: one repair generation after invalid JSON, four tool rounds, eight capability calls, 12k characters of context/observations, and 4k characters of user-facing output. Model-unavailable mode is explicit and offers only documented structured operations; it never revives keyword routing.

Ordinary `/chat` no longer executes API request preview, `RequestUnderstandingService`, `nl_route`, `classify_runtime_chat_route`, or a social-response fast path. Those components remain only behind legacy structured/slash, operational, or exact bound-confirmation paths while their retirement census is completed.

Catalog fields are derived at request time from registry definitions: ID, purpose, availability/dependency, input types, mode, confirmation policy, and provenance. Dynamic pack capabilities therefore appear and disappear with registry registration without a prompt or router patch. Catalog metadata, observations, filenames, and pack data are delimited as untrusted data.
