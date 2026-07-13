# Full Adversarial Authorization Proof v1

Current checkpoint:

- Tag: `v0.2.2-generic-mutation-bypass-hardening-v1`
- Commit: `b56a449ae157a3e96e160d01b436e4f887d5252b`

Full Adversarial Authorization Proof v1 attacks the complete supported
mutation chain:

```text
request -> intent routing -> capability selection -> target normalization
-> Universal Mutation Plan -> confirmation -> runtime revalidation
-> trusted invocation context -> Executor Registry -> low-level primitive
-> final-state verification -> receipt and status
```

It is a proof checkpoint, not a new user-facing capability expansion.

## Security Properties

The proof runner encodes these properties:

- P1 fixed authority: untrusted callers cannot choose capability, executor,
  authorization mode, trusted caller type, grant, context, receipt result, or
  mutation truth.
- P2 exact target binding: authorization applies only to the normalized target
  shown in the Plan.
- P3 single-use authorization: confirmation, trusted context, callback binding,
  and mutation execution cannot be replayed.
- P4 scope isolation: authorization cannot cross thread, account, provider,
  repository, root, service, skill, version, grant, operation, or target.
- P5 runtime truth: policy and target state are revalidated immediately before
  mutation.
- P6 primitive enforcement: direct low-level helpers reject missing or
  mismatched trusted context.
- P7 durable mutation truth: receipts and status reflect committed fixture
  results.
- P8 failure truth: partial, uncertain, failed, no-op, and pre-mutation failures
  are not misreported.
- P9 fail closed: malformed, stale, expired, mismatched, or incomplete
  authorization data returns `mutated=false` unless a fixture explicitly models
  a mutation that already occurred.
- P10 fixture isolation: fixture-only contexts are denied in production mode.

## Attack Matrix

The reviewed case inventory is:

```text
docs/operator/ADVERSARIAL_AUTHORIZATION_CASES.json
```

The inventory records stable case ids, security properties, attack class, entry
surface, caller type, target type, expected decision, expected mutation truth,
expected reason, and fixture requirement.

Covered entry surfaces include:

- direct Python helper calls;
- Executor Registry dispatch;
- capability gate;
- Universal Mutation Plan store;
- skill-pack broker;
- provider adapter;
- shell skill;
- primitive-denial helpers;
- internal callback/context validation;
- API authority-field schema proof.

Covered mutation classes include:

- filesystem create/modify/delete;
- Git commit and shell Git mutation denial;
- service restart target allowlist;
- notification/provider delivery;
- package shell mutation denial;
- backup/restore/support/memory through integrated evidence gates;
- skill grants and target scopes;
- internal DB, secret, HTTP, and shell primitive denial.

## Proof Runner

Run:

```bash
python scripts/full_adversarial_authorization_proof.py
```

Expected summary:

```text
PASS=54 WARN=1 FAIL=0 SKIP=0
PROPERTIES_PROVEN=10/10
RELEASE_BLOCKERS=0
```

The warning is the documented process-isolation limitation. It is not an
authorization failure.

The runner writes machine-readable evidence to:

```text
/tmp/full_adversarial_authorization_proof_evidence.json
```

## Proof Areas

Forged capability/executor proof:

- raw `capability_id` and `executor_id` in Plan/action payloads do not override
  trusted executor metadata;
- unknown executors deny;
- capability/executor mismatches fail Plan validation;
- high-risk mutation without explicit confirmation denies.

Trusted-context proof:

- missing context denies;
- wrong capability, wrong executor, wrong operation, wrong Plan fingerprint,
  wrong target fingerprint, expired context, consumed context, fixture context
  in production, and unknown caller type all deny before mutation.

Plan and confirmation replay proof:

- tampered Plan fingerprints deny;
- reused Plan ids with changed targets deny;
- cancelled and expired Plans are not executable;
- confirmation copied to another Plan denies;
- duplicate notification confirmation returns the previous fixture status
  without sending twice.

Target-drift proof:

- file content hash drift denies;
- file symlink replacement denies;
- staged Git diff drift denies;
- service allowlist drift denies;
- notification destination and content drift deny.

Scope-isolation proof:

- trusted context cannot cross operation or target;
- skill grant revocation denies;
- skill target-scope expansion denies;
- skill updates do not inherit new permissions.

Skill-pack proof:

- undeclared permission denies;
- declared but ungranted permission denies;
- revoked grant denies;
- raw capability/executor fields in skill payload are ignored in favor of the
  mapped permission;
- newly requested update permission requires a new grant.

API and control-plane proof:

- supported API request schemas do not accept authority override fields such as
  capability id, executor id, trusted context, decision id, Plan fingerprint,
  or grant id;
- installed-runtime API override behavior is also exercised by
  `scripts/generic_mutation_bypass_smoke.py` when the local API is reachable.

Background-task proof:

- a background origin without confirmation receives `confirmation_required`;
- no background path receives reusable confirmation or trusted context through
  supported platform APIs.

Primitive bypass proof:

- raw HTTP mutation, arbitrary shell mutation, direct DB domain mutation, raw
  secret read, direct provider send, direct Git shell mutation, and direct
  package shell mutation all deny with `mutated=false`.

Concurrency and duplicate proof:

- duplicate file creation is bounded by target state and returns no second
  mutation;
- duplicate notification delivery is idempotent by operation id in the fixture
  transport.

Partial and uncertain outcome proof:

- a fixture executor returning post-mutation partial status records
  `mutated=true`;
- an exception before verified mutation records `mutated=false`;
- `ExecutorPartialFailure` records a safe no-mutation partial failure according
  to current registry semantics;
- uncertain external provider outcome is represented as uncertain, not
  delivered.

Receipt and status proof:

- a successful fixture mutation writes an Executor Registry journal receipt with
  matching mutation truth;
- a Plan store reload preserves Plan fingerprint/status truth.

## Integrated Evidence

The full proof complements, rather than replaces, these focused gates:

```bash
python scripts/capability_policy_smoke.py
python scripts/capability_policy_audit.py
python scripts/universal_plan_mode_smoke.py
python scripts/universal_plan_mode_audit.py
python scripts/generic_mutation_bypass_audit.py
python scripts/generic_mutation_bypass_smoke.py
python scripts/skill_pack_permission_boundary_smoke.py
python scripts/communications_migration_smoke.py
python scripts/files_git_service_migration_smoke.py
python scripts/executor_authorization_migration_smoke.py
```

`scripts/prove_ready.py` includes the full adversarial proof as a required
authorization gate.

## Known Limitation

This proof does not claim process isolation for arbitrary malicious in-process
Python code. The guarantee is for supported platform APIs, trusted runtime
boundaries, and registered executor/skill-pack invocation surfaces.

External untrusted code execution remains future process-isolation work.

## Recommended Checkpoint

```text
v0.2.2-full-adversarial-authorization-proof-v1
```

This is not final `v0.2.2`.
