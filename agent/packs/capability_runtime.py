from __future__ import annotations

"""Lifecycle-bound dynamic pack capabilities using the sole live registry."""

import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import threading
import time
import uuid
from typing import Any, Mapping

from agent.capability_registry import ApprovalPolicy, CapabilityContract, CapabilityDefinition, CapabilityMode, CapabilityProvenance, CapabilityRegistry
from agent.packs.capability_contracts import PackCapabilityContractError, canonical_json, digest, load_manifest, validate_value
from agent.packs.worker_runtime import SandboxedPackWorker


PACK_PROOF_REQUIREMENTS = (
    "health", "self_test", "verifier", "chat", "policy", "permission", "task",
    "restart", "failure", "timeout", "redaction", "digest", "revocation", "isolation",
)


class PackCapabilityStore:
    def __init__(self, db_path: str, storage_root: str | Path) -> None:
        self.db_path = str(db_path)
        self.storage_root = Path(storage_root).resolve() / "capability-runtime-v1"
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("""CREATE TABLE IF NOT EXISTS external_pack_capability_versions (
                record_id TEXT PRIMARY KEY, pack_id TEXT NOT NULL, version TEXT NOT NULL, pack_class TEXT NOT NULL,
                content_digest TEXT NOT NULL, manifest_json TEXT NOT NULL, artifact_root TEXT NOT NULL,
                review_approved INTEGER NOT NULL DEFAULT 0, enabled INTEGER NOT NULL DEFAULT 0,
                grants_json TEXT NOT NULL DEFAULT '[]', blocked_reason TEXT, self_test_json TEXT,
                last_invocation_json TEXT,
                created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
                UNIQUE(pack_id, content_digest))""")
            self._conn.execute("""CREATE TABLE IF NOT EXISTS external_pack_mutation_plans (
                plan_id TEXT PRIMARY KEY, action TEXT NOT NULL, payload_json TEXT NOT NULL,
                binding_digest TEXT NOT NULL, actor_id TEXT NOT NULL, session_id TEXT NOT NULL,
                thread_id TEXT NOT NULL, state TEXT NOT NULL, created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL, consumed_at INTEGER)""")
            columns = {str(row[1]) for row in self._conn.execute("PRAGMA table_info(external_pack_capability_versions)")}
            if "last_invocation_json" not in columns:
                self._conn.execute("ALTER TABLE external_pack_capability_versions ADD COLUMN last_invocation_json TEXT")
            self._conn.commit()

    def preview_mutation(self, action: str, payload: Mapping[str, Any], *, actor_id: str, session_id: str, thread_id: str) -> dict[str, Any]:
        if action == "import":
            source = Path(str(payload.get("source_dir") or "")).expanduser().resolve()
            manifest = load_manifest(source)
            canonical_payload = {"source_dir": str(source), "content_digest": manifest["content_digest"]}
            consequence = f"Import exact local {manifest['pack_class']} pack {manifest['id']} version {manifest['version']}. This does not approve or enable it."
        elif action == "gate":
            record_id = str(payload.get("record_id") or "")
            gate = str(payload.get("gate") or "")
            if gate not in {"review_approved", "enabled", "grants", "blocked_reason"}:
                raise ValueError("pack_gate_invalid")
            row = self.get(record_id)
            if row is None:
                raise ValueError("pack_not_found")
            value = payload.get("value", True)
            canonical_payload = {"record_id": record_id, "gate": gate, "value": value, "content_digest": row["content_digest"], "updated_at": row["updated_at"]}
            consequence = f"Set {gate} for exact pack {row['pack_id']} version {row['version']} to {canonical_json(value)}."
        elif action == "remove":
            record_id = str(payload.get("record_id") or "")
            row = self.get(record_id)
            if row is None:
                raise ValueError("pack_not_found")
            canonical_payload = {"record_id": record_id, "content_digest": row["content_digest"], "updated_at": row["updated_at"]}
            consequence = f"Remove exact pack {row['pack_id']} version {row['version']} and revoke its capabilities."
        else:
            raise ValueError("pack_mutation_action_invalid")
        plan_id = f"pack-plan-{uuid.uuid4().hex}"
        created = int(time.time()); expires = created + 300
        binding = {"schema_version": "personal-agent.pack-mutation.v1", "plan_id": plan_id, "action": action, "payload": canonical_payload, "actor_id": str(actor_id), "session_id": str(session_id), "thread_id": str(thread_id), "expires_at": expires}
        binding_digest = digest(binding)
        with self._lock:
            self._conn.execute(
                "INSERT INTO external_pack_mutation_plans(plan_id,action,payload_json,binding_digest,actor_id,session_id,thread_id,state,created_at,expires_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (plan_id, action, canonical_json(canonical_payload), binding_digest, str(actor_id), str(session_id), str(thread_id), "pending", created, expires),
            ); self._conn.commit()
        return {**binding, "binding_digest": binding_digest, "preview": consequence, "requires_confirmation": True}

    def apply_mutation(self, plan_id: str, binding_digest: str, *, actor_id: str, session_id: str, thread_id: str) -> dict[str, Any]:
        now = int(time.time())
        with self._lock:
            row = self._conn.execute("SELECT * FROM external_pack_mutation_plans WHERE plan_id=?", (plan_id,)).fetchone()
            if row is None:
                raise ValueError("pack_mutation_plan_not_found")
            if row["state"] != "pending":
                raise PermissionError("pack_mutation_plan_replayed")
            if now > int(row["expires_at"]):
                self._conn.execute("UPDATE external_pack_mutation_plans SET state='expired' WHERE plan_id=?", (plan_id,)); self._conn.commit()
                raise PermissionError("pack_mutation_plan_expired")
            if any((str(row[k]) != str(v)) for k, v in (("binding_digest", binding_digest), ("actor_id", actor_id), ("session_id", session_id), ("thread_id", thread_id))):
                raise PermissionError("pack_mutation_binding_mismatch")
            payload = json.loads(row["payload_json"]); action = str(row["action"])
            if action == "import":
                current_manifest = load_manifest(payload["source_dir"])
                if current_manifest["content_digest"] != payload["content_digest"]:
                    raise PermissionError("pack_content_changed_after_preview")
            else:
                current = self.get(str(payload["record_id"]))
                if current is None or current["content_digest"] != payload["content_digest"] or int(current["updated_at"]) != int(payload["updated_at"]):
                    raise PermissionError("pack_state_changed_after_preview")
            changed = self._conn.execute("UPDATE external_pack_mutation_plans SET state='consumed',consumed_at=? WHERE plan_id=? AND state='pending'", (now, plan_id)).rowcount
            self._conn.commit()
            if changed != 1:
                raise PermissionError("pack_mutation_plan_raced")
        if action == "import":
            return {"action": action, "record": self.import_local(payload["source_dir"])}
        if action == "gate":
            return {"action": action, "record": self.set_gate(payload["record_id"], payload["gate"], payload.get("value", True))}
        return {"action": action, "removed": self.remove(payload["record_id"]), "record_id": payload["record_id"]}

    def import_local(self, source_dir: str | Path) -> dict[str, Any]:
        source = Path(source_dir).expanduser().resolve()
        manifest = load_manifest(source)
        record_id = f"{manifest['id']}-{manifest['content_digest'][:12]}"
        target = self.storage_root / record_id
        if target.exists():
            existing = self.get(record_id)
            if existing:
                return existing
        temp = self.storage_root / f".{record_id}.{time.time_ns()}.tmp"
        temp.mkdir(mode=0o700)
        try:
            (temp / "personal-agent-pack.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
            for content in manifest.get("content_files") or []:
                name = str(content["path"]); src = source / name; destination = temp / name
                if src.is_symlink() or not src.is_file() or src.stat().st_nlink != 1:
                    raise PackCapabilityContractError("pack_content_changed_during_import")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, destination); destination.chmod(0o400)
                if hashlib.sha256(destination.read_bytes()).hexdigest() != content["sha256"]:
                    raise PackCapabilityContractError("pack_content_changed_during_import")
            temp.rename(target)
        finally:
            if temp.exists():
                shutil.rmtree(temp, ignore_errors=True)
        now = int(time.time())
        with self._lock:
            # A content or contract update is a new authority.  It invalidates
            # every gate on older versions before the new record can exist.
            self._conn.execute(
                "UPDATE external_pack_capability_versions SET review_approved=0,enabled=0,grants_json='[]',self_test_json=NULL,blocked_reason='superseded',updated_at=? WHERE pack_id=? AND content_digest<>?",
                (now, manifest["id"], manifest["content_digest"]),
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO external_pack_capability_versions(record_id,pack_id,version,pack_class,content_digest,manifest_json,artifact_root,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (record_id, manifest["id"], manifest["version"], manifest["pack_class"], manifest["content_digest"], canonical_json(manifest), str(target), now, now),
            )
            self._conn.commit()
        return self.get(record_id) or {}

    def _row(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        try:
            manifest = json.loads(row["manifest_json"]); grants = json.loads(row["grants_json"]); self_test = json.loads(row["self_test_json"] or "null"); last_invocation = json.loads(row["last_invocation_json"] or "null")
        except Exception:
            manifest, grants, self_test, last_invocation = {}, [], None, None
        return {"record_id": row["record_id"], "pack_id": row["pack_id"], "version": row["version"], "pack_class": row["pack_class"], "content_digest": row["content_digest"], "manifest": manifest, "artifact_root": row["artifact_root"], "review_approved": bool(row["review_approved"]), "enabled": bool(row["enabled"]), "grants": grants, "blocked_reason": row["blocked_reason"], "self_test": self_test, "last_invocation": last_invocation, "created_at": row["created_at"], "updated_at": row["updated_at"]}

    def record_invocation(self, record_id: str, *, capability_id: str, ok: bool, outcome: str, worker: Mapping[str, Any] | None = None) -> None:
        safe = {"capability_id": str(capability_id), "ok": bool(ok), "outcome": str(outcome)[:80], "at": int(time.time())}
        if isinstance(worker, Mapping):
            safe["worker"] = {key: worker.get(key) for key in ("isolated", "engine", "namespace", "elapsed_ms", "exit_code", "orphaned") if key in worker}
        with self._lock:
            self._conn.execute("UPDATE external_pack_capability_versions SET last_invocation_json=? WHERE record_id=?", (canonical_json(safe), record_id)); self._conn.commit()

    def get(self, record_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._row(self._conn.execute("SELECT * FROM external_pack_capability_versions WHERE record_id=?", (record_id,)).fetchone())

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [x for row in self._conn.execute("SELECT * FROM external_pack_capability_versions ORDER BY pack_id,created_at") if (x := self._row(row))]

    def set_gate(self, record_id: str, gate: str, value: Any = True) -> dict[str, Any]:
        if gate not in {"review_approved", "enabled", "grants", "blocked_reason", "self_test"}:
            raise ValueError("pack_gate_invalid")
        current = self.get(record_id)
        if current is None:
            raise ValueError("pack_not_found")
        if gate == "enabled" and value and not current["review_approved"]:
            raise PermissionError("pack_review_required")
        column = "grants_json" if gate == "grants" else "self_test_json" if gate == "self_test" else gate
        stored = canonical_json(value) if gate in {"grants", "self_test"} else value
        with self._lock:
            self._conn.execute(f"UPDATE external_pack_capability_versions SET {column}=?,updated_at=? WHERE record_id=?", (stored, int(time.time()), record_id))
            self._conn.commit()
        return self.get(record_id) or {}

    def remove(self, record_id: str) -> bool:
        current = self.get(record_id)
        if current is None:
            return False
        with self._lock:
            self._conn.execute("DELETE FROM external_pack_capability_versions WHERE record_id=?", (record_id,)); self._conn.commit()
        root = Path(current["artifact_root"])
        try:
            root.relative_to(self.storage_root)
            shutil.rmtree(root, ignore_errors=True)
        except ValueError:
            pass
        return True


class DynamicPackCapabilityRuntime:
    def __init__(self, *, store: PackCapabilityStore, registry: CapabilityRegistry, response_factory, worker: SandboxedPackWorker | None = None) -> None:
        self.store = store; self.registry = registry; self.response_factory = response_factory
        self.worker = worker or SandboxedPackWorker()
        self._registered: set[str] = set()

    @staticmethod
    def _python_contract(schema: Mapping[str, Any], *, add_chat_context: bool = True) -> CapabilityContract:
        types = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}
        props = {k: types[v["type"]] for k, v in (schema.get("properties") or {}).items()}
        if add_chat_context:
            props.update({"user_id": str, "text": str})
        return CapabilityContract(properties=props, required=tuple(schema.get("required") or ()), allow_extra=False)

    def lifecycle(self, row: Mapping[str, Any]) -> dict[str, Any]:
        manifest = row.get("manifest") if isinstance(row.get("manifest"), dict) else {}
        reason = str(row.get("blocked_reason") or "") or None
        artifact_root = Path(str(row.get("artifact_root") or ""))
        digest_ok = bool(manifest and digest({k: v for k, v in manifest.items() if k != "content_digest"}) == row.get("content_digest"))
        artifacts_ok = digest_ok and artifact_root.is_dir()
        if artifacts_ok:
            expected_paths = {"personal-agent-pack.json", *(str(item["path"]) for item in manifest.get("content_files") or [])}
            actual_paths = {path.relative_to(artifact_root).as_posix() for path in artifact_root.rglob("*") if path.is_file()}
            artifacts_ok = actual_paths == expected_paths
            for content in manifest.get("content_files") or []:
                path = artifact_root / str(content["path"])
                try:
                    artifacts_ok = artifacts_ok and path.is_file() and not path.is_symlink() and path.stat().st_size == int(content["size"]) and hashlib.sha256(path.read_bytes()).hexdigest() == content["sha256"]
                except OSError:
                    artifacts_ok = False
        requested_set = {p for cap in manifest.get("capabilities") or [] for p in cap.get("permissions") or []}
        for cap in manifest.get("capabilities") or []:
            invocation = cap.get("invocation") if isinstance(cap.get("invocation"), dict) else {}
            if invocation.get("kind") == "registered_capability":
                underlying = self.registry.get(str(invocation.get("capability_id") or ""))
                if underlying is not None:
                    requested_set.update(underlying.permission_requirements)
        requested = sorted(requested_set)
        grants = sorted(str(x) for x in (row.get("grants") or []))
        worker_health = self.worker.health() if row.get("pack_class") == "sandboxed_executable" else None
        mutating_declaration = any(str(cap.get("mode")) == "mutating" for cap in manifest.get("capabilities") or [])
        usable = bool(row.get("review_approved") and row.get("enabled") and not reason and artifacts_ok and not mutating_declaration and set(requested) <= set(grants) and (worker_health is None or worker_health.available))
        if reason: missing = "safety_review"
        elif not artifacts_ok: missing = "integrity"
        elif mutating_declaration: missing = "unsupported_effect_broker"
        elif not row.get("review_approved"): missing = "review_approval"
        elif not row.get("enabled"): missing = "enablement"
        elif not set(requested) <= set(grants): missing = "permission"
        elif worker_health and not worker_health.available: missing = "isolation_runtime"
        else: missing = None
        return {"usable": usable, "missing_gate": missing, "integrity_ok": artifacts_ok, "requested_permissions": requested, "granted_permissions": grants, "worker": {"available": worker_health.available, "reason": worker_health.reason} if worker_health else None}

    def register_usable(self) -> dict[str, Any]:
        result = {"registered": [], "unavailable": [], "failures": []}
        for row in self.store.list():
            life = self.lifecycle(row)
            if not life["usable"]:
                result["unavailable"].append({"record_id": row["record_id"], **life}); continue
            for cap in row["manifest"].get("capabilities") or []:
                try:
                    self._register(row, cap)
                    definition = self.registry.require(str(cap["capability_id"]))
                    self_test = definition.self_test()
                    self.store.set_gate(str(row["record_id"]), "self_test", self_test)
                    if not bool(self_test.get("ok")):
                        self.registry.unregister_external(str(cap["capability_id"]))
                        result["failures"].append({"capability_id": cap.get("capability_id"), "reason": "self_test_failed"})
                    else:
                        result["registered"].append(cap["capability_id"])
                except Exception as exc:
                    result["failures"].append({"capability_id": cap.get("capability_id"), "reason": exc.__class__.__name__})
        return result

    def _register(self, row: Mapping[str, Any], cap: Mapping[str, Any]) -> None:
        capability_id = str(cap["capability_id"])
        if self.registry.get(capability_id):
            raise ValueError("pack_capability_duplicate_or_shadow")
        mode = CapabilityMode(str(cap["mode"]))
        invocation = cap.get("invocation") if isinstance(cap.get("invocation"), dict) else {}
        underlying = self.registry.get(str(invocation.get("capability_id") or "")) if invocation.get("kind") == "registered_capability" else None
        if underlying is not None:
            if underlying.mode is CapabilityMode.MUTATING and mode is not CapabilityMode.MUTATING:
                raise ValueError("pack_policy_downgrade")
            if underlying.approval_policy is ApprovalPolicy.REQUIRED and mode is not CapabilityMode.MUTATING:
                raise ValueError("pack_approval_downgrade")
        record_id = str(row["record_id"]); contract_digest = str(cap["contract_digest"])
        def current() -> tuple[dict[str, Any] | None, dict[str, Any]]:
            now = self.store.get(record_id); return now, self.lifecycle(now or {})
        def health() -> tuple[bool, str | None]:
            now, life = current()
            exact = next((x for x in ((now or {}).get("manifest") or {}).get("capabilities", []) if x.get("capability_id") == capability_id and x.get("contract_digest") == contract_digest), None)
            return (bool(now and life["usable"] and exact), None if now and life["usable"] and exact else str(life.get("missing_gate") or "pack_version_changed"))
        def invoke(payload: Mapping[str, Any]):
            now, life = current()
            if not now or not life["usable"]:
                raise RuntimeError(str(life.get("missing_gate") or "pack_unavailable"))
            pack_input = {k: v for k, v in payload.items() if k not in {"user_id", "text"}}
            pack_input = validate_value(cap["input_schema"], pack_input, field="pack_input")
            is_self_test = str(payload.get("user_id") or "").startswith("pack-self-test:")
            if now["pack_class"] == "declarative":
                spec = cap["invocation"]; underlying = self.registry.require(spec["capability_id"])
                if underlying.provenance is not CapabilityProvenance.NATIVE:
                    raise RuntimeError("pack_declaration_native_authority_required")
                mapped = {}
                for key, value in spec["inputs"].items():
                    if isinstance(value, str) and value.startswith("$input."):
                        mapped[key] = pack_input[value.split(".", 1)[1]]
                    else: mapped[key] = value
                mapped.setdefault("user_id", str(payload.get("user_id") or "pack")); mapped.setdefault("text", str(payload.get("text") or cap["description"]))
                try:
                    underlying_result = self.registry.invoke(spec["capability_id"], mapped)
                except Exception as exc:
                    if not is_self_test:
                        self.store.record_invocation(record_id, capability_id=capability_id, ok=False, outcome=f"underlying_{exc.__class__.__name__}")
                    raise
                data = {"result": getattr(underlying_result, "text", str(underlying_result)), "underlying_capability_id": spec["capability_id"], "pack_binding": self._binding(now, cap)}
            else:
                field = cap["invocation"]["input_field"]
                result = self.worker.invoke(module_path=Path(now["artifact_root"]) / cap["invocation"]["module"], artifact_digest=cap["artifact_digest"], export=cap["invocation"]["export"], value=pack_input[field], limits=cap["limits"])
                if not result.get("ok"):
                    if not is_self_test:
                        self.store.record_invocation(record_id, capability_id=capability_id, ok=False, outcome=str(result.get("error_kind") or "worker_failed"), worker=result.get("worker"))
                    raise RuntimeError(str(result.get("error_kind") or "worker_failed"))
                data = {"result": result["untrusted_output"]["result"], "worker": result["worker"], "pack_binding": self._binding(now, cap)}
            validate_value(cap["output_schema"], {"result": data["result"]}, field="pack_output")
            if not is_self_test:
                self.store.record_invocation(record_id, capability_id=capability_id, ok=True, outcome="verified_result", worker=data.get("worker"))
            return self.response_factory(capability_id, cap["display_name"], data)
        def verify(result: Any) -> bool:
            data = getattr(result, "data", None)
            payload = data if isinstance(data, dict) else {}
            runtime = payload.get("runtime_payload") if isinstance(payload.get("runtime_payload"), dict) else payload.get("runtime") if isinstance(payload.get("runtime"), dict) else payload
            actual = runtime.get("result") if isinstance(runtime, dict) else None
            return isinstance(actual, int) if cap["verifier"]["kind"] == "integer_result" else bool(str(actual or "").strip())
        def self_test() -> dict[str, Any]:
            payload = {**cap["self_test_input"], "user_id": f"pack-self-test:{record_id}", "text": cap["description"]}
            try: return {"ok": verify(invoke(payload)), "status": "pass", "binding": self._binding(row, cap)}
            except Exception as exc: return {"ok": False, "status": "fail", "reason": exc.__class__.__name__}
        definition = CapabilityDefinition(
            capability_id=capability_id,
            description=f"{cap['description']}. Provided by the reviewed external pack {row['manifest'].get('display_name') or row['pack_id']}.",
            example_goals=tuple(cap["examples"]),
            input_contract=self._python_contract(cap["input_schema"]), output_contract=CapabilityContract(properties={"response": object}, required=("response",)),
            mode=mode, approval_policy=ApprovalPolicy.REQUIRED if mode is CapabilityMode.MUTATING else ApprovalPolicy.NEVER,
            invocation_hook=invoke, verification_hook=verify, health_hook=health, provenance=CapabilityProvenance.PACK,
            capability_type=f"external_{row['pack_class']}", material_group=f"pack:{row['pack_id']}",
            permission_requirements=tuple(sorted(set(cap["permissions"]) | set(underlying.permission_requirements if underlying is not None else ()))), mode_requirements=("exact_pack_binding",),
            unavailable_message=f"{cap['display_name']} is installed but not usable until its exact pack lifecycle gates pass.",
            proof_requirements=PACK_PROOF_REQUIREMENTS, proof_nodes={x: ("tests/test_wp4_pack_runtime.py",) for x in PACK_PROOF_REQUIREMENTS},
            self_test_hook=self_test, task_composable=bool(cap["task_composable"]), retry_safety="read_only_safe" if mode is CapabilityMode.READ_ONLY else "reconcile_first", max_task_retries=1 if mode is CapabilityMode.READ_ONLY else 0, resume_policy="revalidate",
        )
        self.registry.register(definition); self._registered.add(capability_id)

    @staticmethod
    def _binding(row: Mapping[str, Any], cap: Mapping[str, Any]) -> dict[str, Any]:
        return {"pack_id": row["pack_id"], "record_id": row["record_id"], "version": row["version"], "content_digest": row["content_digest"], "contract_digest": cap["contract_digest"], "abi": cap["invocation"].get("abi") or "declarative.v1"}
