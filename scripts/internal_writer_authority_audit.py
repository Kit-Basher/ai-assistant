#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.internal_writer_authority import (  # noqa: E402
    INTERNAL_WRITER_REGISTRY_SCHEMA,
    InternalWriterRegistry,
)


REGISTRY_PATH = ROOT / "docs" / "operator" / "INTERNAL_WRITER_REGISTRY_V1.json"
CLASSIFICATION_PATH = ROOT / "docs" / "operator" / "MUTATION_FILE_CLASSIFICATIONS_V2B.json"
REQUIRED_FIELDS = {
    "writer_id", "module", "disposition", "capability_id",
    "allowed_operations", "resource_types", "target_scopes",
    "allowed_triggers", "argument_schema", "modes", "audit", "retry",
    "evidence",
}
INTERNAL_DISPOSITIONS = {
    "trusted_bookkeeping",
    "scheduled_maintenance",
    "mixed_internal_and_public_pending",
}
ENFORCED_INTERNAL_DISPOSITIONS = {"trusted_bookkeeping", "scheduled_maintenance"}


def normalized_registry_bytes() -> bytes:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return (json.dumps(payload, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def run_audit() -> tuple[list[str], list[str], dict[str, int]]:
    failures: list[str] = []
    passes: list[str] = []
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != INTERNAL_WRITER_REGISTRY_SCHEMA:
        failures.append("registry_schema_invalid")
    rows = payload.get("writers") if isinstance(payload.get("writers"), list) else []
    if len(rows) != 24:
        failures.append(f"registry_count:{len(rows)}")
    indexed: dict[str, dict] = {}
    modules: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            failures.append("registry_row_invalid")
            continue
        writer_id = str(row.get("writer_id") or "")
        missing = sorted(REQUIRED_FIELDS - set(row))
        if missing:
            failures.append(f"{writer_id}:missing:{','.join(missing)}")
        if writer_id in indexed:
            failures.append(f"{writer_id}:duplicate")
        indexed[writer_id] = row
        module = str(row.get("module") or "")
        if module in modules:
            failures.append(f"{writer_id}:duplicate_module:{module}")
        modules[module] = writer_id
        if not row.get("allowed_operations") or not row.get("resource_types") or not row.get("target_scopes"):
            failures.append(f"{writer_id}:unbounded_contract")
        if any(str(target).startswith(("/", "~")) for target in row.get("target_scopes", [])):
            failures.append(f"{writer_id}:host_path_in_contract")
        audit = row.get("audit") if isinstance(row.get("audit"), dict) else {}
        if not audit.get("required") or not audit.get("redacted"):
            failures.append(f"{writer_id}:audit_not_required_redacted")
        retry = row.get("retry") if isinstance(row.get("retry"), dict) else {}
        if row.get("disposition") in INTERNAL_DISPOSITIONS and not retry.get("durable_operation_id_required"):
            failures.append(f"{writer_id}:automatic_retry_without_operation_id")
        passes.append(f"registered:{writer_id}:{row.get('disposition')}")
        if row.get("disposition") in ENFORCED_INTERNAL_DISPOSITIONS:
            module_path = ROOT / (module.replace(".", "/") + ".py")
            source = module_path.read_text(encoding="utf-8") if module_path.is_file() else ""
            if not any(marker in source for marker in ("execute_internal_write(", "perform_registered_internal_write(")):
                failures.append(f"{writer_id}:runtime_enforcement_missing")
            else:
                passes.append(f"runtime_enforced:{writer_id}")

    classifications = json.loads(CLASSIFICATION_PATH.read_text(encoding="utf-8")).get("classifications", [])
    by_path = {str(row.get("path") or ""): row for row in classifications if isinstance(row, dict)}
    for module, writer_id in modules.items():
        path = module.replace(".", "/") + ".py"
        row = by_path.get(path)
        if row is None:
            failures.append(f"{writer_id}:classification_missing:{path}")
            continue
        if str(row.get("contract_identity") or "") != writer_id:
            failures.append(f"{writer_id}:classification_identity_mismatch")
    classified_contracts = {
        str(row.get("contract_identity") or "")
        for row in classifications
        if isinstance(row, dict) and str(row.get("contract_identity") or "") != "not_applicable"
    }
    if classified_contracts != set(indexed):
        failures.append("classification_registry_set_mismatch")

    api = (ROOT / "agent" / "api_server.py").read_text(encoding="utf-8")
    control = (ROOT / "agent" / "control_plane.py").read_text(encoding="utf-8")
    broker = (ROOT / "agent" / "skill_pack_permissions.py").read_text(encoding="utf-8")
    for name, source in (("api", api), ("control_plane", control), ("skill_pack", broker)):
        if "reject_public_internal_authority_claim" not in source:
            failures.append(f"public_boundary_missing:{name}")
        else:
            passes.append(f"public_boundary_rejects_internal_claim:{name}")
    packaging = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if 'docs/operator/INTERNAL_WRITER_REGISTRY_V1.json", "share/personal-agent/INTERNAL_WRITER_REGISTRY_V1.json' not in packaging:
        failures.append("internal_writer_registry_not_packaged")
    else:
        passes.append("internal_writer_registry_packaged")
    service_expectations = {
        "stable_service": (ROOT / "systemd" / "personal-agent-api.service", "Environment=AGENT_DB_PATH=%h/.local/share/personal-agent/agent.db"),
        "dev_service": (ROOT / "systemd" / "personal-agent-api-dev.service", "Environment=AGENT_DB_PATH=%h/.local/share/personal-agent/agent.db"),
        "release_bundle": (ROOT / "packaging" / "release_bundle" / "install.sh", "Environment=AGENT_DB_PATH=$state_root/agent.db"),
        "debian_service": (ROOT / "packaging" / "debian" / "personal-agent-api.service.in", "Environment=AGENT_DB_PATH=%h/.local/share/personal-agent/agent.db"),
    }
    for name, (path, expected) in service_expectations.items():
        if expected not in path.read_text(encoding="utf-8"):
            failures.append(f"canonical_state_root_mismatch:{name}")
        else:
            passes.append(f"canonical_state_root:{name}")
    orchestrator = (ROOT / "agent" / "orchestrator.py").read_text(encoding="utf-8")
    for marker in (
        'confirmation_store_path=state_root / "confirmation_transactions.sqlite3"',
        "legacy_confirmation_store_paths=[executor_journal.with_name",
        '"confirmation_transactions": str(state_root / "confirmation_transactions.sqlite3")',
    ):
        if marker not in orchestrator:
            failures.append(f"canonical_confirmation_wiring_missing:{marker}")
    if not any(item.startswith("canonical_confirmation_wiring_missing:") for item in failures):
        passes.append("canonical_confirmation_storage_and_backup_wiring")
    for relative in (
        "agent/cli.py",
        "agent/orchestrator.py",
        "agent/telegram_bridge.py",
        "telegram_adapter/bot.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        if "InternalWriterAuthority" in source or "InternalWriterFactory" in source:
            failures.append(f"public_internal_authority_constructor:{relative}")
        else:
            passes.append(f"public_boundary_cannot_construct_internal_authority:{relative}")
    counts = Counter(str(row.get("disposition") or "") for row in rows if isinstance(row, dict))
    return passes, failures, dict(sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--emit-normalized", action="store_true")
    args = parser.parse_args()
    if args.emit_normalized:
        sys.stdout.buffer.write(normalized_registry_bytes())
        return 0
    passes, failures, counts = run_audit()
    if args.json:
        print(json.dumps({"passes": passes, "failures": failures, "counts": counts}, indent=2, sort_keys=True))
    else:
        for row in passes:
            print(f"PASS: {row}")
        for row in failures:
            print(f"FAIL: {row}")
        print(f"COUNTS={json.dumps(counts, sort_keys=True)}")
        print(f"PASS={len(passes)} WARN=0 FAIL={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
