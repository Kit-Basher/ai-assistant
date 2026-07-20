from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import tempfile

import pytest

from agent.internal_writer_authority import (
    InternalWriterAuthority,
    InternalWriterFactory,
    InternalWriterJournal,
    InternalWriterRegistry,
    execute_internal_write,
    reject_public_internal_authority_claim,
)


def _registry(path: Path) -> InternalWriterRegistry:
    payload = {
        "schema": "personal-agent.internal-writer-registry.v1",
        "writers": [
            {
                "writer_id": "fixture_writer",
                "module": __name__,
                "disposition": "scheduled_maintenance",
                "capability_id": "internal.fixture.write",
                "allowed_operations": ["replace_state"],
                "resource_types": ["fixture_state"],
                "target_scopes": ["state:fixture"],
                "allowed_triggers": ["scheduler"],
                "argument_schema": {"max_properties": 4, "max_total_bytes": 1024},
                "modes": ["safe", "controlled"],
                "audit": {"required": True, "redacted": True},
                "retry": {"durable_operation_id_required": True},
                "evidence": "fixture",
            },
            {
                "writer_id": "public_writer",
                "module": __name__,
                "disposition": "operator_triggered_legacy",
                "capability_id": "legacy.fixture.write",
                "allowed_operations": ["write"],
                "resource_types": ["fixture_state"],
                "target_scopes": ["state:fixture"],
                "allowed_triggers": ["operator"],
                "argument_schema": {"max_properties": 1, "max_total_bytes": 64},
                "modes": ["controlled"],
                "audit": {"required": True, "redacted": True},
                "retry": {"durable_operation_id_required": False},
                "evidence": "fixture",
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return InternalWriterRegistry(path)


def test_scoped_authority_is_single_use_audited_and_redacted() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        registry = _registry(root / "registry.json")
        factory = InternalWriterFactory(trigger="scheduler", registry=registry)
        authority = factory.issue("fixture_writer", operation_id="job-2026-07-20", runtime_mode="safe")
        journal = InternalWriterJournal(root / "journal.sqlite3")
        assert (root / "journal.sqlite3").stat().st_mode & 0o777 == 0o600

        def write_state() -> str:
            return "done"

        result = execute_internal_write(
            authority=authority,
            operation="replace_state",
            resource_type="fixture_state",
            target_scope="state:fixture",
            arguments={"record_id": "one", "token": "not-allowed-before-redaction"},
            callback=write_state,
            journal=journal,
            registry=registry,
        )
        assert result == "done"
        with sqlite3.connect(root / "journal.sqlite3") as connection:
            row = connection.execute("SELECT state, request_json FROM internal_writer_operations").fetchone()
        assert row[0] == "succeeded"
        assert "not-allowed-before-redaction" not in row[1]
        with pytest.raises(ValueError, match="invalid_or_replayed"):
            execute_internal_write(
                authority=authority,
                operation="replace_state",
                resource_type="fixture_state",
                target_scope="state:fixture",
                arguments={},
                callback=write_state,
                journal=journal,
                registry=registry,
            )


def test_forged_scope_escalation_and_arbitrary_arguments_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        registry = _registry(root / "registry.json")
        factory = InternalWriterFactory(trigger="scheduler", registry=registry)
        journal = InternalWriterJournal(root / "journal.sqlite3")

        def write_state() -> None:
            return None

        forged = InternalWriterAuthority("fixture_writer", "internal.fixture.write", "scheduler", "forged", "safe", "made-up")
        with pytest.raises(ValueError, match="invalid_or_replayed"):
            execute_internal_write(authority=forged, operation="replace_state", resource_type="fixture_state", target_scope="state:fixture", arguments={}, callback=write_state, journal=journal, registry=registry)

        wrong_resource = factory.issue("fixture_writer", operation_id="wrong-resource")
        with pytest.raises(ValueError, match="resource_denied"):
            execute_internal_write(authority=wrong_resource, operation="replace_state", resource_type="filesystem", target_scope="state:fixture", arguments={}, callback=write_state, journal=journal, registry=registry)

        bad_arguments = factory.issue("fixture_writer", operation_id="bad-arguments")
        with pytest.raises(ValueError, match="argument_denied:executor_id"):
            execute_internal_write(authority=bad_arguments, operation="replace_state", resource_type="fixture_state", target_scope="state:fixture", arguments={"executor_id": "evil"}, callback=write_state, journal=journal, registry=registry)


def test_public_and_serialized_authority_claims_are_denied() -> None:
    with tempfile.TemporaryDirectory() as raw:
        registry = _registry(Path(raw) / "registry.json")
        with pytest.raises(ValueError, match="not_internal"):
            InternalWriterFactory(trigger="scheduler", registry=registry).issue("public_writer", operation_id="x")
    assert reject_public_internal_authority_claim({"nested": {"internal_writer_id": "fixture_writer"}}) == "internal_writer_id"
    assert reject_public_internal_authority_claim({"message": "internal_writer_id is untrusted prose"}) is None


def test_mixed_writer_cannot_receive_internal_authority_until_separated() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        payload = {
            "schema": "personal-agent.internal-writer-registry.v1",
            "writers": [{
                "writer_id": "mixed_writer", "module": __name__,
                "disposition": "mixed_internal_and_public_pending",
                "capability_id": "internal.fixture.mixed", "allowed_operations": ["write"],
                "resource_types": ["fixture_state"], "target_scopes": ["state:fixture"],
                "allowed_triggers": ["scheduler"],
                "argument_schema": {"max_properties": 1, "max_total_bytes": 64},
                "modes": ["safe"], "audit": {"required": True, "redacted": True},
                "retry": {"durable_operation_id_required": True}, "evidence": "fixture",
            }],
        }
        path = root / "registry.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="not_internal"):
            InternalWriterFactory(trigger="scheduler", registry=InternalWriterRegistry(path)).issue(
                "mixed_writer", operation_id="job"
            )


def test_duplicate_scheduled_operation_is_not_reexecuted() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        registry = _registry(root / "registry.json")
        journal = InternalWriterJournal(root / "journal.sqlite3")
        calls: list[int] = []

        def write_state() -> None:
            calls.append(1)

        for attempt in range(2):
            authority = InternalWriterFactory(trigger="scheduler", registry=registry).issue("fixture_writer", operation_id="same-job")
            if attempt == 0:
                execute_internal_write(authority=authority, operation="replace_state", resource_type="fixture_state", target_scope="state:fixture", arguments={}, callback=write_state, journal=journal, registry=registry)
            else:
                with pytest.raises(ValueError, match="duplicate_operation"):
                    execute_internal_write(authority=authority, operation="replace_state", resource_type="fixture_state", target_scope="state:fixture", arguments={}, callback=write_state, journal=journal, registry=registry)
        assert calls == [1]


def test_stale_internal_execution_becomes_indeterminate_and_stays_consumed() -> None:
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "journal.sqlite3"
        journal = InternalWriterJournal(path)
        assert journal.reserve("operation", {
            "writer_id": "fixture_writer", "operation_id": "job", "capability_id": "internal.fixture.write",
            "operation": "replace_state", "resource_type": "fixture_state", "target_scope": "state:fixture",
            "trigger": "scheduler",
        })
        assert journal.reconcile_stale(before_iso="9999-01-01T00:00:00+00:00") == 1
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT state FROM internal_writer_operations").fetchone()[0] == "indeterminate"
        assert not journal.reserve("operation", {
            "writer_id": "fixture_writer", "operation_id": "job", "capability_id": "internal.fixture.write",
            "operation": "replace_state", "resource_type": "fixture_state", "target_scope": "state:fixture",
            "trigger": "scheduler",
        })
