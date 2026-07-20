from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
import sqlite3
import tempfile
import time

import pytest

from agent.confirmation_transactions import (
    ConfirmationTransactionStore,
    operation_key_for_plan,
)
from agent.executor_registry import ExecutorRegistry, ExecutorSpec
from agent.mutation_plan import build_mutation_confirmation, build_mutation_plan
from memory.db import MemoryDB


def _fixture_plan(plan_id: str = "cross-process-plan") -> tuple[dict, dict]:
    mutation_plan = build_mutation_plan(
        plan_id=plan_id,
        capability_id="files.create",
        executor_id="operator.file.create.v1",
        expires_at_epoch=int(time.time()) + 300,
        actor_id="alice",
        thread_id="thread-a",
        session_id="session-a",
        target_snapshot={"target_path": "/fixture/target", "content_sha256": "fixture"},
        mutation_inventory=[{"operation": "fixture_create"}],
    )
    wrapper = {
        **mutation_plan,
        "mutation_plan": dict(mutation_plan),
        "action_type": "operator.file.create",
        "target": "fixture target",
        "executor_status": "enabled",
        "risk_level": "medium",
    }
    confirmation = build_mutation_confirmation(mutation_plan, confirmation_id="confirmation-one")
    return wrapper, confirmation


def _race_worker(root: str, barrier, queue) -> None:
    base = Path(root)
    wrapper, confirmation = _fixture_plan()
    registry = ExecutorRegistry(base / "journal.jsonl", confirmation_store_path=base / "confirmations.sqlite3")

    def execute(_plan, _action):
        with (base / "effects.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        return {"ok": True, "mutated": True}

    registry.register(ExecutorSpec(
        executor_id="operator.file.create.v1",
        action_type="operator.file.create",
        status="enabled",
        run=execute,
        capability_id="files.create",
    ))
    registry.freeze()
    barrier.wait(timeout=10)
    result = registry.execute_confirmed_plan(
        plan=wrapper,
        action={"pending_id": wrapper["plan_id"]},
        confirmation=confirmation,
    )
    queue.put((result.ok, result.error_code))


def _crash_after_execution_boundary(root: str) -> None:
    base = Path(root)
    wrapper, confirmation = _fixture_plan("crash-plan")
    registry = ExecutorRegistry(base / "journal.jsonl", confirmation_store_path=base / "confirmations.sqlite3")

    def execute(_plan, _action):
        (base / "crash-effect").write_text("mutated", encoding="utf-8")
        os._exit(19)

    registry.register(ExecutorSpec(
        executor_id="operator.file.create.v1",
        action_type="operator.file.create",
        status="enabled",
        run=execute,
        capability_id="files.create",
    ))
    registry.freeze()
    registry.execute_confirmed_plan(
        plan=wrapper,
        action={"pending_id": wrapper["plan_id"]},
        confirmation=confirmation,
    )


def test_two_processes_consume_one_confirmation_once() -> None:
    with tempfile.TemporaryDirectory() as raw:
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(2)
        queue = context.Queue()
        processes = [context.Process(target=_race_worker, args=(raw, barrier, queue)) for _ in range(2)]
        for process in processes:
            process.start()
        for process in processes:
            process.join(20)
            assert process.exitcode == 0
        results = [queue.get(timeout=2) for _ in processes]
        assert sum(1 for ok, _reason in results if ok) == 1
        assert sum(1 for ok, _reason in results if not ok) == 1
        assert len((Path(raw) / "effects.log").read_text(encoding="utf-8").splitlines()) == 1


def test_stale_reserved_and_executing_states_never_retry() -> None:
    with tempfile.TemporaryDirectory() as raw:
        store = ConfirmationTransactionStore(Path(raw) / "confirmations.sqlite3", stale_after_seconds=30)
        wrapper, confirmation = _fixture_plan("reserved-plan")
        plan = wrapper["mutation_plan"]
        reservation = store.reserve(plan=plan, confirmation=confirmation)
        assert reservation.allowed
        recovered = store.reconcile_stale(now_epoch=int(time.time()) + 31)
        assert recovered == {"failed_before_execution": 1, "indeterminate": 0}
        replay = store.reserve(plan=plan, confirmation=confirmation)
        assert not replay.allowed
        assert replay.reason_code == "mutation_confirmation_replayed"

        wrapper2, confirmation2 = _fixture_plan("executing-plan")
        plan2 = wrapper2["mutation_plan"]
        reservation2 = store.reserve(plan=plan2, confirmation=confirmation2)
        assert store.mark_executing(reservation2.operation_key, reservation2.owner_id)
        recovered2 = store.reconcile_stale(now_epoch=int(time.time()) + 31)
        assert recovered2 == {"failed_before_execution": 0, "indeterminate": 1}
        replay2 = store.reserve(plan=plan2, confirmation=confirmation2)
        assert not replay2.allowed
        assert replay2.reason_code == "mutation_confirmation_reconciliation_required"


def test_crash_after_execution_boundary_becomes_indeterminate() -> None:
    with tempfile.TemporaryDirectory() as raw:
        context = multiprocessing.get_context("spawn")
        process = context.Process(target=_crash_after_execution_boundary, args=(raw,))
        process.start()
        process.join(20)
        assert process.exitcode == 19
        assert (Path(raw) / "crash-effect").read_text(encoding="utf-8") == "mutated"

        store = ConfirmationTransactionStore(Path(raw) / "confirmations.sqlite3", stale_after_seconds=30)
        recovered = store.reconcile_stale(now_epoch=int(time.time()) + 31)
        assert recovered["indeterminate"] == 1
        wrapper, confirmation = _fixture_plan("crash-plan")
        retry = store.reserve(plan=wrapper["mutation_plan"], confirmation=confirmation)
        assert not retry.allowed
        assert retry.reason_code == "mutation_confirmation_reconciliation_required"


def test_terminal_result_survives_restart_and_is_redacted_shape() -> None:
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "confirmations.sqlite3"
        wrapper, confirmation = _fixture_plan("restart-plan")
        registry = ExecutorRegistry(Path(raw) / "journal.jsonl", confirmation_store_path=path)
        registry.register(ExecutorSpec(
            executor_id="operator.file.create.v1",
            action_type="operator.file.create",
            status="enabled",
            run=lambda _plan, _action: {"ok": True, "mutated": True, "details": {"token": "must-not-persist"}},
            capability_id="files.create",
        ))
        result = registry.execute_confirmed_plan(plan=wrapper, action={"pending_id": wrapper["plan_id"]}, confirmation=confirmation)
        assert result.ok
        row = ConfirmationTransactionStore(path).get(operation_key_for_plan(wrapper["mutation_plan"]))
        assert row is not None and row["state"] == "succeeded"
        assert "must-not-persist" not in str(row["result_json"])

        restarted = ExecutorRegistry(Path(raw) / "journal.jsonl", confirmation_store_path=path)
        restarted.register(ExecutorSpec(
            executor_id="operator.file.create.v1",
            action_type="operator.file.create",
            status="enabled",
            run=lambda _plan, _action: {"ok": True, "mutated": True},
            capability_id="files.create",
        ))
        replay = restarted.execute_confirmed_plan(plan=wrapper, action={"pending_id": wrapper["plan_id"]}, confirmation=confirmation)
        assert replay.error_code == "mutation_confirmation_replayed"


def test_expiry_is_rechecked_inside_durable_reservation() -> None:
    with tempfile.TemporaryDirectory() as raw:
        store = ConfirmationTransactionStore(Path(raw) / "confirmations.sqlite3")
        plan = build_mutation_plan(
            plan_id="expired-at-reserve",
            capability_id="files.create",
            executor_id="operator.file.create.v1",
            expires_at_epoch=int(time.time()) + 1,
            actor_id="alice",
            thread_id="thread-a",
            session_id="session-a",
            target_snapshot={"target_path": "/fixture/target"},
            mutation_inventory=[{"operation": "fixture_create"}],
        )
        confirmation = build_mutation_confirmation(plan, confirmation_id="confirmation-expiring")
        time.sleep(1.2)
        result = store.reserve(plan=plan, confirmation=confirmation)
        assert not result.allowed
        assert result.reason_code == "mutation_confirmation_expired"
        assert store.get(result.operation_key) is None


def test_schema_is_idempotent_restrictive_and_legacy_rows_migrate() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        legacy_path = root / "executor_registry_journal.jsonl.confirmations.sqlite3"
        legacy = ConfirmationTransactionStore(legacy_path)
        wrapper, confirmation = _fixture_plan("legacy-plan")
        reserved = legacy.reserve(plan=wrapper["mutation_plan"], confirmation=confirmation)
        assert reserved.allowed
        assert legacy.mark_executing(reserved.operation_key, reserved.owner_id)
        assert legacy.finish(reserved.operation_key, reserved.owner_id, state="succeeded", result={"ok": True})

        canonical_path = root / "confirmation_transactions.sqlite3"
        canonical = ConfirmationTransactionStore(canonical_path, legacy_paths=[legacy_path])
        reopened = ConfirmationTransactionStore(canonical_path, legacy_paths=[legacy_path])
        assert reopened.get(reserved.operation_key)["state"] == "succeeded"
        assert canonical_path.stat().st_mode & 0o777 == 0o600
        with sqlite3.connect(canonical_path) as connection:
            assert connection.execute(
                "SELECT schema_version FROM confirmation_transaction_meta WHERE singleton = 1"
            ).fetchone()[0] == 1
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        keeper = reopened._connect()
        try:
            keeper.execute("BEGIN IMMEDIATE")
            keeper.execute("UPDATE confirmation_transaction_meta SET schema_version = schema_version WHERE singleton = 1")
            for suffix in ("-wal", "-shm"):
                sidecar = Path(f"{canonical_path}{suffix}")
                assert sidecar.is_file()
                assert sidecar.stat().st_mode & 0o777 == 0o600
            keeper.execute("COMMIT")
        finally:
            keeper.close()


def test_separate_transaction_schema_does_not_modify_schema_v2_agent_database() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        database = MemoryDB(str(root / "agent.db"))
        database.init_schema(str(Path(__file__).resolve().parents[1] / "memory" / "schema.sql"))
        database.close()
        before = (root / "agent.db").read_bytes()
        ConfirmationTransactionStore(root / "confirmation_transactions.sqlite3")
        assert (root / "agent.db").read_bytes() == before
        with sqlite3.connect(root / "agent.db") as connection:
            assert connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()[0] == "2"


def test_transaction_boundary_revalidates_scope_before_creating_a_row() -> None:
    with tempfile.TemporaryDirectory() as raw:
        store = ConfirmationTransactionStore(Path(raw) / "confirmations.sqlite3")
        wrapper, confirmation = _fixture_plan("scope-plan")
        confirmation["thread_id"] = "other-thread"
        result = store.reserve(plan=wrapper["mutation_plan"], confirmation=confirmation)
        assert not result.allowed
        assert result.reason_code == "mutation_confirmation_thread_id_mismatch"
        assert store.get(result.operation_key) is None


def test_terminal_transaction_blocks_replay_when_append_only_journal_fails() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        wrapper, confirmation = _fixture_plan("receipt-failure-plan")
        registry = ExecutorRegistry(root / "journal.jsonl", confirmation_store_path=root / "confirmations.sqlite3")
        effects: list[int] = []

        def execute(_plan, _action):
            effects.append(1)
            return {"ok": True, "mutated": True}

        registry.register(ExecutorSpec(
            executor_id="operator.file.create.v1",
            action_type="operator.file.create",
            status="enabled",
            run=execute,
            capability_id="files.create",
        ))
        registry.journal.append = lambda _record: (_ for _ in ()).throw(OSError("fixture journal failure"))
        with pytest.raises(OSError, match="fixture journal failure"):
            registry.execute_confirmed_plan(
                plan=wrapper,
                action={"pending_id": wrapper["plan_id"]},
                confirmation=confirmation,
            )
        registry.journal.append = lambda _record: None
        replay = registry.execute_confirmed_plan(
            plan=wrapper,
            action={"pending_id": wrapper["plan_id"]},
            confirmation=confirmation,
        )
        assert replay.error_code == "mutation_confirmation_replayed"
        assert effects == [1]
