#!/usr/bin/env python3
from __future__ import annotations

import multiprocessing
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.confirmation_transactions import ConfirmationTransactionStore
from agent.mutation_plan import build_mutation_confirmation, build_mutation_plan


def _plan() -> dict:
    return build_mutation_plan(
        plan_id="transaction-smoke-plan",
        capability_id="files.create",
        executor_id="operator.file.create.v1",
        expires_at_epoch=4_102_444_800,
        actor_id="actor",
        thread_id="thread",
        session_id="session",
        target_snapshot={"target": "fixture"},
        mutation_inventory=[{"operation": "fixture"}],
    )


def _confirmation(plan: dict) -> dict:
    return build_mutation_confirmation(plan, confirmation_id="transaction-smoke-confirmation")


def _reserve(path: str, start: multiprocessing.synchronize.Event, output: multiprocessing.queues.Queue, plan: dict, confirmation: dict) -> None:
    store = ConfirmationTransactionStore(path)
    start.wait(10)
    result = store.reserve(plan=plan, confirmation=confirmation)
    output.put((result.allowed, result.reason_code, result.operation_key, result.owner_id))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="personal-agent-confirmation-proof-") as tmp:
        path = str(Path(tmp) / "confirmation.sqlite3")
        context = multiprocessing.get_context("spawn")
        start = context.Event()
        output = context.Queue()
        plan = _plan()
        confirmation = _confirmation(plan)
        workers = [context.Process(target=_reserve, args=(path, start, output, plan, confirmation)) for _ in range(4)]
        for worker in workers:
            worker.start()
        start.set()
        rows = [output.get(timeout=20) for _ in workers]
        for worker in workers:
            worker.join(20)
            if worker.exitcode != 0:
                print(f"FAIL: subprocess exit={worker.exitcode}")
                return 1
        winners = [row for row in rows if row[0]]
        if len(winners) != 1:
            print(f"FAIL: expected one reservation winner, got {len(winners)}")
            return 1
        store = ConfirmationTransactionStore(path)
        winner = winners[0]
        if not store.mark_executing(winner[2], winner[3]):
            print("FAIL: winning reservation could not enter executing")
            return 1
        if not store.finish(winner[2], winner[3], state="succeeded", result={"ok": True}):
            print("FAIL: terminal receipt was not durable")
            return 1
        replay = store.reserve(plan=plan, confirmation=confirmation)
        if replay.allowed or replay.reason_code != "mutation_confirmation_replayed":
            print("FAIL: durable replay was not blocked")
            return 1
        print("PASS: four subprocesses produced exactly one durable winner")
        print("PASS: terminal persistence survives reopen and blocks replay")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
