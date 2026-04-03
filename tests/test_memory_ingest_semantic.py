from __future__ import annotations

import unittest

from agent.conversation_memory import record_event
from agent.memory_ingest import ingest_event


class _FakeSemanticService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def ingest_conversation_text(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return {"ok": True}

    def ingest_note_text(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return {"ok": True}


class _FakeDB:
    def __init__(self, service: _FakeSemanticService) -> None:
        self._semantic_memory_service = service


class TestMemoryIngestSemantic(unittest.TestCase):
    def test_skill_events_are_ignored(self) -> None:
        service = _FakeSemanticService()
        db = _FakeDB(service)
        ingest_event(db, "user-1", "skill", "assistant ack", ["tag"])
        self.assertEqual([], service.calls)

    def test_user_events_are_indexed(self) -> None:
        service = _FakeSemanticService()
        db = _FakeDB(service)
        ingest_event(db, "user-1", "user", "remember this", ["tag"])
        self.assertEqual(1, len(service.calls))
        self.assertEqual("remember this", service.calls[0]["text"])

    def test_conversation_record_uses_canonical_ingest(self) -> None:
        service = _FakeSemanticService()
        db = _FakeDB(service)
        record_event(db, "user-1", "topic", "chat")
        self.assertEqual(1, len(service.calls))
        self.assertEqual("topic:user-1:topic", service.calls[0]["source_ref"])


if __name__ == "__main__":
    unittest.main()
