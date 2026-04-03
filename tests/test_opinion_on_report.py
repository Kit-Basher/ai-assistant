import inspect
import unittest

from skills.opinion_on_report import handler


class _FakeRouteInference:
    def __init__(self, text: str, *, ok: bool = True) -> None:
        self.text = text
        self.ok = bool(ok)
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return {
            "ok": self.ok,
            "text": self.text,
            "provider": "ollama",
            "model": "ollama:qwen2.5:3b-instruct",
            "fallback_used": False,
            "attempts": [],
            "duration_ms": 1,
            "error_kind": None if self.ok else "llm_unavailable",
        }


class _ExplodingDirectClient:
    def generate(self, _prompt: str) -> str:
        raise AssertionError("direct presentation client should not be used")


class _ExplodingDB:
    def __getattr__(self, _name):
        raise AssertionError("DB access should not occur.")


class TestOpinionOnReport(unittest.TestCase):
    def test_valid_facts_generate(self) -> None:
        infer = _FakeRouteInference(
            "Based on the report:\n- Potential risks:\n- Things to keep an eye on:\n- Unknowns / missing data:"
        )
        result = handler.opinion_on_report(
            {
                "route_inference": infer,
                "llm_presentation_client": _ExplodingDirectClient(),
                "db": _ExplodingDB(),
            },
            facts={"mounts": [{"mountpoint": "/", "delta_used": 10}]},
            context_note="disk usage review",
        )
        self.assertEqual(1, len(infer.calls))
        self.assertEqual("chat", infer.calls[0].get("purpose"))
        self.assertEqual("skill.opinion_on_report", ((infer.calls[0].get("metadata") or {}) or {}).get("source_surface"))
        self.assertIn("Potential risks", result["text"])
        self.assertIn("facts_hash", result["data"])

    def test_missing_facts_error(self) -> None:
        infer = _FakeRouteInference("should not run")
        result = handler.opinion_on_report({"route_inference": infer}, facts={})
        self.assertIn("Facts are required", result["text"])
        self.assertEqual([], infer.calls)

    def test_new_number_rejected(self) -> None:
        infer = _FakeRouteInference("Risk increased by 99%.")
        result = handler.opinion_on_report(
            {"route_inference": infer},
            facts={"delta_used": 10},
        )
        self.assertEqual(1, len(infer.calls))
        self.assertEqual(result["text"], "I can’t form a reliable opinion from the provided facts.")

    def test_no_db_access(self) -> None:
        infer = _FakeRouteInference("No issues detected.")
        result = handler.opinion_on_report(
            {"route_inference": infer, "db": _ExplodingDB()},
            facts={"note": "ok"},
        )
        self.assertIn("source", result["data"])

    def test_llm_not_invoked_on_invalid(self) -> None:
        infer = _FakeRouteInference("should not run")
        result = handler.opinion_on_report({"route_inference": infer}, facts=None)
        self.assertEqual([], infer.calls)
        self.assertIn("Facts are required", result["text"])

    def test_handler_source_uses_canonical_route_only(self) -> None:
        source = inspect.getsource(handler._run_llm)
        self.assertIn("route_inference", source)
        self.assertNotIn("llm_router", source)
        self.assertNotIn("llm_broker", source)
        self.assertNotIn("llm_presentation_client", source)


if __name__ == "__main__":
    unittest.main()
