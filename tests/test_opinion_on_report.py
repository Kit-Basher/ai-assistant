import unittest

from skills.opinion_on_report import handler


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.called = False

    def generate(self, _prompt: str) -> str:
        self.called = True
        return self.text


class _ExplodingDB:
    def __getattr__(self, _name):
        raise AssertionError("DB access should not occur.")


class TestOpinionOnReport(unittest.TestCase):
    def test_valid_facts_generate(self) -> None:
        client = _FakeClient(
            "Based on the report:\n- Potential risks:\n- Things to keep an eye on:\n- Unknowns / missing data:"
        )
        result = handler.opinion_on_report(
            {"llm_presentation_client": client, "db": _ExplodingDB()},
            facts={"mounts": [{"mountpoint": "/", "delta_used": 10}]},
            context_note="disk usage review",
        )
        self.assertTrue(client.called)
        self.assertIn("Potential risks", result["text"])
        self.assertIn("facts_hash", result["data"])

    def test_missing_facts_error(self) -> None:
        client = _FakeClient("should not run")
        result = handler.opinion_on_report({"llm_presentation_client": client}, facts={})
        self.assertIn("Facts are required", result["text"])
        self.assertFalse(client.called)

    def test_new_number_rejected(self) -> None:
        client = _FakeClient("Risk increased by 99%.")
        result = handler.opinion_on_report(
            {"llm_presentation_client": client},
            facts={"delta_used": 10},
        )
        self.assertEqual(result["text"], "I can’t form a reliable opinion from the provided facts.")

    def test_no_db_access(self) -> None:
        client = _FakeClient("No issues detected.")
        result = handler.opinion_on_report(
            {"llm_presentation_client": client, "db": _ExplodingDB()},
            facts={"note": "ok"},
        )
        self.assertIn("source", result["data"])

    def test_llm_not_invoked_on_invalid(self) -> None:
        client = _FakeClient("should not run")
        result = handler.opinion_on_report({"llm_presentation_client": client}, facts=None)
        self.assertFalse(client.called)
        self.assertIn("Facts are required", result["text"])


if __name__ == "__main__":
    unittest.main()
