# Bad Chat Case Fixtures

Use this folder to turn a live bad chat route into a regression case quickly.

Create a small JSON file with either one object or a list of objects:

```json
{
  "case_id": "live.dot_tts_lookup",
  "category": "fixture/live regression",
  "message": "can you look up dot.tts im wondering if it would be a good model to use for a project",
  "expected_semantic_intent": "web_search",
  "expected_route": "action_tool",
  "expected_kind": "safe_web_search",
  "expect_search": true,
  "must_not_contain": ["Linux Troubleshooting Workflow", "voice output"]
}
```

Then run:

```bash
python scripts/chat_eval.py
python -m pytest -q tests/test_adversarial_chat_routing.py
```

Do not put secrets, private text, raw hostile pack contents, or API tokens in
fixtures. Reduce the transcript to the smallest user message that proves the
bad route.
