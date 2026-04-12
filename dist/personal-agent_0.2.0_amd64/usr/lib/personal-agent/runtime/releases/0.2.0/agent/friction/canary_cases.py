from __future__ import annotations


MUST_HAVE = [
    {
        "name": "long_user_claim_with_command_has_summary_and_next",
        "gate_passed": True,
        "user_text": "run tests and summarize",
        "rendered_answer": "\n".join(
            [
                "Status report:",
                "line 1",
                "line 2",
                "line 3",
                "line 4",
                "line 5",
                "line 6",
                "line 7",
                "Use `pytest -q` to verify.",
            ]
        ),
        "claims": [
            {
                "text": "Status report was prepared for the current request.",
                "support": "user",
                "user_turn_id": "thread-1:u:1",
            }
        ],
        "ctx": {
            "recent_turn_ids": ["thread-1:u:1"],
            "in_scope_memory_ids": [],
            "tool_event_ids": [],
        },
        "expect_summary": True,
        "expect_next": True,
        "expected_next_prefix": "Next: Run ",
    },
    {
        "name": "long_memory_claim_no_command_has_summary_only",
        "gate_passed": True,
        "user_text": "what do you remember",
        "rendered_answer": "\n".join(
            [
                "Memory report:",
                "line 1",
                "line 2",
                "line 3",
                "line 4",
                "line 5",
                "line 6",
                "line 7",
                "No runnable command is included.",
            ]
        ),
        "claims": [
            {
                "text": "Preference memory indicates concise output style.",
                "support": "memory",
                "memory_id": "mem:5",
            }
        ],
        "ctx": {
            "recent_turn_ids": ["thread-1:u:1"],
            "in_scope_memory_ids": ["mem:5"],
            "tool_event_ids": [],
        },
        "expect_summary": True,
        "expect_next": False,
    },
    {
        "name": "short_tool_claim_with_command_has_next_only",
        "gate_passed": True,
        "user_text": "run checks",
        "rendered_answer": "Use `pytest -q` to verify.",
        "claims": [
            {
                "text": "Tool run evidence exists for this check.",
                "support": "tool",
                "tool_event_id": "audit:7",
            }
        ],
        "ctx": {
            "recent_turn_ids": ["thread-1:u:1"],
            "in_scope_memory_ids": [],
            "tool_event_ids": ["audit:7"],
        },
        "expect_summary": False,
        "expect_next": True,
        "expected_next_prefix": "Next: Run ",
    },
    {
        "name": "long_tool_claim_with_command_has_summary_body_next_order",
        "gate_passed": True,
        "user_text": "please run and summarize",
        "rendered_answer": "\n".join(
            [
                "Runbook:",
                "step 1",
                "step 2",
                "step 3",
                "step 4",
                "step 5",
                "step 6",
                "step 7",
                "$ pytest -q",
            ]
        ),
        "claims": [
            {
                "text": "Tool-backed verification command is available in this response.",
                "support": "tool",
                "tool_event_id": "audit:9",
            }
        ],
        "ctx": {
            "recent_turn_ids": ["thread-1:u:1"],
            "in_scope_memory_ids": [],
            "tool_event_ids": ["audit:9"],
        },
        "expect_summary": True,
        "expect_next": True,
        "expected_next_prefix": "Next: Run ",
    },
]


MUST_NOT_APPEAR = [
    {
        "name": "intercept_reply_unchanged",
        "gate_passed": False,
        "intercepted_reply_text": "I’m not sure.\n\nWhat date should I use?",
        "expect_no_summary": True,
        "expect_no_next": True,
    },
    {
        "name": "short_pass_without_extractable_action",
        "gate_passed": True,
        "user_text": "status update",
        "rendered_answer": "Everything looks stable.",
        "claims": [
            {
                "text": "Status reflects the current request.",
                "support": "user",
                "user_turn_id": "thread-1:u:1",
            }
        ],
        "ctx": {
            "recent_turn_ids": ["thread-1:u:1"],
            "in_scope_memory_ids": [],
            "tool_event_ids": [],
        },
        "expect_no_summary": True,
        "expect_no_next": True,
    },
    {
        "name": "long_pass_without_supported_claim",
        "gate_passed": True,
        "user_text": "show long text",
        "rendered_answer": "\n".join(
            [
                "Long text:",
                "line 1",
                "line 2",
                "line 3",
                "line 4",
                "line 5",
                "line 6",
                "line 7",
                "line 8",
            ]
        ),
        "claims": [
            {
                "text": "Unanchored narrative claim.",
                "support": "none",
            }
        ],
        "ctx": {
            "recent_turn_ids": ["thread-1:u:1"],
            "in_scope_memory_ids": [],
            "tool_event_ids": [],
        },
        "expect_no_summary": True,
        "expect_no_next": True,
    },
    {
        "name": "short_command_without_evidence",
        "gate_passed": True,
        "user_text": "run tests",
        "rendered_answer": "Use `pytest -q` to verify.",
        "claims": [
            {
                "text": "Command suggestion with no supported provenance.",
                "support": "none",
            }
        ],
        "ctx": {
            "recent_turn_ids": [],
            "in_scope_memory_ids": [],
            "tool_event_ids": [],
        },
        "expect_no_summary": True,
        "expect_no_next": True,
    },
]

