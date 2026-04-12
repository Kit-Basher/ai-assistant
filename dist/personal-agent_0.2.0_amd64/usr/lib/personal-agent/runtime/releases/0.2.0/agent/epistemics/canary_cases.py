from __future__ import annotations


MUST_INTERCEPT = [
    {
        "name": "unresolved_reference_ambiguous",
        "user_text": "do that again",
        "ctx": {
            "active_thread_id": "thread-1",
            "referents": ["[1] Task A", "[2] Task B"],
            "recent_turn_ids": ["thread-1:u:1"],
            "thread_turn_count": 2,
        },
        "candidate": {
            "final_answer": "Done.",
            "claims": [{"text": "Action completed.", "support": "user"}],
        },
    },
    {
        "name": "missing_required_slot_schedule",
        "user_text": "schedule it",
        "ctx": {
            "active_thread_id": "thread-1",
            "thread_turn_count": 1,
            "recent_turn_ids": ["thread-1:u:1"],
        },
        "candidate": {
            "final_answer": "Scheduled.",
            "claims": [{"text": "Scheduling confirmed.", "support": "user"}],
        },
    },
    {
        "name": "cross_thread_memory_boundary",
        "user_text": "what should I do now",
        "ctx": {
            "active_thread_id": "thread-1",
            "recent_turn_ids": ["thread-1:u:1"],
            "thread_turn_count": 2,
            "in_scope_memory_ids": ["mem:local-1"],
            "out_of_scope_memory": ["mem:other-2"],
            "out_of_scope_relevant_memory": True,
        },
        "candidate": {
            "final_answer": "Use your previous project plan.",
            "claims": [{"text": "Previous plan exists.", "support": "memory", "memory_id": "mem:other-2"}],
        },
    },
    {
        "name": "memory_miss_required",
        "user_text": "what did we decide last week",
        "ctx": {
            "active_thread_id": "thread-1",
            "recent_turn_ids": ["thread-1:u:1"],
            "thread_turn_count": 1,
            "memory_miss": True,
        },
        "candidate": {
            "final_answer": "You decided to launch.",
            "claims": [{"text": "Launch decision exists.", "support": "user"}],
        },
    },
    {
        "name": "tool_failure_signal",
        "user_text": "run the operation",
        "ctx": {
            "active_thread_id": "thread-1",
            "recent_turn_ids": ["thread-1:u:1"],
            "thread_turn_count": 1,
            "tool_failures": ["runner_not_configured"],
            "tool_event_ids": ["audit:10"],
        },
        "candidate": {
            "final_answer": "Executed successfully.",
            "claims": [{"text": "Tool succeeded.", "support": "tool", "tool_event_id": "audit:10"}],
        },
    },
    {
        "name": "soft_cross_thread_phrase",
        "user_text": "summarize this",
        "ctx": {
            "active_thread_id": "thread-1",
            "recent_turn_ids": ["thread-1:u:1"],
            "thread_turn_count": 2,
        },
        "candidate": {
            "final_answer": "As we discussed earlier, continue with the same plan.",
            "claims": [{"text": "Summary generated.", "support": "user"}],
        },
    },
    {
        "name": "new_thread_summary_drift",
        "user_text": "status",
        "ctx": {
            "active_thread_id": "thread-2",
            "recent_turn_ids": ["thread-2:u:1"],
            "thread_turn_count": 0,
        },
        "candidate": {
            "final_answer": "As usual, keep the same cadence.",
            "claims": [{"text": "Cadence is known.", "support": "user"}],
        },
    },
    {
        "name": "provenance_memory_missing_id",
        "user_text": "recall that",
        "ctx": {
            "active_thread_id": "thread-1",
            "recent_turn_ids": ["thread-1:u:1"],
            "thread_turn_count": 1,
            "in_scope_memory_ids": ["mem:1", "mem:2"],
        },
        "candidate": {
            "hydrate_provenance": False,
            "final_answer": "Loaded memory detail.",
            "claims": [{"text": "From memory.", "support": "memory"}],
        },
    },
]


MUST_PASS = [
    {
        "name": "valid_user_claim",
        "user_text": "what did I ask",
        "ctx": {
            "active_thread_id": "thread-1",
            "recent_turn_ids": ["thread-1:u:1"],
            "thread_turn_count": 2,
        },
        "candidate": {
            "final_answer": "You asked for status.",
            "claims": [{"text": "Asked for status.", "support": "user", "user_turn_id": "thread-1:u:1"}],
        },
    },
    {
        "name": "valid_memory_claim_in_scope",
        "user_text": "what do you remember",
        "ctx": {
            "active_thread_id": "thread-1",
            "recent_turn_ids": ["thread-1:u:1"],
            "thread_turn_count": 2,
            "in_scope_memory_ids": ["mem:5"],
            "in_scope_memory": ["mem:5"],
        },
        "candidate": {
            "final_answer": "You prefer concise answers.",
            "claims": [{"text": "Preference stored.", "support": "memory", "memory_id": "mem:5"}],
        },
    },
    {
        "name": "valid_tool_claim_in_scope",
        "user_text": "did the tool run",
        "ctx": {
            "active_thread_id": "thread-1",
            "recent_turn_ids": ["thread-1:u:1"],
            "thread_turn_count": 2,
            "tool_event_ids": ["audit:7"],
        },
        "candidate": {
            "final_answer": "The tool run completed.",
            "claims": [{"text": "Tool event recorded.", "support": "tool", "tool_event_id": "audit:7"}],
        },
    },
    {
        "name": "valid_mixed_claims_hydrated",
        "user_text": "confirm everything",
        "ctx": {
            "active_thread_id": "thread-1",
            "recent_turn_ids": ["thread-1:u:3"],
            "thread_turn_count": 3,
            "in_scope_memory_ids": ["mem:42"],
            "tool_event_ids": ["audit:9"],
        },
        "candidate": {
            "final_answer": "Confirmed.",
            "claims": [
                {"text": "From user.", "support": "user"},
                {"text": "From memory.", "support": "memory"},
                {"text": "From tool.", "support": "tool"},
            ],
        },
    },
    {
        "name": "definitional_none_claim",
        "user_text": "what is 2+2",
        "ctx": {
            "active_thread_id": "thread-1",
            "recent_turn_ids": ["thread-1:u:1"],
            "thread_turn_count": 2,
        },
        "candidate": {
            "final_answer": "2 + 2 = 4.",
            "claims": [{"text": "2 + 2 = 4", "support": "none"}],
        },
    },
]

