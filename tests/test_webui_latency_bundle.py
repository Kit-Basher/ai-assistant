from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestWebuiLatencyBundle(unittest.TestCase):
    def test_webui_chat_bundle_contains_deferred_placeholder_latency_instrumentation(self) -> None:
        bundle = (REPO_ROOT / "agent" / "webui" / "dist" / "assets" / "index-m6LtTPCx.js").read_text(
            encoding="utf-8"
        )

        for snippet in (
            "chatRequestPendingRef",
            "chat/ui.request_start",
            "chat/ui.placeholder_shown",
            "chat/ui.placeholder_skipped",
            "chat/ui.response_received",
            "chat/ui.visible_render",
            "chat/ui.latency_summary",
            "requestAnimationFrame",
        ):
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, bundle)

