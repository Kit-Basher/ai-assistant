from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
import tempfile
import unittest

from agent.model_scout import ModelScout, ModelScoutSettings, ModelScoutStore


def _registry_document() -> dict[str, object]:
    return {
        "providers": {
            "ollama": {
                "provider_type": "openai_compat",
                "base_url": "http://127.0.0.1:11434",
                "chat_path": "/v1/chat/completions",
                "enabled": True,
                "local": True,
            }
        },
        "models": {
            "ollama:llama3": {
                "provider": "ollama",
                "model": "llama3",
                "capabilities": ["chat"],
                "enabled": True,
                "available": True,
                "quality_rank": 3,
                "cost_rank": 1,
                "pricing": {
                    "input_per_million_tokens": None,
                    "output_per_million_tokens": None,
                },
            }
        },
        "defaults": {
            "default_provider": "ollama",
            "default_model": "ollama:llama3",
            "routing_mode": "prefer_local_lowest_cost_capable",
            "allow_remote_fallback": True,
        },
    }


def _router_snapshot() -> dict[str, object]:
    return {
        "providers": [
            {
                "id": "ollama",
                "health": {
                    "status": "ok",
                    "successes": 3,
                    "failures": 0,
                },
            }
        ],
        "models": [
            {
                "id": "ollama:llama3",
                "health": {
                    "status": "ok",
                    "successes": 3,
                    "failures": 0,
                },
            }
        ],
    }


class TestModelScout(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _scout(
        self,
        *,
        payload: dict[str, object],
        now_holder: list[datetime],
        settings: ModelScoutSettings,
    ) -> ModelScout:
        store = ModelScoutStore(self.db_path)
        return ModelScout(
            settings,
            store=store,
            fetch_json=lambda _url: payload,
            now_fn=lambda: now_holder[0],
        )

    def test_scoring_and_sorting_are_deterministic(self) -> None:
        payload = {
            "data": [
                {
                    "id": "bartowski/Qwen2.5-7B-Instruct-GGUF",
                    "likes": 2400,
                    "downloads": 120000,
                    "license": "apache-2.0",
                },
                {
                    "id": "someone/Tiny-1.5B-GGUF",
                    "likes": 800,
                    "downloads": 48000,
                    "license": "mit",
                },
            ]
        }
        now_holder = [datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc)]
        settings = ModelScoutSettings(
            enabled=True,
            notify_delta=0.0,
            absolute_threshold=0.0,
            max_suggestions_per_notify=2,
            license_allowlist=frozenset({"apache-2.0", "mit", "bsd-3-clause"}),
            size_max_b=12.0,
        )

        scout = self._scout(payload=payload, now_holder=now_holder, settings=settings)
        try:
            result_a = scout.run(
                registry_document=_registry_document(),
                router_snapshot=_router_snapshot(),
                usage_stats_snapshot={},
                notify_sender=None,
            )
            now_holder[0] = now_holder[0] + timedelta(minutes=1)
            result_b = scout.run(
                registry_document=_registry_document(),
                router_snapshot=_router_snapshot(),
                usage_stats_snapshot={},
                notify_sender=None,
            )
        finally:
            scout.close()

        ids_a = [row["id"] for row in result_a["suggestions"]]
        ids_b = [row["id"] for row in result_b["suggestions"]]
        self.assertEqual(ids_a, ids_b)
        self.assertEqual(
            [
                "local:bartowski/qwen2.5-7b-instruct-gguf",
                "local:someone/tiny-1.5b-gguf",
            ],
            ids_a,
        )

    def test_notification_sender_called_only_when_threshold_is_met(self) -> None:
        payload = {
            "data": [
                {
                    "id": "bartowski/Qwen2.5-7B-Instruct-GGUF",
                    "likes": 2400,
                    "downloads": 120000,
                    "license": "apache-2.0",
                }
            ]
        }
        now_holder = [datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc)]
        notify_calls: list[list[dict[str, object]]] = []

        strict_scout = self._scout(
            payload=payload,
            now_holder=now_holder,
            settings=ModelScoutSettings(
                enabled=True,
                notify_delta=1_000.0,
                absolute_threshold=1_000.0,
                max_suggestions_per_notify=2,
                license_allowlist=frozenset({"apache-2.0", "mit", "bsd-3-clause"}),
                size_max_b=12.0,
            ),
        )
        try:
            strict_result = strict_scout.run(
                registry_document=_registry_document(),
                router_snapshot=_router_snapshot(),
                usage_stats_snapshot={},
                notify_sender=lambda _message, batch: notify_calls.append(batch),
            )
        finally:
            strict_scout.close()

        self.assertEqual(0, strict_result["notified"])
        self.assertEqual([], notify_calls)

        now_holder[0] = now_holder[0] + timedelta(minutes=1)
        permissive_scout = self._scout(
            payload=payload,
            now_holder=now_holder,
            settings=ModelScoutSettings(
                enabled=True,
                notify_delta=0.0,
                absolute_threshold=0.0,
                max_suggestions_per_notify=2,
                license_allowlist=frozenset({"apache-2.0", "mit", "bsd-3-clause"}),
                size_max_b=12.0,
            ),
        )
        try:
            permissive_result = permissive_scout.run(
                registry_document=_registry_document(),
                router_snapshot=_router_snapshot(),
                usage_stats_snapshot={},
                notify_sender=lambda _message, batch: notify_calls.append(batch),
            )
        finally:
            permissive_scout.close()

        self.assertEqual(1, permissive_result["notified"])
        self.assertEqual(1, len(notify_calls))

    def test_dedupe_and_cooldown_prevent_spam_until_window_passes(self) -> None:
        payload = {
            "data": [
                {
                    "id": "bartowski/Qwen2.5-7B-Instruct-GGUF",
                    "likes": 2400,
                    "downloads": 120000,
                    "license": "apache-2.0",
                }
            ]
        }
        now_holder = [datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc)]
        notifications: list[list[dict[str, object]]] = []

        scout = self._scout(
            payload=payload,
            now_holder=now_holder,
            settings=ModelScoutSettings(
                enabled=True,
                notify_delta=0.0,
                absolute_threshold=0.0,
                max_suggestions_per_notify=2,
                license_allowlist=frozenset({"apache-2.0", "mit", "bsd-3-clause"}),
                size_max_b=12.0,
                cooldown_days=7,
            ),
        )
        try:
            first = scout.run(
                registry_document=_registry_document(),
                router_snapshot=_router_snapshot(),
                usage_stats_snapshot={},
                notify_sender=lambda _message, batch: notifications.append(batch),
            )
            now_holder[0] = now_holder[0] + timedelta(days=1)
            second = scout.run(
                registry_document=_registry_document(),
                router_snapshot=_router_snapshot(),
                usage_stats_snapshot={},
                notify_sender=lambda _message, batch: notifications.append(batch),
            )
            now_holder[0] = now_holder[0] + timedelta(days=8)
            third = scout.run(
                registry_document=_registry_document(),
                router_snapshot=_router_snapshot(),
                usage_stats_snapshot={},
                notify_sender=lambda _message, batch: notifications.append(batch),
            )
        finally:
            scout.close()

        self.assertEqual(1, first["notified"])
        self.assertEqual(0, second["notified"])
        self.assertEqual(1, third["notified"])
        self.assertEqual(2, len(notifications))


if __name__ == "__main__":
    unittest.main()
