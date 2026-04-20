from __future__ import annotations

import unittest

from agent.resource_insights import classify_memory_pressure, summarize_resource_report


GIB = 1024**3


def _proc(pid: int, name: str, *, cpu_ticks: int = 0, rss_gib: float = 0.0) -> dict[str, object]:
    return {
        "pid": pid,
        "name": name,
        "cpu_ticks": cpu_ticks,
        "rss_bytes": int(rss_gib * GIB),
    }


def _payload(
    *,
    used_gib: float,
    total_gib: float,
    available_gib: float,
    top_cpu: list[dict[str, object]] | None = None,
    top_rss: list[dict[str, object]] | None = None,
    cpu_count: int = 8,
    load_1m: float = 1.0,
    text: str = "",
) -> dict[str, object]:
    total = int(total_gib * GIB)
    used = int(used_gib * GIB)
    available = int(available_gib * GIB)
    return {
        "loads": {"1m": load_1m, "5m": load_1m * 0.8, "15m": load_1m * 0.6},
        "memory": {
            "total": total,
            "used": used,
            "available": available,
            "free": available,
            "used_pct": round((used / float(total)) * 100.0, 1) if total else 0.0,
        },
        "swap": {"total": 0, "used": 0},
        "cpu_count": cpu_count,
        "cpu_samples": top_cpu or [],
        "rss_samples": top_rss or [],
        "text": text,
    }


def _assert_cause_first(summary: str) -> None:
    cause_idx = summary.find("Likely cause:")
    normal_idx = summary.find("Normality:")
    evidence_idx = summary.find("Evidence:")
    action_idx = summary.find("Safe next action:")
    assert cause_idx != -1
    assert normal_idx != -1
    assert evidence_idx != -1
    assert action_idx != -1
    assert cause_idx < normal_idx < evidence_idx < action_idx


class TestResourceInsights(unittest.TestCase):
    def test_hidden_vm_process_is_classified_as_likely_cause(self) -> None:
        payload = _payload(
            used_gib=58,
            total_gib=64,
            available_gib=6,
            top_cpu=[_proc(101, "qemu-system-x86_64", cpu_ticks=900, rss_gib=0.0)],
            top_rss=[_proc(101, "qemu-system-x86_64", cpu_ticks=900, rss_gib=18.0)],
            text="my RAM is high",
        )

        result = summarize_resource_report(payload, text="my RAM is high")
        summary = str(result["summary"])

        _assert_cause_first(summary)
        self.assertIn("virtual machine/emulator", summary)
        self.assertIn("looks like the biggest sampled consumer", summary)
        self.assertIn("shut it down or pause it", str(result["safe_action"]))

    def test_ollama_is_classified_as_llm_runtime_and_cpu_dominant(self) -> None:
        payload = _payload(
            used_gib=56,
            total_gib=64,
            available_gib=8,
            top_cpu=[_proc(202, "ollama", cpu_ticks=2200, rss_gib=0.0), _proc(203, "helper", cpu_ticks=600, rss_gib=0.0)],
            top_rss=[_proc(202, "ollama", cpu_ticks=2200, rss_gib=12.0)],
            load_1m=12.0,
            text="nothing open but cpu is high",
        )

        result = summarize_resource_report(payload, text="nothing open but cpu is high")
        summary = str(result["summary"])

        _assert_cause_first(summary)
        self.assertIn("LLM runtime or model server", summary)
        self.assertIn("main CPU contributor", summary)
        self.assertIn("Even if nothing is open", summary)

    def test_ollama_multi_process_explains_server_and_worker_roles(self) -> None:
        payload = _payload(
            used_gib=40,
            total_gib=64,
            available_gib=24,
            top_cpu=[
                _proc(202, "ollama", cpu_ticks=2200, rss_gib=0.0),
                _proc(203, "ollama runner", cpu_ticks=700, rss_gib=0.0),
            ],
            top_rss=[
                _proc(202, "ollama", cpu_ticks=2200, rss_gib=12.0),
                _proc(203, "ollama runner", cpu_ticks=700, rss_gib=6.0),
            ],
            text="why are there 2 ollama instances?",
        )

        result = summarize_resource_report(payload, text="why are there 2 ollama instances?")
        summary = str(result["summary"])

        _assert_cause_first(summary)
        self.assertIn("multiple Ollama processes", summary)
        self.assertRegex(summary.lower(), r"(server|worker|helper)")

    def test_browser_session_is_high_memory_but_still_has_available_ram(self) -> None:
        payload = _payload(
            used_gib=75,
            total_gib=100,
            available_gib=25,
            top_cpu=[_proc(301, "chrome", cpu_ticks=500, rss_gib=0.0), _proc(302, "chrome", cpu_ticks=420, rss_gib=0.0)],
            top_rss=[
                _proc(301, "chrome", cpu_ticks=500, rss_gib=22.0),
                _proc(302, "chrome", cpu_ticks=420, rss_gib=18.0),
                _proc(303, "chrome", cpu_ticks=180, rss_gib=10.0),
            ],
            text="my memory is high",
        )

        result = summarize_resource_report(payload, text="my memory is high")
        summary = str(result["summary"])

        _assert_cause_first(summary)
        self.assertIn("browser session with multiple tabs/processes", summary)
        self.assertIn("plenty of RAM is available", summary)

    def test_high_available_memory_is_normal_not_pressure(self) -> None:
        payload = _payload(
            used_gib=12.8,
            total_gib=62.7,
            available_gib=49.9,
            top_cpu=[],
            top_rss=[],
            text="memory seems high",
        )

        result = summarize_resource_report(payload, text="memory seems high")
        summary = str(result["summary"])

        _assert_cause_first(summary)
        self.assertIn("plenty of available memory", summary)
        self.assertIn("not under pressure", summary)
        self.assertNotIn("getting tight", summary)
        self.assertNotIn("concerning because available memory", summary)

    def test_high_available_memory_stays_normal_even_with_swap_used(self) -> None:
        payload = _payload(
            used_gib=12.8,
            total_gib=62.7,
            available_gib=49.9,
            top_cpu=[],
            top_rss=[],
            text="memory seems high",
        )
        payload["swap"] = {"total": 8 * GIB, "used": 4 * GIB}

        pressure = classify_memory_pressure(
            total_bytes=int(62.7 * GIB),
            available_bytes=int(49.9 * GIB),
            swap_used_bytes=int(4 * GIB),
        )
        result = summarize_resource_report(payload, text="memory seems high")
        summary = str(result["summary"])

        self.assertFalse(bool(pressure["is_pressure"]))
        self.assertEqual("normal", pressure["state"])
        self.assertIn("not under pressure", summary)
        self.assertNotIn("getting tight", summary)

    def test_high_cpu_single_process_is_called_out_as_dominant_workload(self) -> None:
        payload = _payload(
            used_gib=34,
            total_gib=64,
            available_gib=30,
            top_cpu=[_proc(401, "python", cpu_ticks=4000, rss_gib=0.0), _proc(402, "sleep", cpu_ticks=250, rss_gib=0.0)],
            top_rss=[_proc(401, "python", cpu_ticks=4000, rss_gib=1.0)],
            load_1m=5.0,
            text="cpu lag",
        )

        result = summarize_resource_report(payload, text="cpu lag")
        summary = str(result["summary"])

        _assert_cause_first(summary)
        self.assertIn("long-running background job", summary)
        self.assertIn("main CPU contributor", summary)
        self.assertIn("doing most of the work", str(result["normality"]))

    def test_no_clear_culprit_case_is_explicit(self) -> None:
        payload = _payload(
            used_gib=18,
            total_gib=64,
            available_gib=46,
            top_cpu=[],
            top_rss=[],
            text="memory complaint",
        )

        result = summarize_resource_report(payload, text="memory complaint")
        summary = str(result["summary"])

        _assert_cause_first(summary)
        self.assertIn("no single process stands out", summary)
        self.assertIn("No urgent action is obvious", summary)


if __name__ == "__main__":
    unittest.main()
