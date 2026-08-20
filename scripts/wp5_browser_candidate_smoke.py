#!/usr/bin/env python3
"""Non-destructive browser proof for an already-running isolated WP5 candidate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    chrome = os.environ.get("BROWSER_UI_CHROME") or "/usr/bin/google-chrome"
    checks: list[dict[str, object]] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=chrome, headless=True)
        context = browser.new_context(viewport={"width": 390, "height": 844}, reduced_motion="reduce")
        page = context.new_page()
        page.goto(args.base_url, wait_until="networkidle", timeout=30_000)
        checks.append({"name": "mobile_ui_loaded", "passed": page.locator("body").count() == 1})
        page.get_by_role("button", name="Advanced").click(timeout=10_000)
        pack_group = page.locator("summary", has_text="Packs")
        if pack_group.get_attribute("aria-expanded") != "true":
            pack_group.click(timeout=10_000)
        page.locator(".admin-nav-group-buttons").get_by_role("button", name="Packs", exact=True).click(timeout=10_000)
        page.get_by_role("heading", name="Find, fetch, or create a skill").wait_for(timeout=10_000)
        body = page.locator("body").inner_text(timeout=5_000)
        checks.append({"name": "normal_user_pack_flow", "passed": all(text in body for text in ("Preview quarantine fetch", "Preview assistant-created draft", "Presence visualizer"))})
        checks.append({"name": "reduced_motion_context", "passed": page.evaluate("matchMedia('(prefers-reduced-motion: reduce)').matches") is True})
        checks.append({"name": "no_pack_supplied_script_surface", "passed": page.locator("iframe, object, embed").count() == 0})
        context.close()
        browser.close()
    report = {"contract": "personal-agent.wp5-browser-candidate.v1", "checks": checks, "summary": {"passed": sum(bool(row["passed"]) for row in checks), "failed": sum(not bool(row["passed"]) for row in checks)}}
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
