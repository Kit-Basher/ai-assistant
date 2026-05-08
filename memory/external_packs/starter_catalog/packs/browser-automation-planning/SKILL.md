---
name: Browser Automation Planning Guidance
description: Plan browser automation safely without controlling a browser or running automation.
---

# Browser Automation Planning Guidance

Use this pack when the user wants browser automation but no managed browser adapter is approved.

Rules:
- Stay text-only. Do not launch a browser, execute scripts, install drivers, start MCP servers, or operate accounts.
- Explain that real browser control needs a managed adapter with explicit permissions.
- Avoid credential collection. Ask the user to handle sign-in directly.
- For websites with transactions, messaging, purchases, or account changes, require a separate confirmation plan.

Workflow:
1. Identify the user goal, target site, account sensitivity, and whether mutation is involved.
2. Draft a human-reviewable automation plan: pages, selectors, data inputs, waits, and failure checks.
3. Mark every mutating step as confirmation-required.
4. Suggest dry-run validation using screenshots or manual inspection.
5. Route actual execution to a managed adapter explanation, not this text pack.
