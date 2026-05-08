---
name: File Organization Workflow
description: Plan safe file organization with inventory and dry-run review before changes.
---

# File Organization Workflow

Use this pack when the user wants to organize, rename, group, deduplicate, or clean up files.

Rules:
- Stay text-only. Do not move, delete, rename, copy, or create files from this pack.
- Require an inventory before proposing mutations.
- Require a dry-run mapping before any file changes.
- Preserve originals unless the user explicitly approves a mutation through the normal action gate.

Workflow:
1. Identify scope, file types, naming preferences, and protected paths.
2. Propose categories and naming conventions.
3. Create a dry-run plan with source and destination paths.
4. Highlight duplicates, conflicts, and ambiguous files.
5. Route actual moves/deletes/renames through preview-first confirmation.
