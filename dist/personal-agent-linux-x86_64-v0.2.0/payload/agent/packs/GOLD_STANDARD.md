# Pack Gold Standard

This document is the canonical internal authoring and normalization target for external skill packs.
It describes the shape the system should prefer when reducing safe external content into a single
portable internal pack.

## What a skill pack is

A skill pack is a focused bundle of safe text, prompts, reference material, and optional static assets
that helps the agent perform a specific task more reliably.

## Core principle

The pack must remain useful as safe text alone. If the imported content cannot stand on its own as
readable guidance, it should be reduced, synthesized, or blocked rather than executed.

## Required canonical structure

Normalized packs should be written in a canonical internal directory with these top-level entries:

- `PACK.md`
- `manifest.json`
- `SKILL.md`
- `prompts/`
- `assets/`
- `metadata/`

## SKILL.md required sections

- Purpose
- When to Use
- Inputs
- Behavior
- Constraints
- Response Style
- Example Prompts

## Quality rules

- Specific: describe the task and the expected behavior concretely.
- Actionable: tell the user what the pack is for and how to use it.
- Honest: preserve uncertainty, missing context, and dropped content.
- Safe: never require execution of imported code or trust beyond the audited text.
- Minimal: keep the canonical skill focused and avoid noisy duplication.

## Import outcome classes

- `normalized_safe_text`
- `partial_safe_import`
- `blocked_install`

## Anti-patterns

- Copying executable instructions into the canonical skill as if they were safe.
- Hiding dropped files or blocked categories from the review trail.
- Treating a mixed or plugin-like bundle as trusted runnable content.
- Replacing a clear task description with catalog noise or vague marketing language.

## Minimal example skill

```markdown
# Repo Helper

## Purpose
Help explain repository files safely.

## When to Use
Use when the user asks what a repository contains or how to summarize its docs.

## Inputs
Repository docs, safe text files, and the user’s question.

## Behavior
Read the safe material, stay grounded, and explain only what is present.

## Constraints
Do not execute code. Do not invent details. Ignore dropped files.

## Response Style
Be concise, direct, and clear about uncertainty.

## Example Prompts
- What does this repo do?
- Summarize the setup steps.
```
