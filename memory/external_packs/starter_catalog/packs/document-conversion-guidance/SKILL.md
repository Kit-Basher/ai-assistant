---
name: PDF and Document Conversion Guidance
description: Plan document and PDF conversion safely before any file operation.
---

# PDF and Document Conversion Guidance

Use this pack when the user asks to convert, export, or turn content into a PDF or another document format.

Rules:
- Stay text-only. Do not convert files, execute commands, install packages, or write output files from this pack.
- Confirm source material, target format, page size, fonts, images, links, metadata, and accessibility needs.
- Treat file mutation as a separate preview-first action outside this pack.
- Never promise formatting fidelity until a conversion path and verification step are chosen.

Workflow:
1. Identify the source format and target format.
2. List formatting constraints that matter: margins, headings, tables, images, links, and page breaks.
3. Recommend a safe conversion path already available to the user, or describe what capability is missing.
4. Define verification checks: open result, check page count, links, embedded images, and text selection.
5. For batch conversion, require a dry-run inventory before any changes.
