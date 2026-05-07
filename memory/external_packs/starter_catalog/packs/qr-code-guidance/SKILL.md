---
name: QR Code Creation Guidance
description: Plan safe QR code creation and validation without executing code or installing tools.
---

# QR Code Creation Guidance

Use this pack when the user wants a QR code or barcode and no native QR tool is available.

Rules:
- Stay text-only. Do not generate files, execute commands, install packages, start services, or call external QR APIs from this pack.
- Ask for or confirm the exact encoded content before any creation step.
- Warn about sensitive payloads, login tokens, payment addresses, and links that could be misleading.
- Recommend previewing the encoded content and scanning the final QR code with a trusted scanner before sharing it.
- If the user wants actual generation, explain that a separate approved tool or manual workflow is needed.

Workflow:
1. Identify the payload type: URL, plain text, contact info, Wi-Fi details, payment address, or other.
2. Normalize the payload in text and ask for confirmation if it is ambiguous.
3. Recommend output choices: PNG for sharing, SVG for print, and error correction level based on intended use.
4. Provide a manual or approved-tool checklist without running the tool.
5. Verify that the final QR code decodes to exactly the confirmed payload.
