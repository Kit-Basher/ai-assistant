from __future__ import annotations

import io
import logging
import unittest

from agent.security.redaction import RedactingFormatter, redact_text, redact_value


FAKE_TOKEN = "123456:TEST_TOKEN_REDACT_ME"


class TestSecretRedaction(unittest.TestCase):
    def test_telegram_api_url_preserves_method_and_redacts_token(self) -> None:
        redacted = redact_text(f"https://api.telegram.org/bot{FAKE_TOKEN}/getUpdates")
        self.assertEqual("https://api.telegram.org/bot<redacted>/getUpdates", redacted)
        self.assertNotIn(FAKE_TOKEN, redacted)

    def test_redacts_token_in_exception_text_and_repr_like_strings(self) -> None:
        text = redact_text(f"RuntimeError('POST http://api.telegram.org/bot{FAKE_TOKEN}/sendMessage failed')")
        self.assertIn("http://api.telegram.org/bot<redacted>/sendMessage", text)
        self.assertNotIn(FAKE_TOKEN, text)

    def test_nested_payload_redacts_known_secret_and_secret_keys(self) -> None:
        payload = {
            "ok": False,
            "error": f"token was {FAKE_TOKEN}",
            "nested": [{"telegram_bot_token": FAKE_TOKEN}, {"url": f"https://api.telegram.org/bot{FAKE_TOKEN}/getMe"}],
        }
        redacted = redact_value(payload, known_secrets=[FAKE_TOKEN])
        rendered = str(redacted)
        self.assertNotIn(FAKE_TOKEN, rendered)
        self.assertIn("getMe", rendered)

    def test_authorization_and_query_secret_redaction(self) -> None:
        text = redact_text("Authorization: Bearer abc.def token=secret-value api_key=abc123")
        self.assertIn("Authorization: Bearer <redacted>", text)
        self.assertIn("token=<redacted>", text)
        self.assertIn("api_key=<redacted>", text)

    def test_logging_formatter_redacts_httpx_style_request_line(self) -> None:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(RedactingFormatter("%(message)s"))
        logger = logging.getLogger("test.redaction.telegram")
        old_handlers = list(logger.handlers)
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        try:
            logger.info("HTTP Request: POST https://api.telegram.org/bot%s/getUpdates \"HTTP/1.1 200 OK\"", FAKE_TOKEN)
        finally:
            logger.handlers = old_handlers
            logger.propagate = True
        output = stream.getvalue()
        self.assertIn("https://api.telegram.org/bot<redacted>/getUpdates", output)
        self.assertNotIn(FAKE_TOKEN, output)

    def test_non_secret_url_is_not_destroyed(self) -> None:
        url = "https://example.com/docs/tokenization"
        self.assertEqual(url, redact_text(url))
