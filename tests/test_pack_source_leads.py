from __future__ import annotations

import unittest

from agent.packs.source_leads import build_source_leads_from_safe_search


class TestPackSourceLeads(unittest.TestCase):
    def test_normalizes_safe_search_results_into_untrusted_leads(self) -> None:
        result = build_source_leads_from_safe_search(
            {
                "ok": True,
                "results": [
                    {
                        "title": "Example skill",
                        "url": "https://example.com/skills/demo",
                        "snippet": "A metadata snippet.",
                        "engine": "test",
                    }
                ],
            }
        )

        self.assertTrue(result.ok)
        self.assertEqual(1, len(result.leads))
        lead = result.leads[0]
        self.assertEqual("Example skill", lead.title)
        self.assertEqual("https://example.com/skills/demo", lead.url)
        self.assertTrue(lead.untrusted)
        self.assertTrue(lead.requires_source_approval)
        self.assertTrue(lead.blocked_from_fetch)

    def test_github_repo_result_is_labelled_but_untrusted(self) -> None:
        result = build_source_leads_from_safe_search(
            {"ok": True, "results": [{"title": "Repo", "url": "https://github.com/acme/skill-pack"}]}
        )

        self.assertEqual("github_repo", result.leads[0].suspected_source_kind)
        self.assertTrue(result.leads[0].untrusted)
        self.assertTrue(result.leads[0].blocked_from_fetch)

    def test_archive_url_is_labelled_but_untrusted(self) -> None:
        result = build_source_leads_from_safe_search(
            {"ok": True, "results": [{"title": "Archive", "url": "https://example.com/pack.zip"}]}
        )

        self.assertEqual("generic_archive_url", result.leads[0].suspected_source_kind)
        self.assertTrue(result.leads[0].requires_source_approval)

    def test_non_http_urls_ignored_and_duplicates_deduped(self) -> None:
        result = build_source_leads_from_safe_search(
            {
                "ok": True,
                "results": [
                    {"title": "Mail", "url": "mailto:test@example.com"},
                    {"title": "One", "url": "https://example.com/pack?b=2&a=1"},
                    {"title": "Duplicate", "url": "https://example.com/pack?a=1&b=2"},
                ],
            }
        )

        self.assertEqual(1, len(result.leads))
        self.assertEqual("One", result.leads[0].title)

    def test_token_query_params_redacted(self) -> None:
        result = build_source_leads_from_safe_search(
            {
                "ok": True,
                "results": [
                    {
                        "title": "Secret URL",
                        "url": "https://example.com/pack.zip?token=secret&api_key=abc&ok=yes",
                    }
                ],
            }
        )

        url = result.leads[0].url
        self.assertIn("token=%5BREDACTED%5D", url)
        self.assertIn("api_key=%5BREDACTED%5D", url)
        self.assertIn("ok=yes", url)
        self.assertNotIn("secret", url)

    def test_no_network_or_page_fetch_happens(self) -> None:
        class ExplodingResult(dict):
            def read(self):  # pragma: no cover - should never be called
                raise AssertionError("page content should not be read")

        result = build_source_leads_from_safe_search(
            {"ok": True, "results": [ExplodingResult(title="Lead", url="https://example.com/pack")]}
        )

        self.assertEqual(1, len(result.leads))


if __name__ == "__main__":
    unittest.main()

