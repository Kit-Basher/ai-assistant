from __future__ import annotations

from scripts.language_consistency import check_english_language_consistency


def test_plain_english_response_passes() -> None:
    result = check_english_language_consistency(
        "Explain why the sky is blue in two sentences.",
        "The sky looks blue because air scatters shorter blue wavelengths more than red wavelengths.",
    )

    assert result.ok


def test_english_chinese_sentence_fails() -> None:
    result = check_english_language_consistency(
        "Explain why the sky is blue in two sentences.",
        "The sky appears blue because the Earth's atmosphere散射太阳光中的短波蓝光 more strongly.",
    )

    assert not result.ok
    assert "language_consistency_failed" in result.reason
    assert "散射" in result.excerpt


def test_code_block_or_url_symbols_do_not_fail() -> None:
    result = check_english_language_consistency(
        "Explain this code.",
        "Use this URL: https://example.com/%E4%BD%A0 and this code:\n```python\nprint('你好你好你好')\n```",
    )

    assert result.ok


def test_explicit_answer_in_chinese_allows_non_english() -> None:
    result = check_english_language_consistency(
        "Answer in Chinese: why is the sky blue?",
        "天空呈现蓝色是因为大气会更强地散射短波蓝光。",
    )

    assert result.ok
