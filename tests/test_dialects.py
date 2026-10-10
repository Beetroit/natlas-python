"""Unit tests for dialect guards, system prompt mapping, and normalizers."""

from natlas.dialects.guards import apply_dialect_guard
from natlas.dialects.normalizer import normalize_nigerian_text
from natlas.dialects.prompts import (
    get_system_prompt_for_language,
)


def test_system_prompt_mapping() -> None:
    """Verify correct system prompts are retrieved for supported languages."""
    assert "Yorùbá" in get_system_prompt_for_language("yoruba")
    assert "Hausa" in get_system_prompt_for_language("hausa")
    assert "Igbo" in get_system_prompt_for_language("igbo")


def test_apply_dialect_guard_prepends_system_message() -> None:
    """Verify dialect guard prepends system message when none is present."""
    messages = [{"role": "user", "content": "Bawo ni"}]
    guarded = apply_dialect_guard(messages, language="yoruba", tone_enforcement=True)

    assert len(guarded) == 2
    assert guarded[0]["role"] == "system"
    assert "Yorùbá" in guarded[0]["content"]
    assert guarded[1]["content"] == "Bawo ni"


def test_apply_dialect_guard_augments_existing_system_message() -> None:
    """Verify dialect guard augments existing system prompt without overwriting user instructions."""
    messages = [
        {"role": "system", "content": "You are a customer support agent."},
        {"role": "user", "content": "Sannu"},
    ]
    guarded = apply_dialect_guard(messages, language="hausa", tone_enforcement=True)

    assert len(guarded) == 2
    assert guarded[0]["role"] == "system"
    assert "Hausa" in guarded[0]["content"]
    assert "You are a customer support agent." in guarded[0]["content"]


def test_normalize_nigerian_text_nfc() -> None:
    """Verify Unicode NFC normalization."""
    # Decomposed vs composed
    decomposed = "ọ"  # o + dot below
    normalized = normalize_nigerian_text(decomposed, language="yoruba")
    assert normalized == "ọ"


def test_system_prompt_pidgin_anti_drift_boundaries() -> None:
    """Verify Nigerian Pidgin prompt includes strict negative boundaries and lexical anchors."""
    pidgin_prompt = get_system_prompt_for_language("pidgin")
    assert "Nigerian Pidgin English" in pidgin_prompt
    assert "DO NOT speak Yoruba, Igbo, or Hausa" in pidgin_prompt
    assert "No wahala" in pidgin_prompt
    assert "dey" in pidgin_prompt
    assert "Na so!" in pidgin_prompt


def test_apply_dialect_guard_pidgin_guards_against_yoruba_drift() -> None:
    """Verify dialect guard injects anti-drift negative constraints when language=pidgin."""
    messages = [{"role": "user", "content": "How I fit start tech career for Lagos?"}]
    guarded = apply_dialect_guard(messages, language="pidgin", tone_enforcement=True)

    assert len(guarded) == 2
    assert guarded[0]["role"] == "system"
    system_text = guarded[0]["content"]
    assert "DO NOT speak Yoruba, Igbo, or Hausa" in system_text
    assert "No wahala" in system_text
    assert guarded[1]["content"] == "How I fit start tech career for Lagos?"


def test_assert_pidgin_content_avoids_yoruba_markers() -> None:
    """Assert a validator function detects and rejects Yoruba drift in Pidgin responses."""
    yoruba_drift_sample = "O dara, baba, o fẹ́ bẹ̀rẹ̀ ìrìn àjò yìí nínú imọ-ẹrọ, eh? Ọ̀nà tó dára ni láti bẹ̀rẹ̀."
    valid_pidgin_sample = "You fit start tech career for Lagos, you fit do am well. Na so, you fit take online course."

    yoruba_forbidden_markers = [
        "O dara",
        "baba, o fẹ́",
        "ìrìn àjò",
        "ẹ̀kọ́",
        "nínú",
        "ọmọlúwàbí",
        "bẹ̀rẹ̀",
    ]

    def contains_yoruba_drift(text: str) -> bool:
        return any(
            marker.lower() in text.lower() for marker in yoruba_forbidden_markers
        )

    assert contains_yoruba_drift(yoruba_drift_sample) is True
    assert contains_yoruba_drift(valid_pidgin_sample) is False


def test_live_server_pidgin_anti_drift_integration() -> None:
    """Integration test: verify SDK with pidgin dialect guard prevents Yoruba drift against model."""
    import httpx
    import pytest

    from natlas.client import Natlas

    base_url = "http://127.0.0.1:8000"
    try:
        health_resp = httpx.get(f"{base_url}/health", timeout=2.0)
        if health_resp.status_code != 200:
            pytest.skip("Local N-ATLAS server not running")
    except (httpx.RequestError, OSError):
        pytest.skip("Local N-ATLAS server not reachable")

    with Natlas(base_url=f"{base_url}/v1") as client:
        response = client.chat.create(
            messages=[
                {"role": "user", "content": "How I fit start tech career for Lagos?"}
            ],
            language="pidgin",
            tone_enforcement=True,
            temperature=0.3,
            repetition_penalty=1.18,
            max_tokens=250,
        )
        content = response.choices[0].message.content

        # 1. Assert Yoruba phrases are strictly absent
        yoruba_markers = [
            "O dara",
            "ìrìn àjò",
            "ẹ̀kọ́",
            "bẹ̀rẹ̀ pẹ̀lú",
            "nínú imọ-ẹrọ",
            "ọmọlúwàbí",
        ]
        for marker in yoruba_markers:
            assert marker.lower() not in content.lower(), (
                f"Detected Yoruba drift '{marker}' in Pidgin response: {content}"
            )

        # 2. Assert Igbo/Yoruba loan words from code-switching loops are absent
        code_switch_markers = [
            "akwụkwọ",
            "onwe-ọrụ",
            "ọgbọn",
            "gba imọ",
            "jẹ onwe",
        ]
        for marker in code_switch_markers:
            assert marker.lower() not in content.lower(), (
                f"Detected code-switching loan word '{marker}' in Pidgin response: {content}"
            )

        # 3. Assert no degenerate 4-gram repetition loops
        words = content.lower().split()
        if len(words) >= 8:
            four_grams = [" ".join(words[i : i + 4]) for i in range(len(words) - 3)]
            counts = {fg: four_grams.count(fg) for fg in set(four_grams)}
            for fg, count in counts.items():
                assert count <= 2, (
                    f"Detected repetitive loop '{fg}' repeated {count} times in: {content}"
                )

        # 4. Assert authentic Pidgin tokens are present
        pidgin_markers = ["fit", "dey", "na", "so", "wey", "am", "wahala", "wetin"]
        assert any(m in content.lower().split() for m in pidgin_markers), (
            f"Response did not contain common Pidgin lexical tokens: {content}"
        )
