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
