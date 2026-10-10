"""Text normalization and orthography utilities for Nigerian languages."""

import unicodedata

from natlas.config import LanguageCode


def normalize_nigerian_text(text: str, language: LanguageCode | None = None) -> str:
    """Normalize text into canonical NFC Unicode representation.

    Ensures that composite diacritics (such as Yoruba tone marks and Igbo subdots)
    are properly composed for tokenizer consistency.
    """
    normalized = unicodedata.normalize("NFC", text)

    if language == "hausa":
        # Ensure standard Latin representation for hooked letters
        pass
    elif language == "yoruba":
        # Keep Yoruba tonal marks intact
        pass
    elif language == "igbo":
        # Keep Igbo dot-below vowels intact
        pass

    return normalized
