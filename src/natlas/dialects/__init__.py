"""Nigerian dialect and orthography package for N-ATLAS."""

from natlas.dialects.guards import ChatMessage, apply_dialect_guard
from natlas.dialects.normalizer import normalize_nigerian_text
from natlas.dialects.prompts import (
    BASE_SYSTEM_PROMPT,
    ENGLISH_SYSTEM_PROMPT,
    HAUSA_SYSTEM_PROMPT,
    IGBO_SYSTEM_PROMPT,
    PIDGIN_SYSTEM_PROMPT,
    YORUBA_SYSTEM_PROMPT,
    get_system_prompt_for_language,
)

__all__ = [
    "BASE_SYSTEM_PROMPT",
    "ENGLISH_SYSTEM_PROMPT",
    "HAUSA_SYSTEM_PROMPT",
    "IGBO_SYSTEM_PROMPT",
    "PIDGIN_SYSTEM_PROMPT",
    "YORUBA_SYSTEM_PROMPT",
    "ChatMessage",
    "apply_dialect_guard",
    "get_system_prompt_for_language",
    "normalize_nigerian_text",
]
