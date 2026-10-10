"""Dialect guards and anti-drift validators for Nigerian language generations."""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from natlas.config import LanguageCode
from natlas.dialects.prompts import get_system_prompt_for_language


class ChatMessage(BaseModel):
    """Represents a single message in a chat conversation."""

    model_config = ConfigDict(frozen=True)

    role: str
    content: str


def apply_dialect_guard(
    messages: Sequence[ChatMessage | dict[str, str]],
    language: LanguageCode | None,
    tone_enforcement: bool = True,
) -> list[dict[str, str]]:
    """Inject language-tuned system prompts and tone constraints to prevent language drift.

    Args:
        messages: The sequence of chat messages.
        language: The target language code, or None if unspecified.
        tone_enforcement: If True, injects specific tone rules and guardrails.

    Returns:
        A list of sanitized message dictionaries ready for model consumption.
    """
    normalized_messages: list[dict[str, str]] = []
    has_system_message = False

    for msg in messages:
        if isinstance(msg, ChatMessage):
            role = msg.role
            content = msg.content
        else:
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", ""))

        if role == "system":
            has_system_message = True
            if language is not None and tone_enforcement:
                # Augment existing system message with dialect prompt
                tuned_prompt = get_system_prompt_for_language(language)
                content = f"{tuned_prompt}\n\nAdditional instructions:\n{content}"
        normalized_messages.append({"role": role, "content": content})

    if not has_system_message and language is not None and tone_enforcement:
        # Prepend the dedicated language system prompt
        tuned_prompt = get_system_prompt_for_language(language)
        normalized_messages.insert(0, {"role": "system", "content": tuned_prompt})

    return normalized_messages
