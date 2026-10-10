"""Audio model router for NCAIR1 Whisper checkpoints."""

from natlas.config import ASR_MODELS, LanguageCode
from natlas.exceptions import AudioProcessingError


def resolve_asr_model(language: LanguageCode | str) -> str:
    """Resolve the target NCAIR1 Hugging Face ASR model ID for a given language.

    Args:
        language: Language code or name ('yoruba', 'hausa', 'igbo', 'english', 'pidgin').

    Returns:
        The exact Hugging Face model repository identifier.
    """
    normalized = language.strip().lower()
    if normalized in ("pidgin", "nigerian english"):
        normalized = "english"

    if normalized in ASR_MODELS:
        return ASR_MODELS[normalized]

    raise AudioProcessingError(
        f"Unsupported language '{language}' for N-ATLAS ASR. "
        f"Supported languages: {', '.join(ASR_MODELS.keys())}"
    )
