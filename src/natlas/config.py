"""Configuration constants and model defaults for N-ATLAS."""

from enum import StrEnum
from typing import Final, Literal

DEFAULT_BASE_URL: Final[str] = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL: Final[str] = "NCAIR1/N-ATLaS"

# Recommended generation parameters derived from N-ATLaS model evaluation card
DEFAULT_TEMPERATURE: Final[float] = 0.1
DEFAULT_REPETITION_PENALTY: Final[float] = 1.12
DEFAULT_MAX_TOKENS: Final[int] = 1024
DEFAULT_TOP_P: Final[float] = 0.9

# Official Hugging Face repositories for ASR
ASR_MODELS: Final[dict[str, str]] = {
    "yoruba": "NCAIR1/Yoruba-ASR",
    "hausa": "NCAIR1/Hausa-ASR",
    "igbo": "NCAIR1/Igbo-ASR",
    "english": "NCAIR1/NigerianAccentedEnglish",
}


class SupportedLanguage(StrEnum):
    """Officially supported languages for N-ATLAS."""

    YORUBA = "yoruba"
    HAUSA = "hausa"
    IGBO = "igbo"
    ENGLISH = "english"
    PIDGIN = "pidgin"


LanguageCode = Literal["yoruba", "hausa", "igbo", "english", "pidgin"]
