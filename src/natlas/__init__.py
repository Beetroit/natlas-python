"""N-ATLAS: Sovereign AI Toolkit & SDK for Nigerian Languages and ASR."""

from natlas.client import AsyncNatlas, Natlas
from natlas.config import SupportedLanguage
from natlas.exceptions import (
    APIConnectionError,
    APIStatusError,
    AudioProcessingError,
    DialectGuardError,
    ModelNotFoundError,
    NatlasError,
)

__version__ = "0.1.0"

__all__ = [
    "APIConnectionError",
    "APIStatusError",
    "AsyncNatlas",
    "AudioProcessingError",
    "DialectGuardError",
    "ModelNotFoundError",
    "Natlas",
    "NatlasError",
    "SupportedLanguage",
    "__version__",
]
