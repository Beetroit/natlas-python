"""Exception classes for the N-ATLAS SDK and runtime."""


class NatlasError(Exception):
    """Base exception for all Natlas SDK errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message: str = message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r})"


class APIConnectionError(NatlasError):
    """Raised when the client cannot connect to the N-ATLAS server."""


class APIStatusError(NatlasError):
    """Raised when the N-ATLAS server returns a non-2xx HTTP status code."""

    def __init__(self, message: str, status_code: int, response_body: str) -> None:
        super().__init__(message)
        self.status_code: int = status_code
        self.response_body: str = response_body

    def __repr__(self) -> str:
        return (
            f"APIStatusError(status_code={self.status_code}, "
            f"message={self.message!r}, response_body={self.response_body!r})"
        )


class AudioProcessingError(NatlasError):
    """Raised when audio conversion, resampling, or encoding fails."""


class ModelNotFoundError(NatlasError):
    """Raised when a requested model or GGUF file is missing."""


class DialectGuardError(NatlasError):
    """Raised when dialect or language constraint validation fails."""
