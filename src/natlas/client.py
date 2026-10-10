"""Main entrypoint client classes for the N-ATLAS SDK."""

import os
from types import TracebackType
from typing import Self

import httpx

from natlas.api.audio import AsyncAudio, Audio
from natlas.api.chat import AsyncChat, Chat
from natlas.config import DEFAULT_BASE_URL


class Natlas:
    """Synchronous N-ATLAS SDK Client."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Initialize the synchronous N-ATLAS client.

        Args:
            base_url: The base URL of the N-ATLAS API server (defaults to local or NATLAS_BASE_URL).
            api_key: Optional API key for hosted cloud endpoints (or NATLAS_API_KEY).
            timeout: HTTP request timeout in seconds.
        """
        resolved_base_url = (
            base_url or os.environ.get("NATLAS_BASE_URL") or DEFAULT_BASE_URL
        )
        resolved_api_key = api_key or os.environ.get("NATLAS_API_KEY")

        headers: dict[str, str] = {
            "User-Agent": "natlas-python/0.1.0",
        }
        if resolved_api_key:
            headers["Authorization"] = f"Bearer {resolved_api_key}"

        self._http_client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            headers=headers,
        )

        self.chat = Chat(self._http_client, resolved_base_url)
        self.audio = Audio(self._http_client, resolved_base_url)

    def close(self) -> None:
        """Close the underlying HTTP client session."""
        self._http_client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


class AsyncNatlas:
    """Asynchronous N-ATLAS SDK Client."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Initialize the asynchronous N-ATLAS client."""
        resolved_base_url = (
            base_url or os.environ.get("NATLAS_BASE_URL") or DEFAULT_BASE_URL
        )
        resolved_api_key = api_key or os.environ.get("NATLAS_API_KEY")

        headers: dict[str, str] = {
            "User-Agent": "natlas-python/0.1.0",
        }
        if resolved_api_key:
            headers["Authorization"] = f"Bearer {resolved_api_key}"

        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers=headers,
        )

        self.chat = AsyncChat(self._http_client, resolved_base_url)
        self.audio = AsyncAudio(self._http_client, resolved_base_url)

    async def aclose(self) -> None:
        """Close the underlying asynchronous HTTP client session."""
        await self._http_client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()
