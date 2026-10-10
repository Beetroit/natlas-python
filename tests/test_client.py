"""Unit tests for the synchronous and asynchronous Natlas client."""

import httpx

from natlas.client import AsyncNatlas, Natlas


def test_client_initialization_defaults() -> None:
    """Verify default base URL and header initialization."""
    client = Natlas()
    assert client.chat._base_url == "http://127.0.0.1:8000/v1"
    assert client._http_client.headers["User-Agent"] == "natlas-python/0.1.0"
    client.close()


def test_client_custom_base_url_and_key() -> None:
    """Verify custom base URL and bearer token header."""
    client = Natlas(base_url="https://api.natlas.ng/v1", api_key="secret-token")
    assert client.chat._base_url == "https://api.natlas.ng/v1"
    assert client._http_client.headers["Authorization"] == "Bearer secret-token"
    client.close()


def test_client_context_manager() -> None:
    """Verify context manager closes HTTP client properly."""
    with Natlas() as client:
        assert isinstance(client._http_client, httpx.Client)
    assert client._http_client.is_closed


def test_async_client_initialization() -> None:
    """Verify async client headers and attributes."""
    async_client = AsyncNatlas(base_url="http://localhost:9000/v1")
    assert async_client.chat._base_url == "http://localhost:9000/v1"
