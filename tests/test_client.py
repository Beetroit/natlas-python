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


def test_client_chat_create_payload_forwards_language_and_tone() -> None:
    """Verify chat.create forwards language and tone_enforcement in HTTP payload."""
    import json

    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "natlas-test",
                "object": "chat.completion",
                "created": 123456789,
                "model": "N-ATLaS-8B-GGUF",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "No wahala, you fit start so.",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = Natlas()
    client._http_client = mock_client
    client.chat._client = mock_client

    response = client.chat.create(
        messages=[{"role": "user", "content": "How I fit start?"}],
        language="pidgin",
        tone_enforcement=True,
    )

    assert response.choices[0].message.content == "No wahala, you fit start so."
    assert len(captured_requests) == 1
    sent_payload = json.loads(captured_requests[0].content.decode("utf-8"))
    assert sent_payload["language"] == "pidgin"
    assert sent_payload["tone_enforcement"] is True
    client.close()
