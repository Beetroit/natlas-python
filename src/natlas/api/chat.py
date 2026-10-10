"""Chat completions client namespace (synchronous and asynchronous)."""

import json
from collections.abc import AsyncIterator, Iterator, Sequence

import httpx

from natlas.api.types import (
    ChatCompletionChunk,
    ChatCompletionResponse,
)
from natlas.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    LanguageCode,
)
from natlas.dialects.guards import ChatMessage, apply_dialect_guard
from natlas.exceptions import APIConnectionError, APIStatusError


class Chat:
    """Synchronous chat completions resource."""

    def __init__(self, http_client: httpx.Client, base_url: str) -> None:
        self._client: httpx.Client = http_client
        self._base_url: str = base_url.rstrip("/")

    def create(
        self,
        messages: Sequence[ChatMessage | dict[str, str]],
        *,
        model: str = DEFAULT_MODEL,
        language: LanguageCode | None = None,
        tone_enforcement: bool = True,
        temperature: float = DEFAULT_TEMPERATURE,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        top_p: float = DEFAULT_TOP_P,
    ) -> ChatCompletionResponse:
        """Create a chat completion response.

        Args:
            messages: Conversation messages.
            model: Model name/identifier.
            language: Target Nigerian language for prompt tuning.
            tone_enforcement: Whether to enforce diacritics and dialect constraints.
            temperature: Sampling temperature (default 0.1 for N-ATLAS).
            repetition_penalty: Repetition penalty (default 1.12 for N-ATLAS).
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling probability.

        Returns:
            ChatCompletionResponse object.
        """
        guarded_messages = apply_dialect_guard(
            messages, language=language, tone_enforcement=tone_enforcement
        )
        payload = {
            "model": model,
            "messages": guarded_messages,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False,
        }

        try:
            response = self._client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
            )
        except httpx.RequestError as exc:
            raise APIConnectionError(
                f"Could not connect to N-ATLAS server: {exc}"
            ) from exc

        if response.status_code != 200:
            raise APIStatusError(
                message=f"Request failed with status {response.status_code}",
                status_code=response.status_code,
                response_body=response.text,
            )

        return ChatCompletionResponse.model_validate(response.json())

    def stream(
        self,
        messages: Sequence[ChatMessage | dict[str, str]],
        *,
        model: str = DEFAULT_MODEL,
        language: LanguageCode | None = None,
        tone_enforcement: bool = True,
        temperature: float = DEFAULT_TEMPERATURE,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        top_p: float = DEFAULT_TOP_P,
    ) -> Iterator[ChatCompletionChunk]:
        """Stream chat completion chunks in real time via Server-Sent Events (SSE)."""
        guarded_messages = apply_dialect_guard(
            messages, language=language, tone_enforcement=tone_enforcement
        )
        payload = {
            "model": model,
            "messages": guarded_messages,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": True,
        }

        try:
            with self._client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json=payload,
            ) as response:
                if response.status_code != 200:
                    raise APIStatusError(
                        message=f"Streaming request failed with status {response.status_code}",
                        status_code=response.status_code,
                        response_body=response.read().decode("utf-8", errors="replace"),
                    )

                for line in response.iter_lines():
                    cleaned = line.strip()
                    if not cleaned or not cleaned.startswith("data:"):
                        continue
                    data_str = cleaned[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break
                    chunk_dict = json.loads(data_str)
                    yield ChatCompletionChunk.model_validate(chunk_dict)
        except httpx.RequestError as exc:
            raise APIConnectionError(f"Connection lost while streaming: {exc}") from exc


class AsyncChat:
    """Asynchronous chat completions resource."""

    def __init__(self, http_client: httpx.AsyncClient, base_url: str) -> None:
        self._client: httpx.AsyncClient = http_client
        self._base_url: str = base_url.rstrip("/")

    async def create(
        self,
        messages: Sequence[ChatMessage | dict[str, str]],
        *,
        model: str = DEFAULT_MODEL,
        language: LanguageCode | None = None,
        tone_enforcement: bool = True,
        temperature: float = DEFAULT_TEMPERATURE,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        top_p: float = DEFAULT_TOP_P,
    ) -> ChatCompletionResponse:
        """Create an asynchronous chat completion response."""
        guarded_messages = apply_dialect_guard(
            messages, language=language, tone_enforcement=tone_enforcement
        )
        payload = {
            "model": model,
            "messages": guarded_messages,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False,
        }

        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
            )
        except httpx.RequestError as exc:
            raise APIConnectionError(
                f"Could not connect to N-ATLAS server: {exc}"
            ) from exc

        if response.status_code != 200:
            raise APIStatusError(
                message=f"Request failed with status {response.status_code}",
                status_code=response.status_code,
                response_body=response.text,
            )

        return ChatCompletionResponse.model_validate(response.json())

    async def stream(
        self,
        messages: Sequence[ChatMessage | dict[str, str]],
        *,
        model: str = DEFAULT_MODEL,
        language: LanguageCode | None = None,
        tone_enforcement: bool = True,
        temperature: float = DEFAULT_TEMPERATURE,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        top_p: float = DEFAULT_TOP_P,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Stream chat completion chunks asynchronously via SSE."""
        guarded_messages = apply_dialect_guard(
            messages, language=language, tone_enforcement=tone_enforcement
        )
        payload = {
            "model": model,
            "messages": guarded_messages,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": True,
        }

        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json=payload,
            ) as response:
                if response.status_code != 200:
                    body_bytes = await response.aread()
                    raise APIStatusError(
                        message=f"Streaming request failed with status {response.status_code}",
                        status_code=response.status_code,
                        response_body=body_bytes.decode("utf-8", errors="replace"),
                    )

                async for line in response.aiter_lines():
                    cleaned = line.strip()
                    if not cleaned or not cleaned.startswith("data:"):
                        continue
                    data_str = cleaned[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break
                    chunk_dict = json.loads(data_str)
                    yield ChatCompletionChunk.model_validate(chunk_dict)
        except httpx.RequestError as exc:
            raise APIConnectionError(f"Connection lost while streaming: {exc}") from exc
