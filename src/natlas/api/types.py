"""Strict Pydantic schemas for N-ATLAS client and server interactions."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ChatCompletionMessage(BaseModel):
    """Message object returned by completion."""

    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionChoice(BaseModel):
    """Choice object returned in chat completion response."""

    model_config = ConfigDict(frozen=True)

    index: int
    message: ChatCompletionMessage
    finish_reason: str | None = None


class UsageInfo(BaseModel):
    """Token usage metrics."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Standard OpenAI-compatible chat completion payload."""

    model_config = ConfigDict(frozen=True)

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo | None = None


class ChatCompletionChunkDelta(BaseModel):
    """Delta object within streaming chunk."""

    model_config = ConfigDict(frozen=True)

    role: str | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    """Choice object within streaming chunk."""

    model_config = ConfigDict(frozen=True)

    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""

    model_config = ConfigDict(frozen=True)

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class TranscriptionResponse(BaseModel):
    """Audio transcription response."""

    model_config = ConfigDict(frozen=True)

    text: str
    language: str
    model: str
    duration_seconds: float | None = None
    confidence: float | None = None
