"""FastAPI application providing OpenAI-compatible endpoints for N-ATLAS."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from natlas.api.types import (
    ChatCompletionResponse,
    TranscriptionResponse,
)
from natlas.audio.router import resolve_asr_model
from natlas.audio.transcoder import transcode_to_16k_mono_wav
from natlas.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from natlas.dialects.guards import apply_dialect_guard
from natlas.exceptions import AudioProcessingError


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request schema."""

    model: str = DEFAULT_MODEL
    messages: list[dict[str, str]]
    language: str | None = None
    tone_enforcement: bool = True
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    repetition_penalty: float = Field(
        default=DEFAULT_REPETITION_PENALTY, ge=1.0, le=2.0
    )
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1)
    top_p: float = Field(default=DEFAULT_TOP_P, ge=0.0, le=1.0)
    stream: bool = False


from natlas.server.engine import ASREngineProtocol, LLMEngineProtocol


def create_app(
    engine: LLMEngineProtocol | None = None,
    asr_engine: ASREngineProtocol | None = None,
) -> FastAPI:
    """Create and configure the FastAPI N-ATLAS application."""
    app = FastAPI(
        title="N-ATLAS API Server",
        description="OpenAI-compatible inference engine for Nigeria's sovereign AI model and ASR.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health_check() -> dict[str, str]:
        return {
            "status": "healthy",
            "model": DEFAULT_MODEL,
            "version": "0.1.0",
        }

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse | EventSourceResponse:
        """OpenAI-compatible chat completion endpoint with SSE streaming."""
        from natlas.config import LanguageCode

        lang: LanguageCode | None = (
            request.language
            if request.language in ("yoruba", "hausa", "igbo", "english", "pidgin")
            else None
        )
        guarded_messages = apply_dialect_guard(
            request.messages,
            language=lang,
            tone_enforcement=request.tone_enforcement,
        )

        active_engine = engine
        if active_engine is None:
            # Fallback mock/echo response when engine is not initialized
            from natlas.api.types import ChatCompletionChoice, ChatCompletionMessage

            mock_content = f"[N-ATLAS echo] Received {len(guarded_messages)} messages."
            return ChatCompletionResponse(
                id="natlas-echo",
                created=1728345600,
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatCompletionMessage(
                            role="assistant", content=mock_content
                        ),
                        finish_reason="stop",
                    )
                ],
            )

        if request.stream:

            async def event_generator() -> AsyncIterator[str]:
                for chunk in active_engine.stream_generate(
                    guarded_messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    top_p=request.top_p,
                ):
                    yield chunk.model_dump_json()
                yield "[DONE]"

            return EventSourceResponse(event_generator())

        # Non-streaming completion
        response = active_engine.generate(
            guarded_messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
        )
        return response

    @app.post("/v1/audio/transcriptions")
    async def audio_transcriptions(
        file: Annotated[UploadFile, File(...)],
        model: Annotated[str, Form()] = "",
        language: Annotated[str, Form()] = "english",
    ) -> TranscriptionResponse:
        """OpenAI-compatible audio transcription endpoint."""
        resolved_model = model or resolve_asr_model(language)
        audio_bytes = await file.read()
        normalized_wav = transcode_to_16k_mono_wav(audio_bytes)

        active_asr = asr_engine
        if active_asr is None:
            return TranscriptionResponse(
                text="[N-ATLAS Mock ASR] Transcription successful.",
                language=language,
                model=resolved_model,
                duration_seconds=1.0,
            )

        try:
            return active_asr.transcribe(
                normalized_wav,
                model_name=resolved_model,
                language=language,
            )
        except AudioProcessingError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"ASR transcription failed: {exc}"
            ) from exc

    return app
