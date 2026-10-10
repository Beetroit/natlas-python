"""FastAPI application providing OpenAI-compatible endpoints for N-ATLAS."""

import sys
import traceback
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
from natlas.dialects.prompts import PIDGIN_LOGIT_BIAS
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

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Global exception handler to print tracebacks to stderr and return debug details."""
        tb = traceback.format_exc()
        sys.stderr.write(
            f"\n[N-ATLAS SERVER ERROR] Unhandled exception during {request.method} {request.url.path}:\n{tb}\n"
        )
        sys.stderr.flush()
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "detail": f"Internal Server Error: {exc}",
                "traceback": tb,
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        """Log 5xx HTTPExceptions with traceback details."""
        tb = traceback.format_exc()
        if exc.status_code >= 500:
            sys.stderr.write(
                f"\n[N-ATLAS SERVER HTTP {exc.status_code}] Error during {request.method} {request.url.path}:\n"
                f"Detail: {exc.detail}\n"
                f"Traceback:\n{tb}\n"
            )
            sys.stderr.flush()
            content: dict[str, object] = {
                "error": exc.detail,
                "detail": exc.detail,
            }
            if tb and tb.strip() != "NoneType: None":
                content["traceback"] = tb
            return JSONResponse(status_code=exc.status_code, content=content)

        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=exc.headers,
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
            from natlas.api.types import (
                ChatCompletionChoice,
                ChatCompletionChunk,
                ChatCompletionChunkChoice,
                ChatCompletionChunkDelta,
                ChatCompletionMessage,
            )

            user_text = ""
            for msg in reversed(guarded_messages):
                if msg.get("role") == "user":
                    user_text = msg.get("content", "")
                    break

            lang_label = lang.capitalize() if lang else "Nigerian Multilingual"
            mock_reply = (
                f"[N-ATLAS Sovereign Engine (Mock Mode)]: Ẹ n lẹ o! Sannu! Nnọọ! "
                f"Received message: \"{user_text}\" with language target '{lang_label}'. "
                "The dialect tone guard and orthography pipeline are active. "
                "To load the fine-tuned 8B neural weights, launch via: natlas serve --model <path_to_N-ATLaS.gguf>"
            )

            if request.stream:

                async def mock_event_generator() -> AsyncIterator[str]:
                    import asyncio

                    words = mock_reply.split(" ")
                    for word in words:
                        chunk = ChatCompletionChunk(
                            id="natlas-mock-stream",
                            created=1728345600,
                            model=request.model,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChunkDelta(
                                        role="assistant", content=word + " "
                                    ),
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield chunk.model_dump_json()
                        await asyncio.sleep(0.02)

                    final_chunk = ChatCompletionChunk(
                        id="natlas-mock-stream",
                        created=1728345600,
                        model=request.model,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(role=None, content=""),
                                finish_reason="stop",
                            )
                        ],
                    )
                    yield final_chunk.model_dump_json()
                    yield "[DONE]"

                return EventSourceResponse(mock_event_generator())

            return ChatCompletionResponse(
                id="natlas-echo",
                created=1728345600,
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatCompletionMessage(
                            role="assistant", content=mock_reply
                        ),
                        finish_reason="stop",
                    )
                ],
            )

        logit_bias: dict[int, float] | None = (
            PIDGIN_LOGIT_BIAS if lang == "pidgin" else None
        )

        if request.stream:

            async def event_generator() -> AsyncIterator[str]:
                for chunk in active_engine.stream_generate(
                    guarded_messages,
                    temperature=request.temperature,
                    repetition_penalty=request.repetition_penalty,
                    logit_bias=logit_bias,
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
            repetition_penalty=request.repetition_penalty,
            logit_bias=logit_bias,
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
        try:
            resolved_model = model or resolve_asr_model(language)
            audio_bytes = await file.read()
            if not audio_bytes:
                raise AudioProcessingError(
                    "Received empty audio file upload (0 bytes)."
                )
            normalized_wav = transcode_to_16k_mono_wav(audio_bytes)

            active_asr = asr_engine
            if active_asr is None:
                return TranscriptionResponse(
                    text="[N-ATLAS Mock ASR] Transcription successful.",
                    language=language,
                    model=resolved_model,
                    duration_seconds=1.0,
                )

            return active_asr.transcribe(
                normalized_wav,
                model_name=resolved_model,
                language=language,
            )
        except AudioProcessingError as exc:
            tb = traceback.format_exc()
            sys.stderr.write(f"\n[N-ATLAS ASR AudioProcessingError 422]: {exc}\n{tb}\n")
            sys.stderr.flush()
            raise HTTPException(
                status_code=422,
                detail=f"Audio processing failed: {exc}\nTraceback:\n{tb}",
            ) from exc
        except Exception as exc:
            tb = traceback.format_exc()
            sys.stderr.write(f"\n[N-ATLAS ASR 500 Failure]: {exc}\n{tb}\n")
            sys.stderr.flush()
            raise HTTPException(
                status_code=500,
                detail=f"ASR transcription failed: {exc}\nTraceback:\n{tb}",
            ) from exc

    return app
