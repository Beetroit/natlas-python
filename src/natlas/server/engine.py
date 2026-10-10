"""Inference engine managing local GGUF and Whisper ASR models."""

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from natlas.api.types import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionMessage,
    ChatCompletionResponse,
    TranscriptionResponse,
    UsageInfo,
)
from natlas.config import DEFAULT_TEMPERATURE, DEFAULT_TOP_P
from natlas.exceptions import ModelNotFoundError


class LLMEngineProtocol(Protocol):
    """Protocol defining the interface for an LLM inference backend."""

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        top_p: float,
    ) -> ChatCompletionResponse: ...

    def stream_generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        top_p: float,
    ) -> Iterator[ChatCompletionChunk]: ...


class ASREngineProtocol(Protocol):
    """Protocol defining the interface for an ASR inference backend."""

    def transcribe(
        self,
        audio_wav_bytes: bytes,
        model_name: str,
        language: str,
    ) -> TranscriptionResponse: ...


class LlamaCppEngine:
    """Inference engine leveraging llama-cpp-python for GGUF weights with CUDA."""

    def __init__(
        self,
        model_path: str | Path,
        n_gpu_layers: int = 32,
        n_ctx: int = 8192,
    ) -> None:
        self.model_path = str(model_path)
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self._llm = None

        if not os.path.exists(self.model_path):
            raise ModelNotFoundError(f"GGUF model file not found at: {self.model_path}")

        try:
            import importlib

            llama_mod = importlib.import_module("llama_cpp")
            llama_cls = llama_mod.Llama
            self._llm = llama_cls(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                verbose=False,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "llama-cpp-python is required for local inference. "
                "Install via: pip install 'natlas[server]'"
            ) from exc

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = 1024,
        top_p: float = DEFAULT_TOP_P,
    ) -> ChatCompletionResponse:
        """Run full completion through llama-cpp."""
        if self._llm is None:
            raise RuntimeError("LLM is not loaded.")

        res = self._llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=False,
        )

        choice_data = res["choices"][0]
        choice = ChatCompletionChoice(
            index=0,
            message=ChatCompletionMessage(
                role="assistant",
                content=choice_data["message"]["content"],
            ),
            finish_reason=choice_data.get("finish_reason"),
        )
        usage = None
        if "usage" in res:
            usage = UsageInfo(
                prompt_tokens=res["usage"].get("prompt_tokens", 0),
                completion_tokens=res["usage"].get("completion_tokens", 0),
                total_tokens=res["usage"].get("total_tokens", 0),
            )

        return ChatCompletionResponse(
            id=res.get("id", "natlas-local"),
            created=res.get("created", 0),
            model="N-ATLaS-8B-GGUF",
            choices=[choice],
            usage=usage,
        )

    def stream_generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = 1024,
        top_p: float = DEFAULT_TOP_P,
    ) -> Iterator[ChatCompletionChunk]:
        """Stream completion tokens chunk by chunk."""
        if self._llm is None:
            raise RuntimeError("LLM is not loaded.")

        stream_iter = self._llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True,
        )

        for chunk in stream_iter:
            choice_data = chunk["choices"][0]
            delta_content = choice_data.get("delta", {}).get("content", "")
            yield ChatCompletionChunk(
                id=chunk.get("id", "natlas-stream"),
                created=chunk.get("created", 0),
                model="N-ATLaS-8B-GGUF",
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(
                            role="assistant", content=delta_content
                        ),
                        finish_reason=choice_data.get("finish_reason"),
                    )
                ],
            )


import io
import logging
import wave
from collections.abc import Iterable, Iterator
from typing import Final

import numpy as np
from dotenv import find_dotenv, load_dotenv

from natlas.exceptions import AudioProcessingError

logger = logging.getLogger("natlas.asr")

WHISPER_LANG_MAP: Final[dict[str, str | None]] = {
    "yoruba": "yo",
    "hausa": "ha",
    "english": "en",
    "pidgin": "en",
    "nigerian english": "en",
    "igbo": None,
}


def _wav_to_float32_array(audio_wav_bytes: bytes) -> np.ndarray:
    """Convert 16kHz mono WAV bytes to 1D float32 numpy array normalized to [-1.0, 1.0]."""
    try:
        with wave.open(io.BytesIO(audio_wav_bytes), "rb") as wf:
            raw_frames = wf.readframes(wf.getnframes())
            if not raw_frames:
                return np.zeros(0, dtype=np.float32)
            audio_int16 = np.frombuffer(raw_frames, dtype=np.int16)
            return (audio_int16.astype(np.float32) / 32768.0).copy()
    except Exception as exc:
        raise AudioProcessingError(f"Failed to decode WAV audio frames: {exc}") from exc


class WhisperModelProtocol(Protocol):
    """Protocol for faster_whisper WhisperModel."""

    def transcribe(
        self,
        audio: str | io.BytesIO | np.ndarray,
        *,
        language: str | None = None,
    ) -> tuple[Iterable[object], object]: ...


class ASREngine:
    """ASR Inference engine using faster-whisper with resilient fallback."""

    def __init__(
        self,
        device: str = "auto",
        compute_type: str | None = None,
        fallback_model: str = "tiny",
    ) -> None:
        dotenv_path = find_dotenv()
        if dotenv_path:
            load_dotenv(dotenv_path)

        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        self.device = device
        self.compute_type = compute_type
        self.fallback_model = fallback_model
        self._models: dict[str, tuple[WhisperModelProtocol, str]] = {}

    def _init_whisper_model(
        self, model_id: str, device: str, compute_type: str
    ) -> WhisperModelProtocol:
        from faster_whisper import WhisperModel

        token = os.environ.get("HF_TOKEN")
        model: WhisperModelProtocol = WhisperModel(
            model_id,
            device=device,
            compute_type=compute_type,
            use_auth_token=token,
        )
        return model

    def _resolve_and_load(
        self, requested_model: str
    ) -> tuple[WhisperModelProtocol, str]:
        """Load requested model with fallback to open Whisper and CPU if CUDA fails."""
        if requested_model in self._models:
            return self._models[requested_model]

        device = "cuda" if self.device in ("auto", "cuda") else "cpu"
        compute_type = (
            self.compute_type
            if self.compute_type is not None
            else ("float16" if device == "cuda" else "int8")
        )

        model_candidates = [requested_model]
        if requested_model != self.fallback_model:
            model_candidates.append(self.fallback_model)

        last_error: Exception | None = None

        for candidate in model_candidates:
            # First attempt with chosen device (CUDA or CPU)
            try:
                model = self._init_whisper_model(candidate, device, compute_type)
                self._models[requested_model] = (model, candidate)
                return model, candidate
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Failed to load ASR model '%s' on %s (%s): %s",
                    candidate,
                    device,
                    compute_type,
                    exc,
                )

                # If CUDA failed, try CPU fallback for same candidate
                if device == "cuda":
                    try:
                        logger.info("Retrying '%s' on CPU with int8...", candidate)
                        model = self._init_whisper_model(candidate, "cpu", "int8")
                        self._models[requested_model] = (model, candidate)
                        return model, candidate
                    except Exception as cpu_exc:  # noqa: BLE001
                        last_error = cpu_exc
                        logger.warning(
                            "Failed CPU fallback for '%s': %s", candidate, cpu_exc
                        )

        raise AudioProcessingError(
            f"Unable to load ASR model '{requested_model}' or fallback '{self.fallback_model}': {last_error}"
        )

    def transcribe(
        self,
        audio_wav_bytes: bytes,
        model_name: str,
        language: str,
    ) -> TranscriptionResponse:
        """Run Whisper transcription on 16kHz WAV audio bytes."""
        try:
            import faster_whisper  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is required for local ASR serving. "
                "Install via: pip install 'natlas[server]'"
            ) from exc

        model, active_model_name = self._resolve_and_load(model_name)
        audio_np = _wav_to_float32_array(audio_wav_bytes)

        normalized_lang = language.strip().lower()
        whisper_lang = WHISPER_LANG_MAP.get(normalized_lang, None)

        transcribe_fn = model.transcribe
        try:
            segments, info = transcribe_fn(audio_np, language=whisper_lang)
            full_text = " ".join(
                [str(getattr(seg, "text", "")).strip() for seg in segments]
            ).strip()
        except RuntimeError as exc:
            # If cuBLAS or CUDA runtime error occurs during encode, retry on CPU
            if "cublas" in str(exc).lower() or "cuda" in str(exc).lower():
                logger.warning(
                    "CUDA runtime error during ASR (%s). Switching to CPU...", exc
                )
                cpu_model = self._init_whisper_model(active_model_name, "cpu", "int8")
                self._models[model_name] = (cpu_model, active_model_name)
                cpu_transcribe = cpu_model.transcribe
                segments, info = cpu_transcribe(audio_np, language=whisper_lang)
                full_text = " ".join(
                    [str(getattr(seg, "text", "")).strip() for seg in segments]
                ).strip()
            else:
                raise AudioProcessingError(
                    f"ASR transcription runtime failure: {exc}"
                ) from exc
        except Exception as exc:
            raise AudioProcessingError(f"ASR transcription failed: {exc}") from exc

        return TranscriptionResponse(
            text=full_text,
            language=language,
            model=active_model_name,
            duration_seconds=getattr(info, "duration", None),
        )
