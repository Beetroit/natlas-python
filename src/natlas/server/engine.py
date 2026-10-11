"""Inference engine managing local GGUF and Whisper ASR models."""

import os
import sys
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
from natlas.config import (
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from natlas.exceptions import ModelNotFoundError


class LLMEngineProtocol(Protocol):
    """Protocol defining the interface for an LLM inference backend."""

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        logit_bias: dict[int, float] | None = None,
        max_tokens: int,
        top_p: float,
    ) -> ChatCompletionResponse: ...

    def stream_generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        logit_bias: dict[int, float] | None = None,
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

            from natlas.cli.hardware import resolve_cuda_dlls

            resolve_cuda_dlls()

            llama_mod = importlib.import_module("llama_cpp")
            llama_cls = llama_mod.Llama
            self._llm = llama_cls(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                verbose=False,
            )
        except (ImportError, ModuleNotFoundError, RuntimeError, OSError) as exc:
            raise ImportError(
                "llama-cpp-python is required for local GGUF inference. "
                "Install via: uv pip install llama-cpp-python (or uv sync --extra llama)"
            ) from exc

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        logit_bias: dict[int, float] | None = None,
        max_tokens: int = 1024,
        top_p: float = DEFAULT_TOP_P,
    ) -> ChatCompletionResponse:
        """Run full completion through llama-cpp."""
        if self._llm is None:
            raise RuntimeError("LLM is not loaded.")

        res = self._llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            repeat_penalty=repetition_penalty,
            logit_bias=logit_bias,
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
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        logit_bias: dict[int, float] | None = None,
        max_tokens: int = 1024,
        top_p: float = DEFAULT_TOP_P,
    ) -> Iterator[ChatCompletionChunk]:
        """Stream completion tokens chunk by chunk."""
        if self._llm is None:
            raise RuntimeError("LLM is not loaded.")

        stream_iter = self._llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            repeat_penalty=repetition_penalty,
            logit_bias=logit_bias,
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
from collections.abc import Iterator
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


class ASRModelPipelineProtocol(Protocol):
    """Protocol for Hugging Face Transformers ASR pipeline."""

    def __call__(
        self,
        inputs: np.ndarray,
        **kwargs: object,
    ) -> dict[str, object] | list[dict[str, object]]: ...


class ASREngine:
    """Strict Sovereign ASR Inference engine running official NCAIR1 checkpoints."""

    def __init__(
        self,
        device: str = "auto",
        compute_type: str | None = None,
    ) -> None:
        dotenv_path = find_dotenv()
        if dotenv_path:
            load_dotenv(dotenv_path)

        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        # Windows DLL discovery for NVIDIA cuBLAS if installed
        if sys.platform == "win32":
            try:
                import importlib

                cublas_lib = importlib.import_module("nvidia.cublas.lib")
                cublas_file = getattr(cublas_lib, "__file__", None)
                if cublas_file:
                    cublas_dir = os.path.dirname(cublas_file)
                    if os.path.exists(cublas_dir):
                        os.add_dll_directory(cublas_dir)
            except (ImportError, AttributeError):
                pass

        self.requested_device = device
        self.compute_type = compute_type
        self._token: str | None = os.environ.get("HF_TOKEN")
        self._active_device: str = self._resolve_device(device)
        self._pipelines: dict[str, ASRModelPipelineProtocol] = {}

        # Verify access on initialization
        self._verify_sovereign_access()

    def _resolve_device(self, requested: str) -> str:
        """Resolve requested device to available PyTorch device."""
        if requested in ("auto", "cuda"):
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            logger.info(
                "N-ATLAS ASR: CUDA is not active in current PyTorch runtime. Operating on CPU."
            )
            return "cpu"
        return "cpu"

    def _verify_sovereign_access(self) -> None:
        """Validate that user has approved Hugging Face token access to NCAIR1 models."""
        if not self._token:
            raise AudioProcessingError(
                "HF_TOKEN is not set. Access to official sovereign N-ATLAS ASR models "
                "(NCAIR1/Yoruba-ASR, NCAIR1/Hausa-ASR, NCAIR1/Igbo-ASR, NCAIR1/NigerianAccentedEnglish) "
                "requires an authorized Hugging Face token. "
                "Please configure HF_TOKEN in your environment or .env file with access granted at "
                "https://huggingface.co/NCAIR1/Yoruba-ASR."
            )

        import httpx

        headers = {"Authorization": f"Bearer {self._token}"}
        try:
            resp = httpx.get(
                "https://huggingface.co/api/models/NCAIR1/Yoruba-ASR",
                headers=headers,
                timeout=5.0,
            )
            if resp.status_code in (401, 403):
                raise AudioProcessingError(
                    f"Access to sovereign N-ATLAS models is restricted (HTTP {resp.status_code}). "
                    "Verify that your HF_TOKEN has accepted the access terms at "
                    "https://huggingface.co/NCAIR1/Yoruba-ASR."
                )
        except httpx.RequestError as exc:
            logger.warning(
                "Unable to reach Hugging Face to verify token access (proceeding with cached weights): %s",
                exc,
            )

    def _load_pipeline(self, model_name: str) -> ASRModelPipelineProtocol:
        """Load and cache the official transformers ASR pipeline for model_name."""
        if model_name in self._pipelines:
            return self._pipelines[model_name]

        try:
            from transformers import pipeline  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "transformers is required for local ASR serving. "
                "Install via: pip install 'natlas[server]'"
            ) from exc

        logger.info(
            "Loading sovereign ASR checkpoint '%s' on %s...",
            model_name,
            self._active_device,
        )

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            token=self._token,
            device=self._active_device,
        )
        self._pipelines[model_name] = pipe
        return pipe

    def transcribe(
        self,
        audio_wav_bytes: bytes,
        model_name: str,
        language: str,
    ) -> TranscriptionResponse:
        """Run sovereign ASR transcription on 16kHz WAV audio bytes."""
        import unicodedata

        audio_np = _wav_to_float32_array(audio_wav_bytes)
        duration_seconds = float(len(audio_np) / 16000.0) if len(audio_np) > 0 else 0.0

        pipe = self._load_pipeline(model_name)

        try:
            result = pipe(
                audio_np,
                chunk_length_s=30,
                stride_length_s=4,
                return_timestamps=True,
            )
            if isinstance(result, list) and len(result) > 0:
                raw_text = str(result[0].get("text", "")).strip()
            elif isinstance(result, dict):
                raw_text = str(result.get("text", "")).strip()
            else:
                raw_text = ""
            normalized_text = unicodedata.normalize("NFC", raw_text)
        except Exception as exc:
            raise AudioProcessingError(f"ASR transcription failed: {exc}") from exc

        return TranscriptionResponse(
            text=normalized_text,
            language=language,
            model=model_name,
            duration_seconds=duration_seconds,
        )
