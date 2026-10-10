"""Audio transcription client namespace (synchronous and asynchronous)."""

from pathlib import Path
from typing import BinaryIO

import anyio
import httpx

from natlas.api.types import TranscriptionResponse
from natlas.audio.router import resolve_asr_model
from natlas.audio.transcoder import transcode_to_16k_mono_wav
from natlas.config import LanguageCode
from natlas.exceptions import APIConnectionError, APIStatusError


class Audio:
    """Synchronous audio transcription resource."""

    def __init__(self, http_client: httpx.Client, base_url: str) -> None:
        self._client: httpx.Client = http_client
        self._base_url: str = base_url.rstrip("/")

    def transcribe(
        self,
        file: str | Path | bytes | BinaryIO,
        *,
        language: LanguageCode | str = "english",
        transcode: bool = True,
    ) -> TranscriptionResponse:
        """Transcribe speech into text using official NCAIR1 ASR Whisper checkpoints.

        Args:
            file: Path to audio file, raw audio bytes, or binary IO stream.
            language: Target Nigerian language or accent ('yoruba', 'hausa', 'igbo', 'english').
            transcode: If True, automatically transcodes to 16kHz mono WAV before upload.

        Returns:
            TranscriptionResponse containing transcribed text and metadata.
        """
        model_name = resolve_asr_model(language)

        wav_bytes: bytes
        if transcode:
            wav_bytes = transcode_to_16k_mono_wav(file)
        elif isinstance(file, (str, Path)):
            with open(file, "rb") as f:
                wav_bytes = f.read()
        elif isinstance(file, bytes):
            wav_bytes = file
        else:
            file.seek(0)
            wav_bytes = file.read()

        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {
            "model": model_name,
            "language": language,
        }

        try:
            response = self._client.post(
                f"{self._base_url}/audio/transcriptions",
                files=files,
                data=data,
            )
        except httpx.RequestError as exc:
            raise APIConnectionError(
                f"Could not connect to N-ATLAS ASR server: {exc}"
            ) from exc

        if response.status_code != 200:
            raise APIStatusError(
                message=f"Audio transcription failed with status {response.status_code}",
                status_code=response.status_code,
                response_body=response.text,
            )

        return TranscriptionResponse.model_validate(response.json())


class AsyncAudio:
    """Asynchronous audio transcription resource."""

    def __init__(self, http_client: httpx.AsyncClient, base_url: str) -> None:
        self._client: httpx.AsyncClient = http_client
        self._base_url: str = base_url.rstrip("/")

    async def transcribe(
        self,
        file: str | Path | bytes | BinaryIO,
        *,
        language: LanguageCode | str = "english",
        transcode: bool = True,
    ) -> TranscriptionResponse:
        """Transcribe speech into text asynchronously."""
        model_name = resolve_asr_model(language)

        wav_bytes: bytes
        if transcode:
            wav_bytes = transcode_to_16k_mono_wav(file)
        elif isinstance(file, (str, Path)):
            wav_bytes = await anyio.Path(file).read_bytes()
        elif isinstance(file, bytes):
            wav_bytes = file
        else:
            file.seek(0)
            wav_bytes = file.read()

        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {
            "model": model_name,
            "language": language,
        }

        try:
            response = await self._client.post(
                f"{self._base_url}/audio/transcriptions",
                files=files,
                data=data,
            )
        except httpx.RequestError as exc:
            raise APIConnectionError(
                f"Could not connect to N-ATLAS ASR server: {exc}"
            ) from exc

        if response.status_code != 200:
            raise APIStatusError(
                message=f"Audio transcription failed with status {response.status_code}",
                status_code=response.status_code,
                response_body=response.text,
            )

        return TranscriptionResponse.model_validate(response.json())
