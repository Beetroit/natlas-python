"""Audio transcoding and normalization utilities for N-ATLAS voice pipeline."""

import io
import wave
from pathlib import Path
from typing import BinaryIO

from natlas.exceptions import AudioProcessingError

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


def transcode_to_16k_mono_wav(
    audio_source: str | Path | bytes | BinaryIO,
) -> bytes:
    """Normalize any audio file or byte stream into 16kHz mono 16-bit PCM WAV.

    Supports WhatsApp audio (.ogg, .opus), browser mic recordings (.webm),
    M4A, MP3, and WAV.

    Args:
        audio_source: File path, raw audio bytes, or binary IO stream.

    Returns:
        Bytes containing valid 16kHz mono 16-bit PCM WAV audio.
    """
    try:
        import av
        from av.audio.frame import AudioFrame
        from av.audio.resampler import AudioResampler
        from av.container import InputContainer
    except ImportError:
        # Fallback if PyAV is not installed: check if input is already raw WAV
        return _fallback_wav_check(audio_source)

    try:
        input_container: BinaryIO
        if isinstance(audio_source, (str, Path)):
            with open(audio_source, "rb") as f:
                input_bytes = f.read()
            input_container = io.BytesIO(input_bytes)
        elif isinstance(audio_source, bytes):
            input_container = io.BytesIO(audio_source)
        else:
            input_container = audio_source

        input_container.seek(0)
        container = av.open(input_container)

        if not isinstance(container, InputContainer):
            raise AudioProcessingError(
                "Provided audio is not a readable media container."
            )

        audio_streams = [s for s in container.streams if s.type == "audio"]
        if not audio_streams:
            raise AudioProcessingError("No audio stream found in the provided media.")

        resampler = AudioResampler(
            format="s16",
            layout="mono",
            rate=TARGET_SAMPLE_RATE,
        )

        output_io = io.BytesIO()
        with wave.open(output_io, "wb") as wav_file:
            wav_file.setnchannels(TARGET_CHANNELS)
            wav_file.setsampwidth(2)  # 16-bit PCM = 2 bytes
            wav_file.setframerate(TARGET_SAMPLE_RATE)

            for frame in container.decode(audio_streams[0]):
                if isinstance(frame, AudioFrame):
                    resampled_frames = resampler.resample(frame)
                    if resampled_frames is not None:
                        for resampled_frame in resampled_frames:
                            raw_planes = resampled_frame.planes
                            if raw_planes:
                                wav_file.writeframes(bytes(raw_planes[0]))

        return output_io.getvalue()
    except Exception as exc:
        if isinstance(exc, AudioProcessingError):
            raise
        raise AudioProcessingError(f"Failed to transcode audio: {exc}") from exc


def _fallback_wav_check(audio_source: str | Path | bytes | BinaryIO) -> bytes:
    """Fallback handler when PyAV is not installed."""
    raw_data: bytes
    if isinstance(audio_source, (str, Path)):
        with open(audio_source, "rb") as f:
            raw_data = f.read()
    elif isinstance(audio_source, bytes):
        raw_data = audio_source
    else:
        audio_source.seek(0)
        raw_data = audio_source.read()

    # Check for RIFF header
    if raw_data.startswith(b"RIFF") and b"WAVE" in raw_data[:12]:
        return raw_data

    raise AudioProcessingError(
        "Compressed audio (.ogg, .opus, .webm) requires PyAV. "
        "Please install the audio extra: pip install 'natlas[audio]'"
    )
