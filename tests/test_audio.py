"""Unit tests for audio model routing and format handling."""

import pytest

from natlas.audio.router import resolve_asr_model
from natlas.exceptions import AudioProcessingError


def test_resolve_asr_model_valid_languages() -> None:
    """Verify correct Hugging Face repository mapping for Nigerian ASR models."""
    assert resolve_asr_model("yoruba") == "NCAIR1/Yoruba-ASR"
    assert resolve_asr_model("hausa") == "NCAIR1/Hausa-ASR"
    assert resolve_asr_model("igbo") == "NCAIR1/Igbo-ASR"
    assert resolve_asr_model("english") == "NCAIR1/NigerianAccentedEnglish"
    assert resolve_asr_model("pidgin") == "NCAIR1/NigerianAccentedEnglish"


def test_resolve_asr_model_unsupported_language() -> None:
    """Verify exception raised when requesting an unsupported language."""
    with pytest.raises(AudioProcessingError) as exc_info:
        resolve_asr_model("swahili")
    assert "Unsupported language" in str(exc_info.value)


def test_transcode_and_wav_to_float32() -> None:
    """Verify in-memory WAV transcoding and numpy float32 conversion."""
    import io
    import wave

    from natlas.audio.transcoder import transcode_to_16k_mono_wav
    from natlas.server.engine import _wav_to_float32_array

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)

    raw_wav = transcode_to_16k_mono_wav(buf.getvalue())
    assert raw_wav.startswith(b"RIFF")

    float_arr = _wav_to_float32_array(raw_wav)
    assert len(float_arr) == 1600
    assert float_arr.dtype.name == "float32"
