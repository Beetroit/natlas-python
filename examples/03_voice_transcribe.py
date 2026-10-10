"""Voice note transcription using NCAIR1 Whisper ASR models."""

from pathlib import Path

from natlas import Natlas

client = Natlas()

audio_file = Path("sample_voice_note.ogg")

if audio_file.exists():
    # Automatically transcodes WhatsApp .ogg/.opus into 16kHz mono WAV
    # and routes to NCAIR1/Yoruba-ASR
    result = client.audio.transcribe(
        file=audio_file,
        language="yoruba",
        transcode=True,
    )
    print("Transcribed Text:", result.text)
    print("Model Used:", result.model)
    print("Duration (s):", result.duration_seconds)
else:
    print(f"To run this example, place an audio file at {audio_file.resolve()}")

client.close()
