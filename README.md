# natlas (Python SDK & Engine)

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org)

The official Python SDK and local serving runtime for **N-ATLAS** (Nigeria's sovereign multilingual Large Language Model and Automatic Speech Recognition system).

Built for the **National AI Innovation Challenge (NAIC 2026)** by NCAIR and NITDA under **Problem Statement 01: Developer Infrastructure** (Track B: Innovation & Enterprise).

---

## Architecture

```mermaid
flowchart TD
    subgraph ClientSDK ["Python Client (src/natlas/client.py)"]
        Sync["Natlas (Synchronous - httpx.Client)"]
        Async["AsyncNatlas (Asynchronous - httpx.AsyncClient)"]
    end

    subgraph DialectEngine ["Dialect & Tone Guards (src/natlas/dialects/)"]
        Prompts["prompts.py (System Prompts: Yoruba, Hausa, Igbo, Pidgin)"]
        Guards["guards.py (apply_dialect_guard)"]
        Normalizer["normalizer.py (Unicode NFC & Diacritics)"]
        Prompts --> Guards
    end

    subgraph AudioEngine ["Voice Transcoder & Router (src/natlas/audio/)"]
        Transcoder["transcoder.py (PyAV in-memory 16kHz WAV Resampler)"]
        Router["router.py (NCAIR1 Whisper Checkpoint Router)"]
    end

    subgraph ServingEngine ["FastAPI Local Server (src/natlas/server/)"]
        App["app.py (OpenAI-compatible REST API)"]
        Engine["engine.py (CUDA GGUF & faster-whisper)"]
        CLI["cli/main.py ('natlas serve', 'natlas chat')"]
        CLI --> App
        App --> Engine
    end

    Sync --> Guards
    Async --> Guards
    Sync --> Transcoder
    Async --> Transcoder
    Transcoder --> Router
    Sync -. HTTP/SSE .-> App
    Async -. HTTP/SSE .-> App
```

---

## Features

- **Dual Client Architecture**: Both synchronous (`Natlas`) and asynchronous (`AsyncNatlas`) clients with identical API surfaces via `httpx`.
- **Dialect Guards & Tone Integrity**: Automatic system prompt injection preventing language drift back to English for **Yorùbá, Hausa, Igbo, and Nigerian Pidgin**.
- **Turnkey Audio Transcoding (PyAV)**: In-memory transcoding of WhatsApp voice notes (`.ogg`, `.opus`), browser recordings (`.webm`), and mobile audio to 16kHz mono WAV in pure Python—no system `ffmpeg.exe` installation required.
- **Unified ASR Router**: Directly routes voice inputs to the official `NCAIR1` Whisper checkpoints (`Yoruba-ASR`, `Hausa-ASR`, `Igbo-ASR`, `NigerianAccentedEnglish`).
- **OpenAI-Compatible Local Engine**: Run `natlas serve` to spin up a local FastAPI server with CUDA GPU acceleration on consumer hardware (such as RTX 3060).

---

## Installation

### Client SDK (Lightweight)
```bash
# Using uv (Recommended)
uv add natlas

# Using pip
pip install natlas
```

### With Voice / Audio Transcoding
```bash
# Using uv (Recommended)
uv add "natlas[audio]"

# Using pip
pip install "natlas[audio]"
```

### With Local GPU Serving Engine
```bash
# Using uv (Recommended)
uv add "natlas[server]"

# Using pip
pip install "natlas[server]"
```

---

## Quickstart

### Multilingual Chat Completion
```python
from natlas import Natlas

client = Natlas(base_url="http://127.0.0.1:8000/v1")

# Chat with dialect enforcement
response = client.chat.create(
    language="yoruba",
    tone_enforcement=True,
    messages=[
        {"role": "user", "content": "Bawo ni mo se le se iforukọsilẹ ile-iṣẹ mi?"}
    ],
)

print(response.choices[0].message.content)
```

### Streaming Tokens in Real Time
```python
for chunk in client.chat.stream(
    language="hausa",
    messages=[{"role": "user", "content": "Barka dai!"}],
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Voice Note Transcription
```python
result = client.audio.transcribe(
    file="whatsapp_note.ogg",
    language="yoruba",
    transcode=True,
)
print("Transcription:", result.text)
print("Model:", result.model)
```

---

## Local Serving CLI

Launch the local inference server offloading GGUF layers to your CUDA GPU:

```bash
# Using uv (Recommended)
uv run natlas serve --model ./models/N-ATLaS.Q4_K_M.gguf --gpu-layers 32 --port 8000

# Using pip / global environment
natlas serve --model ./models/N-ATLaS.Q4_K_M.gguf --gpu-layers 32 --port 8000
```

Start interactive terminal chat:
```bash
# Using uv (Recommended)
uv run natlas chat --language yoruba

# Using pip
natlas chat --language yoruba
```

---

## Testing & Verification

```bash
# Run tests
uv run pytest tests/

# Lint and type check
uv run ruff check .
uvx ty check src
uvx pyrefly check src
```
