"""Diagnostic and environment setup command for N-ATLAS."""

import os
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from natlas.cli.hardware import (
    detect_gpu_acceleration,
    ensure_asr_dependencies,
    ensure_llama_cpp,
    is_asr_installed,
    is_llama_cpp_installed,
)
from natlas.cli.pull import resolve_cached_llm

console = Console()


def setup(
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Automatically confirm installation of missing acceleration wheels",
        ),
    ] = False,
) -> None:
    """Inspect local hardware, verify model access, and configure GPU acceleration."""
    console.print(
        "[bold cyan]=== N-ATLAS Developer Environment Setup & Diagnostics ===[/bold cyan]\n"
    )

    table = Table(title="System & Acceleration Readiness", show_header=True)
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Status", style="bold")
    table.add_column("Details", style="dim")

    # 1. Python Environment & Environment Variables
    from dotenv import find_dotenv, load_dotenv

    dotenv_file = find_dotenv()
    if dotenv_file:
        load_dotenv(dotenv_file)

    py_ver = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    table.add_row("Python", f"v{py_ver}", sys.executable)

    # 2. Hardware Accelerator
    backend, backend_details = detect_gpu_acceleration()
    backend_status = (
        "[green]Ready[/green]" if backend != "cpu" else "[yellow]CPU Only[/yellow]"
    )
    table.add_row("Hardware", backend_status, backend_details)

    # 3. Hugging Face Authentication
    try:
        from huggingface_hub import get_token

        hf_token = get_token() or os.environ.get("HF_TOKEN")
    except ImportError:
        hf_token = os.environ.get("HF_TOKEN")

    if hf_token:
        hf_status = "[green]Authenticated[/green]"
        hf_info = f"Token set ({hf_token[:4]}...{hf_token[-3:]})"
    else:
        hf_status = "[yellow]Not Configured[/yellow]"
        hf_info = "Set HF_TOKEN in .env for official gated ASR models"
    table.add_row("Hugging Face Auth", hf_status, hf_info)

    # 4. Sovereign LLM Weights
    cached_llm = resolve_cached_llm("n-atlas")
    if cached_llm:
        llm_status = "[green]Cached on Disk[/green]"
        llm_info = cached_llm
    else:
        llm_status = "[yellow]Missing[/yellow]"
        llm_info = "Run 'natlas pull n-atlas:q4' (~4.9 GB)"
    table.add_row("N-ATLAS GGUF Model", llm_status, llm_info)

    # 5. GGUF Engine Library (llama-cpp-python)
    llama_ready = is_llama_cpp_installed()
    if llama_ready:
        llama_status = "[green]Installed[/green]"
        llama_info = "Local GGUF GPU/CPU inference active"
    else:
        llama_status = "[bold red]Not Installed[/bold red]"
        llama_info = "Requires pre-compiled wheel for GPU acceleration"
    table.add_row("LLM Engine (llama-cpp)", llama_status, llama_info)

    # 6. Audio Transcoder
    try:
        import av  # noqa: F401

        audio_status = "[green]Installed[/green]"
        audio_info = "PyAV in-memory WAV resampler active"
    except (ImportError, ModuleNotFoundError):
        audio_status = "[yellow]Missing (Optional)[/yellow]"
        audio_info = "Install via: uv pip install av"
    table.add_row("Audio Transcoder", audio_status, audio_info)

    # 7. Sovereign ASR Engine (transformers / torch)
    asr_ready = is_asr_installed()
    if asr_ready:
        asr_status = "[green]Installed[/green]"
        asr_info = "Official transformers & PyTorch speech pipelines active"
    else:
        asr_status = "[bold red]Not Installed[/bold red]"
        asr_info = "Requires transformers, torch, and soundfile for sovereign ASR"
    table.add_row("Sovereign ASR (transformers)", asr_status, asr_info)

    console.print(table)
    console.print()

    # Bootstrap missing components
    if not llama_ready:
        console.print(
            "[bold yellow]Action Needed:[/bold yellow] LLM inference engine is missing."
        )
        ensure_llama_cpp(auto_yes=yes)

    if not asr_ready:
        console.print(
            "[bold yellow]Action Needed:[/bold yellow] Sovereign ASR dependencies are missing."
        )
        ensure_asr_dependencies(auto_yes=yes)

    if llama_ready and asr_ready:
        console.print(
            "[bold green]All core inference components are installed and ready to serve![/bold green]"
        )
