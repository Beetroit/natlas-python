"""CLI command for launching the N-ATLAS local inference server."""

import os
from pathlib import Path

import typer
from rich.console import Console

console = Console()


def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind host address"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    model: str = typer.Option(
        "", "--model", "-m", help="Path to N-ATLAS GGUF model file"
    ),
    n_gpu_layers: int = typer.Option(
        32, "--gpu-layers", "-g", help="Layers to offload to CUDA GPU"
    ),
    device: str = typer.Option(
        "auto", "--device", "-d", help="Device for ASR (auto, cuda, or cpu)"
    ),
    mock: bool = typer.Option(
        False, "--mock", help="Run in mock/pass-through mode (no weights loaded)"
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Auto-confirm missing acceleration wheel installation",
    ),
) -> None:
    """Launch the local OpenAI-compatible N-ATLAS REST API server."""
    import uvicorn

    from natlas.cli.hardware import (
        ensure_asr_dependencies,
        ensure_llama_cpp,
        is_asr_installed,
        is_llama_cpp_installed,
    )
    from natlas.cli.pull import pull_model, resolve_cached_llm
    from natlas.exceptions import AudioProcessingError
    from natlas.server.app import create_app
    from natlas.server.engine import ASREngine, LlamaCppEngine

    console.print(
        f"[bold green]Starting N-ATLAS Local Server on http://{host}:{port}[/bold green]"
    )

    llm_engine = None
    if mock:
        console.print(
            "[yellow]Running server in mock/pass-through mode (explicit --mock flag).[/yellow]"
        )
    else:
        model_path_to_load: str | None = None
        if model and os.path.exists(model):
            model_path_to_load = model
        else:
            cached_model = resolve_cached_llm("n-atlas")
            if cached_model:
                console.print(
                    f"[green]Found cached sovereign model: {cached_model}[/green]"
                )
                model_path_to_load = cached_model
            else:
                console.print(
                    "[bold cyan]No local N-ATLAS model found on disk. Auto-pulling sovereign weights (~4.9 GB)...[/bold cyan]"
                )
                try:
                    pulled_paths = pull_model("n-atlas")
                    if pulled_paths:
                        model_path_to_load = pulled_paths[0]
                except Exception as pull_exc:  # noqa: BLE001
                    console.print(
                        f"[bold yellow]Auto-pull skipped ({pull_exc}). Falling back to mock mode.[/bold yellow]"
                    )

        if model_path_to_load:
            if not is_llama_cpp_installed():
                ensure_llama_cpp(auto_yes=yes)

            try:
                console.print(
                    f"[cyan]Loading LLM model from {model_path_to_load} with {n_gpu_layers} GPU layers...[/cyan]"
                )
                llm_engine = LlamaCppEngine(
                    Path(model_path_to_load), n_gpu_layers=n_gpu_layers
                )
            except ImportError as imp_exc:
                console.print(
                    f"[yellow]{imp_exc}. Running LLM in mock mode (ASR remains active).[/yellow]"
                )
            except Exception as load_exc:  # noqa: BLE001
                console.print(
                    f"[yellow]Failed to load GGUF model: {load_exc}. Running LLM in mock mode.[/yellow]"
                )

    asr_engine = None
    if not is_asr_installed():
        ensure_asr_dependencies(auto_yes=yes)

    try:
        console.print(f"[cyan]Initializing ASR engine on {device}...[/cyan]")
        asr_engine = ASREngine(device=device)
    except AudioProcessingError as exc:
        console.print(f"[bold red]Strict Sovereign ASR Check Failed: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc
    except (ImportError, RuntimeError, OSError) as exc:
        console.print(f"[yellow]ASR engine initialization skipped: {exc}[/yellow]")

    app = create_app(engine=llm_engine, asr_engine=asr_engine)
    uvicorn.run(app, host=host, port=port)
