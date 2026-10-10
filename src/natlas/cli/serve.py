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
) -> None:
    """Launch the local OpenAI-compatible N-ATLAS REST API server."""
    import uvicorn

    from natlas.server.app import create_app
    from natlas.server.engine import ASREngine, LlamaCppEngine

    console.print(
        f"[bold green]Starting N-ATLAS Local Server on http://{host}:{port}[/bold green]"
    )

    llm_engine = None
    if model and os.path.exists(model):
        console.print(
            f"[cyan]Loading LLM model from {model} with {n_gpu_layers} GPU layers...[/cyan]"
        )
        llm_engine = LlamaCppEngine(Path(model), n_gpu_layers=n_gpu_layers)
    else:
        console.print(
            "[yellow]Running server in mock/pass-through mode (no GGUF model loaded).[/yellow]"
        )

    asr_engine = None
    try:
        console.print(f"[cyan]Initializing ASR engine on {device}...[/cyan]")
        asr_engine = ASREngine(device=device)
    except (ImportError, RuntimeError, OSError) as exc:
        console.print(f"[yellow]ASR engine initialization skipped: {exc}[/yellow]")

    app = create_app(engine=llm_engine, asr_engine=asr_engine)
    uvicorn.run(app, host=host, port=port)
