"""CLI command and helper functions for pulling sovereign N-ATLAS models."""

import os
from typing import Final

import typer
from dotenv import find_dotenv, load_dotenv
from rich.console import Console

console = Console()

DEFAULT_LLM_REPO: Final[str] = "QuantFactory/N-ATLaS-GGUF"
DEFAULT_LLM_FILE: Final[str] = "N-ATLaS.Q4_K_M.gguf"

MODEL_ALIASES: Final[dict[str, tuple[str, str | None]]] = {
    "n-atlas": (DEFAULT_LLM_REPO, DEFAULT_LLM_FILE),
    "n-atlas:q4": (DEFAULT_LLM_REPO, "N-ATLaS.Q4_K_M.gguf"),
    "n-atlas:q4_k_m": (DEFAULT_LLM_REPO, "N-ATLaS.Q4_K_M.gguf"),
    "n-atlas:q8": (DEFAULT_LLM_REPO, "N-ATLaS.Q8_0.gguf"),
    "n-atlas:q8_0": (DEFAULT_LLM_REPO, "N-ATLaS.Q8_0.gguf"),
    "asr:yoruba": ("NCAIR1/Yoruba-ASR", None),
    "asr:hausa": ("NCAIR1/Hausa-ASR", None),
    "asr:igbo": ("NCAIR1/Igbo-ASR", None),
    "asr:english": ("NCAIR1/NigerianAccentedEnglish", None),
}


def resolve_cached_llm(target: str = "n-atlas") -> str | None:
    """Check if the target GGUF model is already cached locally.

    Returns the absolute path to the cached file if present, else None.
    """
    from huggingface_hub import try_to_load_from_cache

    norm_target = target.strip().lower()
    repo, filename = MODEL_ALIASES.get(
        norm_target, (DEFAULT_LLM_REPO, DEFAULT_LLM_FILE)
    )

    if filename is None:
        return None

    cached_path = try_to_load_from_cache(repo, filename)
    if isinstance(cached_path, str) and os.path.exists(cached_path):
        return cached_path
    return None


def pull_model(target: str = "n-atlas") -> list[str]:
    """Download the specified model target and return local paths."""
    from huggingface_hub import hf_hub_download, snapshot_download

    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path)

    token = os.environ.get("HF_TOKEN")
    norm_target = target.strip().lower()
    downloaded_paths: list[str] = []

    # Target: All ASR models
    if norm_target == "asr":
        console.print(
            "[bold cyan]Pulling all 4 official NCAIR1 ASR checkpoints...[/bold cyan]"
        )
        for asr_alias in ["asr:yoruba", "asr:hausa", "asr:igbo", "asr:english"]:
            repo, _ = MODEL_ALIASES[asr_alias]
            console.print(f"[cyan]Downloading {repo}...[/cyan]")
            path = snapshot_download(repo_id=repo, token=token)
            downloaded_paths.append(str(path))
        return downloaded_paths

    # Target: All models (LLM + all ASR)
    if norm_target == "all":
        console.print(
            "[bold cyan]Pulling complete N-ATLAS sovereign stack (LLM + ASR)...[/bold cyan]"
        )
        llm_paths = pull_model("n-atlas")
        asr_paths = pull_model("asr")
        return llm_paths + asr_paths

    # Target: Known alias
    if norm_target in MODEL_ALIASES:
        repo, filename = MODEL_ALIASES[norm_target]
        if filename:
            console.print(
                f"[bold cyan]Pulling {filename} from {repo} (~4.9 GB)...[/bold cyan]"
            )
            file_path = hf_hub_download(
                repo_id=repo,
                filename=filename,
                token=token,
            )
            downloaded_paths.append(str(file_path))
            console.print(f"[green]Successfully cached: {file_path}[/green]")
        else:
            console.print(f"[bold cyan]Pulling ASR checkpoint {repo}...[/bold cyan]")
            dir_path = snapshot_download(repo_id=repo, token=token)
            downloaded_paths.append(str(dir_path))
            console.print(f"[green]Successfully cached: {dir_path}[/green]")
        return downloaded_paths

    # Target: Custom Hugging Face format (repo or repo:filename)
    if "/" in target:
        if ":" in target:
            repo_part, file_part = target.split(":", 1)
            console.print(
                f"[bold cyan]Pulling {file_part} from {repo_part}...[/bold cyan]"
            )
            path = hf_hub_download(
                repo_id=repo_part,
                filename=file_part,
                token=token,
            )
            downloaded_paths.append(str(path))
        else:
            console.print(f"[bold cyan]Pulling repository {target}...[/bold cyan]")
            path = snapshot_download(repo_id=target, token=token)
            downloaded_paths.append(str(path))
        return downloaded_paths

    console.print(
        f"[red]Unknown target '{target}'. Available targets: {', '.join(MODEL_ALIASES.keys())}, asr, all[/red]"
    )
    raise typer.Exit(code=1)


def pull(
    target: str = typer.Argument(
        "n-atlas",
        help="Model target (e.g. n-atlas, n-atlas:q8, asr, asr:yoruba, all)",
    ),
) -> None:
    """Pull sovereign N-ATLAS models from Hugging Face into local cache."""
    try:
        paths = pull_model(target)
        console.print(
            f"[bold green]Ready! Successfully pulled {len(paths)} model component(s).[/bold green]"
        )
    except Exception as exc:
        console.print(f"[bold red]Failed to pull model '{target}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc
