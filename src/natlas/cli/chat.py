"""Interactive terminal chat REPL for N-ATLAS."""

import httpx
import typer
from rich.console import Console

from natlas.client import Natlas
from natlas.config import LanguageCode
from natlas.exceptions import NatlasError

console = Console()


def chat(
    base_url: str = typer.Option(
        "http://127.0.0.1:8000/v1", "--base-url", "-u", help="N-ATLAS API URL"
    ),
    language: str = typer.Option(
        "yoruba", "--language", "-l", help="Target Nigerian language"
    ),
    tone_enforcement: bool = typer.Option(
        True, "--tone/--no-tone", help="Enforce diacritics and dialect guards"
    ),
) -> None:
    """Launch an interactive multilingual chat session with N-ATLAS in your terminal."""
    client = Natlas(base_url=base_url)

    lang_code: LanguageCode | None = (
        language
        if language in ("yoruba", "hausa", "igbo", "english", "pidgin")
        else None  # type: ignore[assignment]
    )

    console.print(f"[bold green]Connected to N-ATLAS at {base_url}[/bold green]")
    console.print(
        f"[bold cyan]Language mode: {language.upper()} (Tone enforcement: {tone_enforcement})[/bold cyan]"
    )
    console.print("[dim]Type 'exit' or 'quit' to leave the chat.[/dim]\n")

    history: list[dict[str, str]] = []

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")
            if not user_input.strip():
                continue
            if user_input.strip().lower() in ("exit", "quit"):
                break

            history.append({"role": "user", "content": user_input})
            console.print(f"[bold green]N-ATLAS ({language}):[/bold green] ", end="")

            full_reply = ""
            for chunk in client.chat.stream(
                messages=history,
                language=lang_code,
                tone_enforcement=tone_enforcement,
            ):
                content = chunk.choices[0].delta.content or ""
                console.print(content, end="")
                full_reply += content
            console.print("\n")
            history.append({"role": "assistant", "content": full_reply})

        except KeyboardInterrupt:
            break
        except (NatlasError, httpx.HTTPError, OSError) as exc:
            console.print(f"\n[bold red]Error: {exc}[/bold red]\n")
