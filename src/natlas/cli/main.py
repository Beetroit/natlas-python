"""Main CLI dispatcher for N-ATLAS."""

import typer

from natlas.cli.chat import chat
from natlas.cli.hardware import resolve_cuda_dlls
from natlas.cli.pull import pull
from natlas.cli.serve import serve
from natlas.cli.setup import setup

resolve_cuda_dlls()

app = typer.Typer(
    name="natlas",
    help="N-ATLAS: Sovereign AI Toolkit for Nigerian Languages",
    add_completion=False,
)

app.command("serve")(serve)
app.command("chat")(chat)
app.command("pull")(pull)
app.command("setup")(setup)

if __name__ == "__main__":
    app()
