"""Main CLI dispatcher for N-ATLAS."""

import typer

from natlas.cli.chat import chat
from natlas.cli.serve import serve

app = typer.Typer(
    name="natlas",
    help="N-ATLAS: Sovereign AI Toolkit for Nigerian Languages",
    add_completion=False,
)

app.command("serve")(serve)
app.command("chat")(chat)

if __name__ == "__main__":
    app()
