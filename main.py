"""
multi-agent-researcher — CLI

Commands
--------
  index   <path>        Index a file or folder into the local FAISS knowledge base
  list                  List indexed collections
  research <question>   Ask a question (routes to RAG, Web, or both automatically)
  ask                   Interactive research REPL
"""

from pathlib import Path
from typing import Optional

from graph.state import ResearchState

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

load_dotenv()

app = typer.Typer(
    name="researcher",
    help="Multi-agent researcher — Orchestrator + RAG + Web, powered by LangGraph & Ollama.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

_OUTPUT_HELP = (
    "Output mode: 'both' (default) saves MD files and prints to CLI, "
    "'cli' prints only, 'md' saves only."
)


# ── index ─────────────────────────────────────────────────────────────────────

@app.command()
def index(
    path: Path = typer.Argument(..., help="PDF, text file, or folder to index"),
    collection: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection name (default: 'default')"
    ),
):
    """Index documents into the local FAISS knowledge base."""
    from rag import config
    from rag.indexer import index as do_index

    if collection:
        config.COLLECTION = collection

    if not path.exists():
        console.print(f"[red]Path not found: {path}[/red]")
        raise typer.Exit(1)

    do_index(path, collection)


# ── list ──────────────────────────────────────────────────────────────────────

@app.command(name="list")
def list_collections():
    """List all indexed collections."""
    from rag.indexer import list_collections as get_cols

    cols = get_cols()
    if not cols:
        console.print("[yellow]No collections found. Run `index` first.[/yellow]")
        return

    console.print("\n[bold]Indexed collections:[/bold]")
    for col in cols:
        chunks = col.get("chunks", "?")
        sources = ", ".join(col.get("sources", []))
        updated = col.get("updated", "")[:10]
        console.print(
            f"  [cyan]{col['name']}[/cyan]  — {chunks} chunks"
            f"  [dim]({sources}) · {updated}[/dim]"
        )


# ── research ──────────────────────────────────────────────────────────────────

@app.command()
def research(
    question: str = typer.Argument(..., help="Question to research"),
    collection: Optional[str] = typer.Option(
        None, "--collection", "-c", help="RAG collection to query"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show routing decisions and agent outputs"
    ),
    output: str = typer.Option(
        "both", "--output", "-o", help=_OUTPUT_HELP
    ),
):
    """Ask a question — the orchestrator decides which agents to use."""
    from rag import config
    from graph.builder import graph

    output = _validate_output(output)

    if collection:
        config.COLLECTION = collection

    initial_state: ResearchState = {
        "question": question,
        "needs_rag": False,
        "needs_web": False,
        "rag_results": [],
        "web_results": [],
        "output_mode": output,
        "final_answer": None,
    }

    if output != "md":
        console.print(f"\n[bold blue]Q:[/bold blue] {question}\n")

    with console.status("[dim]Thinking...[/dim]"):
        result = graph.invoke(initial_state)

    if output != "md":
        if verbose:
            _print_verbose(result)
        answer = result.get("final_answer", "No answer generated.")
        console.print(Panel(Markdown(answer), title="[bold green]Answer[/bold green]", border_style="green"))
    else:
        console.print(f"[dim]Saved to answers/ and web_results/[/dim]")


def _validate_output(value: str) -> str:
    if value not in ("both", "cli", "md"):
        console.print(f"[red]Invalid --output value '{value}'. Use: both, cli, md.[/red]")
        raise typer.Exit(1)
    return value


def _print_verbose(result: dict) -> None:
    needs_rag = result.get("needs_rag", False)
    needs_web = result.get("needs_web", False)

    route_label = (
        "RAG + WEB" if needs_rag and needs_web
        else "RAG only" if needs_rag
        else "WEB only" if needs_web
        else "no search"
    )
    console.print(f"[dim]Route: {route_label}[/dim]")

    if result.get("rag_results"):
        console.print("\n[dim]── RAG context ─────────────────────────────[/dim]")
        for chunk in result["rag_results"]:
            console.print(f"[dim]{chunk[:400]}…[/dim]" if len(chunk) > 400 else f"[dim]{chunk}[/dim]")

    if result.get("web_results"):
        console.print("\n[dim]── Web results ─────────────────────────────[/dim]")
        for chunk in result["web_results"]:
            console.print(f"[dim]{chunk[:400]}…[/dim]" if len(chunk) > 400 else f"[dim]{chunk}[/dim]")

    console.print()


# ── ask (interactive REPL) ────────────────────────────────────────────────────

@app.command()
def ask(
    collection: Optional[str] = typer.Option(
        None, "--collection", "-c", help="RAG collection to query"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show routing decisions and agent outputs"
    ),
    output: str = typer.Option(
        "both", "--output", "-o", help=_OUTPUT_HELP
    ),
):
    """Interactive research REPL — ask multiple questions in sequence."""
    from rag import config
    from graph.builder import graph

    output = _validate_output(output)

    if collection:
        config.COLLECTION = collection

    console.print(
        Panel(
            "[bold]Multi-Agent Researcher — Interactive Mode[/bold]\n\n"
            f"Collection : [cyan]{config.COLLECTION}[/cyan]\n"
            f"LLM        : [cyan]{config.LLM_MODEL}[/cyan]\n"
            f"Embeddings : [cyan]{config.EMBED_MODEL}[/cyan]\n"
            f"Output     : [cyan]{output}[/cyan]\n\n"
            "[dim]Type a question and press Enter. 'exit' or Ctrl-C to quit.[/dim]",
            border_style="blue",
        )
    )

    while True:
        try:
            question = console.input("\n[bold blue]>[/bold blue] ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q", ":q"):
            break

        initial_state: ResearchState = {
            "question": question,
            "needs_rag": False,
            "needs_web": False,
            "rag_results": [],
            "web_results": [],
            "output_mode": output,
            "final_answer": None,
        }

        with console.status("[dim]Thinking...[/dim]"):
            result = graph.invoke(initial_state)

        if output != "md":
            if verbose:
                _print_verbose(result)
            answer = result.get("final_answer", "No answer generated.")
            console.print(Panel(Markdown(answer), title="[bold green]Answer[/bold green]", border_style="green"))
        else:
            console.print(f"[dim]Saved to answers/ and web_results/[/dim]")

    console.print("\n[dim]Bye.[/dim]")


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
