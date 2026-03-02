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


# ── search (RAG debug) ────────────────────────────────────────────────────────

@app.command()
def search(
    query: str = typer.Argument(..., help="Query to search in the FAISS index"),
    collection: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection to search"
    ),
    top_k: int = typer.Option(
        20, "--top-k", "-k", help="Number of chunks to retrieve"
    ),
):
    """Debug RAG retrieval: show all matching chunks with similarity scores."""
    from langchain_ollama import OllamaEmbeddings
    from langchain_community.vectorstores import FAISS
    from rag import config

    if collection:
        config.COLLECTION = collection

    collection_dir = config.INDEX_DIR / config.COLLECTION
    if not (collection_dir / "index.faiss").exists():
        console.print(f"[red]No index found for collection '{config.COLLECTION}'.[/red]")
        raise typer.Exit(1)

    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)
    vectorstore = FAISS.load_local(
        str(collection_dir), embeddings, allow_dangerous_deserialization=True
    )

    results = vectorstore.similarity_search_with_score(query, k=top_k)

    console.print(f"\n[bold]Query:[/bold] {query}")
    console.print(f"[dim]Collection: {config.COLLECTION} — top {top_k} chunks[/dim]\n")

    for i, (doc, score) in enumerate(results, 1):
        src = Path(doc.metadata.get("source", "?")).name
        page = doc.metadata.get("page", "")
        loc = f"{src}, p.{page}" if page != "" else src
        preview = doc.page_content.replace("\n", " ").strip()[:120]

        # FAISS returns L2 distance — lower = more similar
        bar = "█" * max(1, int((1 - min(score, 2) / 2) * 20))
        console.print(
            f"  [bold cyan]{i:>2}.[/bold cyan] "
            f"[dim]score={score:.4f}[/dim] {bar}  "
            f"[yellow]{loc}[/yellow]"
        )
        console.print(f"      [dim]{preview}…[/dim]\n")


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
        for blob in result["rag_results"]:
            # blob is all chunks joined by "\n\n---\n\n" — split and show each one
            sub_chunks = blob.split("\n\n---\n\n")
            for i, sub in enumerate(sub_chunks, 1):
                sub = sub.strip()
                preview = sub[:300] + "…" if len(sub) > 300 else sub
                console.print(f"[dim]  [{i}/{len(sub_chunks)}] {preview}[/dim]")

    if result.get("web_results"):
        console.print("\n[dim]── Web results ─────────────────────────────[/dim]")
        for blob in result["web_results"]:
            # strip the <!-- saved: ... --> comment if present
            lines = blob.split("\n", 1)
            body = lines[1] if lines[0].startswith("<!--") else blob
            sub_chunks = body.split("\n\n")
            for i, sub in enumerate(sub_chunks, 1):
                sub = sub.strip()
                if not sub:
                    continue
                preview = sub[:300] + "…" if len(sub) > 300 else sub
                console.print(f"[dim]  [{i}] {preview}[/dim]")

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
