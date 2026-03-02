"""
Document indexer — loads files into a FAISS vector store.
Adapted from rag-cli (lang-chain-test).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from rich.console import Console

from . import config

console = Console()

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".docx": Docx2txtLoader,
}


def _load_file(path: Path) -> list:
    suffix = path.suffix.lower()
    loader_cls = LOADERS.get(suffix)
    if loader_cls is None:
        console.print(f"  [yellow]Skipping unsupported file: {path.name}[/yellow]")
        return []
    try:
        if loader_cls is TextLoader:
            loader = TextLoader(str(path), autodetect_encoding=True)
        else:
            loader = loader_cls(str(path))
        docs = loader.load()
        console.print(f"  [dim]Loaded[/dim] [bold]{path.name}[/bold] [dim]({len(docs)} pages/chunks)[/dim]")
        return docs
    except Exception as e:
        console.print(f"  [red]Error loading {path.name}: {e}[/red]")
        return []


def load_documents(path: Path) -> list:
    if path.is_file():
        return _load_file(path)
    docs = []
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in LOADERS)
    if not files:
        console.print(f"[yellow]No supported files found in {path}[/yellow]")
        return []
    for f in files:
        docs.extend(_load_file(f))
    return docs


def index(path: Path, collection: str | None = None) -> int:
    collection = collection or config.COLLECTION
    collection_dir = config.INDEX_DIR / collection

    console.print(f"\n[bold blue]Indexing[/bold blue] {path}")

    docs = load_documents(path)
    if not docs:
        console.print("[red]No documents loaded — nothing to index.[/red]")
        return 0

    console.print(f"\n[green]Loaded {len(docs)} raw chunks[/green]")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    console.print(f"[green]Split into {len(chunks)} text chunks[/green]")

    console.print(
        f"\n[bold]Embedding with [cyan]{config.EMBED_MODEL}[/cyan]"
        " (may take a while the first time)...[/bold]"
    )

    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)

    if collection_dir.exists():
        vectorstore = FAISS.load_local(
            str(collection_dir), embeddings, allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
        console.print(f"[dim]Merged into existing collection '{collection}'[/dim]")
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)

    collection_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(collection_dir))

    total = vectorstore.index.ntotal
    sources = sorted({str(Path(c.metadata.get("source", "?")).name) for c in chunks})
    meta = {
        "chunks": total,
        "sources": sources,
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    (collection_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    console.print(
        f"\n[bold green]✓ Indexed {len(chunks)} chunks[/bold green]"
        f" → collection [cyan]'{collection}'[/cyan]"
        f" ([dim]{collection_dir}[/dim])"
    )
    return len(chunks)


def list_collections() -> list[dict]:
    if not config.INDEX_DIR.exists():
        return []
    collections = []
    for d in sorted(config.INDEX_DIR.iterdir()):
        if d.is_dir() and (d / "index.faiss").exists():
            meta_file = d / "meta.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                collections.append({"name": d.name, **meta})
            else:
                collections.append({"name": d.name})
    return collections
