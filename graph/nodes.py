"""
LangGraph node implementations.

Nodes
-----
orchestrate  — classify the question → set needs_rag / needs_web
rag_agent    — retrieve relevant chunks from local FAISS index
web_agent    — search the web via Tavily (or DuckDuckGo fallback)
synthesize   — merge all context and produce the final answer
"""

import os
from pathlib import Path

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from rag import config
from .state import ResearchState


# ── helpers ───────────────────────────────────────────────────────────────────

def _llm() -> ChatOllama:
    return ChatOllama(model=config.LLM_MODEL, temperature=0)


def _format_docs(docs: list) -> str:
    parts = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        loc = f"{Path(src).name}, p.{page}" if page != "" else Path(src).name
        parts.append(f"[{loc}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ── node: orchestrate ─────────────────────────────────────────────────────────

_ROUTE_PROMPT = """\
You are a research router. Analyze the question and decide which sources to query.

Question: {question}

Available sources:
  RAG  — local documents / knowledge base (use for domain-specific, private, or uploaded docs)
  WEB  — live web search (use for current events, recent news, general internet knowledge)
  BOTH — query both sources
  NONE — no search needed (simple math, greetings, trivial facts already known)

Reply with exactly ONE word: RAG, WEB, BOTH, or NONE."""


def orchestrate(state: ResearchState) -> dict:
    """Decide which agents to invoke based on the user question."""
    llm = _llm()
    prompt = _ROUTE_PROMPT.format(question=state["question"])
    response = llm.invoke(prompt)
    route = response.content.strip().upper()

    # Be lenient: any word containing RAG/WEB is accepted
    needs_rag = "RAG" in route or "BOTH" in route
    needs_web = "WEB" in route or "BOTH" in route

    return {"needs_rag": needs_rag, "needs_web": needs_web}


# ── node: rag_agent ───────────────────────────────────────────────────────────

def rag_agent(state: ResearchState) -> dict:
    """Retrieve relevant document chunks from the local FAISS index."""
    collection_dir = config.INDEX_DIR / config.COLLECTION

    if not collection_dir.exists() or not (collection_dir / "index.faiss").exists():
        return {"rag_results": ["[RAG] No index found. Run `index` first."]}

    try:
        embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)
        vectorstore = FAISS.load_local(
            str(collection_dir), embeddings, allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": config.TOP_K, "fetch_k": config.TOP_K * 3},
        )
        docs = retriever.invoke(state["question"])
        formatted = _format_docs(docs)
        return {"rag_results": [formatted]}
    except Exception as e:
        return {"rag_results": [f"[RAG] Error: {e}"]}


# ── node: web_agent ───────────────────────────────────────────────────────────

def web_agent(state: ResearchState) -> dict:
    """Search the web — Tavily if API key is set, else DuckDuckGo."""
    max_results = int(os.getenv("WEB_MAX_RESULTS", "5"))

    try:
        if os.getenv("TAVILY_API_KEY"):
            from langchain_community.tools.tavily_search import TavilySearchResults
            tool = TavilySearchResults(max_results=max_results)
            results = tool.invoke(state["question"])
            if isinstance(results, list):
                formatted = "\n\n".join(
                    f"[{r.get('url', '?')}]\n{r.get('content', '')}" for r in results
                )
            else:
                formatted = str(results)
        else:
            from langchain_community.tools import DuckDuckGoSearchRun
            tool = DuckDuckGoSearchRun()
            formatted = tool.invoke(state["question"])

        return {"web_results": [formatted]}
    except Exception as e:
        return {"web_results": [f"[WEB] Error: {e}"]}


# ── node: synthesize ──────────────────────────────────────────────────────────

_SYNTH_PROMPT = """\
You are a research assistant. Answer the question using the context below.
If no context is provided, answer from your own knowledge and say so.
Be concise, accurate, and cite sources when available.

{context}

Question: {question}

Answer:"""


def synthesize(state: ResearchState) -> dict:
    """Merge RAG + web results and generate the final answer."""
    parts = []

    if state.get("rag_results"):
        parts.append("## Local Documents\n\n" + "\n\n".join(state["rag_results"]))
    if state.get("web_results"):
        parts.append("## Web Search\n\n" + "\n\n".join(state["web_results"]))

    context = "\n\n---\n\n".join(parts) if parts else "(no additional context)"

    prompt = _SYNTH_PROMPT.format(context=context, question=state["question"])
    llm = _llm()
    response = llm.invoke(prompt)
    return {"final_answer": response.content}
