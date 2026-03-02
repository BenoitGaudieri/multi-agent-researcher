import os
from pathlib import Path

# Ollama models
EMBED_MODEL: str = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")
LLM_MODEL: str = os.getenv("RAG_LLM_MODEL", "llama3.2")

# FAISS persistence — each collection is a subdirectory
INDEX_DIR: Path = Path(os.getenv("RAG_INDEX_DIR", "./faiss_db"))
COLLECTION: str = os.getenv("RAG_COLLECTION", "default")

# Chunking — keep small for documents with many short sections
CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "200"))
CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "40"))

# Retrieval
TOP_K: int = int(os.getenv("RAG_TOP_K", "10"))
# Search type: "similarity" (precise) or "mmr" (diverse but may miss relevant chunks)
SEARCH_TYPE: str = os.getenv("RAG_SEARCH_TYPE", "similarity")

# Web results persistence
WEB_RESULTS_DIR: Path = Path(os.getenv("WEB_RESULTS_DIR", "./web_results"))

# Final answers persistence
ANSWERS_DIR: Path = Path(os.getenv("ANSWERS_DIR", "./answers"))
