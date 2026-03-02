import os
from pathlib import Path

# Ollama models
EMBED_MODEL: str = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")
LLM_MODEL: str = os.getenv("RAG_LLM_MODEL", "llama3.2")

# FAISS persistence — each collection is a subdirectory
INDEX_DIR: Path = Path(os.getenv("RAG_INDEX_DIR", "./faiss_db"))
COLLECTION: str = os.getenv("RAG_COLLECTION", "default")

# Chunking
CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

# Retrieval
TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
