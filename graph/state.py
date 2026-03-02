"""
Shared state flowing through the LangGraph research pipeline.

rag_results and web_results use the `add` reducer so that parallel
branches (rag_agent + web_agent) can write concurrently without
overwriting each other — their lists get concatenated automatically.
"""

from operator import add
from typing import Annotated, Optional
from typing_extensions import TypedDict


class ResearchState(TypedDict):
    # Input
    question: str

    # Routing decision (set by orchestrate node)
    needs_rag: bool
    needs_web: bool

    # Agent outputs — reducer allows parallel fan-in
    rag_results: Annotated[list[str], add]
    web_results: Annotated[list[str], add]

    # Final synthesized answer
    final_answer: Optional[str]
