"""
LangGraph pipeline construction.

Graph topology
--------------

                 ┌──────────────┐
                 │  orchestrate │
                 └──────┬───────┘
                        │ conditional fan-out
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
     rag_agent      web_agent      synthesize ◄── (when neither needed)
          │             │
          └──────┬───────┘
                 ▼
            synthesize ◄── fan-in (waits for all parallel branches)
                 │
                END
"""

from langgraph.graph import StateGraph, START, END

from .state import ResearchState
from .nodes import orchestrate, rag_agent, web_agent, synthesize


def _route(state: ResearchState) -> list[str]:
    """Return the list of next nodes to run (parallel fan-out if both needed)."""
    destinations: list[str] = []
    if state["needs_rag"]:
        destinations.append("rag_agent")
    if state["needs_web"]:
        destinations.append("web_agent")
    return destinations or ["synthesize"]


def build_graph():
    """Compile and return the research graph."""
    g = StateGraph(ResearchState)

    # Register nodes
    g.add_node("orchestrate", orchestrate)
    g.add_node("rag_agent", rag_agent)
    g.add_node("web_agent", web_agent)
    g.add_node("synthesize", synthesize)

    # Entry point
    g.add_edge(START, "orchestrate")

    # Dynamic routing after orchestration
    g.add_conditional_edges(
        "orchestrate",
        _route,
        ["rag_agent", "web_agent", "synthesize"],
    )

    # Both agents converge on synthesize (LangGraph waits for all branches)
    g.add_edge("rag_agent", "synthesize")
    g.add_edge("web_agent", "synthesize")

    g.add_edge("synthesize", END)

    return g.compile()


# Module-level singleton — compiled once, reused across calls
graph = build_graph()
