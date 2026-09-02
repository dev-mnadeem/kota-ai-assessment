"""ASHE pension agent: a minimal LangGraph tool-calling agent + critic loop.

Public API: `ask_agent(question) -> dict`. See graph.py for the graph itself,
prompts.py for both LLM roles' instructions, tools.py for the SQLite-backed
LangChain tools, and state.py for the graph state / structured-output schemas.
"""

from __future__ import annotations

from .graph import ask_agent, build_agent_graph
from .state import AgentState, CriticVerdict
from .tools import ALL_TOOLS, SQLiteTools, ToolResult

__all__ = [
    "ask_agent",
    "build_agent_graph",
    "AgentState",
    "CriticVerdict",
    "ALL_TOOLS",
    "SQLiteTools",
    "ToolResult",
]
