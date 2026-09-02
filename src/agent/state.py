"""Graph state and structured LLM-output schemas for the agent."""

from __future__ import annotations

from typing import Any

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class CriticVerdict(BaseModel):
    grounded: bool = Field(
        description=(
            "Every specific number/year/percentage/trend the answer states as true is directly "
            "supported by a tool result in this conversation; nothing invented or remembered as a "
            "figure. Does not apply to an answer that asks a clarifying question instead of "
            "answering, nor to well-established contextual labels (calling this 'pension'/'ONS'/"
            "'ASHE'/'UK' data is ALWAYS grounded — that's the assistant's own known domain, not a "
            "claim needing tool proof) even if that exact wording isn't in a specific tool's JSON "
            "output. Only fail this for an unsupported number/year/trend, never for reasonable "
            "domain framing."
        )
    )
    advice_free: bool = Field(
        description=(
            "No directive recommendation language; if the question asked for a decision, the "
            "answer acknowledges it, shares data actually retrieved via a tool call in this "
            "conversation, and states the decision needs professional guidance. A bare deferral "
            "with no retrieved data fails this check."
        )
    )
    clear_for_hr_reader: bool = Field(
        description="Plain English for a non-technical HR reader, names the data source naturally."
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Concrete, actionable problems if any check fails; empty if all pass.",
    )


class AgentState(MessagesState):
    critic_attempts: int
    critic_verdict: dict[str, Any]
