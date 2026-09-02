from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

from .prompts import CRITIC_PROMPT, SYSTEM_PROMPT
from .state import AgentState, CriticVerdict
from .tools import ALL_TOOLS


LOGGER = logging.getLogger("ashe_agent")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MAX_CRITIC_ATTEMPTS = int(os.getenv("AGENT_MAX_CRITIC_ATTEMPTS", "2"))

_TOOL_NODE = ToolNode(ALL_TOOLS)


def _model_candidates() -> list[str]:
    explicit = os.getenv("OPENAI_MODEL", "").strip()
    defaults = [
        item.strip()
        for item in os.getenv(
            "OPENAI_MODEL_CANDIDATES",
            "gpt-5.1,gpt-5,gpt-4.1,gpt-4o-mini",
        ).split(",")
        if item.strip()
    ]
    ordered = [explicit] + defaults if explicit else defaults
    seen: set[str] = set()
    out: list[str] = []
    for model in ordered:
        if model not in seen:
            out.append(model)
            seen.add(model)
    return out


def _invoke_with_model_fallback(
    messages: list[Any],
    structured_model: type[BaseModel] | None = None,
    bind_tools_to: list[Any] | None = None,
) -> tuple[Any, str]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    errors: dict[str, str] = {}
    for model in _model_candidates():
        llm = ChatOpenAI(model=model, temperature=0, max_retries=0, timeout=30)
        try:
            if structured_model is not None:
                parser = llm.with_structured_output(structured_model)
                return parser.invoke(messages), model
            if bind_tools_to is not None:
                bound = llm.bind_tools(bind_tools_to)
                return bound.invoke(messages), model
            return llm.invoke(messages), model
        except Exception as exc:  # noqa: BLE001
            errors[model] = str(exc)
            continue

    raise RuntimeError(f"All candidate models failed: {errors}")


def _log(node: str, message: str, payload: dict[str, Any] | None = None) -> None:
    data = ""
    if payload is not None:
        try:
            data = " | " + json.dumps(payload, ensure_ascii=True, default=str)
        except Exception:
            data = " | <payload-unserializable>"
    LOGGER.info("node=%s | %s%s", node, message, data)


def _agent_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    _log("agent", "starting", {"message_count": len(messages)})
    response, model = _invoke_with_model_fallback(
        [SystemMessage(content=SYSTEM_PROMPT), *messages], bind_tools_to=ALL_TOOLS
    )
    tool_calls = getattr(response, "tool_calls", None) or []
    _log(
        "agent",
        "completed",
        {"model": model, "tool_calls": [tc.get("name") for tc in tool_calls], "has_final_text": bool(not tool_calls)},
    )
    return {"messages": [response]}


def _log_tools_node(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    for call in tool_calls:
        _log("tools", "invoking", {"tool": call.get("name"), "arguments": call.get("args")})
    result = _TOOL_NODE.invoke(state)
    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage):
            _log("tools", "result", {"tool_call_id": msg.tool_call_id, "content": _truncate(str(msg.content))})
    return result


def _truncate(text: str, limit: int = 400) -> str:
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def _last_ai_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not (getattr(msg, "tool_calls", None) or []):
            content = msg.content
            if isinstance(content, list):
                return "".join(part.get("text", "") for part in content if isinstance(part, dict))
            return str(content)
    return ""


def _critic_view(messages: list[BaseMessage]) -> str:
    """Build the critic's view of the conversation.

    Deliberately NOT a full transcript dump: only the original user question,
    every tool call + full result (accumulated evidence), and the LATEST final
    answer. Earlier rejected draft answers and this node's own past rewrite
    instructions (injected as HumanMessages back into state) are excluded —
    otherwise a critic call reads its own prior verdict's text back as if it
    were fresh conversation content, which both pollutes the "is this
    grounded" judgment and can cause it to re-litigate an issue it was already
    told to stop flagging.
    """
    original_question = next((m for m in messages if isinstance(m, HumanMessage)), None)
    lines: list[str] = []
    if original_question is not None:
        lines.append(f"USER'S ORIGINAL QUESTION: {original_question.content}")

    for msg in messages:
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                calls = ", ".join(f"{tc.get('name')}({tc.get('args')})" for tc in tool_calls)
                lines.append(f"ASSISTANT (tool calls): {calls}")
        elif isinstance(msg, ToolMessage):
            # Full, untruncated content: the critic must see exactly what the
            # agent saw, or it will wrongly flag correct claims as ungrounded
            # just because they were clipped from its own view.
            lines.append(f"TOOL RESULT: {msg.content}")

    lines.append(f"ASSISTANT'S LATEST ANSWER (judge this one): {_last_ai_text(messages)}")
    return "\n".join(lines)


def extract_tool_events(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    pending: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in getattr(msg, "tool_calls", None) or []:
                pending[call["id"]] = {"tool": call["name"], "arguments": call["args"]}
        elif isinstance(msg, ToolMessage):
            meta = pending.get(msg.tool_call_id, {})
            try:
                parsed = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            except json.JSONDecodeError:
                parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}
            events.append(
                {
                    "tool": meta.get("tool", "unknown"),
                    "arguments": meta.get("arguments", {}),
                    "ok": bool(parsed.get("ok")),
                    "result": parsed.get("data"),
                    "sources": parsed.get("sources", []) or [],
                    "error": parsed.get("error"),
                }
            )
    return events


def _dedupe_preserve_order(items: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _critic_node(state: AgentState) -> AgentState:
    attempts = int(state.get("critic_attempts", 0)) + 1
    messages = state["messages"]
    _log("critic", "starting", {"attempt": attempts})

    try:
        verdict, model = _invoke_with_model_fallback(
            [
                SystemMessage(content=CRITIC_PROMPT),
                HumanMessage(content=_critic_view(messages)),
            ],
            structured_model=CriticVerdict,
        )
        _log("critic", "llm verdict", {"model": model, "verdict": verdict.model_dump()})
    except Exception as exc:  # noqa: BLE001
        _log("critic", "llm failed, accepting answer as-is", {"error": str(exc)})
        return {"critic_attempts": attempts, "critic_verdict": {"error": str(exc)}}

    update: dict[str, Any] = {"critic_attempts": attempts, "critic_verdict": verdict.model_dump()}
    passed = verdict.grounded and verdict.advice_free and verdict.clear_for_hr_reader

    if not passed and attempts < MAX_CRITIC_ATTEMPTS:
        issues = "; ".join(verdict.issues) or "unspecified issue"
        _log("critic", "requesting rewrite", {"issues": verdict.issues})
        update["messages"] = [
            HumanMessage(
                content=(
                    "Revise your previous answer. Issues found: "
                    f"{issues}. Address each directly, stay grounded in tool results "
                    "already gathered in this conversation (call more tools only if "
                    "needed), and keep it concise. Output ONLY the corrected answer "
                    "text itself, written as if it were your first response to the "
                    "user's original question — do not mention that this is a "
                    "revision, do not refer to the previous answer or the issues "
                    "found, and do not include any meta-commentary about the "
                    "revision process."
                )
            )
        ]
    elif not passed:
        _log("critic", "attempts exhausted, returning last answer as-is", {"issues": verdict.issues})

    return update


def _route_after_critic(state: AgentState) -> str:
    verdict = state.get("critic_verdict") or {}
    passed = bool(verdict.get("grounded") and verdict.get("advice_free") and verdict.get("clear_for_hr_reader"))
    attempts = int(state.get("critic_attempts", 0))
    if passed or attempts >= MAX_CRITIC_ATTEMPTS or "error" in verdict:
        return END
    return "agent"


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", _log_tools_node)
    graph.add_node("critic", _critic_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": "critic"})
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("critic", _route_after_critic, {"agent": "agent", END: END})

    return graph.compile()


def ask_agent(question: str) -> dict[str, Any]:
    app = build_agent_graph()
    _log("entry", "incoming request", {"question": question})
    result = app.invoke({"messages": [HumanMessage(content=question)], "critic_attempts": 0})

    messages = result["messages"]
    tool_events = extract_tool_events(messages)
    answer = _last_ai_text(messages)

    _log("exit", "completed request", {"answer_chars": len(answer), "tool_calls": [e["tool"] for e in tool_events]})

    return {
        "question": question,
        "answer": answer,
        "sources": _dedupe_preserve_order(s for e in tool_events for s in e.get("sources", [])),
        "tool_calls": tool_events,
        "critic": result.get("critic_verdict"),
    }
