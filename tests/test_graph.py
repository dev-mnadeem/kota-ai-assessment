"""Offline tests for the answer/critic loop.

The model is replaced by a scripted stand-in, so these tests exercise the real
LangGraph wiring, ToolNode dispatch and SQLite tools without a provider key.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agent import graph
from src.agent.state import CriticVerdict
from src.datasets import ROOT, build

PASS = CriticVerdict(grounded=True, advice_free=True, clear_for_hr_reader=True)
FAIL = CriticVerdict(grounded=False, advice_free=True, clear_for_hr_reader=True, issues=["65% is not cited"])


class ScriptedModel:
    """Returns queued agent replies and critic verdicts in order."""

    def __init__(self, agent_replies, verdicts):
        self.agent_replies = list(agent_replies)
        self.verdicts = list(verdicts)
        self.agent_inputs = []
        self.critic_inputs = []

    def __call__(self, messages, structured_model=None, bind_tools_to=None):
        if structured_model is not None:
            self.critic_inputs.append(messages)
            verdict = self.verdicts.pop(0)
            if isinstance(verdict, Exception):
                raise verdict
            return verdict, "scripted-critic"
        self.agent_inputs.append(messages)
        return self.agent_replies.pop(0), "scripted-agent"


def lookup_call(call_id="call-1", hint="finance"):
    return AIMessage(
        content="",
        tool_calls=[{"id": call_id, "name": "lookup_industry_membership_rate",
                     "args": {"industry_hint": hint, "year": 2024}}],
    )


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    path = tmp_path / "ashe.db"
    build(ROOT / "sources/fixture.json", path)
    monkeypatch.setenv("ASHE_DB_PATH", str(path))
    return path


def install(monkeypatch, model):
    monkeypatch.setattr(graph, "_invoke_with_model_fallback", model)
    return model


def test_tool_result_flows_into_answer_sources_and_trace(snapshot, monkeypatch):
    model = install(monkeypatch, ScriptedModel(
        [lookup_call(), AIMessage(content="In 2024, 65% of finance jobs had a DC pension (ONS ASHE P2).")],
        [PASS],
    ))
    result = graph.ask_agent("What share of finance jobs have a DC pension?")

    assert result["answer"].startswith("In 2024, 65%")
    assert result["sources"] == ["P2 (2024)"]
    assert result["tool_calls"] == [{
        "tool": "lookup_industry_membership_rate",
        "arguments": {"industry_hint": "finance", "year": 2024},
        "ok": True,
        "result": {"industry": "Financial and insurance activities", "year": 2024,
                   "pension_family": "defined contribution", "percent": 65.0},
        "sources": ["P2 (2024)"],
        "error": None,
    }]
    assert result["critic"]["grounded"] is True
    assert len(model.agent_inputs) == 2 and len(model.critic_inputs) == 1


def test_failed_critic_triggers_one_bounded_rewrite(snapshot, monkeypatch):
    model = install(monkeypatch, ScriptedModel(
        [lookup_call(), AIMessage(content="It is about 65%."),
         AIMessage(content="In 2024, 65% of finance jobs had a DC pension (ONS ASHE table P2).")],
        [FAIL, PASS],
    ))
    result = graph.ask_agent("What share of finance jobs have a DC pension?")

    assert result["answer"].endswith("(ONS ASHE table P2).")
    rewrite_request = model.agent_inputs[-1][-1]
    assert isinstance(rewrite_request, HumanMessage)
    assert "65% is not cited" in rewrite_request.content
    critic_view = model.critic_inputs[-1][-1].content
    assert "It is about 65%." not in critic_view, "the critic must judge only the latest draft"
    assert "TOOL RESULT:" in critic_view


def test_rewrites_stop_after_max_attempts(snapshot, monkeypatch):
    model = install(monkeypatch, ScriptedModel(
        [AIMessage(content="draft one"), AIMessage(content="draft two"), AIMessage(content="draft three")],
        [FAIL, FAIL, FAIL],
    ))
    result = graph.ask_agent("A question the critic never accepts")

    assert len(model.critic_inputs) == graph.MAX_CRITIC_ATTEMPTS
    assert result["answer"] == "draft two"
    assert result["critic"]["grounded"] is False


def test_critic_outage_returns_the_draft_with_the_error(snapshot, monkeypatch):
    install(monkeypatch, ScriptedModel([AIMessage(content="draft")], [RuntimeError("provider down")]))
    result = graph.ask_agent("Any question at all")
    assert result["answer"] == "draft"
    assert result["critic"] == {"error": "provider down"}


def test_tool_failures_are_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setenv("ASHE_DB_PATH", str(tmp_path / "missing.db"))
    install(monkeypatch, ScriptedModel([lookup_call(), AIMessage(content="The dataset is not available.")], [PASS]))
    result = graph.ask_agent("What share of finance jobs have a DC pension?")
    assert result["tool_calls"][0]["ok"] is False
    assert "Dataset is missing" in result["tool_calls"][0]["error"]
    assert result["sources"] == []


def test_model_fallback_requires_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        graph._invoke_with_model_fallback([HumanMessage(content="hi")])


def test_model_candidates_put_the_explicit_model_first_without_duplicates(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1")
    monkeypatch.setenv("OPENAI_MODEL_CANDIDATES", "gpt-5, gpt-4.1 ,")
    assert graph._model_candidates() == ["gpt-4.1", "gpt-5"]


def test_extract_tool_events_tolerates_non_json_tool_output():
    from langchain_core.messages import ToolMessage

    messages = [lookup_call("c1"), ToolMessage(content="not json", tool_call_id="c1")]
    assert graph.extract_tool_events(messages)[0]["ok"] is False
