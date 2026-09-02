from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AccessDecision:
    data_type: str
    strategy: str
    why: str
    latency_cost: str
    failure_mode_if_wrong: str


ACCESS_DECISIONS = [
    AccessDecision(
        data_type="ONS ASHE observation data (2009-2024)",
        strategy="cache",
        why=(
            "The data updates annually, is large, and is used repeatedly across queries; "
            "serving from local SQLite is fast and deterministic for evals."
        ),
        latency_cost="Low (local disk read, typically milliseconds).",
        failure_mode_if_wrong="Stale answers after a new ONS release if cache is not refreshed.",
    ),
    AccessDecision(
        data_type="Column aliases and taxonomy mapping (industry aliases, pension family mapping)",
        strategy="hardcode",
        why=(
            "Small stable mapping layer needed for robust dimension-value resolution despite naming variation "
            "across years. Implemented in src/agent/tools.py; kept deliberately tiny so it stays reviewable data, "
            "not agent control-flow."
        ),
        latency_cost="Near-zero.",
        failure_mode_if_wrong=(
            "A tool argument can resolve to the wrong dimension value and produce a plausible but incorrect answer."
        ),
    ),
    AccessDecision(
        data_type="Safety boundary text (no recommendations)",
        strategy="LLM-driven (system prompt + structured self-check)",
        why=(
            "This is a hard policy constraint from the assessment, but enforcing it via regex/keyword matching "
            "proved brittle and duplicated across the codebase. It is now carried entirely by the agent's system "
            "prompt (src/agent/prompts.py) plus a dedicated critic node (src/agent/graph.py) that judges "
            "groundedness, advice-free-ness, and clarity as structured LLM output and triggers a bounded rewrite "
            "loop on failure."
        ),
        latency_cost="One extra LLM call per turn (the critic), plus up to one rewrite.",
        failure_mode_if_wrong="Agent might drift into advisory recommendations if the critic itself misjudges an answer.",
    ),
    AccessDecision(
        data_type="Latest ONS release metadata",
        strategy="live (optional)",
        why=(
            "Useful for freshness checks only; not required in the request path once cache exists."
        ),
        latency_cost="Medium (network call, external dependency).",
        failure_mode_if_wrong=(
            "If used in request path, latency spikes and outages can block answers; "
            "if skipped forever, cache may silently age."
        ),
    ),
]
