from __future__ import annotations

import dataclasses
import os
import re
import sqlite3
from contextlib import contextmanager
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field


INDUSTRY_ALIASES = {
    "finance": "Financial and insurance activities",
    "financial": "Financial and insurance activities",
    "retail": "Wholesale and retail trade; repair of motor vehicles and motorcycles",
    "health": "Human health and social work activities",
    "healthcare": "Human health and social work activities",
    "tech": "Information and communication",
    "it": "Information and communication",
    "construction": "Construction",
    "education": "Education",
    "manufacturing": "Manufacturing",
    "public administration": "Public administration and defence; compulsory social security",
}

# Small, stable data-quality constants for dimension-value resolution — not
# agent control-flow. See README.md, "Retrieve / cache / hardcode decisions", item 2.
STOPWORDS = {
    "the", "and", "of", "in", "for", "to", "by", "our", "sector", "industry",
    "industries", "employees", "employee", "jobs", "company", "companies",
}


@dataclass
class ToolResult:
    tool_name: str
    ok: bool
    data: dict[str, Any]
    sources: list[str]
    error: str | None = None


class SQLiteTools:
    """Semantic, read-only tools over ASHE SQLite data.

    These tools are designed for LLM agents:
    - arguments are human-oriented rather than SQL-oriented,
    - output is normalized JSON payloads,
    - every metric tool emits source tags for answer grounding.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        path = Path(self.db_path).resolve()
        if not path.is_file():
            raise FileNotFoundError("Dataset is missing; build it before querying the agent.")
        conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON")
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _normalize_key(value: str) -> str:
        value = value.lower()
        value = re.sub(r"[^a-z0-9\s]", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    def _latest_year(self, table_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(year) AS max_year FROM observations WHERE table_id = ?", (table_id,)
            ).fetchone()
        if not row or row["max_year"] is None:
            raise ValueError(f"No year coverage found for table {table_id}")
        return int(row["max_year"])

    def _distinct_group_values(self, table_id: str, group_type: str, year: int | None) -> list[str]:
        where = ["table_id = ?", "primary_group_type = ?", "earnings_band IS NULL"]
        params: list[Any] = [table_id, group_type]
        if year is not None:
            where.append("year = ?")
            params.append(year)

        query = (
            "SELECT DISTINCT primary_group_value FROM observations "
            f"WHERE {' AND '.join(where)} AND primary_group_value IS NOT NULL"
        )
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        values = [str(row["primary_group_value"]).strip() for row in rows if row["primary_group_value"]]

        dedup: dict[str, str] = {}
        for value in values:
            key = self._normalize_key(value)
            if key not in dedup:
                dedup[key] = value
        return sorted(dedup.values())

    def _canonical_pension_family(self, raw_label: str) -> str:
        key = self._normalize_key(raw_label)
        if "defined contribution" in key:
            return "defined contribution"
        if "defined benefit" in key:
            return "defined benefit"
        if "group personal pension" in key:
            return "group personal pension"
        if "group stakeholder" in key:
            return "group stakeholder pension"
        if "self invested" in key:
            return "group self-invested personal pension"
        if "unknown pension" in key:
            return "unknown pension provision"
        if "no pension" in key:
            return "no pension provision"
        return key

    def _resolve_dimension_value(
        self, table_id: str, group_type: str, hint: str, year: int | None = None
    ) -> tuple[str | None, list[str]]:
        """Resolve a free-text hint to a canonical dimension value.

        Returns (resolved_value, candidates). `candidates` is populated only when
        resolution fails or is ambiguous, so the caller can surface it to the LLM
        for self-correction instead of guessing.
        """
        normalized_hint = self._normalize_key(hint)

        if group_type == "industry":
            for alias, canonical in INDUSTRY_ALIASES.items():
                if re.search(rf"\b{re.escape(alias)}\b", normalized_hint):
                    return canonical, []

        candidates = self._distinct_group_values(table_id, group_type, year)
        if group_type == "industry":
            candidates = [c for c in candidates if not self._normalize_key(c).startswith("all ")]
        if group_type == "employer_size":
            candidates = [c for c in candidates if "employee" in self._normalize_key(c)]
            if normalized_hint in {"small", "small employers"}:
                return "1 to 12 employees", []
            if normalized_hint in {"large", "large employers"}:
                return "1000 or more employees", []

        if not candidates:
            return None, []

        exact = next((c for c in candidates if self._normalize_key(c) == normalized_hint), None)
        if exact:
            return exact, []

        hint_tokens = {t for t in normalized_hint.split() if t not in STOPWORDS}
        scored: list[tuple[int, str]] = []
        for candidate in candidates:
            token_set = {t for t in self._normalize_key(candidate).split() if t not in STOPWORDS}
            score = len(hint_tokens & token_set)
            if score > 0:
                scored.append((score, candidate))

        if not scored:
            return None, candidates[:8]

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top_score = scored[0][0]
        tied = [c for s, c in scored if s == top_score]
        if len(tied) == 1:
            return tied[0], []
        return None, tied[:5]

    @staticmethod
    def _not_found_error(label: str, hint: str, candidates: list[str]) -> str:
        if candidates:
            return (
                f"Could not resolve {label} from '{hint}'. Did you mean: {candidates}? "
                "Call list_dimension_values to see all options, then retry with the exact value."
            )
        return f"Could not resolve {label} from '{hint}'. Call list_dimension_values to see all options."

    def _industry_percent(self, industry: str, year: int, family: str | None) -> float | None:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT pension_type, value_num, suppression_flag
                FROM observations
                WHERE table_id = 'P2'
                  AND year = ?
                  AND primary_group_type = 'industry'
                  AND lower(primary_group_value) = lower(?)
                  AND earnings_band IS NULL
                  AND measure_type = 'percent_jobs'
                """,
                (year, industry),
            ).fetchall()

        usable = [
            (self._canonical_pension_family(row["pension_type"]), float(row["value_num"]))
            for row in rows
            if row["value_num"] is not None and row["suppression_flag"] is None
        ]
        if not usable:
            return None

        if family and family != "overall participation":
            vals = [value for fam, value in usable if fam == family]
            return sum(vals) if vals else None

        no_pension = [value for fam, value in usable if fam == "no pension provision"]
        if not no_pension:
            return None
        return 100.0 - sum(no_pension)

    def _canonical_size_label(self, raw: str) -> str:
        key = self._normalize_key(raw)
        if "1" in key and "12" in key and "employee" in key:
            return "1-12 employees"
        if "1000" in key and "employee" in key:
            return "1000+ employees"
        if "13" in key and "99" in key and "employee" in key:
            return "13-99 employees"
        if "100" in key and "499" in key and "employee" in key:
            return "100-499 employees"
        if "500" in key and "999" in key and "employee" in key:
            return "500-999 employees"
        return raw

    def list_tables(self) -> ToolResult:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT table_id, MIN(year) AS min_year, MAX(year) AS max_year, COUNT(*) AS rows
                FROM observations
                GROUP BY table_id
                ORDER BY table_id
                """
            ).fetchall()
        payload = [
            {
                "table_id": row["table_id"],
                "min_year": row["min_year"],
                "max_year": row["max_year"],
                "row_count": row["rows"],
            }
            for row in rows
        ]
        return ToolResult("list_tables", True, {"tables": payload}, ["data/ashe.db"])

    def describe_schema(self, table_id: str) -> ToolResult:
        table_id = table_id.upper().strip()
        with self._connect() as conn:
            dims = conn.execute(
                """
                SELECT DISTINCT primary_group_type
                FROM observations
                WHERE table_id = ?
                ORDER BY primary_group_type
                """,
                (table_id,),
            ).fetchall()
            measures = conn.execute(
                """
                SELECT DISTINCT measure_type
                FROM observations
                WHERE table_id = ?
                ORDER BY measure_type
                """,
                (table_id,),
            ).fetchall()
            years = conn.execute(
                """
                SELECT MIN(year) AS min_year, MAX(year) AS max_year
                FROM observations
                WHERE table_id = ?
                """,
                (table_id,),
            ).fetchone()
            segments = conn.execute(
                """
                SELECT DISTINCT segment
                FROM observations
                WHERE table_id = ? AND segment IS NOT NULL
                ORDER BY segment
                """,
                (table_id,),
            ).fetchall()
            work_patterns = conn.execute(
                """
                SELECT DISTINCT work_pattern
                FROM observations
                WHERE table_id = ? AND work_pattern IS NOT NULL
                ORDER BY work_pattern
                """,
                (table_id,),
            ).fetchall()

        if years is None or years["max_year"] is None:
            return ToolResult("describe_schema", False, {}, [], error=f"Unknown table_id: {table_id}")

        return ToolResult(
            "describe_schema",
            True,
            {
                "table_id": table_id,
                "dimensions": [row["primary_group_type"] for row in dims],
                "measure_types": [row["measure_type"] for row in measures],
                "min_year": years["min_year"],
                "max_year": years["max_year"],
                "segment_values": [row["segment"] for row in segments],
                "work_pattern_values": [row["work_pattern"] for row in work_patterns],
            },
            [],
        )

    def list_dimension_values(
        self, table_id: str, dimension: str, year: int | None = None, limit: int = 50
    ) -> ToolResult:
        table_id = table_id.upper().strip()
        dim = dimension.strip().lower()

        if dim == "pension_type":
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT DISTINCT pension_type
                    FROM observations
                    WHERE table_id = ?
                    ORDER BY pension_type
                    LIMIT ?
                    """,
                    (table_id, int(limit)),
                ).fetchall()
            values = [row["pension_type"] for row in rows]
        else:
            values = self._distinct_group_values(table_id, dim, year)
            values = values[: int(limit)]

        return ToolResult(
            "list_dimension_values",
            True,
            {"table_id": table_id, "dimension": dim, "year": year, "values": values},
            [],
        )

    def lookup_industry_membership_rate(
        self, industry_hint: str, pension_family: str = "defined contribution", year: int | None = None
    ) -> ToolResult:
        target_year = int(year) if year is not None else self._latest_year("P2")
        family = pension_family.strip().lower()
        industry, candidates = self._resolve_dimension_value("P2", "industry", industry_hint, target_year)
        if not industry:
            return ToolResult(
                "lookup_industry_membership_rate",
                False,
                {},
                [f"P2 ({target_year})"],
                error=self._not_found_error("industry", industry_hint, candidates),
            )

        value = self._industry_percent(industry, target_year, family)
        if value is None:
            return ToolResult(
                "lookup_industry_membership_rate",
                False,
                {"industry": industry, "year": target_year, "pension_family": family},
                [f"P2 ({target_year})"],
                error="No usable value found for this slice.",
            )

        return ToolResult(
            "lookup_industry_membership_rate",
            True,
            {
                "industry": industry,
                "year": target_year,
                "pension_family": family,
                "percent": round(value, 1),
            },
            [f"P2 ({target_year})"],
        )

    def industry_participation_trend(
        self,
        industry_hint: str,
        since_year: int = 2019,
        pension_family: str = "overall participation",
    ) -> ToolResult:
        latest_year = self._latest_year("P2")
        start_year = int(since_year)
        industry, candidates = self._resolve_dimension_value("P2", "industry", industry_hint, latest_year)
        if not industry:
            return ToolResult(
                "industry_participation_trend",
                False,
                {"since_year": start_year, "latest_year": latest_year},
                [f"P2 ({start_year}, {latest_year})"],
                error=self._not_found_error("industry", industry_hint, candidates),
            )

        family = pension_family.strip().lower()
        start = self._industry_percent(industry, start_year, family)
        end = self._industry_percent(industry, latest_year, family)
        if start is None or end is None:
            return ToolResult(
                "industry_participation_trend",
                False,
                {
                    "industry": industry,
                    "pension_family": family,
                    "since_year": start_year,
                    "latest_year": latest_year,
                },
                [f"P2 ({start_year}, {latest_year})"],
                error="Could not retrieve comparable values for both years.",
            )

        delta = end - start
        direction = "up" if delta > 0 else "down" if delta < 0 else "flat"
        return ToolResult(
            "industry_participation_trend",
            True,
            {
                "industry": industry,
                "pension_family": family,
                "since_year": start_year,
                "latest_year": latest_year,
                "start_percent": round(start, 1),
                "end_percent": round(end, 1),
                "delta_percentage_points": round(delta, 1),
                "direction": direction,
            },
            [f"P2 ({start_year}, {latest_year})"],
        )

    def _contribution_distribution(self, year: int, size_group: str) -> dict[str, float]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT pension_type, value_num, suppression_flag
                FROM observations
                WHERE table_id = 'P12'
                  AND year = ?
                  AND primary_group_type = 'employer_size'
                  AND lower(primary_group_value) = lower(?)
                  AND earnings_band IS NULL
                  AND measure_type = 'percent_jobs'
                """,
                (year, size_group),
            ).fetchall()

        dist: dict[str, float] = {}
        for row in rows:
            if row["value_num"] is None or row["suppression_flag"] is not None:
                continue
            dist[self._normalize_key(str(row["pension_type"]))] = float(row["value_num"])
        return dist

    @staticmethod
    def _sum_high_bands(dist: dict[str, float]) -> float:
        total = 0.0
        for label, value in dist.items():
            if "8" in label and "10" in label:
                total += value
            elif "10" in label and "12" in label:
                total += value
            elif "12" in label and "15" in label:
                total += value
            elif "15" in label and "20" in label:
                total += value
            elif "20" in label and "over" in label:
                total += value
        return total

    def compare_contribution_bands_by_size(
        self,
        year: int | None = None,
        small_group_hint: str = "1 to 12 employees",
        large_group_hint: str = "1000 or more employees",
    ) -> ToolResult:
        target_year = int(year) if year is not None else self._latest_year("P12")
        small_group, small_candidates = self._resolve_dimension_value(
            "P12", "employer_size", small_group_hint, target_year
        )
        large_group, large_candidates = self._resolve_dimension_value(
            "P12", "employer_size", large_group_hint, target_year
        )
        if not small_group or not large_group:
            missing = []
            if not small_group:
                missing.append(self._not_found_error("employer size", small_group_hint, small_candidates))
            if not large_group:
                missing.append(self._not_found_error("employer size", large_group_hint, large_candidates))
            return ToolResult(
                "compare_contribution_bands_by_size",
                False,
                {"year": target_year},
                [f"P12 ({target_year})"],
                error=" ".join(missing),
            )

        small_dist = self._contribution_distribution(target_year, small_group)
        large_dist = self._contribution_distribution(target_year, large_group)
        if not small_dist or not large_dist:
            return ToolResult(
                "compare_contribution_bands_by_size",
                False,
                {"year": target_year, "small_group": small_group, "large_group": large_group},
                [f"P12 ({target_year})"],
                error="Missing distribution rows for one or both groups.",
            )

        small_high = self._sum_high_bands(small_dist)
        large_high = self._sum_high_bands(large_dist)
        delta = large_high - small_high
        small_under4 = small_dist.get("under 4", 0.0)
        large_under4 = large_dist.get("under 4", 0.0)

        return ToolResult(
            "compare_contribution_bands_by_size",
            True,
            {
                "year": target_year,
                "small_group": self._canonical_size_label(small_group),
                "large_group": self._canonical_size_label(large_group),
                "small_group_label": small_group,
                "large_group_label": large_group,
                "small_high_band_share": round(small_high, 1),
                "large_high_band_share": round(large_high, 1),
                "delta_percentage_points": round(delta, 1),
                "small_under4_share": round(small_under4, 1),
                "large_under4_share": round(large_under4, 1),
            },
            [f"P12 ({target_year})"],
        )

    def explain_quality_flags(
        self, table_id: str, year: int, group_type: str, group_value_hint: str
    ) -> ToolResult:
        table_id = table_id.upper().strip()
        target_year = int(year)
        resolved, candidates = self._resolve_dimension_value(table_id, group_type, group_value_hint, target_year)
        if not resolved:
            return ToolResult(
                "explain_quality_flags",
                False,
                {},
                [f"{table_id} ({target_year})"],
                error=self._not_found_error(group_type, group_value_hint, candidates),
            )

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT suppression_flag, COUNT(*) AS count_rows
                FROM observations
                WHERE table_id = ?
                  AND year = ?
                  AND primary_group_type = ?
                  AND lower(primary_group_value) = lower(?)
                GROUP BY suppression_flag
                """,
                (table_id, target_year, group_type, resolved),
            ).fetchall()

        breakdown = {str(row["suppression_flag"]): int(row["count_rows"]) for row in rows}
        total = sum(breakdown.values())

        return ToolResult(
            "explain_quality_flags",
            True,
            {
                "table_id": table_id,
                "year": target_year,
                "group_type": group_type,
                "group_value": resolved,
                "suppression_counts": breakdown,
                "total_rows": total,
            },
            [f"{table_id} ({target_year})"],
        )


# ---------------------------------------------------------------------------
# LangChain tool bindings.
#
# Each function below wraps one SQLiteTools method as a real LangChain tool:
# the pydantic args_schema and docstring are what the LLM sees when tools are
# bound via `.bind_tools()`, and the return value is a plain JSON-serializable
# dict so `ToolNode` can pack it straight into a ToolMessage. No hand-rolled
# tool catalog or dispatch table is needed — LangChain derives the schema and
# LangGraph's prebuilt ToolNode handles execution/dispatch natively.
# ---------------------------------------------------------------------------

_INSTANCE = SQLiteTools(os.getenv("ASHE_DB_PATH", "data/ashe.db"))


def _as_dict(result: ToolResult) -> dict[str, Any]:
    return dataclasses.asdict(result)


def _safe_call(tool_name: str, func: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return _as_dict(func(**kwargs))
    except Exception as exc:  # noqa: BLE001
        return _as_dict(ToolResult(tool_name=tool_name, ok=False, data={}, sources=[], error=str(exc)))


@tool("list_tables")
def list_tables_tool() -> dict[str, Any]:
    """Discover available ASHE table families (P1, P2, P4-P12) and their year coverage.

    Use this before planning retrieval if you are unsure which table stores the
    metric you need, or to answer a general "what data do you have" question.
    """
    return _safe_call("list_tables", _INSTANCE.list_tables)


class DescribeSchemaArgs(BaseModel):
    table_id: str = Field(description="ASHE table id, e.g. 'P2', 'P4', 'P12'.")


@tool("describe_schema", args_schema=DescribeSchemaArgs)
def describe_schema_tool(table_id: str) -> dict[str, Any]:
    """Describe one table's dimensions, measure types, year range, and segment/
    work-pattern coverage (e.g. whether it is split by gender or work pattern).

    Use this when you need valid filters before calling a metric tool, or to
    check whether a table covers a breakdown the user asked about (such as
    gender) before claiming it is or isn't available.
    """
    return _safe_call("describe_schema", _INSTANCE.describe_schema, table_id=table_id)


class ListDimensionValuesArgs(BaseModel):
    table_id: str = Field(description="ASHE table id, e.g. 'P2', 'P4', 'P12'.")
    dimension: Literal["industry", "employer_size", "age", "occupation", "pension_type"] = Field(
        description="Which dimension to list canonical values for."
    )
    year: int | None = Field(default=None, description="Optional year filter.")
    limit: int = Field(default=50, description="Maximum number of values to return.")


@tool("list_dimension_values", args_schema=ListDimensionValuesArgs)
def list_dimension_values_tool(
    table_id: str, dimension: str, year: int | None = None, limit: int = 50
) -> dict[str, Any]:
    """List the exact canonical values for one dimension in a table/year.

    Use this to get the precise spelling of industry names, employer-size
    bands, age groups, or pension-type labels before calling a metric tool, or
    to resolve a metric tool's "could not resolve" error with suggestions.
    """
    return _safe_call(
        "list_dimension_values",
        _INSTANCE.list_dimension_values,
        table_id=table_id,
        dimension=dimension,
        year=year,
        limit=limit,
    )


class LookupIndustryRateArgs(BaseModel):
    industry_hint: str = Field(
        description="Industry name or phrase from the user's question, e.g. 'finance sector' or an exact ONS label."
    )
    pension_family: Literal["defined contribution", "defined benefit", "overall participation"] = Field(
        default="defined contribution",
        description="Which pension family's membership rate to return.",
    )
    year: int | None = Field(default=None, description="Defaults to the latest year available in P2.")


@tool("lookup_industry_membership_rate", args_schema=LookupIndustryRateArgs)
def lookup_industry_membership_rate_tool(
    industry_hint: str, pension_family: str = "defined contribution", year: int | None = None
) -> dict[str, Any]:
    """Return a single industry's pension-scheme membership percentage from
    table P2, for one pension family and year.

    If the industry can't be resolved, the response includes candidate
    industry names to retry with instead of guessing.
    """
    return _safe_call(
        "lookup_industry_membership_rate",
        _INSTANCE.lookup_industry_membership_rate,
        industry_hint=industry_hint,
        pension_family=pension_family,
        year=year,
    )


class IndustryTrendArgs(BaseModel):
    industry_hint: str = Field(description="Industry name or phrase from the user's question.")
    since_year: int = Field(default=2019, description="Start year to compare from.")
    pension_family: Literal["defined contribution", "defined benefit", "overall participation"] = Field(
        default="overall participation",
        description="Which pension family's trend to return.",
    )


@tool("industry_participation_trend", args_schema=IndustryTrendArgs)
def industry_participation_trend_tool(
    industry_hint: str, since_year: int = 2019, pension_family: str = "overall participation"
) -> dict[str, Any]:
    """Return start/end/delta trend for an industry from P2, comparing
    since_year to the latest available year.

    Best for "has participation gone up or down since YEAR" questions.
    """
    return _safe_call(
        "industry_participation_trend",
        _INSTANCE.industry_participation_trend,
        industry_hint=industry_hint,
        since_year=since_year,
        pension_family=pension_family,
    )


class CompareContributionBandsArgs(BaseModel):
    year: int | None = Field(default=None, description="Defaults to the latest year available in P12.")
    small_group_hint: str = Field(default="1 to 12 employees", description="Small-employer size band.")
    large_group_hint: str = Field(default="1000 or more employees", description="Large-employer size band.")


@tool("compare_contribution_bands_by_size", args_schema=CompareContributionBandsArgs)
def compare_contribution_bands_by_size_tool(
    year: int | None = None,
    small_group_hint: str = "1 to 12 employees",
    large_group_hint: str = "1000 or more employees",
) -> dict[str, Any]:
    """Compare employer contribution-band distribution between small and large
    employers from P12.

    Returns each cohort's high-band (8%+) share and lower-band (under 4%)
    share, for questions comparing contribution rates by employer size.
    """
    return _safe_call(
        "compare_contribution_bands_by_size",
        _INSTANCE.compare_contribution_bands_by_size,
        year=year,
        small_group_hint=small_group_hint,
        large_group_hint=large_group_hint,
    )


class ExplainQualityFlagsArgs(BaseModel):
    table_id: str = Field(description="ASHE table id, e.g. 'P2', 'P4', 'P12'.")
    year: int = Field(description="Year to check.")
    group_type: Literal["industry", "employer_size", "age", "occupation"] = Field(
        description="Which dimension the group_value_hint belongs to."
    )
    group_value_hint: str = Field(description="Group value or phrase to check reliability for.")


@tool("explain_quality_flags", args_schema=ExplainQualityFlagsArgs)
def explain_quality_flags_tool(
    table_id: str, year: int, group_type: str, group_value_hint: str
) -> dict[str, Any]:
    """Report suppression-flag counts for a selected slice (table/year/group).

    Use this to explain reliability caveats in an answer, e.g. when a slice
    has a meaningful share of suppressed/unreliable estimates.
    """
    return _safe_call(
        "explain_quality_flags",
        _INSTANCE.explain_quality_flags,
        table_id=table_id,
        year=year,
        group_type=group_type,
        group_value_hint=group_value_hint,
    )


ALL_TOOLS = [
    list_tables_tool,
    describe_schema_tool,
    list_dimension_values_tool,
    lookup_industry_membership_rate_tool,
    industry_participation_trend_tool,
    compare_contribution_bands_by_size_tool,
    explain_quality_flags_tool,
]
