import sqlite3

import pytest

from src.agent import tools as tools_module
from src.agent.tools import SQLiteTools
from src.datasets import ROOT, build

INJECTIONS = [
    "finance'; DROP TABLE observations; --",
    "x' UNION SELECT name, sql, 1 FROM sqlite_master --",
    "construction\"); ATTACH DATABASE 'pwned.db' AS p; --",
    "load_extension('/tmp/evil')",
]


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / "ashe.db"
    build(ROOT / "sources/fixture.json", path)
    return path


def test_list_tables_and_schema_describe_the_fixture(snapshot):
    tools = SQLiteTools(snapshot)
    tables = {t["table_id"]: t for t in tools.list_tables().data["tables"]}
    assert tables["P2"] == {"table_id": "P2", "min_year": 2023, "max_year": 2024, "row_count": 4}
    schema = tools.describe_schema(" p2 ")
    assert schema.ok and schema.data["dimensions"] == ["industry"]
    assert tools.describe_schema("P99").ok is False


def test_industry_aliases_and_exact_labels_resolve(snapshot):
    tools = SQLiteTools(snapshot)
    assert tools.lookup_industry_membership_rate("financial services", year=2023).data["percent"] == 60
    assert tools.lookup_industry_membership_rate("Construction", year=2024).data["percent"] == 40


def test_short_it_alias_does_not_hijack_other_industries(snapshot):
    result = SQLiteTools(snapshot).lookup_industry_membership_rate("is it high in construction", year=2024)
    assert result.ok
    assert result.data["industry"] == "Construction"


def test_suppressed_values_are_not_reported(snapshot):
    result = SQLiteTools(snapshot).lookup_industry_membership_rate("manufacturing", year=2024)
    assert result.ok is False
    assert "No usable value" in result.error


def test_unresolvable_industry_offers_candidates(snapshot):
    result = SQLiteTools(snapshot).lookup_industry_membership_rate("zzz quantum farming", year=2024)
    assert result.ok is False
    assert "list_dimension_values" in result.error


def test_list_dimension_values_clamps_limit(snapshot):
    tools = SQLiteTools(snapshot)
    assert len(tools.list_dimension_values("P2", "industry", limit=-1).data["values"]) == 1
    assert len(tools.list_dimension_values("P2", "industry", limit=10_000).data["values"]) == 3


@pytest.mark.parametrize("hint", INJECTIONS)
def test_injection_shaped_hints_are_data_not_sql(snapshot, hint, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = snapshot.read_bytes()
    tools = SQLiteTools(snapshot)
    tools.lookup_industry_membership_rate(hint, year=2024)
    tools.explain_quality_flags("P2", 2024, "industry", hint)
    tools.list_dimension_values(hint, "industry")
    assert snapshot.read_bytes() == before
    assert not (tmp_path / "pwned.db").exists()
    assert tools.list_tables().ok


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM observations",
        "INSERT INTO observations (table_id, source_file) VALUES ('P1', 'x')",
        "CREATE TABLE stolen (x)",
        "ATTACH DATABASE 'other.db' AS other",
        "PRAGMA query_only = OFF",
        "SELECT load_extension('evil')",
        "VACUUM INTO 'copy.db'",
    ],
)
def test_tool_connections_refuse_writes_attach_and_extensions(snapshot, statement, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before_files = set(tmp_path.iterdir())
    before_bytes = snapshot.read_bytes()
    with SQLiteTools(snapshot)._connect() as connection:
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute(statement)
    assert set(tmp_path.iterdir()) == before_files
    assert snapshot.read_bytes() == before_bytes


def test_langchain_tools_read_the_configured_path_at_call_time(snapshot, monkeypatch):
    monkeypatch.setenv("ASHE_DB_PATH", str(snapshot))
    result = tools_module.lookup_industry_membership_rate_tool.invoke(
        {"industry_hint": "finance", "year": 2024}
    )
    assert result["ok"] is True
    assert result["data"]["percent"] == 65
    assert result["sources"] == ["P2 (2024)"]


def test_langchain_tool_errors_become_tool_results(tmp_path, monkeypatch):
    monkeypatch.setenv("ASHE_DB_PATH", str(tmp_path / "missing.db"))
    result = tools_module.list_tables_tool.invoke({})
    assert result["ok"] is False
    assert "Dataset is missing" in result["error"]


def test_tool_schemas_reject_out_of_range_arguments():
    with pytest.raises(ValueError):
        tools_module.list_dimension_values_tool.invoke(
            {"table_id": "P2", "dimension": "industry", "limit": 0}
        )
    with pytest.raises(ValueError):
        tools_module.explain_quality_flags_tool.invoke(
            {"table_id": "P2", "year": 2024, "group_type": "sqlite_master", "group_value_hint": "x"}
        )
