import pytest

from scripts.ashe_data_utils import infer_table_id, parse_numeric, suppression_flag
from scripts.report_ingest_errors import infer_table_id_from_name


@pytest.mark.parametrize(
    ("name", "expected"),
    [("2024_P1__All.csv", "P1"), ("2024_P10__All.csv", "P10"), ("table-p12.xlsx", "P12"),
     ("2019_p7_Full-Time.csv", "P7"), ("notes.csv", "UNKNOWN")],
)
def test_report_groups_errors_by_the_right_table(name, expected):
    assert infer_table_id_from_name(name) == expected


@pytest.mark.parametrize(("name", "expected"), [("2024_P11__All.csv", "P11"), ("table_p4.csv", "P4")])
def test_inventory_table_ids(name, expected):
    assert infer_table_id(name) == expected


@pytest.mark.parametrize(
    ("raw", "value", "flag"),
    [("65.2", 65.2, None), ("1,204", 1204.0, None), ("[u]", None, "u"), ("x", None, "x"), ("..", None, "d")],
)
def test_published_cells_parse_to_values_and_flags(raw, value, flag):
    assert parse_numeric(raw) == value
    assert suppression_flag(raw) == flag
