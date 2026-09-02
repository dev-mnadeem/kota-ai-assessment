from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from ashe_data_utils import (
    classify_file,
    infer_measure_type,
    infer_primary_group_type,
    normalize_text,
    parse_numeric,
    read_csv_rows,
    suppression_flag,
)


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS observations;
        DROP TABLE IF EXISTS ingest_log;

        CREATE TABLE observations (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          table_id TEXT NOT NULL,
          year INTEGER,
          segment TEXT,
          work_pattern TEXT,
          primary_group_type TEXT,
          primary_group_value TEXT,
          earnings_band TEXT,
          pension_type TEXT,
          measure_type TEXT,
          value_num REAL,
          value_raw TEXT,
          suppression_flag TEXT,
          source_file TEXT NOT NULL
        );

        CREATE TABLE ingest_log (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          source_file TEXT NOT NULL,
          status TEXT NOT NULL,
          rows_loaded INTEGER NOT NULL,
          error_message TEXT
        );

        CREATE INDEX idx_observations_table_year ON observations (table_id, year);
        CREATE INDEX idx_observations_group_year ON observations (primary_group_type, primary_group_value, year);
        CREATE INDEX idx_observations_measure ON observations (measure_type);
        """
    )


def find_header_row_index(rows: list[list[str]]) -> int | None:
    for index, row in enumerate(rows):
        if not row:
            continue
        first = row[0].strip().lower()
        if first.startswith("pension type"):
            return index
        if "contribution rate" in first:
            return index
    return None


def find_legacy_description_row(rows: list[list[str]]) -> int | None:
    for index, row in enumerate(rows):
        if not row:
            continue
        first = row[0].strip().lower() if len(row) > 0 else ""
        if first == "description":
            return index
    return None


def build_column_metadata(rows: list[list[str]], header_idx: int) -> tuple[str, dict[int, tuple[str, str]]]:
    pension_row = rows[header_idx]
    measure_row = rows[header_idx + 1] if header_idx + 1 < len(rows) else []

    primary_label = measure_row[0] if measure_row else ""
    col_meta: dict[int, tuple[str, str]] = {}

    for col in range(1, max(len(pension_row), len(measure_row))):
        pension = pension_row[col].strip() if col < len(pension_row) else ""
        measure = measure_row[col].strip() if col < len(measure_row) else ""
        if not pension and not measure:
            continue
        pension = normalize_text(pension) if pension else "Unknown"
        measure = normalize_text(measure) if measure else "Value"
        col_meta[col] = (pension, infer_measure_type(measure))

    return normalize_text(primary_label), col_meta


def build_legacy_column_metadata(rows: list[list[str]], desc_idx: int) -> tuple[str, dict[int, tuple[str, str]]]:
    measure_idx = None
    for idx in range(desc_idx - 1, -1, -1):
        joined = " ".join(cell.strip().lower() for cell in rows[idx] if cell and cell.strip())
        if "per cent" in joined or "jobs" in joined:
            measure_idx = idx
            break

    if measure_idx is None:
        return "Description", {}

    top_idx = None
    mid_idx = None
    for idx in range(measure_idx - 1, -1, -1):
        nonempty = [cell.strip() for cell in rows[idx] if cell and cell.strip()]
        if not nonempty:
            continue
        if top_idx is None:
            top_idx = idx
        elif mid_idx is None:
            mid_idx = idx
            break

    top_row = rows[top_idx] if top_idx is not None else []
    mid_row = rows[mid_idx] if mid_idx is not None else []
    measure_row = rows[measure_idx]

    col_meta: dict[int, tuple[str, str]] = {}
    last_top = ""
    last_mid = ""

    max_cols = max(len(top_row), len(mid_row), len(measure_row))
    for col in range(2, max_cols):
        if col < len(top_row) and top_row[col].strip():
            last_top = normalize_text(top_row[col])
        if col < len(mid_row) and mid_row[col].strip():
            last_mid = normalize_text(mid_row[col])

        measure = measure_row[col].strip() if col < len(measure_row) else ""
        if not measure:
            continue

        measure_norm = normalize_text(measure)
        if not ("jobs" in measure_norm.lower() or "per cent" in measure_norm.lower()):
            continue

        pension_type = last_top or "Unknown"
        if last_mid and "contracted" in last_mid.lower():
            pension_type = f"{pension_type} - {last_mid}"

        col_meta[col] = (pension_type, infer_measure_type(measure_norm))

    return "Description", col_meta


def iter_data_records(file_path: Path) -> list[dict[str, object]]:
    meta = classify_file(file_path)
    rows = read_csv_rows(file_path)
    header_idx = find_header_row_index(rows)
    desc_idx = find_legacy_description_row(rows)

    if header_idx is not None and header_idx + 1 < len(rows):
        primary_header_label, column_meta = build_column_metadata(rows, header_idx)
        data_start = header_idx + 2
    elif desc_idx is not None:
        primary_header_label, column_meta = build_legacy_column_metadata(rows, desc_idx)
        data_start = desc_idx + 1
    else:
        raise ValueError("Could not locate recognized header rows")

    if not column_meta:
        raise ValueError("No data columns detected from headers")

    primary_group_type = infer_primary_group_type(primary_header_label, meta["table_id"])

    records: list[dict[str, object]] = []
    current_group: str | None = None

    for row in rows[data_start:]:
        if not row:
            continue
        label = normalize_text(row[0] if row else "")
        if not label:
            continue

        lower_label = label.lower()
        if lower_label in {"notes", "glossary", "contents", "cv notes"}:
            continue

        if lower_label.startswith("less than") or lower_label.startswith("over") or lower_label.startswith("£") or (" to " in lower_label and "£" in lower_label):
            earnings_band = label
            primary_group_value = current_group or "All employees"
        else:
            current_group = label
            earnings_band = None
            primary_group_value = label

        for col_idx, (pension_type, measure_type) in column_meta.items():
            if col_idx >= len(row):
                continue
            raw_value = row[col_idx].strip()
            if raw_value == "":
                continue

            records.append(
                {
                    "table_id": meta["table_id"],
                    "year": meta["year"],
                    "segment": meta["segment"],
                    "work_pattern": meta["work_pattern"],
                    "primary_group_type": primary_group_type,
                    "primary_group_value": primary_group_value,
                    "earnings_band": earnings_band,
                    "pension_type": pension_type,
                    "measure_type": measure_type,
                    "value_num": parse_numeric(raw_value),
                    "value_raw": raw_value,
                    "suppression_flag": suppression_flag(raw_value),
                    "source_file": str(file_path),
                }
            )

    return records


def load_inventory(inventory_path: Path) -> list[Path]:
    with inventory_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [Path(row["path"]) for row in reader if row.get("include_v1") == "True"]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    db_path = root / "data" / "ashe.db"
    inventory_path = root / "data" / "inventory.csv"

    if not inventory_path.exists():
        raise FileNotFoundError("Missing data/inventory.csv. Run scripts/inventory.py first.")

    source_files = load_inventory(inventory_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        ensure_schema(conn)

        total_rows = 0
        for file_path in source_files:
            try:
                rows = iter_data_records(file_path)
                conn.executemany(
                    """
                    INSERT INTO observations (
                      table_id, year, segment, work_pattern,
                      primary_group_type, primary_group_value, earnings_band,
                      pension_type, measure_type, value_num, value_raw,
                      suppression_flag, source_file
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            row["table_id"],
                            row["year"],
                            row["segment"],
                            row["work_pattern"],
                            row["primary_group_type"],
                            row["primary_group_value"],
                            row["earnings_band"],
                            row["pension_type"],
                            row["measure_type"],
                            row["value_num"],
                            row["value_raw"],
                            row["suppression_flag"],
                            row["source_file"],
                        )
                        for row in rows
                    ],
                )
                conn.execute(
                    "INSERT INTO ingest_log (source_file, status, rows_loaded, error_message) VALUES (?, ?, ?, ?)",
                    (str(file_path), "ok", len(rows), None),
                )
                total_rows += len(rows)
            except Exception as exc:  # noqa: BLE001
                conn.execute(
                    "INSERT INTO ingest_log (source_file, status, rows_loaded, error_message) VALUES (?, ?, ?, ?)",
                    (str(file_path), "error", 0, str(exc)),
                )

        conn.commit()

    print(f"Built SQLite DB: {db_path}")
    print(f"Source files used: {len(source_files)}")
    print(f"Observation rows:  {total_rows}")


if __name__ == "__main__":
    main()
