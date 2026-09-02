from __future__ import annotations

import sqlite3
from pathlib import Path


def fetch_one(conn: sqlite3.Connection, query: str) -> object:
    cur = conn.execute(query)
    return cur.fetchone()[0]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    db_path = root / "data" / "ashe.db"

    if not db_path.exists():
        raise FileNotFoundError("Missing data/ashe.db. Run scripts/normalize_to_sqlite.py first.")

    with sqlite3.connect(db_path) as conn:
        total_obs = fetch_one(conn, "SELECT COUNT(*) FROM observations")
        total_files = fetch_one(conn, "SELECT COUNT(DISTINCT source_file) FROM observations")
        ok_files = fetch_one(conn, "SELECT COUNT(*) FROM ingest_log WHERE status = 'ok'")
        error_files = fetch_one(conn, "SELECT COUNT(*) FROM ingest_log WHERE status = 'error'")

        print(f"DB: {db_path}")
        print(f"Observation rows:      {total_obs}")
        print(f"Distinct source files: {total_files}")
        print(f"Ingest OK files:       {ok_files}")
        print(f"Ingest error files:    {error_files}")

        print("\nRows by table_id:")
        for row in conn.execute(
            "SELECT table_id, COUNT(*) FROM observations GROUP BY table_id ORDER BY table_id"
        ):
            print(f"  {row[0]}: {row[1]}")

        print("\nYear coverage by table_id:")
        for row in conn.execute(
            """
            SELECT table_id, MIN(year), MAX(year), COUNT(DISTINCT year)
            FROM observations
            GROUP BY table_id
            ORDER BY table_id
            """
        ):
            print(f"  {row[0]}: min={row[1]}, max={row[2]}, years={row[3]}")

        print("\nDistinct measure types:")
        for row in conn.execute(
            "SELECT measure_type, COUNT(*) FROM observations GROUP BY measure_type ORDER BY measure_type"
        ):
            print(f"  {row[0]}: {row[1]}")

        print("\nSample rows:")
        for row in conn.execute(
            """
            SELECT table_id, year, primary_group_type, primary_group_value,
                   earnings_band, pension_type, measure_type, value_raw
            FROM observations
            WHERE primary_group_value = 'All employees' AND earnings_band IS NULL
            LIMIT 5
            """
        ):
            print(f"  {row}")


if __name__ == "__main__":
    main()
