from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path


def infer_table_id_from_name(name: str) -> str:
    lowered = name.lower()
    for idx in range(1, 13):
        token = f"p{idx}"
        if token in lowered:
            return token.upper()
    return "UNKNOWN"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    db_path = root / "data" / "ashe.db"

    if not db_path.exists():
        raise FileNotFoundError("Missing data/ashe.db. Run scripts/normalize_to_sqlite.py first.")

    with sqlite3.connect(db_path) as conn:
        rows = list(
            conn.execute(
                """
                SELECT source_file, error_message
                FROM ingest_log
                WHERE status = 'error'
                ORDER BY source_file
                """
            )
        )

    if not rows:
        print("No ingest errors found.")
        return

    by_reason: dict[str, list[str]] = defaultdict(list)
    by_table: dict[str, list[str]] = defaultdict(list)

    for source_file, error_message in rows:
        by_reason[error_message or "Unknown error"].append(source_file)
        by_table[infer_table_id_from_name(Path(source_file).name)].append(source_file)

    print(f"Total ingest errors: {len(rows)}\n")

    print("By reason:")
    for reason, files in sorted(by_reason.items(), key=lambda item: len(item[1]), reverse=True):
        print(f"- {reason}: {len(files)}")

    print("\nBy table:")
    for table_id, files in sorted(by_table.items()):
        print(f"- {table_id}: {len(files)}")

    print("\nSample files:")
    for source_file, error_message in rows[:15]:
        print(f"- {Path(source_file).name} | {error_message}")


if __name__ == "__main__":
    main()
