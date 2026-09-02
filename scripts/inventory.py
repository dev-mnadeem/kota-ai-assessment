from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from ashe_data_utils import classify_file, discover_csv_files


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    # Raw ONS downloads/prepared CSVs aren't checked into this repo (see README.md).
    # Point this at wherever you've staged a fresh prepared/csv folder to re-run ingestion.
    base_dir = root / "prepared"
    prepared_dir = base_dir / "csv"
    out_dir = root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory_path = out_dir / "inventory.csv"
    summary_path = out_dir / "inventory_summary.json"

    rows = [classify_file(path) for path in discover_csv_files(prepared_dir)]

    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "filename",
                "table_id",
                "year",
                "segment",
                "work_pattern",
                "is_cv_like",
                "include_v1",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    v1_rows = [row for row in rows if row["include_v1"]]
    by_table = Counter(row["table_id"] for row in v1_rows)
    by_year = Counter(row["year"] for row in v1_rows)

    summary = {
        "prepared_dir": str(prepared_dir),
        "total_csv_files": len(rows),
        "v1_included_files": len(v1_rows),
        "v1_by_table": dict(sorted(by_table.items())),
        "v1_by_year": dict(sorted(by_year.items(), key=lambda item: (item[0] is None, item[0]))),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote inventory: {inventory_path}")
    print(f"Wrote summary:   {summary_path}")
    print(f"Total CSV files: {len(rows)}")
    print(f"V1 included:     {len(v1_rows)}")


if __name__ == "__main__":
    main()
