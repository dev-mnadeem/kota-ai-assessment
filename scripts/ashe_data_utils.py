from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable


EXCLUDE_TOKENS = (
    "cv__",
    "_cv_",
    "cvs",
    "notes",
    "glossary",
    "contents",
    "cv_notes",
)


def discover_csv_files(prepared_dir: Path) -> list[Path]:
    return sorted(prepared_dir.rglob("*.csv"))


def infer_table_id(filename: str) -> str | None:
    match = re.search(r"(?i)(?:^|[_\-\s])p(\d{1,2})(?:[._\-\s]|$)", filename)
    if not match:
        match = re.search(r"(?i)table[_\-\s]?p(\d{1,2})", filename)
    if not match:
        match = re.search(r"(?i)p(1[0-2]|[1-9])\b", filename)
    if not match:
        return None
    return f"P{int(match.group(1))}"


def infer_year(text: str) -> int | None:
    match = re.search(r"(20\d{2})", text)
    if match:
        return int(match.group(1))
    return None


def infer_segment_from_filename(filename: str) -> tuple[str | None, str | None]:
    segment = None
    work_pattern = None

    if "__" in filename:
        suffix = filename.split("__", 1)[1].rsplit(".", 1)[0]
        parts = [p.strip() for p in suffix.split("_") if p.strip()]
        if parts:
            segment = " ".join(parts)
            lowered = segment.lower()
            if "full-time" in lowered:
                work_pattern = "Full-Time"
            elif "part-time" in lowered:
                work_pattern = "Part-Time"
            elif segment.lower() == "all":
                work_pattern = "All"
    return segment, work_pattern


def classify_file(path: Path) -> dict[str, object]:
    name = path.name
    lowered = name.lower()
    table_id = infer_table_id(name)
    year = infer_year(name)
    segment, work_pattern = infer_segment_from_filename(name)

    is_cv = "cv" in lowered
    is_all_segment = lowered.endswith("__all.csv")

    include_v1 = (
        table_id in {"P1", "P2", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12"}
        and is_all_segment
        and not any(token in lowered for token in EXCLUDE_TOKENS)
    )

    return {
        "path": str(path),
        "filename": name,
        "table_id": table_id,
        "year": year,
        "segment": segment,
        "work_pattern": work_pattern,
        "is_cv_like": is_cv,
        "include_v1": include_v1,
    }


def read_csv_rows(file_path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with file_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            rows.append([cell.strip() for cell in row])
    return rows


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def is_earnings_band(label: str) -> bool:
    text = label.lower()
    return (
        text.startswith("less than")
        or text.startswith("over")
        or text.startswith("gbp")
        or text.startswith("£")
        or " to " in text and "£" in text
    )


def parse_numeric(raw: str) -> float | None:
    text = raw.strip()
    if not text:
        return None
    lowered = text.lower()
    if text.startswith("[") and text.endswith("]"):
        return None
    if lowered in {"x", "..", ":", "-"}:
        return None
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def suppression_flag(raw: str) -> str | None:
    text = raw.strip().lower()
    if text in {"[u]", "[c]", "[w]"}:
        return text[1]
    if text == "x":
        return "x"
    if text == "..":
        return "d"
    if text == "-":
        return "n"
    if text == ":":
        return "a"
    return None


def infer_primary_group_type(header_label: str, table_id: str | None) -> str:
    label = header_label.lower()
    if "sic" in label or "industry" in label:
        return "industry"
    if "age" in label:
        return "age"
    if "size of business" in label or "size of company" in label:
        return "employer_size"
    if "occupation" in label:
        return "occupation"
    if table_id == "P2":
        return "industry"
    if table_id in {"P6", "P10"}:
        return "industry"
    if table_id in {"P1", "P5", "P9"}:
        return "age"
    if table_id in {"P4", "P8", "P12"}:
        return "employer_size"
    if table_id in {"P7", "P11"}:
        return "occupation"
    return "group"


def infer_measure_type(measure_label: str) -> str:
    lowered = measure_label.lower()
    if "percentage of jobs" in lowered:
        return "percent_jobs"
    if "per cent" in lowered:
        return "percent_jobs"
    if "number of jobs" in lowered:
        return "jobs_thousands"
    if "jobs" in lowered and "thousand" in lowered:
        return "jobs_thousands"
    if "jobs" in lowered:
        return "jobs_thousands"
    if "percentage of qualifying earnings" in lowered:
        return "percent_qualifying_earnings"
    if "percentage of full pay" in lowered:
        return "percent_full_pay"
    return "value"


def iter_nonempty_cells(row: Iterable[str]) -> Iterable[tuple[int, str]]:
    for index, cell in enumerate(row):
        if cell and cell.strip():
            yield index, cell.strip()
