from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "ashe.db"
GOLDEN_PATH = ROOT / "evals" / "golden_dataset.json"
EPS = 0.15


def normalize_key(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def canonical_pension_family(value: str) -> str:
    key = normalize_key(value)
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


def fetch_industry_p2_percent(conn: sqlite3.Connection, year: int, industry: str, family: str | None) -> float | None:
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
    usable = [(canonical_pension_family(p), float(v)) for p, v, s in rows if v is not None and s is None]
    if not usable:
        return None
    if family:
        vals = [v for fam, v in usable if fam == family]
        return sum(vals) if vals else None

    no_pension = [v for fam, v in usable if fam == "no pension provision"]
    if no_pension:
        return 100.0 - sum(no_pension)
    return None


def fetch_p12_dist(conn: sqlite3.Connection, year: int, group_value: str) -> dict[str, float]:
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
        (year, group_value),
    ).fetchall()
    out: dict[str, float] = {}
    for band, value, sup in rows:
        if value is None or sup is not None:
            continue
        out[normalize_key(band)] = float(value)
    return out


def sum_high_bands(dist: dict[str, float]) -> float:
    total = 0.0
    for band, value in dist.items():
        if "8" in band and "10" in band:
            total += value
        elif "10" in band and "12" in band:
            total += value
        elif "12" in band and "15" in band:
            total += value
        elif "15" in band and "20" in band:
            total += value
        elif "20" in band and "over" in band:
            total += value
    return total


def close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, abs_tol=EPS)


def main() -> None:
    cases = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []
    checks = 0

    with sqlite3.connect(DB_PATH) as conn:
        for case in cases:
            intent = case["intent"]
            expected = case.get("expected", {})
            case_id = case["id"]

            if not expected:
                # Nothing numeric to check against the DB (e.g. a scope/clarification
                # case where the expected behavior is declining or asking a question,
                # not returning a fact) — this script only verifies numeric facts.
                continue

            if intent == "industry_dc_rate":
                checks += 1
                industry = expected["industry"]
                year = int(expected["year"])
                actual = fetch_industry_p2_percent(conn, year, industry, "defined contribution")
                if actual is None or not close(actual, float(expected["percent"])):
                    failures.append(
                        f"{case_id}: expected DC% {expected['percent']}, actual={actual}"
                    )

            elif intent == "industry_trend":
                checks += 1
                industry = expected["industry"]
                start_year = int(expected["since_year"])
                end_year = int(expected["latest_year"])
                start = fetch_industry_p2_percent(conn, start_year, industry, None)
                end = fetch_industry_p2_percent(conn, end_year, industry, None)
                if start is None or end is None:
                    failures.append(f"{case_id}: missing start/end values start={start}, end={end}")
                    continue
                delta = end - start
                direction = "up" if delta > 0 else "down" if delta < 0 else "flat"
                if not close(start, float(expected["start_percent"])):
                    failures.append(f"{case_id}: start mismatch expected={expected['start_percent']} actual={start}")
                if not close(end, float(expected["end_percent"])):
                    failures.append(f"{case_id}: end mismatch expected={expected['end_percent']} actual={end}")
                if not close(delta, float(expected["delta_percentage_points"])):
                    failures.append(
                        f"{case_id}: delta mismatch expected={expected['delta_percentage_points']} actual={delta}"
                    )
                if direction != expected["direction"]:
                    failures.append(f"{case_id}: direction mismatch expected={expected['direction']} actual={direction}")

            elif intent in {"small_vs_large_contrib", "advisory_edge"}:
                checks += 1
                year = int(expected["year"])
                small = fetch_p12_dist(conn, year, "1 to 12 employees")
                large = fetch_p12_dist(conn, year, "1000 or more employees")
                small_high = sum_high_bands(small)
                large_high = sum_high_bands(large)
                if "small_high_band_share" in expected and not close(
                    small_high, float(expected["small_high_band_share"])
                ):
                    failures.append(
                        f"{case_id}: small high-band mismatch expected={expected['small_high_band_share']} actual={small_high}"
                    )
                if "large_high_band_share" in expected and not close(
                    large_high, float(expected["large_high_band_share"])
                ):
                    failures.append(
                        f"{case_id}: large high-band mismatch expected={expected['large_high_band_share']} actual={large_high}"
                    )
                if "small_under4_share" in expected:
                    actual = small.get("under 4", 0.0)
                    if not close(actual, float(expected["small_under4_share"])):
                        failures.append(
                            f"{case_id}: small under4 mismatch expected={expected['small_under4_share']} actual={actual}"
                        )
                if "large_under4_share" in expected:
                    actual = large.get("under 4", 0.0)
                    if not close(actual, float(expected["large_under4_share"])):
                        failures.append(
                            f"{case_id}: large under4 mismatch expected={expected['large_under4_share']} actual={actual}"
                        )

    if failures:
        print(f"Golden verification failed: {len(failures)} issue(s) across {checks} numeric checks")
        for item in failures:
            print(f"- {item}")
        raise SystemExit(1)

    print(f"Golden verification passed: {checks} numeric checks match data in {DB_PATH}")


if __name__ == "__main__":
    main()
