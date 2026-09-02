from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import ask_agent  # noqa: E402


EPS = 0.15


@dataclass
class CaseResult:
    case_id: str
    retrieval_pass: bool
    generation_pass: bool
    retrieval_errors: list[str]
    generation_errors: list[str]


def load_golden(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _close(a: float, b: float, eps: float = EPS) -> bool:
    return math.isclose(a, b, abs_tol=eps)


def check_retrieval(case: dict[str, object], tool_calls: list[dict[str, object]], sources: list[str]) -> list[str]:
    """Checks retrieval quality: did the agent call the right tool(s) and get
    back the right structured facts? Independent of how the final answer is
    phrased (that's check_generation's job).
    """
    errors: list[str] = []
    expected_tools = set(case.get("expected_tool_calls", []))
    ok_tools = {str(call.get("tool")) for call in tool_calls if call.get("ok")}
    attempted_tools = {str(call.get("tool")) for call in tool_calls}

    if expected_tools:
        missing = expected_tools - ok_tools
        if missing:
            errors.append(
                f"expected tool(s) {sorted(missing)} not called successfully; "
                f"attempted={sorted(attempted_tools)}, succeeded={sorted(ok_tools)}"
            )
    else:
        # Scope/policy cases: the agent should not be fetching a pension-metric
        # tool result for a question this dataset can't answer.
        unexpected = ok_tools - {"list_tables", "describe_schema", "list_dimension_values"}
        if unexpected:
            errors.append(f"expected no metric tool calls for this case; succeeded={sorted(unexpected)}")

    expected_source = case.get("expected_source")
    if expected_source and expected_source not in sources:
        errors.append(f"source missing: expected {expected_source}, actual={sources}")

    expected = case.get("expected", {})
    if expected:
        matched = [call for call in tool_calls if call.get("tool") in expected_tools and call.get("ok")]
        data = matched[-1].get("result") or {} if matched else {}
        if not matched:
            errors.append("no successful matching tool result to check expected fields against")
        for key, exp_value in expected.items():
            if key not in data:
                errors.append(f"missing key in tool result: {key}")
                continue
            actual = data[key]
            if isinstance(exp_value, float):
                if not _close(float(actual), exp_value):
                    errors.append(f"value mismatch for {key}: expected={exp_value}, actual={actual}")
            elif actual != exp_value:
                errors.append(f"value mismatch for {key}: expected={exp_value}, actual={actual}")

    return errors


def check_generation(case: dict[str, object], answer: str) -> list[str]:
    errors: list[str] = []
    required = case.get("required_answer_fragments", [])
    answer_lower = answer.lower()
    for fragment in required:
        if str(fragment).lower() not in answer_lower:
            errors.append(f"missing fragment in answer: {fragment}")
    return errors


def main() -> None:
    golden_path = ROOT / "evals" / "golden_dataset.json"
    report_path = ROOT / "evals" / "eval_report.json"

    cases = load_golden(golden_path)
    results: list[CaseResult] = []

    for case in cases:
        question = str(case["question"])
        output = ask_agent(question)
        tool_calls = list(output.get("tool_calls", []))
        sources = list(output.get("sources", []))
        answer = str(output.get("answer", ""))

        retrieval_errors = check_retrieval(case, tool_calls, sources)
        generation_errors = check_generation(case, answer)

        results.append(
            CaseResult(
                case_id=str(case["id"]),
                retrieval_pass=not retrieval_errors,
                generation_pass=not generation_errors,
                retrieval_errors=retrieval_errors,
                generation_errors=generation_errors,
            )
        )

    retrieval_failures = [r for r in results if not r.retrieval_pass]
    generation_failures = [r for r in results if r.retrieval_pass and not r.generation_pass]
    both_failures = [r for r in results if (not r.retrieval_pass and not r.generation_pass)]

    report = {
        "total_cases": len(results),
        "retrieval_pass": len([r for r in results if r.retrieval_pass]),
        "generation_pass": len([r for r in results if r.generation_pass]),
        "retrieval_failures": [r.__dict__ for r in retrieval_failures],
        "generation_failures": [r.__dict__ for r in generation_failures],
        "both_failures": [r.__dict__ for r in both_failures],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Cases: {report['total_cases']}")
    print(f"Retrieval pass: {report['retrieval_pass']}")
    print(f"Generation pass: {report['generation_pass']}")
    print(f"Retrieval failures: {len(retrieval_failures)}")
    print(f"Generation-only failures: {len(generation_failures)}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
