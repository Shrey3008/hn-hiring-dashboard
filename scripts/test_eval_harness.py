"""Self-tests for the eval harness. No API key, no network.

An eval harness is a measuring instrument, and an instrument nobody calibrated
is decoration. These tests answer one question: if the model were wrong, would
this harness say so?

They work by replacing `src.nl_query._complete` with fake models:

  oracle      returns the reference SQL for every question   -> must score 100%
  saboteur    reintroduces the join fan-out on share queries -> those must FAIL
  overeager   invents an answer to unanswerable questions    -> those must FAIL

Plus unit tests for the comparator, which has to be lenient about column order,
column names and float noise while staying strict about values.

    python scripts/test_eval_harness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

import src.nl_query as nl_query  # noqa: E402
from src.evals.cases import CASES  # noqa: E402
from src.evals.compare import compare_results  # noqa: E402
from src.evals.runner import run_eval_set  # noqa: E402

BY_QUESTION = {c.question: c for c in CASES}

# The naive query for each share question: join job_tech, then divide by
# COUNT(*) over the joined rows. Safe, valid, fast, and about 3x too small.
# This is the exact shape of the bug this project shipped in July 2026.
SABOTAGED = {
    "llm_share_monthly": (
        "SELECT j.month, ROUND(100.0 * SUM(CASE WHEN t.tech = 'LLM' THEN 1 ELSE 0 END) "
        "/ COUNT(*), 2) AS pct FROM jobs j LEFT JOIN job_tech t ON j.post_id = t.post_id "
        "GROUP BY j.month ORDER BY j.month"
    ),
    "rust_share_monthly": (
        "SELECT j.month, ROUND(100.0 * SUM(CASE WHEN t.tech = 'Rust' THEN 1 ELSE 0 END) "
        "/ COUNT(*), 2) AS pct FROM jobs j LEFT JOIN job_tech t ON j.post_id = t.post_id "
        "GROUP BY j.month ORDER BY j.month"
    ),
    "python_share_2026": (
        "SELECT ROUND(100.0 * SUM(CASE WHEN t.tech = 'Python' THEN 1 ELSE 0 END) / COUNT(*), 2) "
        "AS pct FROM jobs j LEFT JOIN job_tech t ON j.post_id = t.post_id "
        "WHERE j.month >= '2026-01-01'"
    ),
    "docker_share_overall": (
        "SELECT ROUND(100.0 * SUM(CASE WHEN t.tech = 'Docker' THEN 1 ELSE 0 END) / COUNT(*), 2) "
        "AS pct FROM jobs j LEFT JOIN job_tech t ON j.post_id = t.post_id"
    ),
}

PASSED = []
FAILED = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    mark = "ok  " if condition else "FAIL"
    print(f"  {mark}  {name}{('  — ' + detail) if detail and not condition else ''}")


# --------------------------------------------------------------------------
# Fake models
# --------------------------------------------------------------------------


def make_oracle():
    def _complete(prompt: str, question: str, model: str) -> str:
        case = BY_QUESTION.get(question)
        if case is None:
            return "UNANSWERABLE"
        if case.expect_unanswerable:
            return "UNANSWERABLE"
        return case.reference_sql
    return _complete


def make_saboteur():
    def _complete(prompt: str, question: str, model: str) -> str:
        case = BY_QUESTION.get(question)
        if case is None:
            return "UNANSWERABLE"
        if case.expect_unanswerable:
            return "UNANSWERABLE"
        if case.id in SABOTAGED:
            return SABOTAGED[case.id]
        return case.reference_sql
    return _complete


def make_overeager():
    def _complete(prompt: str, question: str, model: str) -> str:
        case = BY_QUESTION.get(question)
        if case is None:
            return "UNANSWERABLE"
        if case.expect_unanswerable:
            # Plausible-looking nonsense: the failure mode being measured.
            return "SELECT company, COUNT(*) AS n FROM jobs GROUP BY company LIMIT 5"
        return case.reference_sql
    return _complete


# --------------------------------------------------------------------------
# Comparator unit tests
# --------------------------------------------------------------------------


def test_comparator() -> None:
    print("\ncomparator")

    base = pd.DataFrame({"month": ["2023-08-01", "2023-09-01"], "pct": [4.94, 4.73]})

    swapped_cols = pd.DataFrame({"pct": [4.94, 4.73], "month": ["2023-08-01", "2023-09-01"]})
    check("column order does not matter", compare_results(base, swapped_cols).passed)

    renamed = pd.DataFrame({"m": ["2023-08-01", "2023-09-01"], "share_of_posts": [4.94, 4.73]})
    check("column names do not matter", compare_results(base, renamed).passed)

    reordered_rows = pd.DataFrame({"month": ["2023-09-01", "2023-08-01"], "pct": [4.73, 4.94]})
    check("row order does not matter", compare_results(base, reordered_rows).passed)

    float_noise = pd.DataFrame(
        {"month": ["2023-08-01", "2023-09-01"], "pct": [4.940000000001, 4.7299999999]}
    )
    check("float noise is tolerated", compare_results(base, float_noise).passed)

    extra_col = pd.DataFrame(
        {"month": ["2023-08-01", "2023-09-01"], "pct": [4.94, 4.73], "posts": [17, 14]}
    )
    check("extra helper column is tolerated in contains mode", compare_results(base, extra_col).passed)
    check(
        "extra helper column is rejected in strict mode",
        not compare_results(base, extra_col, mode="strict").passed,
    )

    wrong_value = pd.DataFrame({"month": ["2023-08-01", "2023-09-01"], "pct": [1.50, 1.60]})
    check(
        "the 3x understatement is caught",
        not compare_results(base, wrong_value).passed,
        "THIS IS THE ONE THAT MATTERS",
    )

    missing_row = pd.DataFrame({"month": ["2023-08-01"], "pct": [4.94]})
    check("missing rows are caught", not compare_results(base, missing_row).passed)

    empty = pd.DataFrame()
    check("an empty result is caught", not compare_results(base, empty).passed)


# --------------------------------------------------------------------------
# End-to-end tests with fake models
# --------------------------------------------------------------------------


def test_oracle() -> None:
    print("\noracle model (answers every question correctly)")
    original = nl_query._complete
    nl_query._complete = make_oracle()
    try:
        report = run_eval_set(verbose=False)
    finally:
        nl_query._complete = original

    check(
        f"scores 100% ({report.passed}/{report.total})",
        report.accuracy == 1.0,
        "failures: " + ", ".join(c.id + ": " + c.reason for c in report.cases if not c.passed),
    )


def test_saboteur() -> None:
    print("\nsaboteur model (reintroduces the join fan-out on share queries)")
    original = nl_query._complete
    nl_query._complete = make_saboteur()
    try:
        report = run_eval_set(verbose=False)
    finally:
        nl_query._complete = original

    failed_ids = {c.id for c in report.cases if not c.passed}
    check(
        f"overall accuracy drops below 100% (got {report.accuracy:.1%})",
        report.accuracy < 1.0,
    )
    for case_id in SABOTAGED:
        check(f"catches the fan-out in {case_id}", case_id in failed_ids)
    unexpected = failed_ids - set(SABOTAGED)
    check(
        "no other case is disturbed",
        not unexpected,
        f"unexpected failures: {sorted(unexpected)}",
    )
    fan_out = report.by_tag.get("fan_out", {})
    check(
        f"fan_out tag accuracy falls to {fan_out.get('accuracy', 0):.0%}",
        fan_out.get("accuracy", 1.0) < 1.0,
    )


def test_overeager() -> None:
    print("\nover-eager model (invents answers to unanswerable questions)")
    original = nl_query._complete
    nl_query._complete = make_overeager()
    try:
        report = run_eval_set(verbose=False)
    finally:
        nl_query._complete = original

    failed_ids = {c.id for c in report.cases if not c.passed}
    expected = {c.id for c in CASES if c.expect_unanswerable}
    check("every unanswerable case fails", expected <= failed_ids)
    check("nothing else fails", not (failed_ids - expected), f"also failed: {sorted(failed_ids - expected)}")


def main() -> int:
    print("Eval harness self-tests — no API key required")
    test_comparator()
    test_oracle()
    test_saboteur()
    test_overeager()

    print("\n" + "=" * 60)
    print(f"  {len(PASSED)} passed, {len(FAILED)} failed")
    print("=" * 60)
    if FAILED:
        for name in FAILED:
            print(f"    FAILED: {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
