"""Run the eval set and report accuracy, latency and regressions.

What it measures, and why each one is here:

  accuracy          Did the model's query return the right ANSWER. The only
                    number that matters, and the one guardrails cannot give you.
  accuracy by tag   A single headline figure hides the failure that costs you.
                    `fan_out` is the tag to watch: those are the questions where
                    this project shipped a 3x error.
  unanswerable      Whether the model declines questions the schema cannot
                    answer. A confidently invented query is worse than a refusal.
  repair rate       How often the first query failed and the retry saved it. A
                    rising repair rate means the prompt is drifting.
  latency           p50 and p95, because a correct answer after 30 seconds is a
                    feature nobody uses.

Deliberately NOT measured: token cost. `src.nl_query._complete` discards the
usage object the API returns, so any cost figure here would be a guess. See the
README for the three-line change that makes it real.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.db import get_engine
from src.evals.cases import EvalCase, select_cases
from src.evals.compare import compare_results
from src.nl_query import NLResult, ask, run_sql


@dataclass
class CaseResult:
    id: str
    question: str
    tags: list[str]
    passed: bool
    reason: str
    latency_ms: int
    sql: str | None = None
    repaired: bool = False
    unanswerable: bool = False
    expected_rows: int = 0
    actual_rows: int = 0
    error: str | None = None


@dataclass
class EvalReport:
    started_at: str
    model: str
    total: int
    passed: int
    accuracy: float
    by_tag: dict
    repair_rate: float
    latency_p50_ms: int
    latency_p95_ms: int
    cases: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cases"] = [asdict(c) if not isinstance(c, dict) else c for c in self.cases]
        return d


def reference_result(case: EvalCase) -> pd.DataFrame | None:
    """Run the case's reference SQL through the same read-only path the model uses."""
    if not case.reference_sql:
        return None
    return run_sql(case.reference_sql)


def evaluate_case(case: EvalCase, result: NLResult, elapsed_ms: int) -> CaseResult:
    """Score one model answer against its reference."""
    base = dict(
        id=case.id,
        question=case.question,
        tags=list(case.tags),
        latency_ms=elapsed_ms,
        sql=result.sql,
        repaired=result.repaired,
        unanswerable=result.unanswerable,
        error=result.error,
    )

    # Questions the schema cannot answer: declining IS the correct answer.
    if case.expect_unanswerable:
        passed = bool(result.unanswerable)
        reason = "correctly declined" if passed else "answered an unanswerable question"
        return CaseResult(passed=passed, reason=reason, **base)

    if result.unanswerable:
        return CaseResult(passed=False, reason="declined an answerable question", **base)
    if result.df is None:
        return CaseResult(passed=False, reason=result.error or "no result returned", **base)

    try:
        expected = reference_result(case)
    except Exception as exc:  # a broken reference is a harness bug, not a model failure
        return CaseResult(passed=False, reason=f"REFERENCE SQL FAILED: {exc}", **base)

    cmp = compare_results(expected, result.df, mode=case.mode)
    return CaseResult(
        passed=cmp.passed,
        reason=cmp.reason,
        expected_rows=cmp.expected_rows,
        actual_rows=cmp.actual_rows,
        **base,
    )


def run_eval_set(
    only_tags: tuple[str, ...] = (),
    ids: tuple[str, ...] = (),
    model: str | None = None,
    verbose: bool = True,
) -> EvalReport:
    engine = get_engine()
    cases = select_cases(only_tags, ids)
    results: list[CaseResult] = []

    for i, case in enumerate(cases, 1):
        started = time.monotonic()
        try:
            nl = ask(case.question, engine=engine, model=model)
        except Exception as exc:
            nl = NLResult(question=case.question, error=f"{type(exc).__name__}: {exc}")
        elapsed_ms = int((time.monotonic() - started) * 1000)

        scored = evaluate_case(case, nl, elapsed_ms)
        results.append(scored)

        if verbose:
            mark = "PASS" if scored.passed else "FAIL"
            print(f"  [{i:>2}/{len(cases)}] {mark}  {case.id:<28} {elapsed_ms:>6}ms  {scored.reason}")

    return build_report(results, model=model or "default")


def build_report(results: list[CaseResult], model: str) -> EvalReport:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    latencies = sorted(r.latency_ms for r in results) or [0]

    by_tag: dict[str, dict] = {}
    for r in results:
        for tag in r.tags:
            slot = by_tag.setdefault(tag, {"total": 0, "passed": 0})
            slot["total"] += 1
            slot["passed"] += int(r.passed)
    for tag, slot in by_tag.items():
        slot["accuracy"] = round(slot["passed"] / slot["total"], 4) if slot["total"] else 0.0

    def pct(p: float) -> int:
        if not latencies:
            return 0
        idx = min(int(round(p * (len(latencies) - 1))), len(latencies) - 1)
        return latencies[idx]

    return EvalReport(
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=model,
        total=total,
        passed=passed,
        accuracy=round(passed / total, 4) if total else 0.0,
        by_tag=dict(sorted(by_tag.items())),
        repair_rate=round(sum(1 for r in results if r.repaired) / total, 4) if total else 0.0,
        latency_p50_ms=pct(0.50),
        latency_p95_ms=pct(0.95),
        cases=results,
    )


def format_report(report: EvalReport) -> str:
    lines = []
    lines.append("")
    lines.append("=" * 68)
    lines.append(f"  accuracy   {report.passed}/{report.total}  ({report.accuracy:.1%})")
    lines.append(f"  latency    p50 {report.latency_p50_ms}ms   p95 {report.latency_p95_ms}ms")
    lines.append(f"  repairs    {report.repair_rate:.1%} of queries needed a retry")
    lines.append("=" * 68)
    lines.append("")
    lines.append("  by tag")
    for tag, slot in report.by_tag.items():
        bar_width = int(slot["accuracy"] * 20)
        bar = "#" * bar_width + "." * (20 - bar_width)
        flag = "   <-- the one that shipped a bug" if tag == "fan_out" else ""
        lines.append(
            f"    {tag:<14} {bar}  {slot['passed']:>2}/{slot['total']:<2} {slot['accuracy']:>6.1%}{flag}"
        )
    failures = [c for c in report.cases if not c.passed]
    if failures:
        lines.append("")
        lines.append("  failures")
        for c in failures:
            lines.append(f"    {c.id:<28} {c.reason}")
            if c.sql:
                sql = c.sql if len(c.sql) < 150 else c.sql[:150] + " ..."
                lines.append(f"      {sql}")
    lines.append("")
    return "\n".join(lines)


def save_report(report: EvalReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path


def compare_to_baseline(report: EvalReport, baseline_path: Path) -> tuple[bool, str]:
    """Regression gate: fail if accuracy dropped, or any previously passing case now fails."""
    if not baseline_path.exists():
        return True, f"no baseline at {baseline_path}; nothing to compare"

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    was_passing = {c["id"] for c in baseline.get("cases", []) if c.get("passed")}
    now_passing = {c.id for c in report.cases if c.passed}
    regressed = sorted(was_passing - now_passing)

    msgs = [f"baseline accuracy {baseline.get('accuracy', 0):.1%} -> now {report.accuracy:.1%}"]
    if regressed:
        msgs.append("REGRESSED: " + ", ".join(regressed))
        return False, "\n".join(msgs)
    fixed = sorted(now_passing - was_passing)
    if fixed:
        msgs.append("newly passing: " + ", ".join(fixed))
    return True, "\n".join(msgs)
