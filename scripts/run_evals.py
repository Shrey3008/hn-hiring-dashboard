"""Run the natural-language query eval set.

    python scripts/run_evals.py --dry-run          # no API key needed: checks the eval set itself
    python scripts/run_evals.py                    # full run against Groq
    python scripts/run_evals.py --tags fan_out     # only the questions that shipped a bug
    python scripts/run_evals.py --save baseline.json
    python scripts/run_evals.py --baseline eval_results/baseline.json --min-accuracy 0.80

The last form is the CI gate: exit code 1 if accuracy falls below the floor or
any case that used to pass now fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evals.cases import CASES, all_tags, select_cases  # noqa: E402
from src.evals.runner import (  # noqa: E402
    compare_to_baseline,
    format_report,
    reference_result,
    run_eval_set,
    save_report,
)

DEFAULT_OUT = Path("eval_results")


def dry_run(only_tags: tuple[str, ...], ids: tuple[str, ...]) -> int:
    """Validate the eval set without calling the model.

    Every reference query must execute and return at least one row. A reference
    that silently returns nothing would make its case unfalsifiable: any empty
    model result would 'match'. This is the harness testing itself.
    """
    cases = select_cases(only_tags, ids)
    print(f"Dry run: validating {len(cases)} cases (no API calls)\n")
    problems = 0

    for case in cases:
        if case.expect_unanswerable:
            print(f"  SKIP  {case.id:<28} unanswerable by design, no reference SQL")
            continue
        if not case.reference_sql:
            print(f"  FAIL  {case.id:<28} answerable case with no reference SQL")
            problems += 1
            continue
        try:
            df = reference_result(case)
        except Exception as exc:
            print(f"  FAIL  {case.id:<28} reference SQL error: {exc}")
            problems += 1
            continue
        if df is None or df.empty:
            print(f"  FAIL  {case.id:<28} reference returned 0 rows (case is unfalsifiable)")
            problems += 1
            continue
        preview = ", ".join(str(v) for v in df.iloc[0].tolist()[:4])
        print(f"  ok    {case.id:<28} {len(df):>4} rows   first: {preview}")

    print()
    if problems:
        print(f"{problems} problem(s) in the eval set.")
        return 1
    print(f"All {len(cases)} cases valid.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tags", nargs="*", default=[], help=f"filter by tag: {', '.join(all_tags())}")
    p.add_argument("--ids", nargs="*", default=[], help="run only these case ids")
    p.add_argument("--model", default=None, help="override the Groq model id")
    p.add_argument("--dry-run", action="store_true", help="validate the eval set; no API calls")
    p.add_argument("--list", action="store_true", help="list cases and exit")
    p.add_argument("--save", type=Path, default=None, help="write the JSON report here")
    p.add_argument("--baseline", type=Path, default=None, help="compare against this report")
    p.add_argument("--min-accuracy", type=float, default=None, help="fail below this accuracy (0-1)")
    args = p.parse_args()

    tags, ids = tuple(args.tags), tuple(args.ids)

    if args.list:
        for c in CASES:
            kind = "UNANSWERABLE" if c.expect_unanswerable else ",".join(c.tags)
            print(f"  {c.id:<28} [{kind}]  {c.question}")
        return 0

    if args.dry_run:
        return dry_run(tags, ids)

    print("Running eval set against the model. This makes one API call per case.\n")
    report = run_eval_set(only_tags=tags, ids=ids, model=args.model)
    print(format_report(report))

    out = args.save or (DEFAULT_OUT / f"eval_{report.started_at.replace(':', '').replace('-', '')}.json")
    saved = save_report(report, Path(out))
    print(f"  report saved to {saved}")

    exit_code = 0

    if args.baseline:
        ok, msg = compare_to_baseline(report, Path(args.baseline))
        print(f"\n  {msg}")
        if not ok:
            exit_code = 1

    if args.min_accuracy is not None and report.accuracy < args.min_accuracy:
        print(f"\n  accuracy {report.accuracy:.1%} is below the floor of {args.min_accuracy:.1%}")
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
