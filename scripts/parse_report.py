"""Parse-quality report.

Measures extraction rates instead of assuming them, and surfaces the samples
most likely to reveal a broken rule. Run after every parser change.

    python scripts/parse_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db import get_engine

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def pct(n: int, d: int) -> str:
    return f"{n / d * 100:5.1f}%" if d else "  n/a"


def main() -> None:
    engine = get_engine()
    with engine.connect() as conn:
        jobs = pd.read_sql(text("SELECT * FROM jobs"), conn)
        tech = pd.read_sql(text("SELECT * FROM job_tech"), conn)
        threads = pd.read_sql(text("SELECT * FROM threads ORDER BY month"), conn)
        dim = pd.read_sql(text("SELECT * FROM dim_tech"), conn)

    # SQLite has no boolean type and hands back 0/1 ints, which silently break
    # mask indexing and make `~col` a bitwise negation. Postgres returns real
    # bools, so cast here to keep this report backend-agnostic.
    for col in ("has_salary", "parsed_structured", "is_remote"):
        if col in jobs.columns:
            jobs[col] = jobs[col].astype(bool)

    n = len(jobs)
    if n == 0:
        print("No parsed jobs. Run scripts/run_etl.py first.")
        return

    rule("COVERAGE")
    print(f"Threads:            {len(threads)}")
    print(f"Months:             {threads['month'].min()} .. {threads['month'].max()}")
    print(f"Posts parsed:       {n:,}")
    print(f"Avg posts/thread:   {n / len(threads):.0f}")

    rule("FIELD EXTRACTION RATES")
    fields = {
        "structured header": jobs["parsed_structured"].sum(),
        "company": jobs["company"].notna().sum(),
        "role_title": jobs["role_title"].notna().sum(),
        "location_raw": jobs["location_raw"].notna().sum(),
        "work_mode known": (jobs["work_mode"] != "unknown").sum(),
        "salary range (USD)": jobs["has_salary"].sum(),
    }
    for label, count in fields.items():
        print(f"  {label:<22} {count:>7,}  {pct(int(count), n)}")

    rule("WORK MODE DISTRIBUTION")
    wm = jobs["work_mode"].value_counts()
    for mode, count in wm.items():
        print(f"  {mode:<10} {count:>7,}  {pct(int(count), n)}")

    rule("WORK MODE BY MONTH (share of posts)")
    by_month = (
        jobs.groupby(["month", "work_mode"]).size().unstack(fill_value=0)
    )
    share = (by_month.div(by_month.sum(axis=1), axis=0) * 100).round(1)
    print(share.to_string())

    rule("TECH COVERAGE")
    posts_with_tech = tech["post_id"].nunique()
    print(f"Posts with >=1 tech match: {posts_with_tech:,}  {pct(posts_with_tech, n)}")
    print(f"Total mentions:            {len(tech):,}")
    print(f"Mentions per post (avg):   {len(tech) / n:.1f}")
    unused = sorted(set(dim["tech"]) - set(tech["tech"]))
    print(f"Keywords never matched ({len(unused)}): {', '.join(unused) if unused else 'none'}")

    rule("TOP 25 TECHNOLOGIES")
    top = tech["tech"].value_counts().head(25)
    for t, count in top.items():
        bar = "#" * int(count / top.iloc[0] * 40)
        print(f"  {t:<16} {count:>6,}  {pct(int(count), n)}  {bar}")

    rule("SUSPICIOUSLY COMMON TECH (possible false positives)")
    print("Anything matching >55% of ALL posts is more likely a bad regex than a")
    print("real trend. Verify these against raw text before trusting them.\n")
    suspicious = top[top / n > 0.55]
    if len(suspicious):
        for t, count in suspicious.items():
            print(f"  !! {t:<16} {count:>6,}  {pct(int(count), n)}")
    else:
        print("  none")

    rule("SALARY (disclosed USD ranges only)")
    sal = jobs[jobs["has_salary"]]
    if len(sal):
        print(f"Posts with a parsed range: {len(sal):,}  {pct(len(sal), n)}")
        print(f"  median min: ${sal['salary_min'].median():,.0f}")
        print(f"  median max: ${sal['salary_max'].median():,.0f}")
        print(f"  p10 / p90 of midpoint: "
              f"${((sal['salary_min'] + sal['salary_max']) / 2).quantile(0.1):,.0f}"
              f" / ${((sal['salary_min'] + sal['salary_max']) / 2).quantile(0.9):,.0f}")
        print("\nNote: non-USD postings are intentionally excluded, so this")
        print("undercounts true disclosure. Treat as a USD-only floor.")
    else:
        print("No salary ranges parsed.")

    rule("TOP 15 COMPANIES (sanity-check normalization)")
    print("Repeat posters are expected. Sentence fragments here mean the")
    print("company rule is leaking prose.\n")
    for c, count in jobs["company"].value_counts().head(15).items():
        print(f"  {count:>4}  {c}")

    rule("SAMPLE: UNSTRUCTURED POSTS (no pipe header)")
    unstr = jobs[~jobs["parsed_structured"]]
    print(f"{len(unstr):,} posts ({pct(len(unstr), n)}). Company via prose fallback:")
    got = unstr["company"].notna().sum()
    print(f"  recovered a company for {got:,} of them ({pct(int(got), len(unstr))})\n")
    print(unstr[["company", "work_mode", "has_salary"]].head(10).to_string(index=False))

    rule("SAMPLE: STRUCTURED POSTS WITH NO ROLE EXTRACTED")
    miss = jobs[jobs["parsed_structured"] & jobs["role_title"].isna()]
    print(f"{len(miss):,} posts. If this is large, ROLE_RE is too narrow.\n")
    print(miss[["company", "location_raw", "work_mode"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
