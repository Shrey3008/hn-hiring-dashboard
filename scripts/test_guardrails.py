"""Guardrail tests for the LLM query feature.

No API key needed — these test what happens to SQL *after* the model returns
it, which is the part that has to hold when the model misbehaves or a user
talks it into something.

    python scripts/test_guardrails.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.nl_query import UnsafeSQL, clean_sql, enforce_limit, run_sql, validate_sql

# (label, input, should_be_rejected)
CASES = [
    # --- legitimate queries ---
    ("plain select", "SELECT company FROM jobs LIMIT 5", False),
    ("with CTE", "WITH x AS (SELECT 1 AS n) SELECT n FROM x", False),
    ("fenced markdown", "```sql\nSELECT COUNT(*) FROM jobs\n```", False),
    ("lowercase", "select count(*) from jobs", False),
    ("trailing semicolon", "SELECT 1;", False),
    ("leading whitespace", "   \n  SELECT 1", False),

    # --- direct writes ---
    ("DROP", "DROP TABLE jobs", True),
    ("DELETE", "DELETE FROM jobs", True),
    ("INSERT", "INSERT INTO jobs VALUES (1)", True),
    ("UPDATE", "UPDATE jobs SET company = 'x'", True),
    ("ALTER", "ALTER TABLE jobs ADD COLUMN x INT", True),
    ("CREATE", "CREATE TABLE evil (id INT)", True),
    ("TRUNCATE", "TRUNCATE TABLE jobs", True),

    # --- stacked statements ---
    ("stacked drop", "SELECT 1; DROP TABLE jobs", True),
    ("stacked delete", "SELECT company FROM jobs; DELETE FROM jobs", True),

    # --- hidden behind comments (the reason comments are stripped first) ---
    ("line comment hides drop", "SELECT 1 --\nDROP TABLE jobs", True),
    ("block comment hides drop", "SELECT 1 /* x */; DROP TABLE jobs", True),
    ("comment-only DDL", "/* SELECT */ DROP TABLE jobs", True),

    # --- SQLite-specific escapes ---
    ("PRAGMA", "PRAGMA table_info(jobs)", True),
    ("ATTACH", "ATTACH DATABASE '/tmp/e.db' AS e", True),
    ("load_extension", "SELECT load_extension('evil.so')", True),
    ("VACUUM", "VACUUM", True),

    # --- transaction control ---
    ("BEGIN", "BEGIN; DELETE FROM jobs", True),

    # --- degenerate ---
    ("empty", "", True),
    ("prose only", "I cannot answer that question.", True),
    ("fence with prose", "```\nHere is your query\n```", True),
]


def main() -> int:
    failures = []

    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)
    for label, sql, should_reject in CASES:
        try:
            validate_sql(sql)
            rejected = False
            detail = "accepted"
        except UnsafeSQL as exc:
            rejected = True
            detail = str(exc)

        ok = rejected == should_reject
        mark = "ok  " if ok else "FAIL"
        want = "reject" if should_reject else "accept"
        print(f"  [{mark}] {label:<28} want={want:<6} {detail}")
        if not ok:
            failures.append(label)

    print()
    print("=" * 70)
    print("ROW CAP")
    print("=" * 70)
    # The cap must survive a query that carries its own larger LIMIT.
    capped = enforce_limit("SELECT post_id FROM jobs LIMIT 10000", cap=5)
    print(f"  wrapped: {capped}")
    df = run_sql("SELECT post_id FROM jobs LIMIT 10000", cap=5)
    print(f"  rows returned with cap=5: {len(df)}")
    if len(df) != 5:
        failures.append("row cap not enforced")

    df2 = run_sql("SELECT post_id FROM jobs", cap=3)
    print(f"  uncapped query with cap=3: {len(df2)} rows")
    if len(df2) != 3:
        failures.append("row cap on uncapped query")

    print()
    print("=" * 70)
    print("READ-ONLY CONNECTION (the layer that actually matters)")
    print("=" * 70)
    # Bypass validation entirely — this is what must hold if every earlier
    # layer is defeated.
    try:
        run_sql("SELECT 1")  # sanity: reads work
        print("  [ok  ] reads succeed")
    except Exception as exc:
        print(f"  [FAIL] reads blocked: {exc}")
        failures.append("read blocked")

    from sqlalchemy import text as _text

    from src.db import get_engine

    try:
        eng = get_engine(readonly=True)
        with eng.connect() as conn:
            conn.execute(_text("DELETE FROM jobs"))
            conn.commit()
        print("  [FAIL] write succeeded on read-only connection")
        failures.append("read-only bypassed")
    except Exception as exc:
        print(f"  [ok  ] write refused by driver: {type(exc).__name__}")

    print()
    print("=" * 70)
    if failures:
        print(f"FAILED: {len(failures)} -> {failures}")
        return 1
    print(f"All {len(CASES) + 4} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
