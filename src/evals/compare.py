"""Deciding whether a generated query returned the right answer.

The unit of correctness here is the RESULT, never the SQL text. Many different
queries answer the same question correctly, so string-matching SQL would fail
correct answers and teach the harness nothing. This module compares what came
back from the database.

Three things make naive DataFrame equality useless for this:

1.  Column NAMES differ. The model aliases things however it likes:
    `pct_posts`, `share`, `percentage`. All correct.
2.  Column and row ORDER differ. `SELECT month, pct` and `SELECT pct, month`
    are the same answer.
3.  Floats differ in the last digits. 4.94 vs 4.940000000000001.

So the comparison normalises values, then treats each row as an unordered bag
of values and the result as an unordered bag of rows.

The deliberate trade-off: this cannot tell `(month, posts)` from
`(posts, month)`. That is the price of column-order insensitivity, and it is
the same trade-off the Spider text-to-SQL benchmark makes for execution
accuracy. It has never produced a false pass in this eval set, because the
value TYPES differ between columns in every case here.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

# Floats are rounded to this many places before comparison.
#
# ONE, not two, and that is a calibration decision made against real output.
# The first live run scored a correct answer as wrong: the model returned 4.9
# where the reference returned 4.94, because the prompt's own few-shot examples
# use ROUND(..., 1). Semantically identical; a display-precision difference.
#
# Penalising that would make the accuracy number measure formatting rather than
# correctness. Rounding both sides to one place absorbs it while leaving real
# errors untouched: the fan-out bug this harness exists to catch reports 1.5
# against a true 4.9, which is nowhere near a rounding difference.
#
# The alternative -- matching the reference SQL to the prompt's rounding -- was
# rejected because it couples the eval set to one prompt's formatting habits.
DEFAULT_PRECISION = 1


def _round_half_up(x: float, precision: int) -> float:
    """Round the way SQL does, not the way Python does.

    Python's round() is half-to-EVEN: round(16.65, 1) == 16.6. SQLite's ROUND()
    is half-AWAY-FROM-ZERO: 16.7. Using Python's here made the harness disagree
    with every engine it compares against, and scored correct answers as wrong
    on the first live run.
    """
    try:
        q = Decimal(1).scaleb(-precision)
        return float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))
    except Exception:
        return round(float(x), precision)


def _normalise_value(value, precision: int):
    """Reduce one cell to something comparable across engines and aliases."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, float):
        return _round_half_up(value, precision)
    # pandas/numpy numeric scalars
    try:
        if pd.api.types.is_number(value):
            return _round_half_up(float(value), precision)
    except Exception:
        pass
    # Dates arrive as '2023-08-01' from SQLite and as date objects from a
    # DataFrame parse; str() collapses both.
    return str(value).strip()


def _normalise_frame(df: pd.DataFrame, precision: int) -> list[tuple]:
    """DataFrame -> sorted list of sorted row tuples."""
    if df is None:
        return []
    rows = []
    for row in df.itertuples(index=False, name=None):
        values = [_normalise_value(v, precision) for v in row]
        # Sorted by repr so mixed types (None, float, str) order deterministically.
        rows.append(tuple(sorted(values, key=repr)))
    return sorted(rows, key=repr)


@dataclass
class Comparison:
    passed: bool
    reason: str
    expected_rows: int
    actual_rows: int


def compare_results(
    expected: pd.DataFrame | None,
    actual: pd.DataFrame | None,
    mode: str = "contains",
    precision: int = DEFAULT_PRECISION,
) -> Comparison:
    """Compare a reference result against a model-generated one.

    mode="contains" (default)
        Every expected row must appear inside some actual row. Extra COLUMNS in
        the actual result are tolerated, because a model that answers
        "what share of posts mention Rust" and also returns the raw count is not
        wrong. Extra ROWS are not tolerated: the row count must match.

    mode="strict"
        The two results must be identical as bags of rows.

    "contains" is the default because the most common near-miss in practice is
    a correct answer carrying one helper column, and failing those would make
    the accuracy number measure formatting rather than correctness.
    """
    if expected is None:
        return Comparison(False, "no expected result", 0, 0 if actual is None else len(actual))
    if actual is None:
        return Comparison(False, "query returned nothing", len(expected), 0)

    exp_rows = _normalise_frame(expected, precision)
    act_rows = _normalise_frame(actual, precision)

    if len(exp_rows) != len(act_rows):
        return Comparison(
            False,
            f"row count differs: expected {len(exp_rows)}, got {len(act_rows)}",
            len(exp_rows),
            len(act_rows),
        )

    if mode == "strict":
        if exp_rows == act_rows:
            return Comparison(True, "exact match", len(exp_rows), len(act_rows))
        return Comparison(False, "values differ", len(exp_rows), len(act_rows))

    # contains: pair each expected row with an unused actual row that holds all
    # of its values. Greedy matching is sufficient — rows are already sorted, so
    # identical rows are adjacent and cannot be mismatched.
    remaining = list(act_rows)
    for want in exp_rows:
        hit = None
        for i, got in enumerate(remaining):
            pool = list(got)
            ok = True
            for value in want:
                if value in pool:
                    pool.remove(value)
                else:
                    ok = False
                    break
            if ok:
                hit = i
                break
        if hit is None:
            return Comparison(
                False,
                f"no actual row contains expected values {want!r}",
                len(exp_rows),
                len(act_rows),
            )
        remaining.pop(hit)

    return Comparison(True, "all expected rows matched", len(exp_rows), len(act_rows))
