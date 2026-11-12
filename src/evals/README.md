# Eval harness for the natural-language query feature

Guardrails stop *unsafe* SQL. Nothing in them stops *incorrect* SQL. This
project shipped a query that was safe, valid, fast and wrong by 3x, and the only
reason it was caught is that a validated reference query existed to compare
against. This harness turns that one-off comparison into a measurement.

## Run it

```bash
python scripts/run_evals.py --dry-run        # validates the eval set, no API key needed
python scripts/test_eval_harness.py          # calibrates the harness, no API key needed
python scripts/run_evals.py                  # the real thing: one API call per case
python scripts/run_evals.py --tags fan_out   # only the questions that shipped a bug
```

Regression gate for CI:

```bash
python scripts/run_evals.py \
    --baseline eval_results/baseline.json \
    --min-accuracy 0.80
```

Exit code 1 if accuracy drops below the floor, or if any case that used to pass
now fails. Point it at a saved report to create the baseline:
`python scripts/run_evals.py --save eval_results/baseline.json`.

## How correctness is decided

**Results are compared, never SQL text.** Many different queries answer the same
question correctly, so string-matching would fail correct answers.

Expected answers are stored as **reference SQL**, not literal numbers. Numbers
rot the moment the ETL fetches a new month; reference SQL stays true. Both the
reference and the model's query run through the same read-only connection
against the same database, so the only variable is the query.

The comparator is lenient about presentation and strict about values:

| Tolerated | Rejected |
|---|---|
| different column names (`pct` vs `share`) | different values |
| different column order | different row counts |
| different row order | missing rows |
| float noise (rounded to 2dp) | empty results |
| an extra helper column (`contains` mode) | |

Known trade-off: column-order insensitivity means the comparator cannot tell
`(month, posts)` from `(posts, month)`. That is the same compromise the Spider
text-to-SQL benchmark makes for execution accuracy, and no case in this set has
two columns of the same type where it could bite.

## What gets measured

- **accuracy** overall, and **per tag**. A single headline figure hides the
  failure that costs you: watch `fan_out`.
- **unanswerable handling** — whether the model declines questions the schema
  cannot answer. A confidently invented query is worse than a refusal.
- **repair rate** — how often the first query failed and the retry rescued it.
  Rising repair rate means the prompt is drifting.
- **latency** p50 and p95.

Not measured: **token cost**. `src/nl_query.py::_complete` discards the usage
object the API returns, so any figure here would be a guess. To make it real,
return `resp.usage` alongside the content and record
`prompt_tokens` / `completion_tokens` per case.

## Calibration: the part most harnesses skip

A harness nobody calibrated is decoration. `scripts/test_eval_harness.py`
replaces the model with three fakes and asserts the harness reacts correctly:

| Fake model | Expected |
|---|---|
| **oracle** — returns the reference SQL every time | scores 100% |
| **saboteur** — reintroduces the join fan-out on share queries | those cases FAIL; `fan_out` tag accuracy drops; nothing else moves |
| **over-eager** — invents answers to unanswerable questions | only the unanswerable cases fail |

Measured on this eval set: the oracle scores **25/25**, the saboteur drops
overall accuracy to **84%** with the `fan_out` tag at **56%** and exactly the
four sabotaged cases failing, and the over-eager model fails only the two
unanswerable cases. Nineteen assertions, no API key, runs in seconds.

`--dry-run` is the other half of the calibration: it asserts every reference
query executes **and returns at least one row**. A reference that silently
returns nothing makes its case unfalsifiable, because any empty model result
would "match".

## The eval set

25 cases in `cases.py`, tagged:

| Tag | Cases | What it probes |
|---|---|---|
| `fan_out` | 9 | joining `job_tech` then computing a share — where the bug lives |
| `share` | 7 | percentage questions generally |
| `basic` | 6 | filters, counts, lookups |
| `time_series` | 5 | per-month grouping |
| `ranking` | 3 | top-N with correct tie handling |
| `join` | 4 | multi-table questions |
| `salary` | 3 | the ~17% of posts with parseable USD ranges |
| `spelling` | 2 | exact technology names — the prompt injects all 124 because the model otherwise guesses `postgres` and returns zero rows |
| `unanswerable` | 2 | questions the schema cannot answer |

Adding a case: write the question as a user would ask it, write the reference
SQL, tag it, and run `--dry-run` to confirm the reference returns rows.

## Next

- Capture real token usage and report cost per run.
- Record every generated query so prompt changes can be diffed, not just scored.
- Run the set against two models and compare accuracy against latency and cost.
