"""Natural-language querying over the curated tables, via Groq.

Flow: question -> LLM -> SQL -> guardrails -> read-only execution -> DataFrame.

The schema block in the prompt is GENERATED FROM SQLAlchemy METADATA, never
hand-written. A hand-maintained copy drifts the moment a column is renamed, and
a prompt that disagrees with the database produces confidently wrong SQL rather
than an error — the worst failure mode available here.

Defense in depth, because a prompt instruction is not a security control. An
LLM told "only write SELECT" will still occasionally emit something else, and a
user can ask it to. The layers, in order:

    1. strip markdown fences and SQL comments (comments can hide a second
       statement from a naive keyword scan)
    2. reject anything but a single statement
    3. require SELECT or WITH as the first token
    4. reject DDL/DML keywords outright
    5. wrap the whole query to force a row cap
    6. execute on a connection the driver itself refuses to write through
    7. abort on a statement timeout

Layer 6 is the one that actually matters. Layers 1-5 are cheap and catch honest
mistakes; only the read-only connection is a real boundary.
"""

from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.dialects import sqlite as sqlite_dialect
from sqlalchemy.engine import Engine
from sqlalchemy.schema import CreateTable

from src.db import dim_tech, get_engine, job_tech, jobs, raw_posts, threads

DEFAULT_MODEL = "llama-3.3-70b-versatile"
ROW_CAP = 500
TIMEOUT_SECONDS = 10

# Tables the LLM is allowed to see and query. raw_posts is included because
# full-text questions ("posts mentioning visa sponsorship") need it, but its
# text columns are huge — the prompt warns against SELECT *.
EXPOSED_TABLES = [jobs, job_tech, dim_tech, threads, raw_posts]


# --------------------------------------------------------------------------
# Guardrails
# --------------------------------------------------------------------------

FORBIDDEN_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach"
    r"|pragma|vacuum|reindex|grant|revoke|copy|merge|upsert|begin|commit"
    r"|rollback|savepoint|load_extension)\b",
    re.IGNORECASE,
)
FENCE_RE = re.compile(r"^\s*```(?:sql)?\s*|\s*```\s*$", re.IGNORECASE)
LINE_COMMENT_RE = re.compile(r"--[^\n]*")
BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


class UnsafeSQL(ValueError):
    """The model produced SQL that failed a guardrail."""


# The model sometimes wraps its refusal as SQL — `SELECT 'UNANSWERABLE'` — which
# is harmless but renders as a one-row table instead of a message. Seen in
# practice when a prompt-injection attempt gets refused on the repair attempt.
UNANSWERABLE_SQL_RE = re.compile(r"^select\s+'?unanswerable'?\s*$", re.IGNORECASE)


def is_unanswerable(raw: str) -> bool:
    stripped = raw.strip().strip("`").strip()
    return (
        stripped.upper().startswith("UNANSWERABLE")
        or bool(UNANSWERABLE_SQL_RE.match(clean_sql(raw)))
    )


def clean_sql(raw: str) -> str:
    """Strip markdown fencing and comments, normalise whitespace.

    Comments are removed BEFORE the keyword scan, not after: `SELECT 1 --\\nDROP`
    hides a second statement from a scanner that looks at raw text.
    """
    sql = FENCE_RE.sub("", raw.strip())
    sql = BLOCK_COMMENT_RE.sub(" ", sql)
    sql = LINE_COMMENT_RE.sub(" ", sql)
    return re.sub(r"\s+", " ", sql).strip().rstrip(";").strip()


def validate_sql(raw: str) -> str:
    """Return cleaned, single, read-only SQL or raise UnsafeSQL."""
    sql = clean_sql(raw)

    if not sql:
        raise UnsafeSQL("The model returned an empty query.")
    if ";" in sql:
        raise UnsafeSQL("Multiple statements are not allowed.")
    if not re.match(r"^\s*(select|with)\b", sql, re.IGNORECASE):
        raise UnsafeSQL("Only SELECT queries are allowed.")

    hit = FORBIDDEN_RE.search(sql)
    if hit:
        raise UnsafeSQL(f"Disallowed keyword: {hit.group(0).upper()}")

    return sql


def enforce_limit(sql: str, cap: int = ROW_CAP) -> str:
    """Wrap the query so it cannot return more than `cap` rows.

    Wrapping rather than appending LIMIT: appending is unsafe with CTEs,
    compound SELECTs, and queries that already carry their own LIMIT, and
    regex-matching for a *top-level* LIMIT means parsing SQL. Wrapping caps
    every one of those cases uniformly, and a smaller outer LIMIT still wins
    over a larger inner one.
    """
    return f"SELECT * FROM ({sql}) AS _capped LIMIT {int(cap)}"


@contextmanager
def _statement_timeout(conn, seconds: int):
    """Abort a runaway query.

    SQLite has no statement timeout, so a progress handler polls a deadline and
    aborts by returning non-zero. Postgres gets a real statement_timeout.
    """
    dialect = conn.engine.dialect.name
    if dialect == "postgresql":
        conn.execute(text(f"SET LOCAL statement_timeout = {int(seconds * 1000)}"))
        yield
        return

    raw = getattr(conn.connection, "dbapi_connection", None) or conn.connection
    if not hasattr(raw, "set_progress_handler"):
        yield  # unknown driver; the row cap is still in force
        return

    deadline = time.monotonic() + seconds
    raw.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        yield
    finally:
        raw.set_progress_handler(None, 0)


def run_sql(sql: str, cap: int = ROW_CAP, timeout: int = TIMEOUT_SECONDS) -> pd.DataFrame:
    """Execute validated SQL on a read-only connection."""
    engine = get_engine(readonly=True)
    wrapped = enforce_limit(sql, cap)
    with engine.connect() as conn:
        with _statement_timeout(conn, timeout):
            return pd.read_sql(text(wrapped), conn)


# --------------------------------------------------------------------------
# Schema context
# --------------------------------------------------------------------------


def schema_ddl() -> str:
    """Real DDL, compiled from the live metadata. Cannot drift from the schema."""
    dialect = sqlite_dialect.dialect()
    return "\n".join(
        str(CreateTable(t).compile(dialect=dialect)).strip() for t in EXPOSED_TABLES
    )


def schema_facts(engine: Engine) -> str:
    """Values the DDL cannot express: enum contents, ranges, exact tech spellings.

    The tech list is included in full and verbatim. Without it the model guesses
    'postgres' or 'Postgresql' and silently returns zero rows — a wrong answer
    that looks like a valid one.
    """
    with engine.connect() as conn:
        modes = [r[0] for r in conn.execute(
            select(jobs.c.work_mode).distinct().order_by(jobs.c.work_mode))]
        cats = [r[0] for r in conn.execute(
            select(dim_tech.c.category).distinct().order_by(dim_tech.c.category))]
        techs = [r[0] for r in conn.execute(select(dim_tech.c.tech).order_by(dim_tech.c.tech))]
        lo, hi = conn.execute(text("SELECT MIN(month), MAX(month) FROM jobs")).fetchone()
        n = conn.execute(text("SELECT COUNT(*) FROM jobs")).scalar()

    return f"""
Rows in `jobs`: {n:,}. One row per job post.
`month` is the first day of the thread's month, stored as 'YYYY-MM-DD' text: {lo} .. {hi}.
`work_mode` is exactly one of: {', '.join(repr(m) for m in modes)}.
`dim_tech.category` is one of: {', '.join(repr(c) for c in cats)}.
`has_salary` / `is_remote` / `parsed_structured` are 0 or 1, not true/false.
`jobs.salary_min`/`salary_max` are USD and NULL unless has_salary = 1.
`job_tech` is a long table: one row per (post_id, tech). Join it to `jobs` on post_id.
`raw_posts.text_plain` holds the full post text; never SELECT it without a LIMIT.

Technology names must be matched EXACTLY as spelled here:
{', '.join(techs)}
""".strip()


FEW_SHOT = """
Q: Which companies mentioned Rust in 2026?
A: SELECT DISTINCT j.company FROM jobs j JOIN job_tech t ON t.post_id = j.post_id WHERE t.tech = 'Rust' AND j.month >= '2026-01-01' AND j.company IS NOT NULL ORDER BY j.company

Q: How did remote hiring change over time?
A: SELECT month, ROUND(100.0 * SUM(CASE WHEN work_mode = 'remote' THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_remote FROM jobs GROUP BY month ORDER BY month

Q: How has the share of posts mentioning Rust changed each month?
A: WITH monthly_total AS (SELECT month, COUNT(*) AS total FROM jobs GROUP BY month) SELECT j.month, ROUND(COUNT(DISTINCT t.post_id) * 100.0 / m.total, 1) AS pct_posts FROM job_tech t JOIN jobs j ON j.post_id = t.post_id JOIN monthly_total m ON m.month = j.month WHERE t.tech = 'Rust' GROUP BY j.month, m.total ORDER BY j.month

Q: What are the top 10 databases mentioned?
A: SELECT t.tech, COUNT(DISTINCT t.post_id) AS posts FROM job_tech t JOIN dim_tech d ON d.tech = t.tech WHERE d.category = 'database' GROUP BY t.tech ORDER BY posts DESC LIMIT 10

Q: Which companies posted the most jobs?
A: SELECT company, COUNT(*) AS posts FROM jobs WHERE company IS NOT NULL GROUP BY company ORDER BY posts DESC LIMIT 20

Q: What is the average salary for posts mentioning Kubernetes?
A: SELECT AVG((j.salary_min + j.salary_max) / 2.0) AS avg_midpoint, COUNT(*) AS n FROM jobs j JOIN job_tech t ON t.post_id = j.post_id WHERE t.tech = 'Kubernetes' AND j.has_salary = 1
""".strip()


def build_system_prompt(engine: Engine) -> str:
    return f"""You translate questions about a Hacker News hiring dataset into a single SQLite SELECT query.

SCHEMA
{schema_ddl()}

FACTS
{schema_facts(engine)}

EXAMPLES
{FEW_SHOT}

RULES
- Return ONE SQLite SELECT statement and nothing else. No prose, no markdown, no semicolon.
- Read-only. Never write INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or PRAGMA.
- Prefer share/percentage over raw counts when comparing across months, because
  monthly thread sizes differ.
- CRITICAL — row fan-out: joining `job_tech` to `jobs` multiplies each post into
  one row per technology (about 2.7x). So COUNT(*) after that join counts tech
  mentions, NOT posts. When you need a share of POSTS, take the denominator from
  a separate aggregate over `jobs` alone (see the Rust-per-month example) and
  count the numerator with COUNT(DISTINCT post_id). Dividing by COUNT(*) after
  joining job_tech understates every percentage by roughly 3x.
- Exclude NULL company when listing companies.
- Alias aggregates with clear names.
- SQLite has no MEDIAN, STDDEV, or PERCENTILE function. If asked for a median,
  use AVG and name the column so the answer is not mistaken for a median.
- If the question cannot be answered from this schema, return exactly: UNANSWERABLE
"""


# --------------------------------------------------------------------------
# Groq
# --------------------------------------------------------------------------


class MissingAPIKey(RuntimeError):
    pass


def get_api_key() -> str:
    """Read the key from Streamlit secrets or the environment. Never hard-coded."""
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        import streamlit as st

        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    raise MissingAPIKey(
        "No Groq API key found. Add GROQ_API_KEY to .streamlit/secrets.toml "
        "(gitignored) or export it in your shell. Get a free key at console.groq.com."
    )


def get_model() -> str:
    model = os.environ.get("GROQ_MODEL")
    if model:
        return model
    try:
        import streamlit as st

        if "GROQ_MODEL" in st.secrets:
            return st.secrets["GROQ_MODEL"]
    except Exception:
        pass
    return DEFAULT_MODEL


def available_models() -> list[str]:
    """List models this key can use.

    Groq retires models fairly often, so a hard-coded id eventually 404s. This
    turns that into a fixable message instead of a mystery failure.
    """
    from groq import Groq

    client = Groq(api_key=get_api_key())
    return sorted(m.id for m in client.models.list().data)


def _complete(prompt: str, question: str, model: str) -> str:
    from groq import Groq

    client = Groq(api_key=get_api_key())
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        temperature=0,  # SQL generation wants determinism, not creativity
        max_tokens=800,
    )
    return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


@dataclass
class NLResult:
    question: str
    sql: str | None = None
    df: pd.DataFrame | None = None
    error: str | None = None
    repaired: bool = False
    unanswerable: bool = False
    attempts: list[str] = field(default_factory=list)


def ask(question: str, engine: Engine | None = None, model: str | None = None) -> NLResult:
    """Question -> SQL -> results, with one repair attempt on failure."""
    engine = engine or get_engine()
    model = model or get_model()
    result = NLResult(question=question)

    prompt = build_system_prompt(engine)
    raw = _complete(prompt, question, model)
    result.attempts.append(raw)

    if is_unanswerable(raw):
        result.unanswerable = True
        result.error = "That question can't be answered from this dataset."
        return result

    try:
        sql = validate_sql(raw)
        result.sql = sql
        result.df = run_sql(sql)
        return result
    except Exception as first_error:
        # One repair attempt. Not a loop: if the model cannot fix it with the
        # error in hand, more turns burn quota without converging.
        repair = (
            f"{question}\n\n"
            f"Your previous query failed.\n"
            f"Query: {clean_sql(raw)}\n"
            f"Error: {first_error}\n"
            f"Return one corrected SQLite SELECT statement and nothing else."
        )
        try:
            raw2 = _complete(prompt, repair, model)
            result.attempts.append(raw2)
            if is_unanswerable(raw2):
                result.unanswerable = True
                result.error = "That question can't be answered from this dataset."
                result.sql = None
                return result
            sql2 = validate_sql(raw2)
            result.sql = sql2
            result.df = run_sql(sql2)
            result.repaired = True
            return result
        except Exception as second_error:
            result.error = f"{first_error}  (repair also failed: {second_error})"
            if result.sql is None:
                result.sql = clean_sql(raw)
            return result
