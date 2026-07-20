"""Database engine and schema.

Everything goes through SQLAlchemy Core so that moving from local SQLite to
Neon Postgres is a DATABASE_URL change and nothing else. No ORM — the ETL is
dataframe-shaped, so Core + pandas.to_sql is a better fit than mapped classes.

Layering:
    threads / raw_posts   raw ingest, never edited after write
    jobs / job_tech       parsed output, safe to drop and rebuild
    dim_tech              curated keyword lookup, loaded from data/dim_tech.csv
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
)
from sqlalchemy.engine import Engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Full local build (gitignored) — what the ETL writes.
DEFAULT_DB_PATH = DATA_DIR / "hn_hiring.db"
# Committed read-only snapshot — what a fresh clone and the deployed app get.
SNAPSHOT_DB_PATH = DATA_DIR / "snapshot.db"

metadata = MetaData()


# --------------------------------------------------------------------------
# Raw layer — a faithful copy of what the API returned.
# Re-parsing must never require re-fetching, so these tables are append-only
# and parse.py treats them as read-only input.
# --------------------------------------------------------------------------

threads = Table(
    "threads",
    metadata,
    Column("thread_id", Integer, primary_key=True),
    Column("month", Date, nullable=False, index=True),
    Column("title", Text, nullable=False),
    Column("post_count", Integer, nullable=False, default=0),
    Column("fetched_at", DateTime, nullable=False),
)

raw_posts = Table(
    "raw_posts",
    metadata,
    Column("post_id", Integer, primary_key=True),
    Column("thread_id", Integer, ForeignKey("threads.thread_id"), nullable=False),
    Column("author", String(64)),
    Column("created_at", DateTime),
    Column("text_html", Text),
    Column("text_plain", Text),
    Column("ingested_at", DateTime, nullable=False),
)
Index("ix_raw_posts_thread", raw_posts.c.thread_id)


# --------------------------------------------------------------------------
# Curated layer — rebuilt from raw_posts on every parse run.
# --------------------------------------------------------------------------

jobs = Table(
    "jobs",
    metadata,
    Column("post_id", Integer, ForeignKey("raw_posts.post_id"), primary_key=True),
    Column("month", Date, nullable=False, index=True),
    Column("company", String(200)),
    Column("role_title", String(300)),
    Column("location_raw", String(300)),
    # work_mode: remote | hybrid | onsite | unknown
    Column("work_mode", String(16), nullable=False, default="unknown", index=True),
    Column("is_remote", Boolean, nullable=False, default=False),
    Column("salary_min", Float),
    Column("salary_max", Float),
    Column("has_salary", Boolean, nullable=False, default=False),
    Column("char_count", Integer),
    # True when the post followed the "Company | Role | Location" convention.
    # Tracked so the parse-quality report can show extraction confidence.
    Column("parsed_structured", Boolean, nullable=False, default=False),
)

# Tidy long table: one row per (post, tech). Backs every tech trend chart.
job_tech = Table(
    "job_tech",
    metadata,
    Column("post_id", Integer, ForeignKey("jobs.post_id"), nullable=False),
    Column("tech", String(64), ForeignKey("dim_tech.tech"), nullable=False),
)
Index("ix_job_tech_post", job_tech.c.post_id)
Index("ix_job_tech_tech", job_tech.c.tech)

dim_tech = Table(
    "dim_tech",
    metadata,
    Column("tech", String(64), primary_key=True),
    # Case-insensitive regex used to detect this tech in post text.
    Column("pattern", String(300), nullable=False),
    # language | framework | data | database | cloud | tooling
    Column("category", String(32), nullable=False, index=True),
)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


def get_database_url() -> str:
    """Resolve the database, in precedence order.

    1. DATABASE_URL — set this to a Neon connection string to switch backends:
       postgresql+psycopg://user:pass@host/db?sslmode=require
    2. data/hn_hiring.db — the full local build, if the ETL has been run.
    3. data/snapshot.db — the committed read-only snapshot.

    Order matters: a developer who has run the ETL gets their own fresh data,
    while a fresh clone or the deployed app falls through to the snapshot and
    works with no setup at all.
    """
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    DATA_DIR.mkdir(exist_ok=True)
    if DEFAULT_DB_PATH.exists():
        return f"sqlite:///{DEFAULT_DB_PATH}"
    if SNAPSHOT_DB_PATH.exists():
        return f"sqlite:///{SNAPSHOT_DB_PATH}"
    # Neither exists yet — return the ETL target so init_db/run_etl can create it.
    return f"sqlite:///{DEFAULT_DB_PATH}"


def get_engine(echo: bool = False, readonly: bool = False) -> Engine:
    """Create an engine.

    readonly=True is a real guard for SQLite (the driver refuses writes at the
    file level), which is what the LLM query feature runs against. On Postgres
    it is advisory only — enforce it there with a read-only role instead.
    """
    url = get_database_url()

    if readonly and url.startswith("sqlite:///"):
        db_path = url.replace("sqlite:///", "", 1)
        if not Path(db_path).exists():
            raise FileNotFoundError(
                f"No database at {db_path}. Run scripts/init_db.py first."
            )
        url = f"sqlite:///file:{db_path}?mode=ro&uri=true"

    return create_engine(url, echo=echo, future=True)


def create_all(engine: Engine) -> None:
    metadata.create_all(engine)


def drop_curated(engine: Engine) -> None:
    """Drop only the parsed tables, leaving raw ingest intact.

    This is what makes parser iteration cheap: change a regex, rebuild the
    curated layer in seconds, never touch the API again.
    """
    metadata.drop_all(engine, tables=[job_tech, jobs])
    metadata.create_all(engine, tables=[jobs, job_tech])
