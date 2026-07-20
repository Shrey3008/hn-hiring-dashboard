"""Create the schema and load the curated tech lookup.

Idempotent — safe to re-run. Reloads dim_tech from CSV every time so editing
the keyword list is just: edit CSV, re-run this, re-run the parser.

Usage:
    python scripts/init_db.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, insert, select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db import DATA_DIR, create_all, dim_tech, get_database_url, get_engine

DIM_TECH_CSV = DATA_DIR / "dim_tech.csv"

VALID_CATEGORIES = {"language", "framework", "data", "database", "cloud", "tooling"}


def load_dim_tech(engine) -> int:
    """Validate and load data/dim_tech.csv into the dim_tech table."""
    df = pd.read_csv(DIM_TECH_CSV)

    missing = {"tech", "pattern", "category"} - set(df.columns)
    if missing:
        raise ValueError(f"dim_tech.csv missing columns: {sorted(missing)}")

    dupes = df["tech"][df["tech"].duplicated()].tolist()
    if dupes:
        raise ValueError(f"Duplicate tech names in dim_tech.csv: {dupes}")

    bad_cats = set(df["category"]) - VALID_CATEGORIES
    if bad_cats:
        raise ValueError(
            f"Unknown categories {sorted(bad_cats)}; expected {sorted(VALID_CATEGORIES)}"
        )

    # Fail loudly here rather than mid-parse on 20k posts.
    for row in df.itertuples():
        try:
            re.compile(row.pattern, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"Bad regex for {row.tech!r}: {exc}") from exc

    with engine.begin() as conn:
        conn.execute(delete(dim_tech))
        conn.execute(insert(dim_tech), df.to_dict("records"))

    return len(df)


def main() -> None:
    engine = get_engine()
    print(f"Database: {get_database_url()}")

    create_all(engine)
    print("Schema created (existing tables left untouched).")

    n = load_dim_tech(engine)
    print(f"Loaded {n} tech keywords into dim_tech.")

    with engine.connect() as conn:
        by_cat = conn.execute(
            select(dim_tech.c.category, dim_tech.c.tech)
        ).fetchall()
    counts = pd.DataFrame(by_cat, columns=["category", "tech"]).groupby("category").size()
    for cat, count in counts.items():
        print(f"  {cat:<10} {count}")


if __name__ == "__main__":
    main()
