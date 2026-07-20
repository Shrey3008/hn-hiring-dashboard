"""Build the committed read-only snapshot from the local development database.

Two databases, on purpose:

    data/hn_hiring.db   full local build, gitignored
    data/snapshot.db    deploy artifact, committed

The only difference is `raw_posts.text_html`, which is nulled out here. Nothing
reads it at runtime — it exists so that improving html_to_text() can re-derive
text_plain without re-fetching from the API. That is worth 15MB on disk and not
worth 15MB in git history on every rebuild.

The schema stays byte-identical between the two (the column remains, just
empty), so the app, the queries, and the LLM's metadata-derived DDL behave the
same against either.

    python scripts/build_snapshot.py
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db import DATA_DIR, DEFAULT_DB_PATH, SNAPSHOT_DB_PATH


def main() -> None:
    if not DEFAULT_DB_PATH.exists():
        raise SystemExit(
            f"No local database at {DEFAULT_DB_PATH}. Run scripts/run_etl.py first."
        )

    before = DEFAULT_DB_PATH.stat().st_size
    print(f"Source: {DEFAULT_DB_PATH.name}  ({before / 1e6:.1f} MB)")

    DATA_DIR.mkdir(exist_ok=True)
    shutil.copyfile(DEFAULT_DB_PATH, SNAPSHOT_DB_PATH)

    conn = sqlite3.connect(SNAPSHOT_DB_PATH)
    try:
        conn.execute("UPDATE raw_posts SET text_html = NULL")
        conn.commit()
        # Reclaim the freed pages; without VACUUM the file stays the same size.
        conn.execute("VACUUM")
        counts = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("threads", "raw_posts", "jobs", "job_tech", "dim_tech")
        }
    finally:
        conn.close()

    after = SNAPSHOT_DB_PATH.stat().st_size
    print(f"Snapshot: {SNAPSHOT_DB_PATH.name}  ({after / 1e6:.1f} MB, "
          f"{100 * (1 - after / before):.0f}% smaller)")
    print()
    for table, n in counts.items():
        print(f"  {table:<12} {n:>7,} rows")


if __name__ == "__main__":
    main()
