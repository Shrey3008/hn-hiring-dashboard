"""Run the ETL: fetch raw posts, then parse them into the curated layer.

    python scripts/run_etl.py                  # fetch missing threads + parse
    python scripts/run_etl.py --parse-only     # re-parse without re-fetching
    python scripts/run_etl.py --months 12      # smaller window
    python scripts/run_etl.py --force          # re-fetch everything

--parse-only is the one to use while iterating on extraction rules.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db import create_all, get_engine
from src.fetch import fetch_threads
from src.parse import parse_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--months", type=int, default=36)
    ap.add_argument("--force", action="store_true", help="re-fetch threads already stored")
    ap.add_argument("--parse-only", action="store_true", help="skip the fetch stage")
    ap.add_argument("--fetch-only", action="store_true", help="skip the parse stage")
    args = ap.parse_args()

    engine = get_engine()
    create_all(engine)

    if not args.parse_only:
        print("=" * 60)
        print("FETCH")
        print("=" * 60)
        t0 = time.time()
        stats = fetch_threads(engine, months=args.months, force=args.force)
        print(f"\nFetched {stats['threads_fetched']} threads "
              f"({stats['threads_skipped']} already stored), "
              f"{stats['posts']:,} posts in {time.time() - t0:.1f}s")

    if not args.fetch_only:
        print()
        print("=" * 60)
        print("PARSE")
        print("=" * 60)
        t0 = time.time()
        stats = parse_all(engine)
        pct = stats["structured"] / stats["posts"] * 100 if stats["posts"] else 0
        print(f"\nParsed {stats['posts']:,} posts "
              f"({stats['structured']:,} structured, {pct:.0f}%), "
              f"{stats['tech_mentions']:,} tech mentions in {time.time() - t0:.1f}s")
        print("\nNext: python scripts/parse_report.py")


if __name__ == "__main__":
    main()
