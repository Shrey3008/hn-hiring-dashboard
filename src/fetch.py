"""Fetch HN "Who is hiring?" threads into the raw layer.

Source: Algolia's HN API (free, no key, no auth).
    search_by_date?tags=story,author_whoishiring   -> the monthly threads
    items/{story_id}                               -> full comment tree

Two things this has to get right:

1. The whoishiring bot posts TWO threads every month — "Who is hiring?"
   (employers) and "Who wants to be hired?" (job seekers). They interleave in
   the API response and look nearly identical. Mixing them would invert half
   the dataset, so titles are filtered strictly.

2. Ingest must be idempotent. Re-running replaces a thread's posts rather than
   duplicating them, so a partial run is always safe to repeat.
"""

from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterator

import requests
from sqlalchemy import delete, insert, select
from sqlalchemy.engine import Engine

from src.db import raw_posts, threads

ALGOLIA_SEARCH = "https://hn.algolia.com/api/v1/search_by_date"
ALGOLIA_ITEM = "https://hn.algolia.com/api/v1/items/{}"

USER_AGENT = "hn-hiring-dashboard/0.1 (portfolio project)"

# "Ask HN: Who is hiring? (July 2026)" — the month lives in the title, which is
# more reliable than created_at (threads post at midnight UTC on the 1st and can
# land in the previous day depending on timezone).
TITLE_MONTH_RE = re.compile(r"\(([A-Za-z]+)\s+(\d{4})\)")


@dataclass(frozen=True)
class ThreadRef:
    thread_id: int
    month: date
    title: str


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _get(session: requests.Session, url: str, params: dict | None = None,
         timeout: int = 120, retries: int = 4) -> dict:
    """GET with exponential backoff. Algolia is generous but occasionally 429s."""
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                raise requests.HTTPError("429 rate limited", response=resp)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"    request failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def is_hiring_thread(title: str) -> bool:
    """True only for employer threads.

    Excludes "Who wants to be hired?" and "Freelancer? Seeking freelancer?",
    which the same account posts monthly.
    """
    t = title.lower()
    if "who is hiring" not in t:
        return False
    if "wants to be hired" in t or "freelancer" in t:
        return False
    return True


def parse_month(title: str) -> date | None:
    m = TITLE_MONTH_RE.search(title)
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%B %Y").date()
    except ValueError:
        return None


def list_hiring_threads(months: int, session: requests.Session | None = None) -> list[ThreadRef]:
    """Return the most recent `months` hiring threads, newest first."""
    session = session or _session()
    payload = _get(
        session,
        ALGOLIA_SEARCH,
        params={"tags": "story,author_whoishiring", "hitsPerPage": 1000},
    )

    found: list[ThreadRef] = []
    for hit in payload.get("hits", []):
        title = hit.get("title") or ""
        if not is_hiring_thread(title):
            continue
        month = parse_month(title)
        if month is None:
            continue
        found.append(ThreadRef(int(hit["objectID"]), month, title))

    # Dedupe by month, keeping the first (newest) occurrence.
    by_month: dict[date, ThreadRef] = {}
    for ref in found:
        by_month.setdefault(ref.month, ref)

    ordered = sorted(by_month.values(), key=lambda r: r.month, reverse=True)
    return ordered[:months]


def html_to_text(raw: str | None) -> str:
    """Convert HN's comment HTML to plain text, preserving line structure.

    HN separates paragraphs with bare <p> tags and wraps links in <a href>.
    Newlines matter here: the parser relies on the first line being the
    structured "Company | Role | Location" header.
    """
    if not raw:
        return ""
    text = re.sub(r"<p>", "\n", raw, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    # Keep the href, drop the tag — URLs often carry the company domain.
    text = re.sub(r'<a[^>]*href="([^"]*)"[^>]*>.*?</a>', r"\1", text,
                  flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def iter_thread_posts(story_id: int, session: requests.Session) -> Iterator[dict]:
    """Yield top-level comments (job posts) for a thread.

    Only direct children count — replies are questions and discussion, not jobs.
    """
    item = _get(session, ALGOLIA_ITEM.format(story_id))
    for child in item.get("children") or []:
        if not child.get("text"):
            continue  # deleted or dead
        yield child


def fetch_threads(engine: Engine, months: int = 36, force: bool = False) -> dict:
    """Fetch threads into `threads` and `raw_posts`.

    Skips threads already present unless force=True, so an interrupted run
    resumes cheaply instead of re-downloading everything.
    """
    session = _session()
    refs = list_hiring_threads(months, session)
    print(f"Found {len(refs)} hiring threads "
          f"({refs[-1].month:%b %Y} .. {refs[0].month:%b %Y})")

    with engine.connect() as conn:
        existing = {row[0] for row in conn.execute(select(threads.c.thread_id))}

    stats = {"threads_fetched": 0, "threads_skipped": 0, "posts": 0}

    for ref in refs:
        if ref.thread_id in existing and not force:
            stats["threads_skipped"] += 1
            continue

        print(f"  {ref.month:%Y-%m}  fetching {ref.thread_id} ...", end="", flush=True)
        posts = list(iter_thread_posts(ref.thread_id, session))
        now = datetime.utcnow()

        rows = [
            {
                "post_id": int(p["id"]),
                "thread_id": ref.thread_id,
                "author": p.get("author"),
                "created_at": _parse_ts(p.get("created_at")),
                "text_html": p.get("text"),
                "text_plain": html_to_text(p.get("text")),
                "ingested_at": now,
            }
            for p in posts
        ]
        # Guard against the same comment id appearing twice in a tree.
        rows = list({r["post_id"]: r for r in rows}.values())

        with engine.begin() as conn:
            conn.execute(delete(raw_posts).where(raw_posts.c.thread_id == ref.thread_id))
            conn.execute(delete(threads).where(threads.c.thread_id == ref.thread_id))
            conn.execute(
                insert(threads),
                {
                    "thread_id": ref.thread_id,
                    "month": ref.month,
                    "title": ref.title,
                    "post_count": len(rows),
                    "fetched_at": now,
                },
            )
            if rows:
                conn.execute(insert(raw_posts), rows)

        stats["threads_fetched"] += 1
        stats["posts"] += len(rows)
        print(f" {len(rows)} posts")
        time.sleep(0.3)  # be polite; the API is free

    return stats


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None
