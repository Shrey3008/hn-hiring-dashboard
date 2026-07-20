"""Parse raw job posts into the curated layer.

Reads raw_posts, writes jobs + job_tech. Never touches the network, so
iterating on the rules is a seconds-long loop.

Extraction strategy
-------------------
~89% of posts open with a pipe-delimited header:

    Adacore | Software Engineers | Full-time | Remote or On-Site
    SmarterDx | 150-250k+ + equity | Remote (US only) | Multiple roles
    Safi | Product Engineer | London HQ or Remote | Full-time | £110,000–

Field ORDER is not fixed — only position 1 (company) is reliable. So the
remaining fields are classified by CONTENT (does it look like money? a work
mode? a job title?) rather than by position. Position-based parsing would
misread the majority of posts.

Everything here is best-effort by design. Recall is measured by
scripts/parse_report.py rather than assumed, and known gaps are documented
rather than papered over.
"""

from __future__ import annotations

import re
from datetime import date

import pandas as pd
from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from src.db import dim_tech, drop_curated, job_tech, jobs, raw_posts, threads

# --------------------------------------------------------------------------
# Work mode
# --------------------------------------------------------------------------

# Checked FIRST — "no remote" must not be read as a remote job. This negation
# pass is the difference between a credible remote-trend chart and a wrong one.
NOT_REMOTE_RE = re.compile(
    r"\bno\s+remote\b|\bnot\s+remote\b|\bnon[- ]remote\b|\bno\s+fully\s+remote\b"
    r"|\bonsite\s+only\b|\bon[- ]site\s+only\b|\bin[- ]office\s+only\b"
    r"|\bremote\s*[:=-]?\s*no\b",
    re.IGNORECASE,
)
HYBRID_RE = re.compile(r"\bhybrid\b", re.IGNORECASE)
REMOTE_RE = re.compile(r"\bremote\b|\bwork from home\b|\bwfh\b|\bdistributed team\b",
                       re.IGNORECASE)
ONSITE_RE = re.compile(r"\bon[- ]?site\b|\bin[- ]office\b|\bin person\b", re.IGNORECASE)


def detect_work_mode(header: str, full_text: str) -> str:
    """Return remote | hybrid | onsite | unknown.

    Prefers the structured header, which states the offer, over body text,
    where "we've been remote-first since 2015" is background noise rather
    than a statement about this role.
    """
    for scope in (header, full_text):
        if not scope:
            continue
        if NOT_REMOTE_RE.search(scope):
            return "onsite"
        if HYBRID_RE.search(scope):
            return "hybrid"
        if REMOTE_RE.search(scope):
            return "remote"
        if ONSITE_RE.search(scope):
            return "onsite"
    return "unknown"


# --------------------------------------------------------------------------
# Salary
# --------------------------------------------------------------------------

# Only ranges, and only USD. Mixed currencies in one distribution would be
# misleading, and there is no FX rate that makes a 2023 GBP figure comparable
# to a 2026 USD one. Non-USD posts are counted as "not disclosed" and that
# limitation is reported explicitly.
SALARY_RANGE_RE = re.compile(
    r"(?P<cur1>[$£€]|\bUSD\b|\bGBP\b|\bEUR\b)?\s*"
    r"(?P<lo>\d{2,3}(?:,\d{3})?)\s*(?P<lok>[kK])?\s*"
    r"(?:-|–|—|\bto\b)\s*"
    r"(?P<cur2>[$£€]|\bUSD\b)?\s*"
    r"(?P<hi>\d{2,3}(?:,\d{3})?)\s*(?P<hik>[kK])?",
)
NON_USD = {"£", "€", "GBP", "EUR"}


def _to_dollars(num: str, k_flag: str | None) -> float:
    n = float(num.replace(",", ""))
    if k_flag or n < 1000:
        n *= 1000
    return n


def extract_salary(text: str) -> tuple[float | None, float | None]:
    """Best-effort USD salary range. Returns (min, max) or (None, None)."""
    if not text:
        return None, None

    for m in SALARY_RANGE_RE.finditer(text):
        cur1, cur2 = m.group("cur1"), m.group("cur2")
        if (cur1 in NON_USD) or (cur2 in NON_USD):
            continue

        # Require an explicit money signal, else "10-20 people" parses as pay.
        has_signal = bool(cur1 or cur2 or m.group("lok") or m.group("hik"))
        if not has_signal:
            continue

        lo = _to_dollars(m.group("lo"), m.group("lok") or m.group("hik"))
        hi = _to_dollars(m.group("hi"), m.group("hik"))

        # Plausibility gate: filters out years, headcounts, and equity percents.
        if not (20_000 <= lo < hi <= 2_000_000):
            continue
        return lo, hi

    return None, None


# --------------------------------------------------------------------------
# Field classification
# --------------------------------------------------------------------------

ROLE_RE = re.compile(
    r"\bengineer|\bdeveloper\b|\bdev\b|\bprogrammer\b|\bscientist\b|\banalyst\b"
    r"|\bdesigner\b|\bmanager\b|\barchitect\b|\blead\b|\bdirector\b|\bcto\b"
    r"|\bdevops\b|\bsre\b|\bsysadmin\b|\bresearcher\b|\bintern\b|\bfounder\b"
    r"|\bfront[- ]?end\b|\bback[- ]?end\b|\bfull[- ]?stack\b|\bmobile\b"
    r"|\bqa\b|\bsecurity\b|\bproduct\b|\bdata\b|\bml\b|\bai\b|\bstaff\b"
    r"|\bsenior\b|\bjunior\b|\bprincipal\b|\bhead of\b",
    re.IGNORECASE,
)

EMPLOYMENT_RE = re.compile(
    r"\bfull[- ]?time\b|\bpart[- ]?time\b|\bcontract\b|\bfreelance\b|\bfte\b"
    r"|\bpermanent\b|\bcontractor\b|\bc2c\b|\binternship\b",
    re.IGNORECASE,
)

URL_RE = re.compile(r"https?://|\bwww\.|\.(com|io|ai|dev|org|net|co)\b", re.IGNORECASE)

MONEY_HINT_RE = re.compile(r"[$£€]|\bequity\b|\bsalary\b|\bcomp\b|\d{2,3}\s?[kK]\b",
                           re.IGNORECASE)

LOCATION_WORDS = {
    # A pragmatic list: the cities and regions that actually recur in these
    # threads. Not exhaustive geography — just enough for a location signal.
    "usa", "us", "u.s.", "uk", "u.k.", "canada", "germany", "france", "spain",
    "italy", "netherlands", "poland", "portugal", "sweden", "norway", "denmark",
    "finland", "switzerland", "austria", "belgium", "ireland", "israel", "india",
    "china", "japan", "korea", "australia", "brazil", "mexico", "argentina",
    "europe", "emea", "apac", "latam", "worldwide", "anywhere", "global",
    "san francisco", "sf", "bay area", "nyc", "new york", "brooklyn", "london",
    "berlin", "seattle", "austin", "boston", "chicago", "los angeles", "toronto",
    "vancouver", "montreal", "amsterdam", "paris", "dublin", "tokyo", "singapore",
    "sydney", "melbourne", "bangalore", "bengaluru", "mumbai", "delhi", "denver",
    "atlanta", "portland", "munich", "zurich", "stockholm", "tel aviv", "warsaw",
    "lisbon", "madrid", "barcelona", "prague", "copenhagen", "oslo", "helsinki",
    "miami", "dallas", "houston", "phoenix", "philadelphia", "washington",
    "san diego", "pittsburgh", "detroit", "minneapolis", "nashville", "raleigh",
    "durham", "boulder", "cambridge", "oxford", "manchester", "edinburgh",
    "bristol", "hamburg", "frankfurt", "cologne", "vienna", "brussels", "milan",
    "rome", "athens", "istanbul", "dubai", "hong kong", "shanghai", "beijing",
    "seoul", "taipei", "hyderabad", "pune", "chennai", "jakarta", "bangkok",
    "auckland", "cape town", "nairobi", "lagos", "sao paulo", "buenos aires",
    "mexico city", "remote-first", "salt lake city", "st. louis", "kansas city",
}
# Two-letter US state codes after a comma: "Austin, TX"
US_STATE_RE = re.compile(r",\s*[A-Z]{2}\b")


def looks_like_location(field: str) -> bool:
    low = field.lower()
    if US_STATE_RE.search(field):
        return True
    return any(w in low for w in LOCATION_WORDS)


def classify_field(field: str) -> str:
    """Label a header field by content.

    Order matters: money and work-mode signals are the most distinctive, so
    they are tested before the broader role/location heuristics.
    """
    if not field:
        return "empty"
    if MONEY_HINT_RE.search(field) and SALARY_RANGE_RE.search(field):
        return "salary"
    if NOT_REMOTE_RE.search(field) or HYBRID_RE.search(field) or REMOTE_RE.search(field):
        return "work_mode"
    if URL_RE.search(field):
        return "url"
    if ROLE_RE.search(field):
        return "role"
    if looks_like_location(field):
        return "location"
    if EMPLOYMENT_RE.search(field):
        return "employment"
    return "unknown"


# --------------------------------------------------------------------------
# Company
# --------------------------------------------------------------------------

# Fallback for the ~11% of posts with no pipe header:
# "Air Space Intelligence (https://...) is hiring in Boston"
PROSE_COMPANY_RE = re.compile(
    r"^(?:hey hn[,!]?\s*)?(?P<name>[A-Z][\w&.\-' ]{1,50}?)\s+"
    r"(?:\(|is|are)\s*(?:hiring|looking|seeking|growing)",
    re.IGNORECASE,
)


def clean_company(raw: str | None) -> str | None:
    if not raw:
        return None
    name = re.sub(r"\s+", " ", raw).strip(" *_-—–|:,.")
    # Drop a trailing URL that some posts append to the company field.
    name = re.sub(r"\s*https?://\S+$", "", name).strip()
    if not name or len(name) > 200:
        return name[:200] if name else None
    # A "company" that is really a sentence is not a company.
    if len(name.split()) > 8:
        return None
    return name


def extract_company_prose(text: str) -> str | None:
    m = PROSE_COMPANY_RE.match(text.strip())
    if m:
        return clean_company(m.group("name"))
    return None


# --------------------------------------------------------------------------
# Tech
# --------------------------------------------------------------------------


def load_tech_patterns(engine: Engine) -> list[tuple[str, re.Pattern]]:
    with engine.connect() as conn:
        rows = conn.execute(select(dim_tech.c.tech, dim_tech.c.pattern)).fetchall()
    return [(tech, re.compile(pattern, re.IGNORECASE)) for tech, pattern in rows]


def extract_tech(text: str, patterns: list[tuple[str, re.Pattern]]) -> list[str]:
    if not text:
        return []
    return [tech for tech, rx in patterns if rx.search(text)]


# --------------------------------------------------------------------------
# Main parse
# --------------------------------------------------------------------------


def parse_post(text_plain: str, patterns: list[tuple[str, re.Pattern]]) -> dict:
    """Parse one post into a jobs row (minus post_id/month) plus tech list."""
    text = text_plain or ""
    first_line = text.split("\n")[0].strip() if text else ""
    fields = [f.strip() for f in first_line.split("|")]
    structured = len(fields) >= 3

    company = role = location = None
    salary_field = None

    if structured:
        company = clean_company(fields[0])
        for field in fields[1:]:
            kind = classify_field(field)
            if kind == "role" and role is None:
                role = field[:300]
            elif kind == "location" and location is None:
                location = field[:300]
            elif kind == "salary" and salary_field is None:
                salary_field = field
            elif kind == "work_mode" and location is None and looks_like_location(field):
                # "Remote (US only)" carries both signals.
                location = field[:300]
    else:
        company = extract_company_prose(text)

    # Salary: trust the dedicated header field first, then the whole post.
    lo, hi = (None, None)
    if salary_field:
        lo, hi = extract_salary(salary_field)
    if lo is None:
        lo, hi = extract_salary(text)

    return {
        "company": company,
        "role_title": role,
        "location_raw": location,
        "work_mode": detect_work_mode(first_line, text),
        "salary_min": lo,
        "salary_max": hi,
        "has_salary": lo is not None,
        "char_count": len(text),
        "parsed_structured": structured,
        "_tech": extract_tech(text, patterns),
    }


def parse_all(engine: Engine) -> dict:
    """Rebuild jobs + job_tech from raw_posts. Destroys and recreates both."""
    patterns = load_tech_patterns(engine)

    sql = (
        select(
            raw_posts.c.post_id,
            raw_posts.c.text_plain,
            threads.c.month,
        )
        .select_from(raw_posts.join(threads, raw_posts.c.thread_id == threads.c.thread_id))
    )
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)

    if df.empty:
        raise RuntimeError("No raw posts found — run the fetcher first.")

    print(f"Parsing {len(df):,} posts against {len(patterns)} tech patterns ...")

    job_rows: list[dict] = []
    tech_rows: list[dict] = []

    for row in df.itertuples(index=False):
        parsed = parse_post(row.text_plain, patterns)
        techs = parsed.pop("_tech")
        month = row.month
        if isinstance(month, str):
            month = date.fromisoformat(month)
        parsed["post_id"] = int(row.post_id)
        parsed["month"] = month
        parsed["is_remote"] = parsed["work_mode"] == "remote"
        job_rows.append(parsed)
        tech_rows.extend({"post_id": int(row.post_id), "tech": t} for t in techs)

    drop_curated(engine)
    with engine.begin() as conn:
        for i in range(0, len(job_rows), 1000):
            conn.execute(insert(jobs), job_rows[i:i + 1000])
        for i in range(0, len(tech_rows), 5000):
            conn.execute(insert(job_tech), tech_rows[i:i + 5000])

    return {
        "posts": len(job_rows),
        "tech_mentions": len(tech_rows),
        "structured": sum(1 for r in job_rows if r["parsed_structured"]),
    }
