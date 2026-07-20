"""Chart-backing queries.

Every function takes plain hashable args and returns a DataFrame, so Streamlit
can cache them directly. SQL is written as literal text rather than built by the
ORM — it doubles as the schema documentation the LLM query feature reads in
Phase 4, and keeping it readable matters more here than composability.

Filter convention: month range + work mode apply globally (sidebar). Tech is NOT
a global filter — the trend chart and the jobs table each own their tech
selector. A global tech filter would corrupt the trend denominator, since
"% of posts mentioning Rust" is meaningless once the population is already
filtered to Rust posts.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

# Applied to every query. Kept in one place so the filter semantics can't drift
# between the KPI row and the charts.
_MONTH_WHERE = "j.month BETWEEN :start AND :end"


def _mode_clause(work_modes: tuple[str, ...]) -> str:
    if not work_modes:
        return ""
    slots = ", ".join(f":wm{i}" for i in range(len(work_modes)))
    return f" AND j.work_mode IN ({slots})"


def _params(start: str, end: str, work_modes: tuple[str, ...]) -> dict:
    p = {"start": start, "end": end}
    p.update({f"wm{i}": m for i, m in enumerate(work_modes)})
    return p


def month_bounds(engine: Engine) -> tuple[str, str]:
    with engine.connect() as conn:
        row = conn.execute(text("SELECT MIN(month), MAX(month) FROM jobs")).fetchone()
    return str(row[0]), str(row[1])


def kpis(engine: Engine, start: str, end: str, work_modes: tuple[str, ...]) -> dict:
    sql = text(f"""
        SELECT
            COUNT(*)                                        AS posts,
            COUNT(DISTINCT LOWER(TRIM(j.company)))          AS companies,
            SUM(CASE WHEN j.work_mode = 'remote' THEN 1 ELSE 0 END) AS remote_posts,
            SUM(CASE WHEN j.has_salary = 1 THEN 1 ELSE 0 END)       AS salary_posts,
            MAX(j.month)                                    AS latest_month
        FROM jobs j
        WHERE {_MONTH_WHERE}{_mode_clause(work_modes)}
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, _params(start, end, work_modes)).mappings().fetchone()

    posts = row["posts"] or 0
    return {
        "posts": posts,
        "companies": row["companies"] or 0,
        "pct_remote": (row["remote_posts"] or 0) / posts * 100 if posts else 0.0,
        "pct_salary": (row["salary_posts"] or 0) / posts * 100 if posts else 0.0,
        "latest_month": row["latest_month"],
    }


def volume_by_month(engine: Engine, start: str, end: str,
                    work_modes: tuple[str, ...]) -> pd.DataFrame:
    sql = text(f"""
        SELECT j.month, COUNT(*) AS posts
        FROM jobs j
        WHERE {_MONTH_WHERE}{_mode_clause(work_modes)}
        GROUP BY j.month
        ORDER BY j.month
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn, params=_params(start, end, work_modes))


def work_mode_by_month(engine: Engine, start: str, end: str) -> pd.DataFrame:
    """Share of posts by work mode per month.

    Deliberately ignores the work-mode filter — filtering a composition chart by
    the very dimension it decomposes would always render 100% of one band.
    """
    sql = text(f"""
        SELECT j.month, j.work_mode, COUNT(*) AS posts
        FROM jobs j
        WHERE {_MONTH_WHERE}
        GROUP BY j.month, j.work_mode
        ORDER BY j.month
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"start": start, "end": end})
    totals = df.groupby("month")["posts"].transform("sum")
    df["share"] = df["posts"] / totals * 100
    return df


def tech_rank(engine: Engine) -> pd.DataFrame:
    """Every tech by total mentions across the WHOLE dataset.

    Unfiltered on purpose: this drives the stable tech->color assignment, which
    must not shift when the user changes filters.
    """
    sql = text("""
        SELECT t.tech, d.category, COUNT(*) AS mentions
        FROM job_tech t
        JOIN dim_tech d ON d.tech = t.tech
        GROUP BY t.tech, d.category
        ORDER BY mentions DESC, t.tech
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)


def top_tech(engine: Engine, start: str, end: str, work_modes: tuple[str, ...],
             limit: int = 20, categories: tuple[str, ...] = ()) -> pd.DataFrame:
    cat_clause = ""
    params = _params(start, end, work_modes)
    if categories:
        slots = ", ".join(f":cat{i}" for i in range(len(categories)))
        cat_clause = f" AND d.category IN ({slots})"
        params.update({f"cat{i}": c for i, c in enumerate(categories)})
    params["limit"] = limit

    sql = text(f"""
        SELECT t.tech, d.category, COUNT(DISTINCT t.post_id) AS posts
        FROM job_tech t
        JOIN jobs j     ON j.post_id = t.post_id
        JOIN dim_tech d ON d.tech = t.tech
        WHERE {_MONTH_WHERE}{_mode_clause(work_modes)}{cat_clause}
        GROUP BY t.tech, d.category
        ORDER BY posts DESC
        LIMIT :limit
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn, params=params)


def tech_trend(engine: Engine, start: str, end: str, work_modes: tuple[str, ...],
               techs: tuple[str, ...]) -> pd.DataFrame:
    """Monthly share of posts mentioning each selected tech.

    Share, not raw count — thread sizes vary month to month (258 to 415 in this
    dataset), so raw counts would show thread-volume noise as a technology
    trend. This is the single most important normalization in the project.
    """
    if not techs:
        return pd.DataFrame(columns=["month", "tech", "posts", "share"])

    params = _params(start, end, work_modes)
    slots = ", ".join(f":t{i}" for i in range(len(techs)))
    params.update({f"t{i}": t for i, t in enumerate(techs)})

    sql = text(f"""
        WITH monthly_total AS (
            SELECT j.month, COUNT(*) AS total
            FROM jobs j
            WHERE {_MONTH_WHERE}{_mode_clause(work_modes)}
            GROUP BY j.month
        )
        SELECT
            j.month,
            t.tech,
            COUNT(DISTINCT t.post_id)                  AS posts,
            COUNT(DISTINCT t.post_id) * 100.0 / m.total AS share
        FROM job_tech t
        JOIN jobs j          ON j.post_id = t.post_id
        JOIN monthly_total m ON m.month = j.month
        WHERE {_MONTH_WHERE}{_mode_clause(work_modes)} AND t.tech IN ({slots})
        GROUP BY j.month, t.tech, m.total
        ORDER BY j.month, t.tech
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn, params=params)


def salary_by_month(engine: Engine, start: str, end: str,
                    work_modes: tuple[str, ...]) -> pd.DataFrame:
    sql = text(f"""
        SELECT
            j.month,
            COUNT(*)                                          AS posts,
            SUM(CASE WHEN j.has_salary = 1 THEN 1 ELSE 0 END) AS disclosed,
            AVG(CASE WHEN j.has_salary = 1
                     THEN (j.salary_min + j.salary_max) / 2.0 END) AS avg_midpoint
        FROM jobs j
        WHERE {_MONTH_WHERE}{_mode_clause(work_modes)}
        GROUP BY j.month
        ORDER BY j.month
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params=_params(start, end, work_modes))
    df["pct_disclosed"] = df["disclosed"] / df["posts"] * 100
    return df


def jobs_table(engine: Engine, start: str, end: str, work_modes: tuple[str, ...],
               techs: tuple[str, ...] = (), search: str = "",
               limit: int = 500) -> pd.DataFrame:
    params = _params(start, end, work_modes)
    params["limit"] = limit

    tech_clause = ""
    if techs:
        slots = ", ".join(f":t{i}" for i in range(len(techs)))
        params.update({f"t{i}": t for i, t in enumerate(techs)})
        # ALL selected techs must be present, not any — "Rust AND Kubernetes"
        # is the useful question when you are filtering a job list.
        tech_clause = f"""
            AND (SELECT COUNT(DISTINCT jt.tech) FROM job_tech jt
                 WHERE jt.post_id = j.post_id AND jt.tech IN ({slots})) = {len(techs)}
        """

    search_clause = ""
    if search:
        params["q"] = f"%{search.lower()}%"
        search_clause = " AND LOWER(r.text_plain) LIKE :q"

    sql = text(f"""
        SELECT
            j.month, j.company, j.role_title, j.location_raw, j.work_mode,
            j.salary_min, j.salary_max,
            'https://news.ycombinator.com/item?id=' || j.post_id AS hn_link
        FROM jobs j
        JOIN raw_posts r ON r.post_id = j.post_id
        WHERE {_MONTH_WHERE}{_mode_clause(work_modes)}{tech_clause}{search_clause}
        -- NULLS LAST matters: both SQLite and Postgres sort NULLs first by
        -- default, which would fill the entire first page with the ~9% of posts
        -- where company extraction failed and make the table look empty.
        ORDER BY j.month DESC, j.company NULLS LAST
        LIMIT :limit
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn, params=params)
