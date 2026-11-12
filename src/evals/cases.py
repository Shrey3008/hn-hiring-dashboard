"""The eval set: questions with known-correct answers.

Expected answers are stored as REFERENCE SQL, not as literal numbers. Hard-coded
numbers rot the moment the ETL fetches a new month; reference SQL stays true.
Both the reference and the model's query run through the same read-only
connection against the same database, so the only difference being measured is
the query itself.

Tags drive per-category accuracy. The one that matters most is `fan_out`: those
are the questions where `job_tech` must be joined and a share computed, which is
exactly where this project shipped a 3x error. If overall accuracy is 90% but
fan_out accuracy is 40%, the headline number is hiding the failure that actually
costs you.

Adding a case: write the question as a user would ask it, write the reference
SQL, tag it, and run `python scripts/run_evals.py --dry-run` to confirm the
reference executes and returns rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    tags: tuple[str, ...]
    reference_sql: str | None = None
    # True when the only correct behaviour is to refuse: the question cannot be
    # answered from this schema. A model that invents a plausible query here is
    # worse than one that declines.
    expect_unanswerable: bool = False
    # "contains" tolerates extra helper columns; "strict" does not.
    mode: str = "contains"
    notes: str = ""


CASES: tuple[EvalCase, ...] = (
    # ---------------------------------------------------------------- basics
    EvalCase(
        id="total_posts",
        question="How many job posts are in the dataset?",
        tags=("basic", "aggregate"),
        reference_sql="SELECT COUNT(*) AS posts FROM jobs",
    ),
    EvalCase(
        id="posts_per_month",
        question="How many posts are there each month?",
        tags=("basic", "time_series"),
        reference_sql="SELECT month, COUNT(*) AS posts FROM jobs GROUP BY month ORDER BY month",
    ),
    EvalCase(
        id="hybrid_count",
        question="How many posts are hybrid?",
        tags=("basic", "filter"),
        reference_sql="SELECT COUNT(*) AS posts FROM jobs WHERE work_mode = 'hybrid'",
    ),
    EvalCase(
        id="month_range",
        question="What is the earliest and latest month covered by the data?",
        tags=("basic", "aggregate"),
        reference_sql="SELECT MIN(month) AS first_month, MAX(month) AS last_month FROM jobs",
    ),
    EvalCase(
        id="tech_count",
        question="How many technologies are tracked in the lookup table?",
        tags=("basic", "aggregate"),
        reference_sql="SELECT COUNT(*) AS technologies FROM dim_tech",
    ),
    EvalCase(
        id="database_category",
        question="Which technologies are in the database category?",
        tags=("basic", "filter"),
        reference_sql="SELECT tech FROM dim_tech WHERE category = 'database' ORDER BY tech",
    ),

    # ------------------------------------------------- shares WITHOUT fan-out
    # These join nothing, so COUNT(*) is already a post count. They are the
    # control group: if these fail too, the problem is not fan-out.
    EvalCase(
        id="remote_share_monthly",
        question="What percentage of posts were remote in each month?",
        tags=("share", "time_series"),
        reference_sql=(
            "SELECT month, (100.0 * SUM(CASE WHEN work_mode = 'remote' THEN 1 ELSE 0 END) "
            "/ COUNT(*)) AS pct_remote FROM jobs GROUP BY month ORDER BY month"
        ),
    ),
    EvalCase(
        id="salary_disclosure_share",
        question="What percentage of posts disclosed a salary?",
        tags=("share", "salary"),
        reference_sql=(
            "SELECT (100.0 * SUM(CASE WHEN has_salary = 1 THEN 1 ELSE 0 END) "
            "/ COUNT(*)) AS pct_disclosed FROM jobs"
        ),
    ),
    EvalCase(
        id="structured_header_share",
        question="What share of posts used the structured pipe-delimited header?",
        tags=("share",),
        reference_sql=(
            "SELECT (100.0 * SUM(CASE WHEN parsed_structured = 1 THEN 1 ELSE 0 END) "
            "/ COUNT(*)) AS pct_structured FROM jobs"
        ),
    ),

    # ------------------------------------------------------ THE FAN-OUT CASES
    # Every one of these requires joining job_tech (about 2.7 rows per post) and
    # then dividing by a POST count. Taking the denominator from the joined table
    # understates the answer roughly 3x. This is the class of bug that shipped.
    EvalCase(
        id="llm_share_monthly",
        question="How has the share of posts mentioning LLM changed each month?",
        tags=("share", "fan_out", "time_series"),
        reference_sql=(
            "WITH monthly_total AS (SELECT month, COUNT(*) AS total FROM jobs GROUP BY month) "
            "SELECT j.month, (COUNT(DISTINCT t.post_id) * 100.0 / m.total) AS pct_posts "
            "FROM job_tech t JOIN jobs j ON j.post_id = t.post_id "
            "JOIN monthly_total m ON m.month = j.month "
            "WHERE t.tech = 'LLM' GROUP BY j.month, m.total ORDER BY j.month"
        ),
        notes="The exact question that produced the shipped 3x bug: 1.5% reported vs 4.94% true.",
    ),
    EvalCase(
        id="rust_share_monthly",
        question="What share of posts mention Rust each month?",
        tags=("share", "fan_out", "time_series"),
        reference_sql=(
            "WITH monthly_total AS (SELECT month, COUNT(*) AS total FROM jobs GROUP BY month) "
            "SELECT j.month, (COUNT(DISTINCT t.post_id) * 100.0 / m.total) AS pct_posts "
            "FROM job_tech t JOIN jobs j ON j.post_id = t.post_id "
            "JOIN monthly_total m ON m.month = j.month "
            "WHERE t.tech = 'Rust' GROUP BY j.month, m.total ORDER BY j.month"
        ),
    ),
    EvalCase(
        id="python_share_2026",
        question="What percentage of posts in 2026 mentioned Python?",
        tags=("share", "fan_out"),
        reference_sql=(
            "SELECT (100.0 * (SELECT COUNT(DISTINCT t.post_id) FROM job_tech t "
            "JOIN jobs j ON j.post_id = t.post_id "
            "WHERE t.tech = 'Python' AND j.month >= '2026-01-01') "
            "/ (SELECT COUNT(*) FROM jobs WHERE month >= '2026-01-01')) AS pct_posts"
        ),
    ),
    EvalCase(
        id="docker_share_overall",
        question="What share of all posts mention Docker?",
        tags=("share", "fan_out"),
        reference_sql=(
            "SELECT (100.0 * (SELECT COUNT(DISTINCT post_id) FROM job_tech WHERE tech = 'Docker') "
            "/ (SELECT COUNT(*) FROM jobs)) AS pct_posts"
        ),
    ),

    # ------------------------------------------------------------- ranking
    EvalCase(
        id="top_technologies",
        question="What are the top 10 technologies by number of posts mentioning them?",
        tags=("ranking", "fan_out"),
        reference_sql=(
            "SELECT tech, COUNT(DISTINCT post_id) AS posts FROM job_tech "
            "GROUP BY tech ORDER BY posts DESC, tech LIMIT 10"
        ),
        notes="COUNT(DISTINCT post_id), not COUNT(*) — a post can only count once per tech.",
    ),
    EvalCase(
        id="top_databases",
        question="What are the top 5 databases mentioned?",
        tags=("ranking", "fan_out", "join"),
        reference_sql=(
            "SELECT t.tech, COUNT(DISTINCT t.post_id) AS posts FROM job_tech t "
            "JOIN dim_tech d ON d.tech = t.tech WHERE d.category = 'database' "
            "GROUP BY t.tech ORDER BY posts DESC, t.tech LIMIT 5"
        ),
    ),
    EvalCase(
        id="top_companies",
        question="Which companies posted the most jobs?",
        tags=("ranking",),
        reference_sql=(
            "SELECT company, COUNT(*) AS posts FROM jobs WHERE company IS NOT NULL "
            "GROUP BY company ORDER BY posts DESC, company LIMIT 20"
        ),
    ),

    # ---------------------------------------------------------------- joins
    EvalCase(
        id="companies_using_rust_2026",
        question="Which companies mentioned Rust in 2026?",
        tags=("join", "filter"),
        reference_sql=(
            "SELECT DISTINCT j.company FROM jobs j JOIN job_tech t ON t.post_id = j.post_id "
            "WHERE t.tech = 'Rust' AND j.month >= '2026-01-01' AND j.company IS NOT NULL "
            "ORDER BY j.company"
        ),
    ),
    EvalCase(
        id="python_and_aws",
        question="How many posts mention both Python and AWS?",
        tags=("join", "fan_out"),
        reference_sql=(
            "SELECT COUNT(*) AS posts FROM (SELECT post_id FROM job_tech "
            "WHERE tech IN ('Python', 'AWS') GROUP BY post_id HAVING COUNT(DISTINCT tech) = 2)"
        ),
        notes="Requires HAVING COUNT(DISTINCT tech)=2, not a WHERE with AND, which returns nothing.",
    ),
    EvalCase(
        id="distinct_companies_2025",
        question=("How many distinct companies posted in 2025? Treat names that differ only "
                  "by capitalisation or surrounding whitespace as the same company."),
        tags=("aggregate", "filter"),
        notes="First live run exposed this as under-specified: literal COUNT(DISTINCT company) gives 1890, normalised gives 1881. The question now says which.",
        reference_sql=(
            "SELECT COUNT(DISTINCT LOWER(TRIM(company))) AS companies FROM jobs "
            "WHERE month >= '2025-01-01' AND month < '2026-01-01' AND company IS NOT NULL"
        ),
    ),

    # --------------------------------------------------------------- salary
    EvalCase(
        id="kubernetes_salary",
        question="What is the average salary midpoint for posts mentioning Kubernetes?",
        tags=("salary", "join"),
        reference_sql=(
            "SELECT (AVG((j.salary_min + j.salary_max) / 2.0)) AS avg_midpoint "
            "FROM jobs j JOIN job_tech t ON t.post_id = j.post_id "
            "WHERE t.tech = 'Kubernetes' AND j.has_salary = 1"
        ),
    ),
    EvalCase(
        id="salary_disclosure_monthly",
        question="What percentage of posts disclosed a salary in each month?",
        tags=("share", "salary", "time_series"),
        reference_sql=(
            "SELECT month, (100.0 * SUM(CASE WHEN has_salary = 1 THEN 1 ELSE 0 END) "
            "/ COUNT(*)) AS pct_disclosed FROM jobs GROUP BY month ORDER BY month"
        ),
    ),

    # ------------------------------------------------------- exact spellings
    # The prompt injects all 124 technology names verbatim precisely because the
    # model otherwise guesses 'postgres' and silently returns zero rows — a wrong
    # answer dressed as a valid one. These cases guard that.
    EvalCase(
        id="postgresql_spelling",
        question="How many posts mention PostgreSQL?",
        tags=("spelling", "fan_out"),
        reference_sql=(
            "SELECT COUNT(DISTINCT post_id) AS posts FROM job_tech WHERE tech = 'PostgreSQL'"
        ),
    ),
    EvalCase(
        id="typescript_spelling",
        question="How many posts mention TypeScript?",
        tags=("spelling", "fan_out"),
        reference_sql=(
            "SELECT COUNT(DISTINCT post_id) AS posts FROM job_tech WHERE tech = 'TypeScript'"
        ),
    ),

    # --------------------------------------------------------- unanswerable
    # Declining is the correct answer. A confident invented query is the failure
    # mode being measured here.
    EvalCase(
        id="unanswerable_stock_price",
        question="What is the current stock price of each company that posted?",
        tags=("unanswerable",),
        expect_unanswerable=True,
    ),
    EvalCase(
        id="unanswerable_applicants",
        question="How many people applied to each job posting?",
        tags=("unanswerable",),
        expect_unanswerable=True,
    ),
)


def all_tags() -> list[str]:
    tags: set[str] = set()
    for case in CASES:
        tags.update(case.tags)
    return sorted(tags)


def select_cases(only_tags: tuple[str, ...] = (), ids: tuple[str, ...] = ()) -> list[EvalCase]:
    chosen = list(CASES)
    if ids:
        chosen = [c for c in chosen if c.id in ids]
    if only_tags:
        chosen = [c for c in chosen if set(c.tags) & set(only_tags)]
    return chosen
