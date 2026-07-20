"""HN Hiring Trends — Streamlit dashboard.

Reads the curated layer built by scripts/run_etl.py. Charts are ordered by how
much they actually say: hiring volume, work-mode mix, and tech trend first;
top-tech fourth; salary last and collapsed, because its 16.7% coverage makes it
the weakest evidence in the project.

Every chart ships a "Show data" table. That is not a nicety — three light-mode
palette slots sit below 3:1 contrast against the chart surface, and the table
view plus direct labels are what make them legal. See src/theme.py.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src import queries as q
from src.db import get_engine
from src.theme import (
    WORK_MODE_ORDER,
    altair_config,
    palette,
    resolve_tech_colors,
    work_mode_colors,
)

st.set_page_config(
    page_title="HN Hiring Trends",
    page_icon="📊",
    layout="wide",
)

CHART_HEIGHT = 300


# --------------------------------------------------------------------------
# Data access — cached
# --------------------------------------------------------------------------


@st.cache_resource
def engine():
    return get_engine()


@st.cache_data(ttl=3600)
def _month_bounds():
    return q.month_bounds(engine())


@st.cache_data(ttl=3600)
def _tech_rank():
    return q.tech_rank(engine())


@st.cache_data(ttl=3600)
def _kpis(start, end, modes):
    return q.kpis(engine(), start, end, modes)


@st.cache_data(ttl=3600)
def _volume(start, end, modes):
    return q.volume_by_month(engine(), start, end, modes)


@st.cache_data(ttl=3600)
def _work_mode(start, end):
    return q.work_mode_by_month(engine(), start, end)


@st.cache_data(ttl=3600)
def _top_tech(start, end, modes, limit, cats):
    return q.top_tech(engine(), start, end, modes, limit, cats)


@st.cache_data(ttl=3600)
def _tech_trend(start, end, modes, techs):
    return q.tech_trend(engine(), start, end, modes, techs)


@st.cache_data(ttl=3600)
def _salary(start, end, modes):
    return q.salary_by_month(engine(), start, end, modes)


@st.cache_data(ttl=3600)
def _jobs(start, end, modes, techs, search):
    return q.jobs_table(engine(), start, end, modes, techs, search)


def is_dark() -> bool:
    """Follow the viewer's Streamlit theme where the API exposes it."""
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


DARK = is_dark()
P = palette(DARK)


def show_chart(chart, key: str, data: pd.DataFrame, caption: str | None = None):
    """Render a chart with its table view.

    theme=None matters: Streamlit's default chart theme would override the
    validated palette with its own colors.
    """
    st.altair_chart(altair_config(chart, DARK), use_container_width=True, theme=None)
    if caption:
        st.caption(caption)
    with st.expander("Show data"):
        st.dataframe(data, use_container_width=True, hide_index=True)


def month_axis(title: str | None = None):
    return alt.X(
        "month:T",
        title=title,
        axis=alt.Axis(format="%b %Y", tickCount=8, labelAngle=0),
    )


# --------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------

min_month, max_month = _month_bounds()
months = pd.date_range(min_month, max_month, freq="MS")

st.sidebar.title("Filters")

start_dt, end_dt = st.sidebar.select_slider(
    "Month range",
    options=list(months),
    value=(months[0], months[-1]),
    format_func=lambda d: d.strftime("%b %Y"),
)
start, end = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

selected_modes = st.sidebar.multiselect(
    "Work mode",
    options=WORK_MODE_ORDER,
    default=[],
    help="Empty = all modes. Does not affect the work-mode composition chart.",
)
modes = tuple(selected_modes)

st.sidebar.divider()
st.sidebar.caption(
    "Source: Hacker News *Ask HN: Who is hiring?* threads via the Algolia HN API. "
    "Rebuild with `python scripts/run_etl.py`."
)

# --------------------------------------------------------------------------
# Header + KPIs
# --------------------------------------------------------------------------

st.title("Hacker News Hiring Trends")
st.caption(
    f"{len(months)} monthly threads · {min_month[:7]} to {max_month[:7]} · "
    "parsed from free-text job posts"
)

k = _kpis(start, end, modes)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Job posts", f"{k['posts']:,}")
c2.metric("Distinct companies", f"{k['companies']:,}")
c3.metric("Remote", f"{k['pct_remote']:.0f}%")
c4.metric("Disclosing salary", f"{k['pct_salary']:.0f}%")
c5.metric("Latest month", str(k["latest_month"])[:7] if k["latest_month"] else "—")

st.divider()

# --------------------------------------------------------------------------
# 1. Hiring volume
# --------------------------------------------------------------------------

st.subheader("Hiring volume")

vol = _volume(start, end, modes)
vol["month"] = pd.to_datetime(vol["month"])

# Single series — no legend box; the heading names it.
vol_base = alt.Chart(vol).encode(
    month_axis(),
    alt.Y("posts:Q", title="Job posts", scale=alt.Scale(zero=True)),
)
vol_chart = (
    vol_base.mark_line(strokeWidth=2, color=P["categorical"][0])
    + vol_base.mark_point(size=60, filled=True, color=P["categorical"][0]).encode(
        tooltip=[
            alt.Tooltip("month:T", title="Month", format="%B %Y"),
            alt.Tooltip("posts:Q", title="Posts"),
        ]
    )
).properties(height=CHART_HEIGHT)

show_chart(
    vol_chart, "volume", vol,
    "Posts per monthly thread. Thread size varies enough (258–415) that every "
    "other chart normalizes against it rather than using raw counts.",
)

# --------------------------------------------------------------------------
# 2. Work mode mix
# --------------------------------------------------------------------------

st.subheader("Remote vs hybrid vs onsite")

wm = _work_mode(start, end)
wm["month"] = pd.to_datetime(wm["month"])

# Explicit stack order. Vega-Lite only emits the `color_<field>_sort_index`
# field when the color scale declares `sort`; with a `domain` it does not, so
# an order channel referencing it is silently dropped and the stack falls back
# to alphabetical. That buries `remote` between two varying bands, which is the
# one series whose trend this chart exists to show. Anchor it to the baseline.
wm["stack_order"] = wm["work_mode"].map({m: i for i, m in enumerate(WORK_MODE_ORDER)})

wm_chart = (
    alt.Chart(wm)
    .mark_area(stroke=P["surface"], strokeWidth=1.5)
    .encode(
        month_axis(),
        alt.Y("share:Q", title="Share of posts (%)", stack="normalize",
              axis=alt.Axis(format="%")),
        alt.Color(
            "work_mode:N",
            title="Work mode",
            # Fixed domain+range: the band for "remote" is the same color no
            # matter which modes are present in the filtered data.
            scale=alt.Scale(domain=WORK_MODE_ORDER, range=work_mode_colors(DARK)),
            legend=alt.Legend(orient="top", direction="horizontal", symbolType="square"),
        ),
        order=alt.Order("stack_order:Q", sort="ascending"),
        tooltip=[
            alt.Tooltip("month:T", title="Month", format="%B %Y"),
            alt.Tooltip("work_mode:N", title="Mode"),
            alt.Tooltip("share:Q", title="Share", format=".1f"),
            alt.Tooltip("posts:Q", title="Posts"),
        ],
    )
    .properties(height=CHART_HEIGHT)
)

show_chart(
    wm_chart, "work_mode", wm,
    "Ignores the sidebar work-mode filter by design — filtering a composition "
    "chart by the dimension it decomposes would always render one solid band. "
    "*unknown* is missing data, not a category, so it wears neutral gray.",
)

# --------------------------------------------------------------------------
# 3. Tech trend
# --------------------------------------------------------------------------

st.subheader("Technology trend")

rank = _tech_rank()
ranked_techs = rank["tech"].tolist()

default_techs = [t for t in ["Python", "TypeScript", "Rust", "LLM", "Kubernetes"]
                 if t in ranked_techs]
picked = st.multiselect(
    "Compare technologies (max 8)",
    options=ranked_techs,
    default=default_techs,
    # 8 = the number of categorical slots. Past that, hues would have to be
    # cycled or generated, and two series would share a color.
    max_selections=8,
    help="Series are colored in order of overall popularity, so changing the "
         "selection can reassign colors. The legend and line-end labels always "
         "carry identity.",
)

if not picked:
    st.info("Select at least one technology.")
else:
    tt = _tech_trend(start, end, modes, tuple(picked))
    tt["month"] = pd.to_datetime(tt["month"])

    domain, colors = resolve_tech_colors(picked, ranked_techs, DARK)
    scale = alt.Scale(domain=domain, range=colors)

    tt_base = alt.Chart(tt).encode(
        month_axis(),
        alt.Y("share:Q", title="Share of posts (%)"),
        alt.Color("tech:N", title="Technology", scale=scale,
                  legend=alt.Legend(orient="top", direction="horizontal")),
    )
    tt_line = tt_base.mark_line(strokeWidth=2)
    tt_pts = tt_base.mark_point(size=55, filled=True).encode(
        tooltip=[
            alt.Tooltip("month:T", title="Month", format="%B %Y"),
            alt.Tooltip("tech:N", title="Tech"),
            alt.Tooltip("share:Q", title="Share of posts", format=".1f"),
            alt.Tooltip("posts:Q", title="Posts"),
        ]
    )

    # Direct labels at the line ends. Text wears secondary ink, never the series
    # color — the adjacent colored mark carries identity. This is also the
    # "relief" the sub-3:1 light slots require.
    last = tt.loc[tt.groupby("tech")["month"].idxmax()]
    tt_labels = (
        alt.Chart(last)
        .mark_text(align="left", dx=8, dy=0, fontSize=11, color=P["text_secondary"])
        .encode(month_axis(), alt.Y("share:Q"), text="tech:N")
    )

    tech_chart = (tt_line + tt_pts + tt_labels).properties(height=CHART_HEIGHT + 40)
    show_chart(
        tech_chart, "tech_trend", tt,
        "Share of that month's posts mentioning each technology — not raw "
        "counts, which would show thread-size noise as a hiring trend.",
    )

# --------------------------------------------------------------------------
# 4. Top technologies
# --------------------------------------------------------------------------

st.subheader("Most-mentioned technologies")

cats = st.multiselect(
    "Category",
    options=sorted(rank["category"].unique()),
    default=[],
    help="Empty = all categories.",
)
top = _top_tech(start, end, modes, 20, tuple(cats))

# Magnitude, one measure -> a single hue. Ranking is carried by bar length and
# sort order; coloring bars by rank would imply a second variable that is not there.
top_chart = (
    alt.Chart(top)
    .mark_bar(color=P["categorical"][0], cornerRadiusEnd=4, height=alt.RelativeBandSize(0.72))
    .encode(
        alt.X("posts:Q", title="Job posts mentioning"),
        alt.Y("tech:N", title=None, sort="-x"),
        tooltip=[
            alt.Tooltip("tech:N", title="Tech"),
            alt.Tooltip("category:N", title="Category"),
            alt.Tooltip("posts:Q", title="Posts"),
        ],
    )
    .properties(height=460)
)
show_chart(top_chart, "top_tech", top)

# --------------------------------------------------------------------------
# 5. Salary — demoted, collapsed
# --------------------------------------------------------------------------

st.subheader("Salary transparency")
st.caption(
    "⚠️ **Weakest data in the project — read the caveat before citing it.** Only "
    "explicit USD *ranges* are parsed. Single figures (\"$150k\") and non-USD "
    "postings count as undisclosed, so this is a floor on real disclosure, not a "
    "measurement of it."
)

with st.expander("Show salary charts"):
    sal = _salary(start, end, modes)
    sal["month"] = pd.to_datetime(sal["month"])

    sal_base = alt.Chart(sal).encode(
        month_axis(),
        alt.Y("pct_disclosed:Q", title="Posts disclosing a USD range (%)"),
    )
    sal_chart = (
        sal_base.mark_line(strokeWidth=2, color=P["categorical"][0])
        + sal_base.mark_point(size=60, filled=True, color=P["categorical"][0]).encode(
            tooltip=[
                alt.Tooltip("month:T", title="Month", format="%B %Y"),
                alt.Tooltip("pct_disclosed:Q", title="Disclosed", format=".1f"),
                alt.Tooltip("disclosed:Q", title="Posts with range"),
            ]
        )
    ).properties(height=240)
    show_chart(sal_chart, "salary_pct", sal)

    # Separate chart, not a second axis on the one above. Two measures of
    # different scale never share a y-axis.
    mid_base = alt.Chart(sal.dropna(subset=["avg_midpoint"])).encode(
        month_axis(),
        alt.Y("avg_midpoint:Q", title="Average midpoint (USD)",
              axis=alt.Axis(format="$,.0f"), scale=alt.Scale(zero=False)),
    )
    mid_chart = (
        mid_base.mark_line(strokeWidth=2, color=P["categorical"][1])
        + mid_base.mark_point(size=60, filled=True, color=P["categorical"][1]).encode(
            tooltip=[
                alt.Tooltip("month:T", title="Month", format="%B %Y"),
                alt.Tooltip("avg_midpoint:Q", title="Avg midpoint", format="$,.0f"),
            ]
        )
    ).properties(height=240)
    show_chart(mid_chart, "salary_mid", sal[["month", "avg_midpoint"]])

# --------------------------------------------------------------------------
# Jobs table
# --------------------------------------------------------------------------

st.divider()
st.subheader("Browse job posts")

tcol, scol = st.columns([2, 3])
with tcol:
    table_techs = st.multiselect("Must mention all of", options=ranked_techs, default=[])
with scol:
    search = st.text_input("Search post text", placeholder="e.g. visa sponsorship")

jobs_df = _jobs(start, end, modes, tuple(table_techs), search)
st.caption(f"{len(jobs_df):,} posts shown (capped at 500).")
st.dataframe(
    jobs_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "month": st.column_config.TextColumn("Month"),
        "company": st.column_config.TextColumn("Company"),
        "role_title": st.column_config.TextColumn("Role"),
        "location_raw": st.column_config.TextColumn("Location"),
        "work_mode": st.column_config.TextColumn("Mode"),
        "salary_min": st.column_config.NumberColumn("Salary min", format="$%d"),
        "salary_max": st.column_config.NumberColumn("Salary max", format="$%d"),
        "hn_link": st.column_config.LinkColumn("HN", display_text="open"),
    },
)

# --------------------------------------------------------------------------
# Caveats
# --------------------------------------------------------------------------

with st.expander("Data quality & known limitations"):
    st.markdown(
        """
Parsed from unstructured free text, so extraction is best-effort and measured
rather than assumed (`python scripts/parse_report.py`):

| Field | Coverage |
|---|---|
| structured header | 90.1% |
| company | 90.6% |
| work mode | 90.7% |
| location | 74.3% |
| ≥1 technology | 70.5% |
| role title | 62.3% |
| salary range (USD) | 16.7% |

**Known limitations**

- **Go is undercounted.** It matches only `\\bgolang\\b`; matching the English
  word "go" would swamp the chart with false positives. Real Go demand is higher
  than shown.
- **Salary is a floor, not a rate** — USD ranges only, as described above.
- **Role title is 62%**, but 87% of the misses are posts with no role in the
  header at all, not extraction failures.
- **Technology mentions are not weighted.** A tech named once in a long list
  counts the same as the core stack.
- **Companies are not deduplicated** beyond case/whitespace normalization, so
  "Acme" and "Acme Inc" are two companies.
        """
    )
