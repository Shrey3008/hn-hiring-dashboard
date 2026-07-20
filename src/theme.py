"""Chart palette and Altair styling.

Colors are not chosen by taste — they come from a validated categorical palette
and were re-checked for the exact series counts this dashboard renders:

    work mode (3 slots)   light: PASS   dark: PASS
    tech trend (6 slots)  light: PASS   dark: PASS

Light mode reports a sub-3:1 contrast WARN on magenta/yellow/aqua against the
chart surface. That is not dismissable — it obligates "relief": visible direct
labels or a table view. Both ship here (line-end labels, plus a "Show data"
table under every chart), which is what makes those slots legal.
"""

from __future__ import annotations

# Categorical slots in FIXED order. The ordering is the CVD-safety mechanism,
# not cosmetic — do not reshuffle to taste.
CATEGORICAL_LIGHT = [
    "#2a78d6",  # 1 blue
    "#008300",  # 2 green
    "#e87ba4",  # 3 magenta
    "#eda100",  # 4 yellow
    "#1baf7a",  # 5 aqua
    "#eb6834",  # 6 orange
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
CATEGORICAL_DARK = [
    "#3987e5", "#008300", "#d55181", "#c98500",
    "#199e70", "#d95926", "#9085e9", "#e66767",
]

LIGHT = {
    "surface": "#fcfcfb",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "categorical": CATEGORICAL_LIGHT,
}
DARK = {
    "surface": "#1a1a19",
    "text_primary": "#ffffff",
    "text_secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "categorical": CATEGORICAL_DARK,
}

# Work mode is a fixed 3-way split plus an absence. "unknown" is not a category
# with an identity — it is missing data — so it wears muted gray rather than
# consuming a categorical hue.
WORK_MODE_ORDER = ["remote", "hybrid", "onsite", "unknown"]


def palette(dark: bool = False) -> dict:
    return DARK if dark else LIGHT


def work_mode_colors(dark: bool = False) -> list[str]:
    p = palette(dark)
    return [p["categorical"][0], p["categorical"][1], p["categorical"][2], p["muted"]]


def resolve_tech_colors(
    selected: list[str], ranked_techs: list[str], dark: bool = False
) -> tuple[list[str], list[str]]:
    """Assign colors to the selected techs. Returns (domain, range) for Altair.

    Series always occupy CONSECUTIVE slots from the top of the palette, ordered
    by each tech's global mention rank.

    Why not a fixed tech->slot map, which would keep a tech's color constant as
    others come and go? Two reasons, both discovered by validation rather than
    argument:

      1. With 124 techs and 8 slots, a fixed map collides. Python (rank 0) and
         Rust (rank 8) both land on blue, and that pair is in the default
         selection.
      2. Resolving collisions by skipping to the next free slot produces a
         non-consecutive subset — and the palette's CVD guarantees only cover
         ADJACENT slots in the documented order. Slots {0,1,2,5,6} put magenta
         beside orange: normal-vision deltaE 11.6, a hard FAIL against the floor
         of 15. Two series full-color readers cannot reliably tell apart.

    So compaction wins. The cost is that changing the selection can recolor the
    other series; the benefit is that every rendered combination inherits the
    validated adjacency and no two series ever share a hue. Identity is carried
    by the legend and the line-end labels, not by color memory across edits.
    """
    pal = palette(dark)["categorical"]
    rank = {t: i for i, t in enumerate(ranked_techs)}
    ordered = sorted(selected, key=lambda t: rank.get(t, len(rank)))[: len(pal)]
    return ordered, pal[: len(ordered)]


def altair_config(chart, dark: bool = False):
    """Recessive chrome: hairline grid, muted axes, no view border."""
    p = palette(dark)
    return (
        chart.configure_view(strokeWidth=0, fill=p["surface"])
        .configure_axis(
            grid=True,
            gridColor=p["grid"],
            gridWidth=1,
            domainColor=p["axis"],
            tickColor=p["axis"],
            labelColor=p["text_secondary"],
            titleColor=p["text_secondary"],
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight="normal",
        )
        .configure_legend(
            labelColor=p["text_secondary"],
            titleColor=p["text_secondary"],
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight="normal",
            symbolType="stroke",
            symbolStrokeWidth=3,
        )
        .configure_title(color=p["text_primary"], fontSize=13, fontWeight=600, anchor="start")
    )
