"""Figure 11: current Mendelian zero-shot and linear-probe leaderboards.

The zero-shot panel contains the six headline models from the original Figure
11 selection. The probe panel is deliberately restricted to the four models
that overlap that zero-shot set, matching the canonical review in issue #370.

Run:
    uv run python -m plots.blog.figure11_leaderboard_heatmap

Outputs:
    plots/output/blog/figure11_leaderboard_heatmap__mendelian_{llr,probe}.{svg,png,pdf}
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from marin_dna.pipelines.evals.leaderboard import (
    DEFAULT_PROTOCOL,
    normalized_rows,
    probe_normalized_rows,
)
from plots.blog._leaderboard import render_heatmap, table_from_normalized

ZERO_SHOT_DISPLAYS: frozenset[str] = frozenset(
    {
        "GPN-Star (M)",
        "AlphaGenome",
        "exp135-1B-m5.1",
        "Evo 2 (40B)",
        "Evo 2 (7B)",
        "Evo 2 (1B base)",
    }
)
PROBE_DISPLAYS: frozenset[str] = frozenset(
    {
        "exp135-1B-m5.1",
        "Evo 2 (40B)",
        "Evo 2 (7B)",
        "Evo 2 (1B base)",
    }
)
DISPLAY_OVERRIDES: dict[str, str] = {
    "exp135-1B-m5.1": "MarinDNA (1B/m5.1)",
}


def _table_for_displays(
    rows: pl.DataFrame,
    expected_displays: frozenset[str],
) -> pd.DataFrame:
    """Select and require the exact editorial model set for one panel."""
    table = table_from_normalized(rows)
    mask = table.index.get_level_values("method_display").isin(expected_displays)
    table = table[mask]
    actual_displays = frozenset(table.index.get_level_values("method_display"))
    assert actual_displays == expected_displays, (
        "leaderboard model set mismatch: "
        f"missing={sorted(expected_displays - actual_displays)}, "
        f"unexpected={sorted(actual_displays - expected_displays)}"
    )
    table.index = pd.MultiIndex.from_tuples(
        [
            (DISPLAY_OVERRIDES.get(display, display), family)
            for display, family in table.index
        ],
        names=table.index.names,
    )
    return table


def build_mendelian_llr() -> None:
    """Render the six canonical-protocol zero-shot headline rows."""
    rows = (
        normalized_rows("mendelian_traits")
        .with_columns(
            pl.col("family")
            .replace_strict(DEFAULT_PROTOCOL, default=None)
            .alias("_default")
        )
        .filter(pl.col("protocol") == pl.col("_default"))
    )
    render_heatmap(
        _table_for_displays(rows, ZERO_SHOT_DISPLAYS),
        title="Mendelian VEP benchmark — AUPRC (%) · zero-shot LLR",
        output_name="figure11_leaderboard_heatmap__mendelian_llr",
    )


def build_mendelian_probe() -> None:
    """Render only the four probe rows that overlap the zero-shot panel."""
    render_heatmap(
        _table_for_displays(
            probe_normalized_rows("mendelian_traits"),
            PROBE_DISPLAYS,
        ),
        title="Mendelian VEP benchmark — AUPRC (%) · frozen-embedding linear probe",
        output_name="figure11_leaderboard_heatmap__mendelian_probe",
    )


def build() -> None:
    build_mendelian_llr()
    build_mendelian_probe()


if __name__ == "__main__":
    build()
