"""Figure 11 — Mendelian VEP leaderboard heatmap, AUPRC %, across worlds.

Redo of the blog's Figure 11 in Eric's style, driven from LIVE evals_v2 metrics
(the new Mendelian eval) instead of the hand-extracted CSV snapshot in Eric's post
repo. Renders one parallel figure per ready world:

  * M·LLR   — zero-shot LLR, each family at its ``DEFAULT_PROTOCOL`` (the canonical
              leaderboard number: marin_dna LLR, gpn_star cLLR, conservation score,
              alphagenome L2, evo2 LLR).
  * M·Probe — frozen-embedding linear probe (#347/#348), probe-capable families
              (marin_dna on S3, evo2 on the pinned gist).

The SGE worlds (S·LLR / S·Probe) live in a separate recipe — SGE is a per-accession
grid, not the matched-pair models × subset shape this heatmap renders.

Run:  uv run python -m plots.blog.figure11_leaderboard_heatmap
Out:  plots/output/blog/figure11_leaderboard_heatmap__mendelian_{llr,probe}.{svg,png,pdf}
"""

from __future__ import annotations

import polars as pl

from marin_dna.pipelines.evals.leaderboard import (
    DEFAULT_PROTOCOL,
    normalized_rows,
    probe_normalized_rows,
    sge_normalized_rows,
)
from plots.blog._leaderboard import (
    SGE_SUBSET_DISPLAY,
    SGE_SUBSET_ORDER,
    render_heatmap,
    table_from_normalized,
    table_from_sge,
)

# Headline models for the leaderboard figure — mirrors the blog's Fig 11 selection
# (editorial + tentative; easy to extend, e.g. add "scaling-v0.5 4B" / "exp166-v0.1-p1B").
# Applied to M·LLR to trim the full ~33-model registry down to the headline comparison;
# M·Probe / S·LLR are already curated by which models have probe / SGE metrics on S3.
HEADLINE_DISPLAYS: frozenset[str] = frozenset(
    {
        "GPN-Star (M)",
        "AlphaGenome",
        "exp135-1B-m5.1",
        "Evo 2 (40B)",
        "Evo 2 (7B)",
        "Evo 2 (1B base)",
    }
)


def build_mendelian_llr() -> None:
    """M·LLR: each family at its default (canonical zero-shot) protocol, headline set."""
    rows = (
        normalized_rows("mendelian_traits")
        .with_columns(
            pl.col("family")
            .replace_strict(DEFAULT_PROTOCOL, default=None)
            .alias("_default")
        )
        .filter(pl.col("protocol") == pl.col("_default"))
    )
    table = table_from_normalized(rows)
    table = table[
        table.index.get_level_values("method_display").isin(HEADLINE_DISPLAYS)
    ]
    render_heatmap(
        table,
        title="Mendelian VEP benchmark — AUPRC (%) · zero-shot LLR (new eval)",
        out_name="figure11_leaderboard_heatmap__mendelian_llr",
    )


def build_mendelian_probe() -> None:
    """M·Probe: frozen-embedding linear probe (single ``probe`` protocol)."""
    rows = probe_normalized_rows("mendelian_traits")
    if rows.height == 0:
        print("figure11: no probe rows for mendelian_traits — skipping M·Probe")
        return
    render_heatmap(
        table_from_normalized(rows),
        title="Mendelian VEP benchmark — AUPRC (%) · frozen-embedding linear probe",
        out_name="figure11_leaderboard_heatmap__mendelian_probe",
    )


def build_sge_llr() -> None:
    """S·LLR: SGE (saturation genome editing) AUPRC, across-gene macro, zero-shot LLR.

    SGE assays only coding/splice, so this is a narrower Macro / Missense / Splicing
    heatmap; ``score_type='minus_llr_avg'`` selects the FWD+RC-averaged LLR world.
    """
    rows = sge_normalized_rows("sge")
    if rows.height == 0:
        print("figure11: no SGE rows — skipping S·LLR")
        return
    render_heatmap(
        table_from_sge(rows, score_type="minus_llr_avg"),
        title="SGE benchmark — AUPRC (%) · zero-shot LLR",
        out_name="figure11_leaderboard_heatmap__sge_llr",
        subset_order=SGE_SUBSET_ORDER,
        subset_display=SGE_SUBSET_DISPLAY,
    )


def build() -> None:
    build_mendelian_llr()
    build_mendelian_probe()
    build_sge_llr()


if __name__ == "__main__":
    build()
