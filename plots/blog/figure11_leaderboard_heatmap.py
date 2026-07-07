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
)
from plots.blog._leaderboard import render_heatmap, table_from_normalized


def build_mendelian_llr() -> None:
    """M·LLR: each family at its default (canonical zero-shot) protocol."""
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
        table_from_normalized(rows),
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


def build() -> None:
    build_mendelian_llr()
    build_mendelian_probe()


if __name__ == "__main__":
    build()
