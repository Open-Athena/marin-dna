"""Evo 2 (and combined MarinDNA + Evo 2) scaling-probe Mendelian figures — the #341 analog.

Issue #341 showed, for the MarinDNA 8-rung scaling ladder (46M→4B, step-215573), that the
across-scale **missense** degradation is *readout-localized*: the frozen-embedding linear
probe's per-chromosome-weighted Mendelian AUPRC rises ~monotonically with scale while the
matched **zero-shot LLR** peaks (~128M) then degrades — a cheap linear head on the frozen
embedding recovers the loss. This recipe asks the same question of the **Evo 2** family
(1B / 7B / 40B) and overlays both families on shared axes.

The Evo 2 probe-metrics parquets (pinned Evo 2 gist — see `EVO2_PROBE_METRICS_GIST_BASE` in
`marin_dna.pipelines.evals.leaderboard`) carry BOTH `probe_score` and the matched
`minus_llr_avg` baseline scored on identical rows through the same `per_chrom_ap_table` as the
MarinDNA `compute_probe_metrics` output — same `concat_ref_delta` feature, same per-chrom-
weighted AUPRC, same 6 consequence subsets, and **identical per-subset row counts** as the
MarinDNA ladder. So probe-vs-LLR is paired within each model, and MarinDNA-vs-Evo 2 is
apples-to-apples (a `_sanity` gate asserts the n's match).

One visual language throughout — COLOR = model, STYLE = score (solid linear probe / dashed
zero-shot LLR); no score=color mapping. Six figures (seaborn figure-level `relplot`), each SVG +
PNG to `plots/output/family_scaling_probe_mendelian/`:

  marin_figure     — MarinDNA per-subset facet grid (blue).   } the #341 issue figures,
  marin_missense   — MarinDNA missense alone (blue).          } regenerated in the unified palette
  evo2_figure      — Evo 2 per-subset facet grid (orange).
  evo2_missense    — Evo 2 missense alone (orange; the direct #341/#302 analog).
  combined_figure  — both families per-subset grid (MarinDNA blue / Evo 2 orange).
  combined_missense— both families, missense alone (the money plot: the zero-shot missense
                     degradation across two independently-trained gLM families).

Metric = per-chromosome-weighted AUPRC (TraitGym / #331): a **point estimate**; the MarinDNA
ladder parquet carries no SE, so no error bars are drawn (matches #341). The 1:9-matched set
gives a 0.10 random baseline (dashed). Reported **per subset** (no macro-avg).

Usage:
    uv run python plots/family_scaling_probe_mendelian.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import polars as pl

from marin_dna.pipelines.evals.leaderboard import (
    EVO2_DATASET_SHORT,
    EVO2_PROBE_METRICS_GIST_BASE,
)

OUT_DIR = Path(__file__).resolve().parent / "output" / "family_scaling_probe_mendelian"
DATASET = "mendelian_traits"

# Prevalence baseline for the 1:9-matched Mendelian set (exactly 10% positives).
BASELINE = 0.10

# The two score types both parquets carry.
PROBE_ST, LLR_ST = "probe_score", "minus_llr_avg"
PROBE_LABEL, LLR_LABEL = "linear probe", "zero-shot LLR"
LABEL_BY_ST = {PROBE_ST: PROBE_LABEL, LLR_ST: LLR_LABEL}
SCORE_ORDER = [PROBE_LABEL, LLR_LABEL]

# One visual language for every figure: COLOR = model (family), STYLE = score (solid linear
# probe / dashed zero-shot LLR — a natural "two conditions of one series" cue). Single-family
# figures use that family's one color for both its lines; the combined figure uses both family
# colors. No score=color mapping anywhere (no probe=blue / LLR=red). Family colors are the
# dashboard palette (dashboard/family-colors.css) so every view agrees.
MARIN, EVO2 = "MarinDNA", "Evo 2"
FAMILY_ORDER = [MARIN, EVO2]
FAMILY_PALETTE = {MARIN: "#1f77b4", EVO2: "#ff7f0e"}  # dashboard family colors
SCORE_DASHES = {
    PROBE_LABEL: "",
    LLR_LABEL: (4, 1.5),
}  # "" == solid: probe solid, LLR dashed
SCORE_MARKERS = {
    PROBE_LABEL: "o",
    LLR_LABEL: "o",
}  # neutral dot — color+dash already carry it

# MarinDNA 8-rung ladder (issues #274 / #302 / #341), all at step-215573. Total param count
# per rung (#274 table) → x-axis position.
MARIN_SIZES = [
    "h640-p46M",
    "h768-p76M",
    "h896-p128M",
    "h1152-p255M",
    "h1408-p476M",
    "h1920-p1B",
    "h2432-p2B",
    "h2944-p4B",
]
MARIN_PARAMS = {
    "h640-p46M": 45.9e6,
    "h768-p76M": 75.5e6,
    "h896-p128M": 128.5e6,
    "h1152-p255M": 254.8e6,
    "h1408-p476M": 475.9e6,
    "h1920-p1B": 1.12e9,
    "h2432-p2B": 2.27e9,
    "h2944-p4B": 4.02e9,
}
MARIN_LABEL = {
    "h640-p46M": "46M",
    "h768-p76M": "76M",
    "h896-p128M": "128M",
    "h1152-p255M": "255M",
    "h1408-p476M": "476M",
    "h1920-p1B": "1B",
    "h2432-p2B": "2B",
    "h2944-p4B": "4B",
}

# Evo 2 rungs (issue #131 / #352). Params from dashboard/models.yaml.
EVO2_MODELS = ["evo2_1b_base", "evo2_7b", "evo2_40b"]
EVO2_PARAMS = {"evo2_1b_base": 1.0e9, "evo2_7b": 7.0e9, "evo2_40b": 4.0e10}
EVO2_LABEL = {"evo2_1b_base": "1B", "evo2_7b": "7B", "evo2_40b": "40B"}

# Subsets to plot, in facet order (missense first — the #302 focus). The coding / regulatory
# consequence classes shared with the #341 ladder, so the two families' panels align. (distal /
# non_coding_transcript_exon also clear the n>=300 probe gate but are dropped for cross-family
# alignment; mature_miRNA n=40 is below-gate.)
KEEP_SUBSETS = [
    "missense_variant",
    "synonymous_variant",
    "splicing",
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
    "tss_proximal",
]

# X-axis ticks. Per-family for the single-family figures; a curated spread for the combined
# ones (all 3 Evo 2 points + a readable MarinDNA spread — 10 raw ticks would collide in a
# small facet).
MARIN_TICKS = [(MARIN_PARAMS[s], MARIN_LABEL[s]) for s in MARIN_SIZES]
EVO2_TICKS = [(EVO2_PARAMS[m], EVO2_LABEL[m]) for m in EVO2_MODELS]
COMBINED_TICKS = [
    (45.9e6, "46M"),
    (128.5e6, "128M"),
    (475.9e6, "476M"),
    (1.12e9, "1B"),
    (4.02e9, "4B"),
    (7.0e9, "7B"),
    (4.0e10, "40B"),
]


def _disp(subset: str) -> str:
    return subset.replace("_variant", "")


def _marin_path(prefix: str, size: str) -> str:
    return f"{prefix}/scaling-v0.5-{size}-step-215573/{DATASET}.parquet"


def _evo2_path(model: str) -> str:
    short = EVO2_DATASET_SHORT[DATASET]  # "mendelian"
    return f"{EVO2_PROBE_METRICS_GIST_BASE}/{short}_{model}_train_probe_metrics.parquet"


def _read(path: str, what: str) -> pl.DataFrame:
    opts = {"aws_region": "us-east-2"} if path.startswith("s3://") else None
    try:
        return pl.read_parquet(path, storage_options=opts)
    except Exception as e:  # fail loud, name the missing cell
        raise RuntimeError(f"{what}: could not read {path} — is it published? ({e})")


def _normalize(
    df: pl.DataFrame, family: str, size_key: str, params: float, size_label: str
) -> pd.DataFrame:
    """One model's parquet → tidy rows [family, size_label, params, subset, score_type,
    value, se, n, n_pos], restricted to the two score types and the kept subsets. `se` is
    null-filled for the MarinDNA ladder (point estimates, no SE column)."""
    if "se" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("se"))
    out = (
        df.filter(
            pl.col("score_type").is_in([PROBE_ST, LLR_ST])
            & pl.col("subset").is_in(KEEP_SUBSETS)
        )
        .select(["score_type", "subset", "value", "se", "n", "n_pos"])
        .with_columns(
            pl.lit(family).alias("family"),
            pl.lit(size_key).alias("size_key"),
            pl.lit(size_label).alias("size_label"),
            pl.lit(params).alias("params"),
        )
    )
    return out.to_pandas()


def load(marin_prefix: str) -> pd.DataFrame:
    """Concatenate every MarinDNA rung + every Evo 2 rung into one tidy long frame, adding the
    seaborn display columns (`subset_disp`, `score`)."""
    frames = [
        _normalize(
            _read(_marin_path(marin_prefix, s), f"marin_dna {s}"),
            MARIN,
            s,
            MARIN_PARAMS[s],
            MARIN_LABEL[s],
        )
        for s in MARIN_SIZES
    ]
    frames += [
        _normalize(
            _read(_evo2_path(m), f"evo2 {m}"), EVO2, m, EVO2_PARAMS[m], EVO2_LABEL[m]
        )
        for m in EVO2_MODELS
    ]
    pdf = pd.concat(frames, ignore_index=True)
    pdf["subset_disp"] = pdf["subset"].map(_disp)
    pdf["score"] = pdf["score_type"].map(LABEL_BY_ST)
    return pdf


def _sanity(pdf: pd.DataFrame) -> None:
    """Guard the claims the figures rest on: finite values in [0,1], exact 1:9 prevalence, all 6
    subsets present per model, and — the apples-to-apples claim — identical per-subset row
    counts across every model of both families."""
    assert pdf["value"].notna().all(), "unexpected NaN AUPRC among kept subsets"
    assert ((pdf["value"] >= 0) & (pdf["value"] <= 1)).all(), "AUPRC outside [0,1]"
    # 1:9 matched → exactly 10% positives (n == 10 * n_pos), per subset.
    assert (pdf["n"] == 10 * pdf["n_pos"]).all(), "prevalence is not exactly 1:9"
    # Every model carries all 6 kept subsets (both score types).
    per_model = pdf.groupby(["family", "size_key"])["subset"].nunique()
    assert (per_model == len(KEEP_SUBSETS)).all(), (
        f"a model is missing kept subsets:\n{per_model}"
    )
    # Same variant set across families ⇒ one n per subset. This is what makes the combined
    # plot a level comparison rather than two differently-sampled curves.
    per_subset_n = pdf.groupby("subset")["n"].nunique()
    assert (per_subset_n == 1).all(), (
        f"per-subset n differs across models — rows are NOT identical:\n"
        f"{pdf.groupby('subset')['n'].unique()}"
    )


def _style_axes(g, ticks: list[tuple[float, str]]) -> None:
    """Log-x with size-label ticks, a dashed prevalence baseline, and a light grid."""
    x_vals = [v for v, _ in ticks]
    x_labels = [lab for _, lab in ticks]
    g.set(xscale="log")
    for ax in g.axes.flat:
        ax.axhline(BASELINE, ls="--", lw=0.8, color="gray", alpha=0.7)
        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.grid(True, alpha=0.3)


def _save(g, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = OUT_DIR / f"{stem}.svg"
    g.savefig(svg, dpi=200, bbox_inches="tight")
    g.savefig(svg.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {svg} (+ .png)")


def print_table(pdf: pd.DataFrame, family: str, order: list[str]) -> None:
    fam = pdf[pdf["family"] == family]
    print(
        f"\n=== {family}: per-chrom-weighted AUPRC — probe (P) vs zero-shot LLR (Z) ==="
    )
    for subset in KEEP_SUBSETS:
        sub = fam[fam["subset"] == subset]
        if sub.empty:
            continue
        print(f"\n{subset}")
        print(f"{'size':>6} | {'probe':>7} {'LLR':>7} {'Δ(P−Z)':>8}")
        for size_label in order:
            cell = sub[sub["size_label"] == size_label]

            def _v(st: str) -> float:
                hit = cell[cell["score_type"] == st]["value"]
                return float(hit.iloc[0]) if len(hit) else float("nan")

            pv, zv = _v(PROBE_ST), _v(LLR_ST)
            print(f"{size_label:>6} | {pv:>7.3f} {zv:>7.3f} {pv - zv:>+8.3f}")


def _encoding_kw(family: str | None) -> dict:
    """seaborn kwargs for the shared encoding — color = model, style = score. Combined
    (``family is None``): color varies by family (``hue``). Single family: one fixed family color
    for both its score lines. Either way STYLE (solid probe / dashed LLR) carries the score, so no
    figure maps score to color."""
    style_kw = dict(
        style="score",
        style_order=SCORE_ORDER,
        dashes=SCORE_DASHES,
        markers=SCORE_MARKERS,
    )
    if family is None:
        return dict(
            hue="family", hue_order=FAMILY_ORDER, palette=FAMILY_PALETTE, **style_kw
        )
    return dict(color=FAMILY_PALETTE[family], **style_kw)


def _finish_legend(sns, g, family: str | None) -> None:
    """Rebuild relplot's legend with long, marker-free proxy handles so solid (probe) vs dashed
    (zero-shot LLR) is unmistakable — relplot's own style handles are too short and carry a
    centered marker. Combined: a ``model`` color legend + a neutral ``score`` style legend. Single
    family: just the ``score`` legend, drawn in that family's color (the model is named in the
    title, not the legend)."""
    if g.legend is None:
        return
    from matplotlib.lines import Line2D

    g.legend.remove()
    fig = g.figure
    # Anchor snug to the right of the actual axes. relplot reserves a wide legend band by
    # shrinking the axes; anchoring at the figure edge (x=1.0) would leave that band as dead space.
    x = max(ax.get_position().x1 for ax in fig.axes) + 0.015
    if family is None:
        model_h = [
            Line2D([], [], color=FAMILY_PALETTE[m], lw=2.6, label=m)
            for m in FAMILY_ORDER
        ]
        leg_model = fig.legend(
            handles=model_h,
            title="model",
            loc="center left",
            bbox_to_anchor=(x, 0.60),
            frameon=False,
            handlelength=2.6,
        )
        fig.add_artist(leg_model)
        score_color, score_y = "0.25", 0.40  # neutral: color already means family
    else:
        score_color, score_y = (
            FAMILY_PALETTE[family],
            0.5,
        )  # match the single family's lines
    score_h = [
        Line2D([], [], color=score_color, lw=2.2, ls="-", label=PROBE_LABEL),
        Line2D([], [], color=score_color, lw=2.2, ls=(0, (6, 3)), label=LLR_LABEL),
    ]
    fig.legend(
        handles=score_h,
        title="score",
        loc="center left",
        bbox_to_anchor=(x, score_y),
        frameon=False,
        handlelength=3.4,
    )


def build_grid(
    sns, pdf: pd.DataFrame, ticks, suptitle: str, stem: str, family: str | None = None
):
    g = sns.relplot(
        data=pdf,
        x="params",
        y="value",
        col="subset_disp",
        col_order=[_disp(s) for s in KEEP_SUBSETS],
        col_wrap=3,
        kind="line",
        linewidth=1.8,
        markersize=7 if family is None else 8,
        facet_kws={"sharey": False},
        height=3.0,
        aspect=1.35,
        **_encoding_kw(family),
    )
    _style_axes(g, ticks)
    g.set_axis_labels("model size (params, log)", "per-chrom AUPRC")
    g.set_titles("{col_name}")
    _finish_legend(sns, g, family)
    g.figure.suptitle(suptitle, y=1.02)
    _save(g, stem)


def build_missense(
    sns, pdf: pd.DataFrame, ticks, suptitle: str, stem: str, family: str | None = None
):
    mdf = pdf[pdf["subset"] == "missense_variant"]
    g = sns.relplot(
        data=mdf,
        x="params",
        y="value",
        kind="line",
        linewidth=2.0,
        markersize=9,
        height=5,
        aspect=1.4,
        **_encoding_kw(family),
    )
    _style_axes(g, ticks)
    g.set_axis_labels("model size (params, log)", "per-chrom-weighted AUPRC")
    _finish_legend(sns, g, family)
    g.figure.suptitle(suptitle, y=1.02)
    _save(g, stem)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--marin-prefix",
        default="s3://oa-bolinas/snakemake/analysis/evals_v2/results/probe_metrics",
        help="S3 prefix holding the MarinDNA scaling-ladder probe_metrics parquets.",
    )
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")

    pdf = load(args.marin_prefix)
    _sanity(pdf)

    print_table(pdf, EVO2, [EVO2_LABEL[m] for m in EVO2_MODELS])
    print_table(pdf, MARIN, [MARIN_LABEL[s] for s in MARIN_SIZES])

    # Per-family views (each in its own family color, solid probe / dashed LLR). The MarinDNA
    # pair are the unified-palette regeneration of the #341 issue figures.
    marin = pdf[pdf["family"] == MARIN]
    evo2 = pdf[pdf["family"] == EVO2]
    build_grid(
        sns,
        marin,
        MARIN_TICKS,
        "MarinDNA — Mendelian VEP by consequence",
        "marin_figure",
        MARIN,
    )
    build_missense(
        sns,
        marin,
        MARIN_TICKS,
        "MarinDNA — Mendelian missense",
        "marin_missense",
        MARIN,
    )
    build_grid(
        sns,
        evo2,
        EVO2_TICKS,
        "Evo 2 — Mendelian VEP by consequence",
        "evo2_figure",
        EVO2,
    )
    build_missense(
        sns, evo2, EVO2_TICKS, "Evo 2 — Mendelian missense", "evo2_missense", EVO2
    )
    # Combined (both family colors), family=None (default).
    build_grid(
        sns, pdf, COMBINED_TICKS, "Mendelian VEP by consequence", "combined_figure"
    )
    build_missense(sns, pdf, COMBINED_TICKS, "Mendelian missense", "combined_missense")


if __name__ == "__main__":
    main()
