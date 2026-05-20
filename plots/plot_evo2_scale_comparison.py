"""Evo 2 scale-comparison analysis for issue #203 — first iteration.

Loads the three Evo 2 prediction parquets posted in the 2026-05-20 comment
of issue #131 (PR #194 k=9 Mendelian train split). Runs a bug-hunt block
(per-checkpoint LLR correlation, sign-direction sanity, statistical
significance per subset), then computes AUPRC + MRR-within-group per
(model, subset) and writes a score-histogram figure plus an outlier table.

Outputs (all under `plots/output/evo2_scale_comparison/`):

- `metrics.parquet` — per (model, score_type, subset) AUPRC + MRR with SE.
- `significance.parquet` — per (subset, score_type, model_pair) Wald Z + p
  for AUPRC and MRR gaps.
- `correlations.parquet` — per-subset Pearson and Spearman between each
  pair of checkpoint LLRs.
- `outliers_top20.parquet` — per subset, the 20 variants with the largest
  |Δscore(40B − 1B_base)|.
- `score_histograms_{minus_llr,next_token_jsd_mean}.{svg,png}` — per-subset
  pos-vs-neg score histograms, one panel per subset, models overlaid.

Usage:
    uv run python plots/plot_evo2_scale_comparison.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score

from bolinas.pipelines.evals.metrics import (
    GLOBAL_SUBSET,
    auprc_with_bootstrap_se,
    mrr_within_group,
    paired_metric_delta_bootstrap,
)

GIST_URLS: dict[str, str] = {
    "evo2_1b_base": "https://gist.githubusercontent.com/gonzalobenegas/3649e68fb63ca1f3443e4486078eb4d8/raw/b6c254849a71ca0783a24218a4fe9037e887e8f7/evo2_1b_base_train.parquet",
    "evo2_7b": "https://gist.githubusercontent.com/gonzalobenegas/3649e68fb63ca1f3443e4486078eb4d8/raw/e72d3d2e14955a670b8229dc8d525a69ea88c05c/evo2_7b_train.parquet",
    "evo2_40b": "https://gist.githubusercontent.com/gonzalobenegas/3649e68fb63ca1f3443e4486078eb4d8/raw/2b425e759811c201ca806ae4c8733fd7732220a6/evo2_40b_train.parquet",
}
MODELS: tuple[str, ...] = ("evo2_1b_base", "evo2_7b", "evo2_40b")
MODEL_COLORS: dict[str, str] = {
    "evo2_1b_base": "#3b1a64",
    "evo2_7b": "#3a6fa0",
    "evo2_40b": "#e8d840",
}
SCORE_COLS: tuple[str, ...] = ("minus_llr", "next_token_jsd_mean")
KEY_COLS: tuple[str, ...] = (
    "chrom",
    "pos",
    "ref",
    "alt",
    "label",
    "subset",
    "match_group",
)
OUT_DIR = Path(__file__).parent / "output" / "evo2_scale_comparison"

# Dashboard parquet checked as regression anchor — re-fetched at runtime.
DASHBOARD_PARQUET_URL = (
    "https://openathena.ai/bolinas-dna/_file/data/leaderboard.9c4a2b57.parquet"
)


def load_predictions() -> pd.DataFrame:
    """Load and merge the three gist parquets, keyed on (chrom,pos,ref,alt).

    Returns a long-format frame with one row per (variant, model)."""
    rows: list[pd.DataFrame] = []
    for model, url in GIST_URLS.items():
        df = pd.read_parquet(url)
        df = df.assign(model=model)
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    # Sanity: every (chrom,pos,ref,alt) should have one row per model.
    counts = df.groupby(["chrom", "pos", "ref", "alt"])["model"].nunique()
    assert (counts == len(MODELS)).all(), (
        f"variant-row counts uneven across models: {counts.value_counts().to_dict()}"
    )
    return df


def pivot_score(long_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Wide frame: one row per variant with columns label, subset,
    match_group, {model}_{score_col} for each model."""
    keys = list(KEY_COLS)
    pivoted = long_df.pivot(index=keys, columns="model", values=score_col)
    pivoted.columns = [f"{m}__{score_col}" for m in pivoted.columns]
    return pivoted.reset_index()


def bug_hunt_correlations(long_df: pd.DataFrame) -> pd.DataFrame:
    """Per-subset (and global) Pearson + Spearman correlation between each
    pair of checkpoints on `minus_llr`. Flags any pair < 0.3."""
    wide = pivot_score(long_df, "minus_llr")
    rows: list[dict] = []
    subsets = sorted(wide["subset"].unique().tolist()) + [GLOBAL_SUBSET]
    for subset in subsets:
        sub = wide if subset == GLOBAL_SUBSET else wide[wide["subset"] == subset]
        for i, m1 in enumerate(MODELS):
            for m2 in MODELS[i + 1 :]:
                x = sub[f"{m1}__minus_llr"].to_numpy()
                y = sub[f"{m2}__minus_llr"].to_numpy()
                pear = float(pearsonr(x, y).statistic)
                spear = float(spearmanr(x, y).statistic)
                rows.append(
                    {
                        "subset": subset,
                        "model_a": m1,
                        "model_b": m2,
                        "n": len(sub),
                        "pearson": pear,
                        "spearman": spear,
                        "flag_low": pear < 0.3,
                    }
                )
    return pd.DataFrame(rows)


def bug_hunt_dashboard_anchor(long_df: pd.DataFrame) -> pd.DataFrame:
    """Regression check: AUPRC computed locally from gist parquets must
    reproduce the dashboard's LLR-protocol values to ≤ 1e-4."""
    dash = pd.read_parquet(DASHBOARD_PARQUET_URL)
    dash = dash[
        (dash["dataset"] == "mendelian_traits")
        & (dash["protocol"] == "LLR")
        & (dash["method_id"].isin(MODELS))
    ][["method_id", "subset", "value", "n_positives"]]

    rows: list[dict] = []
    for model in MODELS:
        sub = long_df[long_df["model"] == model]
        # Per-subset
        for subset in sorted(sub["subset"].unique()):
            s = sub[sub["subset"] == subset]
            local = float(average_precision_score(s["label"], s["minus_llr"]))
            rows.append({"method_id": model, "subset": subset, "local_auprc": local})
        # Global
        rows.append(
            {
                "method_id": model,
                "subset": GLOBAL_SUBSET,
                "local_auprc": float(
                    average_precision_score(sub["label"], sub["minus_llr"])
                ),
            }
        )
    local_df = pd.DataFrame(rows)
    merged = local_df.merge(
        dash.rename(columns={"value": "dashboard_auprc"}),
        on=["method_id", "subset"],
        how="left",
    )
    merged["delta"] = merged["local_auprc"] - merged["dashboard_auprc"]
    merged["matches"] = merged["delta"].abs() < 1e-4
    n_matchable = int(merged["dashboard_auprc"].notna().sum())
    n_match = int((merged["matches"] & merged["dashboard_auprc"].notna()).sum())
    print(
        f"  dashboard-anchor regression: {n_match}/{n_matchable} (subset, model) "
        f"cells reproduce dashboard AUPRC within 1e-4."
    )
    assert n_match == n_matchable, (
        "Local AUPRC diverges from dashboard — bug somewhere. "
        f"Diffs > 1e-4: {merged[~merged['matches'] & merged['dashboard_auprc'].notna()]}"
    )
    return merged


def bug_hunt_sign_direction(long_df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, subset) Pearson correlation between `label` and
    `minus_llr`. All should be positive — minus_llr is meant to assign
    larger values to deleterious (label=1). Flags any negative
    correlation as a sign-flip bug."""
    rows: list[dict] = []
    for model in MODELS:
        sub = long_df[long_df["model"] == model]
        for subset in sorted(sub["subset"].unique()):
            s = sub[sub["subset"] == subset]
            if s["label"].nunique() < 2:
                continue
            r = float(pearsonr(s["label"], s["minus_llr"]).statistic)
            rows.append(
                {
                    "model": model,
                    "subset": subset,
                    "n": len(s),
                    "pearson_label_vs_score": r,
                    "flag_negative": r < 0,
                }
            )
    return pd.DataFrame(rows)


def per_model_metrics(
    long_df: pd.DataFrame, score_col: str, *, n_bootstrap: int = 1000, rng: int = 0
) -> pd.DataFrame:
    """Per (model, subset) AUPRC and MRR with cluster-bootstrap SE."""
    rows: list[dict] = []
    for model in MODELS:
        sub = long_df[long_df["model"] == model]
        subsets = sorted(sub["subset"].unique())
        for subset in subsets:
            s = sub[sub["subset"] == subset]
            auprc = auprc_with_bootstrap_se(
                s["label"],
                s[score_col],
                s["match_group"],
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
            mrr = mrr_within_group(
                s["label"],
                s[score_col],
                s["match_group"],
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
            rows.append(
                {
                    "model": model,
                    "score_type": score_col,
                    "subset": subset,
                    "auprc_value": auprc["value"],
                    "auprc_se": auprc["se"],
                    "mrr_value": mrr["value"],
                    "mrr_se": mrr["se"],
                    "n_groups": auprc["n_groups"],
                    "n_rows": auprc["n_rows"],
                }
            )
        # Global
        auprc = auprc_with_bootstrap_se(
            sub["label"],
            sub[score_col],
            sub["match_group"],
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        mrr = mrr_within_group(
            sub["label"],
            sub[score_col],
            sub["match_group"],
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        rows.append(
            {
                "model": model,
                "score_type": score_col,
                "subset": GLOBAL_SUBSET,
                "auprc_value": auprc["value"],
                "auprc_se": auprc["se"],
                "mrr_value": mrr["value"],
                "mrr_se": mrr["se"],
                "n_groups": auprc["n_groups"],
                "n_rows": auprc["n_rows"],
            }
        )
    return pd.DataFrame(rows)


def model_pair_significance(
    long_df: pd.DataFrame, *, n_bootstrap: int = 1000, rng: int = 0
) -> pd.DataFrame:
    """Paired cluster-bootstrap Z + p-values for AUPRC and MRR gaps
    between each pair of models, per (score_type, subset).

    Uses ``paired_metric_delta_bootstrap`` so the SE picks up the
    cross-model correlation (both models score the same match_groups).
    This is generally tighter than the independence formula
    ``sqrt(SE_a² + SE_b²)`` and gives the correct frequentist p for a
    paired comparison."""
    rows: list[dict] = []
    for score_type in SCORE_COLS:
        for subset in sorted(long_df["subset"].unique().tolist()) + [GLOBAL_SUBSET]:
            sub = (
                long_df
                if subset == GLOBAL_SUBSET
                else long_df[long_df["subset"] == subset]
            )
            wide = pivot_score(sub, score_type)
            label = wide["label"].to_numpy()
            mg = wide["match_group"].to_numpy()
            for i, m1 in enumerate(MODELS):
                for m2 in MODELS[i + 1 :]:
                    sa = wide[f"{m1}__{score_type}"].to_numpy()
                    sb = wide[f"{m2}__{score_type}"].to_numpy()
                    for metric in ("auprc", "mrr"):
                        res = paired_metric_delta_bootstrap(
                            label,
                            sa,
                            sb,
                            mg,
                            metric,
                            n_bootstrap=n_bootstrap,
                            rng=rng,
                        )
                        rows.append(
                            {
                                "score_type": score_type,
                                "subset": subset,
                                "metric": metric,
                                "model_a": m1,
                                "model_b": m2,
                                "value_a": res["value_a"],
                                "value_b": res["value_b"],
                                "delta": res["delta"],
                                "paired_se": res["se"],
                                "z": res["z"],
                                "p_two_sided": res["p_two_sided"],
                                "significant_05": res["p_two_sided"] < 0.05,
                                "n_groups": res["n_groups"],
                            }
                        )
    return pd.DataFrame(rows)


def find_outliers(
    long_df: pd.DataFrame, *, score_col: str = "minus_llr", top_n: int = 20
) -> pd.DataFrame:
    """Per subset, top_n variants with the largest |Δscore(40B − 1B_base)|."""
    wide = pivot_score(long_df, score_col)
    wide["delta_40b_minus_1b"] = (
        wide[f"evo2_40b__{score_col}"] - wide[f"evo2_1b_base__{score_col}"]
    )
    rows: list[pd.DataFrame] = []
    for subset in sorted(wide["subset"].unique()):
        sub = wide[wide["subset"] == subset].copy()
        sub["abs_delta"] = sub["delta_40b_minus_1b"].abs()
        sub_top = sub.nlargest(top_n, "abs_delta")
        rows.append(sub_top)
    return pd.concat(rows, ignore_index=True)


def plot_score_histograms(long_df: pd.DataFrame, score_col: str, out_dir: Path) -> None:
    """Per-subset score histograms: positives vs negatives, one panel per
    subset, models overlaid as transparent histograms."""
    subsets = sorted(long_df["subset"].unique())
    n_cols = 3
    n_rows = math.ceil(len(subsets) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 2.8 * n_rows),
        sharex=False,
    )
    axes = np.array(axes).reshape(-1)
    for ax, subset in zip(axes, subsets):
        sub = long_df[long_df["subset"] == subset]
        # Auto-range across all models for a fair side-by-side view.
        lo, hi = (
            float(sub[score_col].quantile(0.005)),
            float(sub[score_col].quantile(0.995)),
        )
        bins = np.linspace(lo, hi, 41)
        for model in MODELS:
            sm = sub[sub["model"] == model]
            pos = sm[sm["label"] == 1][score_col].to_numpy()
            neg = sm[sm["label"] == 0][score_col].to_numpy()
            color = MODEL_COLORS[model]
            ax.hist(
                neg,
                bins=bins,
                density=True,
                alpha=0.25,
                color=color,
                label=f"{model.split('_', 1)[1]} neg",
            )
            ax.hist(
                pos,
                bins=bins,
                density=True,
                alpha=0.6,
                color=color,
                histtype="step",
                linewidth=1.4,
                label=f"{model.split('_', 1)[1]} pos",
            )
        ax.set_title(
            f"{subset} (n_pos={int((sub['label'] == 1).sum() / len(MODELS))})",
            fontsize=9,
        )
        ax.set_xlabel(score_col, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
    # Hide unused axes
    for ax in axes[len(subsets) :]:
        ax.set_visible(False)
    # Single legend on the first axis
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"Evo 2 scale comparison — per-subset score histograms ({score_col})\n"
        "Filled = negatives; step = positives. Same data; only model varies.",
        fontsize=10,
    )
    # Reserve bottom band for the legend so the panels don't overlap it,
    # and rely on bbox_inches='tight' to include the legend in the saved
    # bbox even though it sits below the figure rect.
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    stem = f"score_histograms_{score_col}"
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def find_outliers_extended(
    long_df: pd.DataFrame, *, score_col: str = "minus_llr", top_n: int = 20
) -> dict[str, pd.DataFrame]:
    """For each model and each large subset, return the top-N highest-
    scoring negatives ('false positives' by score-ranking) and the top-N
    lowest-scoring positives ('false negatives'). Useful for the
    misclassification-character analysis."""
    out: dict[str, pd.DataFrame] = {}
    for model in MODELS:
        fp_rows: list[pd.DataFrame] = []
        fn_rows: list[pd.DataFrame] = []
        sub = long_df[long_df["model"] == model]
        for subset in sorted(sub["subset"].unique()):
            s = sub[sub["subset"] == subset]
            neg = s[s["label"] == 0]
            pos = s[s["label"] == 1]
            if len(neg) >= top_n:
                fp_rows.append(neg.nlargest(top_n, score_col).assign(kind="FP"))
            if len(pos) >= top_n:
                fn_rows.append(pos.nsmallest(top_n, score_col).assign(kind="FN"))
        out[f"{model}_FP"] = (
            pd.concat(fp_rows, ignore_index=True) if fp_rows else pd.DataFrame()
        )
        out[f"{model}_FN"] = (
            pd.concat(fn_rows, ignore_index=True) if fn_rows else pd.DataFrame()
        )
    return out


def merge_annotations(long_df: pd.DataFrame) -> pd.DataFrame:
    """Join the HF dataset's richer annotations (AF, consequence, gene
    ids, distance bins, clinvar_id, trait) onto the gist predictions by
    (chrom, pos, ref, alt). Downloads + caches the HF dataset once."""
    from datasets import load_dataset

    ds = load_dataset(
        "bolinas-dna/evals_mendelian_traits",
        revision="4aed58e50c5dea0b878a665007af2ef9e5108e9f",
        split="train",
    )
    ann = ds.to_pandas()[
        [
            "chrom",
            "pos",
            "ref",
            "alt",
            "AF",
            "consequence_final",
            "exon_closest_pc_gene_id",
            "distance_exon_pc",
            "tss_closest_pc_gene_id",
            "distance_tss_pc",
            "clinvar_id",
            "trait",
            "source",
            "distance_exon_pc_bin",
        ]
    ]
    ann["chrom"] = ann["chrom"].astype(str)
    long = long_df.copy()
    long["chrom"] = long["chrom"].astype(str)
    merged = long.merge(ann, on=["chrom", "pos", "ref", "alt"], how="left")
    assert len(merged) == len(long_df), (
        f"merge changed row count: {len(long_df)} → {len(merged)}"
    )
    n_unmatched = int(merged["consequence_final"].isna().sum())
    if n_unmatched:
        print(f"  ⚠ {n_unmatched} rows unmatched in HF annotation join")
    return merged


def rank_within_group(
    long_df: pd.DataFrame, score_col: str = "minus_llr"
) -> pd.DataFrame:
    """One row per (match_group, model) with rank-of-positive (1=best)."""
    from scipy.stats import rankdata

    rows: list[dict] = []
    for model in MODELS:
        sub = long_df[long_df["model"] == model]
        for (mg, subset), g in sub.groupby(["match_group", "subset"], sort=False):
            scores = g[score_col].to_numpy()
            labels = g["label"].astype(int).to_numpy()
            ranks = rankdata(-scores, method="average")
            rank_pos = float(ranks[labels == 1][0])
            rows.append(
                {
                    "match_group": int(mg),
                    "subset": subset,
                    "model": model,
                    "rank_pos": rank_pos,
                    "group_size": int(len(g)),
                }
            )
    return pd.DataFrame(rows)


def cross_scale_misclassification(
    ann_df: pd.DataFrame,
    score_col: str = "minus_llr",
    subset: str = "missense_variant",
) -> dict[str, pd.DataFrame]:
    """For a subset, classify each match_group by rank-of-positive in
    each model and bucket cross-scale patterns. Returns per-group
    classification and a summary table."""
    sub = ann_df[ann_df["subset"] == subset]
    ranks = rank_within_group(sub, score_col=score_col)
    rank_wide = ranks.pivot(
        index="match_group", columns="model", values="rank_pos"
    ).reset_index()
    rank_wide.columns.name = None

    pos = sub[(sub["model"] == MODELS[0]) & (sub["label"] == 1)][
        [
            "match_group",
            "chrom",
            "pos",
            "ref",
            "alt",
            "AF",
            "consequence_final",
            "exon_closest_pc_gene_id",
            "distance_exon_pc",
            "clinvar_id",
            "trait",
        ]
    ].drop_duplicates("match_group")
    merged = rank_wide.merge(pos, on="match_group", how="left")

    r1 = merged["evo2_1b_base"]
    r4 = merged["evo2_40b"]
    merged["scale_failure"] = (r1 <= 3) & (r4 >= 7)
    merged["scale_success"] = (r1 >= 7) & (r4 <= 3)
    merged["always_hit"] = (merged[list(MODELS)] == 1).all(axis=1)
    merged["always_miss"] = (merged[list(MODELS)] >= 7).all(axis=1)

    summary_rows: list[dict] = []
    for bucket in ("always_hit", "scale_success", "scale_failure", "always_miss"):
        b = merged[merged[bucket]]
        summary_rows.append(
            {
                "bucket": bucket,
                "n_match_groups": int(b.shape[0]),
                "mean_AF": float(b["AF"].mean()) if len(b) else float("nan"),
                "median_AF": float(b["AF"].median()) if len(b) else float("nan"),
                "frac_rare_under_1pct": (
                    float((b["AF"] < 0.01).mean()) if len(b) else float("nan")
                ),
                "n_unique_genes": int(b["exon_closest_pc_gene_id"].nunique()),
                "mean_distance_exon": (
                    float(b["distance_exon_pc"].mean()) if len(b) else float("nan")
                ),
                "n_with_clinvar_id": int(b["clinvar_id"].notna().sum()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    return {"per_group": merged, "summary": summary_df}


def plot_rank_crosstab(per_group: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap: rank-of-positive in 1B vs in 40B (missense match_groups)."""
    import matplotlib.colors as mcolors

    r1 = per_group["evo2_1b_base"].round().astype(int)
    r4 = per_group["evo2_40b"].round().astype(int)
    counts = pd.crosstab(r1, r4)
    counts = counts.reindex(index=range(1, 11), columns=range(1, 11), fill_value=0)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    vmax = max(int(counts.values.max()), 1)
    im = ax.imshow(
        counts.values,
        cmap="viridis",
        norm=mcolors.LogNorm(vmin=1, vmax=vmax),
        origin="lower",
    )
    ax.set_xticks(range(10))
    ax.set_xticklabels(range(1, 11))
    ax.set_yticks(range(10))
    ax.set_yticklabels(range(1, 11))
    ax.set_xlabel("rank-of-positive in evo2_40b (1 = best)")
    ax.set_ylabel("rank-of-positive in evo2_1b_base")
    ax.set_title(
        "Missense match_groups — rank-of-positive cross-tab\n"
        "Off-diagonal upper-right = 40B worse than 1B (scale-failure)"
    )
    for i in range(10):
        for j in range(10):
            v = int(counts.values[i, j])
            if v:
                ax.text(
                    j,
                    i,
                    str(v),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if v < vmax / 4 else "black",
                )
    plt.colorbar(im, ax=ax, label="match_group count (log)")
    fig.tight_layout()
    stem = "rank_crosstab_missense_1b_vs_40b"
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scale_bucket_characteristics(per_group: pd.DataFrame, out_dir: Path) -> None:
    """Compare AF + distance-to-exon distributions across cross-scale buckets."""
    bucket_labels = {
        "always_hit": "All models rank pos = 1",
        "scale_success": "1B rank ≥7 → 40B rank ≤3",
        "scale_failure": "1B rank ≤3 → 40B rank ≥7",
        "always_miss": "All models rank pos ≥7",
    }
    bucket_colors = {
        "always_hit": "#2e8c8c",
        "scale_success": "#3a6fa0",
        "scale_failure": "#d75f00",
        "always_miss": "#888888",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    af_bins = np.logspace(-6, 0, 40)
    for bucket, label in bucket_labels.items():
        af = per_group[per_group[bucket]]["AF"].dropna()
        if len(af):
            axes[0].hist(
                np.clip(af, 1e-6, 1),
                bins=af_bins,
                alpha=0.7,
                label=f"{label} (n={len(af)})",
                histtype="step",
                linewidth=1.7,
                color=bucket_colors[bucket],
            )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("AF (gnomAD)")
    axes[0].set_ylabel("# match_groups")
    axes[0].set_title("Allele frequency of positive, by cross-scale bucket")
    axes[0].legend(fontsize=8, loc="upper left")

    d_bins = np.linspace(0, 100, 41)
    for bucket, label in bucket_labels.items():
        d = per_group[per_group[bucket]]["distance_exon_pc"].dropna()
        if len(d):
            axes[1].hist(
                np.clip(d, 0, 100),
                bins=d_bins,
                alpha=0.7,
                label=f"{label} (n={len(d)})",
                histtype="step",
                linewidth=1.7,
                color=bucket_colors[bucket],
            )
    axes[1].set_xlabel("distance to nearest exon (clipped at 100 nt)")
    axes[1].set_ylabel("# match_groups")
    axes[1].set_title("Distance-to-exon of positive, by cross-scale bucket")
    axes[1].legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Missense — what kinds of variants degrade with scale?\n"
        "(positives' allele frequency + distance to nearest exon)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    stem = "scale_bucket_characteristics_missense"
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scale_curves(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    """Per-subset {AUPRC, MRR} vs model scale, with bootstrap-SE bands."""
    score_col = "minus_llr"
    df = metrics_df[metrics_df["score_type"] == score_col].copy()
    subsets = sorted([s for s in df["subset"].unique() if s != GLOBAL_SUBSET])
    n_cols = 3
    n_rows = math.ceil(len(subsets) / n_cols)
    for metric_label, value_col, se_col in [
        ("AUPRC", "auprc_value", "auprc_se"),
        ("MRR-within-group", "mrr_value", "mrr_se"),
    ]:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.5 * n_cols, 2.6 * n_rows),
            sharex=True,
            sharey=False,
        )
        axes = np.array(axes).reshape(-1)
        x = np.arange(len(MODELS))
        for ax, subset in zip(axes, subsets):
            sub = df[df["subset"] == subset].set_index("model").reindex(MODELS)
            y = sub[value_col].to_numpy()
            yerr = sub[se_col].to_numpy()
            ax.errorbar(
                x, y, yerr=yerr, marker="o", color="#222", capsize=3, linewidth=1.2
            )
            n_pos = int(sub["n_groups"].iloc[0])
            ax.set_title(f"{subset} (n_pos={n_pos})", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels(["1B_base", "7B", "40B"], fontsize=8)
            ax.set_ylabel(metric_label, fontsize=8)
        for ax in axes[len(subsets) :]:
            ax.set_visible(False)
        fig.suptitle(
            f"Evo 2 scale curves — {metric_label} on {score_col}\n"
            "Error bars: cluster-bootstrap SE over match_groups.",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0.0, 1, 0.94])
        stem = f"scale_curves_{value_col}"
        fig.savefig(out_dir / f"{stem}.svg")
        fig.savefig(out_dir / f"{stem}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")

    print("Loading 3 gist parquets…")
    long_df = load_predictions()
    print(
        f"  {len(long_df)} rows across {long_df['model'].nunique()} models, "
        f"{long_df['subset'].nunique()} subsets, "
        f"{long_df['match_group'].nunique()} match_groups."
    )

    print("Bug-hunt: dashboard regression anchor…")
    bug_hunt_dashboard_anchor(long_df).to_parquet(
        OUT_DIR / "regression_dashboard_anchor.parquet",
        index=False,
    )

    print("Bug-hunt: per-checkpoint LLR correlation…")
    corr = bug_hunt_correlations(long_df)
    corr.to_parquet(OUT_DIR / "correlations.parquet", index=False)
    low = corr[corr["flag_low"]]
    if not low.empty:
        print(f"  ⚠ {len(low)} subset-pair Pearson < 0.3:")
        print(low.to_string(index=False))
    else:
        print(f"  all Pearson ≥ 0.3 (min = {corr['pearson'].min():.3f}).")

    print("Bug-hunt: sign-direction sanity…")
    signs = bug_hunt_sign_direction(long_df)
    signs.to_parquet(OUT_DIR / "sign_direction.parquet", index=False)
    neg = signs[signs["flag_negative"]]
    if not neg.empty:
        print(
            f"  ⚠ {len(neg)} (model, subset) cells have negative label↔score correlation:"
        )
        print(neg.to_string(index=False))
    else:
        print(
            f"  all label↔minus_llr correlations positive "
            f"(min = {signs['pearson_label_vs_score'].min():.3f})."
        )

    print("Computing per-model metrics (AUPRC + MRR, n_bootstrap=1000)…")
    metrics_parts = [
        per_model_metrics(long_df, sc, n_bootstrap=1000, rng=0) for sc in SCORE_COLS
    ]
    metrics_df = pd.concat(metrics_parts, ignore_index=True)
    metrics_df.to_parquet(OUT_DIR / "metrics.parquet", index=False)
    print(f"  {len(metrics_df)} rows in metrics.parquet")

    print("Computing model-pair significance (paired cluster bootstrap)…")
    sig_df = model_pair_significance(long_df, n_bootstrap=1000, rng=0)
    sig_df.to_parquet(OUT_DIR / "significance.parquet", index=False)

    print("Finding outlier variants…")
    outliers = find_outliers(long_df, score_col="minus_llr", top_n=20)
    outliers.to_parquet(OUT_DIR / "outliers_top20.parquet", index=False)

    print("Plotting score histograms…")
    for sc in SCORE_COLS:
        plot_score_histograms(long_df, sc, OUT_DIR)

    print("Plotting scale curves…")
    plot_scale_curves(metrics_df, OUT_DIR)

    print("Annotating with HF dataset metadata (AF, gene, consequence, …)…")
    ann_df = merge_annotations(long_df)

    print(
        "Per-model top-N FP / FN (highest-scoring negatives, lowest-scoring positives)…"
    )
    fp_fn = find_outliers_extended(ann_df, score_col="minus_llr", top_n=20)
    fp_fn_df = pd.concat(
        [df.assign(slice=name) for name, df in fp_fn.items()], ignore_index=True
    )
    fp_fn_df.to_parquet(OUT_DIR / "top_fp_fn_per_model.parquet", index=False)

    print("Cross-scale misclassification analysis on missense…")
    miss = cross_scale_misclassification(
        ann_df, score_col="minus_llr", subset="missense_variant"
    )
    miss["per_group"].to_parquet(
        OUT_DIR / "missense_rank_per_group.parquet", index=False
    )
    miss["summary"].to_parquet(
        OUT_DIR / "missense_scale_bucket_summary.parquet", index=False
    )
    print(miss["summary"].to_string(index=False))

    print("Plotting rank cross-tab + scale-bucket characteristics…")
    plot_rank_crosstab(miss["per_group"], OUT_DIR)
    plot_scale_bucket_characteristics(miss["per_group"], OUT_DIR)

    print("Done. Inspect:")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p}")


if __name__ == "__main__":
    main()
