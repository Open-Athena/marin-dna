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
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.metrics import average_precision_score

from bolinas.pipelines.evals.metrics import (
    GLOBAL_SUBSET,
    auprc_with_bootstrap_se,
    mrr_within_group,
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


def model_pair_significance(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Wald Z + p-values for AUPRC and MRR gaps between each
    pair of models, per (score_type, subset). Independence-assumed
    combined SE — paired bootstrap would tighten."""
    rows: list[dict] = []
    grouped = metrics_df.groupby(["score_type", "subset"])
    for (score_type, subset), g in grouped:
        gm = g.set_index("model")
        for i, m1 in enumerate(MODELS):
            for m2 in MODELS[i + 1 :]:
                if m1 not in gm.index or m2 not in gm.index:
                    continue
                for metric in ("auprc", "mrr"):
                    v1, se1 = gm.loc[m1, f"{metric}_value"], gm.loc[m1, f"{metric}_se"]
                    v2, se2 = gm.loc[m2, f"{metric}_value"], gm.loc[m2, f"{metric}_se"]
                    delta = float(v2 - v1)
                    combined_se = float(math.sqrt(se1**2 + se2**2))
                    z = delta / combined_se if combined_se > 0 else 0.0
                    p = float(2 * (1 - norm.cdf(abs(z))))
                    rows.append(
                        {
                            "score_type": score_type,
                            "subset": subset,
                            "metric": metric,
                            "model_a": m1,
                            "model_b": m2,
                            "delta": delta,
                            "combined_se": combined_se,
                            "z": z,
                            "p_two_sided": p,
                            "significant_05": p < 0.05,
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
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    stem = f"score_histograms_{score_col}"
    fig.savefig(out_dir / f"{stem}.svg")
    fig.savefig(out_dir / f"{stem}.png", dpi=150)
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

    print("Computing model-pair significance…")
    sig_df = model_pair_significance(metrics_df)
    sig_df.to_parquet(OUT_DIR / "significance.parquet", index=False)

    print("Finding outlier variants…")
    outliers = find_outliers(long_df, score_col="minus_llr", top_n=20)
    outliers.to_parquet(OUT_DIR / "outliers_top20.parquet", index=False)

    print("Plotting score histograms…")
    for sc in SCORE_COLS:
        plot_score_histograms(long_df, sc, OUT_DIR)

    print("Plotting scale curves…")
    plot_scale_curves(metrics_df, OUT_DIR)

    print("Done. Inspect:")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p}")


if __name__ == "__main__":
    main()
