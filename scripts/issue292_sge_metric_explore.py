"""Explore candidate metrics for the SGE eval (issue #292).

Reads the evals_v2 *scores* parquet for one model on the `sge` dataset
(`results/scores/<model>/sge.parquet` — per-strand `llr_*`/`jsd_*` atoms +
every `evals_sge` column) plus the dataset's `calibrations.parquet` companion,
and computes a *menu* of candidate metrics. Nothing here is committed as "the"
SGE metric — the point is to look at several and compare (see #292).

Why the structure below:

- `function_score` is on a per-study scale that is NOT comparable across genes
  (BAP1 sigma ~ 0.03 vs RAD51C ~ 4.7), so every correlation is computed
  **per gene** and then macro-averaged (one vote per gene). Raw `function_score`
  is never pooled across genes.
- Score orientation: the model's signed `llr` = log P(alt)/P(ref); a disruptive
  variant has `llr < 0` and (for a loss-of-function assay) a low `function_score`,
  so we expect **Spearman(llr, function_score) > 0**. For the "predict abnormal"
  binary tasks the natural oriented score is `minus_llr = -llr` (higher = more
  damaging); `abs_llr` and `jsd` are direction-agnostic disruption magnitudes.

Run (CPU; reads the parquet from S3 via polars):

    uv run --group genome-s3 python scripts/issue292_sge_metric_explore.py \
        [s3://.../results/scores/<model>/sge.parquet]

Outputs tidy tables to stdout and writes `scratch/sge_metric_menu_<model>.csv`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from huggingface_hub import hf_hub_download
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from marin_dna.pipelines.evals.metrics import _correlation_with_bootstrap_se

DEFAULT_MODEL = "mix-v0.9-p1B-i24-exp135-m5.1-step-59158"
S3_SCORES = (
    "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores/"
    f"{{model}}/sge.parquet"
)
SGE_REPO = "bolinas-dna/evals_sge"

N_BOOTSTRAP = 1000
SEED = 0
N_MIN_CORR = 30  # min variants in a (gene[, consequence]) cell to report a correlation
N_MIN_BINARY = 20  # min variants (with >=5 of the minority class) for a binary metric


def load_scores(path: str) -> pd.DataFrame:
    """Scores parquet → pandas, with the derived strand-averaged atoms added."""
    df = pl.read_parquet(path).to_pandas()
    assert {"llr_fwd", "llr_rc", "jsd_fwd", "jsd_rc"}.issubset(df.columns), (
        f"missing per-strand atoms; have {sorted(df.columns)[:12]}…"
    )
    df["llr_avg"] = (df["llr_fwd"] + df["llr_rc"]) / 2
    df["jsd_avg"] = (df["jsd_fwd"] + df["jsd_rc"]) / 2
    assert df["function_score"].notna().all(), "function_score has NaN"
    return df


# Strand-resolved score columns the menu draws from.
LLR_COLS = ["llr_fwd", "llr_rc", "llr_avg"]
JSD_COLS = ["jsd_fwd", "jsd_rc", "jsd_avg"]


def _macro(values: list[float], ses: list[float]) -> tuple[float, float, int]:
    """Unweighted mean of per-gene estimates + SE-of-the-mean (sqrt(sum se^2)/k)."""
    vals = np.array(values, dtype=float)
    se_arr = np.array(ses, dtype=float)
    ok = ~np.isnan(vals)
    k = int(ok.sum())
    if k == 0:
        return float("nan"), float("nan"), 0
    macro = float(np.mean(vals[ok]))
    macro_se = float(np.sqrt(np.nansum(se_arr[ok] ** 2)) / k)
    return macro, macro_se, k


# --------------------------------------------------------------------------- #
# T1 — per-gene correlation of the model score vs the continuous function_score.
# --------------------------------------------------------------------------- #
def correlation_menu(df: pd.DataFrame) -> pd.DataFrame:
    """Per-gene Spearman/Pearson(score, function_score) → macro + macro|.|."""
    rng = np.random.default_rng(SEED)
    genes = sorted(df["gene"].unique())
    rows = []
    # Signed llr/jsd: correlation sign tells us direction; abs_llr is a distinct
    # (non-monotone) transform so it gets its own correlation. minus_llr is
    # redundant for correlation (just flips the sign), so it's omitted here.
    score_cols = LLR_COLS + JSD_COLS + [f"abs_{c}" for c in LLR_COLS]
    work = df.copy()
    for c in LLR_COLS:
        work[f"abs_{c}"] = work[c].abs()
    for method in ("spearman", "pearson"):
        for col in score_cols:
            per_gene, per_gene_se = [], []
            for g in genes:
                sub = work[work["gene"] == g]
                if len(sub) < N_MIN_CORR:
                    per_gene.append(np.nan)
                    per_gene_se.append(np.nan)
                    continue
                v, se = _correlation_with_bootstrap_se(
                    sub[col].to_numpy(float),
                    sub["function_score"].to_numpy(float),
                    method=method,
                    n_bootstrap=N_BOOTSTRAP,
                    rng=rng,
                )
                per_gene.append(v)
                per_gene_se.append(se)
            macro, macro_se, k = _macro(per_gene, per_gene_se)
            macro_abs, _, _ = _macro([abs(x) for x in per_gene], per_gene_se)
            rows.append(
                {
                    "target": "function_score",
                    "metric": method,
                    "score_type": col,
                    "macro": macro,
                    "macro_se": macro_se,
                    "macro_abs": macro_abs,
                    "n_genes": k,
                }
            )
    return pd.DataFrame(rows).sort_values(["metric", "macro_abs"], ascending=[True, False])


def correlation_per_consequence(df: pd.DataFrame, score_col: str = "llr_avg") -> pd.DataFrame:
    """Per-(gene x consequence_final) Spearman → macro within each consequence.

    Pooling raw function_score across genes is invalid, so each consequence's
    number is a macro-average over per-(gene) Spearmans within that consequence.
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for cons, csub in df.groupby("consequence_final"):
        per_gene, per_gene_se = [], []
        for g, gsub in csub.groupby("gene"):
            if len(gsub) < N_MIN_CORR:
                continue
            v, se = _correlation_with_bootstrap_se(
                gsub[score_col].to_numpy(float),
                gsub["function_score"].to_numpy(float),
                method="spearman",
                n_bootstrap=N_BOOTSTRAP,
                rng=rng,
            )
            per_gene.append(v)
            per_gene_se.append(se)
        if not per_gene:
            continue
        macro, macro_se, k = _macro(per_gene, per_gene_se)
        rows.append(
            {
                "consequence_final": cons,
                "score_type": score_col,
                "macro_spearman": macro,
                "macro_se": macro_se,
                "n_genes": k,
                "n_variants": len(csub),
            }
        )
    return pd.DataFrame(rows).sort_values("n_variants", ascending=False)


# --------------------------------------------------------------------------- #
# Binary "predict abnormal" tasks. Oriented score = minus_llr / abs_llr / jsd
# (higher = more damaging). AUROC/AUPRC point + row-bootstrap SE.
# --------------------------------------------------------------------------- #
def _binary_score_cols(df: pd.DataFrame) -> dict[str, np.ndarray]:
    cols = {}
    for c in LLR_COLS:
        cols[f"minus_{c}"] = (-df[c]).to_numpy(float)
        cols[f"abs_{c}"] = df[c].abs().to_numpy(float)
    for c in JSD_COLS:
        cols[c] = df[c].to_numpy(float)
    return cols


def _auroc_auprc_bootstrap(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    auroc = float(roc_auc_score(y, s))
    auprc = float(average_precision_score(y, s))
    n = len(y)
    bo_roc, bo_prc = np.empty(N_BOOTSTRAP), np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.sum() in (0, len(yb)):
            bo_roc[b] = bo_prc[b] = np.nan
            continue
        bo_roc[b] = roc_auc_score(yb, s[idx])
        bo_prc[b] = average_precision_score(yb, s[idx])
    return {
        "auroc": auroc,
        "auroc_se": float(np.nanstd(bo_roc, ddof=1)),
        "auprc": auprc,
        "auprc_se": float(np.nanstd(bo_prc, ddof=1)),
        "n": n,
        "n_pos": int(y.sum()),
    }


def binary_menu(
    df: pd.DataFrame, label: np.ndarray, name: str, per_gene: bool
) -> pd.DataFrame:
    """AUROC/AUPRC for a 0/1 `label` (NaN = drop). If per_gene, macro-average."""
    mask = ~np.isnan(label)
    sub = df[mask].reset_index(drop=True)
    y_all = label[mask].astype(int)
    score_cols = _binary_score_cols(sub)
    rows = []
    for col, s_all in score_cols.items():
        if per_gene:
            per_roc, per_roc_se, per_prc, per_prc_se = [], [], [], []
            for g in sorted(sub["gene"].unique()):
                gi = (sub["gene"] == g).to_numpy()
                yg = y_all[gi]
                if len(yg) < N_MIN_BINARY or yg.sum() < 5 or yg.sum() > len(yg) - 5:
                    continue
                r = _auroc_auprc_bootstrap(yg, s_all[gi])
                per_roc.append(r["auroc"])
                per_roc_se.append(r["auroc_se"])
                per_prc.append(r["auprc"])
                per_prc_se.append(r["auprc_se"])
            if not per_roc:
                continue
            roc, roc_se, k = _macro(per_roc, per_roc_se)
            prc, prc_se, _ = _macro(per_prc, per_prc_se)
            rows.append(
                {
                    "target": name,
                    "grain": "macro_per_gene",
                    "score_type": col,
                    "auroc": roc,
                    "auroc_se": roc_se,
                    "auprc": prc,
                    "auprc_se": prc_se,
                    "n_genes": k,
                    "n_pos": int(y_all.sum()),
                }
            )
        else:
            if y_all.sum() < 5 or y_all.sum() > len(y_all) - 5:
                continue
            r = _auroc_auprc_bootstrap(y_all, s_all)
            rows.append(
                {
                    "target": name,
                    "grain": "pooled",
                    "score_type": col,
                    "auroc": r["auroc"],
                    "auroc_se": r["auroc_se"],
                    "auprc": r["auprc"],
                    "auprc_se": r["auprc_se"],
                    "n_genes": sub["gene"].nunique(),
                    "n_pos": r["n_pos"],
                }
            )
    return pd.DataFrame(rows).sort_values("auroc", ascending=False)


def calibrated_binary_label(df: pd.DataFrame, cal: pd.DataFrame) -> np.ndarray:
    """Per-gene abnormal(1)/normal(0)/NaN(drop) via ClinGen/ExCALIBR thresholds.

    For each gene pick the coarsest calibration that has BOTH a `normal` and an
    `abnormal` class (fewest class rows). A variant is abnormal if its
    `function_score` falls in an abnormal class range, normal if in a normal
    range, else dropped (intermediate / not_specified / no calibration).
    """

    def in_range(x, lo, hi, inc_lo, inc_hi):
        if pd.notna(lo):
            if (x < lo) or (x == lo and not inc_lo):
                return False
        if pd.notna(hi):
            if (x > hi) or (x == hi and not inc_hi):
                return False
        return True

    label = np.full(len(df), np.nan)
    fs = df["function_score"].to_numpy(float)
    for gene, gsub in df.groupby("gene"):
        gcal = cal[cal["gene"] == gene]
        if gcal.empty:
            print(f"  [T2] {gene}: no calibration — skipped")
            continue
        # Coarsest binary-capable scheme.
        best_title, best_n = None, None
        for title, tsub in gcal.groupby("calibration_title"):
            gos = set(tsub["go_classification"])
            if {"normal", "abnormal"}.issubset(gos):
                if best_n is None or len(tsub) < best_n:
                    best_title, best_n = title, len(tsub)
        if best_title is None:
            print(f"  [T2] {gene}: no normal+abnormal scheme — skipped")
            continue
        scheme = gcal[gcal["calibration_title"] == best_title]
        norm = scheme[scheme["go_classification"] == "normal"]
        abn = scheme[scheme["go_classification"] == "abnormal"]
        idx = np.where((df["gene"] == gene).to_numpy())[0]
        n_abn = n_norm = 0
        for i in idx:
            x = fs[i]
            is_abn = any(
                in_range(x, r.range_lower, r.range_upper, r.inclusive_lower, r.inclusive_upper)
                for r in abn.itertuples()
            )
            is_norm = any(
                in_range(x, r.range_lower, r.range_upper, r.inclusive_lower, r.inclusive_upper)
                for r in norm.itertuples()
            )
            if is_abn and not is_norm:
                label[i] = 1.0
                n_abn += 1
            elif is_norm and not is_abn:
                label[i] = 0.0
                n_norm += 1
        print(f"  [T2] {gene}: scheme={best_title!r} → {n_abn} abnormal / {n_norm} normal")
    return label


def main() -> None:
    model = DEFAULT_MODEL
    path = sys.argv[1] if len(sys.argv) > 1 else S3_SCORES.format(model=model)
    print(f"# SGE metric exploration — {model}\n# scores: {path}\n")
    df = load_scores(path)
    print(f"loaded {len(df)} variants, {df['gene'].nunique()} genes\n")

    print("=" * 70)
    print("T1 — per-gene correlation vs function_score (macro over genes)")
    print("=" * 70)
    corr = correlation_menu(df)
    print(corr.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 70)
    print("T1 — per-consequence_final macro Spearman (score_type=llr_avg)")
    print("=" * 70)
    pc = correlation_per_consequence(df, "llr_avg")
    print(pc.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 70)
    print("T2 — ClinGen-calibrated binary (predict abnormal), macro over genes")
    print("=" * 70)
    cal = pd.read_parquet(
        hf_hub_download(SGE_REPO, "calibrations.parquet", repo_type="dataset")
    )
    t2 = calibrated_binary_label(df, cal)
    cal_menu = binary_menu(df, t2, "calibrated_abnormal", per_gene=True)
    print(cal_menu.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 70)
    print("T3 — BRCA1 author_func_class (LOF vs FUNC, drop INT), pooled")
    print("=" * 70)
    brca1 = df[df["gene"] == "BRCA1"]
    fc = brca1["author_func_class"]
    t3 = np.full(len(brca1), np.nan)
    t3[(fc == "LOF").to_numpy()] = 1.0
    t3[(fc == "FUNC").to_numpy()] = 0.0
    t3_menu = binary_menu(brca1.reset_index(drop=True), t3, "brca1_func_class", per_gene=False)
    print(t3_menu.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 70)
    print("T4 (sanity) — BRCA1 ClinVar Pathogenic vs Benign, pooled")
    print("=" * 70)
    cv = brca1["author_clinvar_simple"].astype("string")
    t4 = np.full(len(brca1), np.nan)
    t4[cv.isin(["Pathogenic", "Likely pathogenic"]).to_numpy()] = 1.0
    t4[cv.isin(["Benign", "Likely benign"]).to_numpy()] = 0.0
    t4_menu = binary_menu(brca1.reset_index(drop=True), t4, "brca1_clinvar", per_gene=False)
    print(t4_menu.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    out = Path("scratch") / f"sge_metric_menu_{model}.csv"
    out.parent.mkdir(exist_ok=True)
    pd.concat(
        [corr.assign(table="T1_corr"), cal_menu.assign(table="T2_calibrated"),
         t3_menu.assign(table="T3_brca1_funcclass"), t4_menu.assign(table="T4_brca1_clinvar")],
        ignore_index=True,
    ).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
