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
# Min variants in a (gene x consequence_group) cell to compute a correlation (#292:
# agreed n>=100). Per-gene cells span a whole gene (thousands of variants), so this
# threshold only bites the per-(gene x group) cells in the grid / per-consequence tables.
N_MIN_CORR = 100
N_MIN_BINARY_PER_LABEL = 30  # min variants PER label (>=30 abnormal AND >=30 normal)
# for a gene (or pooled set) to contribute a binary metric — both classes need a
# non-degenerate sample, same n_min=30 convention as the correlation macro.

# Coarse consequence grouping — the SAME map the matched-pair datasets use
# (snakemake/evals/config/config.yaml `consequence_groups`), replicated here because
# evals_sge ships only `consequence_final`, no `consequence_group` (flagged in #297).
# Consequences absent from the map keep their own value (single-consequence category),
# matching trait_intervals.build_dataset's `.replace(...)` semantics. Only the subtypes
# that actually occur in SGE are listed; the rest of the config's `distal` members
# (other cCRE classes) never appear here.
CONSEQUENCE_GROUPS = {
    "splice_donor_5th_base_variant": "splicing",
    "splice_region_variant": "splicing",
    "splice_donor_region_variant": "splicing",
    "splice_polypyrimidine_tract_variant": "splicing",
    "exon_proximal": "splicing",
    "intron_variant": "distal",
    "intergenic_variant": "distal",
    "upstream_gene_variant": "distal",
    "downstream_gene_variant": "distal",
    "dELS_flank": "distal",
    "CA_flank": "distal",
}


def load_scores(path: str) -> pd.DataFrame:
    """Scores parquet → pandas, with the derived strand-averaged atoms + the coarse
    `consequence_group` (evals_sge ships only `consequence_final`; #297)."""
    df = pl.read_parquet(path).to_pandas()
    assert {"llr_fwd", "llr_rc", "jsd_fwd", "jsd_rc"}.issubset(df.columns), (
        f"missing per-strand atoms; have {sorted(df.columns)[:12]}…"
    )
    df["llr_avg"] = (df["llr_fwd"] + df["llr_rc"]) / 2
    df["jsd_avg"] = (df["jsd_fwd"] + df["jsd_rc"]) / 2
    df["consequence_group"] = df["consequence_final"].replace(CONSEQUENCE_GROUPS)
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
def correlation_menu(df: pd.DataFrame, sign_align: bool = True) -> pd.DataFrame:
    """Per-gene Spearman/Pearson(score, function_score) → macro + macro|.|.

    When ``sign_align`` (default), each gene's correlation is multiplied by its
    :func:`gene_signs` (+1/-1) before the macro so an inverted assay (DDX3X) reinforces
    rather than cancels. ``macro_abs`` (mean |corr|) is sign-invariant by construction.
    """
    rng = np.random.default_rng(SEED)
    genes = sorted(df["gene"].unique())
    signs = gene_signs(df) if sign_align else {g: 1.0 for g in genes}
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
                per_gene.append(signs[g] * v)
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


def correlation_per_consequence(
    df: pd.DataFrame,
    score_col: str = "llr_avg",
    method: str = "spearman",
    sign_align: bool = True,
) -> pd.DataFrame:
    """Per-(gene x consequence_group) correlation → macro within each group.

    Grouped on the coarse ``consequence_group`` (#292: splice classes + exon_proximal
    collapse to ``splicing``, cCRE flanks to ``distal``). Pooling raw function_score
    across genes is invalid (per-gene scales differ), so each group's number is a
    macro-average over per-(gene) correlations on cells with n >= ``N_MIN_CORR`` (100),
    sign-aligned per gene (DDX3X inverted) when ``sign_align``. ``method`` in
    {"spearman", "pearson"}: Spearman is rank-based and robust to the heavy-tailed
    per-gene function_score (e.g. RAD51C's -29 tail); Pearson is linear.
    """
    rng = np.random.default_rng(SEED)
    signs = gene_signs(df, score_col) if sign_align else {}
    rows = []
    for cons, csub in df.groupby("consequence_group"):
        per_gene, per_gene_se = [], []
        for g, gsub in csub.groupby("gene"):
            if len(gsub) < N_MIN_CORR:
                continue
            v, se = _correlation_with_bootstrap_se(
                gsub[score_col].to_numpy(float),
                gsub["function_score"].to_numpy(float),
                method=method,
                n_bootstrap=N_BOOTSTRAP,
                rng=rng,
            )
            per_gene.append(signs.get(g, 1.0) * v)
            per_gene_se.append(se)
        if not per_gene:
            continue
        macro, macro_se, k = _macro(per_gene, per_gene_se)
        rows.append(
            {
                "consequence_group": cons,
                "score_type": score_col,
                "method": method,
                "macro": macro,
                "macro_se": macro_se,
                "n_genes": k,
                "n_variants": len(csub),
            }
        )
    return pd.DataFrame(rows).sort_values("n_variants", ascending=False)


def gene_signs(df: pd.DataFrame, score_col: str = "llr_avg") -> dict[str, float]:
    """Per-gene +1/-1 alignment sign = sign of the gene's *overall*
    corr(score, function_score) (Pearson, all variants).

    Some SGE assays encode ``function_score`` in the opposite direction: DDX3X's
    0-1 NDD readout is high = abnormal, so it comes out -1, while the others are
    high = function-retained (+1). Sign-aligning per gene before macro-averaging
    stops an inverted assay from cancelling the rest. Robust because each gene has
    thousands of (missense-dominated) variants, so the overall sign is unambiguous.
    """
    signs: dict[str, float] = {}
    for g, sub in df.groupby("gene"):
        x, y = sub[score_col].to_numpy(float), sub["function_score"].to_numpy(float)
        r = pearsonr(x, y)[0] if x.std() > 0 and y.std() > 0 else 0.0
        signs[g] = -1.0 if r < 0 else 1.0
    return signs


def gene_consequence_grid(
    df: pd.DataFrame,
    score_col: str = "llr_avg",
    method: str = "spearman",
    n_min: int = N_MIN_CORR,
    sign_align: bool = True,
) -> pd.DataFrame:
    """2-way grid of per-(gene x consequence_group) correlation point estimates.

    Rows = consequence_group (#292 coarse grouping: splicing / distal / missense / …),
    columns = gene; each cell = corr(score, function_score) over that (gene, group)
    subset, or NaN if n < ``n_min`` (100; or a degenerate constant column). When
    ``sign_align`` (default), each gene's cells are multiplied by its :func:`gene_signs`
    (+1/-1) so an inverted assay (DDX3X) is oriented with the rest before the macro means
    are taken. A ``macro`` column holds the row mean over genes (per-group macro) and a
    ``macro`` row holds the column mean over groups (per-gene macro). Point estimates only
    — this is the overview grid; per-group bootstrap SEs are in
    :func:`correlation_per_consequence`.
    """
    corr_fn = spearmanr if method == "spearman" else pearsonr
    genes = sorted(df["gene"].unique())
    signs = gene_signs(df, score_col) if sign_align else {g: 1.0 for g in genes}
    conss = list(df["consequence_group"].value_counts().index)  # most-frequent first
    grid = pd.DataFrame(index=conss, columns=genes, dtype=float)
    for (cons, g), sub in df.groupby(["consequence_group", "gene"]):
        x, y = sub[score_col].to_numpy(float), sub["function_score"].to_numpy(float)
        if len(sub) >= n_min and x.std() > 0 and y.std() > 0:
            grid.loc[cons, g] = signs[g] * float(corr_fn(x, y)[0])
    grid["macro"] = grid[genes].mean(axis=1, skipna=True)  # per-group macro
    grid.loc["macro"] = grid.mean(axis=0, skipna=True)  # per-gene macro
    return grid


def group_trustworthiness(
    df: pd.DataFrame, cal: pd.DataFrame, score_col: str = "llr_avg"
) -> pd.DataFrame:
    """Per-consequence_group diagnostics for deciding which groups the SGE assay can
    actually be trusted to measure (#292).

    Three axes per group:
    - **coverage**: total n and the number of genes with an n>=100 cell.
    - **model sign-consistency**: ``frac_genes_pos`` = fraction of those qualifying
      genes whose sign-aligned per-gene correlation is positive (1.0 = the model tracks
      the assay the same way in every gene; low = scattered/unreliable).
    - **assay functional variation**: ``abnormal_rate`` = fraction of the
      calibration-labeled variants the assay calls abnormal (:func:`calibrated_binary_label`).
      Near-zero means the assay sees ~no function in that group — so a low correlation
      there is the assay's limitation, not the model's.

    ``macro_pearson`` (sign-aligned, the grid's per-group margin) is included for context.
    Coverage + abnormal_rate are model-independent; ``macro_pearson`` /
    ``frac_genes_pos`` depend on the scored model.
    """
    genes = sorted(df["gene"].unique())
    grid = gene_consequence_grid(df, score_col, method="pearson")
    abn = calibrated_binary_label(df, cal)
    work = df.assign(_abn=abn)
    rows = []
    for grp in [g for g in grid.index if g != "macro"]:
        cells = grid.loc[grp, genes].dropna()  # per-gene sign-aligned corr, n>=100 cells
        sub = work[work["consequence_group"] == grp]
        lab = sub["_abn"].dropna()
        rows.append(
            {
                "group": grp,
                "n": len(sub),
                "n_genes_ge100": int(len(cells)),
                "macro_pearson": float(grid.loc[grp, "macro"]),
                "frac_genes_pos": float((cells > 0).mean()) if len(cells) else float("nan"),
                "abnormal_rate": float(lab.mean()) if len(lab) else float("nan"),
                "n_labeled": int(len(lab)),
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False)


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
                n_pos = int(yg.sum())
                if n_pos < N_MIN_BINARY_PER_LABEL or (len(yg) - n_pos) < N_MIN_BINARY_PER_LABEL:
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
            n_pos_all = int(y_all.sum())
            if n_pos_all < N_MIN_BINARY_PER_LABEL or (len(y_all) - n_pos_all) < N_MIN_BINARY_PER_LABEL:
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

    For each gene pick a **numeric-threshold** scheme: its `normal` AND `abnormal`
    classes must each have >=1 row with a finite `function_score` range bound.
    Categorical schemes (e.g. "Investigator-provided functional classes" for some
    genes, IGVF control sets) carry `[nan, nan]` ranges — they can't threshold
    function_score, so a gene with only categorical schemes (DDX3X) is skipped here
    (its abnormal/normal call lives in an `author_` class column instead). Among
    numeric schemes prefer the most granular (most finite-bound rows ≈ ExCALIBR),
    tie-break by title for determinism. A variant is abnormal if its function_score
    falls in any abnormal range, normal if in any normal range, else dropped (the
    intermediate gap / no calibration).
    """

    def in_range(x, lo, hi, inc_lo, inc_hi):
        if pd.notna(lo):
            if (x < lo) or (x == lo and not inc_lo):
                return False
        if pd.notna(hi):
            if (x > hi) or (x == hi and not inc_hi):
                return False
        return True

    def numeric_rows(t, go):
        r = t[t["go_classification"] == go]
        return r[r["range_lower"].notna() | r["range_upper"].notna()]

    label = np.full(len(df), np.nan)
    fs = df["function_score"].to_numpy(float)
    for gene, gsub in df.groupby("gene"):
        gcal = cal[cal["gene"] == gene]
        if gcal.empty:
            print(f"  [T2] {gene}: no calibration — skipped")
            continue
        # Most-granular numeric-threshold scheme (finite bounds on both classes).
        best_title, best_n = None, -1
        for title, tsub in sorted(gcal.groupby("calibration_title"), key=lambda kv: kv[0]):
            n_norm, n_abn = len(numeric_rows(tsub, "normal")), len(numeric_rows(tsub, "abnormal"))
            if n_norm and n_abn and (n_norm + n_abn) > best_n:
                best_title, best_n = title, n_norm + n_abn
        if best_title is None:
            print(f"  [T2] {gene}: only categorical (no numeric thresholds) — skipped")
            continue
        scheme = gcal[gcal["calibration_title"] == best_title]
        norm = numeric_rows(scheme, "normal")
        abn = numeric_rows(scheme, "abnormal")
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
    print("T1 — per-consequence_group macro correlation (sign-aligned, n>=100, llr_avg)")
    print("=" * 70)
    for _meth in ("spearman", "pearson"):
        pc = correlation_per_consequence(df, "llr_avg", method=_meth)
        print(f"\n[{_meth}]")
        print(pc.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 70)
    print("T1 — gene x consequence_group grid (sign-aligned; cell = corr(llr_avg, function_score))")
    print("=" * 70)
    grids = {}
    for _meth in ("pearson", "spearman"):
        grid = gene_consequence_grid(df, "llr_avg", method=_meth)
        grids[_meth] = grid
        print(f"\n[{_meth}]  blank = n<{N_MIN_CORR}")
        print(grid.to_string(float_format=lambda x: f"{x:.2f}", na_rep="·"))
        grid.to_csv(Path("scratch") / f"sge_grid_{_meth}_{model}.csv")

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
