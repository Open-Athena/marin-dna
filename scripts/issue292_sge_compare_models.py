"""3-model SGE comparison (#292): correlation (sign-aligned, per-consequence_group,
n>=100) + ClinGen-calibrated binary AUROC (n>=30/label), restricted to the trustworthy
groups (missense + splicing). Point estimates — fast model ranking; a paired bootstrap
would firm up the deltas. Reuses the metric helpers from issue292_sge_metric_explore.py.

Run from the repo root:
    uv run --group genome-s3 python scripts/issue292_sge_compare_models.py
Writes scratch/sge_compare_{corr,binary}.csv + scratch/sge_model_compare.{png,svg}.
"""
import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "expl", str(Path(__file__).with_name("issue292_sge_metric_explore.py"))
)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

S3 = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores/{}/sge.parquet"
MODELS = {  # display -> evals_v2 model dir
    "exp13-mp-1.7B": S3.format("exp13-mixture-proportional-step-26000"),
    "exp135-1B-m5.1": S3.format("mix-v0.9-p1B-i24-exp135-m5.1-step-59158"),
    "exp166-1B": S3.format("exp166-v0.1-p1B-step-27329"),
}
cal = pd.read_parquet(hf_hub_download("bolinas-dna/evals_sge", "calibrations.parquet", repo_type="dataset"))
GROUPS_ALL = ["missense_variant", "splicing", "non_coding_transcript_exon_variant",
              "synonymous_variant", "tss_proximal", "5_prime_UTR_variant", "3_prime_UTR_variant"]


def overall_corr(df, signs, method):
    fn = spearmanr if method == "spearman" else pearsonr
    vals = [signs[g] * fn(s.llr_avg, s.function_score)[0]
            for g, s in df.groupby("gene") if len(s) >= 100 and s.llr_avg.std() > 0]
    return float(np.mean(vals))


def binary_macro(df, label, score):
    mask = ~np.isnan(label)
    aur = []
    for g in sorted(df.gene.unique()):
        gi = (df.gene.to_numpy() == g) & mask
        yg = label[gi].astype(int)
        if yg.sum() >= 30 and (len(yg) - yg.sum()) >= 30:  # n>=30 per label
            aur.append(roc_auc_score(yg, score[gi]))
    return float(np.mean(aur)), len(aur)


def binary_macro_in_group(df, label, score, group):
    """Per-gene binary AUROC restricted to one consequence_group → macro (n>=30/label)."""
    gmask = (df.consequence_group == group).to_numpy()
    return binary_macro(df[gmask].reset_index(drop=True), label[gmask], score[gmask])


TRUST_GROUPS = ["missense_variant", "splicing"]
corr_rows, bin_rows, grid_rows, by_group_rows = [], [], [], []
for label, path in MODELS.items():
    df = m.load_scores(path)
    signs = m.gene_signs(df)
    gp = m.gene_consequence_grid(df, "llr_avg", method="pearson")
    gs = m.gene_consequence_grid(df, "llr_avg", method="spearman")
    row = {"model": label}
    for meth, g in (("pearson", gp), ("spearman", gs)):
        row[f"overall_{meth}"] = round(overall_corr(df, signs, meth), 3)
        row[f"missense_{meth}"] = round(g.loc["missense_variant", "macro"], 3)
        row[f"splicing_{meth}"] = round(g.loc["splicing", "macro"], 3)
        row[f"trust(mis+spl)_{meth}"] = round(np.mean([g.loc["missense_variant", "macro"], g.loc["splicing", "macro"]]), 3)
    corr_rows.append(row)
    lab = m.calibrated_binary_label(df, cal)
    for sc in ("minus_llr_avg", "jsd_avg"):
        score = (-df.llr_avg).to_numpy() if sc == "minus_llr_avg" else df.jsd_avg.to_numpy()
        auroc, k = binary_macro(df, lab, score)
        bin_rows.append({"model": label, "score": sc, "macro_AUROC": round(auroc, 3), "n_genes": k})
    grid_rows.append({"model": label, **{grp: round(gp.loc[grp, "macro"], 3) if grp in gp.index else None for grp in GROUPS_ALL}})
    # binary AUROC vs correlation, split BY trustworthy group (per gene -> macro)
    for grp in TRUST_GROUPS:
        auc_mll, kb = binary_macro_in_group(df, lab, (-df.llr_avg).to_numpy(), grp)
        auc_jsd, _ = binary_macro_in_group(df, lab, df.jsd_avg.to_numpy(), grp)
        by_group_rows.append({
            "model": label, "group": grp.replace("_variant", ""),
            "spearman": round(gs.loc[grp, "macro"], 3), "pearson": round(gp.loc[grp, "macro"], 3),
            "n_genes_corr": int(gp.loc[grp, sorted(df.gene.unique())].notna().sum()),
            "AUROC_minusllr": None if np.isnan(auc_mll) else round(auc_mll, 3),
            "AUROC_jsd": None if np.isnan(auc_jsd) else round(auc_jsd, 3),
            "n_genes_bin": kb,
        })

corr_df, bin_df = pd.DataFrame(corr_rows), pd.DataFrame(bin_rows)
by_group_df = pd.DataFrame(by_group_rows)
grid_df = pd.DataFrame(grid_rows).set_index("model")
print("=== correlation (sign-aligned, n>=100, macro over 9 genes) ===")
print(corr_df.to_string(index=False))
print("\n=== binary AUROC (ClinGen-calibrated abnormal; 7 genes; n>=30/label) ===")
print(bin_df.to_string(index=False))
print("\n=== per-consequence_group macro Pearson ===")
print(grid_df.to_string())
print("\n=== binary AUROC vs correlation, split by trustworthy group (per gene -> macro) ===")
print(by_group_df.to_string(index=False))
Path("scratch").mkdir(exist_ok=True)
corr_df.to_csv("scratch/sge_compare_corr.csv", index=False)
bin_df.to_csv("scratch/sge_compare_binary.csv", index=False)
by_group_df.to_csv("scratch/sge_compare_by_group.csv", index=False)

models = list(MODELS)
fig, (axc, axb) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
x, w = np.arange(len(models)), 0.2
for i, (col, lab) in enumerate([("overall_pearson", "overall"), ("missense_pearson", "missense"),
                                ("splicing_pearson", "splicing"), ("trust(mis+spl)_pearson", "missense+splicing")]):
    axc.bar(x + (i - 1.5) * w, [corr_df.loc[corr_df.model == md, col].iloc[0] for md in models], w, label=lab)
axc.set_xticks(x, models, rotation=12, fontsize=9); axc.set_ylabel("macro Pearson (sign-aligned, n>=100)")
axc.set_title("Correlation vs function_score"); axc.legend(fontsize=8); axc.grid(axis="y", alpha=.3)
for i, sc in enumerate(["minus_llr_avg", "jsd_avg"]):
    axb.bar(x + (i - 0.5) * w * 1.5, [bin_df[(bin_df.model == md) & (bin_df.score == sc)]["macro_AUROC"].iloc[0] for md in models], w * 1.5, label=sc)
axb.set_xticks(x, models, rotation=12, fontsize=9); axb.set_ylabel("macro AUROC (7 calibrated genes, n>=30/label)")
axb.set_ylim(0.5, 0.9); axb.set_title("Binary: predict ClinGen-calibrated abnormal"); axb.legend(fontsize=8); axb.grid(axis="y", alpha=.3)
fig.suptitle("SGE: 3-model comparison (exp135-1B-m5.1 · exp166-1B · exp13-mixture-proportional-1.7B)", fontsize=12)
fig.savefig("scratch/sge_model_compare.png", dpi=150, bbox_inches="tight")
fig.savefig("scratch/sge_model_compare.svg", bbox_inches="tight")
print("\nwrote scratch/sge_model_compare.{png,svg}")
