"""SGE vs Mendelian model ranking, compared WITHIN each consequence class (#292).

Like-for-like: Mendelian per-subset AUPRC (minus_llr_avg) vs SGE within-group metrics
(Spearman per-gene macro + binary AUROC jsd, n>=30/label), separately for missense and
splicing — no cross-group macro. Removes the consequence-mix confound that made the
Mendelian *macro* ranking look reversed vs SGE: within missense and within splicing the
two benchmarks AGREE (both rank exp13-mp-1.7B top).

Run from the repo root:
    uv run --group genome-s3 python scripts/issue292_sge_vs_mendelian.py
"""
import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import polars as pl
from huggingface_hub import hf_hub_download
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

m = importlib.util.module_from_spec(
    s := importlib.util.spec_from_file_location("expl", str(Path(__file__).with_name("issue292_sge_metric_explore.py")))
)
s.loader.exec_module(m)
cal = pd.read_parquet(hf_hub_download("bolinas-dna/evals_sge", "calibrations.parquet", repo_type="dataset"))

SGE = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores/{}/sge.parquet"
MEN = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics/{}/mendelian_traits.parquet"
MODELS = {  # display -> evals_v2 model dir
    "exp13-mp-1.7B": "exp13-mixture-proportional-step-26000",
    "exp135-1B-m5.1": "mix-v0.9-p1B-i24-exp135-m5.1-step-59158",
    "exp166-1B": "exp166-v0.1-p1B-step-27329",
}
GROUPS = ["missense_variant", "splicing"]


def binary_in(df, lab, score, group):
    gm = (df.consequence_group == group).to_numpy()
    dfg, labg, sg = df[gm].reset_index(drop=True), lab[gm], score[gm]
    mask, aur = ~np.isnan(labg), []
    for g in sorted(dfg.gene.unique()):
        gi = (dfg.gene.to_numpy() == g) & mask
        yg = labg[gi].astype(int)
        if yg.sum() >= 30 and (len(yg) - yg.sum()) >= 30:
            aur.append(roc_auc_score(yg, sg[gi]))
    return (round(float(np.mean(aur)), 3), len(aur)) if aur else (np.nan, 0)


rows = []
for label, mdir in MODELS.items():
    df = m.load_scores(SGE.format(mdir))
    gs = m.gene_consequence_grid(df, "llr_avg", method="spearman")
    lab = m.calibrated_binary_label(df, cal)
    md = pl.read_parquet(MEN.format(mdir)).to_pandas()
    for grp in GROUPS:
        mend = md[(md.subset == grp) & (md.score_type == "minus_llr_avg")]["value"]
        auc_jsd, kb = binary_in(df, lab, df.jsd_avg.to_numpy(), grp)
        rows.append({
            "group": grp.replace("_variant", ""), "model": label,
            "mendel_AUPRC": round(float(mend.iloc[0]), 3) if len(mend) else None,
            "sge_spearman": round(gs.loc[grp, "macro"], 3),
            "sge_AUROC_jsd": auc_jsd, "sge_n_genes_bin": kb,
        })
t = pd.DataFrame(rows)
for grp in ["missense", "splicing"]:
    sub = t[t.group == grp]
    print(f"\n================= WITHIN {grp.upper()} =================")
    print(sub.drop(columns="group").to_string(index=False))
    for col in ["mendel_AUPRC", "sge_spearman", "sge_AUROC_jsd"]:
        print(f"  rank by {col:14s}: {' > '.join(sub.sort_values(col, ascending=False)['model'].tolist())}")
Path("scratch").mkdir(exist_ok=True)
t.to_csv("scratch/sge_vs_mendelian.csv", index=False)

# scatter: Mendelian AUPRC (x) vs SGE binary AUROC (y), per (model, group)
fig, ax = plt.subplots(figsize=(7, 6))
colors = {"exp13-mp-1.7B": "tab:green", "exp135-1B-m5.1": "tab:blue", "exp166-1B": "tab:orange"}
markers = {"missense": "o", "splicing": "s"}
for _, r in t.iterrows():
    ax.scatter(r.mendel_AUPRC, r.sge_AUROC_jsd, s=170, c=colors[r.model], marker=markers[r.group],
               edgecolor="k", zorder=3)
    ax.annotate(r.model.split("-")[0], (r.mendel_AUPRC, r.sge_AUROC_jsd), fontsize=8,
                xytext=(5, 4), textcoords="offset points")
ax.set_xlabel("Mendelian AUPRC (per-subset, minus_llr_avg)")
ax.set_ylabel("SGE binary AUROC (jsd, within group)")
ax.set_title("SGE vs Mendelian agree WITHIN consequence class\n(exp13-mp-1.7B tops both, on both missense & splicing)")
from matplotlib.lines import Line2D
leg1 = [Line2D([0], [0], marker=mk, color="w", markerfacecolor="gray", markeredgecolor="k", markersize=11, label=g)
        for g, mk in markers.items()]
leg2 = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="k", markersize=11, label=md)
        for md, c in colors.items()]
ax.legend(handles=leg1 + leg2, fontsize=8, loc="lower right")
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig("scratch/sge_vs_mendelian.png", dpi=150, bbox_inches="tight")
fig.savefig("scratch/sge_vs_mendelian.svg", bbox_inches="tight")
print("\nwrote scratch/sge_vs_mendelian.{png,svg}")
