"""exp326 curated-enhancer trajectories (#326) — LL + mendelian AUPRC vs step.

Figure-level seaborn (relplot). Two 2-panel figures:
  fig1_ll_val_enhancer         LL_functional | LL gap  (val_enhancer probe), vs step.
  fig2_auprc_distal_splicing   distal (P2) | splicing (P1) mendelian AUPRC, vs step.

exp326 A/B mendelian AUPRC = in-training lm_eval; exp232 baseline = evals_v2 per-step
from S3. (Methodology stays in the issue text, not the figures.)

Run:  uv run python plots/exp326_curated_enhancers.py
Outputs: plots/output/exp326_curated_enhancers/{fig1_ll_val_enhancer,fig2_auprc_distal_splicing}.{svg,png}
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import seaborn as sns
import wandb

sns.set_theme(style="whitegrid", context="notebook")

PROJ = "gonzalobenegas/marin"
OUT = Path(__file__).parent / "output" / "exp326_curated_enhancers"
OUT.mkdir(parents=True, exist_ok=True)

# (arm label, wandb group, run-name substring)
ARMS = [
    ("exp232 ccre (baseline)", "dna-exp232-v0.1", "-v4_ccre_non_promoter-v0.1"),
    ("A · v4_ccre_noexon", "dna-exp326-v0.1", "-v4_ccre_noexon-v0.1"),
    ("B · v4_ccre_noexon_enhancer", "dna-exp326-v0.1", "-v4_ccre_noexon_enhancer-v0.1"),
]
ORDER = [a[0] for a in ARMS]
PALETTE = dict(zip(ORDER, ["#8c8c8c", "#1f77b4", "#ff7f0e"]))
BASELINE_OFFLINE_STEPS = [500, 1000, 1500, 2000, 2500, 3000, 4000, 4500, 4999]
S3 = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics"

api = wandb.Api(timeout=180)


def get_run(group: str, sub: str):
    runs = [r for r in api.runs(PROJ, filters={"group": group}) if sub in r.name]
    runs.sort(key=lambda r: (r.state == "finished", r.created_at), reverse=True)
    return runs[0]


def wandb_hist(run, key: str) -> dict[int, float]:
    h = run.history(keys=[key], samples=10000, pandas=True)
    if key not in h or "_step" not in h:
        return {}
    h = h[["_step", key]].dropna()
    return {int(s): float(v) for s, v in zip(h["_step"], h[key])}


runs = {label: get_run(grp, sub) for label, grp, sub in ARMS}
print("runs:", {k: v.name for k, v in runs.items()})

# ---- Fig 1 frame: val_enhancer LL_functional (= -loss) and LL gap (nonfunc - func) ----
ll_rows = []
for label, *_ in ARMS:
    r = runs[label]
    f = wandb_hist(r, "eval/val_enhancer_functional/loss")
    n = wandb_hist(r, "eval/val_enhancer_nonfunctional/loss")
    for step in sorted(set(f) & set(n)):
        ll_rows.append({"step": step, "value": -f[step], "arm": label, "metric": "LL_functional"})
        ll_rows.append({"step": step, "value": n[step] - f[step], "arm": label, "metric": "LL gap"})
ll_df = pd.DataFrame(ll_rows)

# ---- Fig 2 frame: distal + splicing AUPRC ----
au_rows = []
for subset in ("distal", "splicing"):
    for step in BASELINE_OFFLINE_STEPS:
        df = pl.read_parquet(
            f"{S3}/exp232-v4_ccre_non_promoter-step-{step}/mendelian_traits.parquet"
        ).filter((pl.col("score_type") == "minus_llr_avg") & (pl.col("subset") == subset))
        if len(df):
            au_rows.append({"step": step, "AUPRC": float(df["value"][0]), "arm": ORDER[0], "subset": subset})
    for label, *_ in ARMS[1:]:
        for step, v in wandb_hist(runs[label], f"lm_eval/mendelian_traits_255/{subset}/avg/auprc").items():
            au_rows.append({"step": step, "AUPRC": v, "arm": label, "subset": subset})
au_df = pd.DataFrame(au_rows)

# ---- Fig 1: LL_functional | LL gap ----
g1 = sns.relplot(
    data=ll_df, x="step", y="value", hue="arm", hue_order=ORDER, palette=PALETTE,
    col="metric", col_order=["LL_functional", "LL gap"], kind="line", marker="o",
    height=4.4, aspect=1.15, facet_kws={"sharey": False},
)
g1.set_titles("{col_name}")
g1.set_axis_labels("training step", "")
g1.axes_dict["LL_functional"].set_ylabel("val_enhancer  LL$_{func}$  (= −loss)")
g1.axes_dict["LL gap"].set_ylabel("val_enhancer  LL gap  (LL$_{func}$ − LL$_{nonfunc}$)")
g1.axes_dict["LL gap"].axhline(0, color="k", lw=0.8, alpha=0.3)
g1.legend.set_title("")
g1.figure.suptitle("exp326 · val_enhancer constraint LL vs step", y=1.03)
g1.savefig(OUT / "fig1_ll_val_enhancer.svg")
g1.savefig(OUT / "fig1_ll_val_enhancer.png", dpi=130)
print("  wrote fig1_ll_val_enhancer.{svg,png}")

# ---- Fig 2: distal | splicing AUPRC ----
g2 = sns.relplot(
    data=au_df, x="step", y="AUPRC", hue="arm", hue_order=ORDER, palette=PALETTE,
    col="subset", col_order=["distal", "splicing"], kind="line", marker="o",
    height=4.4, aspect=1.1, facet_kws={"sharey": False},
)
for _subset, ax in g2.axes_dict.items():
    ax.axhline(0.10, color="k", lw=0.8, ls=":", alpha=0.4)
g2.set_axis_labels("training step", "mendelian AUPRC")
g2.set_titles("{col_name}")
g2.legend.set_title("")
g2.figure.suptitle(
    "exp326 · mendelian AUPRC vs step  (distal = P2 target, splicing = P1 off-diagonal leak)", y=1.03
)
g2.savefig(OUT / "fig2_auprc_distal_splicing.svg")
g2.savefig(OUT / "fig2_auprc_distal_splicing.png", dpi=130)
print("  wrote fig2_auprc_distal_splicing.{svg,png}")
print("done ->", OUT)
