"""exp353 (#353) — CDS annotation-vs-projection mendelian AUPRC trajectories vs step.

Pulls the online ``mendelian_traits_255`` AUPRC (per consequence, FWD/RC-averaged)
logged every 500 steps for the four Qwen3-0.25B arms (wandb group
``dna-exp353-v0.1``) and draws AUPRC-vs-step line plots faceted by consequence:
colour = method (annotation / projection), line style = species scope
(all-204 / vertebrates-125). Emits SVG (for the issue embed) + PNG.

    uv run --with seaborn --with pyarrow python plots/exp353_cds_proj_vs_annot.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import wandb  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "output", "exp353_cds_proj_vs_annot")
os.makedirs(OUT, exist_ok=True)

# arm -> (method, scope). SWEEP is 2 methods x 2 species scopes.
ARMS = {
    "annot_all204": ("annotation", "all-204"),
    "proj_all204": ("projection", "all-204"),
    "annot_vert125": ("annotation", "vertebrates-125"),
    "proj_vert125": ("projection", "vertebrates-125"),
}
# consequence label -> wandb sub-key
CONS = [
    ("missense", "missense_variant"),
    ("synonymous", "synonymous_variant"),
    ("splicing", "splicing"),
    ("global (all)", "_global_"),
]
PREFIX = "lm_eval/mendelian_traits_255/"


def _load() -> pd.DataFrame:
    api = wandb.Api()
    runs = {
        r.name.replace("dna-exp353-cds-0p25b-", "").replace("-v0.1", ""): r
        for r in api.runs("gonzalobenegas/marin", filters={"group": "dna-exp353-v0.1"})
    }
    rows = []
    for arm, (method, scope) in ARMS.items():
        r = runs[arm]
        keys = ["_step"] + [PREFIX + k + "/avg/auprc" for _, k in CONS]
        hist = r.history(keys=keys, pandas=True)
        for _, row in hist.iterrows():
            for label, k in CONS:
                v = row.get(PREFIX + k + "/avg/auprc")
                if pd.notna(v):
                    rows.append(
                        {
                            "step": int(row["_step"]),
                            "AUPRC": float(v),
                            "method": method,
                            "scope": scope,
                            "consequence": label,
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(OUT, "metrics.parquet"))
    return df


# CDS self-supervised LL panels. Convention (per #8 / exp232 / exp351): LL = -loss,
# read from the per-region val *loss* keys (NOT bpb). LL-functional = -loss_functional
# (how well the model models conserved CDS positions); LL-gap = LL_functional -
# LL_nonfunctional = loss_nonfunctional - loss_functional (conserved-vs-nonconserved
# separation). val_cds is the same held-out probe for every arm.
LL_REGION = "val_cds"


def _load_ll() -> pd.DataFrame:
    api = wandb.Api()
    runs = {
        r.name.replace("dna-exp353-cds-0p25b-", "").replace("-v0.1", ""): r
        for r in api.runs("gonzalobenegas/marin", filters={"group": "dna-exp353-v0.1"})
    }
    fk = f"eval/{LL_REGION}_functional/loss"
    nk = f"eval/{LL_REGION}_nonfunctional/loss"
    rows = []
    for arm, (method, scope) in ARMS.items():
        r = runs[arm]
        hist = r.history(keys=["_step", fk, nk], pandas=True)
        for _, row in hist.iterrows():
            lf, ln = row.get(fk), row.get(nk)
            if pd.notna(lf) and pd.notna(ln):
                step = int(row["_step"])
                common = {"step": step, "method": method, "scope": scope}
                rows.append({**common, "value": -lf, "metric": "LL-functional"})
                rows.append({**common, "value": (-lf) - (-ln), "metric": "LL-gap"})
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(OUT, "metrics_ll.parquet"))
    return df


def main() -> None:
    df = _load()
    sns.set_theme(style="whitegrid", context="talk")
    g = sns.relplot(
        data=df,
        kind="line",
        x="step",
        y="AUPRC",
        hue="method",
        style="scope",
        col="consequence",
        col_wrap=2,
        marker="o",
        markersize=6,
        height=3.7,
        aspect=1.35,
        palette={"annotation": "#4C72B0", "projection": "#DD8452"},
        facet_kws={"sharey": False},
    )
    g.set_axis_labels("training step", "AUPRC")
    g.set_titles("{col_name}")
    g.figure.suptitle(
        "exp353 — CDS annotation vs projection: mendelian AUPRC (FWD/RC-avg) vs "
        "step (Qwen3-0.25B, 5K x 8192)",
        y=1.02,
        fontsize=14,
    )
    g.figure.subplots_adjust(left=0.09, wspace=0.18, hspace=0.28)
    for ext in ("svg", "png"):
        g.figure.savefig(os.path.join(OUT, f"figure.{ext}"), bbox_inches="tight", dpi=140)
    plt.close(g.figure)

    # Second figure — CDS self-supervised LL (functional level + gap) vs step.
    df_ll = _load_ll()
    gll = sns.relplot(
        data=df_ll,
        kind="line",
        x="step",
        y="value",
        hue="method",
        style="scope",
        col="metric",
        col_order=["LL-functional", "LL-gap"],
        marker="o",
        markersize=6,
        height=3.9,
        aspect=1.3,
        palette={"annotation": "#4C72B0", "projection": "#DD8452"},
        facet_kws={"sharey": False},
    )
    gll.set_axis_labels("training step", "LL  (= −loss, nats)")
    gll.set_titles("CDS {col_name}")
    gll.figure.suptitle(
        "exp353 — CDS self-supervised LL: functional level + conserved-vs-nonconserved "
        "gap (LL = −loss)",
        y=1.03,
        fontsize=13,
    )
    gll.figure.subplots_adjust(top=0.85, left=0.08, wspace=0.2)
    for ext in ("svg", "png"):
        gll.figure.savefig(os.path.join(OUT, f"figure_ll.{ext}"), bbox_inches="tight", dpi=140)
    plt.close(gll.figure)
    print(f"wrote {OUT}  (auprc_rows={len(df)}, ll_rows={len(df_ll)})")


if __name__ == "__main__":
    main()
