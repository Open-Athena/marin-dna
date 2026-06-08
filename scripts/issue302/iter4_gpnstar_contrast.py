"""issue #302 — iteration 4: is GPN-Star fooled by the SAME conserved-benign FP
set our 4B is? (GPN-Star is itself conservation-driven — #203 found it even more
phyloP-correlated than Evo 2 — so this isolates whether conservation-reliance is
the failure, or the missing human-pathogenicity refinement is.)

Takes our 4B-defined missense FP set (top-10% negatives by score; the
"conserved-looking benigns" characterized in iter3) and asks what GPN-Star scores
them. GPN-Star-V/M/P calibrated predictions from the #203 gist (cLLR =
-llr_calibrated; higher = more deleterious), joined to our enriched missense
table on (chrom,pos,ref,alt).

Outputs (scratch/issue302/figs/, PNG+SVG):
  gpn_score_on_our_groups   GPN-V cLLR distribution on pos / our-FP / typ / easy neg
  gpn_vs_4b_scatter         per-variant 4B vs GPN-V (missense), pos vs neg + FP region
  gpn_auprc_by_gene_age     missense AUPRC by gene-age bucket: 4B vs GPN-V (does GPN hold in old genes?)
Prints: GPN score per group; frac of our FPs GPN also flags; FP-set overlap;
overall + restricted (pos + our-FP-negs) AUPRC for 4B vs GPN-V/M/P.

Run:  uv run python scripts/issue302/iter4_gpnstar_contrast.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score

ENRICHED = Path("scratch/issue302/missense_enriched.parquet")  # from iter3
OUT = Path("scratch/issue302/figs")
KEY = ["chrom", "pos", "ref", "alt"]
GPN_GISTS = {
    v: f"https://gist.githubusercontent.com/gonzalobenegas/db282f89aa00244fbb7437dce0f069ef/raw/02484d50d9bfd80337e313652b26f98a9362b6b1/bolinas_mendelian_traits_GPN-Star-{v}.parquet"
    for v in ("V", "M", "P")
}
AGE_ORDER = [
    "Vertebrata",
    "Eumetazoa",
    "Opisthokonta",
    "Eukaryota",
    "Euk+Bac",
    "Cellular_organisms",
]
AGE_MYA = {
    "Vertebrata": 615,
    "Eumetazoa": 824,
    "Opisthokonta": 1105,
    "Eukaryota": 1962,
    "Euk+Bac": 3500,
    "Cellular_organisms": 4290,
}


def load() -> pd.DataFrame:
    w = pl.read_parquet(ENRICHED).with_columns(pl.col("chrom").cast(str))
    for v, url in GPN_GISTS.items():
        g = (
            pl.read_parquet(url)
            .filter(pl.col("split") == "train")
            .with_columns(
                [
                    pl.col("chrom").cast(str),
                    (-pl.col("llr_calibrated")).alias(f"gpn_{v}"),
                ]
            )
            .select([*KEY, f"gpn_{v}"])
        )
        w = w.join(g, on=KEY, how="left")
    return w.to_pandas()


def groups(jp: pd.DataFrame) -> dict[str, pd.DataFrame]:
    neg = jp[jp.label == 0]
    q90, q10 = neg["4B"].quantile(0.90), neg["4B"].quantile(0.10)
    return {
        "pathogenic (pos)": jp[jp.label == 1],
        "OUR FP (top-10% neg @4B)": neg[neg["4B"] >= q90],
        "typical neg": neg[(neg["4B"] > q10) & (neg["4B"] < q90)],
        "easy neg (bot-10%)": neg[neg["4B"] <= q10],
    }


def analyse(jp: pd.DataFrame) -> dict:
    g = groups(jp)
    pos, fp = g["pathogenic (pos)"], g["OUR FP (top-10% neg @4B)"]
    print("\n=== GPN-Star-V cLLR on OUR groups (median / mean) ===")
    for nm, s in g.items():
        print(
            f"  {nm:26} median={s['gpn_V'].median():+6.2f}  mean={s['gpn_V'].mean():+6.2f}"
        )
    p50, p90 = pos["gpn_V"].median(), pos["gpn_V"].quantile(0.90)
    print(f"\n  GPN positive p50={p50:+.2f} p90={p90:+.2f}")
    print(
        f"  frac of OUR FP set GPN also scores >= GPN-pos-p50: {(fp['gpn_V'] >= p50).mean():.3f}"
    )
    print(
        f"  frac of OUR FP set GPN also scores >= GPN-pos-p90: {(fp['gpn_V'] >= p90).mean():.3f}"
    )

    def top(df, col, n=100):
        return set(map(tuple, df.nlargest(n, col)[KEY].values))

    neg = jp[jp.label == 0]
    o4b, ogpn = top(neg, "4B"), top(neg, "gpn_V")
    print(
        f"\n  top-100 FP overlap OUR-4B vs GPN-V: |∩|={len(o4b & ogpn)} Jaccard={len(o4b & ogpn) / len(o4b | ogpn):.3f}"
    )

    print("\n=== AUPRC: overall and restricted to (positives + OUR-FP negatives) ===")
    fp_idx = set(fp.index)
    rsub = jp[(jp.label == 1) | (jp.index.isin(fp_idx))]
    print(f"  restricted-set prevalence = {rsub.label.mean():.3f} (chance baseline)")
    for col, name in [
        ("4B", "our 4B"),
        ("gpn_V", "GPN-Star-V"),
        ("gpn_M", "GPN-Star-M"),
        ("gpn_P", "GPN-Star-P"),
    ]:
        ov = average_precision_score(jp.label, jp[col])
        rs = average_precision_score(rsub.label, rsub[col])
        print(f"  {name:12} overall AUPRC={ov:.3f}   restricted AUPRC={rs:.3f}")
    return g


def auprc_by_age(jp: pd.DataFrame) -> pd.DataFrame:
    wa = jp[jp["age_mya"].notna()]
    rows = []
    for b in AGE_ORDER:
        sub = wa[wa["modeAge"] == b]
        if len(sub) < 20 or sub.label.sum() < 3:
            continue
        rows.append(
            {
                "bucket": b,
                "mya": AGE_MYA[b],
                "n": len(sub),
                "4B": average_precision_score(sub.label, sub["4B"]),
                "gpn_V": average_precision_score(sub.label, sub["gpn_V"]),
            }
        )
    df = pd.DataFrame(rows)
    print("\n=== missense AUPRC by gene-age: our 4B vs GPN-V ===")
    for r in df.itertuples():
        print(
            f"  {r.bucket:>20} (~{r.mya:>4}) n={r.n:<5} 4B={getattr(r, '_4'):.2f}  GPN-V={r.gpn_V:.2f}"
        )
    return df


# --------------------------------------------------------------------------- #
def fig_group_box(g: dict[str, pd.DataFrame]) -> None:
    names = list(g.keys())
    colors = ["tab:red", "tab:orange", "tab:gray", "tab:blue"]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    data = [g[n]["gpn_V"].dropna().values for n in names]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    p50 = g["pathogenic (pos)"]["gpn_V"].median()
    ax.axhline(
        p50, color="tab:red", ls=":", lw=1, label=f"GPN pathogenic median = {p50:.1f}"
    )
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
    ax.set_ylabel("GPN-Star-V cLLR (higher = more pathogenic)")
    ax.set_title(
        "Does GPN-Star over-call OUR FP set?\nIt sees the conservation (FP > typical neg) but keeps them below pathogenic"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "gpn_score_on_our_groups")


def fig_scatter(jp: pd.DataFrame) -> None:
    neg = jp[jp.label == 0]
    pos = jp[jp.label == 1]
    q90 = neg["4B"].quantile(0.90)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.scatter(
        neg["4B"], neg["gpn_V"], s=7, alpha=0.35, color="tab:gray", label="benign (neg)"
    )
    ax.scatter(
        pos["4B"],
        pos["gpn_V"],
        s=10,
        alpha=0.6,
        color="tab:red",
        label="pathogenic (pos)",
    )
    ax.axvline(
        q90, color="tab:orange", ls="--", lw=1, label="our 4B FP threshold (neg p90)"
    )
    ax.axhline(
        pos["gpn_V"].median(),
        color="tab:red",
        ls=":",
        lw=1,
        label="GPN pathogenic median",
    )
    ax.set_xlabel("our 4B  minus_llr  (high = our model calls pathogenic)")
    ax.set_ylabel("GPN-Star-V  cLLR")
    ax.set_title(
        "4B vs GPN-Star on missense — our FPs (right of orange) sit\nLOW on GPN (below red line): GPN isn't fooled by them"
    )
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)
    _save(fig, "gpn_vs_4b_scatter")


def fig_age(df: pd.DataFrame) -> None:
    df = df.sort_values("mya")
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, df["4B"], "o-", color="tab:blue", label="our 4B")
    ax.plot(x, df["gpn_V"], "s-", color="tab:green", label="GPN-Star-V")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{b}\n(~{m})" for b, m in zip(df["bucket"], df["mya"])], fontsize=7
    )
    ax.set_xlabel("gene age bucket (older →)")
    ax.set_ylabel("missense AUPRC")
    ax.set_title("GPN-Star holds in old genes where our 4B drops")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "gpn_auprc_by_gene_age")


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / name}.{{png,svg}}")


def main() -> None:
    jp = load()
    print(
        f"joined missense rows: {len(jp)} (GPN-V non-null: {jp['gpn_V'].notna().sum()})"
    )
    g = analyse(jp)
    age = auprc_by_age(jp)
    print("\nFigures:")
    fig_group_box(g)
    fig_scatter(jp)
    fig_age(age)
    print("\nDone.")


if __name__ == "__main__":
    main()
