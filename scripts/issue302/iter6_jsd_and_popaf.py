"""issue #302 — iteration 6: (b) jsd_avg protocol robustness + population-AF probe.

PART B (b) — protocol robustness. Re-run the iter1 pathogenic-vs-benign separation
on `jsd_avg` (= mean of jsd_fwd/jsd_rc; the next-token Jensen-Shannon score)
instead of minus_llr_avg, across the ladder. If missense still compresses while
splicing expands, the effect is not an LLR-scoring artifact.

PART C — does the FP set look human-population-tolerated? phyloP measures
cross-species (long-term) constraint; a site can be conserved across mammals yet
tolerated specifically in humans (relaxed/changed constraint or drift). The
cheap human-tolerance proxy is gnomAD PER-POPULATION AF: popmax (max ancestry-
group AF; the clinical "too common to be pathogenic" statistic) and lopsidedness
(popmax / global). Pulls gnomAD exome+genome per-pop AF (myvariant.info, hg38),
computes popmax over the major ancestry groups, and characterizes the FP set vs
typical-neg vs pathogenic. Caveat: lopsidedness can be drift OR selection —
distinguishing them needs Fst/iHS/SDS (out of scope); either way it marks
human-specific tolerance the cross-species model cannot see.

Inputs: scratch/issue302/missense_enriched.parquet (iter3) for groups+global AF;
ladder scores on S3 for jsd. External pulls cached.
Outputs (scratch/issue302/figs/): jsd_separation_vs_scale, popaf_characterization.

Run:  uv run python scripts/issue302/iter6_jsd_and_popaf.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import requests
from sklearn.metrics import average_precision_score, roc_auc_score

ENRICHED = Path("scratch/issue302/missense_enriched.parquet")
OUT = Path("scratch/issue302/figs")
ANNO = Path("scratch/issue302")
KEY = ["chrom", "pos", "ref", "alt"]
SCOR = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
MODELS = [
    ("46M", 46, "scaling-v0.5-h640-p46M-step-215573"),
    ("76M", 76, "scaling-v0.5-h768-p76M-step-215573"),
    ("128M", 128, "scaling-v0.5-h896-p128M-step-215573"),
    ("255M", 255, "scaling-v0.5-h1152-p255M-step-215573"),
    ("476M", 476, "scaling-v0.5-h1408-p476M-step-215573"),
    ("1B", 1120, "scaling-v0.5-h1920-p1B-step-215573"),
    ("2B", 2270, "scaling-v0.5-h2432-p2B-step-215573"),
    ("4B", 4020, "scaling-v0.5-h2944-p4B-step-215573"),
]
SUBSETS = ["missense_variant", "synonymous_variant", "splicing"]
SHORT = {
    "missense_variant": "missense",
    "synonymous_variant": "synonymous",
    "splicing": "splicing",
}
# gnomAD major ancestry groups (exclude 'oth'); popmax = max over these from exome+genome
POPS = ["afr", "amr", "eas", "nfe", "sas", "asj", "fin", "ami", "mid"]


def sep_stats(label, score):
    pos, neg = score[label == 1], score[label == 0]
    sd = np.sqrt(
        ((pos.var(ddof=1) * (len(pos) - 1)) + (neg.var(ddof=1) * (len(neg) - 1)))
        / (len(pos) + len(neg) - 2)
    )
    return {
        "auprc": average_precision_score(label, score),
        "auroc": roc_auc_score(label, score),
        "cohen_d": (pos.mean() - neg.mean()) / sd,
        "frac_neg_above_pos_p90": float((neg > np.percentile(pos, 90)).mean()),
    }


# --------------------------------------------------------------------------- #
# (b) jsd_avg robustness
# --------------------------------------------------------------------------- #
def jsd_robustness() -> pd.DataFrame:
    print("\n" + "=" * 72)
    print("(b) jsd_avg protocol robustness — pathogenic-vs-benign separation vs scale")
    print("=" * 72)
    rows = []
    for lab, params, d in MODELS:
        df = (
            pl.read_parquet(f"{SCOR}/{d}/mendelian_traits.parquet")
            .filter(pl.col("subset").is_in(SUBSETS))
            .with_columns(((pl.col("jsd_fwd") + pl.col("jsd_rc")) / 2).alias("jsd"))
        )
        for s in SUBSETS:
            sub = df.filter(pl.col("subset") == s)
            rows.append(
                {
                    "subset": SHORT[s],
                    "size": lab,
                    "params": params,
                    **sep_stats(sub["label"].to_numpy(), sub["jsd"].to_numpy()),
                }
            )
    res = pd.DataFrame(rows)
    for s in ["missense", "synonymous", "splicing"]:
        r = res[res.subset == s].sort_values("params")
        print(
            f"  {s:11}: AUPRC {r['auprc'].iloc[0]:.3f}->{r['auprc'].iloc[-1]:.3f}  "
            f"Cohen_d {r['cohen_d'].iloc[0]:.2f}->peak{r['cohen_d'].max():.2f}->{r['cohen_d'].iloc[-1]:.2f}  "
            f"neg>posP90 {r['frac_neg_above_pos_p90'].iloc[0]:.3f}->{r['frac_neg_above_pos_p90'].iloc[-1]:.3f}"
        )
    mis = res[res.subset == "missense"].sort_values("params")
    print(
        f"\n  => missense jsd AUPRC peaks @ {mis.loc[mis.auprc.idxmax(), 'size']}, "
        f"Cohen_d peaks @ {mis.loc[mis.cohen_d.idxmax(), 'size']} then declines: "
        f"{'REPRODUCES minus_llr compression' if mis['cohen_d'].iloc[-1] < mis['cohen_d'].max() - 0.05 else 'does NOT reproduce'}"
    )
    return res


# --------------------------------------------------------------------------- #
# population AF
# --------------------------------------------------------------------------- #
def load_popaf(variants: pd.DataFrame) -> pd.DataFrame:
    cache = ANNO / "myvariant_popaf.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    ids = [
        f"chr{c}:g.{p}{r}>{a}"
        for c, p, r, a in zip(
            variants["chrom"], variants["pos"], variants["ref"], variants["alt"]
        )
    ]
    rows = []
    for i in range(0, len(ids), 1000):
        rr = requests.post(
            "https://myvariant.info/v1/variant",
            data={
                "ids": ",".join(ids[i : i + 1000]),
                "assembly": "hg38",
                "fields": "gnomad_exome.af,gnomad_genome.af",
            },
            timeout=180,
        )
        if rr.status_code != 200:
            print(f"  popaf batch {i}: HTTP {rr.status_code}")
            continue
        for h in rr.json():
            vals = {}
            for src in ("gnomad_exome", "gnomad_genome"):
                af = (h.get(src) or {}).get("af") or {}
                for p in POPS:
                    v = af.get(f"af_{p}")
                    if isinstance(v, (int, float)):
                        vals[p] = max(vals.get(p, 0.0), v)
            popmax = max(vals.values()) if vals else None
            popmax_pop = max(vals, key=vals.get) if vals else None
            rows.append(
                {
                    "_id": h.get("query"),
                    "popmax": popmax,
                    "popmax_pop": popmax_pop,
                    "n_pops": len(vals),
                }
            )
        time.sleep(0.5)
    out = variants.copy()
    out["_id"] = ids
    out = out.merge(pd.DataFrame(rows), on="_id", how="left")
    out.to_parquet(cache, index=False)
    return out


def popaf_probe() -> dict:
    print("\n" + "=" * 72)
    print("PART C — population-AF / human-tolerance probe (missense)")
    print("=" * 72)
    w = pl.read_parquet(ENRICHED).with_columns(pl.col("chrom").cast(str)).to_pandas()
    pa = load_popaf(w[KEY].copy())
    w = w.merge(pa[[*KEY, "popmax", "popmax_pop"]], on=KEY, how="left")
    w["lopsided"] = w["popmax"] / w["AF"].clip(lower=1e-6)
    neg = w[w.label == 0]
    q90, q10 = neg["4B"].quantile(0.90), neg["4B"].quantile(0.10)
    g = {
        "pathogenic (pos)": w[w.label == 1],
        "FP (top-10% neg @4B)": neg[neg["4B"] >= q90],
        "typical neg": neg[(neg["4B"] > q10) & (neg["4B"] < q90)],
        "easy neg (bot-10%)": neg[neg["4B"] <= q10],
    }
    print(f"\n  popmax coverage: {w['popmax'].notna().mean():.0%} of missense")
    print(
        f"\n  {'group':>22} {'n':>5} {'med globalAF':>12} {'med popmax':>10} {'med popmax/global':>17} {'%popmax>=5%':>11} {'%popmax>=1%':>11}"
    )
    for nm, s in g.items():
        sp = s.dropna(subset=["popmax"])
        print(
            f"  {nm:>22} {len(sp):>5} {sp['AF'].median():12.4f} {sp['popmax'].median():10.4f} "
            f"{(sp['popmax'] / sp['AF'].clip(lower=1e-6)).median():17.2f} "
            f"{(sp['popmax'] >= 0.05).mean() * 100:11.0f} {(sp['popmax'] >= 0.01).mean() * 100:11.0f}"
        )
    # FP split by popmax: robustly-common vs globally-rare
    fp = g["FP (top-10% neg @4B)"].dropna(subset=["popmax"])
    print(f"\n  FP set (n={len(fp)}) by popmax:")
    print(
        f"    robustly common (popmax>=5%): {(fp['popmax'] >= 0.05).sum():4d} ({(fp['popmax'] >= 0.05).mean() * 100:.0f}%)  "
        f"<- clearly human-tolerated, yet model calls pathogenic"
    )
    print(
        f"    common in a pop (popmax>=1%): {(fp['popmax'] >= 0.01).sum():4d} ({(fp['popmax'] >= 0.01).mean() * 100:.0f}%)"
    )
    print(
        f"    globally rare  (popmax<1%):   {(fp['popmax'] < 0.01).sum():4d} ({(fp['popmax'] < 0.01).mean() * 100:.0f}%)"
    )
    # population of the max for the robustly-common FPs
    rc = fp[fp["popmax"] >= 0.05]
    print(
        f"\n  popmax-population among robustly-common FPs (n={len(rc)}): "
        f"{dict(rc['popmax_pop'].value_counts().head(6))}"
    )
    # lopsided examples
    lop = (
        fp[(fp["popmax"] >= 0.05)]
        .assign(ratio=lambda d: d["popmax"] / d["AF"].clip(lower=1e-6))
        .nlargest(8, "ratio")
    )
    print("\n  Most population-lopsided FPs (popmax >> global):")
    print(
        f"    {'variant':>20} {'global':>7} {'popmax':>7} {'pop':>5} {'ratio':>6} {'phyloP':>6}"
    )
    for r in lop.itertuples():
        print(
            f"    {r.chrom}:{r.pos}{r.ref}>{r.alt:>3} {r.AF:7.4f} {r.popmax:7.3f} {str(r.popmax_pop):>5} {r.ratio:6.1f} {r.phyloP:6.2f}"
        )
    return {"w": w, "groups": g}


# --------------------------------------------------------------------------- #
def fig_jsd(res: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for s, c in [
        ("missense", "tab:blue"),
        ("synonymous", "tab:green"),
        ("splicing", "tab:orange"),
    ]:
        r = res[res.subset == s].sort_values("params")
        axes[0].plot(r["params"], r["cohen_d"], "o-", color=c, label=s)
        axes[1].plot(r["params"], r["auprc"], "o-", color=c, label=s)
    for ax, t, yl in [
        (axes[0], "pos/neg separation (Cohen's d)", "Cohen's d"),
        (axes[1], "AUPRC", "AUPRC"),
    ]:
        ax.set_xscale("log")
        ax.set_xlabel("params (M, log)")
        ax.set_ylabel(yl)
        ax.set_title(t)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(
        "(b) jsd_avg protocol — missense still compresses, splicing expands (robustness)",
        y=1.02,
    )
    _save(fig, "jsd_separation_vs_scale")


def fig_popaf(d: dict) -> None:
    g = d["groups"]
    names = [
        "pathogenic (pos)",
        "FP (top-10% neg @4B)",
        "typical neg",
        "easy neg (bot-10%)",
    ]
    colors = ["tab:red", "tab:orange", "tab:gray", "tab:blue"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    # panel 1: global AF vs popmax (medians) per group
    x = np.arange(len(names))
    gl = [g[n]["AF"].median() for n in names]
    pm = [g[n].dropna(subset=["popmax"])["popmax"].median() for n in names]
    axes[0].bar(x - 0.2, gl, 0.4, label="global AF", color="tab:gray")
    axes[0].bar(x + 0.2, pm, 0.4, label="popmax AF", color="tab:purple")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=7)
    axes[0].set_ylabel("median AF (log)")
    axes[0].set_title(
        "global vs popmax AF — FPs' human tolerance is\nunderstated by global AF"
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")
    # panel 2: fraction common by popmax
    fr5 = [
        (g[n].dropna(subset=["popmax"])["popmax"] >= 0.05).mean() * 100 for n in names
    ]
    fr1 = [
        (g[n].dropna(subset=["popmax"])["popmax"] >= 0.01).mean() * 100 for n in names
    ]
    axes[1].bar(x - 0.2, fr1, 0.4, label="popmax ≥ 1%", color="tab:cyan")
    axes[1].bar(x + 0.2, fr5, 0.4, label="popmax ≥ 5% (BS1-like)", color="tab:olive")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=7)
    axes[1].set_ylabel("% of variants")
    axes[1].set_title("Common-in-a-population by popmax")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    _save(fig, "popaf_characterization")


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / name}.{{png,svg}}")


def main() -> None:
    res = jsd_robustness()
    d = popaf_probe()
    print("\nFigures:")
    fig_jsd(res)
    fig_popaf(d)
    print("\nDone.")


if __name__ == "__main__":
    main()
