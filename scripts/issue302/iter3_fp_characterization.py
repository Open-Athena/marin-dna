"""issue #302 — iteration 3: characterize the "conserved-looking benign" missense
false-positive set (the failure mode from iter1/iter2) across every axis.

The FP set = missense NEGATIVES (benign; AF >= 1e-3) that the 4B ladder model
scores high (pathogenic-like) — the over-confident calls that drive the missense
AUPRC degradation. We profile that set, vs the pathogenic positives and the
benign baseline, on:

  AF              gnomAD allele frequency (in the scores parquet)
  conservation    phyloP_241m (mammalian) + phyloP_100v (vertebrate, deeper)
  ClinVar/CADD/REVEL   myvariant.info (hg38) — is the FP benign / low by other
                       predictors (⇒ our-model-specific) and absent from ClinVar?
  gene constraint LOEUF + pLI (gnomAD v2.1.1 by-gene), by exon_closest_pc_gene_id
  gene age        Liebeskind 2016 modeAge (Zenodo 51708) -> MYA, ENSG->UniProt
                  via mygene.info  (reuses #203 iter-5's source/method)

Gene-age blocks mirror #203 iter-5: top-FP median age vs baseline per size
(Mann-Whitney), missense AUPRC stratified by age bucket across the ladder, and
Spearman(age, score) on negatives.

Inputs: iter1 ladder cache (scratch/issue302/combined_scores.parquet — run iter1
first); gnomAD LOEUF (scratch/issue302/loeuf.txt.bgz). All external pulls cached.

Outputs (scratch/issue302/figs/, PNG+SVG):
  fp_characterization_panels   AF / phyloP / LOEUF / gene-age, FP vs pos vs neg
  missense_auprc_by_gene_age   per-size AUPRC by age bucket (does the gap sit in old genes?)
  fp_revel_clinvar             REVEL distribution + ClinVar-significance breakdown, FP vs pos

Run:  uv run python scripts/issue302/iter3_fp_characterization.py
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import requests
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import average_precision_score

LADDER_CACHE = Path("scratch/issue302/combined_scores.parquet")  # from iter1
LOEUF_BGZ = Path("scratch/issue302/loeuf.txt.bgz")
LOEUF_URL = "https://storage.googleapis.com/gcp-public-data--gnomad/release/2.1.1/constraint/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz"
OUT = Path("scratch/issue302/figs")
ANNO = Path("scratch/issue302")  # caches
KEY = ["chrom", "pos", "ref", "alt"]
PHYLOP_100V = "s3://oa-bolinas/snakemake/conservation_eval/results/mendelian_traits/phyloP_100v_train.parquet"

GENE_AGES_ZIP = "https://zenodo.org/api/records/51708/files/Gene-Ages-v1.0.zip/content"
GENE_AGES_PATH_IN_ZIP = "marcottelab-Gene-Ages-fee8d00/Main/main_HUMAN.csv"
MODE_AGE_MYA: dict[str, int] = {  # Liebeskind 2016 / TimeTree approx divergence (MYA)
    "Cellular_organisms": 4290,
    "Euk_Archaea": 3000,
    "Euk+Bac": 3500,
    "Eukaryota": 1962,
    "Opisthokonta": 1105,
    "Eumetazoa": 824,
    "Vertebrata": 615,
    "Mammalia": 320,
}
AGE_ORDER = [
    "Vertebrata",
    "Eumetazoa",
    "Opisthokonta",
    "Eukaryota",
    "Euk+Bac",
    "Cellular_organisms",
]


# --------------------------------------------------------------------------- #
# External annotations (cached)
# --------------------------------------------------------------------------- #
def load_gene_age_for(ensgs: list[str]) -> pd.DataFrame:
    """ENSG -> modeAge/age_mya via Zenodo-51708 (UniProt-keyed) + mygene.info."""
    cache = ANNO / "ensg_to_age.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    r = requests.get(GENE_AGES_ZIP, timeout=120)
    r.raise_for_status()
    with (
        zipfile.ZipFile(io.BytesIO(r.content)) as zf,
        zf.open(GENE_AGES_PATH_IN_ZIP) as f,
    ):
        ages = pd.read_csv(f, index_col=0)
    ages["age_mya"] = ages["modeAge"].map(MODE_AGE_MYA)
    ages = ages[["modeAge", "age_mya"]]
    # ENSG -> UniProt (Swiss-Prot) via mygene.info
    uni: dict[str, str] = {}
    for i in range(0, len(ensgs), 500):
        chunk = ensgs[i : i + 500]
        rr = requests.post(
            "https://mygene.info/v3/query",
            data={
                "q": ",".join(chunk),
                "scopes": "ensembl.gene",
                "fields": "uniprot.Swiss-Prot",
                "species": "human",
            },
            timeout=120,
        )
        if rr.status_code != 200:
            continue
        for hit in rr.json():
            q, u = hit.get("query"), (hit.get("uniprot") or {}).get("Swiss-Prot")
            if isinstance(u, list):
                u = u[0] if u else None
            if u and q and q not in uni:
                uni[q] = u
        time.sleep(0.3)
    df = pd.DataFrame([{"ensg": k, "uniprot": v} for k, v in uni.items()]).merge(
        ages.reset_index().rename(columns={"index": "uniprot"}),
        on="uniprot",
        how="left",
    )
    df = df[["ensg", "modeAge", "age_mya"]]
    df.to_parquet(cache, index=False)
    print(
        f"  gene age: mapped {df['ensg'].nunique()} / {len(ensgs)} ENSGs, {df['age_mya'].notna().sum()} numeric"
    )
    return df


def load_myvariant_for(variants: pd.DataFrame) -> pd.DataFrame:
    """ClinVar significance + CADD phred + REVEL via myvariant.info (hg38), cached."""
    cache = ANNO / "myvariant_anno.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    ids = [
        f"chr{c}:g.{p}{r}>{a}"
        for c, p, r, a in zip(
            variants["chrom"], variants["pos"], variants["ref"], variants["alt"]
        )
    ]
    rows: list[dict] = []
    fields = "clinvar.rcv.clinical_significance,cadd.phred,dbnsfp.revel.score"
    for i in range(0, len(ids), 1000):
        chunk = ids[i : i + 1000]
        rr = requests.post(
            "https://myvariant.info/v1/variant",
            data={"ids": ",".join(chunk), "assembly": "hg38", "fields": fields},
            timeout=180,
        )
        if rr.status_code != 200:
            print(f"  myvariant batch {i}: HTTP {rr.status_code}")
            continue
        for h in rr.json():
            cv = h.get("clinvar")
            sig = None
            if cv:
                rcv = cv.get("rcv")
                rcv = rcv if isinstance(rcv, list) else [rcv]
                sigs = [
                    x.get("clinical_significance", "")
                    for x in rcv
                    if isinstance(x, dict)
                ]
                sig = "; ".join(s for s in sigs if s) or None
            revel = ((h.get("dbnsfp") or {}).get("revel") or {}).get("score")
            if isinstance(revel, list):
                revel = max(revel) if revel else None
            cadd = (h.get("cadd") or {}).get("phred")
            if isinstance(cadd, list):
                cadd = max(cadd) if cadd else None
            rows.append(
                {
                    "_id": h.get("query"),
                    "clinvar_sig": sig,
                    "revel": revel,
                    "cadd_phred": cadd,
                }
            )
        time.sleep(0.5)
    out = variants.copy()
    out["_id"] = ids
    out = out.merge(pd.DataFrame(rows), on="_id", how="left")
    out.to_parquet(cache, index=False)
    return out


def load_loeuf() -> pl.DataFrame:
    import gzip

    if not LOEUF_BGZ.exists():
        r = requests.get(LOEUF_URL, timeout=180)
        r.raise_for_status()
        LOEUF_BGZ.write_bytes(r.content)
    with gzip.open(LOEUF_BGZ, "rt") as f:
        loeuf = pd.read_csv(f, sep="\t")[
            ["gene_id", "oe_lof_upper", "pLI", "oe_lof_upper_bin"]
        ]
    loeuf = loeuf.rename(columns={"gene_id": "ensg", "oe_lof_upper": "loeuf"}).dropna(
        subset=["ensg"]
    )
    loeuf["ensg"] = loeuf["ensg"].str.replace(r"\.\d+$", "", regex=True)
    return pl.from_pandas(loeuf).unique(subset=["ensg"], keep="first")


# --------------------------------------------------------------------------- #
# Build enriched missense table
# --------------------------------------------------------------------------- #
def build() -> pd.DataFrame:
    lad = pl.read_parquet(LADDER_CACHE).filter(pl.col("subset") == "missense_variant")
    wide = (
        lad.pivot(
            values="mll",
            index=[
                *KEY,
                "label",
                "match_group",
                "AF",
                "phyloP",
                "clinvar_id",
                "exon_closest_pc_gene_id",
            ],
            on="size",
            aggregate_function="first",
        )
        .rename({"exon_closest_pc_gene_id": "ensg"})
        .with_columns(pl.col("ensg").str.replace(r"\.\d+$", ""))
    )
    # deeper conservation track
    p100 = pl.read_parquet(PHYLOP_100V).select(
        [*KEY, pl.col("score").alias("phyloP_100v")]
    )
    wide = wide.join(p100, on=KEY, how="left")
    # LOEUF
    wide = wide.join(load_loeuf(), on="ensg", how="left")
    # gene age
    age = pl.from_pandas(
        load_gene_age_for(wide["ensg"].drop_nulls().unique().to_list())
    )
    wide = wide.join(age, on="ensg", how="left")
    w = wide.to_pandas()
    # myvariant (ClinVar/CADD/REVEL) for all missense
    mv = load_myvariant_for(w[KEY].copy())
    w = w.merge(mv[[*KEY, "clinvar_sig", "revel", "cadd_phred"]], on=KEY, how="left")
    return w


def groups(w: pd.DataFrame) -> dict[str, pd.DataFrame]:
    neg = w[w["label"] == 0]
    q90, q10 = neg["4B"].quantile(0.90), neg["4B"].quantile(0.10)
    return {
        "pathogenic (pos)": w[w["label"] == 1],
        "FP (top-10% neg @4B)": neg[neg["4B"] >= q90],
        "typical neg": neg[(neg["4B"] > q10) & (neg["4B"] < q90)],
        "easy neg (bot-10%)": neg[neg["4B"] <= q10],
    }


def characterize(w: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("FP-SET CHARACTERIZATION (missense; 4B-defined FP set)")
    print("=" * 78)
    g = groups(w)

    def benign_frac(s):
        sig = s["clinvar_sig"].dropna().str.lower()
        return (sig.str.contains("benign")).mean() if len(sig) else np.nan

    hdr = f"{'group':>22} {'n':>5} {'medAF':>7} {'%com':>5} {'medP241':>7} {'medP100v':>8} {'medLOEUF':>8} {'%constr':>7} {'medAgeMYA':>9} {'%old':>5} {'medREVEL':>8} {'medCADD':>7} {'CV%benign':>9} {'CV%none':>7}"
    print("\n" + hdr)
    for name, s in g.items():
        old = (s["age_mya"] >= 1962).mean()
        constr = (s["loeuf"] < 0.35).mean()
        cv_none = s["clinvar_sig"].isna().mean()
        print(
            f"{name:>22} {len(s):>5} {s['AF'].median():7.4f} {(s['AF'] >= 0.01).mean() * 100:5.0f} "
            f"{s['phyloP'].median():7.2f} {s['phyloP_100v'].median():8.2f} "
            f"{s['loeuf'].median():8.2f} {constr * 100:7.0f} {s['age_mya'].median():9.0f} {old * 100:5.0f} "
            f"{s['revel'].median():8.3f} {s['cadd_phred'].median():7.1f} {benign_frac(s) * 100:9.0f} {cv_none * 100:7.0f}"
        )
    print(
        "\n  %com = AF>=1%; %constr = LOEUF<0.35 (constrained gene); %old = gene age>=Eukaryota(1962 MYA);"
    )
    print(
        "  CV%benign = of ClinVar-annotated, fraction benign/likely-benign; CV%none = fraction absent from ClinVar."
    )

    # gene-age: top-FP median age vs baseline, per size
    print(
        "\n  Gene-age: top-50 FP median age (MYA) vs all-neg baseline, per size (Mann-Whitney p):"
    )
    neg = w[(w["label"] == 0) & w["age_mya"].notna()]
    base_med = neg["age_mya"].median()
    print(f"    baseline all-neg median age = {base_med:.0f} MYA (n={len(neg)})")
    for size in ["128M", "1B", "4B"]:
        top = neg.nlargest(50, size)
        p = mannwhitneyu(top["age_mya"], neg["age_mya"], alternative="greater").pvalue
        rho = spearmanr(neg[size], neg["age_mya"]).correlation
        print(
            f"    {size:>4}: top-50 FP median age = {top['age_mya'].median():.0f}  mean={top['age_mya'].mean():.0f}  "
            f"MW-p(older)={p:.3f}  Spearman(score,age)_allneg={rho:+.3f}"
        )

    # missense AUPRC by age bucket across the ladder
    print("\n  Missense AUPRC by gene-age bucket (per size):")
    wa = w[w["age_mya"].notna()]
    print(
        f"    {'bucket (MYA)':>26} {'n':>5} {'npos':>5} {'128M':>6} {'1B':>6} {'4B':>6}"
    )
    age_rows = []
    for b in AGE_ORDER:
        sub = wa[wa["modeAge"] == b]
        if len(sub) < 20 or sub["label"].sum() < 3:
            continue
        row = {
            "bucket": b,
            "mya": MODE_AGE_MYA[b],
            "n": len(sub),
            "npos": int(sub["label"].sum()),
        }
        cells = []
        for size in ["128M", "1B", "4B"]:
            ap = average_precision_score(sub["label"], sub[size])
            row[size] = ap
            cells.append(f"{ap:6.2f}")
        age_rows.append(row)
        print(
            f"    {b + f' (~{MODE_AGE_MYA[b]})':>26} {len(sub):>5} {int(sub['label'].sum()):>5} "
            + " ".join(cells)
        )
    pd.DataFrame(age_rows).to_parquet(ANNO / "auprc_by_age_bucket.parquet", index=False)
    return g, age_rows


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_panels(g: dict[str, pd.DataFrame]) -> None:
    specs = [
        ("AF", "AF (log)", True),
        ("phyloP", "phyloP_241m", False),
        ("loeuf", "LOEUF (gene constraint; low=constrained)", False),
        ("age_mya", "gene age (MYA)", False),
    ]
    names = list(g.keys())
    colors = ["tab:red", "tab:orange", "tab:gray", "tab:blue"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, (col, lab, logy) in zip(axes, specs):
        data = [g[n][col].dropna().values for n in names]
        if logy:
            data = [np.clip(d, 1e-6, None) for d in data]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(
            [n.replace(" (", "\n(") for n in names], fontsize=7, rotation=0
        )
        ax.set_ylabel(lab, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(
        "Iter3 — missense FP set vs pathogenic vs benign baseline (4B-defined FP = top-10% negatives by score)",
        y=1.02,
    )
    _save(fig, "fp_characterization_panels")


def fig_age_auprc(age_rows: list[dict]) -> None:
    if not age_rows:
        return
    df = pd.DataFrame(age_rows).sort_values("mya")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(df))
    for size, c in [("128M", "tab:green"), ("1B", "tab:orange"), ("4B", "tab:blue")]:
        ax.plot(x, df[size], "o-", color=c, label=size)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{b}\n(~{m})" for b, m in zip(df["bucket"], df["mya"])], fontsize=7
    )
    ax.set_xlabel("gene age bucket (older →)")
    ax.set_ylabel("missense AUPRC")
    ax.set_title("Missense AUPRC by gene age — does the scale gap sit in OLD genes?")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "missense_auprc_by_gene_age")


def fig_revel_clinvar(g: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # REVEL distributions: FP vs pathogenic vs typical neg
    for name, c in [
        ("pathogenic (pos)", "tab:red"),
        ("FP (top-10% neg @4B)", "tab:orange"),
        ("typical neg", "tab:gray"),
    ]:
        r = g[name]["revel"].dropna().values
        if len(r):
            axes[0].hist(
                r,
                bins=np.linspace(0, 1, 31),
                density=True,
                histtype="step",
                lw=2,
                color=c,
                label=f"{name} (n={len(r)})",
            )
    axes[0].axvline(0.5, color="k", ls=":", lw=1, label="REVEL=0.5 (path. threshold)")
    axes[0].set_xlabel("REVEL score")
    axes[0].set_ylabel("density")
    axes[0].set_title("Do other predictors flag the FPs? (REVEL)")
    axes[0].legend(fontsize=7)
    # ClinVar significance breakdown for FP set
    fp = g["FP (top-10% neg @4B)"]
    sig = fp["clinvar_sig"].fillna("(absent from ClinVar)").str.lower()
    cat = pd.Series(
        np.where(
            sig.str.contains("benign"),
            "benign/LB",
            np.where(
                sig.str.contains("pathogenic"),
                "path/LP",
                np.where(sig.str.contains("absent"), "absent", "VUS/other"),
            ),
        )
    )
    vc = cat.value_counts()
    axes[1].bar(
        vc.index,
        vc.values,
        color=["tab:green", "tab:red", "lightgray", "tab:purple"][: len(vc)],
    )
    axes[1].set_ylabel("# FP variants")
    axes[1].set_title(f"ClinVar significance of the FP set (n={len(fp)})")
    for i, v in enumerate(vc.values):
        axes[1].text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    _save(fig, "fp_revel_clinvar")


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / name}.{{png,svg}}")


def main() -> None:
    print(
        "Building enriched missense table (conservation + LOEUF + gene age + myvariant)..."
    )
    w = build()
    g, age_rows = characterize(w)
    print("\nFigures:")
    fig_panels(g)
    fig_age_auprc(age_rows)
    fig_revel_clinvar(g)
    w.to_parquet(ANNO / "missense_enriched.parquet", index=False)
    print("\nDone.")


if __name__ == "__main__":
    main()
