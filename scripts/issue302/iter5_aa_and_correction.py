"""issue #302 — iteration 5: amino-acid cut (E) + cheap correction probe.

E (the H1 vs H2 discriminator). Annotate each missense variant with its ref->alt
amino-acid change (from myvariant.info dbnsfp.hgvsp) and score the substitution's
conservativeness: Grantham distance (property formula; high = radical) and
BLOSUM62 (high = conservative). Then ask:
  - Are the conserved-benign FPs more *conservative* substitutions than true
    pathogenics? (H1: model over-tolerates / mis-handles tolerated AA changes)
  - Does AA-radicalness add signal *on top of* phyloP (partial), and does the
    score's AA-dependence GROW with scale? (H1 channel active vs pure H2 site-conservation)
  - Does the per-size missense degradation concentrate in conservative or radical
    substitutions?
  Bonus: AlphaMissense / PrimateAI (protein-LM VEP peers) on the FP set.

Correction probe (cheap, supervised — NOT LFB). Grouped-CV logistic AUPRC for
[4B], [4B,age], [4B,phyloP], [4B,age,phyloP,grantham]: is the bias *linearly*
correctable with the confounds we identified? Reference ceiling = GPN-Star-V.
(Uses labels, so it's an upper bound on correctability; the unsupervised LFB
remedy is a separate effort.)

Inputs: scratch/issue302/missense_enriched.parquet (iter3). External pulls cached.
Outputs (scratch/issue302/figs/): aa_grantham_blosum, aa_scale_by_stratum, correction_auprc.

Run:  uv run python scripts/issue302/iter5_aa_and_correction.py
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import requests
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ENRICHED = Path("scratch/issue302/missense_enriched.parquet")
OUT = Path("scratch/issue302/figs")
ANNO = Path("scratch/issue302")
KEY = ["chrom", "pos", "ref", "alt"]
SIZES = ["46M", "76M", "128M", "255M", "476M", "1B", "2B", "4B"]
PARAMS = {
    "46M": 46,
    "76M": 76,
    "128M": 128,
    "255M": 255,
    "476M": 476,
    "1B": 1120,
    "2B": 2270,
    "4B": 4020,
}
GPN_V = "https://gist.githubusercontent.com/gonzalobenegas/db282f89aa00244fbb7437dce0f069ef/raw/02484d50d9bfd80337e313652b26f98a9362b6b1/bolinas_mendelian_traits_GPN-Star-V.parquet"

# Grantham (1974) physicochemical properties: composition c, polarity p, volume v.
GRANTHAM_PROPS = {
    "A": (0.00, 8.1, 31),
    "R": (0.65, 10.5, 124),
    "N": (1.33, 11.6, 56),
    "D": (1.38, 13.0, 54),
    "C": (2.75, 5.5, 55),
    "Q": (0.89, 10.5, 85),
    "E": (0.92, 12.3, 83),
    "G": (0.74, 9.0, 3),
    "H": (0.58, 10.4, 96),
    "I": (0.00, 5.2, 111),
    "L": (0.00, 4.9, 111),
    "K": (0.33, 11.3, 119),
    "M": (0.00, 5.7, 105),
    "F": (0.00, 5.0, 132),
    "P": (0.39, 8.0, 32.5),
    "S": (1.42, 9.2, 32),
    "T": (0.71, 8.6, 61),
    "W": (0.13, 5.4, 170),
    "Y": (0.20, 6.2, 136),
    "V": (0.00, 5.9, 84),
}
THREE_TO_ONE = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
}
_HGVSP3 = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")


def grantham(a: str, b: str) -> float | None:
    if a not in GRANTHAM_PROPS or b not in GRANTHAM_PROPS or a == b:
        return None if a not in GRANTHAM_PROPS or b not in GRANTHAM_PROPS else 0.0
    (ca, pa, va), (cb, pb, vb) = GRANTHAM_PROPS[a], GRANTHAM_PROPS[b]
    return 50.723 * np.sqrt(
        1.833 * (ca - cb) ** 2 + 0.1018 * (pa - pb) ** 2 + 0.000399 * (va - vb) ** 2
    )


def blosum62():
    try:
        from Bio.Align import substitution_matrices

        m = substitution_matrices.load("BLOSUM62")
        return lambda a, b: (
            float(m[a, b]) if a in m.alphabet and b in m.alphabet else None
        )
    except Exception:
        return None


def parse_aa(hgvsp) -> tuple[str, str] | tuple[None, None]:
    """First 3-letter missense AA change from a dbnsfp.hgvsp list/str."""
    if hgvsp is None:
        return None, None
    items = hgvsp if isinstance(hgvsp, (list, np.ndarray)) else [hgvsp]
    for s in items:
        m = _HGVSP3.search(str(s))
        if m and m.group(1) in THREE_TO_ONE and m.group(3) in THREE_TO_ONE:
            return THREE_TO_ONE[m.group(1)], THREE_TO_ONE[m.group(3)]
    return None, None


def load_aa(variants: pd.DataFrame) -> pd.DataFrame:
    """myvariant.info hgvsp + AlphaMissense + PrimateAI for missense (cached)."""
    cache = ANNO / "myvariant_aa.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    ids = [
        f"chr{c}:g.{p}{r}>{a}"
        for c, p, r, a in zip(
            variants["chrom"], variants["pos"], variants["ref"], variants["alt"]
        )
    ]
    fields = "dbnsfp.hgvsp,dbnsfp.alphamissense.score,dbnsfp.primateai.score"
    rows = []
    for i in range(0, len(ids), 1000):
        rr = requests.post(
            "https://myvariant.info/v1/variant",
            data={
                "ids": ",".join(ids[i : i + 1000]),
                "assembly": "hg38",
                "fields": fields,
            },
            timeout=180,
        )
        if rr.status_code != 200:
            print(f"  myvariant aa batch {i}: HTTP {rr.status_code}")
            continue
        for h in rr.json():
            db = h.get("dbnsfp") or {}
            am = (db.get("alphamissense") or {}).get("score")
            pai = (db.get("primateai") or {}).get("score")
            rows.append(
                {
                    "_id": h.get("query"),
                    "hgvsp": db.get("hgvsp"),
                    "alphamissense": max(am) if isinstance(am, list) else am,
                    "primateai": max(pai) if isinstance(pai, list) else pai,
                }
            )
        time.sleep(0.5)
    out = variants.copy()
    out["_id"] = ids
    out = out.merge(pd.DataFrame(rows), on="_id", how="left")
    aa = out["hgvsp"].apply(parse_aa)
    out["aa_ref"] = [x[0] for x in aa]
    out["aa_alt"] = [x[1] for x in aa]
    out = out.drop(
        columns=["hgvsp"]
    )  # mixed list/None — not parquet-serializable; aa_ref/alt extracted
    out.to_parquet(cache, index=False)
    return out


def build() -> pd.DataFrame:
    w = pl.read_parquet(ENRICHED).with_columns(pl.col("chrom").cast(str)).to_pandas()
    aa = load_aa(w[KEY].copy())
    w = w.merge(
        aa[[*KEY, "aa_ref", "aa_alt", "alphamissense", "primateai"]], on=KEY, how="left"
    )
    b62 = blosum62()
    w["grantham"] = [
        grantham(r, a) if isinstance(r, str) and isinstance(a, str) else None
        for r, a in zip(w["aa_ref"], w["aa_alt"])
    ]
    w["blosum62"] = [
        b62(r, a) if (b62 and isinstance(r, str) and isinstance(a, str)) else None
        for r, a in zip(w["aa_ref"], w["aa_alt"])
    ]
    # GPN-V reference
    g = (
        pl.read_parquet(GPN_V)
        .filter(pl.col("split") == "train")
        .with_columns(
            [pl.col("chrom").cast(str), (-pl.col("llr_calibrated")).alias("gpn_V")]
        )
        .select([*KEY, "gpn_V"])
        .to_pandas()
    )
    w = w.merge(g, on=KEY, how="left")
    cov = w["grantham"].notna().mean()
    # sanity: Grantham reproduces known pairs
    assert abs(grantham("L", "I") - 5) < 2 and abs(grantham("C", "W") - 215) < 5, (
        "Grantham formula off"
    )
    print(
        f"  AA change parsed for {w['grantham'].notna().sum()}/{len(w)} missense ({cov:.0%}); "
        f"AlphaMissense {w['alphamissense'].notna().mean():.0%}, BLOSUM {'on' if b62 else 'OFF'}"
    )
    return w


def groups(w: pd.DataFrame) -> dict[str, pd.DataFrame]:
    neg = w[w["label"] == 0]
    q90, q10 = neg["4B"].quantile(0.90), neg["4B"].quantile(0.10)
    return {
        "pathogenic (pos)": w[w["label"] == 1],
        "FP (top-10% neg @4B)": neg[neg["4B"] >= q90],
        "typical neg": neg[(neg["4B"] > q10) & (neg["4B"] < q90)],
    }


def block_E(w: pd.DataFrame) -> dict:
    print("\n" + "=" * 74)
    print("E — amino-acid cut (Grantham high=radical; BLOSUM62 high=conservative)")
    print("=" * 74)
    g = groups(w)
    print(
        f"\n  {'group':>22} {'n_aa':>5} {'medGrantham':>11} {'medBLOSUM':>9} {'%conserv(G<100)':>15} {'medAlphaMis':>11} {'medPrimateAI':>12}"
    )
    for nm, s in g.items():
        sa = s.dropna(subset=["grantham"])
        print(
            f"  {nm:>22} {len(sa):>5} {sa['grantham'].median():11.0f} {sa['blosum62'].median():9.1f} "
            f"{(sa['grantham'] < 100).mean() * 100:15.0f} {s['alphamissense'].median():11.3f} {s['primateai'].median():12.3f}"
        )

    # does AA-radicalness add on top of phyloP? partial: corr(grantham, score | phyloP) on negatives
    neg = w[(w["label"] == 0) & w["grantham"].notna() & w["phyloP"].notna()].copy()
    print(
        "\n  Spearman(Grantham, score) on negatives, raw and phyloP-residualized, per size:"
    )
    print(f"    {'size':>5} {'raw':>7} {'|phyloP':>8}")
    aa_dep = []
    for size in SIZES:
        raw = spearmanr(neg["grantham"], neg[size]).correlation
        # residualize score and grantham on phyloP (rank), then correlate residuals
        from numpy.polynomial import polynomial as P  # noqa

        sx = neg["phyloP"].rank()
        rs = neg[size] - np.polyval(np.polyfit(sx, neg[size], 1), sx)
        rg = neg["grantham"] - np.polyval(np.polyfit(sx, neg["grantham"], 1), sx)
        part = spearmanr(rg, rs).correlation
        aa_dep.append(
            {"size": size, "params": PARAMS[size], "raw": raw, "partial_phyloP": part}
        )
        print(f"    {size:>5} {raw:+7.3f} {part:+8.3f}")

    # per-size missense AUPRC in conservative vs radical strata
    wg = w[w["grantham"].notna()]
    print("\n  Missense AUPRC by substitution class, per size:")
    print(
        f"    {'class':>22} {'n':>5} {'npos':>5} "
        + " ".join(f"{s:>6}" for s in ["128M", "1B", "4B"])
    )
    strat_rows = []
    for label_, mask in [
        ("conservative (G<100)", wg["grantham"] < 100),
        ("radical (G>=100)", wg["grantham"] >= 100),
    ]:
        sub = wg[mask]
        row = {"klass": label_, "n": len(sub), "npos": int(sub["label"].sum())}
        cells = []
        for size in SIZES:
            row[size] = average_precision_score(sub["label"], sub[size])
        for size in ["128M", "1B", "4B"]:
            cells.append(f"{row[size]:6.2f}")
        strat_rows.append(row)
        print(
            f"    {label_:>22} {len(sub):>5} {int(sub['label'].sum()):>5} "
            + " ".join(cells)
        )
    return {
        "groups": g,
        "aa_dep": pd.DataFrame(aa_dep),
        "strat": pd.DataFrame(strat_rows),
    }


def block_correction(w: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 74)
    print("Correction probe (grouped-CV logistic AUPRC; supervised → upper bound)")
    print("=" * 74)
    d = w.dropna(subset=["phyloP", "age_mya", "grantham", "gpn_V"]).copy()
    y = d["label"].values
    grp = d["match_group"].values
    feature_sets = {
        "4B only": ["4B"],
        "4B + age": ["4B", "age_mya"],
        "4B + phyloP": ["4B", "phyloP"],
        "4B + age + phyloP + Grantham": ["4B", "age_mya", "phyloP", "grantham"],
    }
    rows = []
    gkf = GroupKFold(n_splits=5)
    for name, feats in feature_sets.items():
        oof = np.zeros(len(d))
        X = d[feats].values
        for tr, te in gkf.split(X, y, grp):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
            oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        ap = average_precision_score(y, oof)
        rows.append({"model": name, "auprc": ap})
        print(f"  {name:>32}: CV-AUPRC = {ap:.3f}")
    gpn_ap = average_precision_score(y, d["gpn_V"].values)
    rows.append({"model": "GPN-Star-V (reference ceiling)", "auprc": gpn_ap})
    print(f"  {'GPN-Star-V (reference)':>32}: AUPRC = {gpn_ap:.3f}")
    print(f"  (n={len(d)}, prevalence={y.mean():.3f})")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def fig_aa_box(g: dict) -> None:
    names = list(g.keys())
    colors = ["tab:red", "tab:orange", "tab:gray"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (col, lab, line) in zip(
        axes,
        [
            ("grantham", "Grantham distance (high = radical)", 100),
            ("blosum62", "BLOSUM62 (high = conservative)", 0),
        ],
    ):
        data = [g[n][col].dropna().values for n in names]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax.axhline(line, color="k", ls=":", lw=1)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
        ax.set_ylabel(lab, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(
        "E — AA-substitution conservativeness: FP set vs pathogenic vs benign", y=1.02
    )
    _save(fig, "aa_grantham_blosum")


def fig_strat(res: dict) -> None:
    strat, aa_dep = res["strat"], res["aa_dep"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for _, row in strat.iterrows():
        ys = [row[s] for s in SIZES]
        axes[0].plot(
            [PARAMS[s] for s in SIZES], ys, "o-", label=f"{row['klass']} (n={row['n']})"
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("params (M, log)")
    axes[0].set_ylabel("missense AUPRC")
    axes[0].set_title("Does the scale gap concentrate by substitution class?")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(
        aa_dep["params"], aa_dep["raw"], "o-", label="Spearman(Grantham, score)"
    )
    axes[1].plot(
        aa_dep["params"], aa_dep["partial_phyloP"], "s--", label="partial | phyloP"
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("params (M, log)")
    axes[1].set_ylabel("Spearman on negatives")
    axes[1].set_title("Does the score's AA-radicalness dependence grow with scale?")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    _save(fig, "aa_scale_by_stratum")


def fig_correction(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    colors = ["tab:blue", "tab:cyan", "tab:purple", "tab:olive", "tab:green"]
    ax.barh(df["model"], df["auprc"], color=colors[: len(df)])
    for i, v in enumerate(df["auprc"]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("missense AUPRC (grouped 5-fold CV; GPN-V = fixed score)")
    ax.set_title(
        "Correction probe — is the bias linearly correctable with the confounds?"
    )
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")
    _save(fig, "correction_auprc")


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / name}.{{png,svg}}")


def main() -> None:
    print("Building (AA change + Grantham/BLOSUM + AlphaMissense/PrimateAI + GPN-V)...")
    w = build()
    res = block_E(w)
    corr = block_correction(w)
    print("\nFigures:")
    fig_aa_box(res["groups"])
    fig_strat(res)
    fig_correction(corr)
    print("\nDone.")


if __name__ == "__main__":
    main()
