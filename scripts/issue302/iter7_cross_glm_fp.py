"""issue #302 — iteration 7: cross-gLM false-positive overlap.

Do other likelihood-based gLMs over-call the SAME conserved-benign missense
variants our ladder does? If a shared "universal FP" set exists across model
families, the conserved-benign failure mode is a benchmark-level property of
likelihood-based variant scoring, not specific to our models.

Models (all scored on the same Mendelian-train missense variants; higher =
more pathogenic): our ladder 128M / 1B / 4B; Evo 2 1B / 7B / 40B (#131/#203
gists, minus_llr); GPN-Star-V (#145/#203 gist, cLLR = -llr_calibrated).

For each model, FP set = top-10% of the shared missense NEGATIVES by its score.
Computes the pairwise Jaccard matrix, within- vs cross-family overlap, and the
"universal FP" distribution (how many models flag each negative) + a
characterization of the variants many models flag (phyloP / gene age / popmax /
AlphaMissense) vs the idiosyncratic ones.

Inputs (all cached / gists): enriched missense (iter3), popaf (iter6), aa (iter5).
Outputs (scratch/issue302/figs/): crossglm_fp_jaccard, crossglm_universal_fp.

Run:  uv run python scripts/issue302/iter7_cross_glm_fp.py
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

OUT = Path("scratch/issue302/figs")
ANNO = Path("scratch/issue302")
KEY = ["chrom", "pos", "ref", "alt"]
GIST = "https://gist.githubusercontent.com/gonzalobenegas"
EVO = {
    "evo2-1B": f"{GIST}/3649e68fb63ca1f3443e4486078eb4d8/raw/b6c254849a71ca0783a24218a4fe9037e887e8f7/evo2_1b_base_train.parquet",
    "evo2-7B": f"{GIST}/3649e68fb63ca1f3443e4486078eb4d8/raw/e72d3d2e14955a670b8229dc8d525a69ea88c05c/evo2_7b_train.parquet",
    "evo2-40B": f"{GIST}/3649e68fb63ca1f3443e4486078eb4d8/raw/2b425e759811c201ca806ae4c8733fd7732220a6/evo2_40b_train.parquet",
}
GPN_V = f"{GIST}/db282f89aa00244fbb7437dce0f069ef/raw/02484d50d9bfd80337e313652b26f98a9362b6b1/bolinas_mendelian_traits_GPN-Star-V.parquet"
ORDER = ["ours-128M", "ours-1B", "ours-4B", "evo2-1B", "evo2-7B", "evo2-40B", "GPN-V"]
FAMILY = {
    "ours-128M": "ours",
    "ours-1B": "ours",
    "ours-4B": "ours",
    "evo2-1B": "evo2",
    "evo2-7B": "evo2",
    "evo2-40B": "evo2",
    "GPN-V": "gpn",
}


def load() -> pd.DataFrame:
    w = (
        pl.read_parquet(ANNO / "missense_enriched.parquet")
        .with_columns(pl.col("chrom").cast(str))
        .select(
            [
                *KEY,
                "label",
                "phyloP",
                "age_mya",
                pl.col("128M").alias("ours-128M"),
                pl.col("1B").alias("ours-1B"),
                pl.col("4B").alias("ours-4B"),
            ]
        )
    )
    for name, url in EVO.items():
        e = (
            pl.read_parquet(url)
            .with_columns(pl.col("chrom").cast(str))
            .select([*KEY, pl.col("minus_llr").alias(name)])
        )
        w = w.join(e, on=KEY, how="inner")
    g = (
        pl.read_parquet(GPN_V)
        .filter(pl.col("split") == "train")
        .with_columns(
            [pl.col("chrom").cast(str), (-pl.col("llr_calibrated")).alias("GPN-V")]
        )
        .select([*KEY, "GPN-V"])
    )
    w = w.join(g, on=KEY, how="inner")
    # annotations for universal-FP characterization
    pa = (
        pl.read_parquet(ANNO / "myvariant_popaf.parquet")
        .with_columns(pl.col("chrom").cast(str))
        .select([*KEY, "popmax"])
    )
    aa = (
        pl.read_parquet(ANNO / "myvariant_aa.parquet")
        .with_columns(pl.col("chrom").cast(str))
        .select([*KEY, "alphamissense"])
    )
    w = w.join(pa, on=KEY, how="left").join(aa, on=KEY, how="left")
    return w.to_pandas()


def main() -> None:
    w = load()
    neg = w[w.label == 0].reset_index(drop=True)
    print(f"shared missense negatives across all {len(ORDER)} models: n={len(neg)}")
    k = max(1, int(round(0.10 * len(neg))))
    fp = {
        m: set(neg.nlargest(k, m).index) for m in ORDER
    }  # each model's top-10% FP set

    # pairwise Jaccard
    print("\n=== pairwise FP-set Jaccard (top-10% negatives) ===")
    J = pd.DataFrame(index=ORDER, columns=ORDER, dtype=float)
    for a, b in combinations(ORDER, 2):
        j = len(fp[a] & fp[b]) / len(fp[a] | fp[b])
        J.loc[a, b] = J.loc[b, a] = j
    np.fill_diagonal(J.values, 1.0)
    print(J.round(2).to_string())

    def avg_pairs(pairs):
        vs = [len(fp[a] & fp[b]) / len(fp[a] | fp[b]) for a, b in pairs]
        return np.mean(vs) if vs else float("nan")

    within = {
        fam: avg_pairs(
            [
                (a, b)
                for a, b in combinations(ORDER, 2)
                if FAMILY[a] == fam and FAMILY[b] == fam
            ]
        )
        for fam in ("ours", "evo2")
    }
    cross = avg_pairs(
        [(a, b) for a, b in combinations(ORDER, 2) if FAMILY[a] != FAMILY[b]]
    )
    print(
        f"\n  within-family avg Jaccard: ours={within['ours']:.3f}  evo2={within['evo2']:.3f}"
    )
    print(f"  cross-family avg Jaccard:  {cross:.3f}")
    print(f"  random expectation (two 10% sets): ~{0.10 / (2 - 0.10):.3f}")

    # universal FP: how many models flag each negative
    neg["n_flag"] = [sum(i in fp[m] for m in ORDER) for i in neg.index]
    print(
        "\n=== universal-FP distribution (# of 7 models putting a negative in their top-10%) ==="
    )
    vc = neg["n_flag"].value_counts().sort_index()
    for nf, cnt in vc.items():
        print(f"  flagged by {nf} models: {cnt:4d} negatives")
    print(
        f"\n  negatives flagged by >=5/7 models (intrinsically-hard conserved-benigns): {(neg['n_flag'] >= 5).sum()}"
    )

    # characterize universal vs idiosyncratic
    print("\n=== characterization by # models flagging ===")
    print(
        f"    {'group':>18} {'n':>5} {'med phyloP':>10} {'med ageMYA':>10} {'med popmax':>10} {'med AlphaMis':>12}"
    )
    for lab, mask in [
        ("flagged 0", neg.n_flag == 0),
        ("flagged 1-2", neg.n_flag.between(1, 2)),
        ("flagged 3-4", neg.n_flag.between(3, 4)),
        ("flagged >=5", neg.n_flag >= 5),
    ]:
        s = neg[mask]
        print(
            f"    {lab:>18} {len(s):>5} {s['phyloP'].median():10.2f} {s['age_mya'].median():10.0f} "
            f"{s['popmax'].median():10.4f} {s['alphamissense'].median():12.3f}"
        )

    fig_jaccard(J)
    fig_universal(neg)
    print("\nDone.")


def fig_jaccard(J: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    M = J.astype(float).values
    im = ax.imshow(
        M,
        cmap="viridis",
        vmin=0,
        vmax=max(0.2, np.nanmax(M[~np.eye(len(M), dtype=bool)])),
    )
    ax.set_xticks(range(len(ORDER)))
    ax.set_yticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ORDER, fontsize=8)
    for i in range(len(ORDER)):
        for j in range(len(ORDER)):
            ax.text(
                j,
                i,
                f"{M[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if M[i, j] < 0.3 else "black",
            )
    fig.colorbar(im, ax=ax, label="FP-set Jaccard")
    ax.set_title(
        "Cross-gLM missense FP-set overlap (top-10% negatives)\nlow off-diagonal = different models fail on different variants"
    )
    _save(fig, "crossglm_fp_jaccard")


def fig_universal(neg: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    vc = neg["n_flag"].value_counts().sort_index()
    axes[0].bar(vc.index, vc.values, color="slateblue")
    axes[0].set_xlabel("# of 7 gLMs flagging the negative (top-10%)")
    axes[0].set_ylabel("# negatives")
    axes[0].set_title("How many models over-call each benign?")
    for x, y in zip(vc.index, vc.values):
        axes[0].text(x, y, str(y), ha="center", va="bottom", fontsize=7)
    # phyloP by n_flag
    groups = [
        neg[neg.n_flag == nf]["phyloP"].dropna().values
        for nf in sorted(neg.n_flag.unique())
    ]
    axes[1].boxplot(
        groups,
        showfliers=False,
        positions=sorted(neg.n_flag.unique()),
        widths=0.6,
        patch_artist=True,
    )
    axes[1].set_xlabel("# of 7 gLMs flagging the negative")
    axes[1].set_ylabel("phyloP_241m")
    axes[1].set_title("More-universally-flagged benigns are more conserved")
    axes[1].grid(alpha=0.3, axis="y")
    _save(fig, "crossglm_universal_fp")


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / name}.{{png,svg}}")


if __name__ == "__main__":
    main()
