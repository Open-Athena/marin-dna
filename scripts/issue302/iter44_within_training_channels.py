"""issue #302 — iteration 44: which of the two plausibility channels drives the WITHIN-TRAINING
degradation? The param-scale recruitment is driven by BOTH conservation (iter41/39) and amino-acid
severity (iter42), synergizing (iter43). Does the #232 0.25B v4_cds within-training reversal
(step 4000 peak -> 4999) show the same two channels, or only one?

Result: within-training the CONSERVATION channel is strong (phyloP_241m β=+0.128, z=8.5) but the
AA-SEVERITY channel is ABSENT (grantham β=+0.014, z=1.0, ns) and so is the interaction (z=1.4, ns).
So the small CDS specialist degrades via conservation ALONE. This explains iter34's depth difference
and confirms iter42's finding that the AA-severity channel is SCALE-EMERGENT (turns on past ~128M);
the 0.25B specialist is below that, so it lacks the protein-severity over-weighting the big models
develop.

Recruitment Δ = z(mll@step4999) − z(mll@step4000). Grantham from cached aa_ref/aa_alt. CPU; S3.
Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter44_within_training_channels.py
"""

from __future__ import annotations

import numpy as np
import polars as pl

SC = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
CB = "s3://oa-bolinas/snakemake/conservation_eval/results/mendelian_traits"
KEY = ["chrom", "pos", "ref", "alt"]
GP = {
    "A": (0, 8.1, 31),
    "R": (0.65, 10.5, 124),
    "N": (1.33, 11.6, 56),
    "D": (1.38, 13, 54),
    "C": (2.75, 5.5, 55),
    "Q": (0.89, 10.5, 85),
    "E": (0.92, 12.3, 83),
    "G": (0.74, 9, 3),
    "H": (0.58, 10.4, 96),
    "I": (0, 5.2, 111),
    "L": (0, 4.9, 111),
    "K": (0.33, 11.3, 119),
    "M": (0, 5.7, 105),
    "F": (0, 5, 132),
    "P": (0.39, 8, 32.5),
    "S": (1.42, 9.2, 32),
    "T": (0.71, 8.6, 61),
    "W": (0.13, 5.4, 170),
    "Y": (0.2, 6.2, 136),
    "V": (0, 5.9, 84),
}


def gr(a, b):
    if a not in GP or b not in GP or a == b:
        return 0.0 if a in GP and b in GP else None
    (ca, pa, va), (cb, pb, vb) = GP[a], GP[b]
    return 50.723 * np.sqrt(
        1.833 * (ca - cb) ** 2 + 0.1018 * (pa - pb) ** 2 + 0.000399 * (va - vb) ** 2
    )


def trk(n):
    return (
        pl.read_parquet(f"{CB}/{n}_train.parquet")
        .with_columns(pl.col("chrom").cast(str))
        .unique(subset=KEY)
        .select([*KEY, pl.col("score").alias(n)])
    )


def z(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def loadc(step):
    return (
        pl.read_parquet(f"{SC}/exp232-v4_cds-step-{step}/mendelian_traits.parquet")
        .filter(pl.col("subset") == "missense_variant")
        .with_columns(
            pl.col("chrom").cast(str),
            (-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("mll"),
        )
        .select([*KEY, "label", "mll"])
    )


def main() -> None:
    aa = (
        pl.read_parquet("scratch/issue302/myvariant_aa.parquet")
        .with_columns(pl.col("chrom").cast(str))
        .drop_nulls(["aa_ref", "aa_alt"])
    )
    aa = (
        aa.with_columns(
            pl.Series(
                "grantham", [gr(r, a) for r, a in zip(aa["aa_ref"], aa["aa_alt"])]
            )
        )
        .drop_nulls("grantham")
        .select([*KEY, "grantham"])
    )
    a = loadc(4000).with_columns(
        ((pl.col("mll") - pl.col("mll").mean()) / pl.col("mll").std()).alias("zpk")
    )
    b = (
        loadc(4999)
        .with_columns(
            ((pl.col("mll") - pl.col("mll").mean()) / pl.col("mll").std()).alias("zend")
        )
        .select([*KEY, "zend"])
    )
    m = (
        a.join(b, on=KEY, how="inner")
        .join(aa, on=KEY, how="left")
        .join(trk("phyloP_241m"), on=KEY, how="left")
        .join(trk("phyloP_100v"), on=KEY, how="left")
        .filter(pl.col("label") == 0)
        .drop_nulls(["grantham", "phyloP_241m", "phyloP_100v", "zpk", "zend"])
    )
    n = m.height
    d = z((m["zend"] - m["zpk"]).to_numpy())
    P, G = z(m["phyloP_241m"].to_numpy()), z(m["grantham"].to_numpy())
    S = z(z(m["phyloP_241m"].to_numpy()) - z(m["phyloP_100v"].to_numpy()))
    X = np.column_stack([np.ones(n), P, G, S, P * G])
    coef, *_ = np.linalg.lstsq(X, d, rcond=None)
    res = d - X @ coef
    se = np.sqrt(np.diag(((res @ res) / (n - X.shape[1])) * np.linalg.inv(X.T @ X)))
    print(
        f"WITHIN-TRAINING v4_cds recruitment (step 4000→4999), benign missense n={n}:"
    )
    for nm, c, s in zip(
        [
            "intercept",
            "phyloP_241m (conservation)",
            "grantham (AA-severity)",
            "mammal-specificity",
            "phyloP×grantham",
        ],
        coef,
        se,
    ):
        print(f"  {nm:<26} β={c:+.3f} z={c / s:+.1f}{'  *' if abs(c / s) > 2 else ''}")


if __name__ == "__main__":
    main()
