"""issue #302 — iteration 40: is the within-training missense reversal CDS-specific? (Bounds
iter34's confound.) iter34 found the #232 0.25B v4_cds arm's within-training reversal localizes to
DEEPLY-conserved benigns (opposite of the across-scale axis), confounded by CDS-only training. To
disentangle, check whether ANY other #232 region-specialist arm (v4_bg/ccre/ncrna_exon) reverses —
if a fuller-data arm reversed, we could compare its depth localization. Result: only v4_cds
reverses; v4_bg sits at chance (never learns missense); the others rise weakly without reversing.
So the reversal is inseparable from CDS-only training with the available checkpoints — iter34's
confound is irreducible (no full-genome within-training trajectory exists; the ladder is FINAL-only).

CPU; reads S3. Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter40_arm_reversal.py
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score as ap

SC = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
STEPS = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 4999]
ARMS = ["v4_cds", "v4_bg", "v4_ccre_non_promoter", "v4_ncrna_exon"]


def main() -> None:
    print("arm                  | " + " ".join(f"{s:>5}" for s in STEPS) + " | verdict")
    for a in ARMS:
        traj = []
        for s in STEPS:
            try:
                m = (
                    pl.read_parquet(
                        f"{SC}/exp232-{a}-step-{s}/mendelian_traits.parquet"
                    )
                    .filter(pl.col("subset") == "missense_variant")
                    .with_columns(
                        (-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("mll")
                    )
                )
                traj.append(float(ap(m["label"].to_numpy(), m["mll"].to_numpy())))
            except Exception:
                traj.append(float("nan"))
        t = np.array(traj)
        pk = int(np.nanargmax(t))
        last = t[~np.isnan(t)][-1]
        drop = float(np.nanmax(t) - last)
        rev = (
            "REVERSES"
            if (drop > 0.01 and pk < len(t) - 1)
            else ("flat@chance" if np.nanmax(t) < 0.13 else "rises, no reversal")
        )
        print(
            f"{a:<20} | "
            + " ".join(f"{v:.3f}" if not np.isnan(v) else "  -  " for v in t)
            + f" | peak@{STEPS[pk]} drop={drop:+.3f} {rev}"
        )


if __name__ == "__main__":
    main()
