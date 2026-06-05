"""Diff per-variant raw LLRs: online (levanter, GCS dump) vs offline (kernel,
S3 scores parquet). Localizes WHICH variants diverge (esp. distal/fwd) and
whether it's a few outliers or a systematic shift.

online dump rows: {llr, label, subset, chrom, pos, ref, alt, match_group, strand}
  strand '+' -> compare to offline llr_fwd ; '-' -> offline llr_rc
"""

from __future__ import annotations

import json
import subprocess

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

GCS = "gs://marin-us-east5/tmp/issue257/online_llrs_dump2.jsonl"
LOCAL = "scratch/issue257/online_llrs_dump2.jsonl"
OFF = (
    "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores/"
    "exp232-v4_ccre_non_promoter-step-4999/mendelian_traits.parquet"
)

subprocess.run(["gcloud", "storage", "cp", GCS, LOCAL], check=True)
on = pl.DataFrame([json.loads(line) for line in open(LOCAL) if line.strip()])
print(
    f"online rows: {on.shape}  strands={on['strand'].unique().to_list()}  "
    f"subsets={on['subset'].n_unique()}"
)

off = pl.read_parquet(OFF)
# offline long form: one (chrom,pos,ref,alt,strand,off_llr) per strand
off_long = pl.concat(
    [
        off.select(
            [
                "chrom",
                "pos",
                "ref",
                "alt",
                "subset",
                "label",
                pl.col("llr_fwd").alias("off_llr"),
                pl.lit("+").alias("strand"),
            ]
        ),
        off.select(
            [
                "chrom",
                "pos",
                "ref",
                "alt",
                "subset",
                "label",
                pl.col("llr_rc").alias("off_llr"),
                pl.lit("-").alias("strand"),
            ]
        ),
    ]
)
on = on.with_columns(pl.col("pos").cast(pl.Int64))
m = on.join(
    off_long.select(["chrom", "pos", "ref", "alt", "strand", "off_llr"]),
    on=["chrom", "pos", "ref", "alt", "strand"],
    how="inner",
)
m = m.with_columns((pl.col("llr") - pl.col("off_llr")).alias("d"))
print(f"merged rows: {m.shape[0]} (expect ~{on.shape[0]})")

print("\n=== per (subset,strand): corr + mean|online-offline| LLR ===")
print(f"{'subset':<30}{'strand':<5}{'n':>5}{'corr':>8}{'mean|d|':>9}{'max|d|':>8}")
for (sub, st), g in sorted(
    m.group_by(["subset", "strand"]), key=lambda x: (x[0][0], x[0][1])
):
    a, b = g["llr"].to_numpy(), g["off_llr"].to_numpy()
    c = np.corrcoef(a, b)[0, 1] if len(a) > 2 else float("nan")
    print(
        f"{sub:<30}{st:<5}{len(a):>5}{c:>8.4f}{np.abs(a - b).mean():>9.4f}{np.abs(a - b).max():>8.3f}"
    )

# --- distal/fwd focus ---
d = m.filter((pl.col("subset") == "distal") & (pl.col("strand") == "+")).sort(
    "d", descending=True
)
lab = d["label"].to_numpy()
print(f"\n=== distal/fwd: n={len(d)} pos={int(lab.sum())} ===")
print(
    f"AUPRC online(-llr)={average_precision_score(lab, -d['llr'].to_numpy()):.4f}  "
    f"offline(-off_llr)={average_precision_score(lab, -d['off_llr'].to_numpy()):.4f}"
)
print(
    f"mean d (online-offline) over distal/fwd: {d['d'].mean():.4f} ; "
    f"pos-only mean d: {d.filter(pl.col('label') == 1)['d'].mean():.4f} ; "
    f"neg-only mean d: {d.filter(pl.col('label') == 0)['d'].mean():.4f}"
)
print("\ntop 12 |online-offline| distal/fwd variants:")
top = d.with_columns(pl.col("d").abs().alias("ad")).sort("ad", descending=True).head(12)
for r in top.iter_rows(named=True):
    print(
        f"  {r['chrom']}:{r['pos']} {r['ref']}>{r['alt']} label={r['label']} "
        f"online={r['llr']:+.3f} offline={r['off_llr']:+.3f} d={r['d']:+.3f}"
    )
