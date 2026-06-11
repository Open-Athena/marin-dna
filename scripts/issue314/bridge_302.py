"""issue #314 — bridge check: reproduce #302's frozen-embedding probe.

Validates the new ``variant_probe`` harness against a *known* #302 result before
we trust any new pooling axis: the last-layer ``delta`` probe AUPRC on Mendelian
missense for ``scaling-v0.5-1B``. #302 iter9 (match_group-grouped 5-fold,
StandardScaler -> PCA(256) -> LogisticRegression(C=0.5)) reported **0.477**; iter10
re-confirmed under chromosome-grouped CV. We read #302's *existing* pooled
(center-100, FWD+RC-averaged) embedding cache from S3 and re-run the probe through
``variant_probe``. CPU-only.

Run:  uv run --group genome-s3 python scripts/issue314/bridge_302.py
"""

import io

import numpy as np
import polars as pl
import s3fs
from sklearn.metrics import average_precision_score

from marin_dna.pipelines.evals.variant_probe import chrom_grouped_oof, probe_auprc

EMB_S3 = "s3://oa-bolinas/analysis/issue302/embeddings"
SCORES_S3 = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
NAME = "scaling-v0.5-1B"
SCORES_DIR = "scaling-v0.5-h1920-p1B-step-215573"
KEY = ["chrom", "pos", "ref", "alt"]


def load_npz(fs: s3fs.S3FileSystem, name: str) -> dict[str, np.ndarray]:
    with fs.open(f"{EMB_S3}/{name}.npz") as f:
        d = np.load(io.BytesIO(f.read()))
        return {k: d[k] for k in d.files}


def main() -> None:
    fs = s3fs.S3FileSystem()
    emb = load_npz(fs, NAME)
    ref = emb["ref"].astype(np.float32)  # [N, n_layers, D]
    alt = emb["alt"].astype(np.float32)
    keys = pl.read_parquet(f"{EMB_S3}/{NAME}.keys.parquet").with_columns(
        pl.col("chrom").cast(str)
    )

    # Defensive: restrict to missense (the #302 cache is missense-only, so this
    # is a no-op, but it pins the comparison even if the cache ever widens).
    mask = (keys["subset"] == "missense_variant").to_numpy()
    ref, alt = ref[mask], alt[mask]
    keys = keys.filter(pl.col("subset") == "missense_variant")
    y = keys["label"].to_numpy().astype(int)
    mg = keys["match_group"].to_numpy()
    chrom = keys["chrom"].to_numpy()
    print(
        f"{NAME}: N={len(y)} D={ref.shape[2]} n_layers={ref.shape[1]} "
        f"n_pos={int(y.sum())} n_chrom={len(np.unique(chrom))} "
        f"n_match_groups={len(np.unique(mg))}"
    )

    delta = alt[:, -1] - ref[:, -1]  # last layer

    # #302 recipe: StandardScaler -> PCA(256) -> LogisticRegression(C=0.5), GroupKFold(5).
    common = dict(loss="logistic", c=0.5, n_pca=256, standardize=True, n_splits=5)
    oof_mg = chrom_grouped_oof(delta, y, mg, **common)
    oof_ch = chrom_grouped_oof(delta, y, chrom, **common)
    ap_mg = average_precision_score(y, oof_mg)
    ap_ch = average_precision_score(y, oof_ch)

    # Zero-shot LLR baseline (minus_llr_avg) on the same missense variants.
    sc = (
        pl.read_parquet(f"{SCORES_S3}/{SCORES_DIR}/mendelian_traits.parquet")
        .with_columns(pl.col("chrom").cast(str))
        .filter(pl.col("subset") == "missense_variant")
        .with_columns((-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("llr"))
        .select([*KEY, "llr"])
    )
    llr = keys.join(sc, on=KEY, how="left")["llr"].to_numpy()
    assert not np.isnan(llr).any(), f"{int(np.isnan(llr).sum())} keys missing an LLR"
    ap_llr = average_precision_score(y, llr)

    boot = probe_auprc(y, oof_ch, mg)
    print(f"\nMendelian missense AUPRC ({NAME}, last layer):")
    print(f"  zero-shot LLR (minus_llr_avg) : {ap_llr:.3f}")
    print(f"  probe delta, match_group-CV   : {ap_mg:.3f}   (#302 iter9 target ~0.477)")
    print(f"  probe delta, chrom-CV         : {ap_ch:.3f}   (#302 iter10, leak-proof)")
    print(f"  probe delta, chrom-CV ± SE    : {boot['value']:.3f} ± {boot['se']:.3f}")

    assert abs(ap_mg - 0.477) < 0.03, (
        f"BRIDGE FAILED: match_group-CV AUPRC {ap_mg:.3f} != #302's 0.477 (±0.03)"
    )
    print(
        "\n✅ bridge OK — variant_probe reproduces #302's match_group-CV probe AUPRC."
    )


if __name__ == "__main__":
    main()
