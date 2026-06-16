"""AlphaGenome GM12878-DNase LFC predictions for the supervised caQTL/dsQTL
benchmark (#311).

The recommended accessibility scorer (Suppl Table 9): `CenterMaskScorer(DNASE,
width=501, DIFF_LOG2_SUM)`, GM12878-matched — one signed log2 fold-change per variant.
Resumable + S3-checkpointed (`score_dnase_lfc_resumable`): it seeds from
`dnase_lfc.s3_prefix/{ds}.parquet` and scores only missing variants, so for caqtl/dsqtl —
whose genome-native predictions already exist (corrected once by
`scripts/correct_ag_predictions.py`) — it spends **0 API** (guarded by `max_new_calls`).

This rule produces only AlphaGenome's per-variant predictions. The model-agnostic
benchmark *metrics* (ChromBPNet/Enformer carried scores + the shared official metrics +
the leaderboard) are the `scripts/qtl_benchmark.py` driver.
"""

import polars as pl

from marin_dna.pipelines.evals.alphagenome import score_dnase_lfc_resumable


rule score_dnase_lfc:
    output:
        "results/dnase_lfc/{ds}.parquet",
    wildcard_constraints:
        ds=DNASE_LFC_CONSTRAINT,
    threads: DNASE_LFC["num_workers"]
    params:
        hf_path=lambda wc: f"{config['input_hf_prefix']}_{wc.ds}",
        hf_revision=lambda wc: get_dnase_lfc_revision(wc.ds),
        # Resumable checkpoint == the canonical S3 artifact (seed + output location).
        checkpoint=lambda wc: f"{DNASE_LFC['s3_prefix']}/{wc.ds}.parquet",
    run:
        # Canonical genome-oriented variants = train ∪ test of the pinned HF build.
        variants = pl.concat(
            [
                pl.from_pandas(
                    load_dataset(
                        params.hf_path, split=split, revision=params.hf_revision
                    ).to_pandas()
                ).select(["chrom", "pos", "ref", "alt"])
                for split in ("train", "test")
            ]
        )
        out = score_dnase_lfc_resumable(
            variants,
            params.checkpoint,
            num_workers=DNASE_LFC["num_workers"],
            chunk_size=DNASE_LFC["chunk_size"],
            max_new_calls=DNASE_LFC["max_new_calls"],
        )
        out.write_parquet(output[0])
        print(
            f"[alphagenome_eval] dnase_lfc {wildcards.ds}: {out.height} variants "
            f"(genome-native GM12878-DNase LFC)"
        )
