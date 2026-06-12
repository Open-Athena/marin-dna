"""Stage the #313 dsQTL train split to S3 as a plain variant parquet for the
issue-314 embedding extractor.

dsQTL (accessibility QTL) is condition-specific chromatin accessibility — orthogonal
to the fitness/conservation signal a gLM likelihood captures — so it's a clean test
of what's *only* linearly decodable from the frozen embedding. We deliberately do NOT
route it through a gLM score parquet (there is no legitimate gLM zero-shot scoring of
QTL; the stale `exp136-proj_v30` caqtl/dsqtl scores were deleted from S3). Instead we
load the standardized dataset (pinned revision = the evals_v2 config pin) and write its
train split — already all train chromosomes (odd+X) by the #313 builder — so the
extractor's existing ``--variants_parquet`` path consumes it with no HF auth on the GPU
node and no code change.
"""

import polars as pl
from datasets import load_dataset

HF_REPO = "bolinas-dna/evals_dsqtl"
REVISION = "b7e02a07beb831c7047286aacd3ddfd299d6f88f"  # evals_v2 config.yaml dsqtl pin
OUT = "s3://oa-bolinas/analysis/issue314/datasets/dsqtl_train.parquet"
TRAIN_CHROMS = tuple(str(i) for i in range(1, 23, 2)) + ("X",)


def main() -> None:
    df = load_dataset(HF_REPO, revision=REVISION, split="train").to_pandas()
    df["chrom"] = df["chrom"].astype(str)
    for c in ("chrom", "pos", "ref", "alt", "label"):
        assert c in df.columns, f"missing column {c!r}"
    bad = sorted(set(df["chrom"]) - set(TRAIN_CHROMS))
    assert not bad, f"non-train chromosomes present (expected pre-split train): {bad}"
    n_pos = int(df["label"].sum())
    assert 0 < n_pos < len(df), f"degenerate labels: {n_pos}/{len(df)}"
    pl.from_pandas(df).write_parquet(OUT)
    print(f"staged {len(df)} dsQTL train variants ({n_pos} pos) -> {OUT}")


if __name__ == "__main__":
    main()
