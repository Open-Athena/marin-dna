"""Apply development-fitted linear probes to held-out Mendelian variants."""

import argparse
from hashlib import file_digest
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from marin_dna_evals.variant_probe import pair_feature_from_bundle

VARIANT_KEY = ["chrom", "pos", "ref", "alt"]
IDENTITY = [*VARIANT_KEY, "subset"]
HEADLINE = (
    "missense_variant",
    "splicing",
    "5_prime_UTR_variant",
    "tss_proximal",
    "non_coding_transcript_exon_variant",
    "3_prime_UTR_variant",
    "distal",
    "synonymous_variant",
)
EXCLUDED = "mature_miRNA_variant"
CLASSIFIER_SHA256 = {
    "marin": "6693ef8c8126ca4182c2b61d6e1ef001f2ae51ac2eb04c5b19b4961fa4cb0cc5",
    "evo2_40b": "a73c1799047bf7a22c63bce9b429af93a73cd20b968c640cc5393fff2cb4193f",
}
TEST_SHA256 = "658dd3c49b2c171d7c5c973e014dd3d2e8d425e25187eb30513df64d9ef58883"


def check_digest(path: Path, expected: str) -> None:
    with path.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError(f"Unexpected file hash for {path}")


def predict_batch(rows: pd.DataFrame, classifiers: dict) -> np.ndarray:
    """Preserve row order, skipping only the predeclared mature-miRNA subset."""
    if set(classifiers) != set(HEADLINE):
        raise ValueError(
            "Expected one development-fitted classifier for each headline class"
        )
    if not rows["subset"].isin((*HEADLINE, EXCLUDED)).all():
        raise ValueError("Unexpected Mendelian subset")
    predictions = np.full(len(rows), np.nan)
    for subset, positions in rows.groupby(
        "subset", sort=False, observed=True
    ).indices.items():
        if subset == EXCLUDED:
            continue
        classifier = classifiers[subset]
        pipeline = classifier["pipeline"]
        if classifier["feature"] != "concat_ref_delta" or not np.array_equal(
            pipeline.classes_, [0, 1]
        ):
            raise ValueError(f"Unexpected feature or class coding for {subset}")
        features = pair_feature_from_bundle(rows.iloc[positions], classifier["feature"])
        probabilities = pipeline.predict_proba(features)[:, 1]
        if not np.isfinite(probabilities).all() or np.any(
            (probabilities < 0) | (probabilities > 1)
        ):
            raise ValueError(f"Invalid probe probability in {subset}")
        predictions[positions] = probabilities
    return predictions


def make_predictions(
    scores: Path, model: str, classifiers: dict, variants: pd.DataFrame
) -> pd.DataFrame:
    """Use only model outputs and variant identities; join the release by key."""
    llr_columns = ["minus_llr"] if model == "evo2_40b" else ["llr_fwd", "llr_rc"]
    columns = [*IDENTITY, *llr_columns, "emb_ref", "emb_alt"]
    parts = []
    for batch in pq.ParquetFile(scores).iter_batches(batch_size=256, columns=columns):
        rows = batch.to_pandas()
        out = rows[IDENTITY].copy()
        out["minus_llr_avg"] = (
            rows["minus_llr"]
            if model == "evo2_40b"
            else -(rows["llr_fwd"] + rows["llr_rc"]) / 2
        )
        if not np.isfinite(out["minus_llr_avg"]).all():
            raise ValueError("Nonfinite zero-shot score")
        out["probe_score"] = predict_batch(rows, classifiers)
        parts.append(out)
    predictions = pd.concat(parts, ignore_index=True)
    if (
        predictions.duplicated(VARIANT_KEY).any()
        or variants.duplicated(VARIANT_KEY).any()
    ):
        raise ValueError("Duplicate variant key")
    joined = variants[IDENTITY].merge(
        predictions, on=IDENTITY, how="left", validate="one_to_one"
    )
    if len(predictions) != len(variants) or joined["minus_llr_avg"].isna().any():
        raise ValueError(
            "Score bundle differs from the supplied release variants or classes"
        )
    return joined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=CLASSIFIER_SHA256)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--classifiers", required=True, type=Path)
    parser.add_argument("--test-release", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve() in {
        p.resolve() for p in (args.scores, args.classifiers, args.test_release)
    }:
        parser.error("choose an output path distinct from the inputs")
    check_digest(args.test_release, TEST_SHA256)
    check_digest(args.classifiers, CLASSIFIER_SHA256[args.model])
    variants = pd.read_parquet(args.test_release, columns=IDENTITY)
    predictions = make_predictions(
        args.scores, args.model, joblib.load(args.classifiers), variants
    )
    predictions["model"] = args.model
    predictions["split"] = "test"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=args.output.parent, suffix=".parquet", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        predictions.to_parquet(temporary, index=False)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"{args.model}: {len(predictions)} test variants, {predictions.probe_score.count()} probe scores"
    )


if __name__ == "__main__":
    main()
