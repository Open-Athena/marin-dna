"""GPN-Star variant-effect scoring on the matched-pair eval datasets.

GPN-Star scoring runs in a separate repo
([songlab-cal/TraitGym](https://github.com/songlab-cal/TraitGym)), not here.
The producer publishes prediction parquets to a gist (keyed by
``(chrom, pos, ref, alt, split)`` with 4 score columns: ``llr``, ``abs_llr``,
``llr_calibrated``, ``abs_llr_calibrated``). This module is the thin
load + align layer used by ``snakemake/gpn_star_eval/`` to merge those
predictions with the HF eval dataset and derive the leaderboard-convention
``minus_*`` columns.

Two upstream protocol notes (relevant when comparing against other
leaderboard rows):

- **Calibration.** ``*_calibrated`` columns are pentanucleotide-context
  background subtracted by the producer:
  ``llr_calibrated = llr − E[llr | 5-mer, mut]`` and similarly for ``|llr|``.
  Definition pinned in
  https://github.com/Open-Athena/marin-dna/issues/145#issuecomment-4444680280.
- **Reverse-complement averaging.** GPN-Star averages predictions over
  forward + reverse-complement strands. The other gLM rows in this repo's
  leaderboards (``exp*``) are forward-only — RC averaging is a planned
  addition.
"""

from __future__ import annotations

import pandas as pd

GPN_STAR_MODELS: tuple[str, ...] = ("V", "M", "P")

# Per-variant model metadata: (filename suffix used by the producer,
# human-readable MSA description). The suffix matches the parquet filename
# convention in the source comment.
GPN_STAR_MODEL_INFO: dict[str, tuple[str, str]] = {
    "V": ("v100-200m", "vertebrate, 100-way MSA"),
    "M": ("m447-200m", "mammal, 447-way MSA"),
    "P": ("p243-200m", "primate, 243-way MSA"),
}

# Prediction parquets uploaded by the producer to a gist; pinned commit so the
# raw URL is stable. Replace if the producer re-uploads.
#
# Pinned to the **current-revision** upload (issue #145 comment
# https://github.com/Open-Athena/marin-dna/issues/145#issuecomment-4489509362):
# row-aligned to evals_mendelian_traits@4aed58e / evals_complex_traits@22f86a89
# (the k=9 / AUPRC revisions this repo's evals_v2 targets). The earlier upload
# (gist facb982f @ 35bfb2f) is the pre-#194 1:1 revision and is NOT compatible
# with the current HF datasets — the alignment assert in score_variants_gpn_star
# would fire. eQTL was retired (#172/#194) and is not in this gist.
GPN_STAR_GIST_BASE: str = (
    "https://gist.githubusercontent.com/gonzalobenegas/"
    "db282f89aa00244fbb7437dce0f069ef/raw/"
    "02484d50d9bfd80337e313652b26f98a9362b6b1"
)

# Per-dataset leaderboard score column. The calibrated variant is what each
# leaderboard issue (#161 / #162) renders. The uncalibrated counterpart is
# reported in #145 for context. (eQTL #172 retired in #194 — no current upload.)
GPN_STAR_SCORE_COLUMN: dict[str, str] = {
    "mendelian_traits": "minus_llr_calibrated",
    "complex_traits": "abs_llr_calibrated",
}


def predictions_url(dataset: str, model: str) -> str:
    """Return the gist raw URL for one GPN-Star prediction parquet.

    The current-revision upload names files ``bolinas_{dataset}_GPN-Star-{model}``
    (underscore separator); the earlier 1:1-revision upload used a ``.`` before
    ``GPN-Star``. See ``GPN_STAR_GIST_BASE``.
    """
    assert model in GPN_STAR_MODELS, f"unknown GPN-Star model {model!r}"
    return f"{GPN_STAR_GIST_BASE}/bolinas_{dataset}_GPN-Star-{model}.parquet"


def score_variants_gpn_star(
    hf_df: pd.DataFrame,
    predictions: pd.DataFrame,
    split: str = "train",
) -> pd.DataFrame:
    """Align GPN-Star predictions with an HF eval dataset, row-by-row.

    The producer (TraitGym ``bolinas_pack_predictions``) builds the prediction
    parquet by horizontal-concatenating ``[chrom, pos, ref, alt, split]``
    pulled from the HF dataset's ``train.parquet + test.parquet`` with the
    GPN-Star feature parquet. So filtering ``predictions`` to one split
    yields the same row order as ``load_dataset(..., split=split).to_pandas()``
    — **no key-based merge is needed**, just an element-wise alignment assert.
    If the assert fires, something has changed upstream and a silent merge
    would mask it.

    Args:
        hf_df: HF eval dataset's train (or test) split as a pandas DataFrame.
            Must include ``[chrom, pos, ref, alt]``.
        predictions: Full GPN-Star prediction parquet (train + test rows).
            Must include ``[chrom, pos, ref, alt, split, llr, abs_llr,
            llr_calibrated, abs_llr_calibrated]``.
        split: Which split of ``predictions`` to filter to. Default ``train``.

    Returns:
        DataFrame row-aligned with ``hf_df`` (same index order), columns
        ``[llr, abs_llr, llr_calibrated, abs_llr_calibrated, minus_llr,
        minus_llr_calibrated]``. The ``minus_*`` columns are ``-llr`` /
        ``-llr_calibrated`` so positives (pathogenic / direction-of-effect)
        score higher than negatives — the leaderboard convention for the
        Mendelian benchmark. For Complex / eQTL the ``abs_*`` columns are
        used directly.

    Raises:
        AssertionError on row-count mismatch, key-order mismatch, missing
        columns, or NaN in any score column.
    """
    for col in ("chrom", "pos", "ref", "alt"):
        assert col in hf_df.columns, f"hf_df missing column {col!r}"
    for col in (
        "chrom",
        "pos",
        "ref",
        "alt",
        "split",
        "llr",
        "abs_llr",
        "llr_calibrated",
        "abs_llr_calibrated",
    ):
        assert col in predictions.columns, f"predictions missing column {col!r}"

    pred_split = predictions[predictions["split"] == split].reset_index(drop=True)
    hf_df = hf_df.reset_index(drop=True)

    assert len(hf_df) == len(pred_split), (
        f"row count mismatch: hf_df={len(hf_df)} vs "
        f"predictions[{split}]={len(pred_split)}"
    )

    key_cols = ["chrom", "pos", "ref", "alt"]
    hf_keys = hf_df[key_cols].copy()
    hf_keys["chrom"] = hf_keys["chrom"].astype(str)
    pred_keys = pred_split[key_cols].copy()
    pred_keys["chrom"] = pred_keys["chrom"].astype(str)
    eq = (hf_keys == pred_keys).all(axis=1)
    if not eq.all():
        bad_idx = int((~eq).idxmax())
        raise AssertionError(
            f"row alignment broken at index {bad_idx}:\n"
            f"  hf:   {hf_keys.iloc[bad_idx].to_dict()}\n"
            f"  pred: {pred_keys.iloc[bad_idx].to_dict()}"
        )

    raw_cols = ["llr", "abs_llr", "llr_calibrated", "abs_llr_calibrated"]
    scores = pred_split[raw_cols].reset_index(drop=True).copy()
    scores["minus_llr"] = -scores["llr"]
    scores["minus_llr_calibrated"] = -scores["llr_calibrated"]

    for col in scores.columns:
        n_nan = int(scores[col].isna().sum())
        assert n_nan == 0, f"score column {col!r} has {n_nan} NaN values"

    return scores
