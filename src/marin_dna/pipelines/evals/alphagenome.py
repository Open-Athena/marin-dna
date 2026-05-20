"""AlphaGenome variant-effect scoring on the matched-pair eval datasets.

Forward-strand only — no reverse-complement averaging (TraitGym averages with
RC; we skip it to halve API calls). Returns per-track L2_DIFF_LOG1P scores in a
wide DataFrame, one row per input variant.

The PyPI ``alphagenome`` package is gated behind the ``alphagenome-eval`` dep
group so the rest of the repo installs without it.
"""

from __future__ import annotations

import concurrent.futures
import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from alphagenome.models import variant_scorers as _vs


# Each scorer requests one assay's output; a single API call returns scores for
# every track AlphaGenome predicts under that assay.
ALPHAGENOME_TRACKS: tuple[str, ...] = (
    "ATAC",
    "DNASE",
    "CHIP_TF",
    "CHIP_HISTONE",
    "CAGE",
    "PROCAP",
    "RNA_SEQ",
)

# 1 MB context = 500 kb each side. Resolved via
# ``dna_client.SUPPORTED_SEQUENCE_LENGTHS[f"SEQUENCE_LENGTH_{SEQUENCE_LENGTH}"]``.
SEQUENCE_LENGTH: str = "1MB"


def make_scorers() -> tuple[list["_vs.CenterMaskScorer"], dict[str, str]]:
    """Build the 7 CenterMaskScorer(width=None, L2_DIFF_LOG1P) scorers.

    Returns ``(scorers, scorer_repr_to_assay)`` — the second element maps
    ``str(scorer)`` back to its assay name so we can reverse-map AlphaGenome's
    ``tidy_scores`` output (which keys by scorer repr) to assay names.
    """
    from alphagenome.models import dna_client, variant_scorers

    scorers = [
        variant_scorers.CenterMaskScorer(
            requested_output=getattr(dna_client.OutputType, track),
            width=None,
            aggregation_type=variant_scorers.AggregationType.L2_DIFF_LOG1P,
        )
        for track in ALPHAGENOME_TRACKS
    ]
    scorer_repr_to_assay = {str(s): t for s, t in zip(scorers, ALPHAGENOME_TRACKS)}
    return scorers, scorer_repr_to_assay


def parse_score_response(
    tidy: pd.DataFrame,
    scorer_repr_to_assay: dict[str, str],
) -> pd.DataFrame:
    """Convert a single-variant ``tidy_scores`` DataFrame to a wide 1-row table.

    Input ``tidy`` (as returned by ``variant_scorers.tidy_scores([scores])``):
    long form with a ``variant_scorer`` column (string repr matching the keys
    of ``scorer_repr_to_assay``) and a ``raw_score`` column. Each (assay, cell
    type) is one row; the same scorer repr appears multiple times within an
    assay.

    Output: 1-row DataFrame, columns = ``"{assay}_{idx}"`` (idx = position
    within assay, 0-indexed), values = raw scores. Underscore (not hyphen) so
    column names are pandas-query-friendly.
    """
    assert "variant_scorer" in tidy.columns and "raw_score" in tidy.columns, (
        f"unexpected tidy_scores columns: {tidy.columns.tolist()}"
    )
    res = tidy.copy()
    res["assay"] = res["variant_scorer"].map(scorer_repr_to_assay)
    assert res["assay"].notna().all(), (
        "tidy_scores contains scorer reprs not in scorer_repr_to_assay; "
        "API may have returned tracks we didn't request"
    )
    # Sequential per-assay index for column naming.
    res["track"] = res["assay"] + "_" + res.groupby("assay").cumcount().astype(str)
    out = res.set_index("track")[["raw_score"]].T
    out = out.reset_index(drop=True)
    return out


def score_variants_alphagenome(
    V: pd.DataFrame,
    num_workers: int = 4,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Score variants with AlphaGenome's per-track L2_DIFF_LOG1P aggregation.

    Forward-strand only. Single API call per variant.

    Parameters
    ----------
    V
        DataFrame with columns ``chrom`` (unprefixed, e.g. ``"1"``, ``"X"``),
        ``pos`` (1-based, VCF convention), ``ref``, ``alt``. Other columns are
        ignored. Order is preserved in the output.
    num_workers
        Threads in the executor. AlphaGenome's API rate-limits — keep low
        (4 is a safe ceiling).
    api_key
        AlphaGenome API key. Defaults to ``os.environ["ALPHA_GENOME_API_KEY"]``.

    Returns
    -------
    pd.DataFrame
        One row per input variant (same order), columns = ``"{assay}_{idx}"``
        track names with raw L2_DIFF_LOG1P scores.
    """
    from alphagenome.data import genome
    from alphagenome.models import dna_client, variant_scorers

    if api_key is None:
        api_key = os.environ.get("ALPHA_GENOME_API_KEY")
    assert api_key, "ALPHA_GENOME_API_KEY not set; pass api_key= or export the env var"

    required_cols = {"chrom", "pos", "ref", "alt"}
    missing = required_cols - set(V.columns)
    assert not missing, f"V missing required columns: {missing}"

    model = dna_client.create(api_key)
    sequence_length = dna_client.SUPPORTED_SEQUENCE_LENGTHS[
        f"SEQUENCE_LENGTH_{SEQUENCE_LENGTH}"
    ]
    organism = dna_client.Organism.HOMO_SAPIENS
    scorers, scorer_repr_to_assay = make_scorers()

    def score_one(row) -> tuple[np.ndarray, list[str]]:
        # AlphaGenome expects a "chr"-prefixed chromosome string.
        chrom = row.chrom if str(row.chrom).startswith("chr") else f"chr{row.chrom}"
        variant = genome.Variant(
            chromosome=chrom,
            position=int(row.pos),
            reference_bases=row.ref,
            alternate_bases=row.alt,
        )
        interval = variant.reference_interval.resize(sequence_length).copy()
        # Default strand is "."; AlphaGenome reads it as unstranded and we
        # specifically want forward-strand to match TraitGym's call.
        interval.strand = "+"

        scores = model.score_variant(
            interval=interval,
            variant=variant,
            organism=organism,
            variant_scorers=scorers,
        )
        tidy = variant_scorers.tidy_scores([scores])
        parsed = parse_score_response(tidy, scorer_repr_to_assay)
        # Detach values from the SDK / pandas objects so each iteration's
        # footprint is just a small numpy array. Without the detachment, the
        # earlier list-of-DataFrames implementation accumulated ~3-4 MB of
        # retained references per variant and OOMed on >10K-variant runs.
        arr = parsed.to_numpy(dtype=np.float32, copy=True).ravel()
        return arr, list(parsed.columns)

    buf: np.ndarray | None = None
    columns: list[str] | None = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
        for i, (arr, cols) in enumerate(
            tqdm(
                ex.map(score_one, V.itertuples(index=False)),
                total=len(V),
            )
        ):
            if buf is None:
                columns = cols
                buf = np.empty((len(V), len(cols)), dtype=np.float32)
            buf[i] = arr

    return pd.DataFrame(buf, columns=columns)
