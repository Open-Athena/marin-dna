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
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
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

# Transient gRPC status codes retried per variant. The SDK's `@retry_rpc` on
# `score_variant` retries only {RESOURCE_EXHAUSTED, UNAVAILABLE}; AlphaGenome's
# known server-side "bad machine" outages surface as INTERNAL (not retried by
# default), which would abort a whole dataset. Stored as names and resolved to
# `grpc.StatusCode` at call time so this module imports without grpc installed.
SCORE_VARIANT_RETRY_STATUS: tuple[str, ...] = (
    "INTERNAL",
    "UNAVAILABLE",
    "RESOURCE_EXHAUSTED",
    "DEADLINE_EXCEEDED",
)
SCORE_VARIANT_MAX_ATTEMPTS: int = 10


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
    import grpc
    from alphagenome.data import genome
    from alphagenome.models import dna_client, variant_scorers

    if api_key is None:
        api_key = os.environ.get("ALPHA_GENOME_API_KEY")
    assert api_key, "ALPHA_GENOME_API_KEY not set; pass api_key= or export the env var"

    required_cols = {"chrom", "pos", "ref", "alt"}
    missing = required_cols - set(V.columns)
    assert not missing, f"V missing required columns: {missing}"

    model = dna_client.create(api_key)

    # The SDK decorates `score_variant` with `@retry_rpc`, but its default
    # `retry_status_codes` is only {RESOURCE_EXHAUSTED, UNAVAILABLE} — it does
    # NOT retry INTERNAL, which is exactly how AlphaGenome's known,
    # maintainer-acknowledged "bad machine" backend outages surface
    # (alphagenomecommunity.com "StatusCode.INTERNAL" thread). Unretried, a
    # single bad-machine hit aborts the whole dataset. Re-wrap with the SDK's
    # own `retry_rpc` over the widened `SCORE_VARIANT_RETRY_STATUS` set so a
    # retry re-establishes the RPC on a (hopefully healthy) different machine.
    # (score_variant is already @retry_rpc-decorated; this outer wrapper is a
    # superset of those codes, so the nesting is redundant-but-benign.)
    score_variant_with_retry = dna_client.retry_rpc(
        model.score_variant,
        max_attempts=SCORE_VARIANT_MAX_ATTEMPTS,
        retry_status_codes=frozenset(
            getattr(grpc.StatusCode, name) for name in SCORE_VARIANT_RETRY_STATUS
        ),
    )
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

        scores = score_variant_with_retry(
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


# --- Chromatin-accessibility QTL path (caQTL/dsQTL, issues #262/#311) ------------
# AlphaGenome's recommended accessibility variant scorer (Suppl Table 9): a single
# center-mask, width=501, DIFF_LOG2_SUM (= signed log2 fold-change) on the DNASE
# output, cell-type-matched to GM12878. Deliberately separate from the 7-assay
# L2_DIFF_LOG1P matched-pair path above — different scorer, aggregation, single track.

# GM12878's Experimental Factor Ontology cell-line CURIE (AlphaGenome track metadata;
# confirmed against the paper's SPI1-in-GM12878 = EFO:0002784).
GM12878_ONTOLOGY_CURIE: str = "EFO:0002784"
DNASE_LFC_MASK_WIDTH: int = 501


def make_dnase_lfc_scorer() -> "_vs.CenterMaskScorer":
    """The GM12878-DNase accessibility scorer for QTLs.

    ``CenterMaskScorer(DNASE, width=501, DIFF_LOG2_SUM)`` — returns a signed log2
    fold-change (alt vs ref) of summed DNase coverage in the 501 bp window centered
    on the variant. A single API call returns this for *all* DNASE tracks (one per
    cell type); :func:`select_gm12878_dnase_lfc` picks GM12878.
    """
    from alphagenome.models import dna_client, variant_scorers

    return variant_scorers.CenterMaskScorer(
        requested_output=dna_client.OutputType.DNASE,
        width=DNASE_LFC_MASK_WIDTH,
        aggregation_type=variant_scorers.AggregationType.DIFF_LOG2_SUM,
    )


def select_gm12878_dnase_lfc(tidy: pd.DataFrame) -> float:
    """Signed GM12878-DNase LFC from a single-variant ``tidy_scores`` frame.

    The DNase scorer returns every DNASE track (one per cell type); we keep the
    GM12878 cell line (``output_type == "DNASE"`` and ``ontology_curie ==
    "EFO:0002784"``) and return the mean ``raw_score`` over the matched track(s) —
    there is exactly one GM12878 DNase track today, but a mean is robust if
    AlphaGenome adds replicates. Asserts at least one match so a metadata-schema
    change fails loud rather than silently returning NaN.
    """
    needed = {"output_type", "ontology_curie", "raw_score"}
    assert needed <= set(tidy.columns), (
        f"tidy_scores missing {needed - set(tidy.columns)}; got {tidy.columns.tolist()}"
    )
    sel = tidy[
        (tidy["output_type"] == "DNASE")
        & (tidy["ontology_curie"] == GM12878_ONTOLOGY_CURIE)
    ]
    assert len(sel) >= 1, (
        f"no GM12878 DNase track (ontology {GM12878_ONTOLOGY_CURIE}) in tidy_scores; "
        f"available DNASE ontologies e.g. "
        f"{tidy.loc[tidy['output_type'] == 'DNASE', 'ontology_curie'].head(3).tolist()}"
    )
    return float(sel["raw_score"].mean())


def score_variants_dnase_lfc(
    V: pd.DataFrame,
    num_workers: int = 4,
    api_key: str | None = None,
) -> np.ndarray:
    """Per-variant signed GM12878-DNase log2 fold-change (alt vs ref).

    Same forward-strand, per-variant, retry-wrapped threading as
    :func:`score_variants_alphagenome`, but with the single DNase-LFC scorer and
    GM12878-track selection. ``V`` needs ``chrom, pos, ref, alt`` (``pos`` 1-based,
    ``chrom`` unprefixed). Returns a float array aligned to ``V``'s row order (the
    signed LFC; correlate against the signed study effect for direction, ``|·|`` for
    causality).

    Genome-orient ``ref``/``alt`` before scoring (e.g. via the #310 dataset build):
    AlphaGenome computes ``log2(alt/ref)`` for the alleles you pass, so passing
    genome-oriented alleles yields a genome-frame signed LFC directly — no
    downstream sign flip.
    """
    import grpc
    from alphagenome.data import genome
    from alphagenome.models import dna_client, variant_scorers

    if api_key is None:
        api_key = os.environ.get("ALPHA_GENOME_API_KEY")
    assert api_key, "ALPHA_GENOME_API_KEY not set; pass api_key= or export the env var"
    missing = {"chrom", "pos", "ref", "alt"} - set(V.columns)
    assert not missing, f"V missing required columns: {missing}"

    model = dna_client.create(api_key)
    score_variant_with_retry = dna_client.retry_rpc(
        model.score_variant,
        max_attempts=SCORE_VARIANT_MAX_ATTEMPTS,
        retry_status_codes=frozenset(
            getattr(grpc.StatusCode, name) for name in SCORE_VARIANT_RETRY_STATUS
        ),
    )
    sequence_length = dna_client.SUPPORTED_SEQUENCE_LENGTHS[
        f"SEQUENCE_LENGTH_{SEQUENCE_LENGTH}"
    ]
    organism = dna_client.Organism.HOMO_SAPIENS
    scorer = make_dnase_lfc_scorer()

    def score_one(row) -> float:
        chrom = row.chrom if str(row.chrom).startswith("chr") else f"chr{row.chrom}"
        variant = genome.Variant(
            chromosome=chrom,
            position=int(row.pos),
            reference_bases=row.ref,
            alternate_bases=row.alt,
        )
        interval = variant.reference_interval.resize(sequence_length).copy()
        interval.strand = "+"
        scores = score_variant_with_retry(
            interval=interval,
            variant=variant,
            organism=organism,
            variant_scorers=[scorer],
        )
        tidy = variant_scorers.tidy_scores([scores])
        return select_gm12878_dnase_lfc(tidy)

    out = np.empty(len(V), dtype=np.float64)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
        for i, lfc in enumerate(
            tqdm(ex.map(score_one, V.itertuples(index=False)), total=len(V))
        ):
            out[i] = lfc
    return out


_DNASE_LFC_COL = "alphagenome_dnase_lfc"
_DNASE_LFC_KEY = ("chrom", "pos", "ref", "alt")


def _read_lfc_checkpoint(path: str) -> pl.DataFrame | None:
    """Read a resumable LFC checkpoint, or ``None`` if it doesn't exist yet.

    ``path`` may be local or ``s3://`` (polars reads both natively). A missing file
    is the expected first-run state → ``None``; any other read error propagates.
    """
    try:
        return pl.read_parquet(path)
    except (FileNotFoundError, OSError, pl.exceptions.ComputeError):
        return None


def score_dnase_lfc_resumable(
    variants: pl.DataFrame,
    checkpoint_path: str,
    *,
    num_workers: int = 4,
    chunk_size: int = 500,
    max_new_calls: int | None = None,
    score_fn: Callable[..., np.ndarray] = score_variants_dnase_lfc,
    api_key: str | None = None,
) -> pl.DataFrame:
    """Resumably score variants' GM12878-DNase LFC, S3-checkpointed to ``checkpoint_path``.

    Reads the checkpoint (if present) keyed by ``(chrom, pos, ref, alt)``, scores only
    the *missing* variants in chunks of ``chunk_size``, and rewrites the checkpoint after
    each chunk — so a crash / spot preemption resumes from the last chunk. Returns
    ``[chrom, pos, ref, alt, alphagenome_dnase_lfc]`` aligned to ``variants`` (full
    coverage asserted). Scoring genome-oriented ``ref``/``alt`` yields genome-frame LFC.

    Args:
        variants: ``[chrom, pos, ref, alt]`` work-list (unique; genome-oriented).
        checkpoint_path: local or ``s3://`` parquet path used as both the seed cache
            and the resumable checkpoint. Point it at an existing fully-scored artifact
            to reuse without re-spending the API (the caqtl/dsqtl case → 0 calls).
        num_workers / chunk_size: API threading + checkpoint granularity.
        max_new_calls: fail-loud cap — raise if more than this many variants would be
            sent to the API (guards against an accidental full re-score). ``None`` =
            uncapped.
        score_fn: injectable for tests (defaults to :func:`score_variants_dnase_lfc`).
    """
    key = list(_DNASE_LFC_KEY)
    missing = set(key) - set(variants.columns)
    assert not missing, f"variants missing required columns: {missing}"
    work = variants.select(key)
    assert work.unique().height == work.height, (
        "duplicate (chrom,pos,ref,alt) in variants — dedupe before scoring"
    )

    prior = _read_lfc_checkpoint(checkpoint_path)
    if prior is not None and prior.height:
        for col in (*key, _DNASE_LFC_COL):
            assert col in prior.columns, f"checkpoint missing column {col!r}"
        cached = prior.select([*key, _DNASE_LFC_COL])
    else:
        cached = work.head(0).with_columns(
            pl.lit(None, dtype=pl.Float64).alias(_DNASE_LFC_COL)
        )

    todo = work.join(cached.select(key), on=key, how="anti")
    n_missing = todo.height
    if max_new_calls is not None and n_missing > max_new_calls:
        raise RuntimeError(
            f"{n_missing} variants need AlphaGenome scoring but max_new_calls="
            f"{max_new_calls}; raise the cap or point checkpoint_path at the cached "
            "predictions (caqtl/dsqtl reuse should need 0 new calls)"
        )

    for start in range(0, n_missing, chunk_size):
        chunk = todo.slice(start, chunk_size)
        scores = np.asarray(
            score_fn(chunk.to_pandas(), num_workers=num_workers, api_key=api_key),
            dtype=np.float64,
        )
        assert len(scores) == chunk.height, "score_fn returned wrong length"
        cached = pl.concat(
            [cached, chunk.with_columns(pl.Series(_DNASE_LFC_COL, scores))],
            how="vertical_relaxed",
        )
        cached.write_parquet(checkpoint_path)

    out = work.join(cached, on=key, how="left")
    # is_null catches variants the join didn't cover; is_nan catches a variant the scorer
    # returned NaN for (polars treats NaN as a non-null float, so is_null alone misses it).
    scored = out.get_column(_DNASE_LFC_COL)
    n_bad = int((scored.is_null() | scored.is_nan()).sum())
    assert n_bad == 0, f"{n_bad} variants unscored or NaN after resumable run"
    return out
