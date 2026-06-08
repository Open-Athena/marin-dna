"""DART-Eval Task 3 cell-type-specific peak dataset (interval, not variant).

Parsing + chromosome-split for the DART-Eval **Task 3** benchmark
("Discriminating Cell-Type-Specific Elements"), brought into ``snakemake/evals``
as a standalone HF dataset (``bolinas-dna/evals_dart_task3``). Unlike the Task-5
QTL datasets (``caqtl``/``dsqtl`` in ``dart_eval.py``) these are **500 bp
interval windows, not variants** — no ref/alt, no consequence/VEP annotation, no
matching/subsampling, no liftover.

Source (DART-Eval, Patel *et al.* NeurIPS 2024 D&B;
https://github.com/kundajelab/DART-Eval), Synapse project ``syn60581042``. We
consume their **top-5,000-per-cell-type** file
``input_data/top_5000_deseq_peaks.tsv`` (``syn62161401``) — the balanced subset
DART-Eval feeds their zero-shot **clustering / UMAP** baseline. It is the top
5,000 DESeq2 differentially-accessible peaks for each of 5 ATAC-seq cell lines
(GM12878, H1ESC, HEPG2, IMR90, K562) → **25,000** windows, exactly 5,000 per cell
type, each a **500 bp** consensus-peak window (±250 bp around the ATAC summit),
GRCh38. Every peak coordinate is unique (a peak in two cell types' top-5,000
would dual-count, but none do).

(The larger ``processed_inputs/peaks_by_cell_label_unique_dataloader_format.tsv``
— 216,746 windows — is DART-Eval's *full* cell-type-specific set for the
supervised probing/fine-tuning task; the top-5,000 file here is the strict subset
used for the unsupervised clustering view.)

The source columns are ``Chr, Start, End, Cell Type``; the parser projects them
onto the standard ``chrom, start, end, label`` schema and asserts loudly if the
real file deviates so a schema drift can't silently corrupt the dataset.
Coordinates are **0-based, half-open** (verified: every peak matches the
0-based dataloader-format file exactly) — the same convention as this codebase —
so they flow through unchanged.
"""

from __future__ import annotations

import polars as pl

# The 5 ATAC-seq cell lines = the classification labels. Casing matches
# DART-Eval's source TSV; a real-file drift trips the membership assert in
# `parse_dart_task3` (fail fast rather than silently mislabel).
CELL_TYPES = ["GM12878", "H1ESC", "HEPG2", "IMR90", "K562"]

# Every consensus peak is a fixed 500 bp window (±250 bp around the summit).
PEAK_WIDTH = 500

# Top 5,000 differentially-accessible peaks selected per cell type.
TOP_K_PER_CELL_TYPE = 5000

# Required columns of the top_5000_deseq_peaks.tsv source.
DART_TASK3_REQUIRED = ["Chr", "Start", "End", "Cell Type"]

# DART-Eval canonical Task-3 chromosome split — a 3-way holdout taken verbatim
# from their training scripts (e.g.
# `experiments/task_3_peak_classification/train/{hyenadna,nucleotide_transformer}.py`).
# Distinct from the evals pipeline's default odd/even 2-way `SPLIT_CHROMS`; kept
# identical to DART-Eval so any future Task-3 numbers stay comparable to theirs.
# The three lists partition all of chr1..22,X,Y disjointly.
SPLIT_CHROMS = {
    "train": [
        "1",
        "2",
        "3",
        "4",
        "7",
        "8",
        "9",
        "11",
        "12",
        "13",
        "15",
        "16",
        "17",
        "19",
        "X",
        "Y",
    ],
    "validation": ["6", "21"],
    "test": ["5", "10", "14", "18", "20", "22"],
}


def _strip_chr(col: pl.Expr) -> pl.Expr:
    """Drop a leading ``chr`` so chromosome names match our convention
    (``"1"``..``"22"``, ``"X"``, ``"Y"``)."""
    return col.cast(pl.Utf8).str.replace(r"^chr", "")


def _require_columns(df: pl.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    assert not missing, (
        f"DART-Eval Task 3 TSV missing expected columns {missing}; got {df.columns}"
    )


def parse_dart_task3(df: pl.DataFrame) -> pl.DataFrame:
    """Parse a raw ``top_5000_deseq_peaks.tsv`` frame into the standard interval
    schema.

    Projects ``Chr, Start, End, Cell Type`` onto ``chrom`` (``chr`` stripped) /
    ``start`` / ``end`` (0-based, half-open) / ``label`` (cell type), sorted by
    ``chrom, start``. Asserts the invariants that must hold for a valid Task-3
    window set: labels are a subset of the 5 cell types, every window is exactly
    ``PEAK_WIDTH`` bp, coordinates are non-negative. (Full-dataset checks — all 5
    cell types present, balanced 5,000-per-type, ~25k unique rows — live in
    ``assert_full_dataset`` so this stays usable on subsets/fixtures.)
    """
    _require_columns(df, DART_TASK3_REQUIRED)
    out = df.select(
        _strip_chr(pl.col("Chr")).alias("chrom"),
        pl.col("Start").cast(pl.Int64).alias("start"),
        pl.col("End").cast(pl.Int64).alias("end"),
        pl.col("Cell Type").cast(pl.Utf8).alias("label"),
    ).sort(["chrom", "start"])

    assert out["label"].null_count() == 0, "DART-Eval Task 3: null labels"
    bad = set(out["label"].unique().to_list()) - set(CELL_TYPES)
    assert not bad, (
        f"DART-Eval Task 3: unexpected cell-type labels {sorted(bad)} "
        f"(allowed {CELL_TYPES})"
    )
    widths = out.select((pl.col("end") - pl.col("start")).alias("w"))["w"]
    assert (widths == PEAK_WIDTH).all(), (
        f"DART-Eval Task 3: not every window is {PEAK_WIDTH} bp "
        f"(widths seen: {sorted(set(widths.to_list()))[:5]})"
    )
    assert (out["start"] >= 0).all(), "DART-Eval Task 3: negative start coordinate"
    return out


def split_frames(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Partition a parsed frame into ``{train, validation, test}`` by the
    DART-Eval canonical chromosome split.

    Asserts every chromosome present is covered by the canonical split and that
    the partition is exact (no row dropped or double-counted) — the split lists
    are disjoint, so each row lands in exactly one frame.
    """
    canonical = set().union(*SPLIT_CHROMS.values())
    present = set(df["chrom"].unique().to_list())
    extra = present - canonical
    assert not extra, (
        f"DART-Eval Task 3: chromosomes {sorted(extra)} are outside the canonical "
        f"Task-3 split (covers {sorted(canonical)})"
    )
    frames = {
        split: df.filter(pl.col("chrom").is_in(chroms))
        for split, chroms in SPLIT_CHROMS.items()
    }
    total = sum(f.height for f in frames.values())
    assert total == df.height, (
        f"DART-Eval Task 3: split partition is not exact "
        f"(sum {total} != input {df.height})"
    )
    return frames


def assert_full_dataset(
    df: pl.DataFrame, *, min_rows: int = 20_000, max_rows: int = 25_000
) -> None:
    """Build-time sanity for the *complete* parsed dataset (not subsets).

    Guards the real upload: all 5 cell types present, each within the documented
    top-5,000-per-cell-type cap, peak coordinates unique (no dual-counted peak),
    and the total near DART-Eval's 25,000 (5,000 x 5). A gross parse/source error
    (wrong file, truncated download, label drift) trips one of these before
    anything is pushed to HuggingFace.
    """
    labels = set(df["label"].unique().to_list())
    assert labels == set(CELL_TYPES), (
        f"DART-Eval Task 3: expected all 5 cell types {CELL_TYPES}, got {sorted(labels)}"
    )
    per = df.group_by("label").len().sort("label")
    counts = dict(zip(per["label"].to_list(), per["len"].to_list()))
    # n >= 1 is already guaranteed by the all-5-cell-types assert above, so only
    # the top-K upper cap is a live check here.
    for ct, n in counts.items():
        assert n <= TOP_K_PER_CELL_TYPE, (
            f"DART-Eval Task 3: cell type {ct!r} has {n} peaks "
            f"(exceeds the top-{TOP_K_PER_CELL_TYPE} cap)"
        )
    # Balanced by construction (top-K selected per cell type) — enforce it so the
    # card's "5,000 balanced per cell type" claim can never silently go stale.
    assert len(set(counts.values())) == 1, (
        f"DART-Eval Task 3: cell types are not balanced (counts {counts}); "
        f"the top-{TOP_K_PER_CELL_TYPE} selection should give equal counts"
    )
    n_unique = df.select(["chrom", "start", "end"]).unique().height
    assert n_unique == df.height, (
        f"DART-Eval Task 3: {df.height - n_unique} duplicate peak coordinates "
        "(a peak in >1 cell type's top-5,000 would dual-count)"
    )
    assert min_rows <= df.height <= max_rows, (
        f"DART-Eval Task 3: total {df.height} rows outside expected "
        f"[{min_rows}, {max_rows}] (DART-Eval ships 25,000)"
    )
