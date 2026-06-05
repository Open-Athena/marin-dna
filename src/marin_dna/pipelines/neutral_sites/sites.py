"""GPN-Star neutral-site construction: ancestral repeats ∩ low conservation.

Replicates the GPN-Star neutral definition used for mutation-rate calibration
(GPN-Star Methods, "Mutation Rate Calibration"; reference implementation
``analysis/gpn-star/train_and_eval/workflow/rules/calibration.smk`` in
songlab-cal/gpn). The library pieces, each a thin testable unit the snakemake
rules call into:

- :func:`parse_rmsk` — RepeatMasker ``.txt.gz`` → repeat-interval BED frame
  (excluding ``Simple_repeat`` / ``Low_complexity``).
- :func:`contiguous_runs` / :func:`neutral_mask` / :func:`scan_neutral_intervals`
  — low-conservation sites from phyloP + phastCons bigWigs
  (``|phyloP| < threshold ∧ phastCons == 0``).
- :func:`enumerate_positions` — neutral intervals → per-base ``(chrom, pos, ref)``.

Coordinate / chromosome-name conventions
----------------------------------------
Everything in "BED land" (rmsk, liftOver, bedtools, the UCSC bigWigs) is
0-based half-open with UCSC ``chr``-prefixed chromosome names. We keep that
convention *unchanged* through the intersect/liftOver steps — it is the native
format of all those tools, and converting earlier would just invite mistakes.

:func:`enumerate_positions` is the single boundary back into our world: it
strips the ``chr`` prefix (our GRCh38 reference is Ensembl, which uses bare
``1``/``2``/``X`` names) and emits **1-based** ``pos`` — matching the HF eval
datasets so the exact same scoring path consumes neutral sites and eval
variants alike. (Per CLAUDE.md, 1-based lives only at this tool boundary; the
rest of the codebase is 0-based half-open.)

Deviation from the reference flagged for review (issue #267): GPN's awk emits
``swScore`` where it then tries to filter ``Simple_repeat`` / ``Low_complexity``,
so that exclusion is a silent no-op there. We filter ``repClass`` correctly —
ancestral *simple* repeats are poor neutral proxies — which is the reference's
evident intent, not its behavior.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pyBigWig

# UCSC RepeatMasker classes that are not suitable neutral references: simple
# tandem repeats and low-complexity regions evolve under non-neutral,
# length-changing processes and align poorly.
RMSK_EXCLUDE_CLASSES: frozenset[str] = frozenset({"Simple_repeat", "Low_complexity"})

# Column layout of the UCSC ``rmsk`` table dump (``hg38/database/rmsk.txt.gz``),
# which carries a leading ``bin`` column. 0-indexed positions we need.
_RMSK_COLS: dict[str, int] = {
    "chrom": 5,  # genoName  (e.g. "chr1")
    "start": 6,  # genoStart (0-based)
    "end": 7,  # genoEnd   (0-based, exclusive)
    "repName": 10,
    "repClass": 11,
}


def parse_rmsk(
    path: str,
    exclude_classes: frozenset[str] = RMSK_EXCLUDE_CLASSES,
) -> pd.DataFrame:
    """Parse a UCSC ``rmsk.txt.gz`` dump into a repeat-interval BED frame.

    Args:
        path: Path to ``rmsk.txt.gz`` (gzipped, tab-separated, ``bin``-column
            layout — the standard ``goldenPath/<db>/database/rmsk.txt.gz``).
        exclude_classes: ``repClass`` values to drop (default
            :data:`RMSK_EXCLUDE_CLASSES`).

    Returns:
        DataFrame ``[chrom, start, end, name]``, 0-based half-open, UCSC
        ``chr``-prefixed ``chrom`` (BED-land convention — fed to liftOver /
        bedtools). ``name`` is the ``repName`` (kept only so the BED has a
        4th column for liftOver; not unique).
    """
    # Read by positional index then rename — combining ``names`` with a
    # positional ``usecols`` is ambiguous across pandas versions, so we select
    # by integer position (columns come back labelled by their original index)
    # and rename explicitly.
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        compression="gzip",
        usecols=sorted(_RMSK_COLS.values()),
        dtype=str,
    )
    df = df.rename(columns={idx: name for name, idx in _RMSK_COLS.items()})
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df = df[~df["repClass"].isin(exclude_classes)].copy()
    assert (df["end"] > df["start"]).all(), "rmsk has empty/inverted intervals"
    out = df[["chrom", "start", "end", "repName"]].rename(columns={"repName": "name"})
    return out.reset_index(drop=True)


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Yield ``(start, end)`` 0-based half-open spans of each ``True`` run.

    Ported from the reference ``contiguous_runs``. ``start``/``end`` are
    indices *into* ``mask`` (the caller adds any window offset).
    """
    assert mask.ndim == 1, "mask must be 1-D"
    if not mask.any():
        return []
    diff = np.diff(mask.astype(np.int8))
    starts = (np.where(diff == 1)[0] + 1).tolist()
    ends = (np.where(diff == -1)[0] + 1).tolist()
    if mask[0]:  # run starts at index 0
        starts = [0] + starts
    if mask[-1]:  # run extends to the array end
        ends = ends + [int(mask.size)]
    assert len(starts) == len(ends)
    return list(zip(starts, ends))


def neutral_mask(
    phylop: np.ndarray,
    phastcons: np.ndarray,
    phylop_threshold: float,
) -> np.ndarray:
    """Boolean per-base neutrality mask: ``|phyloP| < t ∧ phastCons == 0``.

    NaN (bigWig "no data") fails both predicates, so unaligned/uncovered bases
    are correctly excluded — ``abs(nan) < t`` and ``nan == 0`` are both False.
    """
    assert phylop.shape == phastcons.shape, "phyloP/phastCons length mismatch"
    with np.errstate(invalid="ignore"):
        return (np.abs(phylop) < phylop_threshold) & (phastcons == 0)


def scan_neutral_intervals(
    phylop_bw_path: str,
    phastcons_bw_path: str,
    chroms: list[str],
    phylop_threshold: float,
    *,
    window_size: int = 1_000_000,
    chrom_prefix: str = "chr",
) -> pd.DataFrame:
    """Scan phyloP+phastCons bigWigs → low-conservation interval BED frame.

    Iterates each chromosome in fixed windows, reads both tracks, and emits the
    contiguous runs passing :func:`neutral_mask`. Bare ``chroms`` (e.g.
    ``"1"``) are ``chrom_prefix``-prefixed for the bigWig lookup; output
    ``chrom`` keeps the ``chr`` prefix (BED-land, for the downstream bedtools
    intersect with the ancestral-repeat BED).

    Args:
        phylop_bw_path, phastcons_bw_path: local bigWig paths.
        chroms: bare chromosome names to scan.
        phylop_threshold: ``|phyloP|`` cutoff (GPN uses 0.1 with ancestral
            repeats, 0.05 without).
        window_size: scan granularity (rows read per bigWig call).
        chrom_prefix: prefix added to bare names for the bigWig lookup.

    Returns:
        DataFrame ``[chrom, start, end]`` (0-based half-open, ``chr``-prefixed).
    """
    phylo = pyBigWig.open(phylop_bw_path)
    phast = pyBigWig.open(phastcons_bw_path)
    rows: list[tuple[str, int, int]] = []
    try:
        phylo_chroms = phylo.chroms()
        phast_chroms = phast.chroms()
        for chrom in chroms:
            bw_chrom = f"{chrom_prefix}{chrom}" if chrom_prefix else chrom
            if bw_chrom not in phylo_chroms or bw_chrom not in phast_chroms:
                # A track may legitimately lack a contig; skip rather than guess.
                continue
            # Use the overlapping length when the two tracks disagree (rare).
            length = min(int(phylo_chroms[bw_chrom]), int(phast_chroms[bw_chrom]))
            for win_start in range(0, length, window_size):
                win_end = min(win_start + window_size, length)
                pp = np.asarray(
                    phylo.values(bw_chrom, win_start, win_end, numpy=True),
                    dtype=np.float64,
                )
                pc = np.asarray(
                    phast.values(bw_chrom, win_start, win_end, numpy=True),
                    dtype=np.float64,
                )
                mask = neutral_mask(pp, pc, phylop_threshold)
                for s, e in contiguous_runs(mask):
                    rows.append((bw_chrom, win_start + s, win_start + e))
    finally:
        phylo.close()
        phast.close()
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


def enumerate_positions(
    intervals: pd.DataFrame,
    get_seq: Callable[[str, int, int], str],
    chroms: set[str],
) -> pd.DataFrame:
    """Expand neutral intervals to per-base ``(chrom, pos, ref)`` rows.

    The boundary back into our conventions: strip the UCSC ``chr`` prefix to
    address the (Ensembl-named) reference, emit **1-based** ``pos``, and keep
    only ``A/C/G/T`` reference bases (drops ``N`` and soft-masked-to-N).

    Args:
        intervals: ``[chrom, start, end]`` 0-based half-open, ``chr``-prefixed
            (as produced by :func:`scan_neutral_intervals` / bedtools).
        get_seq: ``(chrom, start, end) -> str`` over the reference (e.g. a
            :class:`marin_dna.data.genome.Genome`); called with bare chrom
            names and 0-based half-open coords. The returned sequence is
            upper-cased here, so soft-masking is fine.
        chroms: bare chromosome names to keep (others skipped).

    Returns:
        DataFrame ``[chrom, pos, ref]`` — bare ``chrom``, 1-based ``pos``,
        ``ref`` ∈ {A,C,G,T}.
    """
    for col in ("chrom", "start", "end"):
        assert col in intervals.columns, f"intervals missing column {col!r}"

    valid = set("ACGT")
    chrom_out: list[str] = []
    pos_out: list[int] = []
    ref_out: list[str] = []
    for chrom_raw, start, end in zip(
        intervals["chrom"], intervals["start"], intervals["end"]
    ):
        chrom = str(chrom_raw)
        if chrom.startswith("chr"):
            chrom = chrom[3:]
        if chrom not in chroms:
            continue
        start, end = int(start), int(end)
        assert end > start, f"empty/inverted interval {chrom}:{start}-{end}"
        seq = get_seq(chrom, start, end).upper()
        assert len(seq) == end - start, (
            f"reference returned {len(seq)} bp for {chrom}:{start}-{end} "
            f"(expected {end - start})"
        )
        for i, ref in enumerate(seq):
            if ref in valid:
                chrom_out.append(chrom)
                pos_out.append(start + i + 1)  # 0-based index -> 1-based pos
                ref_out.append(ref)

    out = pd.DataFrame({"chrom": chrom_out, "pos": pos_out, "ref": ref_out})
    # Defensive: the whole point is a clean neutral set — no surprises downstream.
    assert out["ref"].isin(valid).all(), "non-ACGT ref leaked into neutral set"
    return out
