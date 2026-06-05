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
    # Fail fast on a shifted-column mis-parse (e.g. a dump served without the
    # leading ``bin`` column would slide every index by one): UCSC ``genoName``
    # is always ``chr``-prefixed, so a non-``chr`` value means we parsed the
    # wrong columns.
    assert out["chrom"].str.startswith("chr").all(), (
        f"rmsk genoName values are not chr-prefixed in {path} — wrong column "
        "layout? expected the UCSC bin-column rmsk.txt.gz schema"
    )
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
            # Idempotent prefixing — if a caller ever passes already-``chr``ed
            # names, don't produce ``chrchr1`` (which would silently skip the
            # whole contig below).
            bw_chrom = (
                chrom if chrom.startswith(chrom_prefix) else f"{chrom_prefix}{chrom}"
            )
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
    address the (Ensembl-named) reference, emit **1-based** ``pos``, keep only
    ``A/C/G/T`` reference bases (drops ``N`` / soft-masked-to-N), and
    de-duplicate ``(chrom, pos)`` — slightly-overlapping RepeatMasker features
    can cover a base twice, and a duplicate would double-weight a calibration
    bin downstream.

    The reference is read **once per chromosome** (over the span of that
    chromosome's intervals) rather than once per interval: a fragmented neutral
    set is millions of short intervals, and one byte-range read per interval
    against an ``s3://`` genome would be millions of round-trips.

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
        ``ref`` ∈ {A,C,G,T}, unique ``(chrom, pos)``.
    """
    for col in ("chrom", "start", "end"):
        assert col in intervals.columns, f"intervals missing column {col!r}"

    # Group the requested intervals by (chr-stripped) chromosome so each
    # chromosome's reference is read exactly once.
    by_chrom: dict[str, list[tuple[int, int]]] = {}
    for chrom_raw, start, end in zip(
        intervals["chrom"], intervals["start"], intervals["end"]
    ):
        chrom = str(chrom_raw)
        if chrom[:3].lower() == "chr":  # case-insensitive UCSC strip
            chrom = chrom[3:]
        if chrom not in chroms:
            continue
        start, end = int(start), int(end)
        assert end > start, f"empty/inverted interval {chrom}:{start}-{end}"
        by_chrom.setdefault(chrom, []).append((start, end))

    valid = set("ACGT")
    chrom_out: list[str] = []
    pos_out: list[int] = []
    ref_out: list[str] = []
    for chrom, ivs in by_chrom.items():
        lo = min(s for s, _ in ivs)
        hi = max(e for _, e in ivs)
        span = get_seq(chrom, lo, hi).upper()
        assert len(span) == hi - lo, (
            f"reference returned {len(span)} bp for {chrom}:{lo}-{hi} "
            f"(expected {hi - lo})"
        )
        for start, end in ivs:
            for i in range(start, end):
                ref = span[i - lo]
                if ref in valid:
                    chrom_out.append(chrom)
                    pos_out.append(i + 1)  # 0-based index -> 1-based pos
                    ref_out.append(ref)

    out = pd.DataFrame({"chrom": chrom_out, "pos": pos_out, "ref": ref_out})
    # Collapse positions covered by more than one source interval so no
    # calibration bin is double-weighted.
    out = out.drop_duplicates(["chrom", "pos"]).reset_index(drop=True)
    # Defensive: the whole point is a clean neutral set — no surprises downstream.
    assert out["ref"].isin(valid).all(), "non-ACGT ref leaked into neutral set"
    return out


def annotate_pentanucleotide(
    sites: pd.DataFrame,
    get_seq: Callable[[str, int, int], str],
) -> pd.DataFrame:
    """Attach each neutral site's central pentanucleotide (5-mer) context.

    The 5-mer is the window ``[pos-2, pos+2]`` (1-based, variant-centered) — in
    0-based half-open terms the central base is ``pos - 1`` and the 5-mer spans
    ``[pos-3, pos+2)`` — matching ``_get_variant_window`` in
    ``marin_dna.data.transforms`` and GPN-Star's ``pentanuc``. The mutation-rate
    calibration (issue #267) bins by this 5-mer (and, for LLR, ``5mer + "_" +
    alt``), so it must be computed once, model-independently, here.

    The reference is read **once per chromosome** (over the span covering every
    site's 5-mer), same as :func:`enumerate_positions` — a per-site byte-range
    read against an ``s3://`` genome would be millions of round-trips.

    Args:
        sites: ``[chrom, pos, ref]`` — bare ``chrom``, **1-based** ``pos``,
            ``ref`` ∈ {A,C,G,T} (as produced by :func:`enumerate_positions`).
        get_seq: ``(chrom, start, end) -> str`` over the reference (0-based
            half-open, bare chrom names); the sequence is upper-cased here.

    Returns:
        ``sites`` plus a ``pentanuc`` column (uppercase 5-mer). Sites whose 5-mer
        has a non-ACGT flank (``N`` near assembly gaps) are dropped — they can't
        index a calibration bin; the center is ACGT by construction (it is the
        ref). The center base of every returned 5-mer equals ``ref`` (asserted).
    """
    for col in ("chrom", "pos", "ref"):
        assert col in sites.columns, f"sites missing column {col!r}"

    sites = sites.reset_index(drop=True)
    pentanuc = np.empty(len(sites), dtype=object)
    for chrom, sub in sites.groupby("chrom", sort=False):
        pos = sub["pos"].to_numpy()
        # One read spanning every site's 5-mer: leftmost 5-mer starts at
        # (min_pos - 1) - 2 = min_pos - 3; rightmost ends at (max_pos - 1) + 3.
        lo = int(pos.min()) - 3
        hi = int(pos.max()) + 2
        span = get_seq(str(chrom), lo, hi).upper()
        assert len(span) == hi - lo, (
            f"reference returned {len(span)} bp for {chrom}:{lo}-{hi} "
            f"(expected {hi - lo})"
        )
        for i, p in zip(sub.index.to_numpy(), pos):
            start = int(p) - 3 - lo  # offset of this site's 5-mer within span
            pentanuc[i] = span[start : start + 5]

    out = sites.copy()
    out["pentanuc"] = pentanuc
    # Centering check: the middle base of every 5-mer must equal the stored ref
    # (validates the 1-based-pos <-> 0-based-reference convention end-to-end).
    center = out["pentanuc"].str[2]
    assert (center == out["ref"]).all(), (
        f"pentanucleotide center != ref for "
        f"{int((center != out['ref']).sum())} site(s) — centering/coordinate bug"
    )
    valid = out["pentanuc"].str.fullmatch("[ACGT]{5}")
    n_drop = int((~valid).sum())
    if n_drop:
        print(
            f"[annotate_pentanucleotide] dropped {n_drop} site(s) "
            f"with non-ACGT 5-mer flanks"
        )
    return out.loc[valid].reset_index(drop=True)


def subsample_per_context(sites: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Subsample to at most ``n`` neutral sites per pentanucleotide (5-mer).

    The calibration subsampling unit is the **5-mer** (``pentanuc``), *not* the
    per-alt bin (``5mer + "_" + alt``): each site is scored against all 3 non-ref
    alts, so the three bins of a 5-mer share its sites. Keeping ``n`` sites per
    5-mer therefore yields ``n`` observations in *each* of its 3 calibration
    bins, at ``3n`` variant-scorings per 5-mer. Site-level + alt-agnostic so the
    LLR path (3 alts) and the entropy atom (#269, 4-allele marginal) consume one
    set.

    Deterministic given ``(site set, n, seed)``: sites are canonically sorted
    first (so the result is independent of input row order), then a single
    seeded RNG draws ``min(n, count)`` per 5-mer in sorted-5-mer order.

    Args:
        sites: ``[chrom, pos, ref, pentanuc, ...]`` (e.g. from
            :func:`annotate_pentanucleotide`).
        n: max sites to keep per 5-mer. 5-mers with ``count <= n`` are kept whole.
        seed: RNG seed.

    Returns:
        The kept sites (all input columns), canonically sorted by
        ``(chrom, pos)``. Every 5-mer appears at most ``n`` times.
    """
    assert n > 0, f"n must be positive, got {n}"
    for col in ("chrom", "pos", "pentanuc"):
        assert col in sites.columns, f"sites missing column {col!r}"
    if len(sites) == 0:
        return sites.reset_index(drop=True)
    # Unique (chrom, pos) is the contract (enumerate_positions dedups). Enforce it:
    # a duplicate would double-weight a calibration bin and make the seeded draw
    # depend on input row order.
    assert not sites.duplicated(["chrom", "pos"]).any(), (
        "duplicate (chrom, pos) sites — would double-weight a calibration bin"
    )

    # Canonical sort first so the draw depends only on the site set + seed, not on
    # input row order; the reset index is then in (chrom, pos) order (unique keys).
    sites = sites.sort_values(["chrom", "pos"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for _, group in sites.groupby("pentanuc", sort=True):
        idx = group.index.to_numpy()
        if len(idx) <= n:
            keep.append(idx)
        else:
            keep.append(idx[rng.choice(len(idx), size=n, replace=False)])

    # Sorting the kept positional indices restores (chrom, pos) order in one pass
    # (the index is already (chrom, pos)-sorted) — no second DataFrame sort.
    out = sites.loc[np.sort(np.concatenate(keep))].reset_index(drop=True)
    assert out.groupby("pentanuc").size().max() <= n, "a 5-mer exceeded the cap"
    return out
