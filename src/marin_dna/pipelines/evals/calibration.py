"""Per-checkpoint mutation-rate calibration tables (cLLR stage 3, #267/#270).

Builds the ``llr_neutral_mean`` table that stage 4 (#271) subtracts to calibrate
variant LLRs against the local background mutation rate::

    calibrated LLR = LLR − llr_neutral_mean(pentanuc_mut)

The pipeline scores a pinned, **scoreable** subsampled neutral-site set
(``snakemake/neutral_sites`` → ``neutral_sites_n{n}_w{w}.parquet``, columns
``[chrom, pos, ref, pentanuc]``) with the **fast 2-forward LLR bundle**
(:func:`marin_dna.pipelines.evals.inference.compute_variant_scores`, FWD+RC), then
bins by ``pentanuc_mut = pentanuc + "_" + alt``. Each neutral site is scored against
its 3 non-ref alts, so a 5-mer's three calibration cells draw from the same sites.

The neutral set is pre-filtered upstream (``filter_acgt_window_sites`` at window
``w``) so every site's ``w``-bp window is all-ACGT — the scoring kernel asserts ACGT
on the window, and genome-wide neutral sites near assembly gaps otherwise trip it.
That filtering is model-independent (one set per window, ``w`` = the largest model
window) and lives in the neutral-sites pipeline, so this GPU step never reads the
genome for filtering and just consumes the clean set.

Entropy calibration (``entropy_neutral_mean``, via the 4-forward marginal atom
:func:`marin_dna.model.scoring.compute_marginal_clm`) is a **separate, deferred**
path — this module is LLR-only.

The central pentanucleotide is precomputed upstream (``annotate_pentanucleotide``
in ``snakemake/neutral_sites``), so here we only expand to alts and aggregate.
Coordinates follow the eval datasets: 1-based ``pos`` (the scoring path's tool
boundary), as produced by ``enumerate_positions``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from marin_dna.data.dna import NUCLEOTIDES
from marin_dna.pipelines.evals.inference import compute_variant_scores


def expand_sites_to_variants(sites: pd.DataFrame) -> pd.DataFrame:
    """Expand neutral ``(chrom, pos, ref, pentanuc)`` sites to one row per non-ref alt.

    Each site yields 3 SNV rows (``alt`` ∈ ``{A,C,G,T} \\ {ref}``), tagged with the
    calibration cell key ``pentanuc_mut = pentanuc + "_" + alt``. The subsample
    unit upstream is the **5-mer**, so a site's three cells share its sites; the
    binning here is by the 3,072 ``pentanuc_mut`` cells (1,024 five-mers × 3 alts).

    Args:
        sites: ``[chrom, pos, ref, pentanuc]`` — bare ``chrom``, 1-based ``pos``,
            ``ref`` ∈ {A,C,G,T}, ``pentanuc`` the central 5-mer (center == ref).

    Returns:
        ``[chrom, pos, ref, alt, pentanuc, pentanuc_mut]`` — exactly ``3 * len(sites)``
        rows, ``alt != ref`` on every row.
    """
    for col in ("chrom", "pos", "ref", "pentanuc"):
        assert col in sites.columns, f"sites missing column {col!r}"
    assert len(sites) > 0, "empty neutral-site set"

    pentanuc = sites["pentanuc"].astype(str)
    # The 5-mer must be ACGT-only and centered on ref — both guaranteed by the
    # upstream annotate_pentanucleotide, re-checked here so a wrong/stale input
    # parquet fails loud rather than mislabeling every calibration bin.
    assert pentanuc.str.fullmatch("[ACGT]{5}").all(), (
        "pentanuc must be a 5-mer over ACGT — wrong/degenerate neutral input?"
    )
    assert (pentanuc.str[2].to_numpy() == sites["ref"].to_numpy()).all(), (
        "pentanuc center base != ref — centering/coordinate bug in the neutral set"
    )

    base = sites[["chrom", "pos", "ref", "pentanuc"]]
    parts: list[pd.DataFrame] = []
    for alt in NUCLEOTIDES:
        sub = base.loc[base["ref"] != alt].copy()
        sub["alt"] = alt
        parts.append(sub)
    variants = pd.concat(parts, ignore_index=True)
    variants["pentanuc_mut"] = variants["pentanuc"] + "_" + variants["alt"]
    variants = variants[["chrom", "pos", "ref", "alt", "pentanuc", "pentanuc_mut"]]

    assert len(variants) == 3 * len(sites), (
        f"expected {3 * len(sites)} variants, got {len(variants)}"
    )
    assert (variants["ref"] != variants["alt"]).all(), "ref == alt leaked in"
    return variants.reset_index(drop=True)


def aggregate_llr_neutral_mean(
    scored: pd.DataFrame,
    *,
    min_bin_count: int,
    subsample_n: int,
) -> pd.DataFrame:
    """Bin per-variant LLRs into the per-cell ``llr_neutral_mean`` calibration table.

    Averages raw LLR over the neutral sites in each ``pentanuc_mut`` cell, on the
    forward, reverse-complement, and FWD+RC-averaged strands. ``_avg`` follows the
    eval convention (``marin_dna`` evals_v2 metrics rule): average the *raw* LLR
    per variant first (``llr_avg = (llr_fwd + llr_rc) / 2``), then take the cell
    mean — equivalently ``llr_neutral_mean_avg = (mean_fwd + mean_rc) / 2``.

    Args:
        scored: expanded variants (from :func:`expand_sites_to_variants`) joined
            with per-strand atoms — must carry ``pentanuc_mut, pentanuc, ref, alt,
            llr_fwd, llr_rc``.
        min_bin_count: hard floor on observations per cell; asserts ``min(n_sites)
            >= min_bin_count`` so an unexpectedly empty/sparse cell fails fast.
        subsample_n: the upstream per-5-mer cap (recorded as a column for provenance).

    Returns:
        One row per cell, sorted by ``pentanuc_mut``::

            [pentanuc_mut, pentanuc, ref, alt, n_sites,
             llr_neutral_mean_fwd, llr_neutral_mean_rc, llr_neutral_mean_avg,
             llr_neutral_std_avg, subsample_n]
    """
    for col in ("pentanuc_mut", "pentanuc", "ref", "alt", "llr_fwd", "llr_rc"):
        assert col in scored.columns, f"scored frame missing column {col!r}"
    assert len(scored) > 0, "empty scored frame"
    assert scored[["llr_fwd", "llr_rc"]].notna().all().all(), (
        "NaN in per-strand LLR columns"
    )

    scored = scored.copy()
    # Average raw LLR first (eval `_avg` semantics), then bin.
    scored["llr_avg"] = (scored["llr_fwd"] + scored["llr_rc"]) / 2.0

    table = (
        scored.groupby(["pentanuc_mut", "pentanuc", "ref", "alt"], sort=True)
        .agg(
            llr_neutral_mean_fwd=("llr_fwd", "mean"),
            llr_neutral_mean_rc=("llr_rc", "mean"),
            llr_neutral_mean_avg=("llr_avg", "mean"),
            llr_neutral_std_avg=("llr_avg", "std"),
            n_sites=("llr_avg", "size"),
        )
        .reset_index()
    )
    table["subsample_n"] = subsample_n

    # The 3 mean columns must be finite; std is NaN only for size-1 cells, which
    # the min_bin_count assert below rules out for any sane floor (>= 2).
    mean_cols = ["llr_neutral_mean_fwd", "llr_neutral_mean_rc", "llr_neutral_mean_avg"]
    assert table[mean_cols].notna().all().all(), "NaN in a per-cell mean LLR"

    counts = table["n_sites"]
    assert int(counts.min()) >= min_bin_count, (
        f"a calibration cell has only {int(counts.min())} obs "
        f"(< min_bin_count={min_bin_count}) — neutral set too sparse for this cap"
    )

    # Log the cell-count distribution + any cell below the cap (data-limited
    # 5-mers, mostly CpG-depleted; their SE = s/sqrt(count) regardless of cap).
    below = table[table["n_sites"] < subsample_n]
    print(
        f"[calibration] {len(table)} cells; n_sites "
        f"min={int(counts.min())} median={int(counts.median())} max={int(counts.max())}; "
        f"{len(below)} cell(s) below the cap n={subsample_n}"
    )

    cols = [
        "pentanuc_mut",
        "pentanuc",
        "ref",
        "alt",
        "n_sites",
        "llr_neutral_mean_fwd",
        "llr_neutral_mean_rc",
        "llr_neutral_mean_avg",
        "llr_neutral_std_avg",
        "subsample_n",
    ]
    return table[cols].sort_values("pentanuc_mut").reset_index(drop=True)


def compute_llr_neutral_mean(
    checkpoint_path: str | Path,
    sites: pd.DataFrame,
    genome_path: str | Path,
    window_size: int,
    *,
    subsample_n: int,
    min_bin_count: int,
    batch_size: int = 512,
    num_workers: int = 4,
    rc: bool = True,
    data_transform_on_the_fly: bool = True,
    torch_compile: bool = False,
) -> pd.DataFrame:
    """Score a neutral-site set with the LLR bundle and bin into a calibration table.

    Thin orchestrator over :func:`expand_sites_to_variants`,
    :func:`marin_dna.pipelines.evals.inference.compute_variant_scores`, and
    :func:`aggregate_llr_neutral_mean`. Bins in a single pass — only the small
    per-cell table is returned (the ~3·N raw per-variant rows are never persisted).

    Args:
        checkpoint_path: Local HF checkpoint dir.
        sites: neutral sites ``[chrom, pos, ref, pentanuc]`` (read **locally** by the
            caller — never ``pd.read_parquet("s3://…")`` in a process that later
            forks DataLoader workers; that deadlocks the in-worker genome reads).
        genome_path: GRCh38 reference (S3 URI ok — opened lazily inside each worker).
        window_size: DNA context window for the model (per-model ``window_size``).
        subsample_n: upstream per-5-mer cap (recorded for provenance).
        min_bin_count: hard floor on observations per cell.
        batch_size, num_workers, data_transform_on_the_fly, torch_compile: passed to
            ``compute_variant_scores``.
        rc: must be True — calibration averages FWD+RC to match the eval.

    Returns:
        The ``llr_neutral_mean`` table (see :func:`aggregate_llr_neutral_mean`).
    """
    assert rc, (
        "calibration scores FWD+RC (rc must be True) to match the eval convention"
    )
    variants = expand_sites_to_variants(sites)
    scores = compute_variant_scores(
        checkpoint_path=checkpoint_path,
        dataset=variants[["chrom", "pos", "ref", "alt"]],
        genome_path=genome_path,
        context_size=window_size,
        batch_size=batch_size,
        num_workers=num_workers,
        data_transform_on_the_fly=data_transform_on_the_fly,
        torch_compile=torch_compile,
        rc=rc,
    )
    assert len(scores) == len(variants), (len(scores), len(variants))
    scored = pd.concat(
        [variants.reset_index(drop=True), scores.reset_index(drop=True)], axis=1
    )
    return aggregate_llr_neutral_mean(
        scored, min_bin_count=min_bin_count, subsample_n=subsample_n
    )
