"""Per-window phyloP_447m scoring, fanned out across chroms.

Each ``score_windows_chrom`` reads its per-chrom BED, scores via pyBigWig
(see ``marin_dna_zoonomia_projection.conservation.scoring.score_windows``), and writes a
per-chrom Parquet. ``merge_scored`` concatenates them.

Throughput on the 8-vCPU c6id.2xlarge: ~5 min wall for human autosomes +
X + Y (~22.9M windows).
"""

from marin_dna_zoonomia_projection.conservation.scoring import (
    score_windows as _score_windows,
)


rule score_windows_chrom:
    """Score 255 bp windows on a single chromosome against phyloP_447m."""
    input:
        windows="results/human/intervals/windows/{chrom}.bed.gz",
        bw="results/bigwig/phyloP_447m.bw",
    output:
        "results/human/intervals/scored/per_chrom/phyloP_447m_{chrom}.parquet",
    wildcard_constraints:
        chrom="|".join(STANDARD_CHROMS),
    resources:
        # ~30 MB BED + polars frame + pyBigWig handle peak around 1 GB; cap
        # at 1.5 GB so Snakemake's scheduler caps concurrency below the
        # 16 GB ceiling of c6id.2xlarge (an earlier run OOMed at 8x).
        mem_mb=1500,
    params:
        threshold=PHYLOP_447M_THRESHOLD,
    run:
        windows_df = pl.read_csv(
            input.windows,
            separator="\t",
            has_header=False,
            new_columns=["chrom", "start", "end", "name"],
            schema_overrides={
                "chrom": pl.Utf8,
                "start": pl.Int64,
                "end": pl.Int64,
                "name": pl.Utf8,
            },
        )
        assert len(windows_df) > 0, f"no windows in {input.windows}"
        assert (
            windows_df["chrom"] == wildcards.chrom
        ).all(), f"per-chrom BED has rows on other chroms: {windows_df['chrom'].unique()}"
        assert (windows_df["end"] - windows_df["start"] == WINDOW_SIZE).all()
        scored = _score_windows(input.bw, windows_df, params.threshold)
        assert len(scored) == len(windows_df)
        assert (scored["conserved_bases"] >= 0).all()
        assert (
            (scored["conserved_bases"] <= scored["n_valid_bases"])
            | (scored["n_valid_bases"] == 0)
        ).all()
        assert (scored["n_valid_bases"] <= WINDOW_SIZE).all()
        assert scored["proportion_conserved"].min() >= 0.0
        assert scored["proportion_conserved"].max() <= 1.0
        scored.sort("start").write_parquet(output[0])


rule merge_scored:
    """Concatenate per-chrom Parquets into a single sorted Parquet."""
    input:
        expand(
            "results/human/intervals/scored/per_chrom/phyloP_447m_{chrom}.parquet",
            chrom=STANDARD_CHROMS,
        ),
    output:
        "results/human/intervals/scored/phyloP_447m_windows.parquet",
    run:
        dfs = [pl.read_parquet(p) for p in input]
        seen_chroms = {df["chrom"].unique().item() for df in dfs}
        assert seen_chroms == set(STANDARD_CHROMS), (
            f"missing chroms in per-chrom Parquets: "
            f"{set(STANDARD_CHROMS) - seen_chroms}"
        )
        merged = pl.concat(dfs).sort(["chrom", "start"])
        # Defensive cross-check after concat.
        assert (merged["end"] - merged["start"] == WINDOW_SIZE).all()
        assert (merged["conserved_bases"] >= 0).all()
        assert (merged["proportion_conserved"] <= 1.0).all()
        merged.write_parquet(output[0])
