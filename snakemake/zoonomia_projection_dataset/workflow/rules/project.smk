"""Cross-mammal projection step (issue #149, follows benchmark #153).

Projects the conservation-filtered human BED ``results/human/intervals/filtered/min{p}.bed.gz``
onto every species in ``config/species_zoonomia_447_family_dedup.tsv`` via
``halLiftover --noDupes`` against the Zoonomia 447 Cactus HAL (1.18 TiB,
staged outside Snakemake to ``HAL_PATH`` by ``sky/project.yaml``). All
post-processing logic lives in ``src/marin_dna/projection``.

DAG (per ``min_p``):

    results/human/intervals/filtered/min{p}.bed.gz                (existing pipeline output)
        │
        │  prepare_input_ucsc — strip Ensembl bare names, prepend "chr",
        │                       emit BED6 with strand=+ and score=0
        ▼
    results/projection/input/min{p}.ucsc.bed
        │
        │  hal_chrom_sizes (per species)  +  project_one_species (║×SPECIES)
        ▼
    results/projection/min{p}/per_species/{species}.parquet
        │
        │  merge_projected
        ▼
    results/projection/min{p}/all_species.parquet

Smoke tier prepends two ZRS cCRE rows (issue #120) before head-truncating
to chr1, and gates on ``zrs_sanity_check``.
"""

from marin_dna_zoonomia_projection.projection.filter import (
    filter_length,
    filter_single_chrom_strand,
)
from marin_dna_zoonomia_projection.projection.hal import (
    attach_src_size,
    parse_halliftover_bed,
    run_halliftover,
)
from marin_dna_zoonomia_projection.projection.resize import resize_dataframe
from marin_dna_zoonomia_projection.projection.sequence import (
    attach_sequences_to_parquet,
    parquet_to_bed6,
    parse_bedtools_getfasta_output,
)
from marin_dna_zoonomia_projection.projection.subset import (
    filter_to_species,
    filter_to_subset,
    load_species,
)

# Two cCREs from SCREEN Registry V4 inside the canonical ZRS limb enhancer
# (hg38 chr7:156790115-156793672); Mus_musculus orthologs verified at
# mm10 chr5:29315086-29315432 / chr5:29316458-29316801. Used for the
# smoke-tier sanity check (zrs_sanity_check rule).
ZRS_BED6_LINES: list[str] = [
    "chr7\t156791361\t156791613\tzrs_EH38E2604086\t0\t+",
    "chr7\t156792617\t156792954\tzrs_EH38E2604087\t0\t+",
]

SOURCE_SPECIES = "Homo_sapiens"


def projection_targets(min_p: str) -> list[str]:
    """v1 target list: concat Parquet, plus ZRS sanity check on smoke tier.

    Defined once here so ``rule all_projected`` and ``rule all_sequences``
    don't duplicate the smoke-tier branching.
    """
    out = [f"results/projection/min{min_p}/all_species.parquet"]
    if TIER == "smoke":
        out.append(f"results/projection/min{min_p}/zrs_sanity.txt")
    return out


PER_SPECIES_SCHEMA: dict[str, pl.DataType] = {
    "query_name": pl.Utf8,
    "species": pl.Utf8,
    "t_chrom": pl.Utf8,
    "t_start": pl.Int64,
    "t_end": pl.Int64,
    "t_strand": pl.Utf8,
    "t_src_size": pl.Int64,
}


rule prepare_input_ucsc:
    """Convert filtered BED to UCSC chrom names + BED6 (halLiftover input).

    The existing pipeline writes BEDs with bare Ensembl names ("1", "2", ...,
    "X", "Y") and 4 columns. halLiftover (Cactus) requires UCSC "chr1"/etc
    and preserves the input column count — BED6 with strand survives, BED4
    drops it (benchmark patch a84e063). For smoke tier, also prepend the
    ZRS cCRE rows and head-truncate the rest to chr1.
    """
    input:
        "results/human/intervals/filtered/min{min_p}.bed.gz",
    output:
        "results/projection/input/min{min_p}.ucsc.bed",
    params:
        tier=TIER,
    run:
        df = pl.read_csv(
            input[0],
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
        # Ensembl bare name → UCSC "chr"+name. Already-prefixed inputs would
        # double-prefix so guard with an assertion.
        assert not df["chrom"].str.starts_with("chr").any(), (
            f"input BED has UCSC-style chroms; expected Ensembl bare: "
            f"{df['chrom'].unique().to_list()[:5]}"
        )
        if params.tier == "smoke":
            df = df.filter(pl.col("chrom") == "1").head(1000)
        bed6 = df.select(
            chrom=pl.lit("chr") + pl.col("chrom"),
            start="start",
            end="end",
            name="name",
            score=pl.lit(0),
            # BED6 strand=+ (no strand info on tile windows).
            strand=pl.lit("+"),
        )
        with open(output[0], "w") as fout:
            if params.tier == "smoke":
                # ZRS cCREs first so they're guaranteed to be in the
                # head-truncated output regardless of the chr1 row count.
                for line in ZRS_BED6_LINES:
                    fout.write(line + "\n")
            bed6.write_csv(fout, separator="\t", include_header=False)


rule hal_chrom_sizes:
    """Per-species `halStats --chromSizes` → 2-col TSV.

    Single-threaded; 48-way species parallelism. ~30 s/species cold-cache.
    """
    output:
        "results/projection/chrom_sizes/{species}.tsv",
    threads: 1
    resources:
        mem_mb=1500,
    shell:
        "halStats --chromSizes {wildcards.species} %s > {output}" % HAL_PATH


rule project_one_species:
    """Project the UCSC BED onto one species via halLiftover; filter; resize.

    Per (query, species) the final output has at most one row: raw
    ``halLiftover --noDupes`` may emit multiple rows when the query crosses
    alignment block boundaries on the same chrom+strand;
    ``filter_single_chrom_strand`` collapses those to a single merged span
    and drops multi-chrom/multi-strand groups. ``filter_length`` then drops
    merged spans outside ``[pre_resize_min_len, pre_resize_max_len]`` —
    typically 95-100 % of survivors are in the [128, 512] range per #153.
    """
    input:
        bed="results/projection/input/min{min_p}.ucsc.bed",
        chrom_sizes="results/projection/chrom_sizes/{species}.tsv",
    output:
        parquet="results/projection/min{min_p}/per_species/{species}.parquet",
    threads: 1
    resources:
        mem_mb=2000,
    run:
        species = wildcards.species
        # halLiftover writes a temp BED next to the input; keep it on the
        # same NVMe-mounted filesystem as the inputs.
        out_dir = Path(output.parquet).parent
        work_dir = out_dir / "_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        raw_bed = work_dir / f"{species}.bed"
        run_halliftover(
            HAL_PATH, SOURCE_SPECIES, input.bed, species, raw_bed, no_dupes=True
        )
        df = parse_halliftover_bed(raw_bed, species=species)
        df = attach_src_size(df, input.chrom_sizes)
        df = filter_single_chrom_strand(df)
        df = filter_length(df, min_len=PRE_RESIZE_MIN, max_len=PRE_RESIZE_MAX)
        df = df.filter(pl.col("t_src_size") >= TARGET_LEN)
        if df.is_empty():
            pl.DataFrame(schema=PER_SPECIES_SCHEMA).write_parquet(output.parquet)
            return
        # resize_dataframe vectorises the midpoint-clamp; its own asserts
        # cover the bounds invariants. We add the at-most-one-row-per-query
        # invariant here since filter_single_chrom_strand promises it.
        resized = resize_dataframe(df, target_len=TARGET_LEN).select(
            list(PER_SPECIES_SCHEMA.keys())
        )
        assert (
            resized["query_name"].n_unique() == resized.height
        ), "expected at most one row per query_name after filter"
        resized.write_parquet(output.parquet)
        raw_bed.unlink(missing_ok=True)


rule merge_projected:
    """Concatenate per-species Parquets into one all-species Parquet (streaming)."""
    input:
        lambda wc: expand(
            "results/projection/min{min_p}/per_species/{species}.parquet",
            min_p=[wc.min_p],
            species=SPECIES,
        ),
    output:
        "results/projection/min{min_p}/all_species.parquet",
    resources:
        mem_mb=4000,
    run:
        lf = pl.concat([pl.scan_parquet(p) for p in input], how="vertical")
        lf.sink_parquet(output[0])


rule zrs_sanity_check:
    """Smoke-tier-only sanity gate: ZRS cCREs lift human to mouse correctly."""
    input:
        mus="results/projection/min{min_p}/per_species/Mus_musculus.parquet",
    output:
        "results/projection/min{min_p}/zrs_sanity.txt",
    shell:
        "marin-dna-zrs-check --mus-parquet {input.mus} --output {output}"


rule hal_to_fasta:
    """Per-species multi-chromosome FASTA from HAL via hal2fasta.

    Kept on local NVMe only (``local()`` skips the S3 default-storage
    upload — see commit 6fbb5b4 for the convention). 320 GB across 108
    species would be wasteful to mirror; the ``.2bit`` is the archival
    format (~80 GB total).
    """
    output:
        local("results/projection/_genomes_fa/{species}.fa"),
    threads: 1
    resources:
        mem_mb=2000,
    shell:
        "hal2fasta %s {wildcards.species} > {output}" % HAL_PATH


rule fasta_to_2bit:
    """Compact 2bit per species, persisted to S3 (~750 MB/species)."""
    input:
        local("results/projection/_genomes_fa/{species}.fa"),
    output:
        "results/projection/genomes/{species}.2bit",
    conda:
        "../envs/bioinformatics.yaml"
    threads: 1
    resources:
        mem_mb=2000,
    shell:
        "faToTwoBit {input} {output}"


rule extract_sequences:
    """Per-species, per-cutoff: strand-aware sequence extraction at projected coords.

    Output is a Parquet that extends the projection schema with a
    ``sequence`` column (canonical for downstream Polars filtering and
    HuggingFace dataset construction). bedtools getfasta -s revcomps
    strand=- rows natively, so the stored sequence is the
    transcription-aware target genome string — no further revcomp
    needed downstream.
    """
    input:
        parquet="results/projection/min{min_p}/per_species/{species}.parquet",
        fasta=local("results/projection/_genomes_fa/{species}.fa"),
        # Force the 2bit too so it's archived to S3 even though
        # extraction itself uses the .fa (NVMe-resident).
        twobit="results/projection/genomes/{species}.2bit",
    output:
        "results/projection/min{min_p}/sequences/{species}.parquet",
    conda:
        "../envs/bioinformatics.yaml"
    threads: 1
    resources:
        mem_mb=4000,
    run:
        out_path = Path(str(output))
        bed_path = out_path.with_suffix(".bed.tmp")
        fa_path = out_path.with_suffix(".fa.tmp")
        n = parquet_to_bed6(input.parquet, bed_path)
        if n == 0:
            attach_sequences_to_parquet(
                input.parquet, [], out_path, target_len=TARGET_LEN
            )
        else:
            shell(
                "bedtools getfasta -s -fi {input.fasta} -bed "
                + str(bed_path)
                + " -nameOnly -fo "
                + str(fa_path)
            )
            sequences = parse_bedtools_getfasta_output(fa_path)
            attach_sequences_to_parquet(
                input.parquet, sequences, out_path, target_len=TARGET_LEN
            )
            fa_path.unlink(missing_ok=True)
        bed_path.unlink(missing_ok=True)


rule merge_sequences:
    """Concatenate per-species sequence Parquets into one all-species file (streaming).

    This is the canonical artifact that subsetting (rule subset_dataset)
    operates on. Polars sink_parquet keeps peak memory bounded.
    """
    input:
        lambda wc: expand(
            "results/projection/min{min_p}/sequences/{species}.parquet",
            min_p=[wc.min_p],
            species=SPECIES,
        ),
    output:
        "results/projection/min{min_p}/all_species_with_sequence.parquet",
    threads: 1
    resources:
        mem_mb=4000,
    run:
        lf = pl.concat([pl.scan_parquet(p) for p in input], how="vertical")
        lf.sink_parquet(output[0])


rule subset_dataset:
    """Lazy-filter the all-species sequence Parquet to a subset by query_name.

    Subsets are defined by a ``query_names.txt`` file (one name per
    line, ``#`` comments allowed). Filter pushdown + column pruning
    mean only the matching rows are decompressed; cost is bounded by
    NVMe throughput (~30–60 s per subset regardless of count).

    Subset definitions live in
    ``config/subsets/{subset}.query_names.txt`` and are pre-computed
    from human annotation overlaps with ``results/human/intervals/filtered/min{min_p}.bed.gz``
    (out of scope here — bring your own query_names list).
    """
    input:
        all_species="results/projection/min{min_p}/all_species_with_sequence.parquet",
        query_names="config/subsets/{subset}.query_names.txt",
    output:
        "results/projection/min{min_p}/subsets/{subset}.parquet",
    threads: 1
    resources:
        mem_mb=4000,
    run:
        filter_to_subset(input.all_species, input.query_names, output[0])


rule all_projected:
    """Final target for the cross-mammal projection step.

    Smoke tier additionally requires the ZRS sanity check to pass; that
    rule's own ``shell`` block exits 1 on any FAIL and so fails this rule
    transitively. Full tier skips the sanity check (no ZRS rows in the input
    BED — they're only added in smoke).
    """
    input:
        projection_targets(PROJECT_MIN_P),


rule all_sequences:
    """v2 target: subsumes v1 (all_projected) and adds per-species 2bit
    plus the canonical sequence-bearing Parquet (per-species + merged).

    The merged ``all_species_with_sequence.parquet`` is the source for
    rule ``subset_dataset`` (issue #149 v3 — subsetting; see
    ``config/subsets/``).
    """
    input:
        # v1 outputs: concatenated Parquet (+ ZRS sanity on smoke).
        projection_targets(PROJECT_MIN_P),
        # v2 outputs: per-species 2bit (archival) + per-species sequence
        # Parquet + merged all-species sequence Parquet.
        expand(
            "results/projection/genomes/{species}.2bit",
            species=SPECIES,
        ),
        expand(
            "results/projection/min{min_p}/sequences/{species}.parquet",
            min_p=[PROJECT_MIN_P],
            species=SPECIES,
        ),
        f"results/projection/min{PROJECT_MIN_P}/all_species_with_sequence.parquet",
