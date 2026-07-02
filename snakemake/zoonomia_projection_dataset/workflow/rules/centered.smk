"""exp351 (#351): enhancer-CENTERED training windows, projected to the one-per-order cohort.

Parallel to the tiled path (windows → score → filter → label → project), but each
window is defined **directly from a dELS/pELS cCRE** (``make_enhancer_anchors``:
one 255 bp window centered on each enhancer, keep-all) instead of a uniform grid
tile, and projection targets only the 19 ``order`` species. Conservation
(≥ ``project_min_p``) and exon-subtraction (noexon) reuse the SAME
``score_windows`` / ``build_region_beds`` / ``label_windows_bp_majority`` machinery
as the tiled curated arm, so the ONLY difference vs
``bolinas-dna/zoonomia-v1-v4_ccre_noexon_enhancer-order`` (exp326 arm B, the tiled
control) is window anchoring — the exp351 experiment variable.

DAG (all human-side steps are cheap CPU; only the projection needs the HAL):

    cre/all.parquet ──make_centered_anchors──▶ anchors/enhancer_anchors.bed.gz
        ──score_centered_anchors (phyloP_447m)──▶ scored/…parquet
        ──filter_centered_conserved (≥ min_p)──▶ filtered/conserved.bed.gz
        ──label_centered_noexon (drop exon overlap)──▶ filtered/noexon.bed.gz
        ──prepare_centered_input_ucsc──▶ projection/input.ucsc.bed
        ──project_centered_one_species (║ × ORDER_SPECIES, halLiftover)──▶ per_species/{sp}.parquet
        ──extract_centered_sequences──▶ sequences/{sp}.parquet
        ──merge_centered_sequences──▶ order/all_species_with_sequence.parquet

The terminal parquet is registered as ``INTERVALS_SOURCES['v4_ccre_enhancer_centered']``
(via ``extra_intervals_sources`` in config → dataset.smk) and flows through the shared
``subset_species`` (order, a no-op filter) → shard → upload chain to HF
``zoonomia-v1-v4_ccre_enhancer_centered-order``.

Reuses the generic per-species genome rules (``hal_chrom_sizes``, ``hal_to_fasta``,
``fasta_to_2bit``) and the projection library helpers imported by ``project.smk``
(``run_halliftover``, ``filter_single_chrom_strand``, ``filter_length``,
``resize_dataframe``, …) — all in scope here since ``project.smk`` is included first.
"""


# 19 one-per-order leaves (a strict subset of the 108-family projection set).
ORDER_SPECIES = pl.read_csv(config["species_subsets"]["order"], separator="\t")[
    "species"
].to_list()
assert "Homo_sapiens" in ORDER_SPECIES, "order cohort must include Homo_sapiens"
if TIER == "smoke":
    ORDER_SPECIES = [s for s in ORDER_SPECIES if s in set(SPECIES)]

CENTERED_DIR = "results/centered"


rule make_centered_anchors:
    """dELS/pELS cCRE → one WINDOW_SIZE window centered on each (keep-all), minus any
    window overlapping an undefined (N / off-chrom) base. Pure human-side; no HAL.

    Keep-all is per row (``_coverage_bp``, not a ``GenomicSet`` merge), so clustered
    enhancers each keep their own centered window — the #351 decision. The undefined
    subtraction mirrors ``make_windows``' ``bedtools intersect -v -b undefined``.
    """
    input:
        cre="results/human/intervals/cre/all.parquet",
        undefined="results/human/intervals/undefined.bed",
    output:
        f"{CENTERED_DIR}/anchors/enhancer_anchors.bed.gz",
    resources:
        mem_mb=8000,
    run:
        from marin_dna.pipelines.zoonomia_projection_dataset.region_labels import (
            _coverage_bp,
            make_enhancer_anchors,
        )

        cre = pl.read_parquet(input.cre)
        anchors = make_enhancer_anchors(cre, WINDOW_SIZE)
        if TIER == "smoke":
            # Fast smoke: a couple thousand chr1 anchors, projected to the 4
            # smoke species — exercises the whole run-block chain cheaply.
            anchors = anchors.filter(pl.col("chrom") == "1").head(2000)

        undef = pl.read_csv(
            input.undefined,
            separator="\t",
            has_header=False,
            new_columns=["chrom", "start", "end"],
            columns=[0, 1, 2],
            schema_overrides={"chrom": pl.Utf8, "start": pl.Int64, "end": pl.Int64},
        )
        w = anchors.to_pandas().astype({"chrom": str}).reset_index(drop=True)
        u = undef.to_pandas().astype({"chrom": str}).reset_index(drop=True)
        undef_bp = _coverage_bp(w, u)
        kept = anchors.filter(pl.Series(name="_defined", values=undef_bp == 0))
        assert kept.height > 0, "all centered anchors overlap undefined regions?"

        with gzip.open(output[0], "wt") as fout:
            for row in kept.iter_rows(named=True):
                fout.write(
                    f"{row['chrom']}\t{row['start']}\t{row['end']}\t{row['name']}\n"
                )


rule score_centered_anchors:
    """phyloP_447m conservation per centered anchor (reuses ``score_windows``)."""
    input:
        anchors=f"{CENTERED_DIR}/anchors/enhancer_anchors.bed.gz",
        bw="results/bigwig/phyloP_447m.bw",
    output:
        f"{CENTERED_DIR}/scored/enhancer_anchors.parquet",
    resources:
        mem_mb=8000,
    run:
        from marin_dna.pipelines.conservation.scoring import score_windows

        anchors = pl.read_csv(
            input.anchors,
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
        assert (anchors["end"] - anchors["start"] == WINDOW_SIZE).all()
        scored = score_windows(input.bw, anchors, PHYLOP_447M_THRESHOLD)
        assert len(scored) == len(anchors)
        assert scored["proportion_conserved"].min() >= 0.0
        assert scored["proportion_conserved"].max() <= 1.0
        scored.write_parquet(output[0])


rule filter_centered_conserved:
    """Keep anchors with ``proportion_conserved >= project_min_p`` (mirror ``filter_bed``)."""
    input:
        f"{CENTERED_DIR}/scored/enhancer_anchors.parquet",
    output:
        f"{CENTERED_DIR}/filtered/conserved.bed.gz",
    run:
        min_p = float(PROJECT_MIN_P)
        assert 0.0 <= min_p <= 1.0
        df = pl.read_parquet(input[0])
        kept = df.filter(pl.col("proportion_conserved") >= min_p)
        assert 0 < len(kept) <= len(df)
        with gzip.open(output[0], "wt") as fout:
            for row in kept.iter_rows(named=True):
                fout.write(
                    f"{row['chrom']}\t{row['start']}\t{row['end']}\t{row['name']}\n"
                )


rule label_centered_noexon:
    """Drop anchors overlapping any exon (cds / utr3 / ncrna_exon / tss) — the #326
    noexon condition on centered windows via the SAME v4 labeler, so the exon
    definition matches ``v4_ccre_noexon_enhancer`` exactly."""
    input:
        anchors=f"{CENTERED_DIR}/filtered/conserved.bed.gz",
        gtf=f"results/annotation/Homo_sapiens.GRCh38.{config['ensembl_release']}.gtf.gz",
        cre="results/human/intervals/cre/all.parquet",
        defined="results/human/intervals/defined.bed",
    output:
        f"{CENTERED_DIR}/filtered/noexon.bed.gz",
    resources:
        mem_mb=24000,
    run:
        from marin_dna.data.intervals import GenomicSet
        from marin_dna.pipelines.zoonomia_projection_dataset.region_labels import (
            build_region_beds,
            label_windows_bp_majority,
            select_anchors_noexon,
        )

        defined = GenomicSet.read_bed(input.defined)
        beds = build_region_beds(
            input.gtf,
            input.cre,
            defined,
            tss_radius=REGION_LABEL_TSS_RADIUS,
            ccre_flank=REGION_LABEL_CCRE_FLANK_V4,
            tss_pc_only=REGION_LABEL_TSS_PC_ONLY_V4,
        )
        labeled = label_windows_bp_majority(
            input.anchors,
            beds,
            functional_threshold=REGION_LABEL_FUNCTIONAL_THRESHOLD,
            priority=REGION_LABEL_PRIORITY_V4,
        )
        noexon = select_anchors_noexon(labeled)
        with gzip.open(output[0], "wt") as fout:
            for row in noexon.select(["chrom", "start", "end", "name"]).iter_rows(
                named=True
            ):
                fout.write(
                    f"{row['chrom']}\t{row['start']}\t{row['end']}\t{row['name']}\n"
                )


rule prepare_centered_input_ucsc:
    """Centered noexon anchors → UCSC BED6 for halLiftover (mirror ``prepare_input_ucsc``;
    full-tier only — no smoke ZRS injection, the anchors already carry names)."""
    input:
        f"{CENTERED_DIR}/filtered/noexon.bed.gz",
    output:
        f"{CENTERED_DIR}/projection/input.ucsc.bed",
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
        assert not df["chrom"].str.starts_with("chr").any(), (
            "centered anchors already UCSC-style; expected bare Ensembl chroms"
        )
        bed6 = df.select(
            chrom=pl.lit("chr") + pl.col("chrom"),
            start="start",
            end="end",
            name="name",
            score=pl.lit(0),
            strand=pl.lit("+"),
        )
        bed6.write_csv(output[0], separator="\t", include_header=False)


rule project_centered_one_species:
    """halLiftover the centered anchors onto one ``order`` species; filter; resize.

    Body identical to ``project_one_species`` (project.smk) — different input BED and
    output dir; ``hal_chrom_sizes`` (generic, per-species) is reused as-is.
    """
    input:
        bed=f"{CENTERED_DIR}/projection/input.ucsc.bed",
        chrom_sizes="results/projection/chrom_sizes/{species}.tsv",
    output:
        f"{CENTERED_DIR}/projection/per_species/{{species}}.parquet",
    threads: 1
    resources:
        mem_mb=2000,
    run:
        species = wildcards.species
        out_dir = Path(output[0]).parent
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
            pl.DataFrame(schema=PER_SPECIES_SCHEMA).write_parquet(output[0])
            return

        resized = resize_dataframe(df, target_len=TARGET_LEN).select(
            list(PER_SPECIES_SCHEMA.keys())
        )
        assert resized["query_name"].n_unique() == resized.height, (
            "expected at most one row per query_name after filter"
        )
        resized.write_parquet(output[0])
        raw_bed.unlink(missing_ok=True)


rule extract_centered_sequences:
    """Strand-aware sequence at projected centered coords (mirror ``extract_sequences``);
    reuses the per-species ``_genomes_fa/{species}.fa`` (hal2fasta).

    Unlike the tiled ``extract_sequences`` this does NOT force the per-species
    ``.2bit`` as an input: those are already archived on S3 from the original
    108-family run, so re-deriving them (``fasta_to_2bit``) would be pure waste.
    """
    input:
        parquet=f"{CENTERED_DIR}/projection/per_species/{{species}}.parquet",
        fasta=local("results/projection/_genomes_fa/{species}.fa"),
    output:
        f"{CENTERED_DIR}/projection/sequences/{{species}}.parquet",
    threads: 1
    resources:
        mem_mb=4000,
    conda:
        "../envs/bioinformatics.yaml"
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


rule merge_centered_sequences:
    """Concatenate per-species centered sequence Parquets (order cohort). This is the
    canonical artifact registered as ``INTERVALS_SOURCES['v4_ccre_enhancer_centered']``."""
    input:
        expand(
            f"{CENTERED_DIR}/projection/sequences/{{species}}.parquet",
            species=ORDER_SPECIES,
        ),
    output:
        f"{CENTERED_DIR}/order/all_species_with_sequence.parquet",
    resources:
        mem_mb=4000,
    run:
        lf = pl.concat([pl.scan_parquet(p) for p in input], how="vertical")
        lf.sink_parquet(output[0])


rule all_centered:
    """Terminal target for the enhancer-centered projection (order cohort)."""
    input:
        f"{CENTERED_DIR}/order/all_species_with_sequence.parquet",
