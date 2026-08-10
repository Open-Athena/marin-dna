rule chrom_sizes:
    input:
        "results/genome/{g}.2bit",
    output:
        "results/chrom_sizes/{g}.tsv",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitInfo {input} {output}"


# just adds the start=0 column, but in human it also filters to standard chroms
rule extract_all:
    input:
        "results/chrom_sizes/{g}.tsv",
    output:
        "results/intervals/all/{g}.bed.gz",
    run:
        df = pd.read_csv(input[0], sep="\t", header=None, names=["chrom", "end"])
        df["start"] = 0
        # we want to filter to chromosomes, and exclude unplaced scaffolds,
        # alt. haplotypes, etc.
        # unfortunately no way to filter chroms based on prefix, AFAIK
        # in human we want to make sure to keep the standard chroms
        if wildcards.g == "GCF_000001405.40":
            df = df[df.chrom.str[:2] == "NC"]
        df = df[["chrom", "start", "end"]]
        df.to_csv(output[0], sep="\t", header=False, index=False)


rule extract_undefined:
    input:
        "results/genome/{g}.2bit",
    output:
        "results/intervals/undefined/{g}.bed.gz",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitInfo {input} /dev/stdout -nBed | gzip > {output}"


rule extract_defined:
    input:
        "results/intervals/all/{g}.bed.gz",
        "results/intervals/undefined/{g}.bed.gz",
    output:
        "results/intervals/defined/{g}.bed.gz",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "bedtools subtract -a {input[0]} -b {input[1]} | gzip > {output}"


rule make_windows:
    input:
        "results/intervals/{interval_src}/{g}.bed.gz",
    output:
        "results/intervals/windows/{interval_src}/{w}/{s}/{g}.bed.gz",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        """
        mkdir -p $(dirname {output})
        bedtools makewindows -b {input[0]} -w {wildcards.w} -s {wildcards.s} \
            | awk '$3-$2 == {wildcards.w}' \
            | gzip >{output}
        """


# promoters from protein-coding transcripts, similar to gpn-animal-promoter-dataset
# described in TraitGym paper
rule intervals_recipe_v1:
    input:
        "results/annotation/{g}.gtf.gz",
        "results/intervals/defined/{g}.bed.gz",
    output:
        "results/intervals/recipe/v1/{g}.bed.gz",
    run:
        promoter_n_upstream = 256
        promoter_n_downstream = 256
        ann = load_annotation(input[0])
        defined = GenomicSet.read_bed(input[1])
        promoters = get_promoters(
            ann,
            n_upstream=promoter_n_upstream,
            n_downstream=promoter_n_downstream,
            mRNA_only=True,
            within_bounds=defined,
        )
        promoters.write_bed(output[0])


rule intervals_recipe_v4:
    input:
        "results/annotation/{g}.gtf.gz",
        "results/intervals/defined/{g}.bed.gz",
    output:
        "results/intervals/recipe/v4/{g}.bed.gz",
    run:
        promoter_n_upstream = 256
        promoter_n_downstream = 256
        ann = load_annotation(input[0])
        defined = GenomicSet.read_bed(input[1])
        promoters = get_promoters(
            ann,
            n_upstream=promoter_n_upstream,
            n_downstream=promoter_n_downstream,
            mRNA_only=False,
            within_bounds=defined,
        )
        promoters.write_bed(output[0])


# CDS regions only
rule intervals_recipe_v3:
    input:
        "results/annotation/{g}.gtf.gz",
        "results/intervals/defined/{g}.bed.gz",
    output:
        "results/intervals/recipe/v3/{g}.bed.gz",
    run:
        min_size = 512
        ann = load_annotation(input[0])
        cds = get_cds(ann)
        defined = GenomicSet(read_bed_to_pandas(input[1]))
        intervals = cds.expand_min_size(min_size)
        intervals = intervals & defined
        write_pandas_to_bed(intervals.to_pandas(), output[0])


rule extract_cds:
    input:
        "results/annotation/{g}.gtf.gz",
    output:
        "results/intervals/cds/{g}.parquet",
    run:
        ann = load_annotation(input[0])
        cds = get_cds(ann)
        assert cds.n_intervals() > 0, f"No CDS regions found for {wildcards.g}"
        cds.write_parquet(output[0])


rule extract_5_prime_utr:
    input:
        "results/annotation/{g}.gtf.gz",
    output:
        "results/intervals/5_prime_utr/{g}.parquet",
    run:
        ann = load_annotation(input[0])
        utr = get_5_prime_utr(ann)
        # Allow empty UTR sets for genomes without UTR annotations
        utr.write_parquet(output[0])


rule extract_3_prime_utr:
    input:
        "results/annotation/{g}.gtf.gz",
    output:
        "results/intervals/3_prime_utr/{g}.parquet",
    run:
        ann = load_annotation(input[0])
        utr = get_3_prime_utr(ann)
        # Allow empty UTR sets for genomes without UTR annotations
        utr.write_parquet(output[0])


rule extract_promoters:
    input:
        "results/annotation/{g}.gtf.gz",
        "results/intervals/all/{g}.bed.gz",
    output:
        "results/intervals/promoters/{upstream}/{downstream}/{g}.parquet",
    run:
        ann = load_annotation(input[0])
        bounds = GenomicSet.read_bed(input[1])
        promoters = get_promoters(
            ann,
            n_upstream=int(wildcards.upstream),
            n_downstream=int(wildcards.downstream),
            mRNA_only=False,
            within_bounds=bounds,
        )
        assert (
            promoters.n_intervals() > 0
        ), f"No promoter regions found for {wildcards.g}"
        promoters.write_parquet(output[0])


rule extract_promoters_mRNA:
    input:
        "results/annotation/{g}.gtf.gz",
        "results/intervals/all/{g}.bed.gz",
    output:
        "results/intervals/promoters_mRNA/{upstream}/{downstream}/{g}.parquet",
    run:
        ann = load_annotation(input[0])
        bounds = GenomicSet.read_bed(input[1])
        promoters = get_promoters(
            ann,
            n_upstream=int(wildcards.upstream),
            n_downstream=int(wildcards.downstream),
            mRNA_only=True,
            within_bounds=bounds,
        )
        assert (
            promoters.n_intervals() > 0
        ), f"No mRNA promoter regions found for {wildcards.g}"
        promoters.write_parquet(output[0])


rule extract_ncrna_exons:
    input:
        "results/annotation/{g}.gtf.gz",
    output:
        "results/intervals/ncrna_exons/{g}.parquet",
    run:
        ann = load_annotation(input[0])
        ncrna = get_ncrna_exons(ann)
        # Allow empty ncRNA sets for genomes without ncRNA annotations
        ncrna.write_parquet(output[0])


rule parquet_to_bed:
    input:
        "results/intervals/{region}/{g}.parquet",
    output:
        "results/bed/{g}/{region}.bed",
    run:
        GenomicSet.read_parquet(input[0]).write_bed(output[0])


rule all_bed:
    input:
        expand(
            "results/bed/{g}/{region}.bed",
            region=config["functional_regions"],
            g=config["genome_subset_bed"],
        ),


# CDS, another version
rule intervals_recipe_v5:
    input:
        "results/intervals/cds/{g}.parquet",
        "results/intervals/defined/{g}.bed.gz",
    output:
        "results/intervals/recipe/v5/{g}.bed.gz",
    run:
        min_size, max_size = 20, 10_000
        add_flank = 20  # splice region
        expand_min_size = 256
        intervals = GenomicSet.read_parquet(input[0])
        defined = GenomicSet.read_bed(input[1])
        intervals = intervals.filter_size(min_size, max_size)
        intervals = intervals.add_flank(add_flank)
        intervals = intervals.expand_min_size(expand_min_size)
        intervals = intervals & defined
        intervals.write_bed(output[0])


# 5' UTR
rule intervals_recipe_v6:
    input:
        "results/intervals/5_prime_utr/{g}.parquet",
        "results/intervals/defined/{g}.bed.gz",
        "results/intervals/cds/{g}.parquet",
    output:
        "results/intervals/recipe/v6/{g}.bed.gz",
    run:
        min_size, max_size = 20, 10_000
        add_flank = 20  # splice region
        expand_min_size = 256
        intervals = GenomicSet.read_parquet(input[0])
        defined = GenomicSet.read_bed(input[1])
        subtract_regions = [
            GenomicSet.read_parquet(input[2]),  # cds
        ]
        for region in subtract_regions:
            intervals = intervals - region
        intervals = intervals.filter_size(min_size, max_size)
        intervals = intervals.add_flank(add_flank)
        intervals = intervals.expand_min_size(expand_min_size)
        intervals = intervals & defined
        intervals.write_bed(output[0])


# 3' UTR
rule intervals_recipe_v7:
    input:
        "results/intervals/3_prime_utr/{g}.parquet",
        "results/intervals/defined/{g}.bed.gz",
        "results/intervals/cds/{g}.parquet",
        "results/intervals/5_prime_utr/{g}.parquet",
    output:
        "results/intervals/recipe/v7/{g}.bed.gz",
    run:
        min_size, max_size = 20, 10_000
        add_flank = 20  # splice region
        expand_min_size = 256
        intervals = GenomicSet.read_parquet(input[0])
        defined = GenomicSet.read_bed(input[1])
        subtract_regions = [
            GenomicSet.read_parquet(input[2]),  # cds
            GenomicSet.read_parquet(input[3]),  # 5_prime_utr
        ]
        for region in subtract_regions:
            intervals = intervals - region
        intervals = intervals.filter_size(min_size, max_size)
        intervals = intervals.add_flank(add_flank)
        intervals = intervals.expand_min_size(expand_min_size)
        intervals = intervals & defined
        intervals.write_bed(output[0])


# ncRNA exons
rule intervals_recipe_v8:
    input:
        "results/intervals/ncrna_exons/{g}.parquet",
        "results/intervals/defined/{g}.bed.gz",
        "results/intervals/cds/{g}.parquet",
        "results/intervals/5_prime_utr/{g}.parquet",
        "results/intervals/3_prime_utr/{g}.parquet",
    output:
        "results/intervals/recipe/v8/{g}.bed.gz",
    run:
        min_size, max_size = 20, 10_000
        add_flank = 20  # splice region
        expand_min_size = 256
        intervals = GenomicSet.read_parquet(input[0])
        defined = GenomicSet.read_bed(input[1])
        subtract_regions = [
            GenomicSet.read_parquet(input[2]),  # cds
            GenomicSet.read_parquet(input[3]),  # 5_prime_utr
            GenomicSet.read_parquet(input[4]),  # 3_prime_utr
        ]
        for region in subtract_regions:
            intervals = intervals - region
        intervals = intervals.filter_size(min_size, max_size)
        intervals = intervals.add_flank(add_flank)
        intervals = intervals.expand_min_size(expand_min_size)
        intervals = intervals & defined
        intervals.write_bed(output[0])


# promoters
rule intervals_recipe_v9:
    input:
        "results/intervals/promoters/256/256/{g}.parquet",
        "results/intervals/defined/{g}.bed.gz",
        "results/intervals/cds/{g}.parquet",
        "results/intervals/5_prime_utr/{g}.parquet",
        "results/intervals/3_prime_utr/{g}.parquet",
        "results/intervals/ncrna_exons/{g}.parquet",
    output:
        "results/intervals/recipe/v9/{g}.bed.gz",
    run:
        intervals = GenomicSet.read_parquet(input[0])
        defined = GenomicSet.read_bed(input[1])
        subtract_regions = [
            GenomicSet.read_parquet(input[2]),  # cds
            GenomicSet.read_parquet(input[3]),  # 5_prime_utr
            GenomicSet.read_parquet(input[4]),  # 3_prime_utr
            GenomicSet.read_parquet(input[5]),  # ncrna_exons
        ]
        for region in subtract_regions:
            intervals = intervals - region
        intervals = intervals & defined
        intervals.write_bed(output[0])


# promoters (larger context)
rule intervals_recipe_v10:
    input:
        "results/intervals/promoters/2048/2048/{g}.parquet",
        "results/intervals/defined/{g}.bed.gz",
        "results/intervals/cds/{g}.parquet",
        "results/intervals/5_prime_utr/{g}.parquet",
        "results/intervals/3_prime_utr/{g}.parquet",
        "results/intervals/ncrna_exons/{g}.parquet",
    output:
        "results/intervals/recipe/v10/{g}.bed.gz",
    run:
        intervals = GenomicSet.read_parquet(input[0])
        defined = GenomicSet.read_bed(input[1])
        subtract_regions = [
            GenomicSet.read_parquet(input[2]),  # cds
            GenomicSet.read_parquet(input[3]),  # 5_prime_utr
            GenomicSet.read_parquet(input[4]),  # 3_prime_utr
            GenomicSet.read_parquet(input[5]),  # ncrna_exons
        ]
        for region in subtract_regions:
            intervals = intervals - region
        intervals = intervals & defined
        intervals.write_bed(output[0])


rule intervals_recipe_v11:
    input:
        "results/intervals/promoters_mRNA/2048/2048/{g}.parquet",
        "results/intervals/defined/{g}.bed.gz",
    output:
        "results/intervals/recipe/v11/{g}.bed.gz",
    run:
        intervals = GenomicSet.read_parquet(input[0])
        defined = GenomicSet.read_bed(input[1])
        intervals = intervals & defined
        intervals.write_bed(output[0])


rule intervals_recipe_v12:
    input:
        "results/intervals/3_prime_utr/{g}.parquet",
        "results/intervals/defined/{g}.bed.gz",
    output:
        "results/intervals/recipe/v12/{g}.bed.gz",
    run:
        min_size, max_size = 20, 10_000
        add_flank = 20  # splice region
        expand_min_size = 256
        intervals = GenomicSet.read_parquet(input[0])
        defined = GenomicSet.read_bed(input[1])
        intervals = intervals.filter_size(min_size, max_size)
        intervals = intervals.add_flank(add_flank)
        intervals = intervals.expand_min_size(expand_min_size)
        intervals = intervals & defined
        intervals.write_bed(output[0])


# 512bp upstream of CDS (5' direction)
rule intervals_recipe_v13:
    input:
        "results/annotation/{g}.gtf.gz",
        "results/intervals/defined/{g}.bed.gz",
    output:
        "results/intervals/recipe/v13/{g}.bed.gz",
    run:
        ann = load_annotation(input[0])
        defined = GenomicSet.read_bed(input[1])
        intervals = get_upstream_of_CDS(ann, dist=512, within_bounds=defined)
        intervals.write_bed(output[0])


# 512bp downstream of CDS (3' direction)
rule intervals_recipe_v14:
    input:
        "results/annotation/{g}.gtf.gz",
        "results/intervals/defined/{g}.bed.gz",
    output:
        "results/intervals/recipe/v14/{g}.bed.gz",
    run:
        ann = load_annotation(input[0])
        defined = GenomicSet.read_bed(input[1])
        intervals = get_downstream_of_CDS(ann, dist=512, within_bounds=defined)
        intervals.write_bed(output[0])


# 256bp downstream of CDS (3' direction)
rule intervals_recipe_v15:
    input:
        "results/annotation/{g}.gtf.gz",
        "results/intervals/defined/{g}.bed.gz",
    output:
        "results/intervals/recipe/v15/{g}.bed.gz",
    run:
        ann = load_annotation(input[0])
        defined = GenomicSet.read_bed(input[1])
        intervals = get_downstream_of_CDS(ann, dist=256, within_bounds=defined)
        intervals.write_bed(output[0])


# mRNA promoters (254bp upstream + 254bp downstream)
rule intervals_recipe_v16:
    input:
        "results/intervals/promoters_mRNA/254/254/{g}.parquet",
        "results/intervals/defined/{g}.bed.gz",
    output:
        "results/intervals/recipe/v16/{g}.bed.gz",
    run:
        intervals = GenomicSet.read_parquet(input[0])
        defined = GenomicSet.read_bed(input[1])
        intervals = intervals & defined
        intervals.write_bed(output[0])


# ENCODE cCRE enhancers (dELS + pELS)
rule intervals_recipe_v17:
    input:
        cre="results/cre/ELS.parquet",
        defined=f"results/intervals/defined/{HUMAN_GENOME}.bed.gz",
        chrom_mapping=local("config/human_chrom_mapping.tsv"),
    output:
        f"results/intervals/recipe/v17/{HUMAN_GENOME}.bed.gz",
    run:
        chrom_map = pl.read_csv(input.chrom_mapping, separator="\t")
        simple_to_refseq = dict(
            zip(chrom_map["ucsc"].str.replace("chr", ""), chrom_map["refseq"])
        )
        df = pl.read_parquet(input.cre).with_columns(
            pl.col("chrom").replace_strict(simple_to_refseq)
        )
        intervals = GenomicSet(df)
        intervals = intervals.resize(255)
        defined = GenomicSet.read_bed(input.defined)
        intervals = intervals & defined
        intervals.write_bed(output[0])


# ENCODE cCRE conserved enhancers (dELS + pELS, ≥20 conserved bp)
rule intervals_recipe_v18:
    input:
        cre="results/cre/ELS_conserved_20.parquet",
        defined=f"results/intervals/defined/{HUMAN_GENOME}.bed.gz",
        chrom_mapping=local("config/human_chrom_mapping.tsv"),
    output:
        f"results/intervals/recipe/v18/{HUMAN_GENOME}.bed.gz",
    run:
        chrom_map = pl.read_csv(input.chrom_mapping, separator="\t")
        simple_to_refseq = dict(
            zip(chrom_map["ucsc"].str.replace("chr", ""), chrom_map["refseq"])
        )
        df = pl.read_parquet(input.cre).with_columns(
            pl.col("chrom").replace_strict(simple_to_refseq)
        )
        intervals = GenomicSet(df)
        intervals = intervals.resize(255)
        defined = GenomicSet.read_bed(input.defined)
        intervals = intervals & defined
        intervals.write_bed(output[0])


# Cross-species projected ELS_conserved_20 enhancers with functional exons
# subtracted. Per-species: on hg38 this is the `intervals_source_unified`
# passthrough (equivalent to v18 after resize+scannable); on other genomes
# it's the mmseqs2 best-hit projection from hg38. Scannable = defined
# minus low-quality-excluded exons (see `rule scannable_regions` in
# `scannable.smk`).
rule intervals_recipe_v30:
    input:
        projected="results/intervals/ELS_conserved_20_mmseqs2_s75/{g}.parquet",
        scannable="results/intervals/scannable/{g}.bed.gz",
    output:
        "results/intervals/recipe/v30/{g}.bed.gz",
    run:
        intervals = GenomicSet.read_parquet(input.projected)
        intervals = intervals.resize(255)
        scannable = GenomicSet.read_bed(input.scannable)
        intervals = intervals & scannable
        intervals.write_bed(output[0])


rule all_intervals_recipe_v30_mammals_seg20:
    """Convenience target: the recipe v30 bed.gz for each genome in
    `mammals_seg20`. Does NOT trigger the full `rule all` HF upload —
    add `v30/255/128` to `intervals.training` in config to enable that.
    """
    input:
        expand(
            "results/intervals/recipe/v30/{g}.bed.gz",
            g=genome_sets.get("mammals_seg20", []),
        ),


# Source-curation sweep around v30 (issue: phyloP/phastCons × 20/50 bp).
# Each recipe wraps a different ELS_*_mmseqs2_s75 mapping with the same
# resize(255) + scannable intersect as v30, so the four datasets differ
# only in the upstream cCRE conservation filter.
rule intervals_recipe_v31:
    input:
        projected="results/intervals/ELS_conserved_50_mmseqs2_s75/{g}.parquet",
        scannable="results/intervals/scannable/{g}.bed.gz",
    output:
        "results/intervals/recipe/v31/{g}.bed.gz",
    run:
        intervals = GenomicSet.read_parquet(input.projected)
        intervals = intervals.resize(255)
        scannable = GenomicSet.read_bed(input.scannable)
        intervals = intervals & scannable
        intervals.write_bed(output[0])


rule intervals_recipe_v32:
    input:
        projected="results/intervals/ELS_phastCons_43p_conserved_20_mmseqs2_s75/{g}.parquet",
        scannable="results/intervals/scannable/{g}.bed.gz",
    output:
        "results/intervals/recipe/v32/{g}.bed.gz",
    run:
        intervals = GenomicSet.read_parquet(input.projected)
        intervals = intervals.resize(255)
        scannable = GenomicSet.read_bed(input.scannable)
        intervals = intervals & scannable
        intervals.write_bed(output[0])


rule intervals_recipe_v33:
    input:
        projected="results/intervals/ELS_phastCons_43p_conserved_50_mmseqs2_s75/{g}.parquet",
        scannable="results/intervals/scannable/{g}.bed.gz",
    output:
        "results/intervals/recipe/v33/{g}.bed.gz",
    run:
        intervals = GenomicSet.read_parquet(input.projected)
        intervals = intervals.resize(255)
        scannable = GenomicSet.read_bed(input.scannable)
        intervals = intervals & scannable
        intervals.write_bed(output[0])


rule all_intervals_recipe_curation_sweep_mammals_seg20:
    """Convenience target for the v31/v32/v33 curation sweep on mammals_seg20:
    every per-species recipe bed.gz across the three new mappings. Excludes
    v30 (already built) and the HF upload step. Used by
    sky/run_mmseqs2_mammals_seg20_curation_sweep.yaml.
    """
    input:
        expand(
            "results/intervals/recipe/{recipe}/{g}.bed.gz",
            recipe=["v31", "v32", "v33"],
            g=genome_sets.get("mammals_seg20", []),
        ),
