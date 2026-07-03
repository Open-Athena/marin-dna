"""Arm B (human-CDS projection) of the issue #353 CDS projection-vs-annotation
experiment.

A SELF-CONTAINED DUPLICATE of the interval-projection + dataset chain, isolated
end-to-end under ``results/cds_projection/…`` with its own HF prefix, so it
never clashes on S3 with the shared training_dataset outputs (the
"duplicate rules must change output paths" rule for an S3 default-storage
pipeline). It reads shared files — genomes, chrom sizes, and the existing v5
human CDS recipe bed — ONLY as inputs (reading never clashes), and reuses
library helpers + the generic ``compress_shard`` rule unchanged.

Per target species ``g`` in the ``animals_order204`` set:

    results/intervals/recipe/v5/{human}.bed.gz        (shared v5 human CDS, read-only)
        │  makewindows 255/128  →  results/cds_projection/source_windows.bed
        │  twoBitToFa           →  results/cds_projection/query.fa
        ▼
    mmseqs2 (query vs results/genome/{g}.fa, v30 flags)  →  align/{g}.hits.tsv
        │  best-hit + project + midpoint-resize to 255 (resize_dataframe)
        ▼
    results/cds_projection/intervals/{g}.bed.gz       (g == human: identity = source windows)
        │  twoBitToFa -bedPos   →  seq/{g}.fa
        │  load_fasta + add_rc  →  dataset_genome/{g}.parquet
        ▼
    merge + shuffle + shard  →  compress_shard (shared)  →  hf upload-large-folder

Target: ``all_cds_projection``.
"""

CDS = config["cds_projection"]
CDS_SOURCE = CDS["source_genome"]
CDS_SPECIES = genome_sets[CDS["genome_set"]]
CDS_TARGETS = [g for g in CDS_SPECIES if g != CDS_SOURCE]
CDS_TARGET_LEN = int(CDS["target_len"])
# Dataset cohorts (issue #353 sweep): cohort name -> genome_set to merge. All
# cohorts reuse the same per-genome projection parquets (built over the full
# `genome_set`); a cohort just row-filters the merge to its species.
CDS_COHORTS = CDS["cohorts"]

assert CDS_SOURCE in CDS_SPECIES, (
    f"cds_projection source_genome {CDS_SOURCE!r} must be in genome_set "
    f"{CDS['genome_set']!r} (so the human identity row exists)"
)


rule cds_proj_source_windows:
    """Window the shared v5 human CDS recipe bed to 255/128 — the projection query set.

    Reads (does not write) the existing ``results/intervals/recipe/v5/{human}.bed.gz``.
    """
    input:
        f"results/intervals/recipe/{CDS['source_recipe']}/{CDS_SOURCE}.bed.gz",
    output:
        "results/cds_projection/source_windows.bed",
    params:
        w=CDS["window_size"],
        s=CDS["step_size"],
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        """
        mkdir -p $(dirname {output})
        bedtools makewindows -b {input} -w {params.w} -s {params.s} | \
        awk 'BEGIN {{OFS="\\t"}} $3-$2 == {params.w} {{print $1, $2, $3, $1":"$2"-"$3}}' > {output}
        """


rule cds_proj_query_fa:
    """Per-window human query FASTA (record id = chrom:start-end)."""
    input:
        twobit=f"results/genome/{CDS_SOURCE}.2bit",
        bed="results/cds_projection/source_windows.bed",
    output:
        temp("results/cds_projection/query.fa"),
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        """
        mkdir -p $(dirname {output})
        twoBitToFa {input.twobit} {output} -bed={input.bed}
        """


rule cds_proj_align:
    """mmseqs2 nucleotide search of the human query windows against one target genome.

    Duplicate of ``align_intervals_mmseqs2`` (interval_alignment.smk) with the v30
    flags, isolated tmpdir + output. Human is excluded (no self-alignment).
    """
    input:
        query="results/cds_projection/query.fa",
        target="results/genome/{g}.fa",
    output:
        tsv=temp("results/cds_projection/align/{g}.hits.tsv"),
    params:
        sensitivity=CDS["sensitivity"],
        max_accept=CDS["max_accept"],
        split_memory_limit=CDS["split_memory_limit"],
    # ~8 threads/search lets ~4 searches run concurrently on a 32-core box
    # (each peaks ~34 GB RAM — see config.cloud mem_mb); much faster than one
    # 32-thread search at a time across 203 species.
    threads: 8
    resources:
        mem_mb=CDS["mem_mb"],
    wildcard_constraints:
        g="|".join(CDS_TARGETS),
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        """
        mkdir -p $(dirname {output.tsv})
        TMP=$(mktemp -d -t cds_mmseqs2_{wildcards.g}_XXXX)
        trap 'rm -rf "$TMP"' EXIT

        mmseqs createdb {input.target} $TMP/targetDB --mask-lower-case 1
        mmseqs createdb {input.query}  $TMP/queryDB  --mask-lower-case 1

        mmseqs search \
            $TMP/queryDB $TMP/targetDB $TMP/resultDB $TMP/search_tmp \
            --search-type 3 \
            --strand 2 \
            --mask-lower-case 1 \
            --split-memory-limit {params.split_memory_limit} \
            -s {params.sensitivity} \
            --max-accept {params.max_accept} \
            --threads {threads}

        mmseqs convertalis \
            $TMP/queryDB $TMP/targetDB $TMP/resultDB {output.tsv} \
            --format-output "query,target,tstart,tend,bits,evalue,fident,qcov,tcov"
        """


rule cds_proj_intervals:
    """Best hit per query, projected to target coords and midpoint-resized to 255 bp.

    Resizing (``resize_dataframe``) coerces each hit back to exactly the model
    context length, so an indel-shortened hit is centred rather than dropped.
    Contigs shorter than the target length can't hold a 255 bp window and are
    filtered out.
    """
    input:
        tsv="results/cds_projection/align/{g}.hits.tsv",
        chrom_sizes="results/chrom_sizes/{g}.tsv",
    output:
        "results/cds_projection/intervals/{g}.bed.gz",
    wildcard_constraints:
        g="|".join(CDS_TARGETS),
    run:
        import gzip

        from marin_dna.pipelines.alignment.mmseqs2 import (
            best_hit_per_query,
            parse_mmseqs2_hits,
            project_hits_to_intervals,
        )
        from marin_dna.pipelines.projection.resize import resize_dataframe

        best = best_hit_per_query(project_hits_to_intervals(parse_mmseqs2_hits(input.tsv)))

        sizes: dict[str, int] = {}
        with open(input.chrom_sizes) as f:
            for line in f:
                c, s = line.rstrip("\n").split("\t")
                sizes[c] = int(s)

        mkdir = os.path.dirname(output[0])
        os.makedirs(mkdir, exist_ok=True)

        if best.height == 0:
            with gzip.open(output[0], "wt") as fh:
                pass
        else:
            resized = resize_dataframe(
                best.rename({"start": "t_start", "end": "t_end"})
                .with_columns(
                    pl.col("chrom").replace_strict(sizes, default=None).alias("t_src_size")
                )
                .drop_nulls("t_src_size")
                .filter(pl.col("t_src_size") >= CDS_TARGET_LEN),
                CDS_TARGET_LEN,
            ).sort(["chrom", "t_start", "t_end"])
            with gzip.open(output[0], "wt") as fh:
                for c, s, e in resized.select(["chrom", "t_start", "t_end"]).iter_rows():
                    fh.write(f"{c}\t{s}\t{e}\n")
        print(f"  cds_projection → {wildcards.g}: {best.height:,} best hits")


rule cds_proj_intervals_human:
    """Human identity row: the projection of human onto human is the query windows themselves."""
    input:
        "results/cds_projection/source_windows.bed",
    output:
        f"results/cds_projection/intervals/{CDS_SOURCE}.bed.gz",
    shell:
        """
        mkdir -p $(dirname {output})
        gzip -c {input} > {output}
        """


rule cds_proj_seq:
    """Extract the 255 bp target sequence for each projected interval (+ strand)."""
    input:
        twobit="results/genome/{g}.2bit",
        bed="results/cds_projection/intervals/{g}.bed.gz",
    output:
        temp("results/cds_projection/seq/{g}.fa"),
    wildcard_constraints:
        g="|".join(CDS_SPECIES),
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        """
        mkdir -p $(dirname {output})
        zcat {input.bed} | awk 'BEGIN {{OFS="\\t"}} {{print $1, $2, $3, "."}}' > {output}.bed
        twoBitToFa {input.twobit} {output} -bed={output}.bed -bedPos
        rm -f {output}.bed
        """


rule cds_proj_parquet:
    """FASTA → per-genome parquet, adding reverse complements (mirrors make_parquet)."""
    input:
        "results/cds_projection/seq/{g}.fa",
    output:
        "results/cds_projection/dataset_genome/{g}.parquet",
    wildcard_constraints:
        g="|".join(CDS_SPECIES),
    run:
        df = load_fasta(input[0]).to_frame().reset_index(names="id")
        if len(df) == 0:
            pl.DataFrame(
                {"id": [], "seq": []}, schema={"id": pl.String, "seq": pl.String}
            ).write_parquet(output[0])
        else:
            df.id = df.id.astype(str)
            if CDS["add_rc"]:
                df = add_rc(df)
            pl.from_pandas(df[["id", "seq"]]).write_parquet(output[0])


rule cds_proj_merge:
    """Merge one cohort's per-genome parquets, shuffle, shard into JSONL (mirrors merge_datasets)."""
    input:
        lambda w: expand(
            "results/cds_projection/dataset_genome/{g}.parquet",
            g=genome_sets[CDS_COHORTS[w.cohort]],
        ),
    output:
        temp(local(expand(
            "results/cds_projection/dataset/{{cohort}}/data/train/{shard}.jsonl",
            shard=SHARDS,
        ))),
    wildcard_constraints:
        cohort="|".join(CDS_COHORTS),
    threads: workflow.cores
    run:
        df = pl.concat(
            tqdm((pl.read_parquet(p) for p in input), total=len(input)),
        ).sample(fraction=1, shuffle=True, seed=config["shuffle_seed"])
        split_pairs = get_array_split_pairs(len(df), len(output))
        for path, (start, end) in tqdm(zip(output, split_pairs), total=len(output)):
            df.slice(start, end - start).write_ndjson(path)


rule cds_proj_hf_upload:
    """Upload one cohort's shards to its HF repo (`{hf_prefix}-{cohort}`)."""
    input:
        local(expand(
            "results/cds_projection/dataset/{{cohort}}/data/train/{shard}.jsonl.zst",
            shard=SHARDS,
        )),
    output:
        "results/cds_projection/upload.done/{cohort}",
    params:
        name=lambda w: f"{CDS['hf_prefix']}-{w.cohort}",
        data_dir=lambda w: f"results/cds_projection/dataset/{w.cohort}",
    wildcard_constraints:
        cohort="|".join(CDS_COHORTS),
    threads: workflow.cores
    shell:
        """
        hf upload-large-folder {params.name} --repo-type dataset {params.data_dir}
        mkdir -p $(dirname {output})
        touch {output}
        """


rule all_cds_projection_parquets:
    """Full projection through per-genome parquets, WITHOUT the HF upload — the
    Arm B training data + per-species yields (reach-vs-divergence). Draft the HF
    dataset card and get sign-off before running `all_cds_projection` (which
    merges, shards, and pushes to HF)."""
    input:
        expand("results/cds_projection/dataset_genome/{g}.parquet", g=CDS_SPECIES),


rule cds_proj_hf_readme:
    """Generate the HF dataset card for one cohort (exact n_rows from parquet footers)."""
    input:
        lambda w: expand(
            "results/cds_projection/dataset_genome/{g}.parquet",
            g=genome_sets[CDS_COHORTS[w.cohort]],
        ),
    output:
        "results/cds_projection/readme/{cohort}/README.md",
    wildcard_constraints:
        cohort="|".join(CDS_COHORTS),
    run:
        from marin_dna.pipelines.training_dataset.cds_projection import (
            build_cds_projection_readme,
        )
        from marin_dna.pipelines.training_dataset.hf_readme import count_parquet_rows

        prefix = workflow.storage_settings.default_storage_prefix
        assert prefix, "cds_proj_hf_readme needs the S3 default-storage profile"
        species = genome_sets[CDS_COHORTS[wildcards.cohort]]
        uris = [
            f"{prefix.rstrip('/')}/results/cds_projection/dataset_genome/{g}.parquet"
            for g in species
        ]
        md = build_cds_projection_readme(
            cohort=wildcards.cohort,
            n_species=len(species),
            n_rows=count_parquet_rows(uris),
            commit_sha=GIT_COMMIT_SHA,
            is_vertebrate_subset=(wildcards.cohort != "all204"),
        )
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], "w") as f:
            f.write(md)


rule cds_proj_hf_upload_readme:
    """Push the cohort's dataset card (`hf upload-large-folder` skips top-level files)."""
    input:
        "results/cds_projection/readme/{cohort}/README.md",
    output:
        "results/cds_projection/upload.done/readme/{cohort}",
    params:
        name=lambda w: f"{CDS['hf_prefix']}-{w.cohort}",
    wildcard_constraints:
        cohort="|".join(CDS_COHORTS),
    shell:
        """
        hf upload {params.name} {input} README.md --repo-type dataset
        mkdir -p $(dirname {output})
        touch {output}
        """


rule all_cds_projection:
    input:
        expand("results/cds_projection/upload.done/{cohort}", cohort=list(CDS_COHORTS)),
        expand(
            "results/cds_projection/upload.done/readme/{cohort}",
            cohort=list(CDS_COHORTS),
        ),
