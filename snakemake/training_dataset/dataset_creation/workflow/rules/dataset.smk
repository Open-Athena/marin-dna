rule prepare_intervals_for_window_seq:
    """Add placeholder name column ('.') required by twoBitToFa -bedPos.

    `mkdir -p` is needed because with the S3 default-storage backend
    Snakemake doesn't materialize the local parent directory of a
    `send to storage` output before the shell runs — same gotcha that
    hits mmseqs2 createdb in interval_alignment.smk.
    """
    input:
        "results/intervals/{intervals}/{g}.bed.gz",
    output:
        temp("results/intervals_for_window_seq/{intervals}/{g}.bed.gz"),
    shell:
        """
        mkdir -p $(dirname {output})
        zcat {input} |
        awk 'BEGIN {{OFS="\\t"}} {{print $1, $2, $3, "."}}' |
        gzip > {output}
        """


rule window_seq:
    """Extract sequences from 2bit genome using windowed BED intervals."""
    input:
        "results/genome/{g}.2bit",
        "results/intervals_for_window_seq/windows/recipe/{recipe}/{w}/{s}/{g}.bed.gz",
    output:
        temp("results/intervals_seq/{recipe}/{w}/{s}/{g}.fa"),
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        """
        mkdir -p $(dirname {output})
        twoBitToFa {input[0]} {output} -bed={input[1]} -bedPos
        """


rule make_parquet:
    """Convert FASTA to parquet, optionally adding reverse complements."""
    input:
        "results/intervals_seq/{intervals}/{g}.fa",
    output:
        "results/dataset_genome/{intervals}/{g}.parquet",
    run:
        df = load_fasta(input[0]).to_frame().reset_index(names="id")
        if len(df) == 0:
            pl.DataFrame(
                {"id": [], "seq": []}, schema={"id": pl.String, "seq": pl.String}
            ).write_parquet(output[0])
        else:
            df.id = df.id.astype(str)
            if config["add_rc"]:
                df = add_rc(df)
            pl.from_pandas(df[["id", "seq"]]).write_parquet(output[0])


rule create_functional_validation:
    """Create validation parquet with phyloP conservation case encoding.

    Sequences are subsampled from the human genome. For each base,
    uppercase iff phyloP >= threshold, lowercase otherwise (NaN -> lowercase).
    """
    input:
        fasta="results/intervals_seq/{recipe}/{w}/{s}/" + VALIDATION_GENOME + ".fa",
        chrom_mapping=local("config/human_chrom_mapping.tsv"),
        bigwig=config["validation"]["conservation_bigwig"],
    output:
        "results/validation/{recipe}/{w}/{s}/validation.parquet",
    run:
        val_config = config["validation"]
        threshold = val_config["phylop_threshold"]

        # Load chrom name mapping (RefSeq -> UCSC)
        chrom_map = dict(pl.read_csv(input.chrom_mapping, separator="\t").iter_rows())

        # Load and subsample sequences
        series = load_fasta(input.fasta)
        df = series.to_frame().reset_index(names="id")
        if len(df) == 0:
            pl.DataFrame(
                {"id": [], "seq": []}, schema={"id": pl.String, "seq": pl.String}
            ).write_parquet(output[0])
            return

        df.id = df.id.astype(str)
        max_samples = val_config["max_samples"]

        # Filter to sequences on mapped chromosomes before subsampling
        df["chrom"] = df["id"].apply(lambda x: x.rsplit(":", 1)[0])
        df = df[df["chrom"].isin(chrom_map)]
        df = df.drop(columns=["chrom"])

        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=val_config["seed"])

        bw = pyBigWig.open(input.bigwig)


        def encode_case(row):
            """Encode conservation as case: uppercase iff phyloP >= threshold."""
            chrom_refseq, coords = row["id"].rsplit(":", 1)
            start, end = (int(x) for x in coords.split("-"))
            chrom_ucsc = chrom_map[chrom_refseq]
            scores = bw.values(chrom_ucsc, start, end)
            # NaN (missing data) compares False, so NaN -> lowercase
            return "".join(
                b.upper() if s >= threshold else b.lower()
                for b, s in zip(row["seq"], scores)
            )


        df["seq"] = df.apply(encode_case, axis=1)
        bw.close()

        pl.from_pandas(df[["id", "seq"]]).write_parquet(output[0])


rule merge_datasets:
    """Merge per-genome training parquets, shuffle, and shard into JSONL."""
    # Use explicit {recipe}/{w}/{s} wildcards (each bounded by '/' in the
    # path template) instead of a single {intervals} wildcard whose slashes
    # would race with the genome_set wildcard for greedy matching, producing
    # bogus splits like `genome_set=enhancer_seg_mammals_v1/v20/255, intervals=128`.
    input:
        lambda wildcards: expand(
            "results/dataset_genome/{recipe}/{w}/{s}/{g}.parquet",
            recipe=wildcards.recipe,
            w=wildcards.w,
            s=wildcards.s,
            g=genome_sets[wildcards.genome_set],
        ),
    output:
        temp(local(
            expand(
                "results/dataset/{{genome_set}}/{{recipe}}/{{w}}/{{s}}/data/train/{shard}.jsonl",
                shard=SHARDS,
            )
        )),
    threads: workflow.cores
    run:
        df = pl.concat(
            tqdm(
                (pl.read_parquet(path) for path in input),
                total=len(input),
            ),
        ).sample(fraction=1, shuffle=True, seed=config["shuffle_seed"])
        split_pairs = get_array_split_pairs(len(df), len(output))
        for path, (start, end) in tqdm(zip(output, split_pairs), total=len(output)):
            df.slice(start, end - start).write_ndjson(path)


rule compress_shard:
    input:
        local("{anything}.jsonl"),
    output:
        local("{anything}.jsonl.zst"),
    threads: 8
    shell:
        "zstd -T{threads} {input} -o {output}"


rule hf_upload_training:
    """Upload training dataset shards to HuggingFace."""
    input:
        local(expand(
            "results/dataset/{{genome_set}}/{{recipe}}/{{w}}/{{s}}/data/train/{shard}.jsonl.zst",
            shard=SHARDS,
        )),
    output:
        # Explicit `touch {output}` in shell instead of snakemake's `touch()`
        # wrapper -- with default-storage-provider=s3 the wrapper doesn't
        # auto-create the marker, leaving snakemake to declare the output
        # missing even though the upload succeeded.
        "results/upload.done/training/{genome_set}/{recipe}/{w}/{s}",
    params:
        name=lambda wildcards: (
            config["output_hf_prefix"]
            + "-genome_set-" + wildcards.genome_set
            + "-intervals-" + f"{wildcards.recipe}_{wildcards.w}_{wildcards.s}"
        ),
        data_dir=lambda wildcards: (
            f"results/dataset/{wildcards.genome_set}/{wildcards.recipe}/{wildcards.w}/{wildcards.s}"
        ),
    threads: workflow.cores
    shell:
        """
        hf upload-large-folder {params.name} --repo-type dataset {params.data_dir}
        mkdir -p $(dirname {output})
        touch {output}
        """


rule hf_upload_validation:
    """Upload validation dataset to HuggingFace."""
    input:
        "results/validation/{intervals}/validation.parquet",
    output:
        touch("results/upload.done/validation/{intervals}"),
    params:
        name=lambda wildcards: (
            config["output_hf_prefix"]
            + "-validation"
            + "-intervals-" + wildcards.intervals.replace("/", "_")
        ),
    shell:
        "hf upload {params.name} {input} validation.parquet --repo-type dataset"


# ============================================================================
# Per-repo HuggingFace dataset cards (README.md)
#
# `hf upload-large-folder` (used by `hf_upload_training`) silently skips
# top-level files, so the card is generated here and pushed by a *separate*
# `hf upload ... README.md` in dedicated rules. These rules depend on the
# shard/parquet upload markers (so the repo already exists) and write their
# own `*.done` markers — leaving `hf_upload_training` / `hf_upload_validation`
# untouched, so wiring this in does NOT re-trigger any (expensive) shard
# re-upload on the next pipeline run.
# ============================================================================


rule training_hf_readme:
    """Generate the per-repo HF dataset card for a training dataset.

    The exact ``n_samples`` is counted from the per-genome parquet *footers*
    in S3 (cheap metadata reads, no data download) — the shard row total
    equals the sum of these by construction. We gate on the (already-written,
    cheap) ``upload.done/training`` marker purely for ordering: it guarantees
    the per-genome parquets exist before we scan them, without materialising
    the tens-of-GB of shard/parquet data as a snakemake input. ``ancient()``
    keeps that gate from making snakemake explore (or rebuild) the upstream
    data chain just to (re)generate a card.
    """
    input:
        uploaded=ancient(
            "results/upload.done/training/{genome_set}/{recipe}/{w}/{s}"
        ),
    output:
        "results/readme/training/{genome_set}/{recipe}/{w}/{s}/README.md",
    params:
        commit_sha=GIT_COMMIT_SHA,
        hf_prefix=config["output_hf_prefix"],
        n_genomes=lambda wildcards: len(genome_sets[wildcards.genome_set]),
        n_shards=config["n_shards"],
        seed=config["shuffle_seed"],
        add_rc=config["add_rc"],
    run:
        from marin_dna.pipelines.training_dataset.hf_readme import (
            count_parquet_rows,
            write_training_readme,
        )

        prefix = workflow.storage_settings.default_storage_prefix.rstrip("/")
        parquet_uris = [
            f"{prefix}/results/dataset_genome/{wildcards.recipe}/{wildcards.w}/{wildcards.s}/{g}.parquet"
            for g in genome_sets[wildcards.genome_set]
        ]
        n_samples = count_parquet_rows(parquet_uris)

        write_training_readme(
            output[0],
            genome_set=wildcards.genome_set,
            recipe=wildcards.recipe,
            window=int(wildcards.w),
            stride=int(wildcards.s),
            hf_prefix=params.hf_prefix,
            commit_sha=params.commit_sha,
            n_genomes=params.n_genomes,
            n_samples=n_samples,
            n_shards=int(params.n_shards),
            seed=int(params.seed),
            add_rc=bool(params.add_rc),
        )


rule hf_upload_training_readme:
    """Push the training dataset card to its HF repo.

    Depends only on the generated card, *not* on the shard-upload marker:
    ``hf upload`` creates/targets the repo regardless, and gating on the data
    marker would couple this to the (separate) upload rule's rerun state. In a
    full ``rule all`` run the shard upload still runs via its own target.
    """
    input:
        readme="results/readme/training/{genome_set}/{recipe}/{w}/{s}/README.md",
    output:
        "results/upload.done/training_readme/{genome_set}/{recipe}/{w}/{s}",
    params:
        name=lambda wildcards: (
            config["output_hf_prefix"]
            + "-genome_set-" + wildcards.genome_set
            + "-intervals-" + f"{wildcards.recipe}_{wildcards.w}_{wildcards.s}"
        ),
    shell:
        """
        hf upload {params.name} {input.readme} README.md --repo-type dataset
        mkdir -p $(dirname {output})
        touch {output}
        """


rule validation_hf_readme:
    """Generate the per-repo HF dataset card for a validation dataset.

    ``n_samples`` is the exact row count of ``validation.parquet`` (a small,
    <=``max_samples`` subsample) read straight from the parquet input. We gate
    on the parquet itself (not the ``upload.done/validation`` marker, which is
    written via snakemake's S3-incompatible ``touch()`` wrapper and so reads as
    perpetually missing). A plain input — rather than ``ancient()`` — because
    when the parquet is (spuriously, on a fresh checkout) flagged for rebuild,
    an ``ancient()`` wrapper makes snakemake skip scheduling this card; the
    plain input schedules correctly and on the real pipeline host the parquet
    post-dates its config inputs, so no rebuild fires.
    """
    input:
        parquet="results/validation/{recipe}/{w}/{s}/validation.parquet",
    output:
        "results/readme/validation/{recipe}/{w}/{s}/README.md",
    params:
        commit_sha=GIT_COMMIT_SHA,
        hf_prefix=config["output_hf_prefix"],
        threshold=config["validation"]["phylop_threshold"],
        max_samples=config["validation"]["max_samples"],
        seed=config["validation"]["seed"],
        validation_genome=config["validation"]["genome"],
    run:
        from marin_dna.pipelines.training_dataset.hf_readme import (
            count_parquet_rows,
            write_validation_readme,
        )

        n_samples = count_parquet_rows([input.parquet])

        write_validation_readme(
            output[0],
            recipe=wildcards.recipe,
            window=int(wildcards.w),
            stride=int(wildcards.s),
            hf_prefix=params.hf_prefix,
            commit_sha=params.commit_sha,
            n_samples=n_samples,
            phylop_threshold=float(params.threshold),
            max_samples=int(params.max_samples),
            seed=int(params.seed),
            validation_genome=params.validation_genome,
        )


rule hf_upload_validation_readme:
    """Push the validation dataset card to its HF repo.

    Depends only on the generated card (see ``hf_upload_training_readme``).
    Notably it does *not* gate on ``upload.done/validation/...``: that marker
    is written via snakemake's ``touch()`` wrapper, which does not persist
    under the S3 storage provider, so depending on it would spuriously
    re-trigger ``create_functional_validation`` + ``hf_upload_validation``.
    """
    input:
        readme="results/readme/validation/{recipe}/{w}/{s}/README.md",
    output:
        "results/upload.done/validation_readme/{recipe}/{w}/{s}",
    params:
        name=lambda wildcards: (
            config["output_hf_prefix"]
            + "-validation"
            + "-intervals-" + f"{wildcards.recipe}_{wildcards.w}_{wildcards.s}"
        ),
    shell:
        """
        hf upload {params.name} {input.readme} README.md --repo-type dataset
        mkdir -p $(dirname {output})
        touch {output}
        """
