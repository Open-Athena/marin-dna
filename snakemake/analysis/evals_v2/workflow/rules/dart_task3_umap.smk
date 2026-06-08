"""DART-Eval Task 3 cell-type UMAP (interpretation, issue #298).

The cell-type-discrimination analogue of rules/embedding_umap.smk: a GPU
embedding rule + CPU UMAP-fit + CPU plot rule, kept OFF the default ``rule all``
(metrics-only) so they never perturb score/metric reruns. Build explicitly:

    snakemake dart_umap                                                    # every model x split x ctx
    snakemake results/plots/dart_umap/<model>/<split>/ctx<N>/celltype.svg  # one plot (split = train|validation|test)

The *same* model is embedded at each ``dart_umap.context_sizes`` entry (e.g.
255 + 500): each arm sets ``window_size = n_center_bp = ctx`` so the model sees
``ctx`` bp and every DNA token is mean-pooled (whole-window pooling). 255 bp is
exp136's native context; 500 bp is the full peak — ~2x the training context
(RoPE extrapolation), which ``check_window_fits`` allows with a warning. The
``{split}`` wildcard selects the HF split (train|validation|test).

Reuses the embedding kernel (``compute_region_embeddings``) + UMAP fit
(``fit_umap``) from embedding_umap, the ``download_model`` rule for the
checkpoint, the GRCh38 reference straight from S3 (needs ``--group genome-s3``),
and the labeled peak windows from HuggingFace. ``fit_umap`` needs the optional
``umap`` group (``--group umap``).
"""


rule compute_dart_embeddings:
    input:
        checkpoint="results/checkpoints/{model}",
    output:
        "results/interpretation/dart_umap/{model}/{split}/ctx{ctx}/embeddings.parquet",
    wildcard_constraints:
        model="|".join(DART_UMAP_MODELS),
        split="train|validation|test",
        ctx="|".join(str(c) for c in DART_UMAP_CONTEXT_SIZES),
    threads: config["inference"]["num_workers"]
    params:
        # Output-affecting fields only (snakemake `params` rerun trigger);
        # batch_size is execution-only and read inside `run:`. `split` is a
        # wildcard (in the output path), so it already triggers reruns — no param.
        dataset=DART_UMAP_CFG.get("dataset", "bolinas-dna/evals_dart_task3"),
        layer_index=DART_UMAP_CFG.get("layer_index", -1),
    run:
        from marin_dna.pipelines.evals.dart_task3_umap import (
            check_window_fits,
            load_dart_regions,
            read_position_limits,
        )
        from marin_dna.pipelines.evals.embedding_umap import (
            compute_region_embeddings,
        )

        ctx = int(wildcards.ctx)
        # Whole-window pooling: feed `ctx` bp and mean-pool every DNA token
        # (n_center_bp == window_size). Guard the position budget before GPU time:
        # a RoPE model over-budget warns + extrapolates (OOD); non-RoPE hard-fails.
        mpe, uses_rope = read_position_limits(input.checkpoint)
        check_window_fits(mpe, ctx, uses_rope=uses_rope)
        regions = load_dart_regions(params.dataset, split=wildcards.split)
        df = compute_region_embeddings(
            checkpoint_path=input.checkpoint,
            # S3 URI; pyfaidx + fsspec/s3fs reads sequence by byte-range.
            genome_path=config["genome_path"],
            regions=regions,
            window_size=ctx,
            layer_index=int(params.layer_index),
            n_center_bp=ctx,
            batch_size=DART_UMAP_CFG.get(
                "batch_size", config["inference"]["batch_size"]
            ),
            num_workers=config["inference"]["num_workers"],
            torch_compile=config["inference"].get("torch_compile", False),
        )
        df.to_parquet(output[0])
        print(
            f"[dart_umap] {wildcards.model} {wildcards.split} ctx{ctx}: "
            f"{df.shape[0]} windows x {df.shape[1] - 4} embedding dims"
        )


rule fit_dart_umap:
    input:
        "results/interpretation/dart_umap/{model}/{split}/ctx{ctx}/embeddings.parquet",
    output:
        "results/interpretation/dart_umap/{model}/{split}/ctx{ctx}/umap_coords.parquet",
    run:
        from marin_dna.pipelines.evals.embedding_umap import fit_umap

        emb = pd.read_parquet(input[0])
        coords = fit_umap(emb, random_state=DART_UMAP_CFG.get("random_state", 42))
        coords.to_parquet(output[0])
        print(
            f"[dart_umap] {wildcards.model} {wildcards.split} ctx{wildcards.ctx}: "
            f"fit UMAP on {len(coords)} points"
        )


rule plot_dart_umap:
    input:
        "results/interpretation/dart_umap/{model}/{split}/ctx{ctx}/umap_coords.parquet",
    output:
        "results/plots/dart_umap/{model}/{split}/ctx{ctx}/celltype.svg",
    run:
        from marin_dna.pipelines.evals.dart_task3_umap import plot_dart_umap

        coords = pd.read_parquet(input[0])
        plot_dart_umap(coords, output[0], dpi=DART_UMAP_CFG.get("dpi", 200))


rule dart_umap:
    """Cell-type UMAP for every (model, split, context size). Not in `rule all`."""
    input:
        [
            f"results/plots/dart_umap/{model}/{split}/ctx{ctx}/celltype.svg"
            for model in DART_UMAP_MODELS
            for split in DART_UMAP_SPLITS
            for ctx in DART_UMAP_CONTEXT_SIZES
        ],
