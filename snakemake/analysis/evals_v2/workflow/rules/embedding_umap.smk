"""Embedding UMAP (GPN-Star Fig 4A/4B) — interpretation, issue #246.

A GPU embedding rule + CPU UMAP-fit + CPU plot rules, parallel to nuc_dep
(rules/interpretation.smk) and kept OFF the default ``rule all`` (which is
metrics-only) so they never perturb score/metric reruns. Build explicitly:

    snakemake umap                                    # every model, both plots
    snakemake results/plots/umap/<model>/region.svg   # one plot

``compute_region_embeddings`` reuses the existing ``download_model`` rule for
the checkpoint, reads the GRCh38 reference straight from S3 (needs
``--group genome-s3``), and pulls the labeled 100 bp region windows from
HuggingFace. ``fit_umap`` needs the optional ``umap`` group (``--group umap``).
"""


rule compute_region_embeddings:
    input:
        checkpoint="results/checkpoints/{model}",
    output:
        "results/interpretation/umap/{model}/embeddings.parquet",
    wildcard_constraints:
        model="|".join(UMAP_MODELS),
    threads: config["inference"]["num_workers"]
    params:
        # Output-affecting fields only (snakemake `params` rerun trigger);
        # batch_size is execution-only and read inside `run:`.
        dataset=UMAP_CFG.get("dataset", "songlab/gpn-star-umap-regions"),
        window_size=lambda wc: get_model_config(wc.model)["window_size"],
        layer_index=UMAP_CFG.get("layer_index", -1),
        n_center_bp=UMAP_CFG.get("n_center_bp", 100),
    run:
        from marin_dna_evals.embedding_umap import (
            compute_region_embeddings,
            load_umap_regions,
        )

        regions = load_umap_regions(params.dataset)
        df = compute_region_embeddings(
            checkpoint_path=input.checkpoint,
            # S3 URI; pyfaidx + fsspec/s3fs reads sequence by byte-range.
            genome_path=config["genome_path"],
            regions=regions,
            window_size=int(params.window_size),
            layer_index=int(params.layer_index),
            n_center_bp=int(params.n_center_bp),
            batch_size=UMAP_CFG.get(
                "batch_size", config["inference"]["batch_size"]
            ),
            num_workers=config["inference"]["num_workers"],
            torch_compile=config["inference"].get("torch_compile", False),
            bf16=config["inference"]["bf16"],
        )
        df.to_parquet(output[0])
        print(
            f"[umap] {wildcards.model}: {df.shape[0]} windows x "
            f"{df.shape[1] - 5} embedding dims"
        )


rule fit_umap:
    input:
        "results/interpretation/umap/{model}/embeddings.parquet",
    output:
        "results/interpretation/umap/{model}/umap_coords.parquet",
    run:
        from marin_dna_evals.embedding_umap import fit_umap

        emb = pd.read_parquet(input[0])
        coords = fit_umap(emb, random_state=UMAP_CFG.get("random_state", 42))
        coords.to_parquet(output[0])
        print(f"[umap] {wildcards.model}: fit UMAP on {len(coords)} points")


rule plot_umap:
    input:
        "results/interpretation/umap/{model}/umap_coords.parquet",
    output:
        "results/plots/umap/{model}/{color_by}.svg",
    wildcard_constraints:
        color_by="region|conservation",
    run:
        from marin_dna_evals.embedding_umap import plot_umap

        coords = pd.read_parquet(input[0])
        plot_umap(
            coords,
            output[0],
            color_by=wildcards.color_by,
            dpi=UMAP_CFG.get("dpi", 200),
        )


rule umap:
    """Both plots (region + conservation) for every umap model. Not in `rule all`."""
    input:
        [
            f"results/plots/umap/{model}/{color_by}.svg"
            for model in UMAP_MODELS
            for color_by in ("region", "conservation")
        ],
