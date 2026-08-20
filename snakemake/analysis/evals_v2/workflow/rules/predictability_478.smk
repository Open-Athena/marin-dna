"""Conservation × repeat predictability experiment (issue #478).

Additive, versioned, and off the default ``all`` target. Build
``predictability_478_pilot`` first; it runs only the 46M checkpoint and the
issue-274 regression gates. ``predictability_478`` expands the full ladder.
"""

P478_ROOT = f"results/predictability_478/{PREDICTABILITY_478_VERSION}"


rule build_predictability_478_joined:
    input:
        repeat_twobit=storage(PREDICTABILITY_478_CFG["repeat_twobit"]),
        cds_gtf=storage(PREDICTABILITY_478_CFG["cds_gtf"]),
    output:
        parquet=P478_ROOT + "/joined/{region}.parquet",
        manifest=P478_ROOT + "/joined/{region}.manifest.json",
    wildcard_constraints:
        region="|".join(PREDICTABILITY_478_DATASETS),
    threads: 2
    params:
        hf_repo=lambda wc: get_predictability_478_dataset_config(wc.region)["hf_repo"],
        hf_revision=lambda wc: get_predictability_478_dataset_config(wc.region)[
            "hf_revision"
        ],
        split=PREDICTABILITY_478_CFG["split"],
        window_size=PREDICTABILITY_478_CFG["window_size"],
        assembly=PREDICTABILITY_478_CFG["assembly"],
    run:
        import json
        from marin_dna_evals.joined_478 import build_joined_windows

        sequences = load_dataset(
            params.hf_repo,
            split=params.split,
            revision=params.hf_revision,
        ).to_pandas()
        joined, manifest = build_joined_windows(
            sequences,
            region=wildcards.region,
            window_size=int(params.window_size),
            repeat_twobit_path=input.repeat_twobit,
            cds_gtf_path=input.cds_gtf if wildcards.region == "cds" else None,
        )
        manifest.update(
            {
                "assembly": params.assembly,
                "hf_repo": params.hf_repo,
                "hf_revision": params.hf_revision,
                "split": params.split,
                "repeat_twobit": PREDICTABILITY_478_CFG["repeat_twobit"],
                "cds_gtf": (
                    PREDICTABILITY_478_CFG["cds_gtf"]
                    if wildcards.region == "cds"
                    else None
                ),
                "coordinate_system": "0-based half-open",
            }
        )
        joined.to_parquet(output.parquet, index=False)
        Path(output.manifest).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )


rule compute_predictability_478_atoms:
    input:
        checkpoint="results/checkpoints/{model}",
    output:
        fwd=P478_ROOT + "/atoms/{model}/{region}.fwd.parquet",
        rc=P478_ROOT + "/atoms/{model}/{region}.rc.parquet",
    wildcard_constraints:
        model="|".join(PREDICTABILITY_478_MODELS),
        region="|".join(PREDICTABILITY_478_DATASETS),
    threads: config["inference"]["num_workers"]
    resources:
        gpu=1,
    params:
        hf_repo=lambda wc: get_predictability_478_dataset_config(wc.region)["hf_repo"],
        hf_revision=lambda wc: get_predictability_478_dataset_config(wc.region)[
            "hf_revision"
        ],
        split=PREDICTABILITY_478_CFG["split"],
        window_size=PREDICTABILITY_478_CFG["window_size"],
    run:
        from marin_dna_evals.per_base import compute_hf_per_base_stats

        sequences = load_dataset(
            params.hf_repo,
            split=params.split,
            revision=params.hf_revision,
        ).to_pandas()
        atoms = compute_hf_per_base_stats(
            input.checkpoint,
            sequences,
            int(params.window_size),
            batch_size=get_predictability_478_batch_size(wildcards.model),
            num_workers=threads,
            torch_compile=config["inference"].get("torch_compile", False),
        )
        atoms["fwd"].to_parquet(output.fwd, index=False)
        atoms["rc"].to_parquet(output.rc, index=False)


rule check_predictability_478_ll_gap:
    input:
        atoms=P478_ROOT + "/atoms/{model}/{region}.fwd.parquet",
        cached="results/ll_gap/scores/{model}/{region}.parquet",
    output:
        P478_ROOT + "/regression/{model}/{region}.json",
    wildcard_constraints:
        model="|".join(PREDICTABILITY_478_MODELS),
        region="|".join(PREDICTABILITY_478_DATASETS),
    threads: 1
    params:
        hf_repo=lambda wc: get_predictability_478_dataset_config(wc.region)["hf_repo"],
        hf_revision=lambda wc: get_predictability_478_dataset_config(wc.region)[
            "hf_revision"
        ],
        split=PREDICTABILITY_478_CFG["split"],
    run:
        import json
        from marin_dna_evals.per_base import (
            aggregate_by_case,
            compare_ll_gap_cache,
        )

        sequences = load_dataset(
            params.hf_repo,
            split=params.split,
            revision=params.hf_revision,
        ).to_pandas()
        reconstructed = aggregate_by_case(pd.read_parquet(input.atoms), sequences)
        report = compare_ll_gap_cache(
            reconstructed,
            pd.read_parquet(input.cached),
        )
        report.update(
            {
                "model": wildcards.model,
                "region": wildcards.region,
                "hf_repo": params.hf_repo,
                "hf_revision": params.hf_revision,
            }
        )
        Path(output[0]).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )


rule analyze_predictability_478:
    input:
        joined=expand(
            P478_ROOT + "/joined/{region}.parquet",
            region=PREDICTABILITY_478_DATASETS,
        ),
        atoms=expand(
            P478_ROOT + "/atoms/{model}/{region}.{orientation}.parquet",
            model=PREDICTABILITY_478_MODELS,
            region=PREDICTABILITY_478_DATASETS,
            orientation=["fwd", "rc"],
        ),
        regression=expand(
            P478_ROOT + "/regression/{model}/{region}.json",
            model=PREDICTABILITY_478_MODELS,
            region=PREDICTABILITY_478_DATASETS,
        ),
    output:
        summary=P478_ROOT + "/analysis/summary.parquet",
        controlled=P478_ROOT + "/analysis/controlled.parquet",
        manifest=P478_ROOT + "/analysis/manifest.json",
    threads: 2
    run:
        import json
        from marin_dna_evals.analysis_478 import analyze_predictability_478

        joined_paths = dict(
            zip(PREDICTABILITY_478_DATASETS, input.joined, strict=True)
        )
        atom_keys = [
            (model, region, orientation)
            for model in PREDICTABILITY_478_MODELS
            for region in PREDICTABILITY_478_DATASETS
            for orientation in ("fwd", "rc")
        ]
        atom_paths = dict(zip(atom_keys, input.atoms, strict=True))
        summary, controlled, manifest = analyze_predictability_478(
            joined_paths,
            atom_paths,
            model_order=PREDICTABILITY_478_MODELS,
            window_size=PREDICTABILITY_478_CFG["window_size"],
            primary_start=PREDICTABILITY_478_CFG["primary_start"],
            primary_end_exclusive=PREDICTABILITY_478_CFG["primary_end_exclusive"],
            block_bp=PREDICTABILITY_478_CFG["bootstrap_block_bp"],
            bootstrap_replicates=PREDICTABILITY_478_CFG["bootstrap_replicates"],
            seed=PREDICTABILITY_478_CFG["bootstrap_seed"],
        )
        summary.to_parquet(output.summary, index=False)
        controlled.to_parquet(output.controlled, index=False)
        Path(output.manifest).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )


rule analyze_predictability_478_orientations:
    input:
        joined=expand(
            P478_ROOT + "/joined/{region}.parquet",
            region=PREDICTABILITY_478_DATASETS,
        ),
        atoms=expand(
            P478_ROOT + "/atoms/{model}/{region}.{orientation}.parquet",
            model=PREDICTABILITY_478_MODELS,
            region=PREDICTABILITY_478_DATASETS,
            orientation=["fwd", "rc"],
        ),
        regression=expand(
            P478_ROOT + "/regression/{model}/{region}.json",
            model=PREDICTABILITY_478_MODELS,
            region=PREDICTABILITY_478_DATASETS,
        ),
    output:
        summary=P478_ROOT + "/analysis/orientation_summary.parquet",
        controlled=P478_ROOT + "/analysis/orientation_controlled.parquet",
        agreement=P478_ROOT + "/analysis/orientation_agreement.parquet",
        manifest=P478_ROOT + "/analysis/orientation_manifest.json",
    threads: 2
    run:
        import json
        from marin_dna_evals.orientation_478 import (
            analyze_orientation_sensitivity_478,
        )

        joined_paths = dict(
            zip(PREDICTABILITY_478_DATASETS, input.joined, strict=True)
        )
        atom_keys = [
            (model, region, orientation)
            for model in PREDICTABILITY_478_MODELS
            for region in PREDICTABILITY_478_DATASETS
            for orientation in ("fwd", "rc")
        ]
        atom_paths = dict(zip(atom_keys, input.atoms, strict=True))
        summary, controlled, agreement, manifest = (
            analyze_orientation_sensitivity_478(
                joined_paths,
                atom_paths,
                model_order=PREDICTABILITY_478_MODELS,
                window_size=PREDICTABILITY_478_CFG["window_size"],
                primary_start=PREDICTABILITY_478_CFG["primary_start"],
                primary_end_exclusive=PREDICTABILITY_478_CFG[
                    "primary_end_exclusive"
                ],
                block_bp=PREDICTABILITY_478_CFG["bootstrap_block_bp"],
                bootstrap_replicates=PREDICTABILITY_478_CFG["bootstrap_replicates"],
                seed=PREDICTABILITY_478_CFG["orientation_bootstrap_seed"],
                top_fraction=PREDICTABILITY_478_CFG["orientation_top_fraction"],
                rank_sample_size=PREDICTABILITY_478_CFG[
                    "orientation_rank_sample_size"
                ],
            )
        )
        summary.to_parquet(output.summary, index=False)
        controlled.to_parquet(output.controlled, index=False)
        agreement.to_parquet(output.agreement, index=False)
        Path(output.manifest).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )


rule predictability_478_pilot:
    input:
        expand(
            P478_ROOT + "/joined/{region}.manifest.json",
            region=PREDICTABILITY_478_DATASETS,
        ),
        expand(
            P478_ROOT + "/regression/{model}/{region}.json",
            model=[PREDICTABILITY_478_PILOT_MODEL],
            region=PREDICTABILITY_478_DATASETS,
        ),


rule predictability_478:
    input:
        expand(
            P478_ROOT + "/regression/{model}/{region}.json",
            model=PREDICTABILITY_478_MODELS,
            region=PREDICTABILITY_478_DATASETS,
        ),
        P478_ROOT + "/analysis/manifest.json",
        P478_ROOT + "/figure/predictability.png",
        P478_ROOT + "/analysis/orientation_manifest.json",
        P478_ROOT + "/figure/orientation_sensitivity.png",


rule plot_predictability_478:
    input:
        summary=P478_ROOT + "/analysis/summary.parquet",
        controlled=P478_ROOT + "/analysis/controlled.parquet",
    output:
        P478_ROOT + "/figure/predictability.png",
    threads: 1
    run:
        from marin_dna_evals.figure_478 import plot_predictability_478

        plot_predictability_478(input.summary, input.controlled, output[0])


rule plot_predictability_478_orientations:
    input:
        summary=P478_ROOT + "/analysis/orientation_summary.parquet",
        controlled=P478_ROOT + "/analysis/orientation_controlled.parquet",
        agreement=P478_ROOT + "/analysis/orientation_agreement.parquet",
        averaged_controlled=P478_ROOT + "/analysis/controlled.parquet",
    output:
        P478_ROOT + "/figure/orientation_sensitivity.png",
    threads: 1
    run:
        from marin_dna_evals.figure_orientation_478 import (
            plot_orientation_sensitivity_478,
        )

        plot_orientation_sensitivity_478(
            input.summary,
            input.controlled,
            input.agreement,
            input.averaged_controlled,
            output[0],
        )


rule analyze_predictability_478_classification_pilot:
    input:
        joined=expand(
            P478_ROOT + "/joined/{region}.parquet",
            region=PREDICTABILITY_478_DATASETS,
        ),
        atoms=expand(
            P478_ROOT + "/atoms/{model}/{region}.{orientation}.parquet",
            model=PREDICTABILITY_478_MODELS[:2],
            region=PREDICTABILITY_478_DATASETS,
            orientation=["fwd", "rc"],
        ),
    output:
        metrics=P478_ROOT + "/classification/pilot_metrics.parquet",
        block_metrics=P478_ROOT + "/classification/pilot_block_metrics.parquet",
        manifest=P478_ROOT + "/classification/pilot_manifest.json",
    threads: 8
    resources:
        mem_mb=28000,
    run:
        import json
        from marin_dna_evals.classification_478 import (
            analyze_conservation_classification_478,
        )

        pilot_models = PREDICTABILITY_478_MODELS[:2]
        joined_paths = dict(
            zip(PREDICTABILITY_478_DATASETS, input.joined, strict=True)
        )
        atom_keys = [
            (model, region, orientation)
            for model in pilot_models
            for region in PREDICTABILITY_478_DATASETS
            for orientation in ("fwd", "rc")
        ]
        atom_paths = dict(zip(atom_keys, input.atoms, strict=True))
        metrics, block_metrics, manifest = analyze_conservation_classification_478(
            joined_paths,
            atom_paths,
            model_order=pilot_models,
            window_size=PREDICTABILITY_478_CFG["window_size"],
            primary_start=PREDICTABILITY_478_CFG["primary_start"],
            primary_end_exclusive=PREDICTABILITY_478_CFG["primary_end_exclusive"],
            block_bp=PREDICTABILITY_478_CFG["bootstrap_block_bp"],
        )
        metrics.to_parquet(output.metrics, index=False)
        block_metrics.to_parquet(output.block_metrics, index=False)
        Path(output.manifest).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )


rule predictability_478_classification_pilot:
    input:
        P478_ROOT + "/classification/pilot_manifest.json",


rule analyze_predictability_478_classification:
    input:
        joined=expand(
            P478_ROOT + "/joined/{region}.parquet",
            region=PREDICTABILITY_478_DATASETS,
        ),
        atoms=expand(
            P478_ROOT + "/atoms/{model}/{region}.{orientation}.parquet",
            model=PREDICTABILITY_478_MODELS,
            region=PREDICTABILITY_478_DATASETS,
            orientation=["fwd", "rc"],
        ),
    output:
        metrics=P478_ROOT + "/classification/metrics.parquet",
        block_metrics=P478_ROOT + "/classification/block_metrics.parquet",
        manifest=P478_ROOT + "/classification/manifest.json",
    threads: 8
    resources:
        mem_mb=28000,
    run:
        import json
        from marin_dna_evals.classification_478 import (
            analyze_conservation_classification_478,
        )

        joined_paths = dict(
            zip(PREDICTABILITY_478_DATASETS, input.joined, strict=True)
        )
        atom_keys = [
            (model, region, orientation)
            for model in PREDICTABILITY_478_MODELS
            for region in PREDICTABILITY_478_DATASETS
            for orientation in ("fwd", "rc")
        ]
        atom_paths = dict(zip(atom_keys, input.atoms, strict=True))
        metrics, block_metrics, manifest = analyze_conservation_classification_478(
            joined_paths,
            atom_paths,
            model_order=PREDICTABILITY_478_MODELS,
            window_size=PREDICTABILITY_478_CFG["window_size"],
            primary_start=PREDICTABILITY_478_CFG["primary_start"],
            primary_end_exclusive=PREDICTABILITY_478_CFG["primary_end_exclusive"],
            block_bp=PREDICTABILITY_478_CFG["bootstrap_block_bp"],
        )
        metrics.to_parquet(output.metrics, index=False)
        block_metrics.to_parquet(output.block_metrics, index=False)
        Path(output.manifest).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )


rule predictability_478_classification:
    input:
        P478_ROOT + "/classification/manifest.json",
