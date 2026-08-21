"""Likelihood-derived token dynamics through m1 -> m1.3 (issue #489).

The rules are additive, versioned, and excluded from the default ``all`` target.
Build the pilot before any full scoring cell.
"""

import re


LD489_CFG = config["likelihood_dynamics_489"]
LD489_VERSION = LD489_CFG["artifact_version"]
LD489_ROOT = f"results/m13_likelihood_dynamics_489/{LD489_VERSION}"
LD489_DATASETS = [dataset["name"] for dataset in LD489_CFG["datasets"]]
LD489_CHECKPOINTS = [
    checkpoint_cfg["name"] for checkpoint_cfg in LD489_CFG["checkpoints"]
]
LD489_PILOT_CHECKPOINTS = [LD489_CHECKPOINTS[0], LD489_CHECKPOINTS[-1]]
LD489_SCOPES = ["pilot", "full"]


def get_ld489_dataset(name):
    for dataset in LD489_CFG["datasets"]:
        if dataset["name"] == name:
            return dataset
    raise ValueError(f"likelihood_dynamics_489 dataset {name!r} not found")


def get_ld489_checkpoint(name):
    for order, checkpoint_cfg in enumerate(LD489_CFG["checkpoints"]):
        if checkpoint_cfg["name"] == name:
            return {**checkpoint_cfg, "order": order}
    raise ValueError(f"likelihood_dynamics_489 checkpoint {name!r} not found")


def get_ld489_batch_size(name):
    value = LD489_CFG["batch_sizes"][name]
    assert isinstance(value, int) and value > 0
    return value


def get_ld489_reference_input(wildcards):
    dataset = get_ld489_dataset(wildcards.region)
    if dataset["reference_kind"] == "twobit":
        return [storage(dataset["reference_uri"])]
    return []


assert LD489_VERSION == "v1"
assert LD489_DATASETS == ["cds", "upstream", "downstream", "ncrna", "enhancer"]
assert len(LD489_CHECKPOINTS) == 5
assert LD489_CFG["window_size"] == 255
assert (
    LD489_CFG["primary_start"],
    LD489_CFG["primary_end_exclusive"],
) == (32, 223)
assert LD489_CFG["pilot_n_windows"] < LD489_CFG["expected_windows"]
assert LD489_CFG["expected_windows"] == 16384
assert LD489_CFG["tokens_per_step"] == 2097152
assert LD489_CFG["blog_collection"].startswith(
    "https://huggingface.co/collections/marin-dna/"
)
for dataset in LD489_CFG["datasets"]:
    assert set(
        [
            "name",
            "hf_repo",
            "hf_revision",
            "assembly",
            "reference_kind",
            "reference_uri",
            "conservation_label_source",
        ]
    ) <= set(dataset)
    assert dataset["hf_repo"].startswith("marin-dna/")
    assert len(dataset["hf_revision"]) == 40
    assert dataset["reference_kind"] in {"twobit", "fasta"}
    if dataset["reference_kind"] == "twobit":
        assert dataset["assembly"] == "GCF_000001405.40"
        assert dataset["reference_uri"].endswith("GCF_000001405.40.2bit")
    else:
        assert dataset["assembly"] == "GRCh38_Ensembl_release_115"
        assert "ensembl-release-115" in dataset["reference_uri"]

previous_tokens = -1
for checkpoint_cfg in LD489_CFG["checkpoints"]:
    assert checkpoint_cfg["name"] in MODELS
    assert get_model_config(checkpoint_cfg["name"])["window_size"] == 255
    assert (
        checkpoint_cfg["cumulative_tokens"]
        == checkpoint_cfg["step"] * LD489_CFG["tokens_per_step"]
    )
    assert checkpoint_cfg["cumulative_tokens"] > previous_tokens
    previous_tokens = checkpoint_cfg["cumulative_tokens"]
    get_ld489_batch_size(checkpoint_cfg["name"])


rule build_likelihood_dynamics_489_metadata:
    input:
        reference=get_ld489_reference_input,
    output:
        parquet=LD489_ROOT + "/metadata/{scope}/{region}.parquet",
        manifest=LD489_ROOT + "/metadata/{scope}/{region}.manifest.json",
    wildcard_constraints:
        scope="pilot|full",
        region="|".join(map(re.escape, LD489_DATASETS)),
    threads: 2
    params:
        hf_repo=lambda wc: get_ld489_dataset(wc.region)["hf_repo"],
        hf_revision=lambda wc: get_ld489_dataset(wc.region)["hf_revision"],
        split=LD489_CFG["split"],
        window_size=LD489_CFG["window_size"],
        expected_windows=LD489_CFG["expected_windows"],
        pilot_n_windows=LD489_CFG["pilot_n_windows"],
        reference_kind=lambda wc: get_ld489_dataset(wc.region)["reference_kind"],
        reference_uri=lambda wc: get_ld489_dataset(wc.region)["reference_uri"],
        assembly=lambda wc: get_ld489_dataset(wc.region)["assembly"],
        conservation_label_source=lambda wc: get_ld489_dataset(wc.region)[
            "conservation_label_source"
        ],
    run:
        from datasets import load_dataset

        from marin_dna_evals.likelihood_dynamics_489 import (
            build_window_metadata,
            write_json,
        )

        source = load_dataset(
            params.hf_repo,
            split=params.split,
            revision=params.hf_revision,
        )
        assert len(source) == int(params.expected_windows), (
            f"{wildcards.region}: {len(source)} != {params.expected_windows}"
        )
        if wildcards.scope == "pilot":
            source = source.select(range(int(params.pilot_n_windows)))
        sequences = source.to_pandas()
        if params.reference_kind == "twobit":
            assert len(input.reference) == 1
            reference_path = input.reference[0]
        else:
            assert len(input.reference) == 0
            reference_path = params.reference_uri
        metadata, manifest = build_window_metadata(
            sequences,
            region=wildcards.region,
            window_size=int(params.window_size),
            reference_kind=params.reference_kind,
            reference_path=reference_path,
            assembly=params.assembly,
            conservation_label_source=params.conservation_label_source,
        )
        manifest.update(
            {
                "scope": wildcards.scope,
                "hf_repo": params.hf_repo,
                "hf_revision": params.hf_revision,
                "split": params.split,
                "reference_uri": params.reference_uri,
            }
        )
        metadata.to_parquet(output.parquet, index=False)
        write_json(output.manifest, manifest)


rule compute_likelihood_dynamics_489_atoms:
    input:
        checkpoint="results/checkpoints/{model}",
        metadata=LD489_ROOT + "/metadata/{scope}/{region}.parquet",
        metadata_manifest=LD489_ROOT + "/metadata/{scope}/{region}.manifest.json",
    output:
        parquet=LD489_ROOT + "/scoring/{scope}/atoms/{model}/{region}.parquet",
        manifest=LD489_ROOT
        + "/scoring/{scope}/atoms/{model}/{region}.manifest.json",
    wildcard_constraints:
        scope="pilot|full",
        model="|".join(map(re.escape, LD489_CHECKPOINTS)),
        region="|".join(map(re.escape, LD489_DATASETS)),
    threads: config["inference"]["num_workers"]
    resources:
        gpu=1,
        mem_mb=30000,
    params:
        hf_repo=lambda wc: get_ld489_dataset(wc.region)["hf_repo"],
        hf_revision=lambda wc: get_ld489_dataset(wc.region)["hf_revision"],
        split=LD489_CFG["split"],
        window_size=LD489_CFG["window_size"],
        expected_windows=LD489_CFG["expected_windows"],
        pilot_n_windows=LD489_CFG["pilot_n_windows"],
        assembly=lambda wc: get_ld489_dataset(wc.region)["assembly"],
        conservation_label_source=lambda wc: get_ld489_dataset(wc.region)[
            "conservation_label_source"
        ],
        checkpoint_order=lambda wc: get_ld489_checkpoint(wc.model)["order"],
        stage=lambda wc: get_ld489_checkpoint(wc.model)["stage"],
        training_step=lambda wc: get_ld489_checkpoint(wc.model)["step"],
        cumulative_tokens=lambda wc: get_ld489_checkpoint(wc.model)[
            "cumulative_tokens"
        ],
        torch_compile=config["inference"].get("torch_compile", False),
    run:
        import json

        import pandas as pd
        from datasets import load_dataset

        from marin_dna_evals.likelihood_dynamics_489 import (
            assemble_token_atoms,
            compute_hf_per_token_stats,
            write_json,
        )

        source = load_dataset(
            params.hf_repo,
            split=params.split,
            revision=params.hf_revision,
        )
        assert len(source) == int(params.expected_windows)
        if wildcards.scope == "pilot":
            source = source.select(range(int(params.pilot_n_windows)))
        sequences = source.to_pandas()
        stats, score_manifest = compute_hf_per_token_stats(
            input.checkpoint,
            sequences,
            window_size=int(params.window_size),
            batch_size=get_ld489_batch_size(wildcards.model),
            num_workers=threads,
            torch_compile=bool(params.torch_compile),
            validate_aggregate=wildcards.scope == "pilot",
        )
        metadata = pd.read_parquet(input.metadata)
        atoms, atom_manifest = assemble_token_atoms(
            metadata,
            stats,
            checkpoint=wildcards.model,
            checkpoint_order=int(params.checkpoint_order),
            stage=params.stage,
            training_step=int(params.training_step),
            cumulative_tokens=int(params.cumulative_tokens),
            assembly=params.assembly,
            conservation_label_source=params.conservation_label_source,
        )
        metadata_manifest = json.loads(Path(input.metadata_manifest).read_text())
        manifest = {
            "artifact_schema_version": LD489_VERSION,
            "scope": wildcards.scope,
            "dataset": {
                "region": wildcards.region,
                "hf_repo": params.hf_repo,
                "hf_revision": params.hf_revision,
                "split": params.split,
            },
            "checkpoint": get_ld489_checkpoint(wildcards.model),
            "metadata_manifest": metadata_manifest,
            "score_manifest": score_manifest,
            "atom_manifest": atom_manifest,
        }
        atoms.to_parquet(output.parquet, index=False)
        write_json(output.manifest, manifest)


rule validate_likelihood_dynamics_489_pilot:
    input:
        atoms=expand(
            LD489_ROOT + "/scoring/pilot/atoms/{model}/{region}.parquet",
            model=LD489_PILOT_CHECKPOINTS,
            region=LD489_DATASETS,
        ),
        manifests=expand(
            LD489_ROOT + "/scoring/pilot/atoms/{model}/{region}.manifest.json",
            model=LD489_PILOT_CHECKPOINTS,
            region=LD489_DATASETS,
        ),
    output:
        LD489_ROOT + "/pilot/validation_report.json",
    threads: 1
    run:
        from marin_dna_evals.likelihood_dynamics_489 import (
            validate_pilot_artifacts,
            write_json,
        )

        keys = [
            (model, region)
            for model in LD489_PILOT_CHECKPOINTS
            for region in LD489_DATASETS
        ]
        report = validate_pilot_artifacts(
            dict(zip(keys, input.atoms, strict=True)),
            dict(zip(keys, input.manifests, strict=True)),
            checkpoints=LD489_PILOT_CHECKPOINTS,
            regions=LD489_DATASETS,
            expected_windows=int(LD489_CFG["pilot_n_windows"]),
            window_size=int(LD489_CFG["window_size"]),
        )
        write_json(output[0], report)


rule likelihood_dynamics_489_metadata:
    input:
        expand(
            LD489_ROOT + "/metadata/full/{region}.manifest.json",
            region=LD489_DATASETS,
        ),


rule likelihood_dynamics_489_pilot:
    input:
        LD489_ROOT + "/pilot/validation_report.json",


rule likelihood_dynamics_489_atoms:
    input:
        expand(
            LD489_ROOT + "/scoring/full/atoms/{model}/{region}.manifest.json",
            model=LD489_CHECKPOINTS,
            region=LD489_DATASETS,
        ),
