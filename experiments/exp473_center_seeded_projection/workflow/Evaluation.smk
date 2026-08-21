"""Isolated additive development evaluation for MarinDNA issue #473."""

import re

import pandas as pd

from exp473_center_seeded_projection.development_eval import (
    compute_issue473_metrics,
    compute_issue473_variant_scores,
    load_development_dataset,
)

MODELS = [dict(model) for model in config["models"]]
MODEL_BY_NAME = {model["name"]: model for model in MODELS}
DATASETS = [dict(dataset) for dataset in config["datasets"]]
DATASET_BY_NAME = {dataset["name"]: dataset for dataset in DATASETS}
assert MODELS and len(MODEL_BY_NAME) == len(MODELS)
assert DATASETS and len(DATASET_BY_NAME) == len(DATASETS)
assert config["split"] == "train"
assert config["issue_473"]["held_out_access"] is False
assert config["issue_473"]["dataset_file"] == "train.parquet"
RESULTS_ROOT = str(config["issue_473"]["results_root"]).rstrip("/")
assert RESULTS_ROOT.endswith(
    f"/{config['issue_473']['experiment_commit']}/development_eval"
)
MODEL_RE = "|".join(re.escape(name) for name in MODEL_BY_NAME)
DATASET_RE = "|".join(re.escape(name) for name in DATASET_BY_NAME)
MODEL_DATASET_PAIRS = [
    (model["name"], dataset)
    for model in MODELS
    for dataset in model["datasets"]
]
assert len(MODEL_DATASET_PAIRS) == 72
METRIC_OUTPUTS = [
    f"{RESULTS_ROOT}/metrics/{model}/{dataset}.parquet"
    for model, dataset in MODEL_DATASET_PAIRS
]


def model_config(name):
    return MODEL_BY_NAME[name]


def dataset_config(name):
    return DATASET_BY_NAME[name]


def require_registered_cell(model, dataset):
    assert dataset in model_config(model)["datasets"], (
        f"model {model!r} is not registered for dataset {dataset!r}"
    )


rule issue_473_eval_download_model:
    """Stage one immutable HF-format checkpoint in the additive namespace."""
    output:
        directory(f"{RESULTS_ROOT}/checkpoints/{{model}}"),
    wildcard_constraints:
        model=MODEL_RE,
    params:
        gcs_path=lambda wc: model_config(wc.model)["gcs_path"],
    run:
        shell(
            f"mkdir -p {output[0]} && "
            f"gcloud storage cp -r '{params.gcs_path}/*' {output[0]}/"
        )


rule issue_473_eval_score:
    """Score one direct-file development cell with the official runner."""
    input:
        checkpoint=f"{RESULTS_ROOT}/checkpoints/{{model}}",
    output:
        f"{RESULTS_ROOT}/scores/{{model}}/{{dataset}}.parquet",
    wildcard_constraints:
        model=MODEL_RE,
        dataset=DATASET_RE,
    threads: config["inference"]["num_workers"]
    resources:
        gpu=1,
        mem_mb=12000,
    run:
        require_registered_cell(wildcards.model, wildcards.dataset)
        model = model_config(wildcards.model)
        dataset = dataset_config(wildcards.dataset)
        variants = load_development_dataset(
            f"{config['input_hf_prefix']}_{wildcards.dataset}",
            revision=dataset["hf_revision"],
            filename=config["issue_473"]["dataset_file"],
            split=config["split"],
        )
        scores = compute_issue473_variant_scores(
            checkpoint_path=input.checkpoint,
            dataset=variants,
            genome_path=config["genome_path"],
            context_size=int(model["window_size"]),
            batch_size=int(config["inference"]["batch_size"]),
            num_workers=int(config["inference"]["num_workers"]),
            data_transform_on_the_fly=bool(
                config["inference"]["data_transform_on_the_fly"]
            ),
            torch_compile=bool(config["inference"]["torch_compile"]),
            rc=bool(config["inference"]["rc"]),
        )
        assert len(scores) == len(variants)
        output_frame = pd.concat(
            [variants.reset_index(drop=True), scores.reset_index(drop=True)],
            axis=1,
        )
        output_frame.to_parquet(output[0], index=False)
        print(
            f"[issue473] {wildcards.model} {wildcards.dataset} "
            f"(train.parquet only): n={len(output_frame)}"
        )


rule issue_473_eval_metrics:
    """Compute the official metric table for one development score cell."""
    input:
        f"{RESULTS_ROOT}/scores/{{model}}/{{dataset}}.parquet",
    output:
        f"{RESULTS_ROOT}/metrics/{{model}}/{{dataset}}.parquet",
    wildcard_constraints:
        model=MODEL_RE,
        dataset=DATASET_RE,
    resources:
        mem_mb=12000,
    run:
        require_registered_cell(wildcards.model, wildcards.dataset)
        dataset = dataset_config(wildcards.dataset)
        metrics = compute_issue473_metrics(
            pd.read_parquet(input[0]),
            model=wildcards.model,
            dataset=wildcards.dataset,
            split=config["split"],
            score_protocol=dataset["score_protocol"],
            eval_protocol=dataset.get("eval_protocol", "matched_pair"),
            n_bootstrap=int(config["inference"]["n_bootstrap"]),
            bootstrap_seed=int(config["inference"]["bootstrap_seed"]),
        )
        metrics.to_parquet(output[0], index=False)
        print(
            f"[issue473] {wildcards.model} {wildcards.dataset}: "
            f"{len(metrics)} official metric rows"
        )


rule issue_473_development_evaluation:
    """Build every registered development-only score and metric cell."""
    input:
        METRIC_OUTPUTS,
