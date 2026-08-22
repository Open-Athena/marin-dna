"""Additive terminal VEP comparison for the issue #473 random split control."""

import re
from pathlib import Path

import pandas as pd

from exp473_center_seeded_projection.development_eval import (
    compute_issue473_metrics,
    compute_issue473_variant_scores,
    load_development_dataset,
)
from exp473_center_seeded_projection.random_validation_vep import (
    MATURE_MIRNA_SUBSET,
    exclude_mature_mirna_groups,
    run_terminal_comparison,
)

MODELS = [dict(model) for model in config["models"]]
DATASETS = [dict(dataset) for dataset in config["datasets"]]
MODEL_BY_NAME = {model["name"]: model for model in MODELS}
DATASET_BY_NAME = {dataset["name"]: dataset for dataset in DATASETS}
CONTROL = dict(config["issue_473_random_validation_vep"])
assert config["split"] == "train"
assert CONTROL["held_out_access"] is False
assert CONTROL["dataset_file"] == "train.parquet"
assert len(MODELS) == 1
assert len(DATASETS) == 3
MODEL = MODELS[0]
MODEL_NAME = MODEL["name"]
DATASET_NAMES = tuple(MODEL["datasets"])
assert set(DATASET_NAMES) == set(DATASET_BY_NAME)
RESULTS_ROOT = str(CONTROL["results_root"]).rstrip("/")
assert RESULTS_ROOT.endswith(f"/{CONTROL['snapshot_commit']}/random_validation_vep")
BASELINE_RESULTS_ROOT = str(CONTROL["baseline_results_root"]).rstrip("/")
BASELINE_MODEL = str(CONTROL["baseline_model"])
MODEL_RE = re.escape(MODEL_NAME)
DATASET_RE = "|".join(re.escape(name) for name in DATASET_NAMES)
NEW_SCORE_OUTPUTS = [
    f"{RESULTS_ROOT}/scores/{MODEL_NAME}/{dataset}.parquet" for dataset in DATASET_NAMES
]
NEW_METRIC_OUTPUTS = [
    f"{RESULTS_ROOT}/metrics/{MODEL_NAME}/{dataset}.parquet"
    for dataset in DATASET_NAMES
]
BASELINE_SCORE_INPUTS = [
    f"{BASELINE_RESULTS_ROOT}/scores/{BASELINE_MODEL}/{dataset}.parquet"
    for dataset in DATASET_NAMES
]
BASELINE_METRIC_INPUTS = [
    f"{BASELINE_RESULTS_ROOT}/metrics/{BASELINE_MODEL}/{dataset}.parquet"
    for dataset in DATASET_NAMES
]
ANALYSIS_ROOT = f"{RESULTS_ROOT}/analysis"
ANALYSIS_OUTPUTS = {
    "comparison": f"{ANALYSIS_ROOT}/comparison.parquet",
    "samples": f"{ANALYSIS_ROOT}/paired_bootstrap_samples.parquet",
    "summary": f"{ANALYSIS_ROOT}/summary.md",
    "manifest": f"{ANALYSIS_ROOT}/manifest.json",
}


def dataset_config(name):
    return DATASET_BY_NAME[name]


rule issue_473_random_validation_vep_download_model:
    """Stage the immutable terminal random-validation checkpoint."""
    output:
        directory(f"{RESULTS_ROOT}/checkpoints/{MODEL_NAME}"),
    params:
        gcs_path=MODEL["gcs_path"],
    run:
        shell(
            f"mkdir -p {output[0]} && "
            f"gcloud storage cp -r '{params.gcs_path}/*' {output[0]}/"
        )


rule issue_473_random_validation_vep_score:
    """Score one direct-file development VEP cell with the official runner."""
    input:
        checkpoint=f"{RESULTS_ROOT}/checkpoints/{MODEL_NAME}",
    output:
        f"{RESULTS_ROOT}/scores/{MODEL_NAME}/{{dataset}}.parquet",
    wildcard_constraints:
        dataset=DATASET_RE,
    threads: config["inference"]["num_workers"]
    resources:
        gpu=1,
        mem_mb=12000,
    run:
        dataset = dataset_config(wildcards.dataset)
        variants = load_development_dataset(
            f"{config['input_hf_prefix']}_{wildcards.dataset}",
            revision=dataset["hf_revision"],
            filename=CONTROL["dataset_file"],
            split=config["split"],
        )
        scores = compute_issue473_variant_scores(
            checkpoint_path=input.checkpoint,
            dataset=variants,
            genome_path=config["genome_path"],
            context_size=int(MODEL["window_size"]),
            batch_size=int(config["inference"]["batch_size"]),
            num_workers=int(config["inference"]["num_workers"]),
            data_transform_on_the_fly=bool(
                config["inference"]["data_transform_on_the_fly"]
            ),
            torch_compile=bool(config["inference"]["torch_compile"]),
            rc=bool(config["inference"]["rc"]),
        )
        assert len(scores) == len(variants)
        pd.concat(
            [variants.reset_index(drop=True), scores.reset_index(drop=True)],
            axis=1,
        ).to_parquet(output[0], index=False)
        print(
            f"[issue473-random-vep] {MODEL_NAME} {wildcards.dataset} "
            f"(train.parquet only): n={len(variants)}"
        )


rule issue_473_random_validation_vep_metrics:
    """Compute development metrics after mature-miRNA group exclusion."""
    input:
        f"{RESULTS_ROOT}/scores/{MODEL_NAME}/{{dataset}}.parquet",
    output:
        f"{RESULTS_ROOT}/metrics/{MODEL_NAME}/{{dataset}}.parquet",
    wildcard_constraints:
        dataset=DATASET_RE,
    resources:
        mem_mb=12000,
    run:
        dataset = dataset_config(wildcards.dataset)
        scores = pd.read_parquet(input[0])
        if dataset.get("eval_protocol", "matched_pair") == "matched_pair":
            scores = exclude_mature_mirna_groups(scores)
        metrics = compute_issue473_metrics(
            scores,
            model=MODEL_NAME,
            dataset=wildcards.dataset,
            split=config["split"],
            score_protocol=dataset["score_protocol"],
            eval_protocol=dataset.get("eval_protocol", "matched_pair"),
            n_bootstrap=int(config["inference"]["n_bootstrap"]),
            bootstrap_seed=int(config["inference"]["bootstrap_seed"]),
        )
        assert MATURE_MIRNA_SUBSET not in set(metrics["subset"].astype(str))
        metrics.to_parquet(output[0], index=False)
        print(
            f"[issue473-random-vep] {MODEL_NAME} {wildcards.dataset}: "
            f"{len(metrics)} metric rows"
        )


rule issue_473_random_validation_vep_analyze:
    """Compare new terminal scores with the immutable #417 terminal scores."""
    input:
        random_scores=NEW_SCORE_OUTPUTS,
        random_metrics=NEW_METRIC_OUTPUTS,
        chr18_scores=BASELINE_SCORE_INPUTS,
        chr18_metrics=BASELINE_METRIC_INPUTS,
    output:
        comparison=ANALYSIS_OUTPUTS["comparison"],
        samples=ANALYSIS_OUTPUTS["samples"],
        summary=ANALYSIS_OUTPUTS["summary"],
        manifest=ANALYSIS_OUTPUTS["manifest"],
    resources:
        mem_mb=12000,
    run:
        random_score_paths = {
            dataset: Path(str(path))
            for dataset, path in zip(
                DATASET_NAMES, input.random_scores, strict=True
            )
        }
        random_metric_paths = {
            dataset: Path(str(path))
            for dataset, path in zip(
                DATASET_NAMES, input.random_metrics, strict=True
            )
        }
        chr18_score_paths = {
            dataset: Path(str(path))
            for dataset, path in zip(DATASET_NAMES, input.chr18_scores, strict=True)
        }
        chr18_metric_paths = {
            dataset: Path(str(path))
            for dataset, path in zip(
                DATASET_NAMES, input.chr18_metrics, strict=True
            )
        }
        input_uris = {
            "random_scores": dict(
                zip(DATASET_NAMES, NEW_SCORE_OUTPUTS, strict=True)
            ),
            "random_metrics": dict(
                zip(DATASET_NAMES, NEW_METRIC_OUTPUTS, strict=True)
            ),
            "chr18_scores": dict(
                zip(DATASET_NAMES, BASELINE_SCORE_INPUTS, strict=True)
            ),
            "chr18_metrics": dict(
                zip(DATASET_NAMES, BASELINE_METRIC_INPUTS, strict=True)
            ),
        }
        paired = CONTROL["paired_bootstrap"]
        run_terminal_comparison(
            chr18_score_paths=chr18_score_paths,
            random_score_paths=random_score_paths,
            chr18_metric_paths=chr18_metric_paths,
            random_metric_paths=random_metric_paths,
            input_uris=input_uris,
            comparison_output=Path(str(output.comparison)),
            samples_output=Path(str(output.samples)),
            summary_output=Path(str(output.summary)),
            manifest_output=Path(str(output.manifest)),
            snapshot_commit=CONTROL["snapshot_commit"],
            n_bootstrap=int(paired["n_bootstrap"]),
            seed=int(paired["seed"]),
        )


rule issue_473_random_validation_vep_all:
    """Build the three new score cells and terminal comparison bundle."""
    input:
        list(ANALYSIS_OUTPUTS.values()),
