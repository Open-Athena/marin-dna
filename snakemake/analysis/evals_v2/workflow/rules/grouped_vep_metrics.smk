"""Additive AUPRC and Group SMD reports for matched-group VEP datasets.

The named ``grouped_vep_metrics`` target is kept off ``rule all`` so the
existing S3-backed metrics artifacts and their transitive behavior remain
unchanged. Each cell writes a summary report and the aligned match-group
bootstrap draws used for Group SMD intervals and paired comparisons.
"""

from marin_dna_evals.grouped_vep_metrics import compute_grouped_vep_metrics

GROUPED_VEP_DATASETS = [
    dataset for dataset in DATASETS if get_dataset_protocol(dataset) == "matched_pair"
]
GROUPED_VEP_TARGETS = [
    f"results/grouped_vep_metrics/{model['name']}/{dataset}.parquet"
    for model in config["models"]
    for dataset in get_model_datasets(model["name"])
    if dataset in GROUPED_VEP_DATASETS
]


rule grouped_vep_metrics:
    input:
        GROUPED_VEP_TARGETS,


rule compute_grouped_vep_report:
    """Report AUPRC beside Group SMD for one grouped VEP model/dataset cell."""
    input:
        "results/scores/{model}/{dataset}.parquet",
    output:
        summary="results/grouped_vep_metrics/{model}/{dataset}.parquet",
        bootstrap="results/grouped_vep_bootstrap/{model}/{dataset}.parquet",
    wildcard_constraints:
        model="|".join(MODELS),
        dataset="|".join(GROUPED_VEP_DATASETS),
    params:
        n_bootstrap=config["inference"]["n_bootstrap"],
        bootstrap_seed=config["inference"]["bootstrap_seed"],
        score_protocol=lambda wc: get_dataset_config(wc.dataset)["score_protocol"],
    run:
        assert wildcards.dataset in get_model_datasets(wildcards.model), (
            f"model {wildcards.model!r} is not configured for dataset "
            f"{wildcards.dataset!r}"
        )
        protocol = params.score_protocol
        transform = SCORE_PROTOCOLS[protocol]
        df = pd.read_parquet(input[0])
        for column in REQUIRED_VARIANT_COLUMNS:
            assert column in df.columns, f"scores parquet missing column {column!r}"
        if "llr_rc" in df.columns:
            df["llr_avg"] = (df["llr_fwd"] + df["llr_rc"]) / 2
            df["jsd_avg"] = (df["jsd_fwd"] + df["jsd_rc"]) / 2
        score_columns: list[str] = []
        for strand in ("fwd", "rc", "avg"):
            llr_column = f"llr_{strand}"
            jsd_column = f"jsd_{strand}"
            if llr_column in df.columns:
                df[f"{protocol}_{strand}"] = transform(df[llr_column])
                score_columns.append(f"{protocol}_{strand}")
            if jsd_column in df.columns:
                score_columns.append(jsd_column)
        assert (
            score_columns
        ), "no score columns to evaluate — scores parquet schema?"
        summary, bootstrap = compute_grouped_vep_metrics(
            dataset=df[list(REQUIRED_VARIANT_COLUMNS)],
            scores=df[score_columns],
            score_columns=score_columns,
            n_bootstrap=params.n_bootstrap,
            rng=params.bootstrap_seed,
        )
        for table in (summary, bootstrap):
            table["model"] = wildcards.model
            table["dataset"] = wildcards.dataset
            table["split"] = config["split"]
            table["bootstrap_seed"] = params.bootstrap_seed
        summary.to_parquet(output.summary, index=False)
        bootstrap.to_parquet(output.bootstrap, index=False)
        print(
            f"[evals_v2] {wildcards.model} {wildcards.dataset}: "
            f"{len(summary)} grouped VEP metric rows and "
            f"{len(bootstrap)} aligned bootstrap rows"
        )
