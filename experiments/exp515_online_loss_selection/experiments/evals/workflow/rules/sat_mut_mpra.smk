rule sat_mut_mpra_promoter_dataset:
    output:
        expand("results/dataset/sat_mut_mpra_promoter_{promoter}.parquet", promoter=config["sat_mut_mpra_promoter"]),
    run:
        V = pd.read_parquet("hf://datasets/gonzalobenegas/sat_mut_mpra/test.parquet")
        V["label"] = V["label"].abs()  # abs(LFC)
        for promoter, path in zip(config["sat_mut_mpra_promoter"], output):
            V[V["element"] == promoter].to_parquet(path, index=False)


rule combine_sat_mut_mpra_datasets:
    """Combine all sat_mut_mpra promoter datasets into a single dataset for efficient batch inference."""
    input:
        lambda wildcards: expand(
            "results/dataset/{dataset}.parquet",
            dataset=config["combined_dataset_groups"][wildcards.combined_group]["datasets"]
        )
    output:
        "results/dataset/{combined_group}.parquet"
    wildcard_constraints:
        combined_group="sat_mut_mpra_combined"
    run:
        datasets = config["combined_dataset_groups"][wildcards.combined_group]["datasets"]
        dfs = []
        for dataset_name, dataset_path in zip(datasets, input):
            df = pd.read_parquet(dataset_path)
            df["dataset"] = dataset_name  # Add dataset identifier column
            dfs.append(df)

        # Concatenate all datasets
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_parquet(output[0], index=False)
