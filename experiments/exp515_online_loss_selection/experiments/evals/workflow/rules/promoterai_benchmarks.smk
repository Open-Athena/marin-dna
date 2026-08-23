# PromoterAI benchmark datasets from GitHub
# Downloads and processes all benchmarks from: https://github.com/Illumina/PromoterAI/tree/master/data/benchmark

rule promoterai_benchmark:
    output:
        "results/dataset/{dataset}.parquet",
    wildcard_constraints:
        dataset="|".join(config["promoterai_benchmarks"].keys()),
    params:
        filename=lambda wildcards: config["promoterai_benchmarks"][wildcards.dataset],
    run:
        url = f"https://raw.githubusercontent.com/Illumina/PromoterAI/master/data/benchmark/{params.filename}"
        V = pd.read_csv(url, sep="\t")
        V["chrom"] = V["chrom"].str.replace("^chr", "", regex=True)
        # Group by coordinates (variants near multiple genes)
        # Label is True if consequence is not "none" for ANY gene
        V_grouped = V.groupby(COORDINATES, as_index=False).agg({
            "consequence": lambda x: (x != "none").any(),
        })
        V_grouped = V_grouped.rename(columns={"consequence": "label"})
        V_grouped.to_parquet(output[0], index=False)


rule combine_promoterai_datasets:
    """Combine all promoterai benchmarks into a single dataset for efficient batch inference."""
    input:
        lambda wildcards: expand(
            "results/dataset/{dataset}.parquet",
            dataset=config["combined_dataset_groups"][wildcards.combined_group]["datasets"]
        )
    output:
        "results/dataset/{combined_group}.parquet"
    wildcard_constraints:
        combined_group="promoterai_combined"
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
