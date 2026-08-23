rule gnomad_promoter_dataset:
    output:
        "results/dataset/gnomad_promoter.parquet",
    run:
        V = pd.read_parquet("hf://datasets/songlab/gnomad_balanced/test.parquet")
        V = V[V.consequence == "upstream_gene"]
        V = V.groupby("label").sample(n=5000, random_state=42).reset_index(drop=True)
        V = V.sort_values(COORDINATES)
        V.to_parquet(output[0], index=False)
