rule traitgym_mendelian_promoter_dataset:
    output:
        "results/dataset/traitgym_mendelian_promoter.parquet",
    run:
        V = pd.read_parquet("hf://datasets/songlab/TraitGym/mendelian_traits_matched_9/test.parquet")
        subset = pd.read_parquet("hf://datasets/songlab/TraitGym/mendelian_traits_matched_9/subset/nonexonic_AND_proximal.parquet")
        V = V.merge(subset, on=COORDINATES, how="inner")
        V.to_parquet(output[0], index=False)


rule traitgym_complex_promoter_dataset:
    output:
        "results/dataset/traitgym_complex_promoter.parquet",
    run:
        V = pd.read_parquet("hf://datasets/songlab/TraitGym/complex_traits_matched_9/test.parquet")
        subset = pd.read_parquet("hf://datasets/songlab/TraitGym/complex_traits_matched_9/subset/nonexonic_AND_proximal.parquet")
        V = V.merge(subset, on=COORDINATES, how="inner")
        V.to_parquet(output[0], index=False)
