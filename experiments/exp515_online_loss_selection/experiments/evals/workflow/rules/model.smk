rule model_download_checkpoints_first_part:
    output:
        directory("results/model/checkpoints_first_part"),
    shell:
        "hf download songlab/gpn-animal-promoter-checkpoints --repo-type dataset --local-dir {output}"


rule model_download_checkpoints_second_part:
    output:
        directory("results/model/checkpoints_second_part"),
    shell:
        "hf download gonzalobenegas/gpn-animal-promoter-checkpoints-second-part --repo-type dataset --local-dir {output}"


rule model_llr:
    input:
        "results/dataset/{dataset}.parquet",
        "results/genome.fa.gz",
        "results/model/checkpoints_first_part",
        "results/model/checkpoints_second_part",
    output:
        "results/features/{dataset}/{model}_LLR.parquet",
    wildcard_constraints:
        dataset="|".join(get_all_datasets_including_combined()),
        model="|".join(config["models"].keys()),
    threads:
        workflow.cores
    run:
        V = pd.read_parquet(input[0], columns=COORDINATES)
        dataset = Dataset.from_pandas(V, preserve_index=False)
        genome = Genome(input[1])
        model_name = config["models"][wildcards.model]
        tokenizer = HFTokenizer(AutoTokenizer.from_pretrained(model_name))
        model = GPNMaskedLM(AutoModelForMaskedLM.from_pretrained(model_name))
        llr = run_llr_mlm(
            model,
            tokenizer,
            dataset,
            genome,
            config["context_size"],
            data_transform_on_the_fly=True,
            inference_kwargs=dict(
                per_device_eval_batch_size=config["per_device_batch_size"],
                torch_compile=config["torch_compile"],
                bf16_full_eval=True,
                dataloader_num_workers=threads,
                remove_unused_columns=False,
                report_to="none",
            ),
        )
        pd.DataFrame(llr, columns=["score"]).to_parquet(output[0], index=False)


rule model_abs_llr:
    input:
        "results/features/{dataset}/{model}_LLR.parquet",
    output:
        "results/features/{dataset}/{model}_absLLR.parquet",
    wildcard_constraints:
        dataset="|".join(get_all_datasets_including_combined()),
        model="|".join(config["models"].keys()),
    run:
        df = pd.read_parquet(input[0])
        df = df.abs()
        df.to_parquet(output[0], index=False)


rule model_euclidean_distance:
    input:
        "results/dataset/{dataset}.parquet",
        "results/genome.fa.gz",
        "results/model/checkpoints_first_part",
        "results/model/checkpoints_second_part",
    output:
        "results/features/{dataset}/{model}_L2.parquet",
    wildcard_constraints:
        dataset="|".join(get_all_datasets_including_combined()),
        model="|".join(config["models"].keys()),
    threads:
        workflow.cores
    run:
        V = pd.read_parquet(input[0], columns=COORDINATES)
        dataset = Dataset.from_pandas(V, preserve_index=False)
        genome = Genome(input[1])
        model_name = config["models"][wildcards.model]
        tokenizer = HFTokenizer(AutoTokenizer.from_pretrained(model_name))
        model = GPNEmbeddingModel(AutoModel.from_pretrained(model_name), layer="last")
        euclidean_distance = run_euclidean_distance(
            model,
            tokenizer,
            dataset,
            genome,
            config["context_size"],
            data_transform_on_the_fly=True,
            inference_kwargs=dict(
                per_device_eval_batch_size=config["per_device_batch_size"],
                torch_compile=config["torch_compile"],
                bf16_full_eval=True,
                dataloader_num_workers=threads,
                remove_unused_columns=False,
                report_to="none",
            ),
        )
        pd.DataFrame(euclidean_distance, columns=["score"]).to_parquet(output[0], index=False)


rule model_prediction:
    input:
        "results/features/{dataset}/{model}.parquet",
    output:
        "results/prediction/{dataset}/{model}.{sign,plus|minus}.{feature}.parquet",
    run:
        df = (
            pd.read_parquet(input[0], columns=[wildcards.feature])
            .rename(columns={wildcards.feature: "score"})
        )
        if wildcards.sign == "minus":
            df.score = -df.score
        df.to_parquet(output[0], index=False)
