"""One checkpoint load and joint inference for the three RAG development cohorts."""

import hashlib

from marin_dna_evals.rag import compute_combined_rag_scores


def rag_harness_input(wildcards):
    uri = get_model_config(wildcards.model)["rag_harness"]["uri"]
    return storage.s3(uri) if uri.startswith("s3://") else uri


rule compute_rag_scores:
    input:
        checkpoint="results/checkpoints/{model}",
        harness=rag_harness_input,
    output:
        mendelian="results/scores/{model}/mendelian_traits.parquet",
        complex="results/scores/{model}/complex_traits.parquet",
        sge="results/scores/{model}/sge.parquet",
    wildcard_constraints:
        model="|".join(RAG_MODELS) or r"(?!)",
    threads: config["inference"]["num_workers"]
    params:
        **inference_precision_params(config["inference"]),
        harness=lambda wc: get_model_config(wc.model)["rag_harness"],
        revisions=lambda wc: {
            name: get_dataset_config(name)["hf_revision"]
            for name in ("mendelian_traits", "complex_traits", "sge")
        },
    run:
        with open(input.harness, "rb") as handle:
            if (
                hashlib.file_digest(handle, "sha256").hexdigest()
                != params.harness["sha256"]
            ):
                raise ValueError("combined RAG harness SHA-256 mismatch")
        canonical = {
            name: load_dataset(
                f"{config['input_hf_prefix']}_{name}",
                split="train",
                revision=revision,
            ).to_list()
            for name, revision in params.revisions.items()
        }
        inference = config["inference"]
        kwargs = {
            "per_device_eval_batch_size": get_model_batch_size(wildcards.model),
            "torch_compile": inference["torch_compile"],
            "bf16_full_eval": inference["bf16"],
            "tf32": inference.get("tf32"),
            "dataloader_num_workers": inference["num_workers"],
            "dataloader_pin_memory": True,
            "remove_unused_columns": False,
            "eval_accumulation_steps": get_model_eval_accumulation_steps(
                wildcards.model
            ),
        }
        if inference["num_workers"] > 0:
            kwargs["dataloader_prefetch_factor"] = 2
        scores = compute_combined_rag_scores(
            input.checkpoint,
            input.harness,
            canonical,
            params.revisions,
            inference_kwargs=kwargs,
        )
        for name, path in zip(
            ("mendelian_traits", "complex_traits", "sge"), output, strict=True
        ):
            scores[name].to_parquet(path, index=False)
            print(
                f"[evals_v2] {wildcards.model} {name} (train, combined RAG): n={len(scores[name])}"
            )
