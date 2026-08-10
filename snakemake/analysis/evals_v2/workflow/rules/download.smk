"""Download model checkpoint dirs from GCS or HuggingFace Hub.

The genome reference is read directly from S3 by pyfaidx (see
``compute_scores`` in inference.smk) — no `download_genome` rule needed.
"""


rule download_model:
    """Pull a model checkpoint into results/checkpoints/{model}/.

    Each model entry must declare exactly one of:
      - `gcs_path`: full GCS URI (incl. /hf/step-{N}); pulled with
        `gcloud storage cp -r`. Auth uses application-default credentials.
        The `mkdir` + glob is to flatten: `gcloud storage cp -r src dst`
        nests src under dst, but the inference rule expects HF model
        files directly under {output}.
      - `hf_repo`: HuggingFace Hub repo ID; pulled with
        `huggingface_hub.snapshot_download` (mirrors the entire repo).
    """
    output:
        directory("results/checkpoints/{model}"),
    wildcard_constraints:
        model="|".join(MODELS),
    params:
        # Only the source URL/repo affects the download output. Other
        # per-model fields (window_size, batch_size, datasets, …) don't,
        # so they intentionally stay out of `params:` and won't trigger
        # a re-download when tuned.
        gcs_path=lambda wc: get_model_config(wc.model).get("gcs_path", ""),
        hf_repo=lambda wc: get_model_config(wc.model).get("hf_repo", ""),
    run:
        out = output[0]
        if params.gcs_path:
            shell(
                f"mkdir -p {out} && gcloud storage cp -r '{params.gcs_path}/*' {out}/"
            )
        elif params.hf_repo:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=params.hf_repo, local_dir=out)
        else:
            raise ValueError(
                f"model {wildcards.model!r} needs `gcs_path` or `hf_repo`"
            )
