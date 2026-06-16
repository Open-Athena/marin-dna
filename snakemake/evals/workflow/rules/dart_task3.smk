from marin_dna.pipelines.evals import hf_readme
from marin_dna.pipelines.evals.dart_task3 import (
    SPLIT_CHROMS as DART_TASK3_SPLIT_CHROMS,
    assert_full_dataset,
    parse_dart_task3,
    split_frames,
)

# DART-Eval Task 3 ("Discriminating Cell-Type-Specific Elements", issue #293): a
# 500 bp ATAC-seq consensus-peak INTERVAL dataset with 5-class cell-type labels
# (top_5000_deseq_peaks.tsv: the top-5,000 DESeq2 differentially-accessible peaks
# per cell type, 25,000 windows = 5,000 balanced per cell line).
# Because it is intervals — not variants — it bypasses the variant machinery
# (check_ref_alt, consequence/TSS/exon annotation, materialize, matching/QC) AND
# the generic split_dataset_by_chrom / hf_upload rules: it has a 3-way split
# (train/validation/test) on DART-Eval's canonical chromosome holdout, not the
# pipeline's odd/even 2-way split. It lives in its own `results/dart_task3/...`
# output namespace so it never collides with the `{dataset}` wildcard of the
# generic rules. See snakemake/evals/README.md and the dart_task3 library module.

# train, validation, test (canonical 3-way order).
DART_TASK3_SPLITS = list(DART_TASK3_SPLIT_CHROMS)


rule dart_task3_download:
    """Download the top-5,000-per-cell-type peaks TSV (top_5000_deseq_peaks.tsv)
from Synapse over plain HTTP using a Personal Access Token (no synapseclient
dependency). Requires a free Synapse account; export the PAT as
SYNAPSE_AUTH_TOKEN before running."""
    output:
        "results/dart_task3/peaks.tsv",
    params:
        syn_id=config["dart_eval"]["task3_synapse_id"],
    shell:
        'test -n "$SYNAPSE_AUTH_TOKEN" || {{ echo "ERROR: set SYNAPSE_AUTH_TOKEN (a Synapse Personal Access Token) to download DART-Eval data" >&2; exit 1; }}; '
        'curl -fL -H "Authorization: Bearer $SYNAPSE_AUTH_TOKEN" '
        '"https://repo-prod.prod.sagebase.org/repo/v1/entity/{params.syn_id}/file" '
        "-o {output}"


rule dart_task3_dataset:
    """Parse top_5000_deseq_peaks.tsv into the interval schema (chrom/start/end/
label) and route windows to train/validation/test by DART-Eval's canonical
chromosome split. No annotation, no matching, no subsampling. The parse +
split logic (and its asserts) live in the library so they are unit-tested."""
    input:
        tsv="results/dart_task3/peaks.tsv",
    output:
        expand("results/dart_task3/dataset/{split}.parquet", split=DART_TASK3_SPLITS),
    run:
        # infer_schema_length=None: scan the whole TSV for dtype inference (the
        # file is bounded; avoids mis-inferring a column from a short first chunk).
        raw = pl.read_csv(input.tsv, separator="\t", infer_schema_length=None)
        V = parse_dart_task3(raw)
        # Build sanity on the COMPLETE set before splitting/uploading: all 5 cell
        # types present, balanced, 25,000 windows (a truncated download or wrong
        # file trips this).
        assert_full_dataset(V)
        frames = split_frames(V)
        # expand() preserves DART_TASK3_SPLITS order, so output[i] <-> split[i].
        for split, path in zip(DART_TASK3_SPLITS, output):
            frames[split].write_parquet(path)


rule dart_task3_upload:
    """Upload the 3 split parquets + a generated dataset card to
bolinas-dna/evals_dart_task3. Dedicated (not the generic hf_upload) because
of the 3-way split and the interval-specific card."""
    input:
        train="results/dart_task3/dataset/train.parquet",
        validation="results/dart_task3/dataset/validation.parquet",
        test="results/dart_task3/dataset/test.parquet",
    output:
        touch("results/dart_task3/upload.done"),
    params:
        repo_name=f"{config['output_hf_prefix']}_dart_task3",
    run:
        api = HfApi()
        api.create_repo(params.repo_name, repo_type="dataset", exist_ok=True)
        readme = hf_readme.render_dart_task3(
            sha=GIT_SHA,
            train_path=input.train,
            validation_path=input.validation,
            test_path=input.test,
        )
        # Single atomic commit: README + all 3 splits land together, so the repo
        # is never in a half-updated state (mirrors the generic hf_upload). The
        # split inputs are the paths snakemake localized for this rule (the S3
        # storage provider deletes the bare `results/...` copy after the producer).
        ops = [
            CommitOperationAdd(
                path_in_repo="README.md", path_or_fileobj=readme.encode()
            ),
            CommitOperationAdd(
                path_in_repo="train.parquet", path_or_fileobj=str(input.train)
            ),
            CommitOperationAdd(
                path_in_repo="validation.parquet",
                path_or_fileobj=str(input.validation),
            ),
            CommitOperationAdd(
                path_in_repo="test.parquet", path_or_fileobj=str(input.test)
            ),
        ]
        api.create_commit(
            repo_id=params.repo_name,
            repo_type="dataset",
            operations=ops,
            commit_message=f"Upload dart_task3 dataset ({len(ops)} files)",
        )


rule dart_task3:
    """Convenience target: build the 3 split parquets (no upload). Upload
separately with `snakemake results/dart_task3/upload.done`."""
    input:
        expand("results/dart_task3/dataset/{split}.parquet", split=DART_TASK3_SPLITS),
