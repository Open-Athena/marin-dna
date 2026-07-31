"""Plan canonical windows and score them in bounded resumable chunks."""

from pathlib import Path

from marin_dna.pipelines.chinchilla_logo import (
    score_window_plan,
    score_window_plans,
    write_window_plans,
)


rule plan_scope:
    input:
        fasta=FASTA,
        fai=FASTA_INDEX,
        chrom_sizes=CHROM_SIZES,
    output:
        plans=PLAN_OUTPUTS,
        metadata=PLAN_METADATA_OUTPUTS,
        done=PLAN_SCOPE_DONE,
    params:
        context_size=CONTEXT_SIZE,
        stride=STRIDE,
        retain_start=RETAIN_START,
        retain_end=RETAIN_END,
        phase=PHASE,
    run:
        coverage = write_window_plans(
            input.fasta,
            input.chrom_sizes,
            SCAFFOLDS,
            "results/plans",
            context_size=params.context_size,
            stride=params.stride,
            retain_start=params.retain_start,
            retain_end=params.retain_end,
            phase=params.phase,
        )
        assert len(coverage) == len(SCAFFOLDS)
        assert all(Path(path).is_file() for path in output.plans)
        assert all(Path(path).is_file() for path in output.metadata)
        Path(output.done).touch()


rule score_scaffold:
    input:
        plan="results/plans/{scaffold}.parquet",
        fasta=FASTA,
        fai=FASTA_INDEX,
    output:
        done="results/shards/{scaffold}.done.json",
        runtime="results/shards/{scaffold}.runtime.json",
    wildcard_constraints:
        scaffold="|".join(SCAFFOLDS),
    threads: config["inference"]["num_workers"]
    resources:
        gpu=1,
    params:
        shard_dir=lambda wildcards: f"results/shards/{wildcards.scaffold}",
        model_repository=config["model"]["repository"],
        model_revision=config["model"]["revision"],
        context_size=CONTEXT_SIZE,
        # Output-affecting precision policy: changing it must not silently reuse
        # BF16-derived shards as full-precision outputs.
        bf16_full_eval=config["inference"]["bf16_full_eval"],
    run:
        score_window_plan(
            input.plan,
            input.fasta,
            params.shard_dir,
            output.runtime,
            output.done,
            model_repository=params.model_repository,
            model_revision=params.model_revision,
            context_size=params.context_size,
            windows_per_chunk=config["inference"]["windows_per_chunk"],
            batch_size=config["inference"]["batch_size"],
            num_workers=config["inference"]["num_workers"],
            torch_compile=config["inference"]["torch_compile"],
            bf16_full_eval=params.bf16_full_eval,
            eval_accumulation_steps=config["inference"].get(
                "eval_accumulation_steps"
            ),
            application_commit=GIT_COMMIT_SHA,
        )


rule score_scope:
    input:
        plans=PLAN_OUTPUTS,
        plan_done=PLAN_SCOPE_DONE,
        fasta=FASTA,
        fai=FASTA_INDEX,
    output:
        done="results/shards/scope.done.json",
        runtime="results/shards/scope.runtime.json",
    threads: config["inference"]["num_workers"]
    resources:
        gpu=1,
    params:
        shard_root="results/shards",
        model_repository=config["model"]["repository"],
        model_revision=config["model"]["revision"],
        context_size=CONTEXT_SIZE,
        bf16_full_eval=config["inference"]["bf16_full_eval"],
    run:
        assert RESIDENT_MODEL_ACROSS_SCAFFOLDS, (
            "score_scope requires resident_model_across_scaffolds=true"
        )
        score_window_plans(
            input.plans,
            SCAFFOLDS,
            input.fasta,
            params.shard_root,
            output.runtime,
            output.done,
            model_repository=params.model_repository,
            model_revision=params.model_revision,
            context_size=params.context_size,
            windows_per_chunk=config["inference"]["windows_per_chunk"],
            batch_size=config["inference"]["batch_size"],
            num_workers=config["inference"]["num_workers"],
            torch_compile=config["inference"]["torch_compile"],
            bf16_full_eval=params.bf16_full_eval,
            eval_accumulation_steps=config["inference"].get(
                "eval_accumulation_steps"
            ),
            application_commit=GIT_COMMIT_SHA,
        )
