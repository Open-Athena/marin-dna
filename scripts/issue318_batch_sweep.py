"""VRAM batch-size sweep for #318 embedding runs (post base-model-hook).

After switching the embedding capture from `output_hidden_states=True` (all layers)
to a forward hook on `model.base_model` (last layer only), the per-batch forward is
lighter — but the suffix logits over the full Qwen vocab still scale with batch, so
the real ceiling is empirical. This measures peak GPU memory of
`run_variant_score_bundle(return_embeddings=True, rc=True)` on exp135 across batch
sizes, to pick a safe production `batch_size`.

The model is loaded ONCE (a warm-up settles it to bf16); each batch size scores
exactly one batch (`n == batch_size`) so the peak reflects the per-batch FORWARD
footprint — model + activations + suffix logits + the captured last-hidden states +
one batch of `[N, 2+2D]` predictions. (Full-dataset Trainer prediction accumulation
is a separate, N-dependent cost — see `inference.eval_accumulation_steps`.)

Run on a GPU node:
  uv run --group genome-s3 python scripts/issue318_batch_sweep.py
"""

from __future__ import annotations

import argparse
import os
import tempfile

import s3fs
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin_dna.data.genome import Genome
from marin_dna.model.runner import run_variant_score_bundle

MODEL = "mix-v0.9-p1B-i24-exp135-m5.1-step-59158"
CKPT_S3 = "oa-bolinas/snakemake/analysis/evals_v2/results/checkpoints/" + MODEL
GENOME = (
    "s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115/"
    "Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"
)
MEND_REPO = "bolinas-dna/evals_mendelian_traits"
MEND_REV = "4aed58e50c5dea0b878a665007af2ef9e5108e9f"
WINDOW = 255


def localize_checkpoint() -> str:
    fs = s3fs.S3FileSystem()
    dst = tempfile.mkdtemp(prefix="exp135_ckpt_")
    for f in fs.ls(CKPT_S3, detail=False):
        name = f.rstrip("/").split("/")[-1]
        if not name.startswith("."):
            fs.get_file(f, os.path.join(dst, name))
    return dst


def _score(model, tokenizer, genome, df, batch_size: int) -> None:
    run_variant_score_bundle(
        model,
        tokenizer,
        Dataset.from_pandas(df, preserve_index=False),
        genome,
        WINDOW,
        rc=True,
        return_embeddings=True,
        data_transform_on_the_fly=True,
        inference_kwargs={
            "per_device_eval_batch_size": batch_size,
            "torch_compile": False,
            "bf16_full_eval": True,
            "dataloader_num_workers": 0,  # no DataLoader fork → no s3fs deadlock
            "remove_unused_columns": False,
            "report_to": "none",
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--batches", type=int, nargs="+", default=[32, 64, 96, 128, 192, 256]
    )
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA — run on a GPU"
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)} ({total:.1f} GB total)")

    ckpt = localize_checkpoint()
    model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    genome = Genome(GENOME)
    ds = load_dataset(MEND_REPO, split="train", revision=MEND_REV).to_pandas()

    # Warm-up: settle the model to bf16 on-GPU + CUDA init, so per-batch peaks
    # measure only the forward footprint, not the one-time setup.
    _score(model, tokenizer, genome, ds.iloc[:8].reset_index(drop=True), 8)

    best = None
    for bs in args.batches:
        df = ds.iloc[:bs].reset_index(drop=True)  # exactly one batch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            _score(model, tokenizer, genome, df, bs)
        except torch.cuda.OutOfMemoryError:
            print(f"  batch_size={bs:4d}  OOM")
            break
        alloc = torch.cuda.max_memory_allocated() / 1e9
        reserved = torch.cuda.max_memory_reserved() / 1e9
        pct = reserved / total * 100
        print(
            f"  batch_size={bs:4d}  peak_alloc={alloc:5.2f} GB  "
            f"peak_reserved={reserved:5.2f} GB  ({pct:3.0f}% of {total:.0f} GB)"
        )
        if pct < 85:
            best = bs
    print(f"DONE — largest batch under 85% reserved: {best}")


if __name__ == "__main__":
    main()
