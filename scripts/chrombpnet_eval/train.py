"""Train a ChromBPNet head on a gLM's bidirectional embeddings (M2, issue #236).

GM12878 DNase accessibility → ChromBPNet (counts + profile loss), with the
one-hot input replaced by our causal-gLM embedding adapter (FWD+RC concat). Early
stops on ``val_count_pearson``; logs to W&B. The trained checkpoint is then scored
on caQTL/dsQTL via the M1a harness (separate scorer).

Data (stage locally first; from ARSENAL Synapse syn72513540 + a hg38 fasta):
  --peaks  filtered.peaks.bed      (syn73665410)
  --nonpeaks filtered.nonpeaks.bed (syn73665411)
  --bigwig GM12878_unstranded.bw   (syn73665418)
  --bias   bias_model_scaled.h5    (syn73665413)
  --fasta  GRCh38_no_alt_analysis_set...fasta (chr-prefixed; DART-Eval syn60581044)
  --chrom-sizes hg38.chrom.sizes

Model: an HF causal gLM dir + its char tokenizer, e.g. exp136-proj_v30 0.6B
  gs://marin-us-central1/checkpoints/...proj_v30-6692f0/hf/step-9999
  (`gsutil -m cp -r` it local, then pass --hf-model <dir> --tokenizer <dir|repo>).

Example (1-GPU):
  uv run --extra chrombpnet python scripts/chrombpnet_eval/train.py \
    --hf-model ./proj_v30 --tokenizer ./proj_v30 \
    --peaks gm12878_peaks.bed --nonpeaks gm12878_nonpeaks.bed \
    --bigwig GM12878_unstranded.bw --bias bias_model_scaled.h5 \
    --fasta GRCh38.fa --chrom-sizes hg38.chrom.sizes \
    --wandb-name dna-exp236-proj_v30-chrombpnet --out-dir runs/proj_v30
"""

from __future__ import annotations

import argparse

import lightning as L
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin_dna.pipelines.chrombpnet_eval._vendor.chrombpnet.data_config import (
    DataConfig,
)
from marin_dna.pipelines.chrombpnet_eval._vendor.chrombpnet.dataset import DataModule
from marin_dna.pipelines.chrombpnet_eval.lit import ChromBPNetLit
from marin_dna.pipelines.chrombpnet_eval.model import GLMChromBPNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-model", required=True, help="local HF causal-LM dir (gLM)")
    p.add_argument("--tokenizer", required=True, help="HF tokenizer dir or repo id")
    p.add_argument("--peaks", required=True)
    p.add_argument("--nonpeaks", required=True)
    p.add_argument("--bigwig", required=True)
    p.add_argument("--bias", required=True, help="Keras .h5 bias model")
    p.add_argument("--fasta", required=True, help="hg38 fasta (chr-prefixed)")
    p.add_argument("--chrom-sizes", required=True)
    p.add_argument("--out-dir", default="runs/chrombpnet")
    p.add_argument(
        "--wandb-name", default="dna-exp236-chrombpnet", help="includes dna-exp<N>"
    )
    p.add_argument("--wandb-project", default="chrombpnet-eval")
    # representation knobs
    p.add_argument(
        "--chunk-size", type=int, default=255, help="gLM window (255 bp + BOS)"
    )
    p.add_argument("--num-layers-avg", type=int, default=6)
    p.add_argument(
        "--no-bidirectional", action="store_true", help="causal-only ablation"
    )
    p.add_argument(
        "--finetune", action="store_true", help="fine-tune the LM end-to-end"
    )
    # training
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr-head", type=float, default=1e-4)
    p.add_argument("--lr-lm", type=float, default=1e-5)
    p.add_argument("--precision", default="32")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--log-every-n-steps", type=int, default=10)
    p.add_argument(
        "--val-check-interval",
        type=int,
        default=None,
        help="validate every N train batches (default: epoch end)",
    )
    # smoke knobs (cap batches per epoch so a 1-epoch run finishes fast)
    p.add_argument("--limit-train-batches", type=int, default=None)
    p.add_argument("--limit-val-batches", type=int, default=None)
    p.add_argument("--no-wandb", action="store_true", help="CSVLogger instead of W&B")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_config = DataConfig(
        peaks=args.peaks,
        negatives=args.nonpeaks,
        bigwig=args.bigwig,
        fasta=args.fasta,
        chrom_sizes=args.chrom_sizes,
        in_window=2114,
        out_window=1000,
        genome="hg38",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule = DataModule(data_config)
    # ChromBPNet's scale-balancing heuristic for the counts loss weight.
    alpha = datamodule.median_count / 10
    print(f"[train] median_count={datamodule.median_count:.1f} -> alpha={alpha:.3f}")

    hf_model = AutoModelForCausalLM.from_pretrained(args.hf_model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = GLMChromBPNet(
        hf_model,
        tokenizer,
        input_len=2114,
        chunk_size=args.chunk_size,
        num_layers_avg=args.num_layers_avg,
        bidirectional=not args.no_bidirectional,
        finetune=args.finetune,
    )
    model.load_bias(args.bias)
    print(
        f"[train] embedder out_dim={model.embedder.out_dim} (bidirectional={not args.no_bidirectional})"
    )

    lit = ChromBPNetLit(
        model,
        alpha=alpha,
        beta=1.0,
        lr_head=args.lr_head,
        lr_lm=args.lr_lm,
        finetune=args.finetune,
    )

    logger: object
    if args.no_wandb:
        logger = L.pytorch.loggers.CSVLogger(args.out_dir, name=args.wandb_name)
    else:
        logger = L.pytorch.loggers.WandbLogger(
            name=args.wandb_name, project=args.wandb_project, save_dir=args.out_dir
        )
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        reload_dataloaders_every_n_epochs=1,
        accelerator="gpu",
        devices=args.devices,
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval or 1.0,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        logger=logger,
        callbacks=[
            L.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
            L.pytorch.callbacks.EarlyStopping(
                monitor="val_count_pearson", patience=5, mode="max"
            ),
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath=f"{args.out_dir}/checkpoints",
                monitor="val_count_pearson",
                mode="max",
                save_top_k=1,
                filename="best_model",
                save_last=True,
            ),
        ],
    )
    trainer.fit(lit, datamodule)
    print(f"[train] done; best checkpoint under {args.out_dir}/checkpoints")


if __name__ == "__main__":
    main()
