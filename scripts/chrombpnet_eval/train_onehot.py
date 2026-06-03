"""Train the one-hot ChromBPNet baseline on GM12878 DNase (M1b, issue #241).

The fast path that validates the whole training pipeline — data loading, the
counts+profile loss, early-stopping on ``val_count_pearson``, and the W&B
instrumentation (per-step losses + ``grad_norm``, per-epoch ``val_count_pearson``
/ ``val_count_spearman``, ``lr``) — before the slow gLM arm (#243). Faithful
one-hot ChromBPNet (vendored class, 4-channel one-hot, frozen pretrained bias),
fp32 by default (faithful to official one-hot ChromBPNet). A ``--precision
bf16-mixed`` knob is exposed for tuning — on an A10G this arm is GPU-compute-bound
(~97% util at batch 64), so bf16 can actually speed it up.

Data (stage locally first; from ARSENAL Synapse syn72513540 + a hg38 fasta):
  --peaks    filtered.peaks.bed      (syn73665410)
  --nonpeaks filtered.nonpeaks.bed   (syn73665411)
  --bigwig   GM12878_unstranded.bw   (syn73665418)
  --bias     bias_model_scaled.h5    (syn73665413)
  --fasta    GRCh38...fasta (chr-prefixed; DART-Eval syn60756064)
  --chrom-sizes hg38.chrom.sizes

Example (1 GPU, full early-stopped run):
  uv run --extra chrombpnet python scripts/chrombpnet_eval/train_onehot.py \
    --peaks gm12878_peaks.bed --nonpeaks gm12878_nonpeaks.bed \
    --bigwig GM12878_unstranded.bw --bias bias_model_scaled.h5 \
    --fasta GRCh38.fasta --chrom-sizes hg38.chrom.sizes \
    --wandb-name dna-exp236-onehot-chrombpnet --out-dir runs/onehot

Smoke (log every step, validate often, cap batches):
  ... --log-every-n-steps 1 --val-check-interval 50 --limit-train-batches 200 \
      --limit-val-batches 50 --max-epochs 1
"""

from __future__ import annotations

import argparse

import lightning as L
import torch

from marin_dna.pipelines.chrombpnet_eval._vendor.chrombpnet.data_config import (
    DataConfig,
)
from marin_dna.data.genome import Genome
from marin_dna.pipelines.chrombpnet_eval._vendor.chrombpnet.dataset import DataModule
from marin_dna.pipelines.chrombpnet_eval.lit import ChromBPNetLit
from marin_dna.pipelines.chrombpnet_eval.onehot import (
    build_onehot_chrombpnet,
    count_trainable_params,
)
from marin_dna.pipelines.chrombpnet_eval.qtl_eval import (
    QTL_DATASETS,
    QTLEvalCallback,
    build_qtl_specs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # data
    p.add_argument("--peaks", required=True)
    p.add_argument("--nonpeaks", required=True)
    p.add_argument("--bigwig", required=True)
    p.add_argument("--bias", required=True, help="Keras .h5 pretrained bias model")
    p.add_argument("--fasta", required=True, help="hg38 fasta (chr-prefixed)")
    p.add_argument("--chrom-sizes", required=True)
    p.add_argument("--out-dir", default="runs/onehot")
    # training
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3, help="Adam LR (official: 1e-3)")
    p.add_argument(
        "--lr-scheduler",
        choices=["none", "plateau"],
        default="none",
        help="'plateau' = ReduceLROnPlateau on val_count_pearson (ChromBPNet's "
        "commented-out config)",
    )
    p.add_argument("--patience", type=int, default=5, help="early-stop patience")
    p.add_argument(
        "--seed", type=int, default=0, help="seed_everything (reproducibility)"
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="linear LR warmup steps (0=off); tames the early NaN-prone step (#247)",
    )
    p.add_argument(
        "--grad-clip",
        type=float,
        default=1000.0,
        help="global-norm gradient clip (0=off); a generous safety net above the "
        "normal grad_norm (~tens-hundreds) to kill spikes (#247)",
    )
    p.add_argument(
        "--precision",
        default="32",
        help="'32' (default) or 'bf16-mixed'. The model has a bf16-safe forward "
        "(fp32 count-combine), so bf16 won't NaN; it's GPU-bound, so bf16 speeds "
        "it up.",
    )
    p.add_argument(
        "--matmul-precision",
        choices=["highest", "high", "medium"],
        default="highest",
        help="torch.set_float32_matmul_precision; 'highest' = full fp32 matmuls "
        "(default — TF32 'high' contributed to a NaN divergence, see #247). Convs "
        "still use TF32 via cuDNN regardless.",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model (experimental; may fuse the conv tower)",
    )
    p.add_argument("--devices", type=int, default=1)
    # logging cadence
    p.add_argument("--log-every-n-steps", type=int, default=10)
    p.add_argument(
        "--val-check-interval",
        type=float,
        default=None,
        help="validate every N train batches (int >=1) or every fraction of an "
        "epoch (0<f<1, robust to epoch length); default: epoch end",
    )
    p.add_argument("--wandb-name", default="dna-exp236-onehot-chrombpnet")
    p.add_argument("--wandb-project", default="chrombpnet-eval")
    p.add_argument("--no-wandb", action="store_true", help="CSVLogger instead of W&B")
    # online QTL metric: signed Pearson/Spearman of predicted log2FC vs the
    # observed effect, over positives only, logged each validation (the eval
    # target — distinct from val_count_pearson, the accessibility-count fit).
    p.add_argument(
        "--qtl-eval",
        action="store_true",
        help="log live caqtl/dsqtl Pearson/Spearman over positives each val",
    )
    p.add_argument(
        "--qtl-genome",
        default=None,
        help="reference fasta for QTL window extraction — e.g. the staged hg38 "
        "fasta with --qtl-chrom-prefix chr, or the canonical s3:// GRCh38",
    )
    p.add_argument(
        "--qtl-chrom-prefix",
        default="",
        help="prefix prepended to the variant chrom ('chr' for a chr-prefixed fasta)",
    )
    p.add_argument("--qtl-batch-size", type=int, default=256)
    p.add_argument("--qtl-split", default="train", help="QTL split (dev = train)")
    # smoke knobs (cap batches per epoch so a 1-epoch run finishes fast)
    p.add_argument("--limit-train-batches", type=int, default=None)
    p.add_argument("--limit-val-batches", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    L.seed_everything(args.seed, workers=True)
    # matmul precision: 'highest' = full fp32 (default; TF32 contributed to a NaN
    # divergence, #247). Convs use TF32 via cuDNN regardless.
    torch.set_float32_matmul_precision(args.matmul_precision)

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
    # ChromBPNet's scale-balancing heuristic for the counts-loss weight.
    alpha = datamodule.median_count / 10
    print(f"[train] median_count={datamodule.median_count:.1f} -> alpha={alpha:.3f}")

    model = build_onehot_chrombpnet(bias_h5=args.bias)
    n_trainable = count_trainable_params(model)
    n_bias = sum(p.numel() for p in model.bias.parameters())
    print(
        f"[train] one-hot ChromBPNet: {n_trainable:,} trainable params; "
        f"bias frozen ({n_bias:,} params, requires_grad="
        f"{any(p.requires_grad for p in model.bias.parameters())})"
    )
    if args.compile:
        model = torch.compile(model)
        print("[train] torch.compile enabled")

    lit = ChromBPNetLit(
        model,
        alpha=alpha,
        beta=1.0,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        lr_scheduler=None if args.lr_scheduler == "none" else args.lr_scheduler,
    )

    logger: object
    if args.no_wandb:
        logger = L.pytorch.loggers.CSVLogger(args.out_dir, name=args.wandb_name)
    else:
        logger = L.pytorch.loggers.WandbLogger(
            name=args.wandb_name, project=args.wandb_project, save_dir=args.out_dir
        )

    callbacks: list[L.Callback] = [
        L.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        L.pytorch.callbacks.EarlyStopping(
            monitor="val_count_pearson", patience=args.patience, mode="max"
        ),
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=f"{args.out_dir}/checkpoints",
            monitor="val_count_pearson",
            mode="max",
            save_top_k=1,
            filename="best_model",
            save_last=True,
        ),
    ]
    if args.qtl_eval:
        assert args.qtl_genome, "--qtl-eval requires --qtl-genome"
        # Pre-extract the positives' ref/alt windows once; the callback re-scores
        # the cached one-hots each validation.
        specs = build_qtl_specs(
            Genome(args.qtl_genome),
            QTL_DATASETS,
            split=args.qtl_split,
            window=2114,
            chrom_prefix=args.qtl_chrom_prefix,
        )
        callbacks.append(QTLEvalCallback(specs, batch_size=args.qtl_batch_size))

    # int >=1 -> validate every N batches; float in (0,1) -> fraction of an epoch
    # (robust to epoch length); None/0 -> once per epoch.
    vci: int | float = args.val_check_interval or 1.0
    if vci >= 1:
        vci = int(vci)

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        reload_dataloaders_every_n_epochs=1,  # resample train negatives each epoch
        accelerator="gpu",
        devices=args.devices,
        precision=args.precision,
        gradient_clip_val=args.grad_clip or None,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=vci,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(lit, datamodule)
    print(f"[train] done; best checkpoint under {args.out_dir}/checkpoints")


if __name__ == "__main__":
    main()
