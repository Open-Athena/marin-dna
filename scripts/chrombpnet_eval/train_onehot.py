"""Train the one-hot ChromBPNet baseline on GM12878 DNase (#241, #259).

The one-hot arm of the supervised caQTL/dsQTL VEP eval. **No accessibility
validation loop** (#259): the all-chromosome protocol trains on every chromosome
(not leakage — accessibility is trained on the reference genome only and VEP is
zero-shot), so there's no held-out accessibility split and nothing to early-stop
on. Instead we train a **fixed budget** with a **WSD** (Warmup-Stable-Decay) LR
schedule and log the **eval target** — the caQTL/dsQTL variant-effect Pearson —
on a **global-step cadence** (decoupled from epoch boundaries) via
``QTLEvalCallback``. In-training health: the train losses + per-step ``grad_norm``.

Faithful one-hot ChromBPNet (vendored class, 4-channel one-hot, frozen pretrained
bias). fp32 by default; ``--precision bf16-mixed`` is faster on an A10G (the model
has a bf16-safe forward). NB ``--all-chroms`` loads all 24 chromosomes' sequences
into RAM — needs a 32 GB+ host (g5.xlarge's 16 GB OOMs; launch with ``--memory
64+``); see the sky task.

Data (stage locally first; from ARSENAL Synapse syn72513540 + a hg38 fasta):
  --peaks    filtered.peaks.bed      (syn73665410)
  --nonpeaks filtered.nonpeaks.bed   (syn73665411)
  --bigwig   GM12878_unstranded.bw   (syn73665418)
  --bias     bias_model_scaled.h5    (syn73665413)
  --fasta    GRCh38...fasta (chr-prefixed; DART-Eval syn60756064)
  --chrom-sizes hg38.chrom.sizes

Example (1 GPU, all-chroms fixed-budget WSD baseline):
  uv run --extra chrombpnet python scripts/chrombpnet_eval/train_onehot.py \
    --peaks gm12878_peaks.bed --nonpeaks gm12878_nonpeaks.bed \
    --bigwig GM12878_unstranded.bw --bias bias_model_scaled.h5 \
    --fasta GRCh38.fasta --chrom-sizes hg38.chrom.sizes \
    --all-chroms --max-steps 12000 --lr-scheduler wsd \
    --qtl-eval --qtl-genome GRCh38.fasta --qtl-chrom-prefix chr \
    --wandb-name dna-exp259-onehot-allchroms-wsd --out-dir runs/onehot

Smoke (log every step, cap the budget):
  ... --log-every-n-steps 1 --qtl-every-steps 20 --limit-train-batches 200 \
      --max-steps 60
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
from marin_dna.pipelines.chrombpnet_eval.alphagenome import (
    AlphaGenomeLit,
    build_alphagenome_perbase,
    estimate_track_mean,
)
from marin_dna.pipelines.chrombpnet_eval.perbase_mse import (
    PerBaseMSELogLit,
    build_perbase_mse,
)
from marin_dna.pipelines.chrombpnet_eval.samepad import build_samepad_chrombpnet
from marin_dna.pipelines.chrombpnet_eval.ema import EMACallback
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
    p.add_argument(
        "--bias", help="Keras .h5 pretrained bias model (required unless --no-bias)"
    )
    p.add_argument("--fasta", required=True, help="hg38 fasta (chr-prefixed)")
    p.add_argument("--chrom-sizes", required=True)
    p.add_argument(
        "--in-window",
        type=int,
        default=2114,
        help="input one-hot length (255 = gLM-native context, #259)",
    )
    p.add_argument(
        "--out-window",
        type=int,
        default=1000,
        help="profile output length (a valid center-crop of in-window given "
        "--n-layers; e.g. 255bp/n_layers=5 -> 35)",
    )
    p.add_argument(
        "--n-layers",
        type=int,
        default=8,
        help="dilated conv layers; fewer = smaller receptive field for short context",
    )
    p.add_argument(
        "--n-filters",
        type=int,
        default=512,
        help="conv-tower width (#259 capacity knob). Params scale ~n_filters^2 at "
        "constant depth/RF: 362 ~= half params, 724 ~= double vs the 512 default.",
    )
    p.add_argument(
        "--head-kernel",
        type=int,
        default=75,
        help="per-base head conv width (--alphagenome/--perbase-mse, #259). 75 "
        "matches the two-head profile conv (controls the single-head 'less conv' "
        "confounder); 1 = AlphaGenome's literal Linear head.",
    )
    p.add_argument(
        "--jitter",
        type=int,
        default=500,
        help="max random-crop jitter (shift); shrink for short context",
    )
    p.add_argument("--out-dir", default="runs/onehot")
    p.add_argument(
        "--all-chroms",
        action="store_true",
        help="train on ALL chromosomes (1-22,X,Y) — the #259 protocol. Not "
        "leakage: accessibility is trained on the reference genome only and VEP "
        "is zero-shot. Needs a 32 GB+ host (launch with --memory 64+).",
    )
    # #259 simplification ablations (does QTL-Pearson survive removing this?)
    p.add_argument(
        "--no-bias",
        action="store_true",
        help="drop the frozen Tn5/DNase bias entirely (no --bias needed). Expected "
        "~neutral for QTL — total counts are bias-independent (paper p.5).",
    )
    p.add_argument(
        "--same-pad",
        action="store_true",
        help="use the zero-padded ('same') ChromBPNet variant (#259, samepad.py): "
        "width-preserving convs so out_window is free (scale it with in_window) "
        "and n_layers/params stay constant across context sizes. No bias. The "
        "context-size study config.",
    )
    p.add_argument(
        "--alphagenome",
        action="store_true",
        help="use the AlphaGenome/Borzoi-style single per-base head + "
        "Poisson-Multinomial loss with target scaling (#259, alphagenome.py). "
        "Zero-padded tower, no bias; out_window scales with in_window.",
    )
    p.add_argument(
        "--perbase-mse",
        action="store_true",
        help="simplest per-base loss (#259, perbase_mse.py): one head, plain MSE "
        "on log1p(per-base counts). No Poisson/multinomial/scaling/segments. "
        "Zero-padded tower, no bias.",
    )
    p.add_argument(
        "--perbase-norm",
        choices=["batchnorm", "groupnorm"],
        default="batchnorm",
        help="pre-head normalization for --perbase-mse (#259). 'groupnorm' is "
        "batch-independent (no eval running stats) — tests whether BatchNorm is "
        "the per-base seed-variance source.",
    )
    p.add_argument(
        "--n-segments",
        type=int,
        default=1,
        help="AlphaGenome Poisson-Multinomial segments (--alphagenome). 1 = "
        "Poisson on the full-window total (right for our short window). AlphaGenome "
        "uses 8 only because their outputs are 8192 bins (~1 Mb) — pure numerical "
        "stability at that scale; moot for a 256-bin window.",
    )
    p.add_argument(
        "--multinomial-weight",
        type=float,
        default=5.0,
        help="AlphaGenome multinomial-term weight (--alphagenome; Borzoi default 5.0)",
    )
    p.add_argument(
        "--count-only",
        action="store_true",
        help="train the count head only (beta=0, no profile NLL); the QTL score "
        "uses only the count head.",
    )
    # #259 loss-function axis (the "key" lever).
    p.add_argument(
        "--count-loss",
        choices=["mse_log", "poisson"],
        default="mse_log",
        help="count-head loss form: 'mse_log' (official ChromBPNet, MSE on "
        "log1p(total)) or 'poisson' (Poisson NLL on the raw total). The QTL "
        "log2FC score is unchanged; alpha is auto-calibrated per form.",
    )
    p.add_argument(
        "--count-weight-scale",
        type=float,
        default=1.0,
        help="multiply the auto-calibrated count-loss weight (alpha) by this — "
        "sweeps the profile/count two-loss balance (#259). The count head is what "
        "the QTL score reads, yet alpha=median/10 down-weights it ~4x vs profile.",
    )
    # training
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="fixed step budget (#259); when set, runs max_epochs=-1. Pair with "
        "--lr-scheduler wsd so the decay lands at the end of the budget.",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3, help="Adam LR (official: 1e-3)")
    p.add_argument(
        "--optimizer",
        choices=["adam", "adamw"],
        default="adam",
        help="'adam' (official ChromBPNet) or 'adamw' (decoupled weight decay, #259)",
    )
    p.add_argument(
        "--weight-decay", type=float, default=0.0, help="weight decay (#259 knob)"
    )
    p.add_argument(
        "--lr-scheduler",
        choices=["none", "wsd"],
        default="none",
        help="'none' = constant LR (+ optional --warmup-steps); 'wsd' = "
        "Warmup-Stable-Decay over the fixed step budget (#259; pair with "
        "--max-steps, --warmup-frac, --decay-frac)",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="seed_everything (reproducibility)"
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="constant-LR linear warmup steps (--lr-scheduler none); tames the "
        "early NaN-prone step (#247). WSD uses --warmup-frac instead.",
    )
    p.add_argument(
        "--warmup-frac",
        type=float,
        default=0.01,
        help="WSD warmup fraction of the step budget (--lr-scheduler wsd); small "
        "(0.01) for these short supervised runs — the early NaN spike (#247) is "
        "tamed by ~100 warmup steps + grad-clip, not a long LLM-style warmup",
    )
    p.add_argument(
        "--decay-frac",
        type=float,
        default=0.2,
        help="WSD decay fraction of the step budget (--lr-scheduler wsd)",
    )
    p.add_argument(
        "--grad-clip",
        type=float,
        default=1000.0,
        help="global-norm gradient clip (0=off); a generous safety net above the "
        "normal grad_norm (~tens-hundreds) to kill spikes (#247)",
    )
    p.add_argument(
        "--ema",
        action="store_true",
        help="track an EMA of the weights and score the QTL metric on the EMA "
        "weights (#259 knob; clean for the no-running-stats two-head / GroupNorm).",
    )
    p.add_argument(
        "--ema-decay", type=float, default=0.999, help="EMA decay (--ema; #259)"
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
        "--wandb-name",
        required=True,
        help="W&B run display name (REQUIRED — no default, so a misconfigured "
        "launch fails loudly instead of silently logging to a stale default name)",
    )
    p.add_argument("--wandb-project", default="chrombpnet-eval")
    p.add_argument("--no-wandb", action="store_true", help="CSVLogger instead of W&B")
    # online QTL metric: signed Pearson of predicted log2FC vs the observed
    # effect, over positives only — the eval target, logged on a global-step
    # cadence (decoupled from epoch boundaries).
    p.add_argument(
        "--qtl-eval",
        action="store_true",
        help="log live caqtl/dsqtl Pearson (+ qtl_avg_pearson) over positives",
    )
    p.add_argument(
        "--qtl-every-steps",
        type=int,
        default=500,
        help="log the QTL metric every N global optimizer steps (#259; decoupled "
        "from epochs)",
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
    # smoke knob (cap train batches per epoch so a short run finishes fast)
    p.add_argument("--limit-train-batches", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    L.seed_everything(args.seed, workers=True)
    # matmul precision: 'highest' = full fp32 (default; TF32 contributed to a NaN
    # divergence, #247). Convs use TF32 via cuDNN regardless.
    torch.set_float32_matmul_precision(args.matmul_precision)

    # #259: optionally train on all chromosomes (more data; not leakage — VEP is
    # zero-shot on the reference). validation_chroms is unused (no val loop).
    chrom_kwargs: dict = {}
    if args.all_chroms:
        chrom_kwargs["training_chroms"] = [f"chr{c}" for c in [*range(1, 23), "X", "Y"]]
        print("[train] --all-chroms: training on all 24 chromosomes (1-22,X,Y)")
    data_config = DataConfig(
        peaks=args.peaks,
        negatives=args.nonpeaks,
        bigwig=args.bigwig,
        fasta=args.fasta,
        chrom_sizes=args.chrom_sizes,
        in_window=args.in_window,
        out_window=args.out_window,
        shift=args.jitter,
        genome="hg38",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        **chrom_kwargs,
    )
    datamodule = DataModule(data_config)
    # Count-loss weight (alpha), calibrated per form so the count-head gradient
    # scale matches across forms, then scaled by --count-weight-scale (#259):
    #   mse_log: ChromBPNet's median/10 scale-balancing heuristic (grad ~O(1)).
    #   poisson: 0.1 = (median/10)/median — poisson's grad is the count-space
    #            residual (~median), so /median brings it onto the mse-log scale.
    if args.count_loss == "poisson":
        alpha = 0.1 * args.count_weight_scale
    else:
        alpha = datamodule.median_count / 10 * args.count_weight_scale
    print(
        f"[train] median_count={datamodule.median_count:.1f}, "
        f"count_loss={args.count_loss}, scale={args.count_weight_scale} -> "
        f"alpha={alpha:.4f}"
    )

    if args.alphagenome:
        # AlphaGenome/Borzoi per-base head (#259): zero-padded tower, no bias,
        # out_window scaled with in_window; the loss + scaling live in AlphaGenomeLit.
        assert args.out_window <= args.in_window, (args.out_window, args.in_window)
        assert not args.bias, "--alphagenome has no bias model (drop --bias)"
        model: torch.nn.Module = build_alphagenome_perbase(
            out_window=args.out_window,
            n_layers=args.n_layers,
            n_filters=args.n_filters,
            head_kernel_size=args.head_kernel,
        )
        n_trainable = count_trainable_params(model)
        print(
            f"[train] AlphaGenome per-base (#259): {n_trainable:,} trainable, "
            f"in_window={args.in_window}, out_window={args.out_window}, "
            f"n_layers={args.n_layers}, n_filters={args.n_filters}, "
            f"n_segments={args.n_segments}, multinomial_weight={args.multinomial_weight}"
        )
    elif args.perbase_mse:
        # Simplest per-base variant (#259): one head, MSE on log1p(per-base counts).
        assert args.out_window <= args.in_window, (args.out_window, args.in_window)
        assert not args.bias, "--perbase-mse has no bias model (drop --bias)"
        model = build_perbase_mse(
            out_window=args.out_window,
            n_layers=args.n_layers,
            n_filters=args.n_filters,
            head_kernel_size=args.head_kernel,
            norm_type=args.perbase_norm,
        )
        n_trainable = count_trainable_params(model)
        print(
            f"[train] per-base MSE-log (#259): {n_trainable:,} trainable, "
            f"in_window={args.in_window}, out_window={args.out_window}, "
            f"n_layers={args.n_layers}, n_filters={args.n_filters}"
        )
    elif args.same_pad:
        # Zero-padded variant (#259): width-preserving convs, no bias, constant
        # n_layers/params across context; out_window scaled with in_window.
        assert args.out_window <= args.in_window, (args.out_window, args.in_window)
        assert not args.bias, "--same-pad has no bias model (drop --bias)"
        model = build_samepad_chrombpnet(
            out_window=args.out_window,
            n_layers=args.n_layers,
            n_filters=args.n_filters,
        )
        n_trainable = count_trainable_params(model)
        print(
            f"[train] same-pad ChromBPNet (#259): {n_trainable:,} trainable, "
            f"in_window={args.in_window}, out_window={args.out_window}, "
            f"n_layers={args.n_layers}, n_filters={args.n_filters}"
        )
    else:
        if not args.no_bias:
            assert args.bias, "--bias is required unless --no-bias"
        model = build_onehot_chrombpnet(
            bias_h5=args.bias,
            use_bias=not args.no_bias,
            out_dim=args.out_window,
            n_layers=args.n_layers,
            n_filters=args.n_filters,
        )
        n_trainable = count_trainable_params(model)
        if args.no_bias:
            print(
                f"[train] one-hot ChromBPNet (NO bias, #259): {n_trainable:,} trainable"
            )
        else:
            n_bias = sum(p.numel() for p in model.bias.parameters())
            print(
                f"[train] one-hot ChromBPNet: {n_trainable:,} trainable params; "
                f"bias frozen ({n_bias:,} params, requires_grad="
                f"{any(p.requires_grad for p in model.bias.parameters())})"
            )
    if args.compile:
        model = torch.compile(model)
        print("[train] torch.compile enabled")

    lit: L.LightningModule
    if args.alphagenome:
        # AlphaGenome target normaliser = genome-wide mean of non-zero coverage
        # (from the bigWig header; DNase: no squashing). QTL log2FC is invariant to it.
        track_mean = estimate_track_mean(args.bigwig)
        print(
            f"[train] alphagenome track_mean (genome-wide mean non-zero)={track_mean:.4f}"
        )
        lit = AlphaGenomeLit(
            model,
            track_mean=track_mean,
            n_segments=args.n_segments,
            multinomial_weight=args.multinomial_weight,
            apply_squashing=False,  # RNA-seq only; DNase/ATAC off (paper)
            lr=args.lr,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            lr_scheduler=None if args.lr_scheduler == "none" else args.lr_scheduler,
            warmup_frac=args.warmup_frac,
            decay_frac=args.decay_frac,
        )
    elif args.perbase_mse:
        lit = PerBaseMSELogLit(
            model,
            lr=args.lr,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            lr_scheduler=None if args.lr_scheduler == "none" else args.lr_scheduler,
            warmup_frac=args.warmup_frac,
            decay_frac=args.decay_frac,
        )
    else:
        if args.count_only:
            print("[train] --count-only: beta=0 (count head only, no profile NLL)")
        lit = ChromBPNetLit(
            model,
            alpha=alpha,
            beta=0.0 if args.count_only else 1.0,
            count_loss=args.count_loss,
            lr=args.lr,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            lr_scheduler=None if args.lr_scheduler == "none" else args.lr_scheduler,
            warmup_frac=args.warmup_frac,
            decay_frac=args.decay_frac,
        )

    logger: object
    if args.no_wandb:
        logger = L.pytorch.loggers.CSVLogger(args.out_dir, name=args.wandb_name)
    else:
        logger = L.pytorch.loggers.WandbLogger(
            name=args.wandb_name, project=args.wandb_project, save_dir=args.out_dir
        )

    # No early-stopping / no monitored selection (#259, fixed budget): just save
    # the final checkpoint; the WSD decay lands the endpoint.
    callbacks: list[L.Callback] = [
        L.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=f"{args.out_dir}/checkpoints", save_last=True, save_top_k=0
        ),
    ]
    # EMA of weights (#259 knob). Added BEFORE the QTL callback so its per-step
    # update runs first → the QTL eval reads the current EMA.
    ema_cb = None
    if args.ema:
        ema_cb = EMACallback(decay=args.ema_decay)
        callbacks.append(ema_cb)
        print(
            f"[train] EMA enabled (decay={args.ema_decay}); QTL scored on EMA weights"
        )
    if args.qtl_eval:
        assert args.qtl_genome, "--qtl-eval requires --qtl-genome"
        # Pre-extract the positives' ref/alt windows once; the callback re-scores
        # the cached one-hots every --qtl-every-steps global steps.
        specs = build_qtl_specs(
            Genome(args.qtl_genome),
            QTL_DATASETS,
            split=args.qtl_split,
            window=args.in_window,
            chrom_prefix=args.qtl_chrom_prefix,
        )
        callbacks.append(
            QTLEvalCallback(
                specs,
                batch_size=args.qtl_batch_size,
                every_n_steps=args.qtl_every_steps,
                ema_callback=ema_cb,
            )
        )
    else:
        print("[train] WARNING: --qtl-eval not set — only train losses are logged")

    trainer = L.Trainer(
        max_epochs=(-1 if args.max_steps else args.max_epochs),
        max_steps=(args.max_steps if args.max_steps else -1),
        reload_dataloaders_every_n_epochs=1,  # resample train negatives each epoch
        accelerator="gpu",
        devices=args.devices,
        precision=args.precision,
        gradient_clip_val=args.grad_clip or None,
        log_every_n_steps=args.log_every_n_steps,
        limit_train_batches=args.limit_train_batches,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(lit, datamodule)
    print(f"[train] done; final checkpoint under {args.out_dir}/checkpoints")


if __name__ == "__main__":
    main()
