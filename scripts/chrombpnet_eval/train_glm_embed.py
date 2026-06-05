"""Train a ChromBPNet head on FROZEN gLM embeddings — GM12878 DNase (#243 M2).

The simplest M2 cut: a **256 bp** window through a **frozen** gLM
(exp136-proj_v30, hidden 1024), **FWD‖RC** last-layer per-base embeddings
concatenated, LayerNorm, then the #259 **same-padding** ChromBPNet tower — trained
with the **#259 recommended recipe** (AdamW wd=0.01 + WSD over a fixed step budget,
``mse_log`` + multinomial-NLL, all chromosomes, **no validation loop / no
early-stop / no EMA**). The eval target — caQTL/dsQTL variant-effect Pearson — is
logged on a global-step cadence so the whole **trajectory** is visible, not just
the final value.

Experimental (#243); the model/lit/callback are duplicated, not shared (grug).

Data (stage locally first; ARSENAL Synapse syn72513540 + a chr-prefixed hg38
fasta) + the gLM checkpoint (the sky task stages all of these):
  --peaks    filtered.peaks.bed    (syn73665410)
  --nonpeaks filtered.nonpeaks.bed (syn73665411)
  --bigwig   GM12878_unstranded.bw (syn73665418)
  --fasta    GRCh38...fasta (chr-prefixed; DART-Eval syn60756064)
  --chrom-sizes hg38.chrom.sizes
  --glm-dir  exp136 HF checkpoint dir (gs://marin-us-central1/.../proj_v30-6692f0/hf/step-9999)

Example (1 GPU, all-chroms fixed-budget WSD):
  uv run --extra chrombpnet python scripts/chrombpnet_eval/train_glm_embed.py \
    --glm-dir data/exp136 \
    --peaks data/gm12878_peaks.bed --nonpeaks data/gm12878_nonpeaks.bed \
    --bigwig data/GM12878_unstranded.bw \
    --fasta data/GRCh38.fasta --chrom-sizes data/hg38.chrom.sizes \
    --all-chroms --max-steps 12000 --precision bf16-mixed --batch-size 128 \
    --qtl-eval --qtl-genome data/GRCh38.fasta --qtl-chrom-prefix chr \
    --wandb-name dna-exp243-glm-frozen-samepad-wsd --out-dir runs/glm_frozen
"""

from __future__ import annotations

import argparse

import lightning as L
import torch
from transformers import AutoModel, AutoTokenizer

from marin_dna.data.genome import Genome
from marin_dna.pipelines.chrombpnet_eval._vendor.chrombpnet.data_config import (
    DataConfig,
)
from marin_dna.pipelines.chrombpnet_eval._vendor.chrombpnet.dataset import DataModule
from marin_dna.pipelines.chrombpnet_eval.glm_embed import build_glm_samepad_chrombpnet
from marin_dna.pipelines.chrombpnet_eval.glm_lit import (
    GLMChromBPNetLit,
    GLMQTLStepCallback,
)
from marin_dna.pipelines.chrombpnet_eval.onehot import count_trainable_params
from marin_dna.pipelines.chrombpnet_eval.qtl_eval import QTL_DATASETS, build_qtl_specs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # frozen gLM
    p.add_argument(
        "--glm-dir",
        required=True,
        help="local HF checkpoint dir for the frozen gLM (exp136-proj_v30 "
        "hf/step-9999); the sky task stages it from GCS",
    )
    p.add_argument(
        "--no-rc",
        action="store_true",
        help="forward strand only (ablation); default concatenates FWD‖RC",
    )
    p.add_argument(
        "--no-emb-norm",
        action="store_true",
        help="drop the LayerNorm on the embeddings before the conv tower (ablation)",
    )
    p.add_argument(
        "--proj-dim",
        type=int,
        default=512,
        help="pointwise (1x1) FWD‖RC fusion width before the conv tower (#243; "
        "default 512 = n_filters, ~half the head params vs the wide conv ingesting "
        "the full 2H concat)",
    )
    p.add_argument(
        "--no-proj",
        action="store_true",
        help="skip the pointwise projection (concat ablation): the wide iconv "
        "ingests the full 2H concat directly",
    )
    # data
    p.add_argument("--peaks", required=True)
    p.add_argument("--nonpeaks", required=True)
    p.add_argument("--bigwig", required=True)
    p.add_argument("--fasta", required=True, help="hg38 fasta (chr-prefixed)")
    p.add_argument("--chrom-sizes", required=True)
    p.add_argument(
        "--in-window",
        type=int,
        default=256,
        help="DNA window fed to the gLM (256 bp; +BOS = 257 tokens, 1 beyond the "
        "255 bp+BOS training window — fine for RoPE)",
    )
    p.add_argument(
        "--out-window",
        type=int,
        default=256,
        help="profile/count output width (center crop of the same-pad tower; "
        "<= in_window)",
    )
    p.add_argument("--n-filters", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument(
        "--jitter",
        type=int,
        default=0,
        help="max random-crop jitter (shift); 0 = deterministic centered window "
        "for the first cut (the gLM is always fed exactly --in-window bp)",
    )
    p.add_argument("--out-dir", default="runs/glm_frozen")
    p.add_argument(
        "--all-chroms",
        action="store_true",
        help="train on ALL chromosomes (1-22,X,Y) — the #259 protocol (not "
        "leakage: VEP is zero-shot on the reference). Needs a 32 GB+ host.",
    )
    # training (#259 recipe defaults)
    p.add_argument(
        "--max-steps",
        type=int,
        default=12000,
        help="fixed step budget (#259); paired with --lr-scheduler wsd",
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="head LR (frozen gLM, head-only; tunable knob)",
    )
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    p.add_argument("--weight-decay", type=float, default=0.01, help="#259 AdamW knob")
    p.add_argument("--lr-scheduler", choices=["none", "wsd"], default="wsd")
    p.add_argument("--warmup-frac", type=float, default=0.01, help="WSD warmup frac")
    p.add_argument("--decay-frac", type=float, default=0.2, help="WSD decay frac")
    p.add_argument(
        "--warmup-steps", type=int, default=100, help="only for --lr-scheduler none"
    )
    p.add_argument("--grad-clip", type=float, default=1000.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--precision",
        default="bf16-mixed",
        help="'bf16-mixed' (default; bf16-safe forward, faster) or '32'",
    )
    p.add_argument(
        "--matmul-precision", choices=["highest", "high", "medium"], default="highest"
    )
    p.add_argument("--devices", type=int, default=1)
    # logging
    p.add_argument("--log-every-n-steps", type=int, default=10)
    p.add_argument("--wandb-name", required=True, help="must include dna-exp243")
    p.add_argument("--wandb-project", default="chrombpnet-eval")
    p.add_argument("--no-wandb", action="store_true")
    # online QTL metric (the eval target, on a global-step cadence)
    p.add_argument("--qtl-eval", action="store_true")
    p.add_argument("--qtl-every-steps", type=int, default=500)
    p.add_argument("--qtl-genome", default=None)
    p.add_argument("--qtl-chrom-prefix", default="")
    p.add_argument("--qtl-batch-size", type=int, default=256)
    p.add_argument("--qtl-split", default="train", help="dev = train")
    # smoke knob
    p.add_argument("--limit-train-batches", type=int, default=None)
    return p.parse_args()


def load_frozen_glm(glm_dir: str) -> tuple[torch.nn.Module, int, dict]:
    """Load the frozen base gLM + derive the one-hot(ACGT)→token-id mapping from
    its tokenizer. Asserts the four base ids are distinct and non-special, so a
    tokenizer surprise fails loudly rather than silently mis-mapping bases."""
    glm = AutoModel.from_pretrained(glm_dir, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(glm_dir, trust_remote_code=True)
    hidden = glm.config.hidden_size
    # one-hot channels are A,C,G,T (dna_to_one_hot); the char tokenizer vocab is
    # lowercase ("a".."t"). Map channel-order ACGT → token ids.
    acgt_ids = tuple(int(i) for i in tok.convert_tokens_to_ids(list("acgt")))
    bos_id, unk_id = tok.bos_token_id, tok.unk_token_id
    assert len(set(acgt_ids)) == 4 and all(i >= 0 for i in acgt_ids), acgt_ids
    assert bos_id is not None and unk_id is not None, (bos_id, unk_id)
    assert not (set(acgt_ids) & {bos_id, unk_id, tok.pad_token_id}), (
        f"ACGT ids {acgt_ids} collide with a special token "
        f"(bos={bos_id}, unk={unk_id}, pad={tok.pad_token_id})"
    )
    print(
        f"[glm] {glm_dir}: hidden={hidden}, ACGT→ids={acgt_ids}, "
        f"bos={bos_id}, unk={unk_id}"
    )
    return (
        glm,
        hidden,
        {
            "acgt_token_ids": acgt_ids,
            "bos_token_id": int(bos_id),
            "unk_token_id": int(unk_id),
        },
    )


def main() -> None:
    args = parse_args()
    assert "dna-exp243" in args.wandb_name, (
        f"--wandb-name must include 'dna-exp243'; got {args.wandb_name!r}"
    )
    assert args.out_window <= args.in_window, (args.out_window, args.in_window)

    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(args.matmul_precision)

    glm, hidden, tok_ids = load_frozen_glm(args.glm_dir)
    proj_dim = None if args.no_proj else args.proj_dim
    model = build_glm_samepad_chrombpnet(
        glm,
        hidden_size=hidden,
        out_window=args.out_window,
        proj_dim=proj_dim,
        rc=not args.no_rc,
        emb_norm=not args.no_emb_norm,
        n_filters=args.n_filters,
        n_layers=args.n_layers,
        **tok_ids,
    )
    n_trainable = count_trainable_params(model)
    n_frozen = sum(p.numel() for p in model.glm.parameters())
    print(
        f"[train] GLMSamePadChromBPNet: {n_trainable:,} trainable (head); "
        f"gLM frozen ({n_frozen:,} params); rc={not args.no_rc}, "
        f"emb_norm={not args.no_emb_norm}, proj_dim={proj_dim}, "
        f"in={args.in_window}, out={args.out_window}"
    )

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
    alpha = datamodule.median_count / 10  # mse_log count-loss weight (ChromBPNet)
    print(f"[train] median_count={datamodule.median_count:.1f} -> alpha={alpha:.4f}")

    lit = GLMChromBPNetLit(
        model,
        alpha=alpha,
        beta=1.0,
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

    # No early-stopping / no monitored selection (#259, fixed budget): save the
    # final checkpoint; the WSD decay lands the endpoint.
    callbacks: list[L.Callback] = [
        L.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=f"{args.out_dir}/checkpoints", save_last=True, save_top_k=0
        ),
    ]
    if args.qtl_eval:
        assert args.qtl_genome, "--qtl-eval requires --qtl-genome"
        specs = build_qtl_specs(
            Genome(args.qtl_genome),
            QTL_DATASETS,
            split=args.qtl_split,
            window=args.in_window,
            chrom_prefix=args.qtl_chrom_prefix,
        )
        callbacks.append(
            GLMQTLStepCallback(
                specs,
                batch_size=args.qtl_batch_size,
                every_n_steps=args.qtl_every_steps,
            )
        )
    else:
        print("[train] WARNING: --qtl-eval not set — only train losses are logged")

    trainer = L.Trainer(
        max_epochs=-1,
        max_steps=args.max_steps,
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
