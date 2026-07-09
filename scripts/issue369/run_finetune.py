"""Driver for the #369 LoRA missense fine-tuning runs (invoked on a GPU instance).

Modes:
  smoke  — overfit-N sanity (train AUPRC -> ~1.0; gradients flow) + a short dev fold.
  dev    — leave-out-chr1 dev fold, multi-seed, per-epoch val trajectory (wandb).
  nested — full nested leave-one-chromosome-out (the honest headline number).

Example:
  python scripts/issue369/run_finetune.py --model 255M --mode dev --seeds 0,1,2 \
      --rank 8 --lora-lr 1e-4 --max-epochs 30 --out s3://oa-bolinas/analysis/issue369
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from marin_dna.pipelines.finetune.checkpoints import download_checkpoint, model_name
from marin_dna.pipelines.finetune.data import build_or_load_windows, load_missense_train
from marin_dna.pipelines.finetune.model import (
    ATTENTION_MODULES,
    MLP_MODULES,
    build_model,
)
from marin_dna.pipelines.finetune.orchestrate import run_dev_fold, run_nested_loco
from marin_dna.pipelines.finetune.train import TrainConfig, overfit_sanity


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="255M")
    p.add_argument("--mode", default="smoke", choices=["smoke", "dev", "nested"])
    p.add_argument("--window-size", type=int, default=255)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--test-chrom", default="1")
    p.add_argument("--val-chrom", default="3")
    # LoRA capacity
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--target", default="attn", choices=["attn", "attn_mlp"])
    p.add_argument("--top-k-layers", type=int, default=0, help="0 = all layers")
    # training / regularization
    p.add_argument("--lora-lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--pos-weight", type=float, default=0.0, help="0 = plain BCE")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--accum", type=int, default=1, help="grad accumulation (hold eff. batch const)")
    p.add_argument("--compile", action="store_true")
    # wandb / io
    p.add_argument("--wandb-project", default="dna-issue369")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--ckpt-root", default="scratch/issue369/checkpoints")
    p.add_argument("--cache-dir", default="scratch/issue369/windows")
    p.add_argument("--out", default="", help="s3:// prefix to upload results (optional)")
    return p.parse_args()


def make_build_fn(ckpt: str, args: argparse.Namespace, dtype: torch.dtype):
    target = ATTENTION_MODULES if args.target == "attn" else ATTENTION_MODULES + MLP_MODULES
    top_k = args.top_k_layers or None

    def build():
        return build_model(
            ckpt, window_size=args.window_size, lora_rank=args.rank,
            lora_alpha=args.alpha, lora_dropout=args.dropout, target_modules=target,
            top_k_layers=top_k, dtype=dtype,
        )[0]

    return build


def trajectory_rows(results, meta: dict) -> list[dict]:
    rows = []
    for r in results:
        for pt in r.trajectory:
            rows.append({**meta, "seed": r.seed, "test_chrom": r.test_chrom,
                         "val_chrom": r.val_chrom, "num_trainable": r.num_trainable,
                         "best_epoch": r.best_epoch, "best_val_auprc": r.best_val_auprc,
                         "best_test_auprc": r.best_test_auprc, **pt})
    return rows


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")  # A10G Tensor Cores
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[run] device={device} dtype={dtype} model={args.model} mode={args.mode}", flush=True)

    ckpt = str(download_checkpoint(args.model, args.ckpt_root))
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    df = load_missense_train()
    print(f"[run] missense variants: {len(df)} ({int(df['label'].sum())} pos)", flush=True)
    windows = build_or_load_windows(df, tokenizer, args.window_size, args.cache_dir)
    print(f"[run] windows: {len(windows)} (pool [{windows.pool_lo},{windows.pool_hi}))", flush=True)

    build_fn = make_build_fn(ckpt, args, dtype)
    cfg = TrainConfig(
        max_epochs=args.max_epochs, batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size, lora_lr=args.lora_lr, head_lr=args.head_lr,
        weight_decay=args.weight_decay, early_stop_patience=args.patience,
        pos_weight=args.pos_weight or None, num_workers=args.num_workers,
        compile_model=args.compile, accumulate_grad_batches=args.accum,
    )
    seeds = tuple(int(s) for s in args.seeds.split(","))
    meta = {"model": args.model, "mode": args.mode, "rank": args.rank, "target": args.target,
            "top_k_layers": args.top_k_layers, "lora_lr": args.lora_lr,
            "weight_decay": args.weight_decay, "dropout": args.dropout,
            "batch_size": args.batch_size, "max_epochs": args.max_epochs}
    group = (f"{args.model}-r{args.rank}-{args.target}-tk{args.top_k_layers}"
             f"-lr{args.lora_lr:g}-wd{args.weight_decay:g}")

    make_logger = None
    if not args.no_wandb:
        from lightning.pytorch.loggers import WandbLogger

        def make_logger(suffix: str):  # noqa: E731 - small factory
            return WandbLogger(project=args.wandb_project, entity=args.wandb_entity or None,
                               name=f"{group}-{suffix}", group=group, config=meta)

    t0 = time.time()
    rows: list[dict] = []
    summary: dict = {"meta": meta, "group": group}
    if args.mode == "smoke":
        summary["overfit_train_ap"] = overfit_sanity(build_fn(), windows, device)
        short = TrainConfig(**{**cfg.__dict__, "max_epochs": 8})
        results = run_dev_fold(build_fn, windows, short, test_chrom=args.test_chrom,
                               val_chrom=args.val_chrom, seeds=(0,), make_logger=None)
        rows = trajectory_rows(results, meta)
        summary["dev_best_test_auprc"] = [r.best_test_auprc for r in results]
    elif args.mode == "dev":
        results = run_dev_fold(build_fn, windows, cfg, test_chrom=args.test_chrom,
                               val_chrom=args.val_chrom, seeds=seeds, make_logger=make_logger)
        rows = trajectory_rows(results, meta)
        best = [r.best_test_auprc for r in results]
        summary["dev_best_test_auprc"] = best
        summary["dev_mean"], summary["dev_sd"] = float(np.mean(best)), float(np.std(best))
        print(f"[dev] chr{args.test_chrom} best test AUPRC/seed: {[f'{b:.3f}' for b in best]} "
              f"mean={np.mean(best):.3f}+/-{np.std(best):.3f}", flush=True)
    else:  # nested
        _oof, results, overall = run_nested_loco(build_fn, windows, cfg, seed=seeds[0],
                                                 make_logger=make_logger)
        rows = trajectory_rows(results, meta)
        summary["nested_overall_auprc"] = overall
        print(f"[nested] overall per-chrom-weighted AUPRC = {overall:.3f}", flush=True)
    summary["minutes"] = round((time.time() - t0) / 60, 1)

    out_dir = Path("scratch/issue369/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.mode}_{model_name(args.model)}_{int(t0)}"
    if rows:
        pd.DataFrame(rows).to_parquet(out_dir / f"{tag}_trajectory.parquet", index=False)
    (out_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[run] summary: {json.dumps(summary)}", flush=True)
    if args.out:
        import s3fs

        s3fs.S3FileSystem().put(str(out_dir) + "/", args.out.removeprefix("s3://") + f"/{tag}/",
                                recursive=True)
        print(f"[run] uploaded -> {args.out}/{tag}/", flush=True)


if __name__ == "__main__":
    main()
