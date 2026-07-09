"""Driver for the #369 LoRA missense fine-tuning runs (invoked on a GPU instance).

Modes:
  smoke  — overfit-N sanity (train AUPRC -> ~1.0 on a tiny fixed subset; confirms
           gradients flow through LoRA + head) then a short dev-fold trajectory.
  dev    — leave-out-chr1 dev fold, multi-seed, full per-step trajectory vs the probe line.
  nested — full nested leave-one-chromosome-out (the honest headline number).

Example:
  python scripts/issue369/run_finetune.py --model 255M --mode dev --seeds 0,1,2 \
      --rank 8 --target attn --lora-lr 1e-4 --max-epochs 30 --out s3://oa-bolinas/analysis/issue369
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

from marin_dna.pipelines.evals.metrics import per_chrom_weighted_ap
from marin_dna.pipelines.finetune.checkpoints import download_checkpoint, model_name
from marin_dna.pipelines.finetune.data import (
    GENOME_PATH,
    build_or_load_windows,
    iter_minibatches,
    load_missense_train,
)
from marin_dna.pipelines.finetune.model import (
    ATTENTION_MODULES,
    MLP_MODULES,
    build_model,
)
from marin_dna.pipelines.finetune.orchestrate import run_dev_fold, run_nested_loco
from marin_dna.pipelines.finetune.train import TrainConfig, predict_rc_avg


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
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--no-rc-aug", action="store_true")
    p.add_argument("--pos-weight", type=float, default=0.0, help="0 = plain BCE")
    p.add_argument("--ckpt-root", default="scratch/issue369/checkpoints")
    p.add_argument("--cache-dir", default="scratch/issue369/windows")
    p.add_argument("--out", default="", help="s3:// prefix to upload results (optional)")
    return p.parse_args()


def make_build_fn(ckpt: str, args: argparse.Namespace, dtype: torch.dtype):
    target = ATTENTION_MODULES if args.target == "attn" else ATTENTION_MODULES + MLP_MODULES
    top_k = args.top_k_layers or None

    def build():
        return build_model(
            ckpt,
            window_size=args.window_size,
            lora_rank=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules=target,
            top_k_layers=top_k,
            dtype=dtype,
        )[0]

    return build


def overfit_sanity(build_fn, windows, device, *, n: int = 96, steps: int = 200) -> float:
    """Train on a tiny fixed subset; train AUPRC should climb to ~1.0 (gradients flow)."""
    import torch.nn.functional as F

    rng = np.random.default_rng(0)
    pos = np.where(windows.label == 1)[0]
    neg = np.where(windows.label == 0)[0]
    idx = np.concatenate([rng.choice(pos, n // 2, replace=False), rng.choice(neg, n // 2, replace=False)])
    clf = build_fn().to(device)
    params = clf.trainable_parameters()
    print(f"[smoke] trainable params = {clf.num_trainable():,}", flush=True)
    opt = torch.optim.AdamW(params, lr=3e-4)
    gt = torch.as_tensor(idx, dtype=torch.long)
    y = torch.as_tensor(windows.label[idx], dtype=torch.float32, device=device)
    ac = torch.bfloat16 if device.type == "cuda" else None
    clf.train()
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        r, a = windows.ref_fwd[gt].to(device), windows.alt_fwd[gt].to(device)
        if ac is not None:
            with torch.autocast(device_type=device.type, dtype=ac):
                logit = clf(r, a)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y)
        else:
            loss = F.binary_cross_entropy_with_logits(clf(r, a), y)
        loss.backward()
        gnorm = sum(p.grad.norm().item() for p in params if p.grad is not None)
        opt.step()
        if s % 50 == 0 or s == steps - 1:
            sc = predict_rc_avg(clf, windows, idx, device, 64, ac)
            ap = per_chrom_weighted_ap(windows.label[idx], sc, windows.chrom[idx])
            print(f"[smoke] step {s:4d} loss={loss.item():.3f} gnorm={gnorm:.2e} train_ap={ap:.3f}", flush=True)
            clf.train()
    return ap


def trajectory_rows(results, meta: dict) -> list[dict]:
    rows = []
    for r in results:
        for pt in r.trajectory:
            rows.append({**meta, "test_chrom": r.test_chrom, "val_chrom": r.val_chrom,
                         "num_trainable": r.num_trainable, "best_step": r.best_step,
                         "best_val_auprc": r.best_val_auprc, "best_test_auprc": r.best_test_auprc, **pt})
    return rows


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[run] device={device} dtype={dtype} model={args.model} mode={args.mode}", flush=True)

    ckpt = str(download_checkpoint(args.model, args.ckpt_root))
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    df = load_missense_train()
    print(f"[run] missense variants: {len(df)} ({int(df['label'].sum())} pos)", flush=True)
    windows = build_or_load_windows(df, tokenizer, args.window_size, args.cache_dir)
    print(f"[run] windows built: {len(windows)} (pool [{windows.pool_lo},{windows.pool_hi}))", flush=True)

    build_fn = make_build_fn(ckpt, args, dtype)
    cfg = TrainConfig(
        max_epochs=args.max_epochs, batch_size=args.batch_size, lora_lr=args.lora_lr,
        head_lr=args.head_lr, weight_decay=args.weight_decay, rc_augment=not args.no_rc_aug,
        eval_every=args.eval_every, early_stop_patience=args.patience,
        pos_weight=args.pos_weight or None,
    )
    seeds = tuple(int(s) for s in args.seeds.split(","))
    meta = {"model": args.model, "mode": args.mode, "rank": args.rank, "target": args.target,
            "top_k_layers": args.top_k_layers, "lora_lr": args.lora_lr,
            "weight_decay": args.weight_decay, "dropout": args.dropout,
            "batch_size": args.batch_size, "rc_aug": not args.no_rc_aug}

    t0 = time.time()
    rows: list[dict] = []
    summary: dict = {"meta": meta}
    if args.mode == "smoke":
        ap = overfit_sanity(build_fn, windows, device)
        summary["overfit_train_ap"] = ap
        cfg_short = TrainConfig(**{**cfg.__dict__, "max_epochs": 8})
        results = run_dev_fold(build_fn, windows, cfg_short, device,
                               test_chrom=args.test_chrom, val_chrom=args.val_chrom, seeds=(0,))
        rows = trajectory_rows(results, meta)
        summary["dev_best_test_auprc"] = [r.best_test_auprc for r in results]
    elif args.mode == "dev":
        results = run_dev_fold(build_fn, windows, cfg, device,
                               test_chrom=args.test_chrom, val_chrom=args.val_chrom, seeds=seeds)
        rows = trajectory_rows(results, meta)
        best = [r.best_test_auprc for r in results]
        summary["dev_best_test_auprc"] = best
        summary["dev_mean"], summary["dev_sd"] = float(np.mean(best)), float(np.std(best))
        print(f"[dev] chr{args.test_chrom} best test AUPRC per seed: "
              f"{[f'{b:.3f}' for b in best]} mean={np.mean(best):.3f}±{np.std(best):.3f}", flush=True)
    else:  # nested
        oof, results, overall = run_nested_loco(build_fn, windows, cfg, device, seed=seeds[0])
        rows = trajectory_rows(results, meta)
        summary["nested_overall_auprc"] = overall
        print(f"[nested] overall per-chrom-weighted AUPRC = {overall:.3f}", flush=True)
    summary["minutes"] = round((time.time() - t0) / 60, 1)

    out_dir = Path("scratch/issue369/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.mode}_{args.model}_{model_name(args.model)}_{int(t0)}"
    if rows:
        pd.DataFrame(rows).to_parquet(out_dir / f"{tag}_trajectory.parquet", index=False)
    (out_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[run] summary: {json.dumps(summary)}", flush=True)
    if args.out:
        import s3fs

        fs = s3fs.S3FileSystem()
        fs.put(str(out_dir) + "/", args.out.removeprefix("s3://") + f"/{tag}/", recursive=True)
        print(f"[run] uploaded -> {args.out}/{tag}/", flush=True)


if __name__ == "__main__":
    main()
