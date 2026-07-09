"""LoRA fine-tuning via PyTorch Lightning (#369).

One `pl.Trainer` fit per fold: RC-augmented training, per-epoch full-FWD+RC validation on
the val chromosome (the overfitting instrument + early-stop signal, logged to wandb), and
a single full-FWD+RC test pass on the **best-val** trainable state. Only the trainable
LoRA+head params are snapshotted (the frozen base is never checkpointed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader

from marin_dna.pipelines.evals.metrics import per_chrom_weighted_ap
from marin_dna.pipelines.finetune.data import VariantWindows, WindowDataset
from marin_dna.pipelines.finetune.lit import LitVariantFinetune
from marin_dna.pipelines.finetune.model import SiameseLoRAClassifier


@dataclass
class TrainConfig:
    """Fine-tuning hyperparameters. Regularization levers (the primary sweep axis) are
    ``max_epochs``/``early_stop_patience`` (training length), ``weight_decay``, ``lora_lr``;
    the capacity levers (rank, top-K-layers, target modules) live in ``build_model``."""

    max_epochs: int = 30
    batch_size: int = 32
    eval_batch_size: int = 256  # inference has no grad/activations — batch big
    lora_lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 0.0  # on LoRA adapters ("pull toward the frozen model")
    head_weight_decay: float = 0.0
    warmup_frac: float = 0.1
    pos_weight: float | None = None  # None = plain BCE (10% positives)
    early_stop_patience: int = 2  # epochs without val-loss improvement -> stop (sharp overfit)
    num_workers: int = 0  # windows are in-memory tensors — 0 is optimal + fork-safe
    compile_model: bool = False
    accumulate_grad_batches: int = 1  # micro-batches per optimizer step (hold eff. batch const)


@dataclass
class FoldResult:
    test_chrom: str
    val_chrom: str
    seed: int = 0
    trajectory: list[dict] = field(default_factory=list)  # per-epoch losses + val AUPRC
    best_epoch: int = 0
    best_val_auprc: float = float("nan")
    best_test_auprc: float = float("nan")
    best_test_scores: np.ndarray | None = None  # test-chrom logits at best-val
    test_idx: np.ndarray | None = None
    num_trainable: int = 0


def _loader(windows: VariantWindows, idx: np.ndarray, batch: int, cfg: TrainConfig,
            *, shuffle: bool) -> DataLoader:
    kw: dict = {"num_workers": cfg.num_workers, "pin_memory": True}
    if cfg.num_workers > 0:  # s3fs was used in the parent (window build) — spawn, not fork
        kw |= {"multiprocessing_context": "spawn", "persistent_workers": True}
    return DataLoader(WindowDataset(windows, idx), batch_size=batch, shuffle=shuffle, **kw)


def run_fold(
    clf: SiameseLoRAClassifier,
    windows: VariantWindows,
    masks: tuple[np.ndarray, np.ndarray, np.ndarray],
    cfg: TrainConfig,
    *,
    seed: int = 0,
    test_chrom: str = "",
    val_chrom: str = "",
    wandb_logger=None,
) -> FoldResult:
    """Fit on ``train`` mask, early-stop/select on ``val`` mask, report on ``test`` mask."""
    L.seed_everything(seed, workers=True)
    train_mask, val_mask, test_mask = masks
    train_idx, val_idx, test_idx = (np.where(m)[0] for m in (train_mask, val_mask, test_mask))
    num_trainable = clf.num_trainable()

    lit = LitVariantFinetune(
        clf, label=windows.label, chrom=windows.chrom,
        lora_lr=cfg.lora_lr, head_lr=cfg.head_lr, weight_decay=cfg.weight_decay,
        head_weight_decay=cfg.head_weight_decay, warmup_frac=cfg.warmup_frac,
        pos_weight=cfg.pos_weight, compile_model=cfg.compile_model,
    )
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        precision="bf16-mixed",
        accelerator="auto",
        devices=1,
        # Lightning divides by this internally, so `estimated_stepping_batches` (used for the
        # cosine schedule total) already counts OPTIMIZER steps, not micro-batches.
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        logger=wandb_logger or False,
        # Stop on val_loss (smooth); the reported checkpoint is still selected by best
        # val_auprc in-module (AUPRC on ~71 val positives is too noisy a stop signal).
        callbacks=[EarlyStopping("val_loss", mode="min", patience=cfg.early_stop_patience)],
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=False,
        log_every_n_steps=10,
    )
    trainer.fit(
        lit,
        _loader(windows, train_idx, cfg.batch_size, cfg, shuffle=True),
        _loader(windows, val_idx, cfg.eval_batch_size, cfg, shuffle=False),
    )

    # Restore the best-val trainable state (tracked in-module), then one full-FWD+RC pass
    # over the test chrom.
    if lit._best_state is not None:
        lit.load_state_dict(lit._best_state, strict=False)
    preds = trainer.predict(
        lit, _loader(windows, test_idx, cfg.eval_batch_size, cfg, shuffle=False)
    )
    scores = np.concatenate([p[0] for p in preds])
    rows = np.concatenate([p[1] for p in preds])
    order = np.argsort(rows)  # align to ascending row index == test_idx order
    test_scores = scores[order]
    test_auprc = per_chrom_weighted_ap(
        windows.label[test_idx], test_scores, windows.chrom[test_idx]
    )
    return FoldResult(
        test_chrom=test_chrom, val_chrom=val_chrom, seed=seed, trajectory=lit._history,
        best_epoch=lit._best_epoch, best_val_auprc=lit._best_val, best_test_auprc=test_auprc,
        best_test_scores=test_scores, test_idx=test_idx, num_trainable=num_trainable,
    )


@torch.no_grad()
def overfit_sanity(clf: SiameseLoRAClassifier, windows: VariantWindows,
                   device: torch.device, *, n: int = 96, steps: int = 200) -> float:
    """Train on a tiny fixed subset; train AUPRC should climb to ~1.0 (gradients flow)."""
    import torch.nn.functional as F

    rng = np.random.default_rng(0)
    pos = np.where(windows.label == 1)[0]
    neg = np.where(windows.label == 0)[0]
    idx = np.concatenate([rng.choice(pos, n // 2, replace=False),
                          rng.choice(neg, n // 2, replace=False)])
    clf = clf.to(device)
    opt = torch.optim.AdamW(clf.trainable_parameters(), lr=3e-4)
    gt = torch.as_tensor(idx, dtype=torch.long)
    y = torch.as_tensor(windows.label[idx], dtype=torch.float32, device=device)
    ac = torch.bfloat16 if device.type == "cuda" else None
    ap = float("nan")
    for s in range(steps):
        clf.train()
        opt.zero_grad(set_to_none=True)
        r, a = windows.ref_fwd[gt].to(device), windows.alt_fwd[gt].to(device)
        with torch.enable_grad():
            if ac is not None:
                with torch.autocast(device_type=device.type, dtype=ac):
                    logit = clf(r, a)
                loss = F.binary_cross_entropy_with_logits(logit.float(), y)
            else:
                loss = F.binary_cross_entropy_with_logits(clf(r, a), y)
            loss.backward()
        opt.step()
        if s % 50 == 0 or s == steps - 1:
            clf.eval()
            rf, af = windows.ref_fwd[gt].to(device), windows.alt_fwd[gt].to(device)
            rr, ar = windows.ref_rc[gt].to(device), windows.alt_rc[gt].to(device)
            with torch.autocast(device_type=device.type, dtype=ac) if ac else _null():
                sc = clf.logit_rc_avg(rf, af, rr, ar).float().cpu().numpy()
            ap = per_chrom_weighted_ap(windows.label[idx], sc, windows.chrom[idx])
            print(f"[smoke] step {s:4d} loss={loss.item():.3f} train_ap={ap:.3f}", flush=True)
    return ap


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False
