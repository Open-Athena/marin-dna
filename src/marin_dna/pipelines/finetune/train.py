"""LoRA fine-tuning loop with per-step train/val/test AUPRC trajectory logging (#369).

The trajectory *is* the instrument: we watch validation-chromosome AUPRC climb from a
fresh head, (hopefully) clear the frozen-probe baseline line, peak, then overfit. Early
stopping selects the peak on the **validation** chromosome; the reported number is the
**test**-chromosome AUPRC at that val-selected step (leak-free). ``test`` AUPRC is logged
every eval for the plot but never drives selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

from marin_dna.pipelines.evals.metrics import per_chrom_weighted_ap
from marin_dna.pipelines.finetune.data import VariantWindows, iter_minibatches
from marin_dna.pipelines.finetune.model import SiameseLoRAClassifier


@dataclass
class TrainConfig:
    """Fine-tuning hyperparameters. Regularization levers (the primary sweep axis) are
    ``max_epochs``/``early_stop_patience`` (training length), ``weight_decay``,
    ``lora_lr`` — the LoRA capacity levers live in ``build_model``."""

    max_epochs: int = 30
    batch_size: int = 32
    lora_lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 0.0  # on LoRA adapters ("pull toward the frozen model")
    head_weight_decay: float = 0.0
    warmup_frac: float = 0.1
    pos_weight: float | None = None  # None = plain BCE (10% positives)
    rc_augment: bool = True  # random strand per example each step
    eval_every: int = 50  # steps between trajectory points
    early_stop_patience: int = 6  # eval points without val improvement -> stop
    train_probe_subsample: int = 1000  # cap for the train-AUPRC overfit probe


@dataclass
class FoldResult:
    test_chrom: str
    val_chrom: str
    seed: int = 0
    trajectory: list[dict] = field(default_factory=list)  # per eval: step + AUPRCs
    best_step: int = 0
    best_val_auprc: float = float("nan")
    best_test_auprc: float = float("nan")
    best_test_scores: np.ndarray | None = None  # OOF logits on the test chrom
    test_idx: np.ndarray | None = None
    num_trainable: int = 0


def _lora_and_head_params(
    clf: SiameseLoRAClassifier,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    head_ids = {id(p) for p in clf.head.parameters()}
    head = [p for p in clf.head.parameters() if p.requires_grad]
    lora = [
        p
        for p in clf.parameters()
        if p.requires_grad and id(p) not in head_ids
    ]
    return lora, head


@torch.no_grad()
def predict_rc_avg(
    clf: SiameseLoRAClassifier,
    windows: VariantWindows,
    idx: np.ndarray,
    device: torch.device,
    batch_size: int,
    autocast_dtype: torch.dtype | None,
) -> np.ndarray:
    """FWD+RC-averaged logits for the variants at ``idx`` (eval scoring)."""
    clf.eval()
    idx_t = torch.as_tensor(idx, dtype=torch.long)
    out: list[torch.Tensor] = []
    for start in range(0, len(idx_t), batch_size):
        b = idx_t[start : start + batch_size]
        args = [
            windows.ref_fwd[b].to(device),
            windows.alt_fwd[b].to(device),
            windows.ref_rc[b].to(device),
            windows.alt_rc[b].to(device),
        ]
        if autocast_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                logit = clf.logit_rc_avg(*args)
        else:
            logit = clf.logit_rc_avg(*args)
        out.append(logit.float().cpu())
    return torch.cat(out).numpy()


def _auprc(windows: VariantWindows, idx: np.ndarray, scores: np.ndarray) -> float:
    return per_chrom_weighted_ap(windows.label[idx], scores, windows.chrom[idx])


def train_fold(
    clf: SiameseLoRAClassifier,
    windows: VariantWindows,
    masks: tuple[np.ndarray, np.ndarray, np.ndarray],
    cfg: TrainConfig,
    device: torch.device,
    seed: int = 0,
    test_chrom: str = "",
    val_chrom: str = "",
) -> FoldResult:
    """Train on ``train`` mask, early-stop on ``val`` mask, report on ``test`` mask."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    train_mask, val_mask, test_mask = masks
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    # fixed train subsample for the overfit probe (train AUPRC vs val AUPRC gap)
    sub = train_idx
    if len(train_idx) > cfg.train_probe_subsample:
        perm = torch.randperm(len(train_idx), generator=gen).numpy()
        sub = train_idx[perm[: cfg.train_probe_subsample]]

    lora_params, head_params = _lora_and_head_params(clf)
    opt = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": cfg.lora_lr, "weight_decay": cfg.weight_decay},
            {"params": head_params, "lr": cfg.head_lr, "weight_decay": cfg.head_weight_decay},
        ]
    )
    steps_per_epoch = math.ceil(len(train_idx) / cfg.batch_size)
    total_steps = max(1, cfg.max_epochs * steps_per_epoch)
    sched = get_cosine_schedule_with_warmup(
        opt, int(cfg.warmup_frac * total_steps), total_steps
    )
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None
    pos_weight = (
        torch.tensor(cfg.pos_weight, device=device) if cfg.pos_weight else None
    )

    res = FoldResult(
        test_chrom=test_chrom, val_chrom=val_chrom, seed=seed,
        num_trainable=clf.num_trainable(),
    )

    def record(step: int) -> None:
        v = predict_rc_avg(clf, windows, val_idx, device, cfg.batch_size, autocast_dtype)
        t = predict_rc_avg(clf, windows, test_idx, device, cfg.batch_size, autocast_dtype)
        tr = predict_rc_avg(clf, windows, sub, device, cfg.batch_size, autocast_dtype)
        val_ap, test_ap = _auprc(windows, val_idx, v), _auprc(windows, test_idx, t)
        train_ap = _auprc(windows, sub, tr)
        res.trajectory.append(
            {"step": step, "train_auprc": train_ap, "val_auprc": val_ap, "test_auprc": test_ap}
        )
        if not (val_ap <= res.best_val_auprc):  # strictly-better or first finite
            res.best_val_auprc = val_ap
            res.best_test_auprc = test_ap
            res.best_step = step
            res.best_test_scores = t

    ref_fwd, alt_fwd = windows.ref_fwd, windows.alt_fwd
    ref_rc, alt_rc = windows.ref_rc, windows.alt_rc
    labels = torch.as_tensor(windows.label, dtype=torch.float32)

    record(0)  # fresh-head starting point
    step = 0
    stop = False
    for _epoch in range(cfg.max_epochs):
        if stop:
            break
        clf.train()
        for batch in iter_minibatches(len(train_idx), cfg.batch_size, gen):
            g = train_idx[batch.numpy()]
            gt = torch.as_tensor(g, dtype=torch.long)
            if cfg.rc_augment:
                use_rc = torch.randint(0, 2, (len(g),), generator=gen).bool()
                r = torch.where(use_rc[:, None], ref_rc[gt], ref_fwd[gt])
                a = torch.where(use_rc[:, None], alt_rc[gt], alt_fwd[gt])
            else:
                r, a = ref_fwd[gt], alt_fwd[gt]
            r, a = r.to(device), a.to(device)
            y = labels[gt].to(device)
            opt.zero_grad(set_to_none=True)
            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    logit = clf(r, a)
                loss = F.binary_cross_entropy_with_logits(logit.float(), y, pos_weight=pos_weight)
            else:
                logit = clf(r, a)
                loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % cfg.eval_every == 0:
                record(step)
                evals_since_best = (step - res.best_step) // cfg.eval_every
                if evals_since_best >= cfg.early_stop_patience:
                    stop = True
                    break
    if step % cfg.eval_every != 0:
        record(step)  # final point
    res.test_idx = test_idx
    return res
