"""Smoke test: the one-hot ChromBPNet training loop runs end-to-end and logs the
instrumentation we added (val_count_pearson + val_count_spearman, grad_norm).

Synthetic data, CPU, no GPU — validates forward -> counts/profile loss ->
backward -> optimizer step, plus the validation-epoch correlation hook and the
gradient-norm hook, without real GM12878 data.
"""

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from marin_dna.pipelines.chrombpnet_eval.lit import ChromBPNetLit
from marin_dna.pipelines.chrombpnet_eval.onehot import build_onehot_chrombpnet


class _ToyDS(Dataset):
    def __init__(self, n: int = 8):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        oh = torch.zeros(4, 2114)
        oh[torch.randint(0, 4, (2114,)), torch.arange(2114)] = 1.0
        profile = torch.randint(0, 5, (1000,)).float()
        return {"onehot_seq": oh, "profile": profile}


def _fit(lr_scheduler=None):
    model = build_onehot_chrombpnet(bias_h5=None, n_filters=8, n_layers=2)
    lit = ChromBPNetLit(model, alpha=1.0, beta=1.0, lr=1e-3, lr_scheduler=lr_scheduler)
    dl = DataLoader(_ToyDS(8), batch_size=4)
    trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(lit, dl, dl)
    return trainer


def test_train_loop_and_instrumentation():
    trainer = _fit()
    metrics = {**trainer.callback_metrics, **trainer.logged_metrics}
    # The validation-epoch correlation hook logged both counts metrics ...
    for key in ("val_count_pearson", "val_count_spearman"):
        assert key in metrics, f"{key} not logged; got {sorted(metrics)}"
        assert torch.isfinite(torch.as_tensor(float(metrics[key]))), key
    # ... and the per-step gradient-norm hook fired (a positive, finite norm).
    assert "grad_norm" in metrics, sorted(metrics)
    gn = float(metrics["grad_norm"])
    assert gn > 0 and torch.isfinite(torch.as_tensor(gn)), gn


def test_plateau_scheduler_configures():
    # The opt-in ReduceLROnPlateau path builds a valid optimizer+scheduler dict.
    trainer = _fit(lr_scheduler="plateau")
    assert trainer.lr_scheduler_configs, "plateau scheduler not configured"
