"""Simplest per-base loss for the #259 *simplify* track: MSE on log-counts per position.

The minimal per-base model — one head predicting a log-count at every position,
trained with plain MSE against ``log1p(observed per-base counts)``. **No** Poisson,
**no** multinomial, **no** target scaling, **no** segments (contrast the faithful
AlphaGenome variant in ``alphagenome.py``). The #259 "simplify to essence"
question: does the *simplest* per-base loss retain QTL-Pearson? It is also simpler
than the two-head ChromBPNet (one head + one loss, vs profile-multinomial +
count-MSE).

Model: zero-padded ('same') tower + a 1x1-conv per-base head -> ``y[B, L]``
(per-position log-count, unconstrained). The QTL total is ``log(sum(expm1(y)))``
(per-base counts summed), so ``forward`` returns ``(y, log_total)`` and
``score_log2fc`` reads ``log_total`` unchanged (same contract as the other arms).
"""

from __future__ import annotations

import torch

from marin_dna.pipelines.chrombpnet_eval.lit import ChromBPNetLit


class PerBaseMSELog(torch.nn.Module):
    """Per-base log-count head on a zero-padded ('same') tower (#259, simplest).

    Forward: ``onehot[B,4,L] -> (y[B,out_window], log_total[B,1])`` where ``y`` is
    the per-position log-count (MSE target ``log1p(count)``) and ``log_total`` is
    ``log(sum(expm1(y)))`` for the QTL log2FC.
    """

    def __init__(
        self,
        *,
        out_window: int,
        n_filters: int = 512,
        n_layers: int = 8,
        conv1_kernel_size: int = 21,
        head_kernel_size: int = 75,
        norm_type: str = "batchnorm",
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        assert norm_type in ("batchnorm", "groupnorm"), norm_type
        self.out_window = out_window
        self.eps = eps
        # Width-preserving 'same'-pad tower (same construction as samepad.py).
        self.iconv = torch.nn.Conv1d(
            4, n_filters, kernel_size=conv1_kernel_size, padding="same"
        )
        self.rconvs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    n_filters, n_filters, kernel_size=3, padding="same", dilation=2**i
                )
                for i in range(1, n_layers + 1)
            ]
        )
        # Normalize the residual-accumulated tower features before the head — without
        # this the raw head output explodes (MSE ~hundreds at init; the two-head
        # sidesteps it via softmax/pooling). #259. ``batchnorm`` (per-channel, but
        # eval uses seed-dependent running stats) vs ``groupnorm`` (1 group =
        # batch-independent, no running stats — testing whether BatchNorm is the
        # per-base seed-variance source).
        self.norm = (
            torch.nn.BatchNorm1d(n_filters)
            if norm_type == "batchnorm"
            else torch.nn.GroupNorm(1, n_filters)
        )
        # Per-base head: a ``head_kernel_size``-wide conv (default 75, matched to the
        # two-head profile conv — controls the single-head "less conv" confounder).
        # **Link = identity** (the MSE target is log1p(count), so the head predicts
        # an unconstrained log-count/position — NO softplus, which is a Poisson link).
        self.head = torch.nn.Conv1d(
            n_filters, 1, kernel_size=head_kernel_size, padding="same"
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] != 4:
            x = x.permute(0, 2, 1)
        x = torch.relu(self.iconv(x))
        for conv in self.rconvs:
            x = x + torch.relu(conv(x))  # residual; 'same' padding keeps width
        x = self.norm(x)  # normalize tower features before the head (tames init)
        y = self.head(x).squeeze(1)  # [B, in_window] per-base log-count (identity link)
        length = y.shape[-1]
        assert self.out_window <= length, (self.out_window, length)
        start = (length - self.out_window) // 2
        y = y[:, start : start + self.out_window]  # [B, out_window]
        # QTL total = sum of per-base counts = sum(expm1(log-count)); fp32 for the
        # exp (bf16-unstable), then log for the readout. y = log1p(count) (the MSE
        # target), so expm1 inverts it back to counts.
        total = torch.expm1(y.float()).clamp_min(0).sum(dim=-1, keepdim=True)
        log_total = torch.log(total + self.eps)  # [B, 1]
        return y, log_total


def build_perbase_mse(
    *,
    out_window: int,
    n_filters: int = 512,
    n_layers: int = 8,
    head_kernel_size: int = 75,
    norm_type: str = "batchnorm",
) -> PerBaseMSELog:
    """Construct the per-base MSE-log model (#259, simplest). See module docstring."""
    return PerBaseMSELog(
        out_window=out_window,
        n_filters=n_filters,
        n_layers=n_layers,
        head_kernel_size=head_kernel_size,
        norm_type=norm_type,
    )


class PerBaseMSELogLit(ChromBPNetLit):
    """Per-base MSE-on-log-counts loss (#259, simplest per-base).

    Subclasses :class:`ChromBPNetLit` only to reuse its optimizer / WSD-schedule /
    grad-norm machinery; just the loss changes. ``alpha``/``beta``/``count_loss``
    are unused.
    """

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        true_profile = batch["profile"]  # [B, out_window] observed per-bp counts
        y, _ = self(batch["onehot_seq"])  # [B, out_window] predicted per-bp log-count
        with torch.autocast(device_type=y.device.type, enabled=False):
            loss = torch.nn.functional.mse_loss(
                y.float(), torch.log1p(true_profile).float()
            )
        self.log_dict({"train_loss": loss}, on_step=True, on_epoch=True, prog_bar=True)
        return loss
