"""AlphaGenome/Borzoi-style per-base head + Poisson-Multinomial loss (#259).

The faithful version of the "predict per-base counts" idea (vs ChromBPNet's two
separate heads). A *single* per-base head predicts non-negative coverage ``x``,
and **both** loss terms derive from it (as in AlphaGenome / Borzoi):

- a **Poisson** NLL on the per-segment *total* ``sum(x)`` (overall accessibility),
- a **Multinomial** NLL on the within-segment *distribution* ``x / sum(x)`` (shape),
  up-weighted ``5.0`` as in Borzoi.

This is mathematically per-base independent Poisson factorised as Poisson(total) x
Multinomial(shape). The loss is computed in a **scaled** target space and over
**8 segments** for numerical stability — both faithful to AlphaGenome (Methods,
``multinomial_loss``).

Scaling (AlphaGenome ``targets_scaling`` / ``predictions_scaling``): targets are
divided by a per-track mean (of non-zero values), then soft-clipped with a
sqrt-based smooth clip above ``10`` (no ``**0.75`` *squashing* — that's RNA-seq
only, **off for DNase/ATAC**). The model predicts in this scaled space via a
``Softplus(linear) * Softplus(scale)`` head; the inverse ``predictions_scaling``
is only needed to map back to raw units at eval, not for training or for the QTL
log2FC (the track mean cancels in the ref/alt ratio and the soft-clip is
monotone, so ranking is preserved).

The model ``forward`` returns ``(pred[B,out_window], log_total[B,1])`` — the same
``(profile, log_counts)`` contract the QTL scorer reads, so
:class:`~marin_dna.pipelines.chrombpnet_eval.qtl_eval.QTLEvalCallback` /
``score_log2fc`` work unchanged (they read ``log_total`` = log of the predicted
total coverage).
"""

from __future__ import annotations

import torch

from marin_dna.pipelines.chrombpnet_eval.lit import ChromBPNetLit

_CLIP = 10.0  # AlphaGenome soft-clip threshold (in track-mean-normalised units)


def soft_clip_targets(t: torch.Tensor) -> torch.Tensor:
    """AlphaGenome target soft-clip: ``where(t>10, 2*sqrt(10*t)-10, t)``.

    A sqrt-based smooth clip that dampens large (already mean-normalised) values
    above ``10`` while leaving small ones untouched; continuous at the threshold.
    """
    return torch.where(t > _CLIP, 2.0 * torch.sqrt(_CLIP * t.clamp_min(0.0)) - _CLIP, t)


def inverse_soft_clip(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`soft_clip_targets`: ``where(x>10, (x+10)**2/40, x)``."""
    return torch.where(x > _CLIP, (x + _CLIP) ** 2 / (4.0 * _CLIP), x)


def targets_scaling(
    targets: torch.Tensor, track_mean: float, apply_squashing: bool = False
) -> torch.Tensor:
    """Scale experimental targets before the loss (AlphaGenome ``targets_scaling``).

    Divide by ``track_mean`` (per-track mean of non-zero values), optionally apply
    the RNA-seq-only ``**0.75`` squashing (``apply_squashing``; **off for DNase**),
    then soft-clip. Returns the scaled target the model is trained against.
    """
    t = targets / track_mean
    if apply_squashing:
        t = t**0.75
    return soft_clip_targets(t)


def predictions_scaling(
    x: torch.Tensor, track_mean: float, apply_squashing: bool = False
) -> torch.Tensor:
    """Inverse of :func:`targets_scaling` — map scaled predictions back to raw
    units for evaluation against original data (AlphaGenome ``predictions_scaling``).
    """
    x = inverse_soft_clip(x)
    if apply_squashing:
        x = x ** (1.0 / 0.75)
    return x * track_mean


def poisson_multinomial_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_segments: int = 8,
    multinomial_weight: float = 5.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Borzoi/AlphaGenome Poisson-Multinomial loss over ``n_segments`` segments.

    ``pred`` and ``target`` are non-negative ``[B, L]`` (L = out_window; ``target``
    already scaled). The sequence axis is split into ``n_segments`` equal segments;
    within each, a Poisson NLL on the segment total (``/segment_length``) plus a
    Multinomial NLL on the within-segment distribution (``*multinomial_weight``).
    Returns the per-example mean (sum over segments+positions, divided by batch).
    """
    b, length = pred.shape
    assert length % n_segments == 0, (length, n_segments)
    seg = length // n_segments  # multinomial_resolution (segment length)
    pred = pred.reshape(b, n_segments, seg)
    target = target.reshape(b, n_segments, seg)
    sum_pred = pred.sum(dim=-1, keepdim=True)  # [B, n_segments, 1]
    sum_target = target.sum(dim=-1, keepdim=True)
    # Poisson NLL on the segment totals (drop the const log(target!) term).
    poisson = (sum_pred - sum_target * torch.log(sum_pred + eps)).sum()
    # Multinomial NLL on the within-segment distribution.
    mult_prob = pred / (sum_pred + eps)
    positional = (-target * torch.log(mult_prob + eps)).sum()
    return (poisson / seg + multinomial_weight * positional) / b


class AlphaGenomePerBase(torch.nn.Module):
    """Single per-base coverage head on a zero-padded ('same') ChromBPNet tower.

    The same-pad tower (constant width/params, #259) feeds one per-base head:
    ``Softplus(conv) * Softplus(scale)`` (a learnable per-track positive scale,
    init 0). Output is the non-negative scaled-space coverage; the QTL total is
    ``log(sum(coverage))``.

    Forward: ``onehot[B,4,L] -> (pred[B,out_window], log_total[B,1])``.
    """

    def __init__(
        self,
        *,
        out_window: int,
        n_filters: int = 512,
        n_layers: int = 8,
        conv1_kernel_size: int = 21,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
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
        # Per-base head, exactly AlphaGenome ``tracks_scaled_predictions``: a
        # Linear over channels at each position (== a 1x1 conv on the context-rich
        # tower embeddings) -> Softplus, times a learnable per-track positive scale
        # (Softplus(scale), init 0 -> ~0.69), ensuring a non-negative output.
        self.head = torch.nn.Conv1d(n_filters, 1, kernel_size=1)
        self.scale = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] != 4:
            x = x.permute(0, 2, 1)
        x = torch.relu(self.iconv(x))
        for conv in self.rconvs:
            x = x + torch.relu(conv(x))  # residual; 'same' padding keeps width
        h = self.head(x).squeeze(1)  # [B, in_window]
        length = h.shape[-1]
        assert self.out_window <= length, (self.out_window, length)
        start = (length - self.out_window) // 2
        h = h[:, start : start + self.out_window]  # [B, out_window]
        pred = torch.nn.functional.softplus(h) * torch.nn.functional.softplus(
            self.scale
        )
        log_total = torch.log(pred.sum(dim=-1, keepdim=True) + self.eps)  # [B, 1]
        return pred, log_total


def build_alphagenome_perbase(
    *, out_window: int, n_filters: int = 512, n_layers: int = 8
) -> AlphaGenomePerBase:
    """Construct the AlphaGenome-style per-base model (#259). See module docstring."""
    return AlphaGenomePerBase(
        out_window=out_window, n_filters=n_filters, n_layers=n_layers
    )


def estimate_track_mean(datamodule: object) -> float:
    """Per-track mean of non-zero per-bp coverage — the AlphaGenome target
    normaliser (``targets_scaling`` divides by it, setting the soft-clip scale).

    Computed over the same peak+subsampled-negative training loci ``median_count``
    uses (accessing ``median_count`` first populates ``train_val_subsampled``), so
    it is consistent with the rest of the pipeline. The QTL log2FC is invariant to
    this value; it only sets the scaled-space magnitude the model is trained on.
    """
    import pyBigWig

    from marin_dna.pipelines.chrombpnet_eval._vendor.chrombpnet.data_utils import (
        get_cts,
    )

    _ = datamodule.median_count  # populates datamodule.train_val_subsampled
    cts = get_cts(
        datamodule.train_val_subsampled,
        pyBigWig.open(datamodule.config.bigwig),
        datamodule.config.out_window,
    )
    nonzero = cts[cts > 0]
    assert nonzero.size > 0, "no non-zero coverage in the training subsample"
    return float(nonzero.mean())


class AlphaGenomeLit(ChromBPNetLit):
    """Train the per-base model with the scaled-target Poisson-Multinomial loss.

    Subclasses :class:`ChromBPNetLit` purely to reuse its optimizer / WSD-schedule
    / grad-norm machinery; only the loss (``training_step``) changes. The parent's
    ``alpha``/``beta``/``count_loss`` are unused.

    Args:
        model: an :class:`AlphaGenomePerBase` (or any ``forward -> (pred, log_total)``).
        track_mean: per-track mean of non-zero coverage (the AlphaGenome target
            normaliser; sets the soft-clip scale). The QTL log2FC is invariant to it.
        n_segments / multinomial_weight: Poisson-Multinomial loss knobs (8 / 5.0,
            Borzoi/AlphaGenome defaults).
        apply_squashing: the RNA-seq-only ``**0.75`` target squashing; **False for
            DNase/ATAC**.
        lr / optimizer / weight_decay / lr_scheduler / warmup_frac / decay_frac /
            warmup_steps: forwarded to :class:`ChromBPNetLit` (same training recipe).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        track_mean: float,
        n_segments: int = 8,
        multinomial_weight: float = 5.0,
        apply_squashing: bool = False,
        lr: float = 1e-3,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        lr_scheduler: str | None = None,
        warmup_frac: float = 0.01,
        decay_frac: float = 0.2,
        warmup_steps: int = 0,
    ) -> None:
        super().__init__(
            model,
            lr=lr,
            optimizer=optimizer,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler,
            warmup_frac=warmup_frac,
            decay_frac=decay_frac,
            warmup_steps=warmup_steps,
        )
        assert track_mean > 0, track_mean
        self.track_mean = track_mean
        self.n_segments = n_segments
        self.multinomial_weight = multinomial_weight
        self.apply_squashing = apply_squashing

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        true_profile = batch["profile"]  # [B, out_window] observed per-bp counts
        pred, _ = self(batch["onehot_seq"])  # [B, out_window] scaled-space coverage
        # fp32 loss (the log/exp + multinomial are bf16-unstable), as in lit.py.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            target = targets_scaling(
                true_profile.float(), self.track_mean, self.apply_squashing
            )
            loss = poisson_multinomial_loss(
                pred.float(), target, self.n_segments, self.multinomial_weight
            )
        self.log_dict({"train_loss": loss}, on_step=True, on_epoch=True, prog_bar=True)
        return loss
