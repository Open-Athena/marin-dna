"""Zero-padded ('same') ChromBPNet variant for the #259 context-size study.

The faithful vendored ChromBPNet uses **valid** convolutions, so the output
window shrinks to ``in_window - receptive_field``. That couples three things to
``in_window`` and muddies a context-size sweep: the output window collapses to
~36 bp at short context (a sparse ~8-read count target), and ``n_layers`` (hence
param count) has to grow with context so the receptive field still fits inside
the valid-conv output.

This variant uses **dilation-aware 'same' padding** so every conv preserves
width. That buys three simplifications (#259):

- **out_window is free** (any value ``<= in_window``) — the loss/count window can
  be *scaled* with the input (``out = in`` or ``out = in/2``) instead of pinned at
  ~36 bp, and the count target stops being read-starved.
- **n_layers (and params) stay constant** across context sizes — the receptive
  field no longer has to fit inside the valid-conv output, so shrinking
  ``in_window`` doesn't force fewer layers. The curve then varies *only* the input
  window.
- The residual-crop machinery disappears (each conv preserves width).

The bias is dropped (the #259-adopted no-bias default), so this is just the
accessibility tower + profile/count heads. ``forward`` obeys the ChromBPNet
contract (``onehot[B,4,L] -> (profile[B,out_window], log_counts[B,1])``), so it
drops into :class:`~marin_dna.pipelines.chrombpnet_eval.lit.ChromBPNetLit` and the
QTL scorer unchanged.

Edge note: with 'same' padding the profile's *edge* positions integrate zero-pad,
and at short context the deepest dilations (128, 256) convolve mostly over
padding (a couple of layers partly idle — harmless). But the QTL score is the
count log2FC of a variant at the window **center**, where context is real; the
padded-edge contribution is identical for ref/alt and cancels in the difference,
exactly as the dropped bias does.
"""

from __future__ import annotations

import torch


class SamePadChromBPNet(torch.nn.Module):
    """Accessibility-only ChromBPNet with dilation-aware 'same' padding (#259).

    Args:
        out_window: profile/count output width (a center crop of the
            width-preserving tower output; must be ``<= in_window`` at runtime).
        n_filters / n_layers: accessibility-tower width / depth. Defaults are
            official ChromBPNet (512 filters, 8 dilated layers); with 'same'
            padding these are held **constant** across context sizes, so param
            count does not vary with ``in_window``.
        conv1_kernel_size / profile_kernel_size: the wide first conv (21) and the
            wide profile-head conv (75), as in ChromBPNet.

    Forward: ``onehot[B,4,L] -> (profile_logits[B,out_window], log_counts[B,1])``.
    """

    def __init__(
        self,
        *,
        out_window: int,
        n_filters: int = 512,
        n_layers: int = 8,
        conv1_kernel_size: int = 21,
        profile_kernel_size: int = 75,
    ) -> None:
        super().__init__()
        self.out_window = out_window
        self.n_layers = n_layers
        # All convs use 'same' padding (stride 1), so width == in_window throughout
        # — no valid-conv shrinkage, no residual crop. Kernels are odd and the
        # dilated effective kernels stay odd, so 'same' padding is symmetric.
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
        self.fconv = torch.nn.Conv1d(
            n_filters, 1, kernel_size=profile_kernel_size, padding="same"
        )
        self.global_avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.linear = torch.nn.Linear(n_filters, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] != 4:  # accept [B,L,4] too, like the vendored BPNet
            x = x.permute(0, 2, 1)
        x = torch.relu(self.iconv(x))
        for conv in self.rconvs:
            x = x + torch.relu(conv(x))  # residual; 'same' padding keeps width
        # Profile: 'same'-pad conv over the FULL embedding (so the kept positions
        # see real context), then center-crop to out_window.
        profile = self.fconv(x).squeeze(1)  # [B, in_window]
        length = profile.shape[-1]
        assert self.out_window <= length, (self.out_window, length)
        start = (length - self.out_window) // 2
        profile = profile[:, start : start + self.out_window]  # [B, out_window]
        # Count: global average pool over the full embedding -> scalar log-count.
        count = self.linear(self.global_avg_pool(x).squeeze(-1))  # [B, 1]
        return profile, count


def build_samepad_chrombpnet(
    *,
    out_window: int,
    n_filters: int = 512,
    n_layers: int = 8,
) -> SamePadChromBPNet:
    """Construct the zero-padded ChromBPNet variant (#259). See the module docstring."""
    return SamePadChromBPNet(
        out_window=out_window, n_filters=n_filters, n_layers=n_layers
    )
