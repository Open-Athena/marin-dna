"""Frozen-gLM ChromBPNet head for the #243 M2 caQTL/dsQTL VEP arm (first cut).

A 256 bp DNA window is fed through a **frozen** genomic LM (exp136-proj_v30,
``hidden_size=1024``); we take **last-layer per-base embeddings for both the
forward strand and the reverse complement**, concatenate them (→ ``2*hidden``
channels), LayerNorm, and run the #259 **same-padding** ChromBPNet tower on top.

This is the simplest M2 cut (#243): no tiling (256 bp = a single gLM window), the
gLM frozen (no fine-tuning), FWD‖RC concat to recover both flanks from the causal
LM, and "the same ChromBPNet architecture on top".

Experimental — **duplicated, not shared** (grug): the same-padding tower is copied
from ``samepad.py`` (the #259 branch) with the first conv widened from 4 (one-hot)
to the embedding dim; everything else is the faithful #259 no-bias two-head
ChromBPNet (profile multinomial + scalar count). ``forward`` obeys the ChromBPNet
contract — ``onehot[B,4,L] -> (profile_logits[B,out_window], log_counts[B,1])`` —
so it drops into :class:`~marin_dna.pipelines.chrombpnet_eval.glm_lit.GLMChromBPNetLit`
and the QTL scorer (:func:`~marin_dna.pipelines.chrombpnet_eval.qtl_eval.score_log2fc`,
which calls ``model(onehot)``) unchanged.

The model takes one-hot (channels A,C,G,T, as ``dna_to_one_hot`` produces) and
maps it to gLM token ids internally, so the existing vendored ``DataModule`` and
the QTL scorer feed it without any change.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# One-hot channel order is A,C,G,T (``dna_to_one_hot``). The exp136-proj_v30
# tokenizer vocab is ``{[PAD]:0, [UNK]:1, [BOS]:2, a:3, c:4, g:5, t:6}``, so the
# A,C,G,T channels map to ids (3,4,5,6), N (an all-zero one-hot column) → [UNK]=1,
# and BOS=2. These defaults match that checkpoint; the driver re-derives them from
# the loaded tokenizer and asserts equality, so a tokenizer change fails loudly
# rather than silently mis-mapping bases.
ACGT_TOKEN_IDS: tuple[int, int, int, int] = (3, 4, 5, 6)
BOS_TOKEN_ID: int = 2
UNK_TOKEN_ID: int = 1


class GLMSamePadChromBPNet(nn.Module):
    """Frozen-gLM ChromBPNet: FWD‖RC per-base embeddings → same-pad conv tower.

    Args:
        glm: a frozen HF base model (``AutoModel``) whose ``forward(input_ids)``
            returns an output with ``.last_hidden_state`` ``[B, T, hidden]``. Set
            to ``requires_grad=False`` and held in ``.eval()`` regardless of the
            outer module's train/eval state (it is never fine-tuned in this arm).
        embedding_dim: channels fed to the conv tower — ``(2 if rc else 1)*hidden``.
        out_window: profile/count output width (a center crop of the
            width-preserving tower output; must be ``<= in_window`` at runtime).
        acgt_token_ids / bos_token_id / unk_token_id: gLM token ids for the A,C,G,T
            channels, BOS, and N — derived from the checkpoint's tokenizer.
        n_filters / n_layers / conv1_kernel_size / profile_kernel_size: the #259
            same-pad tower (defaults = official ChromBPNet: 512 filters, 8 dilated
            layers, wide first/profile convs).
        emb_norm: LayerNorm the concatenated per-base embeddings before the first
            conv (#243 normalization axis). ``False`` → identity.
        rc: concatenate the reverse-complement embeddings (recovers both flanks
            from the causal LM). ``False`` → forward strand only (ablation).

    Forward: ``onehot[B,4,L] -> (profile_logits[B,out_window], log_counts[B,1])``.
    """

    # ``acgt_to_id`` is a registered buffer; annotate it so mypy sees a Tensor
    # (``register_buffer`` is untyped → otherwise ``Tensor | Module``, not indexable).
    acgt_to_id: torch.Tensor

    def __init__(
        self,
        glm: nn.Module,
        *,
        embedding_dim: int,
        out_window: int,
        acgt_token_ids: tuple[int, int, int, int] = ACGT_TOKEN_IDS,
        bos_token_id: int = BOS_TOKEN_ID,
        unk_token_id: int = UNK_TOKEN_ID,
        n_filters: int = 512,
        n_layers: int = 8,
        conv1_kernel_size: int = 21,
        profile_kernel_size: int = 75,
        emb_norm: bool = True,
        rc: bool = True,
    ) -> None:
        super().__init__()
        assert len(acgt_token_ids) == 4, acgt_token_ids
        self.glm = glm
        for p in self.glm.parameters():
            p.requires_grad = False
        self.glm.eval()

        self.embedding_dim = embedding_dim
        self.out_window = out_window
        self.n_layers = n_layers
        self.rc = rc
        self.bos_token_id = bos_token_id
        self.unk_token_id = unk_token_id
        # ACGT(one-hot channel)->token-id lookup; a non-persistent buffer so it
        # moves with ``.to(device)`` but is not written to the state_dict.
        self.register_buffer(
            "acgt_to_id",
            torch.tensor(acgt_token_ids, dtype=torch.long),
            persistent=False,
        )

        # Normalize the concatenated per-base embeddings before the conv tower
        # (user-requested; #243 normalization axis). Trainable affine LayerNorm
        # over the feature dim, applied per position.
        self.emb_norm: nn.Module = (
            nn.LayerNorm(embedding_dim) if emb_norm else nn.Identity()
        )

        # --- #259 same-pad ChromBPNet tower (duplicated from samepad.py), with the
        # first conv widened from 4 (one-hot) to embedding_dim (gLM embeddings).
        # All convs use 'same' padding (stride 1) → width == in_window throughout,
        # no residual crop; the profile is center-cropped to out_window at the end.
        self.iconv = nn.Conv1d(
            embedding_dim, n_filters, kernel_size=conv1_kernel_size, padding="same"
        )
        self.rconvs = nn.ModuleList(
            [
                nn.Conv1d(
                    n_filters, n_filters, kernel_size=3, padding="same", dilation=2**i
                )
                for i in range(1, n_layers + 1)
            ]
        )
        self.fconv = nn.Conv1d(
            n_filters, 1, kernel_size=profile_kernel_size, padding="same"
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(n_filters, 1)

    def train(self, mode: bool = True) -> "GLMSamePadChromBPNet":
        """Keep the frozen gLM in eval mode regardless of the outer train/eval
        toggle (Lightning calls ``model.train()`` each epoch; the LM has no dropout
        anyway, but eval is the correct, explicit state for a frozen encoder)."""
        super().train(mode)
        self.glm.eval()
        return self

    # ---- frozen gLM embedding front-end -----------------------------------
    def _onehot_to_ids(self, onehot: torch.Tensor) -> torch.Tensor:
        """``onehot[B,4,L]`` (channels A,C,G,T) → token ids ``[B, L+1]`` with BOS
        prepended; all-zero columns (N) → unk."""
        is_n = onehot.sum(dim=1) == 0  # [B,L] — all-zero one-hot column = N
        base_idx = onehot.argmax(dim=1)  # [B,L] in 0..3 (0 where all-zero)
        ids = self.acgt_to_id[base_idx]  # [B,L]
        ids = ids.masked_fill(is_n, self.unk_token_id)
        bos = ids.new_full((ids.shape[0], 1), self.bos_token_id)
        return torch.cat([bos, ids], dim=1)  # [B, L+1]

    @torch.no_grad()
    def _embed_one_strand(self, onehot: torch.Tensor) -> torch.Tensor:
        """``onehot[B,4,L]`` → last-layer per-base embeddings ``[B, L, hidden]``
        (BOS prepended for the gLM forward, then its position dropped). Runs under
        ``no_grad`` — the gLM is frozen, so this never needs a backward graph."""
        ids = self._onehot_to_ids(onehot)  # [B, L+1]
        hidden = self.glm(ids).last_hidden_state  # [B, L+1, hidden]
        return hidden[:, 1:, :]  # drop the BOS position → [B, L, hidden]

    def _embed(self, onehot: torch.Tensor) -> torch.Tensor:
        """FWD (and RC, re-aligned) per-base embeddings concatenated along the
        feature dim → ``[B, L, embedding_dim]``."""
        fwd = self._embed_one_strand(onehot)  # [B, L, hidden]
        if not self.rc:
            return fwd
        # RC at the one-hot level: complement (A<->T, C<->G) + reverse the length.
        rc_onehot = onehot[:, [3, 2, 1, 0], :].flip(-1)
        rc = self._embed_one_strand(rc_onehot)  # [B, L, hidden] in RC coordinates
        rc = rc.flip(1)  # reverse the length axis → back to genomic (FWD) order
        return torch.cat([fwd, rc], dim=-1)  # [B, L, 2*hidden]

    def forward(self, onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if onehot.shape[1] != 4:  # accept [B,L,4] too, like the vendored BPNet
            onehot = onehot.permute(0, 2, 1)
        emb = self._embed(onehot)  # [B, L, embedding_dim]
        emb = self.emb_norm(emb)  # LayerNorm over the feature dim
        x = emb.permute(0, 2, 1)  # [B, embedding_dim, L]
        x = torch.relu(self.iconv(x))
        for conv in self.rconvs:
            x = x + torch.relu(conv(x))  # residual; 'same' padding keeps width
        # Profile: 'same'-pad conv over the full embedding, then center-crop.
        profile = self.fconv(x).squeeze(1)  # [B, L]
        length = profile.shape[-1]
        assert self.out_window <= length, (self.out_window, length)
        start = (length - self.out_window) // 2
        profile = profile[:, start : start + self.out_window]  # [B, out_window]
        # Count: global average pool over the full embedding → scalar log-count.
        count = self.linear(self.global_avg_pool(x).squeeze(-1))  # [B, 1]
        return profile, count


def build_glm_samepad_chrombpnet(
    glm: nn.Module,
    *,
    hidden_size: int,
    out_window: int,
    rc: bool = True,
    emb_norm: bool = True,
    n_filters: int = 512,
    n_layers: int = 8,
    acgt_token_ids: tuple[int, int, int, int] = ACGT_TOKEN_IDS,
    bos_token_id: int = BOS_TOKEN_ID,
    unk_token_id: int = UNK_TOKEN_ID,
) -> GLMSamePadChromBPNet:
    """Construct the frozen-gLM same-pad ChromBPNet (#243). ``embedding_dim`` is
    ``2*hidden_size`` with FWD‖RC concat (default) else ``hidden_size``."""
    embedding_dim = (2 if rc else 1) * hidden_size
    return GLMSamePadChromBPNet(
        glm,
        embedding_dim=embedding_dim,
        out_window=out_window,
        rc=rc,
        emb_norm=emb_norm,
        n_filters=n_filters,
        n_layers=n_layers,
        acgt_token_ids=acgt_token_ids,
        bos_token_id=bos_token_id,
        unk_token_id=unk_token_id,
    )
