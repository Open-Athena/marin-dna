"""Wire our HF causal-LM embedding adapter into the vendored ChromBPNet.

``GLMChromBPNet`` subclasses the vendored ``ArsenalChromBPNet``, overriding only
the **input representation**: one-hot → our nucleotide token ids → our
bidirectional embedder (tiling + FWD/RC concat, :mod:`...embedding`). The
dilated-CNN accessibility tower, the frozen bias model, the counts/profile
combine, and the multinomial+MSE loss all stay the vendored ChromBPNet — faithful
to "preserve the entire ChromBPNet on top of gLM embeddings" (#236).
"""

from __future__ import annotations

from typing import Any, cast

import torch

from marin_dna.data.transforms import (
    _get_nucleotide_token_ids,
    _get_special_token_counts,
)
from marin_dna.pipelines.chrombpnet_eval._vendor.chrombpnet.bpnet import BPNet
from marin_dna.pipelines.chrombpnet_eval._vendor.chrombpnet.chrombpnet import (
    ArsenalChromBPNet,
)
from marin_dna.pipelines.chrombpnet_eval._vendor.chrombpnet.model_config import (
    ArsenalChromBPNetConfig,
)
from marin_dna.pipelines.chrombpnet_eval.embedding import HFCausalChromBPNetEmbedder

# `data_utils.dna_to_one_hot` orders the channels alphabetically: A, C, G, T.
_ONEHOT_ORDER = ("A", "C", "G", "T")


class GLMChromBPNet(ArsenalChromBPNet):
    """ChromBPNet over a causal gLM's bidirectional per-position embeddings.

    Construct with an HF causal LM + its tokenizer. ``forward(one_hot)`` is the
    vendored ChromBPNet forward, with the one-hot→embedding step replaced by our
    adapter. Call :meth:`load_bias` to load the Tn5/DNase bias ``.h5``.
    """

    def __init__(
        self,
        hf_model: torch.nn.Module,
        tokenizer: Any,
        *,
        input_len: int = 2114,
        chunk_size: int = 255,
        num_layers_avg: int = 6,
        bidirectional: bool = True,
        finetune: bool = False,
        config: ArsenalChromBPNetConfig | None = None,
    ) -> None:
        embedder = HFCausalChromBPNetEmbedder(
            hf_model,
            tokenizer,
            seq_input_size=input_len,
            chunk_size=chunk_size,
            num_layers_avg=num_layers_avg,
            bidirectional=bidirectional,
        )
        if config is None:
            config = ArsenalChromBPNetConfig(input_len=input_len)
        config.input_embedding_dim = embedder.out_dim  # 2*hidden if bidirectional
        config.arsenal_input_size = chunk_size
        config.num_layers_avg = num_layers_avg
        config.arsenal_output_type = "embedding"
        config.finetune_arsenal = finetune
        # super().__init__ builds the BPNet (iconv sized to input_embedding_dim) +
        # a fresh bias BPNet, and freezes `arsenal_model` (== our hf_model) unless
        # finetune. `embedder.model` is the same hf_model object, so the freeze
        # applies to the shared weights; nn.Module.parameters() de-dups them.
        super().__init__(config, arsenal_model=hf_model)
        self.embedder = embedder

        # one-hot channel index (0..3 = A,C,G,T) + N (4) -> our token id.
        nuc_to_id = _get_nucleotide_token_ids(tokenizer)
        n_prefix, _ = _get_special_token_counts(tokenizer)
        n_token_id = tokenizer.encode("N")[n_prefix]
        lut = torch.tensor(
            [nuc_to_id[b] for b in _ONEHOT_ORDER] + [n_token_id], dtype=torch.long
        )
        self.register_buffer("_chan_to_token", lut, persistent=False)

    def one_hot_to_tokens(self, X: torch.Tensor) -> torch.Tensor:
        """``[B, L, 4]`` one-hot → ``[B, L]`` of *our* token ids (all-zero → N)."""
        idx = torch.argmax(X, dim=-1).masked_fill(X.sum(dim=-1) == 0, 4)
        lut = cast(torch.Tensor, self._chan_to_token).to(idx.device)
        return lut[idx]

    def get_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        """``[B, L]`` token ids → ``[B, L, out_dim]`` via our bidirectional adapter."""
        return self.embedder(tokens)

    def forward(
        self, x: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same as the vendored ChromBPNet forward, but the **gLM stays in the
        outer autocast (bf16)** while the ChromBPNet CNN + bias-combine run in
        **fp32**. The multinomial profile logits and the ``exp``/``log``
        count-combine are numerically unstable in bf16 (→ NaN loss); the gLM
        (the compute bottleneck) is fine in bf16."""
        tokens = self.one_hot_to_tokens(x.transpose(1, 2))
        x_embs = self.get_embeddings(tokens)  # gLM forward (bf16 under autocast)
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_embs = x_embs.float()
            acc_profile, acc_counts = self.model(x_embs)
            bias_profile, bias_counts = self.bias(x.float())
            y_profile = acc_profile + bias_profile
            y_counts = self._log(self._exp1(acc_counts) + self._exp2(bias_counts))
        return y_profile.squeeze(1), y_counts

    def load_bias(self, keras_h5_path: str) -> None:
        """Load + freeze the Tn5/DNase bias model from a Keras ``.h5`` (matches the
        vendored ``init_bias``)."""
        self.bias = BPNet.from_keras(keras_h5_path, name="bias")
        for p in self.bias.parameters():
            p.requires_grad = False
