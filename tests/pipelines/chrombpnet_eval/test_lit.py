"""Smoke test: the ChromBPNet training loop runs end-to-end on synthetic data.

fast_dev_run = 1 train + 1 val batch through Lightning, with GLMChromBPNet (tiny
Llama). Validates the forward → counts/profile loss → backward → optimizer step
path and the val_count_pearson hook, without GPU or real data.
"""

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaConfig, LlamaForCausalLM

from marin_dna.pipelines.chrombpnet_eval._vendor.chrombpnet.model_config import (
    ArsenalChromBPNetConfig,
)
from marin_dna.pipelines.chrombpnet_eval.lit import ChromBPNetLit
from marin_dna.pipelines.chrombpnet_eval.model import GLMChromBPNet
from marin_dna.tokenizer.char import create_char_tokenizer


class _ToyDS(Dataset):
    def __init__(self, n: int = 4):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        oh = torch.zeros(4, 2114)
        oh[torch.randint(0, 4, (2114,)), torch.arange(2114)] = 1.0
        profile = torch.randint(0, 5, (1000,)).float()
        return {"onehot_seq": oh, "profile": profile}


def test_train_loop_runs():
    tok = create_char_tokenizer(bos=True, eos=True)
    cfg = LlamaConfig(
        vocab_size=len(tok),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=4096,
    )
    acfg = ArsenalChromBPNetConfig(input_len=2114, n_filters=8, n_layers=2)
    model = GLMChromBPNet(
        LlamaForCausalLM(cfg),
        tok,
        input_len=2114,
        chunk_size=255,
        num_layers_avg=2,
        config=acfg,
        finetune=False,
    )
    lit = ChromBPNetLit(model, alpha=1.0, beta=1.0, lr_head=1e-3)
    dl = DataLoader(_ToyDS(4), batch_size=2)
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(lit, dl, dl)  # 1 train + 1 val batch; raises if anything is wrong
