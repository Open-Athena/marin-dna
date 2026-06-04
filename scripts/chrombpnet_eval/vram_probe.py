"""Probe peak GPU memory for a one-hot ChromBPNet training step (#241).

Sweeps batch sizes × precisions and reports peak reserved VRAM (the number that
actually OOMs) so we can pick a batch that fits — especially for bf16, which
roughly halves the activation memory and so admits a bigger batch. Diagnostic,
synthetic data, no real inputs:

    uv run --extra chrombpnet python scripts/chrombpnet_eval/vram_probe.py

Mirrors the real training step: bf16-autocast conv towers + the fp32 count-combine
(OneHotChromBPNet) + the fp32 counts/profile loss + an Adam step, with the bias
frozen (as in a real run).
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from marin_dna.pipelines.chrombpnet_eval._vendor.chrombpnet.model_wrappers import (
    multinomial_nll,
)
from marin_dna.pipelines.chrombpnet_eval.onehot import build_onehot_chrombpnet


def _train_step(model: torch.nn.Module, batch_size: int, bf16: bool) -> None:
    dev = "cuda"
    x = torch.zeros(batch_size, 4, 2114, device=dev)
    x[:, torch.randint(0, 4, (2114,)), torch.arange(2114)] = 1.0
    true_profile = torch.randint(0, 5, (batch_size, 1000), device=dev).float()
    true_counts = torch.log1p(true_profile.sum(dim=-1))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
        y_profile, y_count = model(x)
    with torch.autocast(device_type="cuda", enabled=False):
        loss = multinomial_nll(y_profile.float(), true_profile.float()) + F.mse_loss(
            y_count.squeeze(-1).float(), true_counts.float()
        )
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--batches", type=int, nargs="+", default=[64, 128, 192, 256, 384, 512]
    )
    ap.add_argument("--precisions", nargs="+", default=["fp32", "bf16"])
    args = ap.parse_args()

    assert torch.cuda.is_available(), "need a GPU"
    torch.set_float32_matmul_precision("highest")
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}  total {total:.1f} GB")
    print(f"{'batch':>6} {'precision':>10} {'peak_GB':>8}  status")
    for prec in args.precisions:
        for bs in args.batches:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = build_onehot_chrombpnet(bias_h5=None).cuda()
            for p in model.bias.parameters():  # bias is frozen in a real run
                p.requires_grad = False
            try:
                _train_step(model, bs, bf16=(prec == "bf16"))
                peak = torch.cuda.max_memory_reserved() / 1e9
                print(f"{bs:>6} {prec:>10} {peak:>8.1f}  OK")
            except RuntimeError as e:
                status = "OOM" if "out of memory" in str(e).lower() else str(e)[:40]
                print(f"{bs:>6} {prec:>10} {'-':>8}  {status}")
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
