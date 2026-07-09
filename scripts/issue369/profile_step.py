"""Profile per-step throughput for the #369 harness (issue #369).

Isolates where the ~2 min/epoch at 255M goes: dataloader (num_workers 0 vs 4) vs the
fwd+bwd compute, and how much torch.compile buys. Run on the GPU box:
  .venv/bin/python scripts/issue369/profile_step.py --model 476M --batch 32
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from marin_dna.pipelines.finetune.checkpoints import download_checkpoint
from marin_dna.pipelines.finetune.data import (
    WindowDataset,
    build_or_load_windows,
    chrom_fold_masks,
    load_missense_train,
)
from marin_dna.pipelines.finetune.model import build_model


def bench(fn, n: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="476M")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--rank", type=int, default=8)
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda")

    ckpt = str(download_checkpoint(args.model, "scratch/issue369/checkpoints"))
    tok = AutoTokenizer.from_pretrained(ckpt)
    w = build_or_load_windows(load_missense_train(), tok, 255, "scratch/issue369/windows")
    tr, _, _ = chrom_fold_masks(w.chrom, "1", "3")
    idx = np.where(tr)[0]
    steps_per_epoch = int(np.ceil(len(idx) / args.batch))
    print(f"[prof] {args.model} batch={args.batch} train={len(idx)} steps/epoch={steps_per_epoch}", flush=True)

    # 1. dataloader-only (no model) — is data prep the bottleneck?
    for nw in (0, 4):
        kw = {"multiprocessing_context": "spawn", "persistent_workers": True} if nw else {}
        dl = DataLoader(WindowDataset(w, idx), batch_size=args.batch, shuffle=True,
                        num_workers=nw, pin_memory=True, **kw)
        it = iter(dl)

        def step_dl():
            nonlocal it
            try:
                next(it)
            except StopIteration:
                it = iter(dl)
                next(it)

        ms = bench(step_dl, 40, warmup=5) * 1e3
        print(f"[prof] dataloader-only nw={nw}: {ms:.1f} ms/batch", flush=True)
        del dl, it

    # 2. fwd+bwd compute, no-compile then compile
    r = w.ref_fwd[idx[: args.batch]].to(dev)
    al = w.alt_fwd[idx[: args.batch]].to(dev)
    y = torch.zeros(args.batch, device=dev)
    for comp in (False, True):
        clf = build_model(ckpt, lora_rank=args.rank, dtype=torch.bfloat16)[0].to(dev)
        model = torch.compile(clf) if comp else clf
        opt = torch.optim.AdamW([p for p in clf.parameters() if p.requires_grad], lr=1e-4)

        def fb():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16):
                logit = model(r, al)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y)
            loss.backward()
            opt.step()

        try:
            ms = bench(fb, 20, warmup=8 if comp else 3) * 1e3
            print(f"[prof] fwd+bwd compile={comp}: {ms:.1f} ms/step  "
                  f"=> ~{ms * steps_per_epoch / 1e3:.0f} s/epoch (compute only)", flush=True)
        except Exception as e:
            print(f"[prof] fwd+bwd compile={comp}: FAILED {type(e).__name__}: {e}", flush=True)
        del clf, model, opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
