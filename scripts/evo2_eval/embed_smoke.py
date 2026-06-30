"""GH200 smoke for the Evo2 embedding bundle (issue #131).

Runs the *production* path (``compute_evo2_bundle`` with ``return_embeddings=True``)
on a small slice of ``evals_mendelian_traits`` with a real Evo2 checkpoint — the
things the CPU stub tests can't cover. Touches no production parquet.

Asserts, in order:

  (a) ``--emb-layer`` resolves against the **real** Evo2 module tree. The bundle
      validates this internally and fails loud with the candidate names; ``--list-
      layers`` dumps the module names without scoring (run it first if unsure).
  (b) **VRAM** — peak torch allocation of the embedding forward fits the GPU.
  (c) **LLR unchanged with embeddings on vs off** — the per-strand ``llr``/``jsd``
      from a ``return_embeddings=True`` run must match a ``return_embeddings=False``
      run on the same variants (requesting the hidden state must not perturb the
      logits; the hook is non-invasive in principle, but that's Evo2-internal).
  (d) **f16 storage doesn't wreck the probe feature** — scores once in ``float32``
      (un-rounded), then compares the probe delta ``emb_alt − emb_ref`` computed in
      f32 vs round-tripped through f16. Reports the |emb| magnitude distribution
      (massive-activation channels show up here) and the f16-vs-f32 delta Pearson;
      asserts the correlation is high. If it fails, re-score with
      ``--emb-dtype float32`` (the bundle's lossless escape hatch).

Run on a GH200 in the evo2 docker (the eval_matched_pair image has evo2):
  python scripts/evo2_eval/embed_smoke.py --model evo2_1b_base --n 128 --batch-size 8
  python scripts/evo2_eval/embed_smoke.py --model evo2_1b_base --list-layers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from marin_dna.pipelines.evals.evo2 import EVO2_MODEL_CHOICES

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _evo2_scoring import compute_evo2_bundle  # noqa: E402, I001

MEND_REPO = "bolinas-dna/evals_mendelian_traits"


def list_layers(model_name: str) -> None:
    """Load Evo2 and print its module names (final stack highlighted), then exit.

    Cheap layer-discovery path: no scoring, no genome. Use to pick ``--emb-layer``
    when the default ``norm`` isn't confirmed for a given checkpoint.
    """
    from evo2 import Evo2

    evo2 = Evo2(model_name)
    names = [n for n, _ in evo2.model.named_modules()]
    top = [n for n in names if n and n.count(".") == 0]
    finalish = [n for n in names if "norm" in n or n.endswith("unembed")]
    print(f"[layers] {model_name}: {len(names)} modules")
    print(f"[layers] top-level: {top}")
    print(f"[layers] norm/unembed candidates: {sorted(finalish)}")


def _score(
    df, *, model, genome, bs, window, emb_layer, return_embeddings, emb_dtype="float16"
):
    return compute_evo2_bundle(
        model_name=model,
        df=df[["chrom", "pos", "ref", "alt"]],
        genome_path=genome,
        window_size=window,
        batch_size=bs,
        rc_avg=True,
        return_embeddings=return_embeddings,
        emb_layer=emb_layer,
        emb_dtype=emb_dtype,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="evo2_1b_base", choices=EVO2_MODEL_CHOICES)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--window-size", type=int, default=8192)
    ap.add_argument("--emb-layer", default="norm")
    ap.add_argument("--genome-path", default="results/genome.fa.gz")
    ap.add_argument(
        "--list-layers",
        action="store_true",
        help="Just load the model and print its module names, then exit.",
    )
    ap.add_argument(
        "--pearson-min",
        type=float,
        default=0.98,
        help="Assert the f16-vs-f32 delta Pearson exceeds this (else use "
        "--emb-dtype float32 in production).",
    )
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA — this smoke must run on a GPU"
    print("GPU:", torch.cuda.get_device_name(0))

    if args.list_layers:
        list_layers(args.model)
        return

    ds = load_dataset(MEND_REPO, split="train").to_pandas()
    slice_df = ds.iloc[: args.n].reset_index(drop=True)
    print(
        f"[smoke] {args.model}: scoring {len(slice_df)} variants, emb_layer="
        f"{args.emb_layer!r}, window={args.window_size}"
    )

    # --- (a) layer resolves + (b) VRAM + (d) emb at float32 (un-rounded ref) ---
    torch.cuda.reset_peak_memory_stats()
    emb = _score(
        slice_df,
        model=args.model,
        genome=args.genome_path,
        bs=args.batch_size,
        window=args.window_size,
        emb_layer=args.emb_layer,
        return_embeddings=True,
        emb_dtype="float32",
    )
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[b] peak torch VRAM (embeddings on): {peak_gb:.2f} / {total_gb:.0f} GB")
    assert peak_gb < total_gb, "embedding forward exceeded GPU memory"

    assert "emb_ref" in emb.columns and "emb_alt" in emb.columns
    ref = np.stack(emb["emb_ref"].to_numpy()).astype(np.float32)
    alt = np.stack(emb["emb_alt"].to_numpy()).astype(np.float32)
    assert ref.shape == alt.shape and ref.shape[0] == len(slice_df)
    assert np.isfinite(ref).all() and np.isfinite(alt).all(), "non-finite embeddings"
    d = ref.shape[1]
    mag = np.abs(np.concatenate([ref, alt]))
    print(
        f"[a] emb_layer {args.emb_layer!r} resolved; D={d}; "
        f"|emb| p50={np.percentile(mag, 50):.3f} p99={np.percentile(mag, 99):.3f} "
        f"max={mag.max():.3f}  (channels with |emb|>50: "
        f"{int((np.abs(ref).max(axis=0) > 50).sum())}/{d})"
    )

    # --- (d) f16 vs f32 delta corruption ---
    delta32 = alt - ref
    ref16 = ref.astype(np.float16).astype(np.float32)
    alt16 = alt.astype(np.float16).astype(np.float32)
    delta16 = alt16 - ref16
    # Global Pearson over every (variant, channel) entry.
    a, b = delta16.ravel(), delta32.ravel()
    pearson = float(np.corrcoef(a, b)[0, 1])
    rel_l2 = float(
        np.linalg.norm(delta16 - delta32) / (np.linalg.norm(delta32) + 1e-12)
    )
    print(
        f"[d] probe delta (alt−ref): |delta32| mean={np.abs(delta32).mean():.4f}; "
        f"f16-vs-f32 Pearson={pearson:.5f}  rel_L2_err={rel_l2:.4f}"
    )
    assert pearson > args.pearson_min, (
        f"f16 storage corrupts the probe delta (Pearson {pearson:.4f} < "
        f"{args.pearson_min}) — Evo2 massive activations likely; re-score the "
        f"production bundle with --emb-dtype float32"
    )

    # --- (c) LLR/JSD unchanged with embeddings off ---
    base = _score(
        slice_df,
        model=args.model,
        genome=args.genome_path,
        bs=args.batch_size,
        window=args.window_size,
        emb_layer=args.emb_layer,
        return_embeddings=False,
    )
    max_diff = 0.0
    for col in (
        "llr_fwd",
        "llr_rev",
        "next_token_jsd_mean_fwd",
        "next_token_jsd_mean_rev",
    ):
        max_diff = max(
            max_diff, float(np.abs(emb[col].to_numpy() - base[col].to_numpy()).max())
        )
    print(f"[c] max |llr/jsd diff| (emb on vs off): {max_diff:.2e}")
    assert max_diff < 1e-3, (
        f"llr/jsd drifted with embeddings requested ({max_diff:.2e}) — Evo2's "
        f"return_embeddings path is perturbing the logits"
    )

    print("SMOKE PASSED")


if __name__ == "__main__":
    main()
