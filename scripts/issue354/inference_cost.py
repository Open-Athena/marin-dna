"""GH200 steady-state inference-cost benchmark for exp135-1B-m5.1 (issue #354).

Measures steady-state variant-scoring throughput + $/1k variants + peak VRAM on
whatever single GPU it runs on, using the **real** evals_v2 scoring path
(``run_variant_score_bundle``) with the production inference config: FWD+RC,
bf16, torch.compile, and ``return_embeddings=True`` (the ~2x-heavier
``output_hidden_states`` forward — the like-for-like match to the Evo 2 #131
numbers, which were measured with embeddings on). Storage of the pooled
embeddings stays our standard f16; that cast is post-forward and does not move
throughput or peak VRAM.

Three things happen:

1. **Compile validation** — the #318 embedding overlay disabled torch.compile
   because "compiling the hooked forward is unvalidated". This benchmark needs
   both, so we first score a tiny subset eager vs compiled (embeddings on) and
   report the max |Δllr| / |Δjsd|. A crash here means compile+embeddings is
   broken; a small delta is the expected AUPRC-invariant float-reduction noise.

2. **Batch-size sweep** — for each ``--batch-sizes`` value, one full FWD+RC
   ``return_embeddings=True`` pass over the dataset with a per-batch
   ``TimingCallback``. Steady-state seconds/strand-batch = median of the
   inter-step diffs (robustly drops the one compile batch + the FWD→RC
   boundary gap). ``variants/hr = 3600 * B / (2 * sec_per_strand_batch)`` (the
   factor 2 = both strands per unique variant, matching Evo 2's FWD+RC
   accounting). Peak VRAM from ``torch.cuda.max_memory_{reserved,allocated}``.

3. **Scores dump** — one scores parquet (llr/jsd atoms + f16 emb_ref/emb_alt,
   joined to the variant columns) so the derived SGE AUPRC can be checked
   against the existing official cell offline (regression check).

Outputs ``inference_cost.json`` (per-batch-size rows + metadata) and
``sge_scores.parquet`` to ``--out-dir``. GPU-only (bf16_full_eval errors on
CPU). Run via ``scripts/issue354/run.yaml`` on a Lambda GH200.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

from marin_dna.data.genome import Genome
from marin_dna.model.runner import run_variant_score_bundle
from marin_dna.pipelines.evals.inference import fwd_rc_average_f16

REQUIRED_COLS = ["chrom", "pos", "ref", "alt"]
SECONDS_PER_HOUR = 3600.0
BYTES_PER_GB = 1024**3


class TimingCallback(TrainerCallback):
    """Record ``perf_counter`` after every prediction step (post-forward)."""

    def __init__(self) -> None:
        self.step_times: list[float] = []

    def on_prediction_step(self, args, state, control, **kwargs) -> None:  # noqa: ANN001
        self.step_times.append(time.perf_counter())


def _load_variants(source: str, revision: str | None, split: str) -> pd.DataFrame:
    """Load the variant table from an HF dataset id or a local parquet path."""
    if Path(source).exists():
        df = pd.read_parquet(source)
    else:
        ds = load_dataset(source, split=split, revision=revision)
        df = ds.to_pandas()
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    assert not missing, f"variant table missing required columns {missing}"
    return df.reset_index(drop=True)


def _inference_kwargs(
    batch_size: int,
    torch_compile: bool,
    num_workers: int,
    eval_accumulation_steps: int | None,
) -> dict[str, object]:
    kw: dict[str, object] = {
        "per_device_eval_batch_size": batch_size,
        "torch_compile": torch_compile,
        "bf16_full_eval": True,
        "dataloader_num_workers": num_workers,
        "remove_unused_columns": False,
    }
    if eval_accumulation_steps is not None:
        kw["eval_accumulation_steps"] = eval_accumulation_steps
    return kw


def _score_bundle(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    variants: pd.DataFrame,
    genome: Genome,
    window_size: int,
    inference_kwargs: dict[str, object],
    callbacks: list[TrainerCallback] | None = None,
) -> dict[str, np.ndarray]:
    """FWD+RC + embeddings bundle over ``variants`` (mirrors compute_variant_scores)."""
    hf_dataset = Dataset.from_pandas(variants, preserve_index=False)
    return run_variant_score_bundle(
        model,
        tokenizer,
        hf_dataset,
        genome,
        window_size,
        rc=True,
        return_embeddings=True,
        data_transform_on_the_fly=True,
        inference_kwargs=inference_kwargs,
        callbacks=callbacks,
    )


def _bundle_to_scores_df(
    results: dict[str, np.ndarray], hidden_size: int
) -> pd.DataFrame:
    """Assemble the scores df from per-strand bundle arrays (compute_variant_scores logic)."""
    cols: dict[str, object] = {}
    for strand, arr in results.items():
        cols[f"llr_{strand}"] = arr[:, 0]
        cols[f"jsd_{strand}"] = arr[:, 1]
    d = hidden_size
    width = next(iter(results.values())).shape[1]
    assert width == 2 + 2 * d, f"bundle width {width} != 2 + 2*hidden ({d})"
    cols["emb_ref"] = list(
        fwd_rc_average_f16([a[:, 2 : 2 + d] for a in results.values()])
    )
    cols["emb_alt"] = list(
        fwd_rc_average_f16([a[:, 2 + d : 2 + 2 * d] for a in results.values()])
    )
    return pd.DataFrame(cols)


def _validate_compile(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    variants: pd.DataFrame,
    genome: Genome,
    window_size: int,
    num_workers: int,
    subset: int,
) -> dict[str, object]:
    """Score a tiny subset eager vs compiled (embeddings on); report the deltas.

    The #318 overlay left torch.compile + return_embeddings unvalidated. A crash
    means it's broken; small deltas are the expected float-reduction noise.
    """
    sub = variants.head(min(subset, len(variants)))
    bs = min(16, len(sub))
    out: dict[str, object] = {"subset_n": len(sub), "batch_size": bs}
    try:
        eager = _score_bundle(
            model,
            tokenizer,
            sub,
            genome,
            window_size,
            _inference_kwargs(
                bs,
                torch_compile=False,
                num_workers=num_workers,
                eval_accumulation_steps=None,
            ),
        )
        compiled = _score_bundle(
            model,
            tokenizer,
            sub,
            genome,
            window_size,
            _inference_kwargs(
                bs,
                torch_compile=True,
                num_workers=num_workers,
                eval_accumulation_steps=None,
            ),
        )
        dllr = max(
            float(np.abs(eager[s][:, 0] - compiled[s][:, 0]).max()) for s in eager
        )
        djsd = max(
            float(np.abs(eager[s][:, 1] - compiled[s][:, 1]).max()) for s in eager
        )
        out.update(ok=True, max_abs_dllr=dllr, max_abs_djsd=djsd)
    except Exception as e:  # noqa: BLE001 — the whole point is to surface the failure
        out.update(ok=False, error=f"{type(e).__name__}: {e}")
    return out


def _steady_state_row(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    variants: pd.DataFrame,
    genome: Genome,
    window_size: int,
    batch_size: int,
    num_workers: int,
    eval_accumulation_steps: int | None,
    price_per_hr: float,
) -> dict[str, object]:
    """One timed FWD+RC+embeddings pass over the full table at ``batch_size``."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cb = TimingCallback()
    t0 = time.perf_counter()
    try:
        _score_bundle(
            model,
            tokenizer,
            variants,
            genome,
            window_size,
            _inference_kwargs(
                batch_size,
                torch_compile=True,
                num_workers=num_workers,
                eval_accumulation_steps=eval_accumulation_steps,
            ),
            callbacks=[cb],
        )
    except Exception as e:  # noqa: BLE001
        return {
            "batch_size": batch_size,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
        }
    wall = time.perf_counter() - t0

    times = np.asarray(cb.step_times)
    diffs = np.diff(times)
    assert diffs.size >= 3, (
        f"only {diffs.size + 1} prediction steps at batch_size={batch_size}; need more "
        f"batches (smaller batch or larger dataset) for a robust steady-state median"
    )
    # Median over both FWD+RC passes robustly drops the 1 compile batch + the
    # FWD->RC boundary gap. sec_per_strand_batch = time to score B variants on ONE
    # strand; a unique variant costs 2 of these (FWD+RC).
    sec_per_strand_batch = float(np.median(diffs))
    variants_per_hr = SECONDS_PER_HOUR * batch_size / (2.0 * sec_per_strand_batch)
    return {
        "batch_size": batch_size,
        "ok": True,
        "n_variants": int(len(variants)),
        "n_pred_steps": int(times.size),
        "sec_per_strand_batch_median": sec_per_strand_batch,
        "sec_per_strand_batch_p10": float(np.percentile(diffs, 10)),
        "sec_per_strand_batch_p90": float(np.percentile(diffs, 90)),
        "wall_s_full_pass": wall,
        "variants_per_hr": variants_per_hr,
        "usd_per_1k_variants": 1000.0 / variants_per_hr * price_per_hr,
        "peak_vram_reserved_gb": torch.cuda.max_memory_reserved() / BYTES_PER_GB,
        "peak_vram_allocated_gb": torch.cuda.max_memory_allocated() / BYTES_PER_GB,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="local HF checkpoint dir")
    p.add_argument(
        "--genome", required=True, help="local genome FASTA (bgzipped+faidx)"
    )
    p.add_argument(
        "--dataset",
        default="bolinas-dna/evals_sge",
        help="HF dataset id or local parquet of variants",
    )
    p.add_argument("--revision", default="225d3d1ea32a4af547891b13c33b5e92a5aae849")
    p.add_argument("--split", default="train")
    p.add_argument("--window-size", type=int, default=255)
    p.add_argument("--batch-sizes", default="128,256,512,1024")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--eval-accumulation-steps",
        type=int,
        default=None,
        help="offload predictions to CPU every N steps (None = on-device)",
    )
    p.add_argument("--price-per-hr", type=float, default=2.29, help="GH200 $/hr (#131)")
    p.add_argument("--validate-subset", type=int, default=64)
    p.add_argument("--out-dir", default=".")
    args = p.parse_args()

    assert torch.cuda.is_available(), "GPU required (bf16_full_eval errors on CPU)"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]

    device = torch.cuda.get_device_name(0)
    print(f"[setup] GPU={device}  torch={torch.__version__}", flush=True)

    variants = _load_variants(args.dataset, args.revision, args.split)
    print(
        f"[setup] variants: {len(variants)} rows, cols={list(variants.columns)}",
        flush=True,
    )

    genome = Genome(args.genome)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, trust_remote_code=True
    )
    model = model.cuda().eval()
    hidden_size = int(model.config.hidden_size)

    # 1. compile + embeddings validation
    print("[validate] eager vs compiled (embeddings on)...", flush=True)
    validation = _validate_compile(
        model,
        tokenizer,
        variants,
        genome,
        args.window_size,
        args.num_workers,
        args.validate_subset,
    )
    print(f"[validate] {validation}", flush=True)

    # 2. batch-size sweep
    rows: list[dict[str, object]] = []
    for bs in batch_sizes:
        print(f"[sweep] batch_size={bs} ...", flush=True)
        row = _steady_state_row(
            model,
            tokenizer,
            variants,
            genome,
            args.window_size,
            bs,
            args.num_workers,
            args.eval_accumulation_steps,
            args.price_per_hr,
        )
        print(f"[sweep] {row}", flush=True)
        rows.append(row)

    # 3. scores dump (batch-invariant; use the smallest OK batch for VRAM safety)
    best_ok = next((r["batch_size"] for r in rows if r.get("ok")), batch_sizes[0])
    print(f"[scores] dumping scores at batch_size={best_ok} ...", flush=True)
    results = _score_bundle(
        model,
        tokenizer,
        variants,
        genome,
        args.window_size,
        _inference_kwargs(
            int(best_ok),
            torch_compile=True,
            num_workers=args.num_workers,
            eval_accumulation_steps=args.eval_accumulation_steps,
        ),
    )
    scores = _bundle_to_scores_df(results, hidden_size)
    keep = [
        c
        for c in ["chrom", "pos", "ref", "alt", "label", "subset"]
        if c in variants.columns
    ]
    scores = pd.concat([variants[keep].reset_index(drop=True), scores], axis=1)
    scores_path = out_dir / "sge_scores.parquet"
    scores.to_parquet(scores_path, index=False)

    summary = {
        "gpu": device,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "revision": args.revision,
        "split": args.split,
        "window_size": args.window_size,
        "num_workers": args.num_workers,
        "eval_accumulation_steps": args.eval_accumulation_steps,
        "price_per_hr": args.price_per_hr,
        "n_variants": int(len(variants)),
        "hidden_size": hidden_size,
        "validation": validation,
        "sweep": rows,
    }
    (out_dir / "inference_cost.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[done] wrote {out_dir / 'inference_cost.json'} and {scores_path}", flush=True
    )

    print(
        "\n=== cost table (context = 256 tok; FWD+RC; embeddings on; f16 storage) ==="
    )
    print(f"{'batch':>6} {'variants/hr':>12} {'$/1k':>8} {'peakVRAM(GB)':>13}")
    for r in rows:
        if r.get("ok"):
            print(
                f"{r['batch_size']:>6} {r['variants_per_hr']:>12,.0f} "
                f"{r['usd_per_1k_variants']:>8.3f} {r['peak_vram_reserved_gb']:>13.1f}"
            )
        else:
            print(f"{r['batch_size']:>6}  FAILED: {r.get('error')}")


if __name__ == "__main__":
    main()
