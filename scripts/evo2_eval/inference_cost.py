"""VEP inference-cost table for the score-bundle path (issue #131).

Reports the *steady-state* per-variant cost of variant scoring on a stated
(model, hardware, config), across the axes that actually matter — no single
number hides a knob (cf. the compute-cost-reporting rule):

  - throughput:  variants/hour  AND  accelerator-seconds/variant (composable)
  - feasibility: peak VRAM      -> the minimum viable accelerator
  - economic:    $ / 1000 variants (at a stated $/accel-hour)
  - intrinsic:   TFLOPs/variant (hardware-free) + implied achieved TFLOP/s

Measurement discipline: `s_per_batch` is the *converged* tqdm rate read from a
late-run batch (drop model-load, the first ~10 warmup/compile batches, and
tokenization). The one-time load is excluded — it amortizes over run size; this
is the marginal per-variant cost.

To add our own gLMs later: append a RUN row (params, ctx, batch, s_per_batch,
vram, accel, $/hr, strands, embeddings). The FLOP model is the matmul lower
bound `2 * N_params * tokens`; see `flops_note` for the arch-dependent
correction (transformer attention is quadratic in ctx; Evo2/Hyena is
sub-quadratic — so at 8192 ctx this estimate is tighter for Evo2 than for a
long-context transformer).

    uv run python scripts/evo2_eval/inference_cost.py
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Run:
    name: str
    params_b: float  # nominal parameter count, billions
    ctx: int  # context length in tokens (bp; Evo2 char-level = 1 token/bp)
    batch: int  # per-batch variant count
    s_per_batch: float  # converged steady-state seconds per batch
    vram_gb: float  # observed peak VRAM at this (batch, embeddings)
    accel: str
    usd_per_hr: float
    strands: int = 2  # FWD+RC = 2 full passes; 1 for single-strand
    embeddings: bool = True


def cost(r: Run) -> dict[str, float]:
    # Each strand is a full pass over the variants; per batch of `batch` variants
    # the model spends `s_per_batch`, so per-variant wall time is
    # strands * s_per_batch / batch.
    sec_per_variant = r.strands * r.s_per_batch / r.batch
    variants_per_hr = 3600.0 / sec_per_variant
    usd_per_1k = 1000.0 * sec_per_variant * r.usd_per_hr / 3600.0
    tokens = r.ctx * r.strands  # tokens processed per variant (both strands)
    tflops_per_variant = 2.0 * (r.params_b * 1e9) * tokens / 1e12  # 2*N*T matmul FLOPs
    achieved_tflops = tflops_per_variant / sec_per_variant  # implied TFLOP/s
    return {
        "sec_per_variant": sec_per_variant,
        "variants_per_hr": variants_per_hr,
        "usd_per_1k": usd_per_1k,
        "tflops_per_variant": tflops_per_variant,
        "achieved_tflops": achieved_tflops,
    }


# Measured Evo2 runs — mendelian score bundle, 8192-bp context, FWD+RC,
# float32 embeddings ON, single GH200 (Lambda on-demand $2.29/hr). s_per_batch
# is the converged tqdm rate from each run log.
RUNS: list[Run] = [
    Run("evo2_1b_base", 1.0, 8192, 16, 2.46, 36.0, "GH200-96GB", 2.29),
    Run("evo2_7b", 7.0, 8192, 8, 4.64, 51.0, "GH200-96GB", 2.29),
    Run("evo2_40b", 40.0, 8192, 1, 2.87, 93.0, "GH200-96GB", 2.29),
]


def main() -> None:
    hdr = (
        f"{'model':14s} {'accel':11s} {'bs':>3s} {'s/batch':>8s} {'VRAM':>6s} "
        f"{'var/hr':>9s} {'s/variant':>10s} {'$/1k':>7s} {'TFLOP/var':>10s} "
        f"{'~TFLOP/s':>9s}"
    )
    print("Evo2 VEP scoring cost — 8192 ctx, FWD+RC, f32 embeddings ON")
    print(hdr)
    print("-" * len(hdr))
    for r in RUNS:
        c = cost(r)
        print(
            f"{r.name:14s} {r.accel:11s} {r.batch:>3d} {r.s_per_batch:>8.2f} "
            f"{r.vram_gb:>5.0f}G {c['variants_per_hr']:>9,.0f} "
            f"{c['sec_per_variant']:>10.3f} {c['usd_per_1k']:>6.2f}$ "
            f"{c['tflops_per_variant']:>10.1f} {c['achieved_tflops']:>9.0f}"
        )
    print(
        "\nNotes: TFLOP/var = 2*N*T (matmul lower bound; T = 8192*2 tokens). "
        "Attention/Hyena term extra — sub-quadratic for Evo2, so this is a tight\n"
        "estimate here; a long-context transformer would add a larger ctx^2 term. "
        "~TFLOP/s vs GH200 peak (~990 bf16 / ~1979 fp8) gives the MFU. "
        "Embeddings-off + single-strand are cheaper; re-measure per config."
    )


if __name__ == "__main__":
    main()
