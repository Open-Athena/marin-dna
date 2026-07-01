"""#341: audit the probe C-grid selection across the scaling ladder.

Loads each rung's `compute_probe` joblib (`{subset: {..., c_summary}}`) from S3 and
reports the per-subset chosen `C` plus the built-in grid-edge diagnostics
(`summarize_selected_c`): whether any fold pinned the `logspace(-12, 4, 17)` grid
edge, and — the one that matters — whether `truncation_risk` fired (a *pinned* edge
still improving past tol, i.e. the optimum may lie outside the grid → widen it).

A high-edge pin with a non-positive `high_edge_gain` is benign (minimal-regularization
saturation); a low-edge pin with non-positive `low_edge_gain` is benign (heavy-reg
floor). Only `truncation_risk=True` calls for widening the grid.

Usage:
    uv run --group genome-s3 python scripts/issue341_check_c_grid.py
"""

from __future__ import annotations

import io

import fsspec
import joblib

PREFIX = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/probe"
SIZES = [
    "h640-p46M",
    "h768-p76M",
    "h896-p128M",
    "h1152-p255M",
    "h1408-p476M",
    "h1920-p1B",
    "h2432-p2B",
    "h2944-p4B",
]


def _model(size: str) -> str:
    return f"scaling-v0.5-{size}-step-215573"


def load_joblib(path: str):
    with fsspec.open(path, "rb") as f:
        return joblib.load(io.BytesIO(f.read()))


def main() -> None:
    any_risk = False
    any_unverifiable = False  # an edge pin whose truncation_risk could not be computed
    edge_pins: list[str] = []
    for size in SIZES:
        path = f"{PREFIX}/{_model(size)}/mendelian_traits.joblib"
        try:
            probes = load_joblib(path)
        except FileNotFoundError:
            # Genuinely absent (cell not computed yet) — skip. A corrupt/partial
            # joblib or a deserialization error is a real problem, so let it raise
            # rather than masquerade as MISSING.
            print(f"[{size}] MISSING (not on S3)")
            continue
        print(f"\n=== {size} ({len(probes)} subset probes) ===")
        print(
            f"{'subset':<34} {'full_c':>8} {'c_min':>7} {'c_max':>7} "
            f"{'edge':>5} {'lowGain':>8} {'highGain':>8} {'risk':>5}"
        )
        for subset, rec in probes.items():
            cs = rec["c_summary"]
            at_edge = cs.get("at_edge")
            risk = cs.get("truncation_risk")
            lg = cs.get("low_edge_gain")
            hg = cs.get("high_edge_gain")
            print(
                f"{subset:<34} {cs['full_c']:>8.0e} {cs['c_min']:>7.0e} "
                f"{cs['c_max']:>7.0e} {str(bool(at_edge)):>5} "
                f"{('' if lg is None else f'{lg:+.3f}'):>8} "
                f"{('' if hg is None else f'{hg:+.3f}'):>8} {str(bool(risk)):>5}"
            )
            if at_edge:
                edge_pins.append(f"{size}/{subset} (risk={risk})")
                if risk is None:  # pinned but saturation could not be verified
                    any_unverifiable = True
            if risk:
                any_risk = True

    print("\n" + "=" * 60)
    print(f"edge-pinned (subset, model) cells: {len(edge_pins)}")
    for e in edge_pins:
        print(f"  - {e}")
    if any_risk:
        print("\n⚠️  TRUNCATION RISK present — widen c_grid and re-run the probe.")
    elif any_unverifiable:
        print(
            "\n⚠️  Some edge pins have truncation_risk=None (saturation not verified) "
            "— inspect those cells manually before trusting the grid width."
        )
    else:
        print(
            "\n✅ No truncation_risk: every edge pin is saturated/flat (benign). "
            "The logspace(-12, 4, 17) grid is wide enough."
        )


if __name__ == "__main__":
    main()
