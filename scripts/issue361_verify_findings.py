"""Sharpen the blog #361 "findings to review" with actual numbers + significance.

Reads the same evals_v2 S3 metrics the figures use (with bootstrap SE) and Eric's
vendored old-wandb CSVs, and prints verified statements for each flagged finding.
Run: uv run python scripts/issue361_verify_findings.py

Findings covered here:
  1. Fig 11 leaderboard gaps + significance (M·LLR: m5.1 vs Evo2-40B; S·Probe: m5.1 vs exp13-mp/4B).
  2. Old-wandb vs new-offline scaling (Fig 5/6): does the params→AUPRC trend hold?
Gap "significance" uses the CONSERVATIVE independent-SE combination
(sqrt(se_a^2+se_b^2)); a paired bootstrap (same variants) would be tighter.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `plots`

import pandas as pd
import polars as pl

from marin_dna.pipelines.evals.leaderboard import (
    DEFAULT_PROTOCOL,
    normalized_rows,
    sge_probe_normalized_rows,
)
from marin_dna.pipelines.evals.metrics import MACRO_AVG_SUBSET
from plots.blog._scaling import DATA, LADDER_FINAL_STEP
from plots.blog._worlds import WORLDS


def _macro(rows: pl.DataFrame, display: str) -> tuple[float, float] | None:
    r = rows.filter(
        (pl.col("method_display") == display) & (pl.col("subset") == MACRO_AVG_SUBSET)
    )
    if r.height == 0:
        return None
    return float(r["value"][0]), float(r["se"][0])


def _gap(a: tuple[float, float], b: tuple[float, float]) -> str:
    (va, sa), (vb, sb) = a, b
    gap = va - vb
    comb = math.sqrt(sa**2 + sb**2)
    z = gap / comb if comb else float("inf")
    verdict = (
        "SIGNIFICANT" if abs(z) >= 2 else ("marginal" if abs(z) >= 1 else "NOT sig")
    )
    return (
        f"{va * 100:.1f} vs {vb * 100:.1f}  (Δ={gap * 100:+.1f}, "
        f"±{comb * 100:.1f} combined SE, z={z:+.1f} → {verdict})"
    )


def finding_1_leaderboard() -> None:
    print("\n=== Finding 1 — Fig 11 leaderboard gaps + significance ===")
    # M·LLR: each family at its default protocol (mirrors figure11.build_mendelian_llr).
    rows = normalized_rows("mendelian_traits").with_columns(
        pl.col("family").replace_strict(DEFAULT_PROTOCOL, default=None).alias("_d")
    )
    mllr = rows.filter(pl.col("protocol") == pl.col("_d"))
    m5 = _macro(mllr, "exp135-1B-m5.1")
    for other in ("Evo 2 (40B)", "Evo 2 (7B)", "GPN-Star (M)", "AlphaGenome"):
        o = _macro(mllr, other)
        if m5 and o:
            print(f"  M·LLR  exp135-1B-m5.1 vs {other:14s}: {_gap(m5, o)}")

    # S·Probe: the SGE frozen-embedding-probe leaderboard (marin_dna only).
    sp = sge_probe_normalized_rows("sge").filter(pl.col("subset") == MACRO_AVG_SUBSET)
    vals = {
        r["method_display"]: (float(r["value"]), float(r["se"]))
        for r in sp.iter_rows(named=True)
    }
    print("  S·Probe macro AUPRC ± SE:")
    for disp, (v, s) in sorted(vals.items(), key=lambda kv: -kv[1][0]):
        print(f"    {disp:34s} {v * 100:5.1f} ± {s * 100:.1f}")
    # Is m5.1 significantly below the SGE-probe leaders?
    m5s = next((v for d, v in vals.items() if "m5.1" in d), None)
    for d, v in vals.items():
        if "m5.1" not in d and m5s:
            print(f"  S·Probe  {d} vs m5.1: {_gap(v, m5s)}")


def finding_2_old_vs_new() -> None:
    print("\n=== Finding 2 — old-wandb vs new-offline scaling (8 ladder endpoints) ===")
    old = pd.read_csv(DATA).set_index("run_name")
    world = WORLDS["mendelian_llr"]
    print(
        f"  {'size':>6}  {'params':>6}  {'old wandb':>9}  {'new global':>10}  {'new macro':>9}"
    )
    params, old_vals, newg_vals, newm_vals = [], [], [], []
    for run_name, r in old.iterrows():
        stem = run_name.removeprefix("dna-bolinas-")
        size = stem.split("-")[2] if len(stem.split("-")) > 2 else stem
        old_auprc = float(
            r["lm_eval/traitgym_mendelian_v2_255/auprc"]
        )  # pooled/global, old eval
        try:
            df = world.read(f"{stem}-step-{LADDER_FINAL_STEP}").to_pandas()
        except Exception:
            continue
        g = df[df["subset"] == "_global_"]["value"]
        mac = df[df["subset"] == MACRO_AVG_SUBSET]["value"]
        newg = float(g.iloc[0]) if len(g) else float("nan")
        newm = float(mac.iloc[0]) if len(mac) else float("nan")
        print(
            f"  {size:>6}  {int(r['params']) / 1e6:5.0f}M  {old_auprc * 100:8.1f}  "
            f"{newg * 100:9.1f}  {newm * 100:8.1f}"
        )
        params.append(int(r["params"]))
        old_vals.append(old_auprc)
        newg_vals.append(newg)
        newm_vals.append(newm)

    def rho(a, b):
        return pd.Series(a).corr(pd.Series(b), method="spearman")

    print("\n  Spearman rank correlations across the 8 sizes:")
    print(
        f"    ρ(params, new-global) = {rho(params, newg_vals):+.3f}   "
        f"ρ(params, new-macro) = {rho(params, newm_vals):+.3f}   "
        "(does the NEW eval scale with size?)"
    )
    print(
        f"    ρ(params, old-wandb)  = {rho(params, old_vals):+.3f}   "
        "(did the OLD eval scale with size?)"
    )
    print(
        f"    ρ(old-wandb, new-global) = {rho(old_vals, newg_vals):+.3f}   "
        f"ρ(old-wandb, new-macro) = {rho(old_vals, newm_vals):+.3f}   "
        "(does the old→new per-size ranking transfer?)"
    )


def finding_3_upstream_optimum() -> None:
    print("\n=== Finding 3 — Fig 9 upstream-mix optimum + significance (M·LLR) ===")
    from plots.blog import _mixture as mx
    from plots.blog import _mixture_lineage as ml

    world = WORLDS["mendelian_llr"]

    def macro(mix: str) -> tuple[float, float] | None:
        df = world.read(mx.final_name(mix)).to_pandas()
        r = df[df["subset"] == MACRO_AVG_SUBSET]["value"]
        s = df[df["subset"] == MACRO_AVG_SUBSET]["se"]
        return (float(r.iloc[0]), float(s.iloc[0])) if len(r) else None

    sweep = [
        "uniform_to_upstream_1",
        "uniform_to_upstream_2",
        "uniform_to_upstream_3",
        "uniform_to_upstream_3.5",
        "uniform_to_upstream_3.6",
        "uniform_to_upstream_4",
        "uniform_to_upstream_5",
    ]
    base = macro("uniform")
    pts = sorted(
        (ml.BY_MIX[m].weights.get("upstream", 0.0), m, macro(m)) for m in sweep
    )
    print(f"  uniform baseline: {base[0] * 100:.1f} ± {base[1] * 100:.1f}")
    for up, m, v in pts:
        flag = "  ← peak" if v[0] == max(p[2][0] for p in pts) else ""
        print(f"    upstream={up:.2f}  {v[0] * 100:5.1f} ± {v[1] * 100:.1f}{flag}")
    peak = max(pts, key=lambda p: p[2][0])
    worst = min(pts, key=lambda p: p[2][0])
    print(f"  peak (upstream={peak[0]:.2f}) vs uniform baseline: {_gap(peak[2], base)}")
    print(
        f"  peak (upstream={peak[0]:.2f}) vs worst (upstream={worst[0]:.2f}): {_gap(peak[2], worst[2])}"
    )


def finding_4_probe_payoff() -> None:
    print(
        "\n=== Finding 4 — Fig 7 128M missense: zero-shot LLR vs frozen-embedding probe ==="
    )
    steps = [160000, 170000, 180000, 190000, 200000, 210000, 215573]
    llr, probe = WORLDS["mendelian_llr"], WORLDS["mendelian_probe"]

    def miss(world, step):
        df = world.read(f"scaling-v0.5-h896-p128M-step-{step}").to_pandas()
        r = df[df["subset"] == "missense_variant"]["value"]
        return float(r.iloc[0]) if len(r) and pd.notna(r.iloc[0]) else float("nan")

    print(f"  {'step':>8}  {'LLR miss':>8}  {'probe miss':>10}")
    llr_v, probe_v = [], []
    for s in steps:
        lv, pv = miss(llr, s), miss(probe, s)
        print(f"  {s:>8}  {lv * 100:7.1f}  {pv * 100:9.1f}")
        llr_v.append(lv)
        probe_v.append(pv)
    print(
        f"  LLR   missense 160k→215k: {llr_v[0] * 100:.1f} → {llr_v[-1] * 100:.1f}  "
        f"(Δ={(llr_v[-1] - llr_v[0]) * 100:+.1f})"
    )
    print(
        f"  probe missense 160k→215k: {probe_v[0] * 100:.1f} → {probe_v[-1] * 100:.1f}  "
        f"(Δ={(probe_v[-1] - probe_v[0]) * 100:+.1f})"
    )
    print(
        "  → payoff = probe keeps rising where LLR does not"
        if (probe_v[-1] - probe_v[0]) > (llr_v[-1] - llr_v[0])
        else "  → no clear divergence"
    )


def finding_5_loss_auprc_corr() -> None:
    print(
        "\n=== Finding 5 — Fig 8 loss↔AUPRC Spearman ρ (shared-step intersection) ==="
    )
    from plots.blog import figure8_loss_vs_traitgym_correlation as f8

    for key in ("mendelian_llr", "mendelian_probe", "sge_llr", "sge_probe"):
        corr, n = f8.correlations(WORLDS[key])
        if not corr:
            print(f"  {key:16s}: no data")
            continue
        overall = sum(corr.values()) / len(corr)
        per = "  ".join(f"{k}={v:+.2f}" for k, v in corr.items())
        print(f"  {key:16s} (∩={n} steps)  mean ρ={overall:+.2f}   [{per}]")
    print(
        "  ρ→+1 = loss-drop tracks AUPRC-gain within a scale; near 0 / negative = loss is NOT"
    )
    print("  a reliable within-run proxy for VEP AUPRC on the new eval.")


if __name__ == "__main__":
    finding_1_leaderboard()
    finding_2_old_vs_new()
    finding_3_upstream_optimum()
    finding_4_probe_payoff()
    finding_5_loss_auprc_corr()
