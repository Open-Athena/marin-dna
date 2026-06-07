"""Plots for exp255 — order- vs family-deduplicated per-region 0.25B (issue #255).

exp255 trains two 0.25B arms on ORDER-deduplicated zoonomia (19-species,
one-genome-per-order cohort) for the two largest v4 slices — cds and
ccre_non_promoter — using the IDENTICAL region/hparams/compute as exp232's
FAMILY-cohort (108-species) arms. The question (NOT the diagonal): holding the
training region and budget fixed, does collapsing the cohort to one genome per
order help or hurt the specialist?

Comparison is **matched only** — each order arm vs its family twin, on that arm's
own region (per the exp232 region->subset map in plots/exp232_per_region.py):

    cds  arm:  AUPRC {missense, synonymous, splicing};  LL on val_cds
    ccre arm:  AUPRC {distal};                          LL on val_enhancer

Metrics:
  - offline AUPRC: evals_v2 ``minus_llr_avg`` (BOS-faithful FWD/RC-averaged -LLR),
    final step-4999, cluster-bootstrap SE.
  - LL (= -loss), region-matched, from each run's final-step wandb val losses.
    LL gap = LL_func - LL_nonfunc = nonfunc_loss - func_loss  (>0 = the model puts
    higher likelihood on functional/constrained positions). Convention per #8.

Plus a #266 BOS-fix sanity check: online (in-training lm_eval, BOS-faithful) vs
offline AUPRC for the two ORDER arms across ALL 8 subsets — they should coincide.

Plot conventions: family (108 sp.) is the BASELINE — plotted first / muted color;
order (19 sp.) is the new arm — plotted second / highlighted. Each matched subset
(AUPRC) and each val dataset (LL gap) is a DISTINCT comparison, so it gets its own
subpanel + y-axis — never share a y-axis across different subsets/datasets. AUPRC
error bars are ±1 SE (cluster bootstrap), drawn without caps; the LL gap is a
single-run point estimate (no SE).

Outputs under plots/output/exp255_order_vs_family/ (PNG 130dpi + SVG):
  exp255_matched_auprc.{png,svg}      per matched subset: family vs order AUPRC (±1 SE)
  exp255_matched_llgap.{png,svg}      per matched val set: family vs order LL gap
  exp255_matched_ll_components.{png,svg}  per (component x val set): LL func & non-func
  exp255_online_vs_offline.{png,svg}  order arms, all-subset online vs offline AUPRC
Prints: matched comparison table (AUPRC + LL func/nonfunc/gap) and the
online-vs-offline AUPRC table.

Run:  uv run python plots/exp255_order_vs_family.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import wandb

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

S3_PREFIX = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics"
SCORE_TYPE = "minus_llr_avg"
AUPRC_BASELINE = 0.10
GROUP_ORDER = "dna-exp255-v0.1"
GROUP_FAMILY = "dna-exp232-v0.1"

# arm -> the order/family checkpoint names (evals_v2 registry), the matched wandb
# run substrings, the matched val recipe, and the matched AUPRC subsets.
ARMS: dict[str, dict] = {
    "cds": dict(
        order_model="exp255-v4_cds_order-step-4999",
        family_model="exp232-v4_cds-step-4999",
        order_sub="v4_cds_order",
        family_sub="v4_cds",
        recipe="val_cds",
        subsets=["missense_variant", "synonymous_variant", "splicing"],
    ),
    "ccre": dict(
        order_model="exp255-v4_ccre_non_promoter_order-step-4999",
        family_model="exp232-v4_ccre_non_promoter-step-4999",
        order_sub="v4_ccre_non_promoter_order",
        family_sub="v4_ccre_non_promoter",
        recipe="val_enhancer",
        subsets=["distal"],
    ),
}
ALL_SUBSETS: list[str] = [
    "missense_variant",
    "synonymous_variant",
    "splicing",
    "3_prime_UTR_variant",
    "non_coding_transcript_exon_variant",
    "5_prime_UTR_variant",
    "tss_proximal",
    "distal",
]
C_FAMILY = "#9e9e9e"  # family (108 sp.) — baseline, plotted first (muted)
C_ORDER = "#0072B2"  # order (19 sp.) — the new arm, plotted second (highlight)
OUT_DIR: Path = Path(__file__).parent / "output" / Path(__file__).stem


def _savefig(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", dict(dpi=130)), ("svg", {})):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  wrote {stem}.png + {stem}.svg")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def offline_auprc(model: str) -> dict[str, tuple[float, float]]:
    """{subset: (auprc, se)} for SCORE_TYPE at the model's step-4999 parquet."""
    df = pl.read_parquet(f"{S3_PREFIX}/{model}/mendelian_traits.parquet").filter(
        pl.col("score_type") == SCORE_TYPE
    )
    return {r["subset"]: (r["value"], r["se"]) for r in df.to_dicts()}


def _pick_run(api: wandb.Api, group: str, sub: str):
    """Highest-step run in `group` whose name contains `sub` (the finished arm)."""
    rs = [r for r in api.runs("marin", filters={"group": group}) if sub in r.name]
    if not rs:
        raise RuntimeError(f"no wandb run in {group} matching {sub!r}")
    return max(rs, key=lambda x: dict(x.summary).get("_step") or 0)


def load_wandb() -> dict[tuple[str, str], dict]:
    """(arm, cohort) -> {run, step, func_ll, nonfunc_ll, gap, online_auprc{subset}}.

    LL = -loss on the arm's matched val recipe; online_auprc is the in-training
    lm_eval AUPRC (only meaningful/needed for the order arms, but pulled for both).
    """
    api = wandb.Api()
    out: dict[tuple[str, str], dict] = {}
    for arm, c in ARMS.items():
        for coh, sub, group in (
            ("order", c["order_sub"], GROUP_ORDER),
            ("family", c["family_sub"], GROUP_FAMILY),
        ):
            r = _pick_run(api, group, sub)
            s = dict(r.summary)
            rec = c["recipe"]
            fl = s[f"eval/{rec}_functional/loss"]
            nl = s[f"eval/{rec}_nonfunctional/loss"]
            out[(arm, coh)] = dict(
                run=r.name,
                step=int(s.get("_step") or 0),
                func_ll=-fl,
                nonfunc_ll=-nl,
                gap=nl - fl,
                online_auprc={
                    ss: s.get(f"lm_eval/mendelian_traits_255/{ss}/avg/auprc")
                    for ss in ALL_SUBSETS
                },
            )
            print(f"  {arm:5s}/{coh:6s} <- {r.name} (step {out[(arm, coh)]['step']})")
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def print_tables(wd: dict[tuple[str, str], dict]) -> dict:
    cache = {}  # (arm,cohort)->offline_auprc dict, reused by the plots
    print(
        "\n=== MATCHED: order (19 sp.) vs family (108 sp.) — same region & budget ==="
    )
    print(
        "    offline AUPRC = minus_llr_avg @ step-4999 (±cluster-bootstrap SE); LL = -loss\n"
    )
    for arm, c in ARMS.items():
        om = cache.setdefault((arm, "order"), offline_auprc(c["order_model"]))
        fm = cache.setdefault((arm, "family"), offline_auprc(c["family_model"]))
        print(f"[{arm} arm]  matched AUPRC subsets (family = baseline):")
        print(f"    {'subset':36s}{'family':>16}{'order':>16}{'Δ(order-fam)':>14}")
        for ss in c["subsets"]:
            o, oe = om[ss]
            f, fe = fm[ss]
            print(f"    {ss:36s}{f:8.3f}±{fe:.3f}{o:8.3f}±{oe:.3f}{o - f:+14.3f}")
        lo, lf = wd[(arm, "order")], wd[(arm, "family")]
        print(f"  LL on {c['recipe']} (matched region):")
        print(f"    {'metric':36s}{'family':>16}{'order':>16}{'Δ(order-fam)':>14}")
        for label, key in (
            ("LL functional", "func_ll"),
            ("LL non-functional", "nonfunc_ll"),
            ("LL gap", "gap"),
        ):
            print(
                f"    {label:36s}{lf[key]:>16.3f}{lo[key]:>16.3f}{lo[key] - lf[key]:+14.3f}"
            )
        print()
    return cache


def print_online_offline(wd: dict[tuple[str, str], dict], cache: dict) -> None:
    print(
        "=== SANITY: online (in-training lm_eval, post-#266 BOS) vs offline AUPRC ==="
    )
    print("    order arms, all 8 subsets — they should coincide if the BOS fix holds\n")
    for arm, c in ARMS.items():
        off = cache.setdefault((arm, "order"), offline_auprc(c["order_model"]))
        on = wd[(arm, "order")]["online_auprc"]
        print(f"[{arm}_order]  {'subset':36s}{'online':>9}{'offline':>9}{'Δ':>9}")
        deltas = []
        for ss in ALL_SUBSETS:
            onv = on.get(ss)
            offv = off.get(ss, (None,))[0]
            if onv is None or offv is None:
                continue
            deltas.append(offv - onv)
            print(f"    {'':0s}{ss:36s}{onv:9.3f}{offv:9.3f}{offv - onv:+9.3f}")
        if deltas:
            print(f"    {'mean |Δ|':36s}{'':9}{'':9}{np.mean(np.abs(deltas)):9.3f}\n")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_matched_auprc(cache: dict) -> None:
    """One subpanel per matched subset (its own y-axis — distinct comparisons are
    never put on a shared axis); family (baseline) first, order second. Error bars
    = ±1 SE (cluster bootstrap), drawn without caps."""
    items = [(ss, arm) for arm, c in ARMS.items() for ss in c["subsets"]]
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(2.7 * n, 3.7), squeeze=False)
    for k, (ss, arm) in enumerate(items):
        ax = axes[0][k]
        fv, fe = cache[(arm, "family")][ss]
        ov, oe = cache[(arm, "order")][ss]
        ax.bar(
            [0, 1],
            [fv, ov],
            0.62,
            yerr=[fe, oe],
            color=[C_FAMILY, C_ORDER],
            error_kw=dict(capsize=0, elinewidth=1.4, ecolor="#333"),
        )
        ax.axhline(AUPRC_BASELINE, ls=":", color="gray", lw=0.9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["family\n108 sp.", "order\n19 sp."], fontsize=8.5)
        ax.set_title(f"{ss.replace('_variant', '')}  ({arm})", fontsize=9.5)
        ax.set_ylim(0, max(fv + fe, ov + oe) + 0.05)
        if k == 0:
            ax.set_ylabel("offline AUPRC (minus_llr_avg)")
        ax.grid(axis="y", alpha=0.3)
        ax.text(
            0.5,
            0.97,
            f"Δ={ov - fv:+.3f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85, pad=1.5),
        )
    fig.suptitle(
        "exp255 (#255) — matched-subset Mendelian AUPRC: family (baseline) vs order\n"
        "0.25B, step-4999; error bars = ±1 SE (cluster bootstrap, not CI); dotted = 0.10 prevalence baseline",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _savefig(fig, "exp255_matched_auprc")


def plot_matched_llgap(wd: dict) -> None:
    """One subpanel per matched val dataset (its own y-axis — val_cds and
    val_enhancer are different data, never a shared axis); family (baseline) first,
    order second. The LL gap is a single-run point estimate (no SE)."""
    arms = list(ARMS)
    fig, axes = plt.subplots(
        1, len(arms), figsize=(3.3 * len(arms), 3.9), squeeze=False
    )
    for k, arm in enumerate(arms):
        ax = axes[0][k]
        fg = wd[(arm, "family")]["gap"]
        og = wd[(arm, "order")]["gap"]
        ax.bar([0, 1], [fg, og], 0.62, color=[C_FAMILY, C_ORDER])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["family\n108 sp.", "order\n19 sp."], fontsize=8.5)
        ax.set_title(f"{arm}  ({ARMS[arm]['recipe']})", fontsize=9.5)
        top = max(fg, og)
        ax.set_ylim(0, top * 1.2)
        if k == 0:
            ax.set_ylabel("functional-constraint LL gap (nats)")
        ax.grid(axis="y", alpha=0.3)
        for xb, g in ((0, fg), (1, og)):
            ax.text(
                xb, g + top * 0.012, f"{g:.3f}", ha="center", va="bottom", fontsize=9
            )
        ax.text(
            0.5,
            0.97,
            f"Δ={og - fg:+.3f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85, pad=1.5),
        )
    fig.suptitle(
        "exp255 (#255) — matched-region functional-constraint LL gap: family (baseline) vs order\n"
        "LL = -loss; gap = LL_func - LL_nonfunc; single-run point estimate (no SE)",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _savefig(fig, "exp255_matched_llgap")


def plot_matched_ll_components(wd: dict) -> None:
    """LL functional and LL non-functional, family (baseline) vs order. One panel
    per (component x val dataset), each with its own zoomed y-axis: LL has no
    meaningful zero and the family<->order deltas are sub-0.02 nats, so bars-from-0
    would hide them. Dumbbell = family (gray) and order (blue) dots; no SE."""
    arms = list(ARMS)
    comps = [("func_ll", "LL functional"), ("nonfunc_ll", "LL non-functional")]
    fig, axes = plt.subplots(
        len(comps), len(arms), figsize=(3.3 * len(arms), 6.2), squeeze=False
    )
    for ri, (key, label) in enumerate(comps):
        for ci, arm in enumerate(arms):
            ax = axes[ri][ci]
            fv = wd[(arm, "family")][key]
            ov = wd[(arm, "order")][key]
            ax.plot([0, 1], [fv, ov], color="#cccccc", lw=1.2, zorder=1)
            ax.scatter(
                [0, 1],
                [fv, ov],
                s=120,
                color=[C_FAMILY, C_ORDER],
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )
            ax.set_xlim(-0.6, 1.6)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["family\n108 sp.", "order\n19 sp."], fontsize=8.5)
            lo, hi = min(fv, ov), max(fv, ov)
            pad = max((hi - lo) * 0.9, 0.012)
            ax.set_ylim(lo - pad, hi + pad)
            if ci == 0:
                ax.set_ylabel(f"{label}\n(nats, = -loss; higher better)", fontsize=9)
            if ri == 0:
                ax.set_title(f"{arm}  ({ARMS[arm]['recipe']})", fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            for x, v in ((0, fv), (1, ov)):
                ax.annotate(
                    f"{v:.3f}",
                    (x, v),
                    textcoords="offset points",
                    xytext=(0, 9),
                    ha="center",
                    fontsize=8.5,
                )
            ax.text(
                0.5,
                0.05,
                f"Δ={ov - fv:+.3f}",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85, pad=1.5),
            )
    fig.suptitle(
        "exp255 (#255) — matched-region LL functional & non-functional: family (baseline) vs order\n"
        "LL = -loss (higher = better); per-panel y-axis zoomed (LL has no meaningful zero); single-run (no SE)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _savefig(fig, "exp255_matched_ll_components")


def plot_online_vs_offline(wd: dict, cache: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    markers = {"cds": "o", "ccre": "s"}
    for arm, c in ARMS.items():
        off = cache[(arm, "order")]
        on = wd[(arm, "order")]["online_auprc"]
        xs, ys = [], []
        for ss in ALL_SUBSETS:
            onv, offv = on.get(ss), off.get(ss, (None,))[0]
            if onv is None or offv is None:
                continue
            xs.append(onv)
            ys.append(offv)
        ax.scatter(
            xs,
            ys,
            marker=markers[arm],
            s=60,
            color=C_ORDER,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.6,
            label=f"{arm}_order",
        )
    lo, hi = 0.08, 0.42
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("online AUPRC (in-training lm_eval, BOS)")
    ax.set_ylabel("offline AUPRC (evals_v2 minus_llr_avg)")
    ax.set_title(
        "exp255 — online vs offline AUPRC (all 8 subsets)\npost-#266 BOS fix: points on y=x",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "exp255_online_vs_offline")


def main() -> None:
    print("Loading wandb final-step LL + online AUPRC ...")
    wd = load_wandb()
    print("Tables ...")
    cache = print_tables(wd)
    print_online_offline(wd, cache)
    print("Plotting ...")
    plot_matched_auprc(cache)
    plot_matched_llgap(wd)
    plot_matched_ll_components(wd)
    plot_online_vs_offline(wd, cache)
    print(f"\nDone. Figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()
