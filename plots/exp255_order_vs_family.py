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

Outputs under plots/output/exp255_order_vs_family/ (PNG 130dpi + SVG):
  exp255_matched.{png,svg}            order vs family: matched AUPRC + matched LL gap
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
C_ORDER = "#0072B2"  # order (19 sp.)
C_FAMILY = "#E69F00"  # family (108 sp.)
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
    print("\n=== MATCHED: order (19 sp.) vs family (108 sp.) — same region & budget ===")
    print("    offline AUPRC = minus_llr_avg @ step-4999 (±cluster-bootstrap SE); LL = -loss\n")
    for arm, c in ARMS.items():
        om = cache.setdefault((arm, "order"), offline_auprc(c["order_model"]))
        fm = cache.setdefault((arm, "family"), offline_auprc(c["family_model"]))
        print(f"[{arm} arm]  matched AUPRC subsets:")
        print(f"    {'subset':36s}{'order':>16}{'family':>16}{'Δ(order-fam)':>14}")
        for ss in c["subsets"]:
            o, oe = om[ss]
            f, fe = fm[ss]
            print(f"    {ss:36s}{o:8.3f}±{oe:.3f}{f:8.3f}±{fe:.3f}{o - f:+14.3f}")
        lo, lf = wd[(arm, "order")], wd[(arm, "family")]
        print(f"  LL on {c['recipe']} (matched region):")
        print(f"    {'metric':36s}{'order':>16}{'family':>16}{'Δ(order-fam)':>14}")
        for label, key in (("LL functional", "func_ll"), ("LL non-functional", "nonfunc_ll"), ("LL gap", "gap")):
            print(f"    {label:36s}{lo[key]:>16.3f}{lf[key]:>16.3f}{lo[key] - lf[key]:+14.3f}")
        print()
    return cache


def print_online_offline(wd: dict[tuple[str, str], dict], cache: dict) -> None:
    print("=== SANITY: online (in-training lm_eval, post-#266 BOS) vs offline AUPRC ===")
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


def plot_matched(wd: dict, cache: dict) -> None:
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2.4, 1]})

    # Panel A — matched-subset AUPRC, grouped order vs family
    labels, ovals, oerr, fvals, ferr = [], [], [], [], []
    for arm, c in ARMS.items():
        om = cache[(arm, "order")]
        fm = cache[(arm, "family")]
        for ss in c["subsets"]:
            labels.append(f"{ss.replace('_variant', '')}\n({arm})")
            ovals.append(om[ss][0]); oerr.append(om[ss][1])
            fvals.append(fm[ss][0]); ferr.append(fm[ss][1])
    x = np.arange(len(labels))
    w = 0.38
    axA.bar(x - w / 2, ovals, w, yerr=oerr, capsize=3, color=C_ORDER, label="order (19 sp.)")
    axA.bar(x + w / 2, fvals, w, yerr=ferr, capsize=3, color=C_FAMILY, label="family (108 sp.)")
    axA.axhline(AUPRC_BASELINE, ls=":", color="gray", lw=0.9, label=f"baseline {AUPRC_BASELINE:.2f}")
    axA.set_xticks(x)
    axA.set_xticklabels(labels, fontsize=9)
    axA.set_ylabel("offline AUPRC (minus_llr_avg)")
    axA.set_title("Matched-subset Mendelian AUPRC", fontsize=11)
    axA.legend(fontsize=8, loc="upper right")
    axA.grid(axis="y", alpha=0.3)

    # Panel B — matched-region LL gap, grouped order vs family
    arms = list(ARMS)
    xg = np.arange(len(arms))
    og = [wd[(a, "order")]["gap"] for a in arms]
    fg = [wd[(a, "family")]["gap"] for a in arms]
    axB.bar(xg - w / 2, og, w, color=C_ORDER, label="order")
    axB.bar(xg + w / 2, fg, w, color=C_FAMILY, label="family")
    axB.set_xticks(xg)
    axB.set_xticklabels([f"{a}\n({ARMS[a]['recipe']})" for a in arms], fontsize=9)
    axB.set_ylabel("functional-constraint LL gap (nats)")
    axB.set_title("Matched-region LL gap", fontsize=11)
    axB.grid(axis="y", alpha=0.3)
    for i, a in enumerate(arms):
        for dx, g in ((-w / 2, og[i]), (w / 2, fg[i])):
            axB.text(xg[i] + dx, g + 0.004, f"{g:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "exp255 (#255) — order- vs family-deduplicated cohort, matched region & budget (0.25B, step-4999)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _savefig(fig, "exp255_matched")


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
            xs.append(onv); ys.append(offv)
        ax.scatter(xs, ys, marker=markers[arm], s=60, color=C_ORDER, alpha=0.8,
                   edgecolor="white", linewidth=0.6, label=f"{arm}_order")
    lo, hi = 0.08, 0.42
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("online AUPRC (in-training lm_eval, BOS)")
    ax.set_ylabel("offline AUPRC (evals_v2 minus_llr_avg)")
    ax.set_title("exp255 — online vs offline AUPRC (all 8 subsets)\npost-#266 BOS fix: points on y=x", fontsize=10)
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
    plot_matched(wd, cache)
    plot_online_vs_offline(wd, cache)
    print(f"\nDone. Figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()
