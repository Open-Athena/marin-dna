"""Per-step matched AUPRC trajectory, order vs family — exp255 (issue #255).

The bar plot (`exp255_5region_matched_auprc.py`) is a single step-4999 endpoint per
arm; this shows the whole training trajectory, so the ncRNA/5'UTR family edge can be
read as stable-vs-fluke — the same endpoint->trajectory upgrade the LL plot made.

**Metric provenance (important).** The family arms (exp232) were trained BEFORE the
#266 BOS fix, so their in-training `lm_eval` AUPRC is the bugged no-BOS/OOD version
and is NOT used. Instead:
  - family  -> OFFLINE evals_v2 `minus_llr_avg` at every checkpoint (~9 steps), faithful,
               with cluster-bootstrap SE (error bars).
  - order   -> ONLINE lm_eval AUPRC (faithful, post-#266) as the per-step line.
The order online<->offline agreement (which licenses the online proxy) is verified
separately and reported as a note in the #255 issue text — it is deliberately NOT drawn
here as a graphical element. One panel per matched subset (own y-axis). Run:
    uv run python plots/exp255_auprc_trajectory.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

import wandb
from _exp255_style import COL_FAMILY as C_FAMILY
from _exp255_style import COL_ORDER as C_ORDER
from _exp255_style import set_style

S3 = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics"
SCORE = "minus_llr_avg"
ONLINE_KEY = "lm_eval/mendelian_traits_255/{subset}/avg/auprc"
GROUP_ORDER = "dna-exp255-v0.1"
BASELINE = 0.10
CAND_STEPS = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 4999, 5000]
OUT = Path(__file__).parent / "output" / Path(__file__).stem

# region -> (order wandb sub, offline region token, matched subsets)
REGIONS = {
    "cds": (
        "v4_cds_order",
        "cds",
        ["missense_variant", "synonymous_variant", "splicing"],
    ),
    "utr3": ("v4_utr3_order", "utr3", ["3_prime_UTR_variant"]),
    "ncrna": (
        "v4_ncrna_exon_order",
        "ncrna_exon",
        ["non_coding_transcript_exon_variant"],
    ),
    "tss": (
        "v4_tss_region_and_utr5_order",
        "tss_region_and_utr5",
        ["5_prime_UTR_variant", "tss_proximal"],
    ),
    "ccre": ("v4_ccre_non_promoter_order", "ccre_non_promoter", ["distal"]),
}
PANELS = [(s, r) for r, (_, _, subs) in REGIONS.items() for s in subs]

_cache: dict[str, dict[str, tuple[float, float]] | None] = {}


def read_metrics(model: str) -> dict[str, tuple[float, float]] | None:
    """{subset: (auprc, se)} for minus_llr_avg, or None if that checkpoint wasn't eval'd."""
    if model not in _cache:
        try:
            df = pl.read_parquet(f"{S3}/{model}/mendelian_traits.parquet").filter(
                pl.col("score_type") == SCORE
            )
            _cache[model] = {r["subset"]: (r["value"], r["se"]) for r in df.to_dicts()}
        except Exception:
            _cache[model] = None
    return _cache[model]


def family_offline(
    region: str, subset: str
) -> tuple[list[int], list[float], list[float]]:
    steps, vals, ses = [], [], []
    for n in CAND_STEPS:
        m = read_metrics(f"exp232-v4_{region}-step-{n}")
        if m and subset in m:
            steps.append(n)
            vals.append(m[subset][0])
            ses.append(m[subset][1])
    return steps, vals, ses


def order_online(
    api: wandb.Api, sub: str, subset: str
) -> tuple[list[int], list[float]]:
    key = ONLINE_KEY.format(subset=subset)
    frames = [
        h
        for r in api.runs("marin", filters={"group": GROUP_ORDER})
        if sub in r.name
        for h in [r.history(keys=[key], samples=10000, pandas=True)]
        if len(h) and key in h.columns
    ]
    if not frames:
        return [], []
    df = (
        pd.concat(frames)
        .dropna(subset=[key])
        .drop_duplicates("_step")
        .sort_values("_step")
    )
    return df["_step"].tolist(), df[key].tolist()


def _label(subset: str) -> str:
    return (
        subset.replace("_variant", "")
        .replace("non_coding_transcript_exon", "ncRNA")
        .replace("3_prime_UTR", "3′UTR")
        .replace("5_prime_UTR", "5′UTR")
    )


def main() -> None:
    set_style()
    api = wandb.Api()
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.4), squeeze=False)
    handles_seen = False
    for k, (subset, region) in enumerate(PANELS):
        ax = axes[k // 4][k % 4]
        sub, off_region, _ = REGIONS[region]
        fs, fv, fe = family_offline(off_region, subset)
        os_, ov = order_online(api, sub, subset)

        h1 = ax.errorbar(
            fs,
            fv,
            yerr=fe,
            color=C_FAMILY,
            marker="o",
            ms=5,
            lw=1.8,
            capsize=0,
            elinewidth=1.2,
            label="family (108 sp.) — offline, ±1 SE",
        )
        (h2,) = ax.plot(
            os_,
            ov,
            color=C_ORDER,
            marker=".",
            ms=7,
            lw=1.8,
            label="order (19 sp.) — online",
        )
        ax.axhline(BASELINE, ls=":", color="gray", lw=0.9)
        ax.set_title(f"{_label(subset)}  ({region})", fontsize=10)
        ax.grid(alpha=0.3)
        if k % 4 == 0:
            ax.set_ylabel("matched AUPRC")
        if k // 4 == 1:
            ax.set_xlabel("training step")
        if not handles_seen:
            fig.legend(
                handles=[h1, h2],
                loc="upper center",
                bbox_to_anchor=(0.5, 0.925),
                fontsize=10.5,
                ncol=2,
                frameon=False,
            )
            handles_seen = True
    fig.suptitle(
        "exp255 (#255) — matched AUPRC vs training step: family (108 sp.) vs order (19 sp.)\n"
        "0.25B; family = offline (±1 SE), order = online; dotted = 0.10 prevalence baseline",
        fontsize=10,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", dict(dpi=130)), ("svg", {})):
        fig.savefig(OUT / f"exp255_auprc_trajectory.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"wrote exp255_auprc_trajectory.png + .svg in {OUT}/")


if __name__ == "__main__":
    main()
