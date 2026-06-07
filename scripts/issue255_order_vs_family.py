"""Order- vs family-deduplicated per-region 0.25B — matched comparison (issue #255).

exp255 trains two 0.25B arms on ORDER-deduplicated zoonomia (19-species,
one-genome-per-order cohort) for the two largest v4 slices — cds and
ccre_non_promoter — using the IDENTICAL region/hparams/compute as exp232's
FAMILY-cohort (108-species) arms. Question (NOT the diagonal): holding the
training region and budget fixed, does collapsing to one genome per order help or
hurt the specialist?

Tables only (per #255 review). The only metric here with rigorous uncertainty is
AUPRC (cluster-bootstrap SE from the evals_v2 offline eval), so the conclusion is
drawn from it. The likelihoods are single-run point estimates with NO error bars
and are reported as suggestive context, not claims.

Each order arm is compared to its family twin on that arm's matched region only
(per the exp232 region->subset map in plots/exp232_per_region.py):

    cds  arm:  AUPRC {missense, synonymous, splicing};  LL on val_cds
    ccre arm:  AUPRC {distal};                          LL on val_enhancer

Emits three GitHub-markdown tables to stdout:
  1. matched AUPRC — offline minus_llr_avg @ step-4999, ±1 SE (cluster bootstrap)
  2. matched-region LL functional / non-functional / gap — wandb final-step val
     loss, LL = -loss; single-run point estimate, NO SE
  3. online vs offline AUPRC over all 8 subsets for the two order arms — the
     post-#266 BOS-fix sanity check

Run:  uv run python scripts/issue255_order_vs_family.py
"""

from __future__ import annotations

import numpy as np
import polars as pl

import wandb

S3_PREFIX = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics"
SCORE_TYPE = "minus_llr_avg"
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
    """(arm, cohort) -> {func_ll, nonfunc_ll, gap, online_auprc{subset}}.

    LL = -loss on the arm's matched val recipe; online_auprc is the in-training
    lm_eval AUPRC for the order arms (post-#266, BOS-faithful).
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
                func_ll=-fl,
                nonfunc_ll=-nl,
                gap=nl - fl,
                online_auprc={
                    ss: s.get(f"lm_eval/mendelian_traits_255/{ss}/avg/auprc")
                    for ss in ALL_SUBSETS
                },
            )
    return out


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a GitHub-flavored markdown table."""
    out = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    wd = load_wandb()
    cache = {
        (arm, coh): offline_auprc(ARMS[arm][f"{coh}_model"])
        for arm in ARMS
        for coh in ("order", "family")
    }

    # 1 — matched AUPRC (the rigorous metric).
    rows = []
    for arm, c in ARMS.items():
        for ss in c["subsets"]:
            fv, fe = cache[(arm, "family")][ss]
            ov, oe = cache[(arm, "order")][ss]
            rows.append(
                [
                    arm,
                    ss.replace("_variant", ""),
                    f"{fv:.3f} ± {fe:.3f}",
                    f"{ov:.3f} ± {oe:.3f}",
                    f"{ov - fv:+.3f}",
                ]
            )
    print("### 1. Matched Mendelian AUPRC — offline `minus_llr_avg` @ step-4999\n")
    print("Error = ±1 SE (cluster bootstrap). Prevalence baseline = 0.10.\n")
    print(
        md_table(
            ["arm", "subset", "family (108 sp.)", "order (19 sp.)", "Δ (order−family)"],
            rows,
        )
    )

    # 2 — matched-region LL (suggestive: single-run, no SE).
    rows = []
    for arm, c in ARMS.items():
        lo, lf = wd[(arm, "order")], wd[(arm, "family")]
        for label, key in (
            ("LL functional", "func_ll"),
            ("LL non-functional", "nonfunc_ll"),
            ("LL gap", "gap"),
        ):
            rows.append(
                [
                    arm,
                    c["recipe"],
                    label,
                    f"{lf[key]:+.3f}",
                    f"{lo[key]:+.3f}",
                    f"{lo[key] - lf[key]:+.3f}",
                ]
            )
    print("\n### 2. Matched-region LL — wandb final-step val loss (LL = −loss)\n")
    print(
        "**Single-run point estimates, NO error bars** — suggestive context, not a claim.\n"
    )
    print(
        md_table(
            ["arm", "val set", "metric", "family", "order", "Δ (order−family)"], rows
        )
    )

    # 3 — online vs offline AUPRC, order arms, all subsets (BOS sanity).
    rows, summary = [], []
    for arm, c in ARMS.items():
        off = cache[(arm, "order")]
        on = wd[(arm, "order")]["online_auprc"]
        deltas = []
        for ss in ALL_SUBSETS:
            onv, offv = on.get(ss), off.get(ss, (None,))[0]
            if onv is None or offv is None:
                continue
            deltas.append(offv - onv)
            rows.append(
                [
                    f"{arm}_order",
                    ss.replace("_variant", ""),
                    f"{onv:.3f}",
                    f"{offv:.3f}",
                    f"{offv - onv:+.3f}",
                ]
            )
        summary.append(f"{arm}_order mean|Δ|={np.mean(np.abs(deltas)):.3f}")
    print(
        "\n### 3. Online vs offline AUPRC — order arms, all 8 subsets (post-#266 BOS sanity)\n"
    )
    print(
        f"Agreement: {', '.join(summary)} (max |Δ| = {max(abs(float(r[4])) for r in rows):.3f}).\n"
    )
    print(md_table(["arm", "subset", "online", "offline", "Δ (offline−online)"], rows))


if __name__ == "__main__":
    main()
