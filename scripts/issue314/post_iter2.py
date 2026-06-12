"""Post the iter2 nested-LOCO protocol results to issue #314.

Reads ``nested_<model>.parquet`` (from ``iter2_nested``) for the four models and posts a
factual cross-model table: **per-chromosome-weighted AUPRC** (the primary metric — TraitGym
convention, avoids the cross-fold-calibration artefact of the global metric) for the probe
vs the zero-shot LLR, plus the global metric for contrast (the two disagreeing is itself a
finding), and the **selected-C range per model** (the grid-truncation / fairness check — all
optima should be interior to ``[1e-12, 1e2]``). Narrative synthesis is added in a follow-up.

Run by the iter2 orchestrator (and re-runnable by hand). Skips any model whose parquet is
missing.
"""

import pathlib
import subprocess

import numpy as np
import polars as pl

MARKER = pathlib.Path("scratch/iter2_posted")  # idempotency: post exactly once

MODELS = {"exp135-1B-m5.1": "exp135-1B", "scaling-v0.5-1B": "scaling-1B",
          "exp166-v0.1-p1B": "exp166-1B", "exp166-v0.1-p4B": "exp166-4B"}
ORDER = ["missense_variant", "distal", "tss_proximal", "splicing", "synonymous_variant",
         "5_prime_UTR_variant", "non_coding_transcript_exon_variant", "3_prime_UTR_variant"]
SH = {"missense_variant": "missense", "distal": "distal", "tss_proximal": "tss",
      "splicing": "splicing", "synonymous_variant": "synon", "5_prime_UTR_variant": "5pUTR",
      "non_coding_transcript_exon_variant": "ncRNA", "3_prime_UTR_variant": "3pUTR"}
PRIMARY_REP = "entire_window/abs_delta"
BASE = "s3://oa-bolinas/analysis/issue314/iter2_nested"


def main() -> None:
    if MARKER.exists():
        print("iter2 already posted (marker present); skipping")
        return
    data, present = {}, []
    for mid in MODELS:
        try:
            df = pl.read_parquet(f"{BASE}/nested_{mid}.parquet")
        except Exception:
            continue
        present.append(mid)
        data[mid] = {(r["subset"], r["rep"]): r for r in df.iter_rows(named=True)}
    assert present, "no nested parquets found"

    def row_for(metric: str) -> list[str]:
        lines = ["| subset | " + " | ".join(MODELS[m] for m in present) + " |",
                 "|---|" + "---:|" * len(present)]
        for s in ORDER:
            cells = []
            for m in present:
                r = data[m].get((s, PRIMARY_REP))
                if r is None:
                    cells.append("—")
                elif metric == "delta_pc":
                    cells.append(f"{r['probe_perchrom'] - r['llr_perchrom']:+.3f}")
                else:
                    cells.append(f"{r[metric]:.3f}")
            lines.append(f"| {SH[s]} | " + " | ".join(cells) + " |")
        return lines

    # selected-C interior check (across subsets, both reps) per model
    cdiag = []
    for m in present:
        cmins = [r["c_min"] for r in data[m].values()]
        cmaxs = [r["c_max"] for r in data[m].values()]
        floor = sum(1 for c in cmins if c <= 1.1e-12)
        ceil = sum(1 for c in cmaxs if c >= 0.9e2)
        cdiag.append(f"- **{MODELS[m]}**: C∈10^[{np.log10(min(cmins)):+.1f}, "
                     f"{np.log10(max(cmaxs)):+.1f}]; folds at floor(1e-12)={floor}, "
                     f"at ceil(1e2)={ceil}")

    body = (
        "🤖 **[automated] iter2 — TraitGym nested-LOCO protocol (per-model leakage-free C "
        "tuning).** Fixed rep `entire_window/abs_delta`; leave-one-chromosome-out outer, "
        "inner `GridSearchCV(GroupKFold)` over `logspace(-12, 2)`. **Primary metric = "
        "per-chromosome-weighted AUPRC** (avoids the global metric's cross-fold-calibration "
        "artefact). Factual tables; synthesis follows.\n\n"
        "**Per-chrom AUPRC — probe** (`entire_window/abs_delta`)\n\n"
        + "\n".join(row_for("probe_perchrom")) + "\n\n"
        "**Per-chrom AUPRC — zero-shot LLR**\n\n"
        + "\n".join(row_for("llr_perchrom")) + "\n\n"
        "**Δ = probe − LLR (per-chrom)**\n\n"
        + "\n".join(row_for("delta_pc")) + "\n\n"
        "**Selected-C diagnostic (grid-truncation / fairness check)** — optima should be "
        "interior to `[1e-12, 1e2]`:\n" + "\n".join(cdiag) + "\n\n"
        "Global-metric AUPRC + the signed `entire_window/delta` arm are in the parquets "
        "(`s3://oa-bolinas/analysis/issue314/iter2_nested/`); the global-vs-per-chrom "
        "disagreement is covered in the synthesis."
    )
    subprocess.run(["gh", "issue", "comment", "314", "--repo", "Open-Athena/marin-dna",
                    "--body", body], check=True)
    MARKER.parent.mkdir(exist_ok=True)
    MARKER.touch()
    print(f"posted iter2 ({len(present)} models: {present})")


if __name__ == "__main__":
    main()
