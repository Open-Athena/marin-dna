"""Observable Framework data loader: dataset-level metadata → JSON.

Per-dataset pins (HF commit, score type, tracking issue, etc.) for the
dataset-metadata block shown above the leaderboard.
"""

from __future__ import annotations

import json
import sys

_METRIC = (
    "AUPRC ± cluster-bootstrap SE — area under the precision-recall curve, "
    "with SE estimated by resampling `match_group`s with replacement "
    "(preserving the matched 1:9 clustering). `n` in each column header is "
    "the total variants in the subset (positives + matched negatives); "
    "1:9 matching means 10% are positives by design, so the random-baseline "
    "AUPRC is 0.10."
)

# QTL (caqtl/dsqtl) use the `qtl_global` eval path — no matching, no subsets —
# so the metric story is different from the matched-pair `_METRIC` above
# (which assumes 1:9 clustering and a 0.10 baseline). Pick one metric at a time
# via the selector on the page.
_QTL_METRIC = (
    "Pick a metric with the selector above. **AUPRC** ± bootstrap SE — over "
    "*all* variants (significant QTLs vs unmatched control variants); the "
    "random baseline is the positive rate (not 0.10), which differs per "
    "dataset. **Pearson r** / **Spearman ρ** ± bootstrap SE — correlation of "
    "the variant-effect score with the measured `effect_size`, over the "
    "*positives only*. The color scale and forest-plot axis rescale to the "
    "selected metric."
)

_SGE_METRIC = (
    "**AUPRC** ± bootstrap SE for the ClinGen/ExCALIBR `calibrated_class` "
    "abnormal-vs-normal call, computed **per accession** (MaveDB study) then "
    "macro-averaged over consequence subsets and accessions (scores are "
    "non-comparable across studies). Rank-based, so conservation tracks and gLMs "
    "compare on the same footing. The abnormal base rate varies per gene/subset "
    "(~5–16%), so the random-baseline AUPRC is not a single number. AUPRC-only, "
    "matching the classification framing of the other benchmarks; the continuous "
    "function-score columns remain in the dataset (v3) for provenance."
)

DATASETS = {
    "mendelian_traits": {
        "name": "Mendelian Traits",
        "hf_repo": "bolinas-dna/evals_mendelian_traits",
        "hf_commit": "4aed58e5",
        "score_type": "minus_llr_avg",
        "leading_aggregate": "macro_avg",
        "issue": "https://github.com/Open-Athena/marin-dna/issues/161",
        "split": "train",
        "n_min_per_subset": 30,
        "positives": "OMIM ∪ HGMD ∪ Smedley et al. 2016 pathogenic SNVs (AF < 0.1%)",
        "negatives": "gnomAD AF > 0.1%",
        "matching": "1:9 nearest-neighbor on consequence + chrom strata, with continuous TSS/exon distance features (RobustScaler-scaled Euclidean)",
        "metric": _METRIC,
        "notes": [
            "Per-subset columns exclude subsets with fewer than 30 positives (`n < 300` under 1:9 matching).",
            "Sorted by Macro Avg by default — the consequence-subset distribution is dominated by missense (a ClinVar annotator-history artifact, not pathogenicity reality), so Global AUPRC over-weights protein-coding-specialist methods. Macro Avg gives equal weight to each subset.",
        ],
    },
    "complex_traits": {
        "name": "Complex Traits",
        "hf_repo": "bolinas-dna/evals_complex_traits",
        "hf_commit": "22f86a89",
        "score_type": "abs_llr_avg",
        "leading_aggregate": "global",
        "issue": "https://github.com/Open-Athena/marin-dna/issues/162",
        "split": "train",
        "n_min_per_subset": 30,
        "positives": "UKBB fine-mapped complex-trait variants — SuSiE + FINEMAP `max(PIP across traits) > 0.9`",
        "negatives": "`max(PIP) < 0.01` AND no SuSiE/FINEMAP combine-step null-PIP among those traits (`label_variants_by_pip(use_null_pip_guard=True)`)",
        "matching": "1:9 nearest-neighbor on consequence + chrom strata, with continuous TSS/exon distance + MAF features (RobustScaler-scaled Euclidean)",
        "metric": _METRIC,
        "notes": [
            "Per-subset columns exclude subsets with fewer than 30 positives (`n < 300` under 1:9 matching). Most consequence subsets in this dataset fall below that threshold — distal and missense are the only ones reported.",
            "Sorted by Global by default. Score column is `abs_llr_avg` (magnitude) rather than `minus_llr_avg` — for complex-trait fine-mapped variants we don't have a pathogenicity direction, only that the variant is causal.",
        ],
    },
    "caqtl": {
        "name": "caQTL (chromatin accessibility)",
        "hf_repo": "bolinas-dna/evals_caqtl",
        "hf_commit": "9d004a21",
        "score_type": "abs_llr_avg",
        "issue": "https://github.com/Open-Athena/marin-dna/issues/214",
        "split": "train",
        "positives": "DART-Eval Task-5 significant chromatin-accessibility QTLs (3,173 of 41,382 variants; DeGorter et al. 2023)",
        "negatives": "DART-Eval Task-5 control variants (non-significant)",
        "matching": "none — variants scored as-is (no 1:9 matching, no subsetting)",
        "metric": _QTL_METRIC,
        "notes": [
            "DART-Eval Task-5 caQTL benchmark (`eval_protocol: qtl_global`). No consequence subsets and no matched negatives, so the leaderboard is a single selected-metric column (+ forest plot) rather than the per-consequence heatmap.",
            "Random-baseline AUPRC is the positive rate ≈ 0.077 (3,173 / 41,382). AUPRC is over all variants; Pearson/Spearman over the 3,173 positives only.",
            "MarinDNA defaults to NucDep (Jensen-Shannon divergence, `jsd_avg`) here — a symmetric distributional distance suited to unsigned QTL effects; the LLR magnitude protocol (`abs_llr_avg`) is one click away via the protocol toggle.",
        ],
    },
    "dsqtl": {
        "name": "dsQTL (DNase I sensitivity)",
        "hf_repo": "bolinas-dna/evals_dsqtl",
        "hf_commit": "b7e02a07",
        "score_type": "abs_llr_avg",
        "issue": "https://github.com/Open-Athena/marin-dna/issues/214",
        "split": "train",
        "positives": "DART-Eval Task-5 significant DNase-I-sensitivity QTLs (309 of 15,018 variants; Degner et al. 2012, hg19→GRCh38)",
        "negatives": "DART-Eval Task-5 control variants (non-significant)",
        "matching": "none — variants scored as-is (no 1:9 matching, no subsetting)",
        "metric": _QTL_METRIC,
        "notes": [
            "DART-Eval Task-5 dsQTL benchmark (`eval_protocol: qtl_global`). Single selected-metric column (+ forest plot); no subsets, no matched negatives.",
            "Random-baseline AUPRC is the positive rate ≈ 0.021 (309 / 15,018). AUPRC is over all variants; Pearson/Spearman over the 309 positives only — the small positive count makes the correlation SEs wide (~0.06).",
            "MarinDNA defaults to NucDep (Jensen-Shannon divergence, `jsd_avg`) here — a symmetric distributional distance suited to unsigned QTL effects; the LLR magnitude protocol (`abs_llr_avg`) is one click away via the protocol toggle.",
        ],
    },
    "sge": {
        "name": "Saturation Genome Editing (SGE)",
        "hf_repo": "bolinas-dna/evals_sge",
        "hf_commit": "225d3d1e",
        "score_type": "minus_llr_avg",
        "issue": "https://github.com/Open-Athena/marin-dna/issues/301",
        "split": "train",
        "positives": "Variants whose endogenous-locus function is abnormal (ClinGen/ExCALIBR-calibrated) — the AUPRC positives",
        "negatives": "Variants calibrated normal (intermediate / uncalibrated dropped)",
        "matching": "none — metrics are computed per accession (MaveDB study), since function scores are non-comparable across studies, then macro-averaged over subsets and accessions",
        "metric": _SGE_METRIC,
        "notes": [
            "Saturation-genome-editing VEP (12 genes, missense + splicing only; v2 rebuild #300). `function_score_aligned` is the direction-harmonized continuous score; `calibrated_class` is the uniform ClinGen/ExCALIBR abnormal/normal call.",
            "Use the gene selector to view the macro across accessions or a single accession; an AUPRC heatmap (methods × consequence subsets — Macro / Missense / Splicing / Both), colored on a 0→1 scale (anchored at 0, the metric's full range — the abnormal base rate is well below the matched-pair 0.10; upper end matches the matched-pair heatmap).",
            "MarinDNA defaults to LLR (`minus_llr_avg`, signed — the assayed ALT's direction is informative, so not `abs`); NucDep (`jsd_avg`) is one click away. Conservation tracks use their single per-position score.",
        ],
    },
}


def main() -> None:
    json.dump(DATASETS, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
