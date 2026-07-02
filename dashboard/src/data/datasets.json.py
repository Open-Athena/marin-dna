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

# Supervised-mode (linear-probe) metric caption — shown in place of `_METRIC` when the
# Mendelian page's mode toggle is on Supervised. Same dataset/variants, different scoring.
_PROBE_METRIC = (
    "**Per-chromosome-weighted AUPRC (TraitGym) ± chromosome-cluster bootstrap SE.** For "
    "each consequence subset, `average_precision_score` is computed *within* each "
    "chromosome then size-weighted across chromosomes (the TraitGym metric); SE is a "
    "cluster bootstrap resampling chromosomes (#347). Macro Avg is the unweighted mean over "
    "subsets — there is **no Global** column, because the probe trains a separate classifier "
    "per subset, so a pooled global ranking is undefined. 1:9 matching ⇒ the random-baseline "
    "AUPRC is 0.10."
)

# caQTL/dsQTL moved to the supervised official-metrics Accessibility QTL page (#312),
# which carries its own dataset metadata — so no caqtl/dsqtl entries here anymore.

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
        "probe_metric": _PROBE_METRIC,
        "probe_notes": [
            "**Supervised:** a frozen-embedding L2-logistic linear probe (nested leave-one-chromosome-out; #314/#320) on **MarinDNA models only** — probes read our per-allele embeddings, so no competitor family appears here.",
            "A subset is probed only if it has ≥300 variants and ≥3 chromosomes (`min_variants` / `min_chroms`); smaller subsets (e.g. mature-miRNA) get no probe score.",
            "Not comparable to the Unsupervised metric (matched-pair AUPRC) — different weighting and matching. The toggle switches worlds; the two are never shown together.",
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
