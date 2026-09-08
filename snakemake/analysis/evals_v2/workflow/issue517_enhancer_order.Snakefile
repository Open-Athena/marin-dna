"""Order-control retry with explicitly pinned train-file-only acquisition."""

include: "Snakefile"
include: "rules/issue517_enhancer_order.smk"

ruleorder: compute_scores_exp517_order > compute_scores
