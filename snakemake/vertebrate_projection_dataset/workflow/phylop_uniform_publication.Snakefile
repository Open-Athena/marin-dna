"""Issue #517 strict phyloP-control split and Hugging Face publication."""


configfile: "config/phylop_uniform_publication.yaml"


include: "rules/gpn_star_publication_common.smk"
include: "rules/phylop_uniform_publication.smk"


rule all:
    input:
        expand(
            f"{RESULTS}/datasets/{{region}}/split_summary.json",
            region=COHORTS,
        ),
