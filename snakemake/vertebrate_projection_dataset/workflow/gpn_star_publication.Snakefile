"""Issue #517 GPN-Star-P split and Hugging Face publication workflow."""


configfile: "config/gpn_star_p_publication.yaml"


include: "rules/gpn_star_publication_common.smk"
include: "rules/gpn_star_publication.smk"


rule all:
    input:
        expand(
            f"{RESULTS}/datasets/{{region}}/split_summary.json",
            region=COHORTS,
        ),
