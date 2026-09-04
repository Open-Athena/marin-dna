"""Issue #517 one-per-order strict-phyloP enhancer dataset publication."""


configfile: "config/phylop_uniform_enhancer_order_publication.yaml"


include: "rules/phylop_uniform_enhancer_order_publication.smk"


rule all:
    input:
        SOURCE_AUDIT,
