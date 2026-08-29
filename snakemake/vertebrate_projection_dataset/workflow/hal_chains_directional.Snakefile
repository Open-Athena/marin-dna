"""Issue #523 direction-matched human-to-mammal chain pilot."""


configfile: "config/hal_chain_directional_pilot.yaml"


include: "rules/hal_chain_directional_pilot.smk"


rule all:
    input:
        PRODUCER_MANIFEST,
        expand(SMOKE_CHAIN, species=SMOKE_SPECIES),
        expand(SMOKE_CHAIN_METRICS, species=SMOKE_SPECIES),
        expand(SMOKE_DIRECT_METRICS, species=SMOKE_SPECIES),
        expand(SMOKE_LIFTOVER_METRICS, species=SMOKE_SPECIES),
        expand(SMOKE_PARITY_SUMMARY, species=SMOKE_SPECIES),
        expand(SMOKE_PARITY_DISCREPANCIES, species=SMOKE_SPECIES),


rule smoke:
    input:
        rules.all.input,


rule full_baboon:
    input:
        rules.smoke.input,
        FULL_CHAIN,
        FULL_CHAIN_METRICS,
        FULL_VALIDATION_METRICS,
        FULL_PARITY_SUMMARY,
        FULL_PARITY_DISCREPANCIES,
        FULL_GRID_METRICS,
