"""Issue #523 direction-matched human-to-mammal chain pilot."""


configfile: "config/hal_chain_directional_pilot.yaml"


include: "rules/hal_chain_directional_pilot.smk"


rule smoke:
    input:
        manifest=SMOKE_PRODUCER_MANIFEST,
        chains=[SMOKE_CHAIN.format(species=s) for s in SMOKE_SPECIES],
        chain_metrics=[
            SMOKE_CHAIN_METRICS.format(species=s) for s in SMOKE_SPECIES
        ],
        direct_metrics=[
            SMOKE_DIRECT_METRICS.format(species=s) for s in SMOKE_SPECIES
        ],
        liftover_metrics=[
            SMOKE_LIFTOVER_METRICS.format(species=s) for s in SMOKE_SPECIES
        ],
        parity_summaries=[
            SMOKE_PARITY_SUMMARY.format(species=s) for s in SMOKE_SPECIES
        ],
        parity_discrepancies=[
            SMOKE_PARITY_DISCREPANCIES.format(species=s)
            for s in SMOKE_SPECIES
        ],


rule all:
    input:
        rules.smoke.input,


rule full_baboon:
    input:
        rules.smoke.input,
        FULL_PRODUCER_MANIFEST,
        FULL_CHAIN,
        FULL_CHAIN_METRICS,
        FULL_VALIDATION_METRICS,
        FULL_PARITY_SUMMARY,
        FULL_PARITY_DISCREPANCIES,
        FULL_GRID_METRICS,
