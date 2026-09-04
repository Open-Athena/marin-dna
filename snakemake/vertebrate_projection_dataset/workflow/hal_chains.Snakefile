"""Issue #523 whole-genome human-to-mammal chain pilot."""


configfile: "config/hal_chain_pilot.yaml"


include: "rules/hal_chain_pilot.smk"


rule all:
    input:
        PRODUCER_MANIFEST,
        expand(CHAIN_PATH, species=PILOT_SPECIES, recipe=CHAIN_RECIPES),
        expand(CHAIN_METRICS, species=PILOT_SPECIES, recipe=CHAIN_RECIPES),
        expand(PARITY_SUMMARY, species=PILOT_SPECIES, recipe=CHAIN_RECIPES),
        expand(PARITY_DISCREPANCIES, species=PILOT_SPECIES, recipe=CHAIN_RECIPES),
        expand(FULL_GRID_METRICS, species=PILOT_SPECIES),


rule chains:
    input:
        PRODUCER_MANIFEST,
        expand(CHAIN_PATH, species=PILOT_SPECIES, recipe=CHAIN_RECIPES),
        expand(CHAIN_METRICS, species=PILOT_SPECIES, recipe=CHAIN_RECIPES),
