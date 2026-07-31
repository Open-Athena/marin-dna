"""Explicit bootstrap mirroring and checksum-verified local staging."""

from marin_dna.pipelines.vertebrate_projection_dataset.mirror import (
    mirror_source_object,
    read_mirror_manifest,
    stage_s3_object,
)


rule mirror_multiz_bootstrap:
    input:
        MIRROR_MANIFEST,
    output:
        f"{RESULTS}/bootstrap/multiz_mirror.done",
    run:
        # Explicit bootstrap only. No normal projection target depends on this
        # rule, so workers can never silently fall back to UCSC.
        for expected in read_mirror_manifest(input[0]):
            mirror_source_object(expected)
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(output[0]).write_text("verified and mirrored\n")


rule stage_hal:
    output:
        local(HAL_PATH),
    params:
        source=str(config["hal_s3_uri"]),
    shell:
        "mkdir -p $(dirname {output}) && "
        "aws s3 cp {params.source} {output} --no-progress"


rule stage_multiz_maf:
    input:
        MIRROR_MANIFEST,
    output:
        local(f"{MULTIZ_STAGE_DIR}/maf/{{chrom}}.maf.gz"),
    wildcard_constraints:
        chrom=CHROM_RE,
    run:
        matches = [
            row
            for row in read_mirror_manifest(input[0])
            if row.kind == "primary_chromosome_maf" and row.chrom == wildcards.chrom
        ]
        assert len(matches) == 1
        stage_s3_object(matches[0], output[0])
