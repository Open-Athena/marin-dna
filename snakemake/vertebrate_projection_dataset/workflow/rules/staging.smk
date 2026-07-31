"""Explicit bootstrap mirroring and checksum-verified local staging."""

from marin_dna.pipelines.vertebrate_projection_dataset.mirror import (
    mirror_source_object,
    read_mirror_manifest,
    s3_object_size,
    stage_hal_object,
    stage_s3_object,
    verify_hal_object,
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
    resources:
        mem_mb=1000,
    run:
        stage_hal_object(params.source, output[0])


rule validate_staged_hal:
    input:
        HAL_PATH,
    output:
        HAL_VALIDATION,
    params:
        source=str(config["hal_s3_uri"]),
    resources:
        mem_mb=1000,
    run:
        expected_size = s3_object_size(params.source)
        verify_hal_object(input[0], expected_size=expected_size)
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(output[0]).write_text(
            f"source={params.source}\nbytes={expected_size}\n"
            "required_genome=Homo_sapiens\n"
        )


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
