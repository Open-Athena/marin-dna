from __future__ import annotations

from pathlib import Path

from marin_dna_vertebrate_projection.projection.hal_chain_ramp import (
    RampThresholds,
    next_concurrency,
    read_target_species,
    validate_smoke_gate_payloads,
)


def test_read_target_species_excludes_human_and_preserves_order(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "species.tsv"
    rows = ["species\tfamily"]
    rows.extend(f"species_{index}\tfamily_{index}" for index in range(107))
    rows.insert(51, "Homo_sapiens\tHominidae")
    manifest.write_text("\n".join(rows) + "\n")
    observed = read_target_species(manifest, "Homo_sapiens")
    assert len(observed) == 107
    assert observed[0] == "species_0"
    assert observed[-1] == "species_106"
    assert "Homo_sapiens" not in observed


def test_validate_smoke_gate_payloads_requires_exact_partition() -> None:
    payloads = {
        species: {
            "expected_queries": 9_374,
            "exact_queries": 9_374,
            "exact_fraction": 1.0,
            "chain_multiple_mapping_queries": 0,
            "parity_counts": {"exact_mapped": mapped, "exact_unmapped": 9_374 - mapped},
        }
        for species, mapped in {
            "Papio_anubis": 6_846,
            "Mus_musculus": 1_529,
            "Loxodonta_africana": 2_270,
        }.items()
    }
    observed = validate_smoke_gate_payloads(payloads)
    assert observed["Papio_anubis"]["exact_mapped"] == 6_846
    assert observed["Mus_musculus"]["exact_unmapped"] == 7_845


def test_next_concurrency_doubles_only_when_all_gates_pass() -> None:
    thresholds = RampThresholds(
        minimum_mem_available_bytes=100,
        minimum_disk_free_bytes=200,
        maximum_cpu_busy_fraction=0.85,
        maximum_cpu_iowait_fraction=0.25,
        maximum_load_per_cpu=0.80,
    )
    safe = {
        "mem_available_bytes": 101,
        "disk_free_bytes": 201,
        "cpu_busy_fraction": 0.84,
        "cpu_iowait_fraction": 0.24,
        "load_per_cpu": 0.79,
    }
    assert next_concurrency(2, 40, safe, thresholds) == 4
    assert next_concurrency(32, 40, safe, thresholds) == 40
    unsafe = {**safe, "cpu_iowait_fraction": 0.26}
    assert next_concurrency(4, 40, unsafe, thresholds) == 4
