"""Build response-independent discovery, validation, and test panels for #431."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP429_DIR = REPO_ROOT / "experiments" / "exp429_variant_feature_map"
if str(EXP429_DIR) not in sys.path:
    sys.path.insert(0, str(EXP429_DIR))

import design_heldout_perturbations as builder

ISSUE = 431
SPLITS = ("discovery", "validation", "test")
CONTEXTS_PER_CLASS = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def split_context_group(split: str) -> str:
    assert split in SPLITS
    return f"response_independent_{split}_hash"


def build_split_panels(
    *,
    panel_path: Path,
    panel_manifest_path: Path,
    gtf_path: Path,
    fasta_path: Path,
    output_root: Path,
    contexts_per_class: int = CONTEXTS_PER_CLASS,
) -> dict[str, Any]:
    """Run the #429 coordinate-audited builder once per frozen genomic split."""

    assert contexts_per_class > 0 and not output_root.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert (
        experiment_commit
        and os.environ.get("HELDOUT_DESIGN_COMMIT") == experiment_commit
    )
    output_root.mkdir(parents=True)
    original_split = builder.SOURCE_SPLIT
    original_group = builder.CONTEXT_GROUP
    manifests: dict[str, Any] = {}
    try:
        for split in SPLITS:
            context_group = split_context_group(split)
            builder.SOURCE_SPLIT = split
            builder.CONTEXT_GROUP = context_group
            manifest = builder.design_heldout_perturbations(
                panel_path=panel_path,
                panel_manifest_path=panel_manifest_path,
                gtf_path=gtf_path,
                fasta_path=fasta_path,
                output_dir=output_root / split,
                contexts_per_class=contexts_per_class,
            )
            assert manifest["protocol"]["source_split"] == split
            assert manifest["protocol"]["context_group"] == context_group
            assert manifest["source_rows"] == 4 * contexts_per_class
            manifests[split] = manifest
    finally:
        builder.SOURCE_SPLIT = original_split
        builder.CONTEXT_GROUP = original_group

    panel_hashes = {
        split: {
            "panel_sha256": sha256_file(
                output_root / split / "perturbation_panel.parquet"
            ),
            "sources_sha256": sha256_file(
                output_root / split / "heldout_sources.parquet"
            ),
            "manifest_sha256": sha256_file(output_root / split / "manifest.json"),
            "paired_rows": manifests[split]["rows"],
            "source_rows": manifests[split]["source_rows"],
        }
        for split in SPLITS
    }
    summary = {
        "issue": ISSUE,
        "experiment_commit": experiment_commit,
        "protocol": {
            "splits": list(SPLITS),
            "contexts_per_class": contexts_per_class,
            "selection": "lowest frozen #429 panel sample_hash after unambiguous Ensembl protein-coding annotation; no SAE-response ranking",
            "coordinate_system": "source positions are 1-based; all derived coordinates are 0-based half-open",
        },
        "inputs": {
            "panel_sha256": sha256_file(panel_path),
            "panel_manifest_sha256": sha256_file(panel_manifest_path),
            "gtf_sha256": sha256_file(gtf_path),
            "fasta_sha256": sha256_file(fasta_path),
        },
        "outputs": panel_hashes,
    }
    write_json(output_root / "manifest.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--contexts-per-class", type=int, default=CONTEXTS_PER_CLASS)
    args = parser.parse_args()
    result = build_split_panels(
        panel_path=args.panel,
        panel_manifest_path=args.panel_manifest,
        gtf_path=args.gtf,
        fasta_path=args.fasta,
        output_root=args.output_root,
        contexts_per_class=args.contexts_per_class,
    )
    print(json.dumps(result["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
