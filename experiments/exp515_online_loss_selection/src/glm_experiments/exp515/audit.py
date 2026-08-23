"""CLI for the no-GPU, bounded per-species source-case audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from glm_experiments.exp515.data import audit_case_distribution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-species", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1515)
    args = parser.parse_args()
    payload = audit_case_distribution(
        args.output,
        samples_per_species=args.samples_per_species,
        seed=args.seed,
    )
    if payload["fallback_required"]:
        raise SystemExit("case audit requires the preregistered RefSeq fallback")


if __name__ == "__main__":
    main()
