"""Run one worker or the adaptive HAL-chain controller for issue #523."""

from __future__ import annotations

import argparse

from marin_dna_vertebrate_projection.projection.hal_chain_ramp import (
    run_controller,
    run_worker,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("controller", "worker"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True)
        subparser.add_argument("--pipeline-commit", required=True)
        if command == "worker":
            subparser.add_argument("--species", required=True)
    args = parser.parse_args()
    if args.command == "worker":
        run_worker(args.config, args.pipeline_commit, args.species)
        return
    raise SystemExit(run_controller(args.config, args.pipeline_commit))


if __name__ == "__main__":
    main()
