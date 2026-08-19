"""Command-line entrypoint for exp479 data preparation and training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from exp479_mntp.config import DATA_COMPONENTS, TRAIN_STEPS
from exp479_mntp.data import build_sequence_plan
from exp479_mntp.preflight import run_preflight
from exp479_mntp.train import train_arm


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-data", help="materialize shared train/validation plans"
    )
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--batch-size", type=int, required=True)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--validation-samples-per-component", type=int, default=128)

    preflight = subparsers.add_parser("preflight", help="run actual-checkpoint GH200 gates")
    preflight.add_argument("--output", type=Path, required=True)

    train = subparsers.add_parser("train", help="train one registered arm")
    train.add_argument(
        "--arm",
        choices=("transferred_mntp", "scratch_mntp", "clm_continuation"),
        required=True,
    )
    train.add_argument("--batch-size", type=int, required=True)
    train.add_argument("--train-plan", type=Path, required=True)
    train.add_argument("--validation-plan", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--resume-from", type=Path)
    train.add_argument("--offline-wandb", action="store_true")
    train.add_argument("--accelerator", choices=("cpu", "gpu"), default="gpu")
    train.add_argument("--precision", default="bf16-mixed")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare-data":
        total_samples = TRAIN_STEPS * args.batch_size
        if total_samples % len(DATA_COMPONENTS) != 0:
            raise ValueError("training exposure must divide evenly across five components")
        train_path = args.output_dir / "train.jsonl"
        validation_path = args.output_dir / "validation.jsonl"
        train_hash = build_sequence_plan(
            train_path,
            samples_per_component=total_samples // len(DATA_COMPONENTS),
            seed=args.seed,
            validation=False,
        )
        validation_hash = build_sequence_plan(
            validation_path,
            samples_per_component=args.validation_samples_per_component,
            seed=args.seed + 10_000,
            validation=True,
        )
        print(f"train={train_path} sha256={train_hash}")
        print(f"validation={validation_path} sha256={validation_hash}")
        return

    if args.command == "preflight":
        result = run_preflight(args.output)
        print(json.dumps(result, indent=2))
        return

    train_arm(
        arm=args.arm,
        batch_size=args.batch_size,
        train_plan=args.train_plan,
        validation_plan=args.validation_plan,
        output_dir=args.output_dir,
        seed=args.seed,
        num_workers=args.num_workers,
        resume_from=args.resume_from,
        offline_wandb=args.offline_wandb,
        accelerator=args.accelerator,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()
