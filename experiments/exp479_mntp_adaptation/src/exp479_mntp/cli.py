"""Command-line entrypoint for exp479 data preparation and training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from exp479_mntp.config import DATA_COMPONENTS, TRAIN_STEPS
from exp479_mntp.data import build_sequence_plan
from exp479_mntp.pilot import run_pilot
from exp479_mntp.preflight import run_preflight
from exp479_mntp.train import train_arm
from exp479_mntp.trainer_preflight import run_trainer_preflight


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
    trainer_preflight = subparsers.add_parser(
        "trainer-preflight", help="measure one exact Lightning optimizer step"
    )
    trainer_preflight.add_argument("--batch-size", type=int, required=True)
    trainer_preflight.add_argument("--output", type=Path, required=True)

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

    pilot = subparsers.add_parser(
        "pilot", help="run every trained arm after a passing GH200 preflight"
    )
    pilot.add_argument("--preflight", type=Path, required=True)
    pilot.add_argument("--artifact-dir", type=Path, required=True)
    pilot.add_argument("--hf-repo-id", required=True)
    pilot.add_argument("--model-card", type=Path, required=True)
    pilot.add_argument("--experiment-commit", required=True)
    pilot.add_argument("--seed", type=int, default=0)
    pilot.add_argument("--num-workers", type=int, default=4)
    pilot.add_argument("--offline-wandb", action="store_true")
    pilot.add_argument(
        "--maximum-batch-size",
        type=int,
        help="cap a passing preflight selection after a full-trainer first-step OOM",
    )
    pilot.add_argument("--trainer-preflight", type=Path)
    pilot.add_argument(
        "--model-card-reviewed",
        action="store_true",
        help="assert that a human reviewed the private staging model card",
    )

    evaluate = subparsers.add_parser("evaluate", help="run odd/X VEP diagnostics")
    evaluate.add_argument("--artifact-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--hf-repo-id", required=True)
    evaluate.add_argument("--batch-size", type=int, required=True)
    evaluate.add_argument("--n-bootstrap", type=int, default=1_000)

    nuc_dep = subparsers.add_parser(
        "nuc-dep", help="run the fixed transferred-MNTP dependency-map panel"
    )
    nuc_dep.add_argument("--artifact-dir", type=Path, required=True)
    nuc_dep.add_argument("--output-dir", type=Path, required=True)
    nuc_dep.add_argument("--hf-repo-id", required=True)
    nuc_dep.add_argument("--batch-size", type=int, default=1_024)

    finalize = subparsers.add_parser("finalize", help="publish the pre-autodown cost record")
    finalize.add_argument("--artifact-dir", type=Path, required=True)
    finalize.add_argument("--hf-repo-id", required=True)
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
    if args.command == "trainer-preflight":
        result = run_trainer_preflight(batch_size=args.batch_size, output_path=args.output)
        print(json.dumps(result, indent=2))
        if result["status"] != "passed":
            raise SystemExit(1)
        return

    if args.command == "pilot":
        if not args.model_card_reviewed:
            raise RuntimeError("pilot publication requires human model-card review")
        run_pilot(
            preflight_path=args.preflight,
            artifact_dir=args.artifact_dir,
            hf_repo_id=args.hf_repo_id,
            model_card=args.model_card,
            experiment_commit=args.experiment_commit,
            seed=args.seed,
            num_workers=args.num_workers,
            offline_wandb=args.offline_wandb,
            maximum_batch_size=args.maximum_batch_size,
            trainer_preflight_path=args.trainer_preflight,
        )
        return

    if args.command == "evaluate":
        from exp479_mntp.vep import run_vep_evaluation

        run_vep_evaluation(
            artifact_dir=args.artifact_dir,
            output_dir=args.output_dir,
            hf_repo_id=args.hf_repo_id,
            batch_size=args.batch_size,
            n_bootstrap=args.n_bootstrap,
        )
        return

    if args.command == "nuc-dep":
        from exp479_mntp.nucleotide_dependency import run_dependency_panel

        run_dependency_panel(
            artifact_dir=args.artifact_dir,
            output_dir=args.output_dir,
            hf_repo_id=args.hf_repo_id,
            batch_size=args.batch_size,
        )
        return

    if args.command == "finalize":
        from exp479_mntp.publishing import publish_cost_estimate

        publish_cost_estimate(artifact_dir=args.artifact_dir, repo_id=args.hf_repo_id)
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
        hf_repo_id=None,
    )


if __name__ == "__main__":
    main()
