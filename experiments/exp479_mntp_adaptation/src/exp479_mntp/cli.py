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
    pilot.add_argument("--resume-hf-repo-id")
    pilot.add_argument("--checkpoint-upload-steps", type=int, nargs="*")
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

    diagnostics = subparsers.add_parser(
        "context-window",
        help="run transferred-MNTP context-ablation and window-shift diagnostics",
    )
    diagnostics.add_argument("--artifact-dir", type=Path, required=True)
    diagnostics.add_argument("--output-dir", type=Path, required=True)
    diagnostics.add_argument("--hf-repo-id", required=True)
    diagnostics.add_argument("--batch-size", type=int, default=512)
    diagnostics.add_argument("--n-bootstrap", type=int, default=1_000)

    nuc_dep = subparsers.add_parser(
        "nuc-dep", help="run the fixed transferred-MNTP dependency-map panel"
    )
    nuc_dep.add_argument("--artifact-dir", type=Path, required=True)
    nuc_dep.add_argument("--output-dir", type=Path, required=True)
    nuc_dep.add_argument("--hf-repo-id", required=True)
    nuc_dep.add_argument("--batch-size", type=int, default=1_024)

    audit = subparsers.add_parser(
        "checkpoint-audit",
        help="replay early CLM and audit checkpoint trajectories and alignment",
    )
    audit.add_argument("--artifact-dir", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--train-plan", type=Path, required=True)
    audit.add_argument("--validation-plan", type=Path, required=True)
    audit.add_argument("--logged-loss-csv", type=Path, required=True)
    audit.add_argument("--hf-repo-id", required=True)
    audit.add_argument("--batch-size", type=int, default=64)
    audit.add_argument("--vep-batch-size", type=int, default=1_024)
    audit.add_argument("--dependency-batch-size", type=int, default=1_024)
    audit.add_argument("--n-bootstrap", type=int, default=200)
    audit.add_argument("--num-workers", type=int, default=4)

    stability = subparsers.add_parser(
        "stability-audit",
        help="replay all three arms with pre-clipping gradient-norm tracing",
    )
    stability.add_argument("--artifact-dir", type=Path, required=True)
    stability.add_argument("--output-dir", type=Path, required=True)
    stability.add_argument("--train-plan", type=Path, required=True)
    stability.add_argument("--validation-plan", type=Path, required=True)
    stability.add_argument("--hf-repo-id", required=True)
    stability.add_argument("--batch-size", type=int, default=64)
    stability.add_argument("--num-workers", type=int, default=4)

    recheck = subparsers.add_parser(
        "inference-recheck",
        help="recheck exact-batch VEP parity and paired-baseline dependencies",
    )
    recheck.add_argument("--artifact-dir", type=Path, required=True)
    recheck.add_argument("--output-dir", type=Path, required=True)
    recheck.add_argument("--hf-repo-id", required=True)
    recheck.add_argument("--vep-batch-size", type=int, default=1_024)
    recheck.add_argument("--dependency-batch-size", type=int, default=1_024)
    recheck.add_argument("--n-bootstrap", type=int, default=200)

    final_dependency = subparsers.add_parser(
        "final-dependency",
        help="compare tRNA dependency at each trained arm's final checkpoint",
    )
    final_dependency.add_argument("--artifact-dir", type=Path, required=True)
    final_dependency.add_argument("--output-dir", type=Path, required=True)
    final_dependency.add_argument("--hf-repo-id", required=True)
    final_dependency.add_argument("--batch-size", type=int, default=1_024)

    validation_report = subparsers.add_parser(
        "validation-report",
        help="plot component validation loss for all original training arms",
    )
    validation_report.add_argument("--output-dir", type=Path, required=True)

    calibration = subparsers.add_parser(
        "causal-calibration",
        help="run the single AdamW 1e-6 causal fine-tuning sanity arm",
    )
    calibration.add_argument("--artifact-dir", type=Path, required=True)
    calibration.add_argument("--output-dir", type=Path, required=True)
    calibration.add_argument("--train-plan", type=Path, required=True)
    calibration.add_argument("--validation-plan", type=Path, required=True)
    calibration.add_argument("--hf-repo-id", required=True)
    calibration.add_argument("--batch-size", type=int, default=64)
    calibration.add_argument("--seed", type=int, default=0)
    calibration.add_argument("--num-workers", type=int, default=4)
    calibration.add_argument("--offline-wandb", action="store_true")

    longrun = subparsers.add_parser(
        "causal-longrun",
        help="run the retained AdamW 1e-5 causal trajectory for 1,000 steps",
    )
    longrun.add_argument("--artifact-dir", type=Path, required=True)
    longrun.add_argument("--output-dir", type=Path, required=True)
    longrun.add_argument("--train-plan", type=Path, required=True)
    longrun.add_argument("--validation-plan", type=Path, required=True)
    longrun.add_argument("--batch-size", type=int, default=64)
    longrun.add_argument("--seed", type=int, default=0)
    longrun.add_argument("--num-workers", type=int, default=4)
    longrun.add_argument("--offline-wandb", action="store_true")

    normalization = subparsers.add_parser(
        "loss-normalization-audit",
        help="re-evaluate retained causal checkpoints with Marin-compatible loss",
    )
    normalization.add_argument("--artifact-dir", type=Path, required=True)
    normalization.add_argument("--output-dir", type=Path, required=True)
    normalization.add_argument("--train-plan", type=Path, required=True)
    normalization.add_argument("--validation-plan", type=Path, required=True)
    normalization.add_argument("--batch-size", type=int, default=64)

    source_validation = subparsers.add_parser(
        "source-validation-reproduction",
        help="reproduce the original nine full-dataset validation metrics",
    )
    source_validation.add_argument("--output-dir", type=Path, required=True)
    source_validation.add_argument("--batch-size", type=int, default=128)

    finalize = subparsers.add_parser("finalize", help="publish the pre-autodown cost record")
    finalize.add_argument("--artifact-dir", type=Path, required=True)
    finalize.add_argument("--hf-repo-id", required=True)
    finalize_local = subparsers.add_parser(
        "finalize-local",
        help="print the pre-autodown cost record without external publication",
    )
    finalize_local.add_argument("--artifact-dir", type=Path, required=True)
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
            resume_hf_repo_id=args.resume_hf_repo_id,
            checkpoint_upload_steps=(
                None
                if args.checkpoint_upload_steps is None
                else tuple(args.checkpoint_upload_steps)
            ),
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

    if args.command == "context-window":
        from exp479_mntp.context_window_diagnostics import (
            run_context_window_diagnostics,
        )

        run_context_window_diagnostics(
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

    if args.command == "checkpoint-audit":
        from exp479_mntp.checkpoint_audit import run_checkpoint_audit

        run_checkpoint_audit(
            artifact_dir=args.artifact_dir,
            output_dir=args.output_dir,
            train_plan=args.train_plan,
            validation_plan=args.validation_plan,
            logged_loss_csv=args.logged_loss_csv,
            hf_repo_id=args.hf_repo_id,
            batch_size=args.batch_size,
            vep_batch_size=args.vep_batch_size,
            dependency_batch_size=args.dependency_batch_size,
            n_bootstrap=args.n_bootstrap,
            num_workers=args.num_workers,
        )
        return

    if args.command == "stability-audit":
        from exp479_mntp.checkpoint_audit import run_training_stability_audit

        run_training_stability_audit(
            artifact_dir=args.artifact_dir,
            output_dir=args.output_dir,
            train_plan=args.train_plan,
            validation_plan=args.validation_plan,
            hf_repo_id=args.hf_repo_id,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        return

    if args.command == "inference-recheck":
        from exp479_mntp.inference_recheck import run_inference_recheck

        run_inference_recheck(
            artifact_dir=args.artifact_dir,
            output_dir=args.output_dir,
            hf_repo_id=args.hf_repo_id,
            vep_batch_size=args.vep_batch_size,
            dependency_batch_size=args.dependency_batch_size,
            n_bootstrap=args.n_bootstrap,
        )
        return

    if args.command == "final-dependency":
        from exp479_mntp.final_dependency import run_final_checkpoint_dependency

        run_final_checkpoint_dependency(
            artifact_dir=args.artifact_dir,
            output_dir=args.output_dir,
            hf_repo_id=args.hf_repo_id,
            batch_size=args.batch_size,
        )
        return

    if args.command == "validation-report":
        from exp479_mntp.validation_report import run_validation_report

        run_validation_report(args.output_dir)
        return

    if args.command == "causal-calibration":
        from exp479_mntp.causal_calibration import run_causal_calibration

        run_causal_calibration(
            artifact_dir=args.artifact_dir,
            output_dir=args.output_dir,
            train_plan=args.train_plan,
            validation_plan=args.validation_plan,
            hf_repo_id=args.hf_repo_id,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
            offline_wandb=args.offline_wandb,
        )
        return
    if args.command == "causal-longrun":
        from exp479_mntp.causal_longrun import run_causal_longrun

        run_causal_longrun(
            artifact_dir=args.artifact_dir,
            output_dir=args.output_dir,
            train_plan=args.train_plan,
            validation_plan=args.validation_plan,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
            offline_wandb=args.offline_wandb,
        )
        return
    if args.command == "loss-normalization-audit":
        from exp479_mntp.loss_normalization_audit import run_loss_normalization_audit

        run_loss_normalization_audit(
            artifact_dir=args.artifact_dir,
            output_dir=args.output_dir,
            train_plan=args.train_plan,
            validation_plan=args.validation_plan,
            batch_size=args.batch_size,
        )
        return
    if args.command == "source-validation-reproduction":
        from exp479_mntp.source_validation_reproduction import (
            run_source_validation_reproduction,
        )

        run_source_validation_reproduction(
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
        return
    if args.command == "finalize":
        from exp479_mntp.publishing import publish_cost_estimate

        publish_cost_estimate(artifact_dir=args.artifact_dir, repo_id=args.hf_repo_id)
        return
    if args.command == "finalize-local":
        from exp479_mntp.publishing import write_cost_estimate

        path = write_cost_estimate(artifact_dir=args.artifact_dir)
        print(path.read_text(encoding="utf-8"), end="")
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
