"""Component-level validation trajectories for all issue 479 training arms."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

COMPONENTS = ("cds", "downstream", "enhancer", "ncrna", "upstream")
EXPECTED_STEPS = tuple(range(100, 1_001, 100))
ARM_RUNS = {
    "transferred_mntp": ("6iqcmdm7", "oddka8kk"),
    "scratch_mntp": ("4nstge1d",),
    "clm_continuation": ("yod8l3mb",),
}
ARM_MODES = {
    "transferred_mntp": ("diffusion", "single_mask"),
    "scratch_mntp": ("diffusion", "single_mask"),
    "clm_continuation": ("causal",),
}


def _history_prefix(mode: str) -> str:
    return "val/diffusion" if mode in {"diffusion", "causal"} else "val/single_mask"


def _run_component_history(
    api: wandb.Api,
    *,
    arm: str,
    mode: str,
    run_id: str,
) -> pd.DataFrame:
    prefix = _history_prefix(mode)
    columns = [f"{prefix}/component/{component}/loss" for component in COMPONENTS]
    history = pd.DataFrame(
        api.run(f"gonzalobenegas/marin/{run_id}").scan_history(
            keys=["trainer/global_step", *columns],
            page_size=1_000,
        )
    ).dropna()
    history["step"] = history["trainer/global_step"].astype(int) + 1
    if arm == "transferred_mntp" and run_id == ARM_RUNS[arm][0]:
        history = history[history["step"] <= 800]
    if arm == "transferred_mntp" and run_id == ARM_RUNS[arm][1]:
        history = history[history["step"] > 800]

    rows: list[dict[str, object]] = []
    for _, row in history.iterrows():
        for component, column in zip(COMPONENTS, columns, strict=True):
            rows.append(
                {
                    "arm": arm,
                    "mode": mode,
                    "step": int(row["step"]),
                    "component": component,
                    "loss": float(row[column]),
                    "wandb_run_id": run_id,
                }
            )
    return pd.DataFrame(rows)


def validation_component_history(api: wandb.Api | None = None) -> pd.DataFrame:
    """Fetch and validate all five fixed component trajectories for all arms."""

    api = wandb.Api() if api is None else api
    frames = [
        _run_component_history(api, arm=arm, mode=mode, run_id=run_id)
        for arm, run_ids in ARM_RUNS.items()
        for mode in ARM_MODES[arm]
        for run_id in run_ids
    ]
    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["arm", "mode", "component", "step"])
        .reset_index(drop=True)
    )
    duplicated = combined.duplicated(["arm", "mode", "component", "step"])
    if duplicated.any():
        raise RuntimeError("component validation history contains duplicate arm/mode/step rows")
    for (arm, mode, component), cell in combined.groupby(
        ["arm", "mode", "component"],
        sort=False,
    ):
        observed = tuple(cell["step"].astype(int))
        if observed != EXPECTED_STEPS:
            raise RuntimeError(
                f"{arm}/{mode}/{component} validation steps are {observed}, "
                f"expected {EXPECTED_STEPS}"
            )
        if not cell["loss"].between(0, 10).all():
            raise RuntimeError(f"{arm}/{mode}/{component} has invalid validation loss")
    return combined


def plot_validation_components(frame: pd.DataFrame, output_path: Path) -> None:
    """Plot fixed validation loss for every component and training arm."""

    colors = {
        "transferred_mntp": "#E45756",
        "scratch_mntp": "#72B7B2",
        "clm_continuation": "#4C78A8",
    }
    labels = {
        "transferred_mntp": "Transferred MNTP",
        "scratch_mntp": "Scratch MNTP",
        "clm_continuation": "Continued CLM",
    }
    figure, axes = plt.subplots(
        2,
        len(COMPONENTS),
        figsize=(16, 7),
        sharex=True,
        constrained_layout=True,
    )
    for column, component in enumerate(COMPONENTS):
        for arm in ARM_RUNS:
            mode = "causal" if arm == "clm_continuation" else "diffusion"
            cell = frame[
                (frame["component"] == component) & (frame["arm"] == arm) & (frame["mode"] == mode)
            ].sort_values("step")
            axes[0, column].plot(
                cell["step"],
                cell["loss"],
                color=colors[arm],
                marker="o",
                markersize=3,
                linewidth=1.4,
                label=labels[arm],
            )
        for arm in ("transferred_mntp", "scratch_mntp"):
            cell = frame[
                (frame["component"] == component)
                & (frame["arm"] == arm)
                & (frame["mode"] == "single_mask")
            ].sort_values("step")
            axes[1, column].plot(
                cell["step"],
                cell["loss"],
                color=colors[arm],
                marker="o",
                markersize=3,
                linewidth=1.4,
                label=labels[arm],
            )
        axes[0, column].set_title(component.upper())
        axes[0, column].grid(alpha=0.25)
        axes[1, column].grid(alpha=0.25)
        axes[1, column].set_xlabel("Optimizer step")
        if column == 0:
            axes[0, column].set_ylabel("Diffusion / causal loss")
            axes[1, column].set_ylabel("Single-mask loss")
    axes[0, -1].legend(fontsize=7)
    axes[1, -1].legend(fontsize=7)
    figure.suptitle("Fixed 128-sequence validation loss by dataset component")
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_validation_report(output_dir: Path) -> None:
    """Write compact component history and matched static figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    history = validation_component_history()
    history.to_csv(output_dir / "validation-components.csv", index=False)
    plot_validation_components(history, output_dir / "validation-components")
