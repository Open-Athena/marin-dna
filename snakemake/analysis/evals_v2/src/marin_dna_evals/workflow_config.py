"""Validation and execution-setting resolution for the evals v2 workflow."""

from collections.abc import Mapping, Sequence

GLOBAL_INFERENCE_SWITCHES = frozenset({"return_embeddings", "torch_compile", "bf16"})


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    return value


def validate_inference_config(
    inference: Mapping[str, object], models: Sequence[Mapping[str, object]]
) -> None:
    """Require global-on semantic switches and reject checkpoint overrides."""
    for field in sorted(GLOBAL_INFERENCE_SWITCHES):
        if inference.get(field) is not True:
            raise ValueError(f"inference.{field} must be globally set to true")
    if inference.get("rc") is not True:
        raise ValueError("inference.return_embeddings=true requires inference.rc=true")

    _positive_int(inference.get("batch_size"), field="inference.batch_size")
    global_accumulation = inference.get("eval_accumulation_steps")
    if global_accumulation is not None:
        _positive_int(
            global_accumulation,
            field="inference.eval_accumulation_steps",
        )

    for model in models:
        model_name = str(model.get("name", "<unnamed>"))
        forbidden = sorted(GLOBAL_INFERENCE_SWITCHES.intersection(model))
        if forbidden:
            raise ValueError(
                f"model {model_name!r} cannot override global inference switches: "
                f"{forbidden}"
            )
        if "batch_size" in model:
            _positive_int(
                model["batch_size"],
                field=f"model {model_name!r} batch_size",
            )
        if model.get("eval_accumulation_steps") is not None:
            _positive_int(
                model["eval_accumulation_steps"],
                field=f"model {model_name!r} eval_accumulation_steps",
            )


def resolve_model_batch_size(
    model: Mapping[str, object], inference: Mapping[str, object]
) -> int:
    """Resolve a checkpoint batch size with the global value as fallback."""
    model_name = str(model.get("name", "<unnamed>"))
    return _positive_int(
        model.get("batch_size", inference.get("batch_size")),
        field=f"model {model_name!r} batch_size",
    )


def resolve_model_eval_accumulation_steps(
    model: Mapping[str, object], inference: Mapping[str, object]
) -> int | None:
    """Resolve checkpoint prediction-offload cadence with a global fallback."""
    value = model.get(
        "eval_accumulation_steps",
        inference.get("eval_accumulation_steps"),
    )
    if value is None:
        return None
    model_name = str(model.get("name", "<unnamed>"))
    return _positive_int(
        value,
        field=f"model {model_name!r} eval_accumulation_steps",
    )
