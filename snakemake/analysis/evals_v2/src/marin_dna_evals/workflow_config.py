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
    """Validate global inference policy and documented fp32 fallbacks."""
    for field in sorted(GLOBAL_INFERENCE_SWITCHES - {"bf16"}):
        if inference.get(field) is not True:
            raise ValueError(f"inference.{field} must be globally set to true")
    if type(inference.get("bf16")) is not bool:
        raise ValueError("inference.bf16 must be a boolean")
    tf32 = inference.get("tf32")
    if tf32 is not None and type(tf32) is not bool:
        raise ValueError("inference.tf32 must be a boolean or null")
    if inference["bf16"] is False:
        reason = inference.get("precision_reason")
        if tf32 is not False or not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                "fp32 fallback requires inference.tf32=false and a precision_reason"
            )
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
        forbidden = sorted(
            (GLOBAL_INFERENCE_SWITCHES | {"tf32", "precision_reason"}).intersection(
                model
            )
        )
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


def inference_precision_params(inference: Mapping[str, object]) -> dict[str, object]:
    """Add provenance only for an explicitly selected precision override.

    Existing default runs must retain their original Snakemake parameter map;
    adding even a default-valued key would invalidate completed score outputs.
    """
    precision = {
        key: inference.get(key) for key in ("bf16", "tf32", "precision_reason")
    }
    if precision == {"bf16": True, "tf32": None, "precision_reason": None}:
        return {}
    return {"precision": precision}
