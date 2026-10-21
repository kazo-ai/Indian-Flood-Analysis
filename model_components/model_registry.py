"""
Central model registry for the flood susceptibility framework.

The registry uses the stable M01-M25 identifiers defined in config.py.

This module manages model metadata. Actual model construction is handled
by models.py and the family-specific implementation modules.

It deliberately distinguishes model registration from model availability:
a model can be registered while its required spatial or temporal data
is unavailable.
"""

from __future__ import annotations

from typing import Any, Iterable

import config as C


# ============================================================
# MODEL FAMILIES
# ============================================================

MODEL_FAMILIES = {
    "traditional_ml",
    "hybrid_ml",
    "tabular_dl",
    "spatial_dl",
    "temporal_dl",
}


# ============================================================
# TASK TYPES
# ============================================================

SUPPORTED_TASKS = {
    "static",
    "forecasting",
}


# ============================================================
# REGISTRY ACCESS
# ============================================================

def list_models() -> list[dict[str, Any]]:
    """
    Return metadata for every registered model.

    Returns
    -------
    list[dict]
        One metadata dictionary per registered model.
    """

    result = []

    for model_id, spec in C.MODEL_REGISTRY.items():
        result.append({
            "id": model_id,
            **spec,
        })

    return result


def get_model_spec(model_id: str) -> dict[str, Any]:
    """
    Return the metadata for one model.

    Parameters
    ----------
    model_id:
        Stable identifier, such as M01 or M14.

    Raises
    ------
    KeyError
        If the requested ID is not registered.
    """

    model_id = model_id.strip().upper()

    if model_id not in C.MODEL_REGISTRY:
        available = ", ".join(C.MODEL_REGISTRY)

        raise KeyError(
            f"Unknown model ID '{model_id}'. "
            f"Available IDs: {available}"
        )

    return {
        "id": model_id,
        **C.MODEL_REGISTRY[model_id],
    }


def list_models_by_family(
    family: str,
) -> list[dict[str, Any]]:
    """
    Return models belonging to a specified family.

    Examples
    --------
    list_models_by_family("traditional_ml")
    list_models_by_family("tabular_dl")
    list_models_by_family("spatial_dl")
    list_models_by_family("temporal_dl")
    """

    family = family.strip().lower()

    if family not in MODEL_FAMILIES:
        raise ValueError(
            f"Unknown model family '{family}'. "
            f"Supported families: {sorted(MODEL_FAMILIES)}"
        )

    return [
        model
        for model in list_models()
        if model["family"] == family
    ]


def list_models_by_task(
    task: str,
) -> list[dict[str, Any]]:
    """
    Return models for a specified prediction task.

    Supported tasks
    ---------------
    static:
        Static flood susceptibility classification.

    forecasting:
        Time-dependent flood forecasting.
    """

    task = task.strip().lower()

    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unknown task '{task}'. "
            f"Supported tasks: {sorted(SUPPORTED_TASKS)}"
        )

    return [
        model
        for model in list_models()
        if model["task"] == task
    ]


def validate_model_ids(
    model_ids: Iterable[str],
) -> list[str]:
    """
    Validate model IDs and return their normalised identifiers.

    Duplicate IDs are removed while preserving order.
    """

    validated = []
    seen = set()

    for raw_id in model_ids:
        model_id = str(raw_id).strip().upper()

        if model_id not in C.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model ID '{model_id}'. "
                f"Valid IDs: {list(C.MODEL_REGISTRY)}"
            )

        if model_id not in seen:
            validated.append(model_id)
            seen.add(model_id)

    if not validated:
        raise ValueError(
            "No model IDs were provided."
        )

    return validated


# ============================================================
# INPUT REQUIREMENTS
# ============================================================

def check_data_requirements(
    model_id: str,
    *,
    has_spatial_coordinates: bool,
    has_spatial_graph: bool = False,
    has_raster_patches: bool = False,
    has_temporal_observations: bool = False,
    has_timestamp: bool = False,
    has_repeated_observations: bool = False,
) -> tuple[bool, list[str]]:
    """
    Check whether the declared data inputs are sufficient for a model.

    This checks input availability, not scientific validity or model
    performance. The training pipeline must perform additional checks.

    Returns
    -------
    tuple[bool, list[str]]
        Eligibility and a list of reasons if the model is ineligible.
    """

    spec = get_model_spec(model_id)

    reasons = []

    if spec["requires_spatial_data"]:
        if not has_spatial_coordinates:
            reasons.append(
                "Spatial coordinates are required."
            )

        if spec["id"] in {"M17", "M18", "M19", "M20"}:
            if not has_spatial_graph:
                reasons.append(
                    "A valid spatial graph is required."
                )

        if spec["id"] in {"M15", "M20"}:
            if not has_raster_patches:
                reasons.append(
                    "Structured raster patches are required."
                )

    if spec["requires_temporal_data"]:
        if not has_temporal_observations:
            reasons.append(
                "Time-indexed predictor observations are required."
            )

        if not has_timestamp:
            reasons.append(
                "A timestamp column is required."
            )

        if not has_repeated_observations:
            reasons.append(
                "Repeated observations are required to construct sequences."
            )

    return len(reasons) == 0, reasons


# ============================================================
# REGISTRY VALIDATION
# ============================================================

def validate_registry() -> dict[str, Any]:
    """
    Validate the registry structure and return a diagnostic report.

    This verifies metadata consistency. It does not instantiate or train
    model architectures.
    """

    errors = []
    warnings = []

    registry = C.MODEL_REGISTRY

    expected_ids = {
        f"M{i:02d}"
        for i in range(1, 26)
    }

    actual_ids = set(registry)

    missing_ids = sorted(expected_ids - actual_ids)
    unexpected_ids = sorted(actual_ids - expected_ids)

    if missing_ids:
        errors.append(
            f"Missing model IDs: {missing_ids}"
        )

    if unexpected_ids:
        errors.append(
            f"Unexpected model IDs: {unexpected_ids}"
        )

    for model_id, spec in registry.items():
        required_keys = {
            "name",
            "family",
            "task",
            "requires_spatial_data",
            "requires_temporal_data",
        }

        missing_keys = required_keys - set(spec)

        if missing_keys:
            errors.append(
                f"{model_id}: missing metadata keys "
                f"{sorted(missing_keys)}"
            )
            continue

        if spec["family"] not in MODEL_FAMILIES:
            errors.append(
                f"{model_id}: invalid family "
                f"'{spec['family']}'"
            )

        if spec["task"] not in SUPPORTED_TASKS:
            errors.append(
                f"{model_id}: invalid task "
                f"'{spec['task']}'"
            )

        if (
            spec["requires_temporal_data"]
            and spec["task"] != "forecasting"
        ):
            warnings.append(
                f"{model_id}: temporal inputs are required, "
                "but the task is not marked as forecasting."
            )

    return {
        "valid": len(errors) == 0,
        "registered_count": len(registry),
        "errors": errors,
        "warnings": warnings,
    }


# ============================================================
# COMMAND-LINE DIAGNOSTICS
# ============================================================

if __name__ == "__main__":
    report = validate_registry()

    print("=" * 65)
    print("FLOOD SUSCEPTIBILITY MODEL REGISTRY")
    print("=" * 65)

    print(f"Registered models: {report['registered_count']}")
    print(f"Registry valid: {report['valid']}")

    print("\nModels by family:")

    for family in sorted(MODEL_FAMILIES):
        models = list_models_by_family(family)

        print(f"\n{family} ({len(models)} models)")

        for model in models:
            print(
                f"  {model['id']}: {model['name']}"
            )

    if report["warnings"]:
        print("\nWarnings:")

        for warning in report["warnings"]:
            print(f"  - {warning}")

    if report["errors"]:
        print("\nErrors:")

        for error in report["errors"]:
            print(f"  - {error}")

        raise SystemExit(1)

    print("\nRegistry validation passed.")