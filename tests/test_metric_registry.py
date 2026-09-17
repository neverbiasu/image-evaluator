import subprocess
import sys
from pathlib import Path

import pytest

from image_evaluator.registry import (
    METRIC_REGISTRY,
    DuplicateMetricError,
    MetricRegistry,
    UnknownMetricError,
    filter_metrics,
    get_metric,
    list_metrics,
)
from image_evaluator.specifications import (
    ImplementationRef,
    InputContract,
    MetricSpec,
)

EXPECTED_METRICS = {
    "aesthetic",
    "arcface",
    "clip",
    "directional_clip",
    "fid",
    "kid",
    "lpips",
    "pickscore",
    "psnr",
    "ssim",
}


def _spec(metric_id: str) -> MetricSpec:
    return MetricSpec(
        id=metric_id,
        display_name=metric_id,
        tasks=("custom_task",),
        objectives=("custom_objective",),
        inputs=InputContract(required=("image",)),
        score_direction="higher_is_better",
        implementation=ImplementationRef(
            backend="test", protocol="test-protocol"
        ),
    )


def test_catalog_contains_current_ten_metrics_in_stable_order():
    metrics = list_metrics()

    assert len(metrics) == 10
    assert {metric.id for metric in metrics} == EXPECTED_METRICS
    assert [metric.id for metric in metrics] == sorted(EXPECTED_METRICS)


def test_directional_clip_has_approved_four_input_contract():
    metric = get_metric("directional_clip")

    assert metric.inputs.required == (
        "image",
        "reference_image",
        "prompt",
        "source_prompt",
    )
    assert metric.tasks == ("image_editing",)
    assert metric.score_direction == "higher_is_better"


def test_specs_are_immutable():
    metric = get_metric("clip")

    with pytest.raises(AttributeError):
        metric.id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        METRIC_REGISTRY._metrics["changed"] = metric  # type: ignore[index]


def test_mutable_iterables_are_defensively_normalized():
    required = ["image"]
    optional = ["prompt"]
    tasks = ["custom_task"]
    objectives = ["custom_objective"]
    aggregation = ["mean"]
    dependencies = ["test-backend"]
    citations = ["https://example.invalid/reference"]

    contract = InputContract(  # type: ignore[arg-type]
        required=required,
        optional=optional,
    )
    metric = MetricSpec(  # type: ignore[arg-type]
        id="normalized",
        display_name="Normalized",
        tasks=tasks,
        objectives=objectives,
        inputs=contract,
        score_direction="higher_is_better",
        implementation=ImplementationRef(
            backend="test", protocol="test-protocol"
        ),
        aggregation=aggregation,
        dependencies=dependencies,
        citations=citations,
    )

    required.append("reference_image")
    optional.append("mask")
    tasks.append("later_task")
    objectives.append("later_objective")
    aggregation.append("median")
    dependencies.append("later-backend")
    citations.append("https://example.invalid/later")

    assert contract.required == ("image",)
    assert contract.optional == ("prompt",)
    assert metric.tasks == ("custom_task",)
    assert metric.objectives == ("custom_objective",)
    assert metric.aggregation == ("mean",)
    assert metric.dependencies == ("test-backend",)
    assert metric.citations == ("https://example.invalid/reference",)


def test_invalid_score_direction_is_rejected_at_runtime():
    with pytest.raises(ValueError, match="score_direction must be"):
        MetricSpec(
            id="invalid_direction",
            display_name="Invalid Direction",
            tasks=("custom_task",),
            objectives=("custom_objective",),
            inputs=InputContract(required=("image",)),
            score_direction="sideways",  # type: ignore[arg-type]
            implementation=ImplementationRef(
                backend="test", protocol="test-protocol"
            ),
        )


def test_duplicate_ids_raise_explicit_error():
    with pytest.raises(
        DuplicateMetricError,
        match="Metric id 'duplicate' is already registered",
    ):
        MetricRegistry((_spec("duplicate"), _spec("duplicate")))


def test_unknown_id_raises_explicit_error():
    with pytest.raises(
        UnknownMetricError,
        match="Unknown metric id 'missing'",
    ):
        get_metric("missing")


def test_filter_uses_open_task_and_objective_terms():
    editing = {metric.id for metric in filter_metrics(task="image_editing")}
    preference = {
        metric.id for metric in filter_metrics(objective="human_preference")
    }
    combined = filter_metrics(
        task="image_editing", objective="edit_direction_alignment"
    )

    assert {"aesthetic", "clip", "directional_clip", "lpips"} <= editing
    assert preference == {"pickscore"}
    assert tuple(metric.id for metric in combined) == ("directional_clip",)
    assert filter_metrics(task="future_unregistered_task") == ()


def test_traceability_fields_use_confirmed_registry_values():
    fid = get_metric("fid")
    clip = get_metric("clip")

    assert fid.implementation.backend_version == "0.1.35"
    assert fid.implementation.model == "inception_v3"
    assert fid.implementation.model_revision is None
    assert fid.dependencies == ("clean-fid",)
    assert fid.docs_path == "docs/fid.md"
    assert clip.implementation.backend_version is None
    assert clip.implementation.model_revision is None
    assert clip.citations == (
        "https://arxiv.org/abs/2104.08718",
        "https://github.com/Taited/clip-score",
    )


def test_custom_open_vocabulary_terms_require_no_enum_change():
    custom = _spec("custom_metric")
    registry = MetricRegistry((custom,))

    assert registry.filter(task="custom_task") == (custom,)
    assert registry.filter(objective="custom_objective") == (custom,)


def test_input_contract_rejects_overlapping_roles():
    with pytest.raises(
        ValueError, match="cannot be both required and optional"
    ):
        InputContract(required=("image",), optional=("image",))


def test_registry_import_does_not_load_heavy_backends():
    script = """
import importlib
import sys

blocked = ("torch", "transformers", "open_clip", "clip", "cleanfid", "lpips")
before = set(sys.modules)
importlib.import_module("image_evaluator.registry")
loaded = set(sys.modules) - before
unexpected = sorted(
    name for name in loaded
    if any(
        name == prefix or name.startswith(prefix + ".")
        for prefix in blocked
    )
)
if unexpected:
    raise SystemExit("heavy modules loaded: " + ", ".join(unexpected))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_registry_module_can_be_reloaded_without_backend_imports():
    script = """
import importlib
import image_evaluator.registry as registry

assert importlib.reload(registry).get_metric("ssim").id == "ssim"
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_all_registered_metrics_have_existing_docs_paths():
    repo_root = Path(__file__).resolve().parent.parent
    for metric in list_metrics():
        assert metric.docs_path is not None, (
            f"Metric '{metric.id}' has no docs_path"
        )
        doc_file = repo_root / metric.docs_path
        assert doc_file.is_file(), (
            f"Metric '{metric.id}' docs_path "
            f"'{metric.docs_path}' does not exist"
        )
