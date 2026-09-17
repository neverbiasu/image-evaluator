"""Tests for EvaluationResult data structure and evaluate_detailed() API."""

import json
from typing import Any

import pytest
from PIL import Image

from image_evaluator import EvaluationResult, evaluate, evaluate_detailed
from image_evaluator.registry import get_metric


@pytest.fixture
def dummy_image() -> Image.Image:
    """Create a 32x32 solid RGB PIL Image fixture."""
    return Image.new("RGB", (32, 32), color="blue")


def test_evaluation_result_attributes() -> None:
    """Verify attributes and immutability of EvaluationResult."""
    spec_ssim = get_metric("ssim")
    res = EvaluationResult(
        scores={"ssim": 0.95},
        specs={"ssim": spec_ssim},
        inputs={"image_type": "Image", "prompt": None},
        duration_seconds=0.1234,
    )

    assert res.scores == {"ssim": 0.95}
    assert res.specs["ssim"] == spec_ssim
    assert res.inputs["image_type"] == "Image"
    assert res.duration_seconds == 0.1234

    # Frozen immutability
    with pytest.raises(AttributeError):
        res.duration_seconds = 1.0  # type: ignore[misc]


def test_evaluation_result_detaches_and_freezes_public_mappings() -> None:
    """Inputs are deeply detached and public mappings reject mutation."""
    scores = {"fid": {"score": 4.2, "counts": [10, 12]}}
    specs = {"fid": get_metric("fid")}
    inputs = {"paths": {"generated": ["a", "b"]}}
    res = EvaluationResult(scores=scores, specs=specs, inputs=inputs)

    scores["fid"]["score"] = 99.0
    specs.clear()
    inputs["paths"]["generated"].append("c")

    assert res.scores["fid"]["score"] == 4.2
    assert res.specs["fid"].id == "fid"
    assert res.inputs["paths"]["generated"] == ("a", "b")
    with pytest.raises(TypeError):
        res.scores["fid"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        res.scores["fid"]["score"] = 0.0  # type: ignore[index]
    with pytest.raises(TypeError):
        res.specs["fid"] = get_metric("fid")  # type: ignore[index]
    with pytest.raises(TypeError):
        res.inputs["paths"] = {}  # type: ignore[index]


def test_evaluation_result_mapping_protocol() -> None:
    """Verify Mapping protocol (dict-like access) on EvaluationResult."""
    spec_ssim = get_metric("ssim")
    spec_psnr = get_metric("psnr")
    res = EvaluationResult(
        scores={"ssim": 0.95, "psnr": 34.5},
        specs={"ssim": spec_ssim, "psnr": spec_psnr},
    )

    assert res["ssim"] == 0.95
    assert res["psnr"] == 34.5
    assert "ssim" in res
    assert "psnr" in res
    assert "lpips" not in res
    assert len(res) == 2
    assert sorted(list(res)) == ["psnr", "ssim"]
    assert res.get("ssim") == 0.95
    assert res.get("unknown", 99.9) == 99.9

    items = dict(res.items())
    assert items == {"ssim": 0.95, "psnr": 34.5}


def test_evaluation_result_to_dict() -> None:
    """Verify to_dict() returns an isolated dict matching evaluate() shape."""
    res = EvaluationResult(
        scores={"ssim": 0.95},
        specs={"ssim": get_metric("ssim")},
    )
    d = res.to_dict()
    assert d == {"ssim": 0.95}
    assert isinstance(d, dict)

    # Mutation safety
    d["ssim"] = 0.0
    assert res["ssim"] == 0.95


def test_evaluation_result_to_json() -> None:
    """Verify RFC 8259 JSON serialization including nulls for non-finites."""
    res = EvaluationResult(
        scores={
            "ssim": 0.95,
            "psnr_inf": float("inf"),
            "nan_val": float("nan"),
        },
        specs={"ssim": get_metric("ssim")},
        inputs={"device": "cpu"},
        duration_seconds=0.5,
    )

    json_str = res.to_json(indent=2)
    parsed: dict[str, Any] = json.loads(json_str)

    assert parsed["scores"]["ssim"] == 0.95
    assert parsed["scores"]["psnr_inf"] is None
    assert parsed["scores"]["nan_val"] is None
    assert "ssim" in parsed["specs"]
    assert parsed["specs"]["ssim"]["id"] == "ssim"
    assert parsed["specs"]["ssim"]["score_direction"] == "higher_is_better"
    assert parsed["inputs"]["device"] == "cpu"
    assert parsed["duration_seconds"] == 0.5


def test_evaluation_result_validation_errors() -> None:
    """Verify defensive validation on invalid types or negative duration."""
    with pytest.raises(ValueError, match="Metric name must be a str"):
        EvaluationResult(scores={123: 1.0}, specs={})  # type: ignore[dict-item]

    with pytest.raises(
        ValueError,
        match="duration_seconds must be non-negative",
    ):
        EvaluationResult(scores={"ssim": 1.0}, specs={}, duration_seconds=-0.5)


def test_evaluation_result_repr() -> None:
    """Verify human-readable repr string."""
    res = EvaluationResult(
        scores={"ssim": 0.95231, "psnr": float("inf")},
        specs={},
    )
    rep = repr(res)
    assert "EvaluationResult" in rep
    assert "ssim=0.9523" in rep
    assert "psnr=inf" in rep


def test_evaluate_detailed_flag(dummy_image: Image.Image) -> None:
    """Verify evaluate(..., detailed=True) returns EvaluationResult."""
    # Default is dict
    dict_res = evaluate(
        image=dummy_image,
        reference=dummy_image,
        metrics="ssim",
    )
    assert isinstance(dict_res, dict)
    assert not isinstance(dict_res, EvaluationResult)
    assert "ssim" in dict_res

    # detailed=False is dict
    dict_res2 = evaluate(
        image=dummy_image,
        reference=dummy_image,
        metrics="ssim",
        detailed=False,
    )
    assert isinstance(dict_res2, dict)
    assert not isinstance(dict_res2, EvaluationResult)

    # detailed=True is EvaluationResult
    detailed_res = evaluate(
        image=dummy_image,
        reference=dummy_image,
        metrics="ssim",
        detailed=True,
    )
    assert isinstance(detailed_res, EvaluationResult)
    assert detailed_res["ssim"] == 1.0
    assert "ssim" in detailed_res.specs
    assert detailed_res.specs["ssim"].id == "ssim"
    assert detailed_res.inputs["image_type"] == "Image"
    assert detailed_res.duration_seconds >= 0.0


def test_evaluate_detailed_helper(dummy_image: Image.Image) -> None:
    """Verify evaluate_detailed() helper function behavior."""
    res = evaluate_detailed(
        image=dummy_image,
        reference=dummy_image,
        metrics=["ssim", "psnr"],
    )
    assert isinstance(res, EvaluationResult)
    assert "ssim" in res
    assert "psnr" in res
    assert res.specs["ssim"].score_direction == "higher_is_better"
    assert res.specs["psnr"].score_direction == "higher_is_better"
    assert res.inputs["reference_type"] == "Image"


@pytest.mark.parametrize(
    ("metric_id", "predictor_path", "method_name", "payload"),
    [
        (
            "fid",
            "image_evaluator.fid_predictor.FIDPredictor",
            "evaluate_folder_fid",
            {"fid": 3.5, "score": 3.5, "backend": "clean-fid"},
        ),
        (
            "kid",
            "image_evaluator.kid_predictor.KIDPredictor",
            "evaluate_folder_kid",
            {"kid": -0.01, "score": -0.01, "seed": 0},
        ),
    ],
)
def test_evaluate_detailed_preserves_distribution_result_mappings(
    metric_id: str,
    predictor_path: str,
    method_name: str,
    payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Detailed FID/KID results retain existing structured return values."""
    module_name, class_name = predictor_path.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    predictor = getattr(module, class_name)
    monkeypatch.setattr(predictor, method_name, lambda *args: payload)
    generated = tmp_path / "generated"
    reference = tmp_path / "reference"
    generated.mkdir()
    reference.mkdir()

    result = evaluate_detailed(
        metrics=metric_id,
        image=generated,
        reference=reference,
    )

    assert result.to_dict()[metric_id] == payload
    assert json.loads(result.to_json())["scores"][metric_id] == payload


def test_top_level_import_exports() -> None:
    """Verify EvaluationResult can be imported from top level."""
    from image_evaluator import (
        EvaluationResult as TopRes,
        evaluate_detailed as top_evaluate_detailed,
    )

    assert TopRes is EvaluationResult
    assert top_evaluate_detailed is evaluate_detailed
