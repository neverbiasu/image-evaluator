import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from image_evaluator.pickscore_predictor import (
    PickScorePredictor,
    PickScoreResult,
    _discover_and_validate_images,
    _validate_image_file,
)


@pytest.fixture
def sample_image(tmp_path):
    img_path = tmp_path / "test_img.png"
    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def sample_folder(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for i in range(3):
        img_path = folder / f"img_{i}.png"
        img = Image.new("RGB", (32, 32), color=(i * 50, 100, 150))
        img.save(img_path)
    return str(folder)


# ---------------------------------------------------------------------------
# Mathematical Formula Verification (Pure, No Model Required)
# ---------------------------------------------------------------------------


def test_pickscore_math_identical_vectors_unit_scale():
    feat = torch.tensor([[1.0, 2.0, 3.0]])
    logit_scale = 0.0  # exp(0) = 1.0
    score = PickScorePredictor.compute_score_from_features(
        feat, feat, logit_scale
    )
    assert math.isclose(score, 1.0, rel_tol=1e-5)


def test_pickscore_math_orthogonal_vectors_zero():
    img_feat = torch.tensor([[1.0, 0.0, 0.0]])
    txt_feat = torch.tensor([[0.0, 1.0, 0.0]])
    logit_scale = 2.5
    score = PickScorePredictor.compute_score_from_features(
        img_feat, txt_feat, logit_scale
    )
    assert math.isclose(score, 0.0, abs_tol=1e-6)


def test_pickscore_math_opposite_vectors_negative_scale():
    img_feat = torch.tensor([[1.0, 0.0]])
    txt_feat = torch.tensor([[-1.0, 0.0]])
    logit_scale = 1.5
    score = PickScorePredictor.compute_score_from_features(
        img_feat, txt_feat, logit_scale
    )
    expected = -float(np.exp(1.5))
    assert math.isclose(score, expected, rel_tol=1e-5)


def test_pickscore_math_with_torch_tensor_logit_scale():
    img_feat = torch.tensor([3.0, 4.0])  # 1D tensor handling
    txt_feat = torch.tensor([3.0, 4.0])
    logit_scale = torch.tensor(2.0)
    score = PickScorePredictor.compute_score_from_features(
        img_feat, txt_feat, logit_scale
    )
    expected = float(torch.exp(logit_scale).item())
    assert math.isclose(score, expected, rel_tol=1e-5)


def test_pickscore_math_dimension_mismatch_raises_value_error():
    img_feat = torch.tensor([[1.0, 2.0]])
    txt_feat = torch.tensor([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        PickScorePredictor.compute_score_from_features(
            img_feat, txt_feat, logit_scale=1.0
        )


def test_pickscore_math_non_finite_raises_value_error():
    img_feat = torch.tensor([[float("nan"), 1.0]])
    txt_feat = torch.tensor([[1.0, 1.0]])
    with pytest.raises(ValueError, match="non-finite value"):
        PickScorePredictor.compute_score_from_features(
            img_feat, txt_feat, logit_scale=1.0
        )


# ---------------------------------------------------------------------------
# File & Input Validation Tests
# ---------------------------------------------------------------------------


def test_validate_image_file_valid(sample_image):
    _validate_image_file(sample_image)


def test_validate_image_file_non_existent(tmp_path):
    missing = str(tmp_path / "missing.png")
    with pytest.raises(FileNotFoundError, match="Image not found"):
        _validate_image_file(missing)


def test_validate_image_file_directory(tmp_path):
    folder = str(tmp_path)
    with pytest.raises(ValueError, match="Expected image file, got directory"):
        _validate_image_file(folder)


def test_validate_image_file_corrupted(tmp_path):
    corrupted = tmp_path / "corrupted.png"
    corrupted.write_bytes(b"not an image file content")
    with pytest.raises(ValueError, match="Cannot read or decode image"):
        _validate_image_file(str(corrupted))


def test_discover_images_valid(sample_folder):
    discovered = _discover_and_validate_images(sample_folder)
    assert len(discovered) == 3
    assert all(p.endswith(".png") for p in discovered)


def test_discover_images_non_existent_folder(tmp_path):
    missing = str(tmp_path / "no_folder")
    with pytest.raises(FileNotFoundError, match="Directory not found"):
        _discover_and_validate_images(missing)


def test_discover_images_path_is_file(sample_image):
    with pytest.raises(ValueError, match="Expected directory path, got file"):
        _discover_and_validate_images(sample_image)


def test_discover_images_empty_folder(tmp_path):
    empty = tmp_path / "empty_dir"
    empty.mkdir()
    with pytest.raises(ValueError, match="No supported image files found"):
        _discover_and_validate_images(str(empty))


def test_discover_images_corrupted_image_raises_error(tmp_path):
    folder = tmp_path / "corrupt_folder"
    folder.mkdir()
    (folder / "bad.jpg").write_bytes(b"bad content")
    with pytest.raises(ValueError, match="Cannot read or decode image"):
        _discover_and_validate_images(str(folder))


# ---------------------------------------------------------------------------
# PickScoreResult Container Tests
# ---------------------------------------------------------------------------


def test_pickscore_result_attributes():
    res = PickScoreResult(
        score=21.5,
        model="yuvalkirstain/PickScore_v1",
        processor="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        device="cpu",
        prompt="a scenic lake",
        sample_count=2,
        scores={"a.png": 21.0, "b.png": 22.0},
    )
    assert res.score == 21.5
    assert res.mean_score == 21.5
    assert res["pickscore"] == 21.5
    assert res.model == "yuvalkirstain/PickScore_v1"
    assert res.device == "cpu"
    assert res.prompt == "a scenic lake"
    assert res.sample_count == 2
    assert res.scores == {"a.png": 21.0, "b.png": 22.0}
    assert "PickScoreResult(score=21.5000" in repr(res)


# ---------------------------------------------------------------------------
# Predictor Execution with Mocked Model (Isolated, Zero-Download)
# ---------------------------------------------------------------------------


def test_predictor_device_explicit():
    pred = PickScorePredictor(device="cpu")
    assert pred.device == torch.device("cpu")


def test_predictor_empty_prompt_raises_value_error(sample_image):
    pred = PickScorePredictor(device="cpu")
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        pred.evaluate(sample_image, "")
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        pred.evaluate(sample_image, "   ")


def test_predictor_evaluate_single_image_mocked(sample_image):
    pred = PickScorePredictor(device="cpu")

    mock_model = MagicMock()
    # Normalize vector [1.0, 0.0] -> norm 1.0
    mock_model.get_image_features.return_value = torch.tensor([[1.0, 0.0]])
    mock_model.get_text_features.return_value = torch.tensor([[1.0, 0.0]])
    mock_model.logit_scale = torch.tensor(3.0)  # exp(3.0) ~= 20.0855

    mock_processor = MagicMock()
    mock_processor.return_value = {"input_ids": torch.tensor([[1, 2]])}

    with patch.object(
        pred, "_load_model", return_value=(mock_model, mock_processor)
    ):
        score = pred.evaluate(sample_image, "a red square")

    expected = float(torch.exp(torch.tensor(3.0)).item())
    assert math.isclose(score, expected, rel_tol=1e-5)


def test_predictor_evaluate_folder_mocked(sample_folder):
    pred = PickScorePredictor(device="cpu")

    mock_model = MagicMock()
    # Returns image vector that aligns with text vector
    mock_model.get_image_features.return_value = torch.tensor([[1.0, 0.0]])
    mock_model.get_text_features.return_value = torch.tensor([[1.0, 0.0]])
    mock_model.logit_scale = torch.tensor(2.0)  # exp(2.0) ~= 7.389

    mock_processor = MagicMock()
    mock_processor.return_value = {"dummy": torch.tensor([1])}

    with patch.object(
        pred, "_load_model", return_value=(mock_model, mock_processor)
    ):
        result = pred.evaluate_folder(sample_folder, "a test prompt")

    assert isinstance(result, PickScoreResult)
    assert result.sample_count == 3
    assert len(result.scores) == 3
    expected_score = float(torch.exp(torch.tensor(2.0)).item())
    assert math.isclose(result.mean_score, expected_score, rel_tol=1e-5)


def test_predictor_evaluate_pickscore_convenience_alias(
    sample_image, sample_folder
):
    pred = PickScorePredictor(device="cpu")

    with patch.object(pred, "evaluate", return_value=19.5) as mock_single:
        score_file = pred.evaluate_pickscore(sample_image, "prompt")
        assert score_file == 19.5
        mock_single.assert_called_once_with(sample_image, "prompt")

    mock_res = PickScoreResult(
        score=20.2,
        model="m",
        processor="p",
        device="cpu",
        prompt="prompt",
        sample_count=3,
    )
    with patch.object(
        pred, "evaluate_folder", return_value=mock_res
    ) as mock_folder:
        score_dir = pred.evaluate_pickscore(sample_folder, "prompt")
        assert score_dir == 20.2
        mock_folder.assert_called_once_with(sample_folder, "prompt")
