from unittest.mock import MagicMock, patch

import pytest
from PIL import UnidentifiedImageError

from image_evaluator.laion_ai_aesthetic_predictor import (
    LaionAIAestheticPredictor,
)


def test_predictor_loads_open_clip_once_and_encodes_each_image():
    """A predictor should reuse one OpenCLIP model for multiple images."""
    clip_model = MagicMock()
    preprocess = MagicMock()
    image_tensor = MagicMock()
    preprocess.return_value.unsqueeze.return_value.to.return_value = (
        image_tensor
    )

    image_features = MagicMock()
    image_features.norm.return_value = MagicMock()
    clip_model.encode_image.return_value = image_features

    score_value = MagicMock()
    score_value.item.return_value = 6.12
    aesthetic_model = MagicMock(return_value=[[score_value]])

    image = MagicMock()
    image.convert.return_value = image

    with patch.object(
        LaionAIAestheticPredictor,
        "get_aesthetic_model",
        return_value=aesthetic_model,
    ), patch(
        "image_evaluator.laion_ai_aesthetic_predictor."
        "open_clip.create_model_and_transforms",
        return_value=(clip_model, None, preprocess),
    ) as create_model, patch(
        "image_evaluator.laion_ai_aesthetic_predictor.Image.open",
        return_value=image,
    ):
        predictor = LaionAIAestheticPredictor()
        first_score = predictor.evaluate_aesthetic_score("first.png")
        second_score = predictor.evaluate_aesthetic_score("second.png")

    create_model.assert_called_once_with(
        "ViT-L-14", pretrained="openai", force_quick_gelu=True
    )
    clip_model.to.assert_called_once_with(predictor.device)
    clip_model.eval.assert_called_once_with()
    assert clip_model.encode_image.call_count == 2
    assert preprocess.call_count == 2
    assert first_score == 6.12
    assert second_score == 6.12


def test_predictor_initialization_passes_force_quick_gelu():
    """Predictor must explicitly set force_quick_gelu=True for OpenCLIP."""
    with patch.object(
        LaionAIAestheticPredictor,
        "get_aesthetic_model",
        return_value=MagicMock(),
    ), patch(
        "image_evaluator.laion_ai_aesthetic_predictor."
        "open_clip.create_model_and_transforms",
        return_value=(MagicMock(), None, MagicMock()),
    ) as create_model:
        LaionAIAestheticPredictor()

    create_model.assert_called_once_with(
        "ViT-L-14", pretrained="openai", force_quick_gelu=True
    )


def test_evaluate_returns_none_for_invalid_image():
    with patch.object(
        LaionAIAestheticPredictor,
        "get_aesthetic_model",
        return_value=MagicMock(),
    ), patch(
        "image_evaluator.laion_ai_aesthetic_predictor."
        "open_clip.create_model_and_transforms",
        return_value=(MagicMock(), None, MagicMock()),
    ), patch(
        "image_evaluator.laion_ai_aesthetic_predictor.Image.open",
        side_effect=UnidentifiedImageError("invalid image"),
    ):
        predictor = LaionAIAestheticPredictor()
        assert predictor.evaluate_aesthetic_score("invalid.png") is None


def test_folder_score_filters_extensions_and_averages(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "first.png").touch()
    (image_dir / "second.jpg").touch()
    (image_dir / "ignored.txt").touch()

    predictor = LaionAIAestheticPredictor.__new__(
        LaionAIAestheticPredictor
    )
    with patch.object(
        predictor, "evaluate_aesthetic_score", side_effect=[6.0, 8.0]
    ) as evaluate:
        result = predictor.evaluate_folder_aesthetic_score(str(image_dir))

    assert result == pytest.approx(7.0)
    assert evaluate.call_count == 2

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert predictor.evaluate_folder_aesthetic_score(str(empty_dir)) is None

    with pytest.raises(ValueError, match="not a valid folder"):
        predictor.evaluate_folder_aesthetic_score(str(tmp_path / "missing"))
