"""Tests for ImageReward predictor and evaluation pipeline integration."""

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

import image_evaluator
from image_evaluator.image_reward_predictor import (
    ImageRewardPredictor,
)
from image_evaluator.main import main
from image_evaluator.model_assets import DownloadNotAllowedError


class DummyImageRewardModel(torch.nn.Module):
    """Dummy model returning controlled scalar reward values."""

    def __init__(self, fixed_score: float = 0.5) -> None:
        super().__init__()
        self.fixed_score = fixed_score

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = image.shape[0]
        if input_ids.shape[1] > 0 and input_ids[0, 0].item() == 999:
            return torch.full((batch_size, 1), -1.25)
        return torch.full((batch_size, 1), self.fixed_score)


@dataclass
class _DummyTokenizerOutput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class _DummyTokenizer:
    def __call__(self, prompt: str, **kwargs) -> _DummyTokenizerOutput:
        if "negative" in prompt:
            ids = torch.tensor([[999, 1, 2]])
        else:
            ids = torch.tensor([[100, 1, 2]])
        return _DummyTokenizerOutput(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
        )


@pytest.fixture
def sample_image(tmp_path):
    """Create a temporary test image file."""
    img_path = tmp_path / "test_sample.png"
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def sample_folder(tmp_path):
    """Create a temporary folder with multiple valid images."""
    folder = tmp_path / "img_dir"
    folder.mkdir()
    for i in range(2):
        img = Image.new("RGB", (32, 32), color=(i * 50, 100, 150))
        img.save(folder / f"img_{i:02d}.png")
    return str(folder)


def _mock_image_reward_predictor_loaded(
    predictor: ImageRewardPredictor, fixed_score: float = 0.5
) -> None:
    """Inject dummy model, tokenizer, and preprocess."""
    predictor._model = DummyImageRewardModel(fixed_score=fixed_score)
    predictor._preprocess = lambda pil_img: torch.zeros(3, 224, 224)
    predictor._tokenizer = _DummyTokenizer()


def test_package_exports():
    """Verify ImageRewardPredictor is exported in top-level package."""
    assert hasattr(image_evaluator, "ImageRewardPredictor")
    assert "ImageRewardPredictor" in dir(image_evaluator)


def test_download_gating_uncached_raises_error(sample_image):
    """Verify that an uncached model without --allow-download raises error."""
    with patch(
        "image_evaluator.image_reward_predictor._is_image_reward_cached",
        return_value=False,
    ):
        predictor = ImageRewardPredictor(allow_download=False)
        with pytest.raises(DownloadNotAllowedError) as exc_info:
            predictor.compute_image_reward(sample_image, "a photo of a cat")

        msg = str(exc_info.value)
        assert "THUDM/ImageReward" in msg
        assert "image_reward" in msg
        assert "--allow-download" in msg
        assert "image-evaluator[preference]" in msg


def test_download_gating_disclosure_called():
    """Verify download disclosure callback is triggered on download."""
    from image_evaluator.image_reward_predictor import IMAGE_REWARD_ASSET
    from image_evaluator.model_assets import check_asset_and_permit_download

    disclosures = []

    def callback(asset, message):
        disclosures.append((asset, message))

    check_asset_and_permit_download(
        asset=IMAGE_REWARD_ASSET,
        is_cached=False,
        allow_download=True,
        disclosure_callback=callback,
    )
    assert len(disclosures) == 1
    asset, msg = disclosures[0]
    assert asset.metric_id == "image_reward"
    assert "THUDM/ImageReward" in msg
    assert "1786880927" in msg or "1.8 GB" in msg or "1.7 GB" in msg


def test_scalar_reward_math(sample_image):
    """Verify scalar reward outputs positive and negative values."""
    predictor = ImageRewardPredictor(allow_download=True)
    _mock_image_reward_predictor_loaded(predictor, fixed_score=0.75)

    pos_score = predictor.compute_image_reward(sample_image, "a positive cat")
    assert pos_score == pytest.approx(0.75, abs=1e-4)

    neg_score = predictor.compute_image_reward(sample_image, "a negative cat")
    assert neg_score == pytest.approx(-1.25, abs=1e-4)


def test_input_formats(tmp_path, sample_image):
    """Verify support for str path, Path, PIL, numpy, and Tensor inputs."""
    predictor = ImageRewardPredictor(allow_download=True)
    _mock_image_reward_predictor_loaded(predictor)

    prompt = "a cute puppy"

    # 1. str path
    score1 = predictor.compute_image_reward(sample_image, prompt)
    assert isinstance(score1, float)

    # 2. Path object
    score2 = predictor.compute_image_reward(Path(sample_image), prompt)
    assert score2 == score1

    # 3. PIL.Image
    with Image.open(sample_image) as img:
        score3 = predictor.compute_image_reward(img, prompt)
    assert score3 == score1

    # 4. numpy ndarray
    arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    score4 = predictor.compute_image_reward(arr, prompt)
    assert isinstance(score4, float)

    # 5. torch.Tensor
    tensor_img = torch.rand(3, 64, 64)
    score5 = predictor.compute_image_reward(tensor_img, prompt)
    assert isinstance(score5, float)


def test_invalid_inputs(sample_image):
    """Verify clear exceptions for invalid image or prompt inputs."""
    predictor = ImageRewardPredictor(allow_download=True)
    _mock_image_reward_predictor_loaded(predictor)

    # Non-existent file
    with pytest.raises(FileNotFoundError, match="Image not found"):
        predictor.compute_image_reward(
            "non_existent_image.png", "valid prompt"
        )

    # Empty prompt
    with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
        predictor.compute_image_reward(sample_image, "   ")

    # Non-string prompt
    with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
        predictor.compute_image_reward(sample_image, None)  # type: ignore

    # Unsupported image type
    with pytest.raises(TypeError, match="Unsupported image input type"):
        predictor.compute_image_reward(12345, "valid prompt")


def test_evaluate_folder_image_reward(sample_folder):
    """Verify folder arithmetic mean evaluation across multiple images."""
    predictor = ImageRewardPredictor(allow_download=True)
    _mock_image_reward_predictor_loaded(predictor, fixed_score=0.42)

    mean_score = predictor.evaluate_folder_image_reward(
        sample_folder, "a nice landscape"
    )
    assert mean_score == pytest.approx(0.42, abs=1e-4)


def test_core_evaluate_single_image(sample_image):
    """Verify top-level evaluate() dispatches to image_reward."""
    with patch(
        "image_evaluator.image_reward_predictor.ImageRewardPredictor.compute_image_reward",
        return_value=0.68,
    ):
        results = image_evaluator.evaluate(
            metrics="image_reward",
            image=sample_image,
            prompt="a detailed oil painting",
        )
        assert "image_reward" in results
        assert results["image_reward"] == pytest.approx(0.68)


def test_core_evaluate_detailed(sample_image):
    """Verify evaluate_detailed() produces structured EvaluationResult."""
    with patch(
        "image_evaluator.image_reward_predictor.ImageRewardPredictor.compute_image_reward",
        return_value=0.55,
    ):
        result = image_evaluator.evaluate_detailed(
            metrics=["image_reward"],
            image=sample_image,
            prompt="a vibrant portrait",
        )
        assert result.scores["image_reward"] == pytest.approx(0.55)
        assert "image_reward" in result.specs
        spec = result.specs["image_reward"]
        assert spec.display_name == "ImageReward"
        assert spec.score_direction == "higher_is_better"
        assert result.duration_seconds >= 0.0


def test_core_evaluate_folder(sample_folder):
    """Verify top-level evaluate() with directory input for image_reward."""
    with patch(
        "image_evaluator.image_reward_predictor.ImageRewardPredictor.evaluate_folder_image_reward",
        return_value=0.81,
    ):
        results = image_evaluator.evaluate(
            metrics="image_reward",
            image=sample_folder,
            prompt="a scenic coastline",
        )
        assert results["image_reward"] == pytest.approx(0.81)


def test_core_evaluate_missing_prompt_raises(sample_image):
    """Verify error when prompt is omitted for image_reward."""
    with pytest.raises(ValueError, match="prompt is required"):
        image_evaluator.evaluate(
            metrics="image_reward",
            image=sample_image,
        )


def test_core_evaluate_prohibited_reference_raises(sample_image):
    """Verify error when reference is provided for image_reward."""
    with pytest.raises(ValueError, match="reference was provided"):
        image_evaluator.evaluate(
            metrics="image_reward",
            image=sample_image,
            reference=sample_image,
            prompt="some prompt",
        )


def test_cli_image_reward_text_output(sample_image, capsys):
    """Verify CLI text output format for image_reward."""
    with patch(
        "image_evaluator.image_reward_predictor.ImageRewardPredictor.evaluate_image_reward",
        return_value=0.33,
    ):
        res = main(
            [
                "--metrics",
                "image_reward",
                "--image",
                sample_image,
                "--prompt",
                "a beautiful sunset",
            ]
        )
        captured = capsys.readouterr()
        assert "ImageReward: 0.33" in captured.out
        assert res["metrics"]["image_reward"] == pytest.approx(0.33)


def test_cli_image_reward_json_output(sample_image, capsys):
    """Verify CLI JSON output format for image_reward."""
    with patch(
        "image_evaluator.image_reward_predictor.ImageRewardPredictor.evaluate_image_reward",
        return_value=0.45,
    ):
        res = main(
            [
                "--metrics",
                "image_reward",
                "--image",
                sample_image,
                "--prompt",
                "a cyberpunk skyline",
                "--format",
                "json",
            ]
        )
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "image_reward" in data["metrics"]
        assert data["metrics"]["image_reward"] == pytest.approx(0.45)
        assert res["metrics"]["image_reward"] == pytest.approx(0.45)


def test_cli_missing_prompt_fails(sample_image):
    """Verify CLI exits with code 2 when required prompt is omitted."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--metrics", "image_reward", "--image", sample_image])
    assert exc_info.value.code == 2
