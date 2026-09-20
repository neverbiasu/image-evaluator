"""Tests for VQAScore predictor and evaluation pipeline integration."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

import image_evaluator
from image_evaluator.core import evaluate, evaluate_detailed
from image_evaluator.main import main
from image_evaluator.model_assets import DownloadNotAllowedError
from image_evaluator.vqascore_predictor import (
    VQA_SCORE_ASSET,
    VQA_SCORE_TEXT_ASSET,
    VQA_SCORE_VISION_ASSET,
    VQAScorePredictor,
)


class DummyVQAScoreOutputs:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class DummyVQAScoreModel(torch.nn.Module):
    """Dummy model returning controlled logits for VQAScore posterior."""

    def __init__(self, fixed_score: float = 0.5) -> None:
        super().__init__()
        self.fixed_score = fixed_score

    def forward(self, *args: Any, **kwargs: Any) -> DummyVQAScoreOutputs:
        # With 2 classes and uniform logits (0.0), CE loss = ln(2)
        # exp(-ln(2)) = 0.5
        # If fixed_score is different, compute target loss = -ln(score)
        target_loss = -float(np.log(max(1e-6, min(1.0, self.fixed_score))))
        # Shape: (1, 2, 2)
        # For class 0: logit_0 = log(p0), logit_1 = log(p1)
        # where p0 = exp(-target_loss), p1 = 1 - p0
        p0 = float(np.exp(-target_loss))
        p1 = max(1e-9, 1.0 - p0)
        l0 = float(np.log(p0))
        l1 = float(np.log(p1))
        logits = torch.tensor([[[l0, l1], [l0, l1]]], dtype=torch.float32)
        return DummyVQAScoreOutputs(logits=logits)


@dataclass
class _DummyTokenizerOutput:
    input_ids: list[int]


class _DummyTokenizer:
    pad_token_id = 0

    def __call__(self, prompt: str, **kwargs: Any) -> _DummyTokenizerOutput:
        # Return dummy tokens, label tokens will be index 0
        return _DummyTokenizerOutput(input_ids=[0, 0])


class _DummyImageProcessor:
    image_mean = [0.48145466, 0.4578275, 0.40821073]

    def preprocess(self, img: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        return {"pixel_values": torch.zeros(1, 3, 336, 336)}


@pytest.fixture
def sample_image(tmp_path: Path) -> str:
    """Create a temporary test image file."""
    img_path = tmp_path / "test_sample.png"
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def sample_folder(tmp_path: Path) -> str:
    """Create a temporary folder with multiple valid images."""
    folder = tmp_path / "img_dir"
    folder.mkdir()
    for i in range(2):
        img = Image.new("RGB", (32, 32), color=(i * 50, 100, 150))
        img.save(folder / f"img_{i:02d}.png")
    return str(folder)


def _mock_vqascore_predictor_loaded(
    predictor: VQAScorePredictor, fixed_score: float = 0.5
) -> None:
    """Inject dummy model, tokenizer, and preprocess."""
    predictor._model = DummyVQAScoreModel(fixed_score=fixed_score)
    predictor._tokenizer = _DummyTokenizer()
    predictor._image_processor = _DummyImageProcessor()


def test_package_exports() -> None:
    """Verify VQAScorePredictor is exported in top-level package."""
    assert hasattr(image_evaluator, "VQAScorePredictor")
    assert "VQAScorePredictor" in dir(image_evaluator)


def test_download_gating_uncached_raises_error(sample_image: str) -> None:
    """Verify that uncached model without --allow-download raises error."""
    with patch(
        "image_evaluator.vqascore_predictor._is_vqascore_cached",
        return_value=False,
    ), patch(
        "image_evaluator.vqascore_predictor._get_vqascore_cached_paths",
        return_value=(None, None),
    ):
        predictor = VQAScorePredictor(allow_download=False)
        with pytest.raises(DownloadNotAllowedError) as exc_info:
            predictor.compute_vqascore(sample_image, "a photo of a cat")

        msg = str(exc_info.value)
        assert "zhiqiulin/clip-flant5-xl" in msg
        assert "vqascore" in msg
        assert "--allow-download" in msg
        assert "image-evaluator[vqa]" in msg


def test_download_gating_disclosure_called() -> None:
    """Verify download disclosure callback is triggered on download."""
    from image_evaluator.model_assets import check_asset_and_permit_download

    disclosures: list[tuple[Any, str]] = []

    def callback(asset: Any, message: str) -> None:
        disclosures.append((asset, message))

    check_asset_and_permit_download(
        asset=VQA_SCORE_ASSET,
        is_cached=False,
        allow_download=True,
        disclosure_callback=callback,
    )
    assert len(disclosures) == 1
    asset, msg = disclosures[0]
    assert asset.metric_id == "vqascore"
    assert "zhiqiulin/clip-flant5-xl" in msg
    assert "6327057688" in msg or "6.3 GB" in msg or "5.9 GB" in msg


def test_posterior_math_and_boundedness(sample_image: str) -> None:
    """Verify posterior probability math produces value in [0.0, 1.0]."""
    predictor = VQAScorePredictor(allow_download=True)
    _mock_vqascore_predictor_loaded(predictor, fixed_score=0.72)

    score = predictor.compute_vqascore(sample_image, "a red car")
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(0.72, abs=1e-3)


def test_input_formats(tmp_path: Path, sample_image: str) -> None:
    """Verify support for str path, Path, PIL, numpy, and Tensor inputs."""
    predictor = VQAScorePredictor(allow_download=True)
    _mock_vqascore_predictor_loaded(predictor, fixed_score=0.65)

    prompt = "a cute puppy"

    # 1. str path
    score1 = predictor.compute_vqascore(sample_image, prompt)
    assert isinstance(score1, float)

    # 2. Path object
    score2 = predictor.compute_vqascore(Path(sample_image), prompt)
    assert score2 == score1

    # 3. PIL.Image
    pil_img = Image.open(sample_image)
    score3 = predictor.compute_vqascore(pil_img, prompt)
    assert score3 == score1

    # 4. numpy uint8 array (H, W, 3)
    np_uint8 = np.array(pil_img)
    score4 = predictor.compute_vqascore(np_uint8, prompt)
    assert score4 == score1

    # 5. numpy float32 array in [0, 1]
    np_float = np_uint8.astype(np.float32) / 255.0
    score5 = predictor.compute_vqascore(np_float, prompt)
    assert score5 == score1

    # 6. torch Tensor (3, H, W)
    t_chw = torch.from_numpy(np_float).permute(2, 0, 1)
    score6 = predictor.compute_vqascore(t_chw, prompt)
    assert score6 == score1

    # 7. torch Tensor (1, 3, H, W)
    t_bchw = t_chw.unsqueeze(0)
    score7 = predictor.compute_vqascore(t_bchw, prompt)
    assert score7 == score1


def test_prompt_validation(sample_image: str) -> None:
    """Verify that invalid prompt values raise ValueError."""
    predictor = VQAScorePredictor(allow_download=True)
    _mock_vqascore_predictor_loaded(predictor)

    with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
        predictor.compute_vqascore(sample_image, "")

    with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
        predictor.compute_vqascore(sample_image, "   ")


def test_invalid_image_type() -> None:
    """Verify that unsupported image types raise TypeError or ValueError."""
    predictor = VQAScorePredictor(allow_download=True)
    _mock_vqascore_predictor_loaded(predictor)

    with pytest.raises(TypeError, match="Unsupported image input type"):
        predictor.compute_vqascore(12345, "a prompt")

    with pytest.raises(ValueError, match="Unsupported numpy array"):
        predictor.compute_vqascore(np.zeros((10,)), "a prompt")


def test_folder_evaluation(sample_folder: str, tmp_path: Path) -> None:
    """Verify directory evaluation computes arithmetic mean across images."""
    predictor = VQAScorePredictor(allow_download=True)
    _mock_vqascore_predictor_loaded(predictor, fixed_score=0.55)

    mean_score = predictor.evaluate_folder_vqascore(
        sample_folder, "a test prompt"
    )
    assert isinstance(mean_score, float)
    assert mean_score == pytest.approx(0.55, abs=1e-3)

    # Missing directory
    with pytest.raises(FileNotFoundError):
        predictor.evaluate_folder_vqascore(
            str(tmp_path / "nonexistent"), "prompt"
        )

    # Empty directory
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No supported image files"):
        predictor.evaluate_folder_vqascore(str(empty_dir), "prompt")


def test_cli_vqascore_missing_prompt(sample_image: str) -> None:
    """Verify CLI requires --prompt when --metrics vqascore is selected."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--metrics", "vqascore", "--image", sample_image])
    assert exc_info.value.code == 2


def test_cli_vqascore_success_json(
    sample_image: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify CLI execution with --metrics vqascore and --format json."""
    with patch(
        "image_evaluator.vqascore_predictor.VQAScorePredictor"
    ) as MockPredictor:
        instance = MockPredictor.return_value
        instance.evaluate_vqascore.return_value = 0.812345

        res = main(
            [
                "--metrics",
                "vqascore",
                "--image",
                sample_image,
                "--prompt",
                "a sports car",
                "--format",
                "json",
            ]
        )
        assert isinstance(res, dict)
        assert "metrics" in res
        assert res["metrics"]["vqascore"] == pytest.approx(0.812345, abs=1e-5)


def test_high_level_evaluate_and_detailed(sample_image: str) -> None:
    """Verify high-level evaluate() and evaluate_detailed() for vqascore."""
    with patch(
        "image_evaluator.vqascore_predictor.VQAScorePredictor"
    ) as MockPredictor:
        instance = MockPredictor.return_value
        instance.evaluate_vqascore.return_value = 0.7788

        # 1. Plain dictionary evaluate
        res = evaluate(
            metrics="vqascore",
            image=sample_image,
            prompt="a cat",
            allow_download=True,
        )
        assert isinstance(res, dict)
        assert "vqascore" in res
        assert res["vqascore"] == pytest.approx(0.7788, abs=1e-4)

        # 2. Detailed result evaluate_detailed
        detail = evaluate_detailed(
            metrics="vqascore",
            image=sample_image,
            prompt="a cat",
            allow_download=True,
        )
        assert "vqascore" in detail.scores
        assert detail.scores["vqascore"] == pytest.approx(0.7788, abs=1e-4)
        assert detail.specs["vqascore"].id == "vqascore"
        assert detail.specs["vqascore"].display_name == "VQAScore"
        assert detail.inputs["prompt"] == "a cat"

        # 3. JSON serialization of EvaluationResult
        json_str = detail.to_json()
        parsed = json.loads(json_str)
        assert parsed["scores"]["vqascore"] == pytest.approx(0.7788, abs=1e-4)


def test_download_gating_vision_uncached_raises_error(
    sample_image: str,
) -> None:
    """Verify uncached vision model raises DownloadNotAllowedError."""
    with patch(
        "image_evaluator.vqascore_predictor._get_vqascore_cached_paths",
        return_value=("/fake/xl_dir", None),
    ), patch(
        "os.path.exists",
        side_effect=lambda p: True if p == "/fake/xl_dir" else False,
    ):
        predictor = VQAScorePredictor(allow_download=False)
        with pytest.raises(DownloadNotAllowedError) as exc_info:
            predictor.compute_vqascore(sample_image, "a photo of a cat")

        msg = str(exc_info.value)
        assert "openai/clip-vit-large-patch14-336" in msg
        assert "vqascore" in msg
        assert "--allow-download" in msg
        assert "1715593675" in msg or "1.7 GB" in msg


def test_download_gating_discloses_both_assets() -> None:
    """Verify both text backbone and vision tower assets are disclosed."""
    from image_evaluator.model_assets import check_asset_and_permit_download

    disclosures: list[tuple[Any, str]] = []

    def callback(asset: Any, message: str) -> None:
        disclosures.append((asset, message))

    # 1. Text backbone disclosure
    check_asset_and_permit_download(
        asset=VQA_SCORE_TEXT_ASSET,
        is_cached=False,
        allow_download=True,
        disclosure_callback=callback,
    )
    # 2. Vision tower disclosure
    check_asset_and_permit_download(
        asset=VQA_SCORE_VISION_ASSET,
        is_cached=False,
        allow_download=True,
        disclosure_callback=callback,
    )

    assert len(disclosures) == 2
    text_asset, text_msg = disclosures[0]
    vis_asset, vis_msg = disclosures[1]

    assert text_asset.model_id == "zhiqiulin/clip-flant5-xl"
    assert "6.3 GB" in text_msg or "6327057688" in text_msg
    assert vis_asset.model_id == "openai/clip-vit-large-patch14-336"
    assert "1.7 GB" in vis_msg or "1715593675" in vis_msg

