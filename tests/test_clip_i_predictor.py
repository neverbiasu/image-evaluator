"""Targeted test suite for CLIP-I image-image similarity predictor.

Verifies:
1. Package exports and lazy resolution.
2. Identical images produce similarity 1.0.
3. Multi-format support: PIL Image, torch Tensor, file paths.
4. Stem-based directory pairing evaluation.
5. Missing reference validation in SDK and CLI.
6. Gating and DownloadNotAllowedError behavior when uncached.
7. High-level evaluate() and evaluate_detailed() integration.
8. CLI execution across text and json formats.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

import image_evaluator
from image_evaluator.clip_i_predictor import (
    CLIP_I_ASSET,
    ClipIPredictor,
)
from image_evaluator.core import evaluate, evaluate_detailed
from image_evaluator.main import cli
from image_evaluator.model_assets import DownloadNotAllowedError


@pytest.fixture(scope="module")
def predictor() -> ClipIPredictor:
    """Module-scoped predictor reusing cached model weights."""
    return ClipIPredictor(device="cpu")


@pytest.fixture
def sample_images(tmp_path):
    """Generate sample PIL images and saved disk paths."""
    img1 = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img2 = Image.new("RGB", (100, 100), color=(0, 255, 0))

    path1 = tmp_path / "img1.png"
    path2 = tmp_path / "img2.png"
    img1.save(path1)
    img2.save(path2)

    return {
        "img1": img1,
        "img2": img2,
        "path1": str(path1),
        "path2": str(path2),
    }


def test_package_exports():
    """Verify ClipIPredictor is exported from top-level package."""
    assert hasattr(image_evaluator, "ClipIPredictor")
    assert image_evaluator.ClipIPredictor is ClipIPredictor


def test_identical_image_similarity(predictor, sample_images):
    """Identical images must yield a cosine similarity of approximately 1.0."""
    score = predictor.evaluate_clip_i(
        sample_images["img1"], sample_images["img1"]
    )
    assert isinstance(score, float)
    assert score == pytest.approx(1.0, abs=1e-3)


def test_different_images_lower_similarity(predictor, sample_images):
    """Different images must yield similarity strictly lower than identical."""
    score = predictor.evaluate_clip_i(
        sample_images["img1"], sample_images["img2"]
    )
    assert isinstance(score, float)
    assert score < 0.95


def test_input_formats_support(predictor, sample_images):
    """Supports file paths, PIL Images, and torch Tensors seamlessly."""
    # File path vs PIL Image
    score_path = predictor.evaluate_clip_i(
        sample_images["path1"], sample_images["img1"]
    )
    assert score_path == pytest.approx(1.0, abs=1e-3)

    # torch Tensor (C, H, W) float in [0, 1]
    tensor_img = torch.zeros(3, 64, 64, dtype=torch.float32)
    score_tensor = predictor.evaluate_clip_i(tensor_img, tensor_img)
    assert score_tensor == pytest.approx(1.0, abs=1e-3)


def test_evaluate_folder_clip_i(predictor, tmp_path):
    """Folder evaluation matches images by file stem and averages scores."""
    ref_dir = tmp_path / "refs"
    gen_dir = tmp_path / "gens"
    ref_dir.mkdir()
    gen_dir.mkdir()

    img_a = Image.new("RGB", (64, 64), color=(200, 50, 50))
    img_b = Image.new("RGB", (64, 64), color=(50, 200, 50))

    img_a.save(ref_dir / "sample_01.png")
    img_a.save(gen_dir / "sample_01.png")
    img_b.save(ref_dir / "sample_02.png")
    img_b.save(gen_dir / "sample_02.png")

    mean_score = predictor.evaluate_folder_clip_i(str(ref_dir), str(gen_dir))
    assert isinstance(mean_score, float)
    assert mean_score == pytest.approx(1.0, abs=1e-3)


def test_download_gating_uncached_raises_error():
    """Uncached weights without allow_download must raise error."""
    with patch(
        "image_evaluator.clip_i_predictor._is_clip_i_cached",
        return_value=False,
    ):
        with pytest.raises(DownloadNotAllowedError) as exc_info:
            ClipIPredictor(allow_download=False)
        assert exc_info.value.asset.metric_id == "clip_i"
        assert exc_info.value.asset.model_id == CLIP_I_ASSET.model_id


def test_download_gating_uncached_with_allow_download_discloses():
    """Uncached weights with allow_download emits disclosure callback."""
    cb = MagicMock()
    with patch(
        "image_evaluator.clip_i_predictor._is_clip_i_cached",
        return_value=False,
    ), patch("open_clip.create_model_and_transforms") as mock_create:
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_create.return_value = (mock_model, None, MagicMock())

        ClipIPredictor(allow_download=True, download_callback=cb)
        cb.assert_called_once()
        called_asset, called_msg = cb.call_args[0]
        assert called_asset.metric_id == "clip_i"
        assert "Downloading model" in called_msg


def test_core_evaluate_integration(sample_images):
    """High-level evaluate() dispatches clip_i correctly."""
    result = evaluate(
        metrics="clip_i",
        image=sample_images["img1"],
        reference=sample_images["img1"],
        device="cpu",
    )
    assert "clip_i" in result
    assert result["clip_i"] == pytest.approx(1.0, abs=1e-3)


def test_core_evaluate_detailed_integration(sample_images):
    """evaluate_detailed() packages clip_i score and metric spec."""
    result = evaluate_detailed(
        metrics=["clip_i"],
        image=sample_images["path1"],
        reference=sample_images["path1"],
        device="cpu",
    )
    assert result["clip_i"] == pytest.approx(1.0, abs=1e-3)
    assert "clip_i" in result.specs
    assert result.specs["clip_i"].tasks == (
        "image_editing",
        "subject_driven_generation",
    )
    payload = json.loads(result.to_json())
    assert "clip_i" in payload["scores"]


def test_core_evaluate_missing_reference_raises(sample_images):
    """Evaluating clip_i without reference raises ValueError."""
    with pytest.raises(ValueError, match="reference is required"):
        evaluate(metrics="clip_i", image=sample_images["img1"])


def test_cli_execution_text(sample_images, capsys):
    """CLI evaluates clip_i and prints text result."""
    code = cli(
        [
            "--metrics",
            "clip_i",
            "--image",
            sample_images["path1"],
            "--reference",
            sample_images["path1"],
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "CLIP-I Similarity:" in captured.out


def test_cli_execution_json(sample_images, capsys):
    """CLI evaluates clip_i and outputs structured JSON."""
    code = cli(
        [
            "--metrics",
            "clip_i",
            "--image",
            sample_images["path1"],
            "--reference",
            sample_images["path1"],
            "--format",
            "json",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "success"
    assert "clip_i" in data["metrics"]
    assert data["metrics"]["clip_i"] == pytest.approx(1.0, abs=1e-3)


def test_cli_missing_reference_fails(sample_images, capsys):
    """CLI without reference for clip_i exits with status 2."""
    with pytest.raises(SystemExit) as exc_info:
        cli(
            [
                "--metrics",
                "clip_i",
                "--image",
                sample_images["path1"],
            ]
        )
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--reference is required" in captured.err

