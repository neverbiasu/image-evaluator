"""Targeted test suite for DINOv2 image-image similarity predictor.

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
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

import image_evaluator
from image_evaluator.core import evaluate, evaluate_detailed
from image_evaluator.dino_similarity_predictor import (
    DINO_SIMILARITY_ASSET,
    DinoSimilarityPredictor,
)
from image_evaluator.main import cli
from image_evaluator.model_assets import DownloadNotAllowedError


class DummyDinoOutput:
    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class DummyDinoModel(torch.nn.Module):
    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, pixel_values: torch.Tensor) -> DummyDinoOutput:
        batch = pixel_values.shape[0]
        val = pixel_values.mean(dim=(-2, -1))
        feat = torch.zeros(
            batch, 1, 768, dtype=torch.float32, device=pixel_values.device
        )
        feat[:, 0, :3] = val
        feat[:, 0, 3] = 1.0
        return DummyDinoOutput(feat)


class DummyDinoProcessor:
    def __call__(
        self, images: Any = None, return_tensors: str = "pt"
    ) -> dict[str, torch.Tensor]:
        if isinstance(images, torch.Tensor):
            t = images
        elif isinstance(images, Image.Image):
            arr = np.array(images).astype(np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1)
        elif isinstance(images, np.ndarray):
            t = torch.from_numpy(images).float()
            if t.ndim == 3 and t.shape[-1] in (1, 3):
                t = t.permute(2, 0, 1)
        else:
            t = torch.zeros(3, 224, 224, dtype=torch.float32)
        if t.ndim == 3:
            t = t.unsqueeze(0)
        return {"pixel_values": t}


@contextmanager
def mock_dinov2():
    with patch(
        "image_evaluator.dino_similarity_predictor._is_dinov2_cached",
        return_value=True,
    ), patch(
        "transformers.AutoImageProcessor.from_pretrained",
        return_value=DummyDinoProcessor(),
    ), patch(
        "transformers.AutoModel.from_pretrained",
        return_value=DummyDinoModel(),
    ):
        yield


@pytest.fixture
def predictor() -> DinoSimilarityPredictor:
    """Predictor using DummyDinoModel for offline deterministic tests."""
    with mock_dinov2():
        return DinoSimilarityPredictor(device="cpu")


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
    """Verify DinoSimilarityPredictor is exported from top-level package."""
    assert hasattr(image_evaluator, "DinoSimilarityPredictor")
    assert (
        image_evaluator.DinoSimilarityPredictor is DinoSimilarityPredictor
    )


def test_identical_image_similarity(predictor, sample_images):
    """Identical images must yield a cosine similarity of approx 1.0."""
    score = predictor.evaluate_dino_similarity(
        sample_images["img1"], sample_images["img1"]
    )
    assert isinstance(score, float)
    assert score == pytest.approx(1.0, abs=1e-3)


def test_different_images_lower_similarity(predictor, sample_images):
    """Different images must yield similarity lower than identical."""
    score = predictor.evaluate_dino_similarity(
        sample_images["img1"], sample_images["img2"]
    )
    assert isinstance(score, float)
    assert score < 0.98


def test_input_formats_support(predictor, sample_images):
    """Supports file paths, PIL Images, and torch Tensors seamlessly."""
    score_path = predictor.evaluate_dino_similarity(
        sample_images["path1"], sample_images["img1"]
    )
    assert score_path == pytest.approx(1.0, abs=1e-3)

    tensor_img = torch.zeros(3, 64, 64, dtype=torch.float32)
    score_tensor = predictor.evaluate_dino_similarity(tensor_img, tensor_img)
    assert score_tensor == pytest.approx(1.0, abs=1e-3)


def test_evaluate_folder_dino_similarity(predictor, tmp_path):
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

    mean_score = predictor.evaluate_folder_dino_similarity(
        str(ref_dir), str(gen_dir)
    )
    assert isinstance(mean_score, float)
    assert mean_score == pytest.approx(1.0, abs=1e-3)


def test_download_gating_uncached_raises_error():
    """Uncached weights without allow_download must raise error."""
    with patch(
        "image_evaluator.dino_similarity_predictor._is_dinov2_cached",
        return_value=False,
    ):
        with pytest.raises(DownloadNotAllowedError) as exc_info:
            DinoSimilarityPredictor(allow_download=False)
        assert exc_info.value.asset.metric_id == "dino_similarity"
        assert exc_info.value.asset.model_id == DINO_SIMILARITY_ASSET.model_id


def test_download_gating_uncached_with_allow_download_discloses():
    """Uncached weights with allow_download emits disclosure callback."""
    cb = MagicMock()
    with patch(
        "image_evaluator.dino_similarity_predictor._is_dinov2_cached",
        return_value=False,
    ), patch(
        "transformers.AutoImageProcessor.from_pretrained"
    ) as mock_proc, patch(
        "transformers.AutoModel.from_pretrained"
    ) as mock_model:
        mock_proc.return_value = MagicMock()
        mock_m = MagicMock()
        mock_m.to.return_value = mock_m
        mock_m.eval.return_value = None
        mock_model.return_value = mock_m

        DinoSimilarityPredictor(allow_download=True, download_callback=cb)
        cb.assert_called_once()
        called_asset, called_msg = cb.call_args[0]
        assert called_asset.metric_id == "dino_similarity"
        assert "Downloading model" in called_msg


def test_core_evaluate_integration(sample_images):
    """High-level evaluate() dispatches dino_similarity correctly."""
    with mock_dinov2():
        result = evaluate(
            metrics="dino_similarity",
            image=sample_images["img1"],
            reference=sample_images["img1"],
            device="cpu",
        )
    assert "dino_similarity" in result
    assert result["dino_similarity"] == pytest.approx(1.0, abs=1e-3)


def test_core_evaluate_detailed_integration(sample_images):
    """evaluate_detailed() packages dino_similarity score and metric spec."""
    with mock_dinov2():
        result = evaluate_detailed(
            metrics=["dino_similarity"],
            image=sample_images["path1"],
            reference=sample_images["path1"],
            device="cpu",
        )
    assert result["dino_similarity"] == pytest.approx(1.0, abs=1e-3)
    assert "dino_similarity" in result.specs
    assert result.specs["dino_similarity"].tasks == (
        "image_editing",
        "subject_driven_generation",
    )
    payload = json.loads(result.to_json())
    assert "dino_similarity" in payload["scores"]


def test_core_evaluate_missing_reference_raises(sample_images):
    """Evaluating dino_similarity without reference raises ValueError."""
    with pytest.raises(ValueError, match="reference is required"):
        evaluate(metrics="dino_similarity", image=sample_images["img1"])


def test_cli_execution_text(sample_images, capsys):
    """CLI evaluates dino_similarity and prints text result."""
    with mock_dinov2():
        code = cli(
            [
                "--metrics",
                "dino_similarity",
                "--image",
                sample_images["path1"],
                "--reference",
                sample_images["path1"],
            ]
        )
    assert code == 0
    captured = capsys.readouterr()
    assert "DINOv2 Similarity:" in captured.out


def test_cli_execution_json(sample_images, capsys):
    """CLI evaluates dino_similarity and outputs structured JSON."""
    with mock_dinov2():
        code = cli(
            [
                "--metrics",
                "dino_similarity",
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
    assert "dino_similarity" in data["metrics"]
    assert data["metrics"]["dino_similarity"] == pytest.approx(1.0, abs=1e-3)


def test_cli_missing_reference_fails(sample_images, capsys):
    """CLI without reference for dino_similarity exits with status 2."""
    with pytest.raises(SystemExit) as exc_info:
        cli(
            [
                "--metrics",
                "dino_similarity",
                "--image",
                sample_images["path1"],
            ]
        )
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--reference is required" in captured.err

