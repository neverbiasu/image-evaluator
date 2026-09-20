"""Tests for HPS v2.1 predictor and evaluation pipeline integration."""

import json
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

import image_evaluator
from image_evaluator.hpsv2_predictor import (
    Hpsv2Predictor,
)
from image_evaluator.main import main
from image_evaluator.model_assets import DownloadNotAllowedError


class DummyOpenClipModel(torch.nn.Module):
    """Dummy OpenCLIP model returning controlled normalized feature vectors."""

    def __init__(self, feature_dim: int = 1024) -> None:
        super().__init__()
        self.feature_dim = feature_dim

    def forward(
        self, image: torch.Tensor, text: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        batch_size = image.shape[0]
        # Return deterministic unit vectors for predictable cosine similarity
        img_feats = torch.zeros(batch_size, self.feature_dim)
        img_feats[:, 0] = 1.0  # Unit vector along dimension 0

        # If text token contains specific pattern, adjust text vector
        txt_feats = torch.zeros(batch_size, self.feature_dim)
        if text.shape[1] > 0 and text[0, 0].item() == 999:
            txt_feats[:, 1] = 1.0  # Orthogonal vector along dimension 1
        else:
            txt_feats[:, 0] = 1.0  # Parallel vector along dimension 0

        return {
            "image_features": img_feats,
            "text_features": txt_feats,
        }


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


def _mock_hps_predictor_loaded(predictor: Hpsv2Predictor) -> None:
    """Inject dummy model, tokenizer, and preprocess."""
    predictor._model = DummyOpenClipModel(feature_dim=1024)
    predictor._preprocess = lambda pil_img: torch.zeros(3, 224, 224)

    def dummy_tokenizer(texts: list[str]) -> torch.Tensor:
        if texts and "orthogonal" in texts[0]:
            return torch.tensor([[999, 1, 2]])
        return torch.tensor([[100, 1, 2]])

    predictor._tokenizer = dummy_tokenizer


def test_package_exports():
    """Verify Hpsv2Predictor is exported in top-level package."""
    assert hasattr(image_evaluator, "Hpsv2Predictor")
    assert "Hpsv2Predictor" in dir(image_evaluator)


def test_download_gating_uncached_raises_error(sample_image):
    """Verify that an uncached model without --allow-download raises error."""
    with patch(
        "image_evaluator.hpsv2_predictor._is_hpsv2_cached",
        return_value=False,
    ):
        predictor = Hpsv2Predictor(allow_download=False)
        with pytest.raises(DownloadNotAllowedError) as exc_info:
            predictor.compute_hpsv2(sample_image, "a photo of a cat")

        msg = str(exc_info.value)
        assert "xswu/HPSv2" in msg
        assert "hpsv2" in msg
        assert "--allow-download" in msg
        assert "image-evaluator[preference]" in msg


def test_download_gating_uncached_with_allow_download_discloses(sample_image):
    """Verify download disclosure occurs when allow_download=True."""
    disclosed_assets = []

    def mock_callback(asset, message):
        disclosed_assets.append((asset, message))

    with patch(
        "image_evaluator.hpsv2_predictor._is_hpsv2_cached",
        return_value=False,
    ), patch(
        "huggingface_hub.hf_hub_download",
        return_value="/tmp/mock_hps.pt",
    ), patch(
        "open_clip.create_model_and_transforms",
        return_value=(
            DummyOpenClipModel(),
            None,
            lambda x: torch.zeros(3, 224, 224),
        ),
    ), patch(
        "open_clip.get_tokenizer",
        return_value=lambda x: torch.tensor([[1, 2, 3]]),
    ), patch(
        "torch.load",
        return_value={"state_dict": {}},
    ):
        predictor = Hpsv2Predictor(
            allow_download=True,
            download_callback=mock_callback,
        )
        score = predictor.compute_hpsv2(sample_image, "a photo of a cat")
        assert isinstance(score, float)
        assert len(disclosed_assets) == 1
        asset, msg = disclosed_assets[0]
        assert asset.metric_id == "hpsv2"
        assert asset.model_id == "xswu/HPSv2"
        assert "1972490005" in msg or "1.8 GB" in msg


def test_hpsv2_math_cosine_exact():
    """Verify exact cosine score calculation using controlled embeddings."""
    predictor = Hpsv2Predictor()
    _mock_hps_predictor_loaded(predictor)

    img = Image.new("RGB", (64, 64), (100, 100, 100))
    # Parallel features -> cosine similarity 1.0
    score_parallel = predictor.compute_hpsv2(img, "normal prompt")
    assert pytest.approx(score_parallel, rel=1e-5) == 1.0

    # Orthogonal features -> cosine similarity 0.0
    score_orthogonal = predictor.compute_hpsv2(img, "orthogonal prompt")
    assert pytest.approx(score_orthogonal, rel=1e-5) == 0.0


def test_hpsv2_input_formats_support(sample_image):
    """Verify support for PIL.Image, torch.Tensor, ndarray, and str path."""
    predictor = Hpsv2Predictor()
    _mock_hps_predictor_loaded(predictor)

    prompt = "a majestic cat"

    # 1. str path
    s1 = predictor.compute_hpsv2(sample_image, prompt)
    assert isinstance(s1, float)

    # 2. PIL Image
    pil_img = Image.open(sample_image)
    s2 = predictor.compute_hpsv2(pil_img, prompt)
    assert isinstance(s2, float)

    # 3. torch.Tensor (3, H, W)
    t1 = torch.rand(3, 64, 64)
    s3 = predictor.compute_hpsv2(t1, prompt)
    assert isinstance(s3, float)

    # 4. torch.Tensor (1, 3, H, W)
    t2 = torch.rand(1, 3, 64, 64)
    s4 = predictor.compute_hpsv2(t2, prompt)
    assert isinstance(s4, float)

    # 5. numpy.ndarray (H, W, 3)
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    s5 = predictor.compute_hpsv2(arr, prompt)
    assert isinstance(s5, float)


def test_hpsv2_invalid_inputs_raise(sample_image):
    """Verify input validation errors are cleanly raised."""
    predictor = Hpsv2Predictor()
    _mock_hps_predictor_loaded(predictor)

    # Empty prompt
    with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
        predictor.compute_hpsv2(sample_image, "")

    with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
        predictor.compute_hpsv2(sample_image, "   ")

    # Non-existent file
    with pytest.raises(FileNotFoundError, match="Image not found"):
        predictor.compute_hpsv2("/tmp/non_existent_img_12345.png", "prompt")

    # Directory passed as image
    with pytest.raises(ValueError, match="Expected single image file"):
        predictor.compute_hpsv2("/tmp", "prompt")

    # Invalid tensor shape
    with pytest.raises(ValueError, match="Expected 3-channel image tensor"):
        predictor.compute_hpsv2(torch.rand(2, 64, 64), "prompt")

    # Unsupported type
    with pytest.raises(TypeError, match="Unsupported image input type"):
        predictor.compute_hpsv2(12345, "prompt")


def test_evaluate_folder_hpsv2(sample_folder):
    """Verify directory evaluation aggregates scores by arithmetic mean."""
    predictor = Hpsv2Predictor()
    _mock_hps_predictor_loaded(predictor)

    mean_score = predictor.evaluate_folder_hpsv2(sample_folder, "test prompt")
    assert isinstance(mean_score, float)
    assert pytest.approx(mean_score, rel=1e-5) == 1.0

    # Non-existent directory
    with pytest.raises(FileNotFoundError, match="Directory not found"):
        predictor.evaluate_folder_hpsv2(
            "/tmp/non_existent_folder_abc", "prompt"
        )


def test_core_evaluate_integration(sample_image):
    """Verify evaluate() core API dispatches hpsv2."""
    with patch(
        "image_evaluator.hpsv2_predictor.Hpsv2Predictor._ensure_loaded",
    ), patch(
        "image_evaluator.hpsv2_predictor.Hpsv2Predictor.compute_hpsv2",
        return_value=0.2854,
    ):
        res = image_evaluator.evaluate(
            metrics=["hpsv2"],
            image=sample_image,
            prompt="a cozy fireplace",
        )
        assert "hpsv2" in res
        assert pytest.approx(res["hpsv2"], rel=1e-4) == 0.2854


def test_core_evaluate_detailed_integration(sample_image):
    """Verify evaluate_detailed() core API dispatches hpsv2 with metadata."""
    with patch(
        "image_evaluator.hpsv2_predictor.Hpsv2Predictor.compute_hpsv2",
        return_value=0.3120,
    ):
        result = image_evaluator.evaluate_detailed(
            metrics=["hpsv2"],
            image=sample_image,
            prompt="a sunny beach",
        )
        assert "hpsv2" in result.scores
        assert result.scores["hpsv2"] == 0.3120
        assert "hpsv2" in result.specs
        assert result.specs["hpsv2"].display_name == "HPS v2.1"
        assert result.duration_seconds >= 0.0


def test_core_evaluate_missing_prompt_raises(sample_image):
    """Verify evaluate() raises ValueError when prompt is omitted for hpsv2."""
    with pytest.raises(ValueError, match="prompt is required when 'hpsv2'"):
        image_evaluator.evaluate(
            metrics=["hpsv2"],
            image=sample_image,
        )


def test_core_evaluate_prohibited_reference_raises(sample_image):
    """Verify evaluate() raises ValueError when reference is provided."""
    with pytest.raises(
        ValueError,
        match="reference was provided but no reference-based metric",
    ):
        image_evaluator.evaluate(
            metrics=["hpsv2"],
            image=sample_image,
            reference=sample_image,
            prompt="a cat",
        )


def test_cli_execution_text(sample_image, capsys):
    """Verify CLI execution in text mode."""
    with patch(
        "image_evaluator.hpsv2_predictor.Hpsv2Predictor.evaluate_hpsv2",
        return_value=0.2985,
    ):
        res = main(
            [
                "--metrics",
                "hpsv2",
                "--image",
                sample_image,
                "--prompt",
                "cyberpunk street",
            ]
        )
        out = capsys.readouterr().out
        assert "HPS v2.1: 0.2985" in out
        assert res["metrics"]["hpsv2"] == 0.2985


def test_cli_execution_json(sample_image, capsys):
    """Verify CLI execution in JSON mode."""
    with patch(
        "image_evaluator.hpsv2_predictor.Hpsv2Predictor.evaluate_hpsv2",
        return_value=0.3250,
    ):
        res = main(
            [
                "--metrics",
                "hpsv2",
                "--image",
                sample_image,
                "--prompt",
                "starry night",
                "--format",
                "json",
            ]
        )
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["metrics"]["hpsv2"] == 0.3250
        assert res == data


def test_cli_missing_prompt_fails(sample_image):
    """Verify CLI exits with code 2 when required prompt is omitted."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--metrics", "hpsv2", "--image", sample_image])
    assert exc_info.value.code == 2
