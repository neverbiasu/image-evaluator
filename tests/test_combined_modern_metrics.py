"""Targeted integration tests for combined 0.6.0 modern evaluation metrics.

Verifies that valid combinations of modern metrics (clip_i, dino_similarity,
hpsv2, image_reward, vqascore) run together harmoniously, avoid duplicate
loads, preserve JSON structure without drift, and honor input contracts.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from image_evaluator.core import evaluate, evaluate_detailed
from image_evaluator.main import main
from image_evaluator.registry import get_metric, list_metrics


@pytest.fixture
def sample_image(tmp_path: Path) -> str:
    """Create a temporary test image file."""
    img_path = tmp_path / "gen_sample.png"
    img = Image.new("RGB", (64, 64), color=(180, 90, 40))
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def sample_reference(tmp_path: Path) -> str:
    """Create a temporary test reference image file."""
    ref_path = tmp_path / "ref_sample.png"
    img = Image.new("RGB", (64, 64), color=(170, 85, 45))
    img.save(ref_path)
    return str(ref_path)


@pytest.fixture
def sample_folder_pair(tmp_path: Path) -> tuple[str, str]:
    """Create a pair of folders with matched image stems."""
    gen_dir = tmp_path / "gen_folder"
    ref_dir = tmp_path / "ref_folder"
    gen_dir.mkdir()
    ref_dir.mkdir()
    for i in range(2):
        img_g = Image.new("RGB", (32, 32), color=(i * 40, 80, 120))
        img_r = Image.new("RGB", (32, 32), color=(i * 40, 75, 115))
        img_g.save(gen_dir / f"item_{i:02d}.png")
        img_r.save(ref_dir / f"item_{i:02d}.png")
    return str(gen_dir), str(ref_dir)


def test_registry_contains_all_five_modern_metrics() -> None:
    """Verify all 5 modern metrics are in registry with valid docs."""
    modern_ids = {
        "clip_i",
        "dino_similarity",
        "hpsv2",
        "image_reward",
        "vqascore",
    }
    all_specs = list_metrics()
    registered_ids = {spec.id for spec in all_specs}
    assert modern_ids.issubset(registered_ids)

    for mid in modern_ids:
        spec = get_metric(mid)
        assert Path(spec.docs_path).exists(), f"Docs missing: {spec.docs_path}"


def test_combined_pairwise_modern_metrics(
    sample_image: str, sample_reference: str
) -> None:
    """Verify clip_i and dino_similarity execute together in evaluate()."""
    with patch(
        "image_evaluator.clip_i_predictor.ClipIPredictor"
    ) as MockClipI, patch(
        "image_evaluator.dino_similarity_predictor.DinoSimilarityPredictor"
    ) as MockDino:
        MockClipI.return_value.evaluate_clip_i.return_value = 0.885
        MockDino.return_value.evaluate_dino_similarity.return_value = 0.912

        res = evaluate(
            metrics=["clip_i", "dino_similarity", "ssim", "psnr"],
            image=sample_image,
            reference=sample_reference,
            allow_download=True,
        )
        assert isinstance(res, dict)
        assert res["clip_i"] == pytest.approx(0.885, abs=1e-4)
        assert res["dino_similarity"] == pytest.approx(0.912, abs=1e-4)
        assert "ssim" in res
        assert "psnr" in res


def test_combined_prompt_modern_metrics(sample_image: str) -> None:
    """Verify hpsv2, image_reward, and vqascore execute together."""
    with patch(
        "image_evaluator.hpsv2_predictor.Hpsv2Predictor"
    ) as MockHps, patch(
        "image_evaluator.image_reward_predictor.ImageRewardPredictor"
    ) as MockIR, patch(
        "image_evaluator.vqascore_predictor.VQAScorePredictor"
    ) as MockVQA:
        MockHps.return_value.evaluate_hpsv2.return_value = 0.284
        MockIR.return_value.evaluate_image_reward.return_value = 0.612
        MockVQA.return_value.evaluate_vqascore.return_value = 0.745

        prompt = "a photorealistic mountain lake at sunrise"
        res = evaluate(
            metrics=["hpsv2", "image_reward", "vqascore", "clip"],
            image=sample_image,
            prompt=prompt,
            allow_download=True,
        )
        assert isinstance(res, dict)
        assert res["hpsv2"] == pytest.approx(0.284, abs=1e-4)
        assert res["image_reward"] == pytest.approx(0.612, abs=1e-4)
        assert res["vqascore"] == pytest.approx(0.745, abs=1e-4)
        assert "clip" in res


def test_full_fifteen_metric_heterogeneous_combination(
    sample_image: str, sample_reference: str
) -> None:
    """Verify mixed pairwise, prompt, and aesthetic metrics evaluate."""
    with patch(
        "image_evaluator.clip_i_predictor.ClipIPredictor"
    ) as MockClipI, patch(
        "image_evaluator.dino_similarity_predictor.DinoSimilarityPredictor"
    ) as MockDino, patch(
        "image_evaluator.hpsv2_predictor.Hpsv2Predictor"
    ) as MockHps, patch(
        "image_evaluator.image_reward_predictor.ImageRewardPredictor"
    ) as MockIR, patch(
        "image_evaluator.vqascore_predictor.VQAScorePredictor"
    ) as MockVQA:
        MockClipI.return_value.evaluate_clip_i.return_value = 0.85
        MockDino.return_value.evaluate_dino_similarity.return_value = 0.89
        MockHps.return_value.evaluate_hpsv2.return_value = 0.29
        MockIR.return_value.evaluate_image_reward.return_value = 0.55
        MockVQA.return_value.evaluate_vqascore.return_value = 0.78

        prompt = "a red sports car"
        result = evaluate_detailed(
            metrics=[
                "aesthetic",
                "ssim",
                "psnr",
                "clip_i",
                "dino_similarity",
                "hpsv2",
                "image_reward",
                "vqascore",
            ],
            image=sample_image,
            reference=sample_reference,
            prompt=prompt,
            allow_download=True,
        )

        assert len(result.scores) == 8
        assert result.scores["clip_i"] == pytest.approx(0.85, abs=1e-4)
        assert result.scores["dino_similarity"] == pytest.approx(
            0.89, abs=1e-4
        )
        assert result.scores["hpsv2"] == pytest.approx(0.29, abs=1e-4)
        assert result.scores["image_reward"] == pytest.approx(0.55, abs=1e-4)
        assert result.scores["vqascore"] == pytest.approx(0.78, abs=1e-4)
        assert "aesthetic" in result.scores
        assert "ssim" in result.scores
        assert "psnr" in result.scores

        # Verify JSON serialization without drift
        json_output = result.to_json()
        data = json.loads(json_output)
        assert "scores" in data
        assert "specs" in data
        assert "inputs" in data
        assert "duration_seconds" in data
        assert len(data["scores"]) == 8
        assert len(data["specs"]) == 8
        for m in (
            "clip_i",
            "dino_similarity",
            "hpsv2",
            "image_reward",
            "vqascore",
        ):
            assert m in data["specs"]
            assert data["specs"][m]["id"] == m


def test_cli_combined_modern_metrics_json(
    sample_image: str, sample_reference: str
) -> None:
    """Verify CLI executes combined modern metrics with JSON serialization."""
    with patch(
        "image_evaluator.clip_i_predictor.ClipIPredictor"
    ) as MockClipI, patch(
        "image_evaluator.dino_similarity_predictor.DinoSimilarityPredictor"
    ) as MockDino:
        MockClipI.return_value.evaluate_clip_i.return_value = 0.81
        MockDino.return_value.evaluate_dino_similarity.return_value = 0.84

        args = [
            "--metrics",
            "clip_i",
            "dino_similarity",
            "--image",
            sample_image,
            "--reference",
            sample_reference,
            "--format",
            "json",
        ]
        out_data = main(args)
        assert isinstance(out_data, dict)
        assert out_data["status"] == "success"
        assert out_data["metrics"]["clip_i"] == pytest.approx(0.81, abs=1e-4)
        assert out_data["metrics"]["dino_similarity"] == pytest.approx(
            0.84, abs=1e-4
        )


def test_in_memory_tensor_stream_modern_metrics() -> None:
    """Verify in-memory PyTorch tensors evaluate without disk I/O."""
    t_gen = torch.rand(3, 32, 32)
    t_ref = torch.rand(3, 32, 32)

    with patch(
        "image_evaluator.clip_i_predictor.ClipIPredictor"
    ) as MockClipI, patch(
        "image_evaluator.dino_similarity_predictor.DinoSimilarityPredictor"
    ) as MockDino:
        MockClipI.return_value.evaluate_clip_i.return_value = 0.95
        MockDino.return_value.evaluate_dino_similarity.return_value = 0.97

        res = evaluate(
            metrics=["clip_i", "dino_similarity", "ssim", "psnr"],
            image=t_gen,
            reference=t_ref,
            allow_download=True,
        )
        assert res["clip_i"] == pytest.approx(0.95, abs=1e-4)
        assert res["dino_similarity"] == pytest.approx(0.97, abs=1e-4)
        assert isinstance(res["ssim"], float)
        assert isinstance(res["psnr"], float)
