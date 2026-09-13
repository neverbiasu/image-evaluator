"""Executable test suite for documentation examples (T-040-04).

Exercises every Python and CLI code example published in README.md and
the batch-performance guide to guarantee zero code drift.
"""

import json
import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from image_evaluator import (
    ClipScorePredictor,
    PSNRPredictor,
    SSIMPredictor,
    evaluate,
)


@pytest.fixture
def make_png(tmp_path):
    """Helper to create a valid 32x32 PNG image."""
    def _create(path, color="red"):
        img = Image.new("RGB", (32, 32), color=color)
        img.save(path)
        return path
    return _create


def test_readme_python_sdk_tensor_example():
    """Verify exact in-memory PyTorch tensor example from README.md."""
    t_img = torch.rand(3, 32, 32)
    t_ref = torch.rand(3, 32, 32)

    scores = evaluate(
        metrics=["ssim", "psnr"],
        image=t_img,
        reference=t_ref,
        device="cpu",
    )
    assert isinstance(scores, dict)
    assert "ssim" in scores
    assert "psnr" in scores
    assert isinstance(scores["ssim"], float)
    assert isinstance(scores["psnr"], float)


def test_batch_performance_in_memory_pil_reuse():
    """Verify in-memory PIL batch reuse pattern from guide."""
    ssim_pred = SSIMPredictor()
    psnr_pred = PSNRPredictor()

    images = [
        Image.new("RGB", (64, 64), color=(i * 20, 100, 100)) for i in range(5)
    ]
    reference = Image.new("RGB", (64, 64), color=(0, 100, 100))

    scores = []
    for img in images:
        s = ssim_pred.evaluate_ssim(reference, img)
        p = psnr_pred.evaluate_psnr(reference, img)
        scores.append({"ssim": s, "psnr": p})

    assert len(scores) == 5
    for row in scores:
        assert isinstance(row["ssim"], float)
        assert isinstance(row["psnr"], float)


def test_batch_performance_predictor_reuse_across_file_pairs(
    tmp_path, make_png
):
    """Verify Paradigm 2 Predictor reuse across file pairs."""
    ssim_pred = SSIMPredictor()

    pairs = []
    for i in range(3):
        gen_path = tmp_path / f"gen_{i:02d}.png"
        ref_path = tmp_path / f"ref_{i:02d}.png"
        make_png(gen_path, color="blue")
        make_png(ref_path, color="blue")
        pairs.append((str(gen_path), str(ref_path)))

    results = []
    for gen_p, ref_p in pairs:
        score = ssim_pred.evaluate_ssim(ref_p, gen_p)
        results.append(score)

    assert len(results) == 3
    for s in results:
        assert s == pytest.approx(1.0, abs=1e-5)


def test_batch_performance_clip_reuse_with_mock(tmp_path):
    """Verify ClipScorePredictor reuse across multiple sample items."""
    img_paths = []
    for i in range(3):
        p = tmp_path / f"sample_{i}.png"
        Image.new("RGB", (64, 64), color="blue").save(p)
        img_paths.append(str(p))

    mock_model = MagicMock()
    mock_model.get_image_features.return_value = torch.ones((1, 512))
    mock_model.get_text_features.return_value = torch.ones((1, 512))
    mock_processor = MagicMock(
        side_effect=lambda text=None, images=None: {
            "pixel_values": torch.zeros((1, 3, 224, 224), dtype=torch.float32)
        }
    )
    mock_tokenizer = MagicMock(
        side_effect=lambda data, **kwargs: {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
    )

    with patch(
        "image_evaluator.clip_score_predictor.AutoModel.from_pretrained",
        return_value=mock_model,
    ), patch(
        "image_evaluator.clip_score_predictor.AutoProcessor.from_pretrained",
        return_value=mock_processor,
    ), patch(
        "image_evaluator.clip_score_predictor.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    ):
        clip_pred = ClipScorePredictor(device="cpu")
        clip_pred.model = mock_model

        results = []
        for path in img_paths:
            score = clip_pred.evaluate_clip_score(path, "a serene landscape")
            results.append({"image": path, "clip": score})

        assert len(results) == 3
        for r in results:
            assert isinstance(r["clip"], float)


def test_batch_performance_cli_directory_mode_subprocess(
    tmp_path, make_png
):
    """Verify Paradigm 1 CLI directory mode invocation via OS subprocess."""
    gen_dir = tmp_path / "generated"
    ref_dir = tmp_path / "reference"
    gen_dir.mkdir()
    ref_dir.mkdir()

    for name in ["sample_a.png", "sample_b.png"]:
        make_png(gen_dir / name, color="green")
        make_png(ref_dir / name, color="green")

    cmd = [
        sys.executable,
        "-m",
        "image_evaluator.main",
        "--metrics",
        "ssim",
        "psnr",
        "--image",
        str(gen_dir),
        "--reference",
        str(ref_dir),
        "--format",
        "json",
    ]
    env = dict(os.environ, PYTHONPATH=".")
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=False, env=env
    )

    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert "metrics" in data
    assert "ssim" in data["metrics"]
    assert "psnr" in data["metrics"]
    assert data["metrics"]["ssim"] == pytest.approx(1.0, abs=1e-5)
