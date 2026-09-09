import numpy as np
import pytest
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage

from image_evaluator.psnr_predictor import PSNRPredictor


def test_compute_psnr_tensor_identical_images_infinite():
    """Verify PSNR of identical images is infinite."""
    predictor = PSNRPredictor(device="cpu")
    t = torch.rand(1, 3, 64, 64)
    score = predictor.compute_psnr_tensor(t, t)
    assert math_is_inf(score)


def math_is_inf(val):
    import math

    return math.isinf(val) and val > 0


def test_compute_psnr_tensor_matches_skimage_numerical_accuracy():
    """Verify PSNR implementation matches skimage to high float precision."""
    predictor = PSNRPredictor(device="cpu")
    np.random.seed(42)

    x_np = np.random.rand(128, 128, 3).astype(np.float32)
    noise = np.random.normal(0, 0.05, x_np.shape).astype(np.float32)
    y_np = np.clip(x_np + noise, 0.0, 1.0)

    sk_psnr = psnr_skimage(x_np, y_np, data_range=1.0)

    x_torch = (
        torch.from_numpy(x_np).permute(2, 0, 1).unsqueeze(0)
    )
    y_torch = (
        torch.from_numpy(y_np).permute(2, 0, 1).unsqueeze(0)
    )
    torch_psnr = predictor.compute_psnr_tensor(
        x_torch, y_torch, data_range=1.0
    )

    abs_diff = abs(sk_psnr - torch_psnr)
    assert abs_diff < 1e-5, f"PSNR diff {abs_diff} exceeds threshold 1e-5"


def test_evaluate_psnr_size_mismatch_raises_value_error(tmp_path):
    """Verify size mismatch strictly raises ValueError with guidance."""
    predictor = PSNRPredictor(device="cpu")

    img1_path = tmp_path / "img1.png"
    img2_path = tmp_path / "img2.png"

    Image.new("RGB", (64, 64), color="red").save(img1_path)
    Image.new("RGB", (128, 128), color="red").save(img2_path)

    with pytest.raises(ValueError) as excinfo:
        predictor.evaluate_psnr(str(img1_path), str(img2_path))

    msg = str(excinfo.value)
    assert "Image size mismatch" in msg
    assert "reference" in msg
    assert "(64, 64)" in msg
    assert "(128, 128)" in msg
    assert "PSNR requires identical spatial dimensions" in msg
    assert "super-resolution" in msg or "downsampling" in msg


def test_evaluate_folder_psnr_averages_scores(tmp_path):
    """Verify evaluate_folder_psnr correctly averages paired images."""
    predictor = PSNRPredictor(device="cpu")

    ref_dir = tmp_path / "ref"
    gen_dir = tmp_path / "gen"
    ref_dir.mkdir()
    gen_dir.mkdir()

    # Create 2 image pairs
    im1 = Image.new("RGB", (32, 32), color=(100, 100, 100))
    im1_noisy = Image.new("RGB", (32, 32), color=(110, 110, 110))
    im1.save(ref_dir / "sample1.png")
    im1_noisy.save(gen_dir / "sample1.png")

    im2 = Image.new("RGB", (32, 32), color=(50, 50, 50))
    im2_noisy = Image.new("RGB", (32, 32), color=(70, 70, 70))
    im2.save(ref_dir / "sample2.png")
    im2_noisy.save(gen_dir / "sample2.png")

    s1 = predictor.evaluate_psnr(
        str(ref_dir / "sample1.png"), str(gen_dir / "sample1.png")
    )
    s2 = predictor.evaluate_psnr(
        str(ref_dir / "sample2.png"), str(gen_dir / "sample2.png")
    )
    expected_mean = (s1 + s2) / 2.0

    folder_score = predictor.evaluate_folder_psnr(str(ref_dir), str(gen_dir))
    assert abs(folder_score - expected_mean) < 1e-5
