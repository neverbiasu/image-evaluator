import numpy as np
import pytest
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim_sk

from image_evaluator.ssim_predictor import SSIMPredictor


@pytest.fixture
def predictor():
    return SSIMPredictor(device="cpu")


def test_init_sets_up_gaussian_kernel_and_device():
    pred = SSIMPredictor(window_size=11, sigma=1.5, device="cpu")
    assert pred.window_size == 11
    assert pred.sigma == 1.5
    assert pred.pad == 5
    assert pred.window.shape == (3, 1, 11, 11)


def test_evaluate_ssim_identical_images_is_one(predictor, tmp_path):
    img_path1 = tmp_path / "img1.png"
    img_path2 = tmp_path / "img2.png"

    arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    Image.fromarray(arr).save(img_path1)
    Image.fromarray(arr).save(img_path2)

    score = predictor.evaluate_ssim(str(img_path1), str(img_path2))
    assert score == pytest.approx(1.0, abs=1e-5)


def test_evaluate_ssim_black_vs_white_near_zero(predictor, tmp_path):
    black_path = tmp_path / "black.png"
    white_path = tmp_path / "white.png"

    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(black_path)
    Image.new("RGB", (64, 64), color=(255, 255, 255)).save(white_path)

    score = predictor.evaluate_ssim(str(black_path), str(white_path))
    assert score == pytest.approx(0.0, abs=1e-3)


def test_ssim_matches_skimage_wang_2004_reference(predictor, tmp_path):
    p1 = tmp_path / "sample_a.png"
    p2 = tmp_path / "sample_b.png"

    np.random.seed(42)
    arr1 = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    noise = np.random.normal(0, 15, arr1.shape)
    arr2 = np.clip(arr1.astype(float) + noise, 0, 255).astype(np.uint8)

    Image.fromarray(arr1).save(p1)
    Image.fromarray(arr2).save(p2)

    score_pred = predictor.evaluate_ssim(str(p1), str(p2))

    # Reference skimage computation with exact Wang et al. 2004 parameters
    norm1 = arr1.astype(np.float32) / 255.0
    norm2 = arr2.astype(np.float32) / 255.0
    score_sk = ssim_sk(
        norm1,
        norm2,
        channel_axis=-1,
        data_range=1.0,
        gaussian_weights=True,
        win_size=11,
        sigma=1.5,
        use_sample_covariance=False,
    )

    assert score_pred == pytest.approx(score_sk, abs=1e-4)


def test_evaluate_ssim_size_mismatch_raises_value_error(predictor, tmp_path):
    ref_path = tmp_path / "ref.png"
    gen_path = tmp_path / "gen.png"

    Image.new("RGB", (64, 64)).save(ref_path)
    Image.new("RGB", (128, 128)).save(gen_path)

    with pytest.raises(ValueError, match="Image size mismatch"):
        predictor.evaluate_ssim(str(ref_path), str(gen_path))


def test_evaluate_folder_ssim_averages_scores(predictor, tmp_path):
    ref_dir = tmp_path / "ref"
    gen_dir = tmp_path / "gen"
    ref_dir.mkdir()
    gen_dir.mkdir()

    for name in ["sample_a.png", "sample_b.png"]:
        Image.new("RGB", (32, 32)).save(ref_dir / name)
        Image.new("RGB", (32, 32)).save(gen_dir / name)

    from unittest.mock import patch

    with patch.object(predictor, "evaluate_ssim", side_effect=[0.85, 0.95]):
        score = predictor.evaluate_folder_ssim(str(ref_dir), str(gen_dir))

    assert score == pytest.approx(0.90)


@pytest.mark.parametrize("size", [(6, 6), (10, 10)])
def test_evaluate_ssim_small_image_raises_value_error(
    predictor, tmp_path, size
):
    p1 = tmp_path / "img1.png"
    p2 = tmp_path / "img2.png"
    Image.new("RGB", size, color=(128, 128, 128)).save(p1)
    Image.new("RGB", size, color=(128, 128, 128)).save(p2)

    with pytest.raises(
        ValueError, match="smaller than SSIM window_size"
    ):
        predictor.evaluate_ssim(str(p1), str(p2))


@pytest.mark.parametrize("size", [(10, 20), (20, 10)])
def test_evaluate_ssim_one_dimension_smaller_raises_value_error(
    predictor, tmp_path, size
):
    p1 = tmp_path / "img1.png"
    p2 = tmp_path / "img2.png"
    Image.new("RGB", size, color=(128, 128, 128)).save(p1)
    Image.new("RGB", size, color=(128, 128, 128)).save(p2)

    with pytest.raises(
        ValueError, match="smaller than SSIM window_size"
    ):
        predictor.evaluate_ssim(str(p1), str(p2))


def test_evaluate_ssim_11x11_boundary_success(predictor, tmp_path):
    p1 = tmp_path / "img1.png"
    p2 = tmp_path / "img2.png"
    Image.new("RGB", (11, 11), color=(128, 128, 128)).save(p1)
    Image.new("RGB", (11, 11), color=(128, 128, 128)).save(p2)

    score = predictor.evaluate_ssim(str(p1), str(p2))
    assert score == pytest.approx(1.0, abs=1e-5)


def test_compute_ssim_tensor_dimensions_too_small_raises_value_error(
    predictor,
):
    t1 = torch.rand(1, 3, 10, 10)
    t2 = torch.rand(1, 3, 10, 10)
    with pytest.raises(
        ValueError, match="smaller than SSIM window_size"
    ):
        predictor.compute_ssim_tensor(t1, t2)


def test_compute_ssim_tensor_shape_mismatch_raises_value_error(predictor):
    t1 = torch.rand(1, 3, 32, 32)
    t2 = torch.rand(1, 3, 16, 16)
    with pytest.raises(ValueError, match="Tensor shape mismatch"):
        predictor.compute_ssim_tensor(t1, t2)
