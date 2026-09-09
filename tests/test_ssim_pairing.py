from unittest.mock import patch

import pytest

from image_evaluator.ssim_predictor import SSIMPredictor


def _mk(d, names):
    for n in names:
        (d / n).write_text("dummy")


def _dirs(p, a_names, b_names):
    a = p / "ref"
    b = p / "gen"
    a.mkdir()
    b.mkdir()
    _mk(a, a_names)
    _mk(b, b_names)
    return str(a), str(b)


@pytest.fixture
def predictor():
    return SSIMPredictor(device="cpu")


def test_ssim_pairs_by_stem_ignoring_ext(tmp_path, predictor):
    r, g = _dirs(tmp_path, ["b.png", "a.jpg"], ["a.png", "b.jpeg"])
    with patch.object(predictor, "evaluate_ssim", side_effect=[0.75, 0.85]):
        out = predictor.evaluate_folder_ssim(r, g)
    assert out == pytest.approx(0.80)


def test_ssim_missing_stem_raises_filenotfound(tmp_path, predictor):
    r, g = _dirs(tmp_path, ["a.png"], ["b.png"])
    with pytest.raises(FileNotFoundError):
        predictor.evaluate_folder_ssim(r, g)


def test_ssim_duplicate_stem_raises_value_error(tmp_path, predictor):
    r, g = _dirs(tmp_path, ["a.png", "a.jpg"], ["a.png"])
    with pytest.raises(ValueError, match="Duplicate stem"):
        predictor.evaluate_folder_ssim(r, g)


def test_ssim_unsupported_file_raises_value_error(tmp_path, predictor):
    r, g = _dirs(tmp_path, ["a.png", "data.csv"], ["a.png"])
    with pytest.raises(ValueError, match="Unsupported"):
        predictor.evaluate_folder_ssim(r, g)


def test_ssim_folder_with_unscorable_small_image_fails_batch(
    tmp_path, predictor
):
    from PIL import Image

    ref_dir = tmp_path / "ref"
    gen_dir = tmp_path / "gen"
    ref_dir.mkdir()
    gen_dir.mkdir()

    # One valid pair and one pair smaller than window size
    Image.new("RGB", (32, 32), (128, 128, 128)).save(ref_dir / "valid.png")
    Image.new("RGB", (32, 32), (128, 128, 128)).save(gen_dir / "valid.png")
    Image.new("RGB", (6, 6), (128, 128, 128)).save(ref_dir / "too_small.png")
    Image.new("RGB", (6, 6), (128, 128, 128)).save(gen_dir / "too_small.png")

    with pytest.raises(
        ValueError, match="smaller than SSIM window_size"
    ):
        predictor.evaluate_folder_ssim(str(ref_dir), str(gen_dir))
