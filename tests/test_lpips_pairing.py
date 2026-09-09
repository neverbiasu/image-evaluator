from unittest.mock import patch

import pytest

from image_evaluator.lpips_predictor import LPIPSPredictor


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
def mocked_predictor():
    with patch("lpips.LPIPS"):
        return LPIPSPredictor(device="cpu")


def test_lpips_pairs_by_stem_ignoring_ext(tmp_path, mocked_predictor):
    r, g = _dirs(tmp_path, ["b.png", "a.jpg"], ["a.png", "b.jpeg"])
    with patch.object(
        mocked_predictor, "evaluate_lpips", side_effect=[0.1, 0.3]
    ):
        out = mocked_predictor.evaluate_folder_lpips(r, g)
    assert out == pytest.approx(0.2)


def test_lpips_missing_stem_raises_filenotfound(tmp_path, mocked_predictor):
    r, g = _dirs(tmp_path, ["a.png"], ["b.png"])
    with pytest.raises(FileNotFoundError):
        mocked_predictor.evaluate_folder_lpips(r, g)


def test_lpips_duplicate_stem_raises_value_error(tmp_path, mocked_predictor):
    r, g = _dirs(tmp_path, ["a.png", "a.jpg"], ["a.png"])
    with pytest.raises(ValueError, match="Duplicate stem"):
        mocked_predictor.evaluate_folder_lpips(r, g)


def test_lpips_unsupported_file_raises_value_error(tmp_path, mocked_predictor):
    r, g = _dirs(tmp_path, ["a.png", "data.csv"], ["a.png"])
    with pytest.raises(ValueError, match="Unsupported"):
        mocked_predictor.evaluate_folder_lpips(r, g)
