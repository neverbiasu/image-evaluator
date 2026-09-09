from unittest.mock import patch

import pytest

from image_evaluator.arcface_dist_predictor import ArcFaceDistPredictor


def _mk(d, names):
    for n in names:
        (d / n).write_text("x")


def _pred():
    with patch("image_evaluator.arcface_dist_predictor.FaceAnalysis"):
        return ArcFaceDistPredictor(device="cpu")


def _dirs(p, a_names, b_names):
    a = p / "ref"
    b = p / "gen"
    a.mkdir()
    b.mkdir()
    _mk(a, a_names)
    _mk(b, b_names)
    return str(a), str(b)


def test_pairs_by_stem_ignore_ext(tmp_path):
    r, g = _dirs(tmp_path, ["b.png", "a.jpg"], ["a.png", "b.jpeg"])
    pred = _pred()
    with patch.object(
        pred, "evaluate_arcface_distance", side_effect=[0.2, 0.4]
    ):
        out = pred.evaluate_folder_arcface_distance(r, g)
    assert out == pytest.approx(0.3)


def test_missing_stem_raises(tmp_path):
    r, g = _dirs(tmp_path, ["a.png"], ["b.png"])
    with pytest.raises(FileNotFoundError):
        _pred().evaluate_folder_arcface_distance(r, g)


def test_duplicate_stem_raises(tmp_path):
    r, g = _dirs(tmp_path, ["a.png", "a.jpg"], ["a.png"])
    with pytest.raises(ValueError, match="Duplicate stem"):
        _pred().evaluate_folder_arcface_distance(r, g)


def test_unsupported_visible_raises(tmp_path):
    r, g = _dirs(tmp_path, ["a.png", "x.csv"], ["a.png"])
    with pytest.raises(ValueError, match="Unsupported"):
        _pred().evaluate_folder_arcface_distance(r, g)


def test_unscorable_fails_whole_batch(tmp_path):
    r, g = _dirs(tmp_path, ["a.png", "b.png"], ["a.png", "b.png"])
    p = _pred()
    with patch.object(p, "evaluate_arcface_distance",
                      side_effect=[0.1, None]):
        with pytest.raises(ValueError, match="Unscorable"):
            p.evaluate_folder_arcface_distance(r, g)
