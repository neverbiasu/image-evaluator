from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from image_evaluator.lpips_predictor import LPIPSPredictor


@pytest.fixture
def predictor():
    with patch("lpips.LPIPS") as mock_lpips_cls:
        loss_fn = MagicMock()
        mock_lpips_cls.return_value = loss_fn
        pred = LPIPSPredictor(device="cpu")
        return pred


def test_init_sets_up_model_on_requested_device():
    with patch("lpips.LPIPS") as mock_lpips_cls:
        loss_fn = MagicMock()
        mock_lpips_cls.return_value = loss_fn

        pred = LPIPSPredictor(net="alex", version="0.1", device="cpu")
        mock_lpips_cls.assert_called_once_with(net="alex", version="0.1")
        loss_fn.to.assert_called_once_with(torch.device("cpu"))
        loss_fn.eval.assert_called_once()
        assert pred.device == torch.device("cpu")


def test_evaluate_lpips_identical_images_zero(tmp_path):
    pred = LPIPSPredictor(device="cpu")
    img_path1 = tmp_path / "img1.png"
    img_path2 = tmp_path / "img2.png"

    arr = np.ones((64, 64, 3), dtype=np.uint8) * 128
    Image.fromarray(arr).save(img_path1)
    Image.fromarray(arr).save(img_path2)

    dist = pred.evaluate_lpips(str(img_path1), str(img_path2))
    assert dist == pytest.approx(0.0, abs=1e-5)


def test_evaluate_lpips_size_mismatch_raises_value_error(tmp_path):
    pred = LPIPSPredictor(device="cpu")
    ref_path = tmp_path / "ref.png"
    gen_path = tmp_path / "gen.png"

    Image.new("RGB", (64, 64)).save(ref_path)
    Image.new("RGB", (128, 128)).save(gen_path)

    with pytest.raises(ValueError, match="Image size mismatch"):
        pred.evaluate_lpips(str(ref_path), str(gen_path))


def test_evaluate_folder_lpips_averages_scores(tmp_path):
    ref_dir = tmp_path / "ref"
    gen_dir = tmp_path / "gen"
    ref_dir.mkdir()
    gen_dir.mkdir()

    for name in ["sample_a.png", "sample_b.png"]:
        Image.new("RGB", (32, 32)).save(ref_dir / name)
        Image.new("RGB", (32, 32)).save(gen_dir / name)

    pred = LPIPSPredictor(device="cpu")
    with patch.object(pred, "evaluate_lpips", side_effect=[0.12, 0.24]):
        score = pred.evaluate_folder_lpips(str(ref_dir), str(gen_dir))

    assert score == pytest.approx(0.18)
