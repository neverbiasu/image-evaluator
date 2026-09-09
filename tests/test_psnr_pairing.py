from unittest.mock import patch

import pytest
from PIL import Image

from image_evaluator.psnr_predictor import PSNRPredictor


def test_pairing_matching_stems_different_exts(tmp_path):
    """Verify PSNR folder pairing matches stems across png/jpg."""
    ref_dir = tmp_path / "ref"
    gen_dir = tmp_path / "gen"
    ref_dir.mkdir()
    gen_dir.mkdir()

    Image.new("RGB", (32, 32)).save(ref_dir / "case_01.png")
    Image.new("RGB", (32, 32)).save(gen_dir / "case_01.jpg")

    predictor = PSNRPredictor(device="cpu")
    with patch.object(
        predictor, "evaluate_psnr", return_value=35.5
    ) as mock_eval:
        score = predictor.evaluate_folder_psnr(str(ref_dir), str(gen_dir))
        assert score == 35.5
        assert mock_eval.call_count == 1


def test_pairing_mismatched_stems_raises_file_not_found(tmp_path):
    """Verify PSNR folder pairing raises FileNotFoundError on missing pair."""
    ref_dir = tmp_path / "ref"
    gen_dir = tmp_path / "gen"
    ref_dir.mkdir()
    gen_dir.mkdir()

    Image.new("RGB", (32, 32)).save(ref_dir / "case_01.png")
    Image.new("RGB", (32, 32)).save(gen_dir / "case_02.png")

    predictor = PSNRPredictor(device="cpu")
    with pytest.raises(FileNotFoundError):
        predictor.evaluate_folder_psnr(str(ref_dir), str(gen_dir))
