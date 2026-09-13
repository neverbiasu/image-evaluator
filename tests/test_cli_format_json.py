import json
import math
import subprocess
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from image_evaluator.main import main


def test_cli_format_invalid_choice():
    """Verify invalid --format choice exits with code 2."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--metrics", "ssim", "--image", "a.png", "--format", "xml"])
    assert excinfo.value.code == 2


def test_cli_format_json_single_metric(tmp_path, capsys):
    """Verify --format json outputs valid JSON to stdout."""
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    Image.new("RGB", (32, 32), color=(100, 100, 100)).save(img1)
    Image.new("RGB", (32, 32), color=(100, 100, 100)).save(img2)

    with patch("image_evaluator.ssim_predictor.SSIMPredictor") as mock_ssim:
        inst = MagicMock()
        inst.evaluate_ssim.return_value = 0.9876
        mock_ssim.return_value = inst

        res = main(
            [
                "--metrics",
                "ssim",
                "--image",
                str(img1),
                "--reference",
                str(img2),
                "--format",
                "json",
            ]
        )

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "success"
    assert "ssim" in data["metrics"]
    assert data["metrics"]["ssim"] == pytest.approx(0.9876)
    assert res == data


def test_cli_format_json_psnr_infinite(tmp_path, capsys):
    """Verify PSNR infinite value produces RFC 8259 compliant null in JSON."""
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    Image.new("RGB", (32, 32)).save(img1)
    Image.new("RGB", (32, 32)).save(img2)

    with patch("image_evaluator.psnr_predictor.PSNRPredictor") as mock_psnr:
        inst = MagicMock()
        inst.evaluate_psnr.return_value = float("inf")
        mock_psnr.return_value = inst

        res = main(
            [
                "--metrics",
                "psnr",
                "--image",
                str(img1),
                "--reference",
                str(img2),
                "--format",
                "json",
            ]
        )

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["metrics"]["psnr"] is None
    assert data["metrics"]["psnr_raw"] == "inf"
    assert res == data


def test_cli_format_json_multiple_metrics(tmp_path, capsys):
    """Verify --format json with multiple metrics including dataset."""
    ref_dir = tmp_path / "ref"
    gen_dir = tmp_path / "gen"
    ref_dir.mkdir()
    gen_dir.mkdir()

    with (
        patch("image_evaluator.fid_predictor.FIDPredictor") as mock_fid,
        patch("image_evaluator.kid_predictor.KIDPredictor") as mock_kid,
    ):
        fid_inst = MagicMock()
        fid_inst.evaluate_folder_fid.return_value = {
            "fid": 14.5,
            "backend": "clean-fid",
            "version": "0.1.35",
            "mode": "clean",
            "model": "inception_v3",
            "device": "cpu",
            "Nref": 20,
            "Ngen": 20,
        }
        mock_fid.return_value = fid_inst

        kid_inst = MagicMock()
        kid_inst.evaluate_folder_kid.return_value = {
            "kid": 0.005,
            "backend": "clean-fid",
            "version": "0.1.35",
            "mode": "clean",
            "model": "inception_v3",
            "device": "cpu",
            "num_subsets": 100,
            "max_subset_size": 1000,
            "seed": 0,
            "Nref": 20,
            "Ngen": 20,
        }
        mock_kid.return_value = kid_inst

        res = main(
            [
                "--metrics",
                "fid",
                "kid",
                "--image",
                str(gen_dir),
                "--reference",
                str(ref_dir),
                "--format",
                "json",
            ]
        )

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "success"
    assert data["metrics"]["fid"]["fid"] == 14.5
    assert data["metrics"]["kid"]["kid"] == 0.005
    assert res == data


def test_cli_subprocess_stdout_purity_with_json(tmp_path):
    """Test pure stdout output in separate subprocess mimicking pipeline."""
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    Image.new("RGB", (32, 32)).save(img1)
    Image.new("RGB", (32, 32)).save(img2)

    cmd = [
        sys.executable,
        "-m",
        "image_evaluator.main",
        "--metrics",
        "ssim",
        "--image",
        str(img1),
        "--reference",
        str(img2),
        "--format",
        "json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["status"] == "success"
    assert "ssim" in data["metrics"]
    assert 0.0 <= data["metrics"]["ssim"] <= 1.0


def test_cli_format_json_nan_and_negative_inf(tmp_path, capsys):
    """Verify non-finite float values (NaN, -Inf) are serialized as null."""
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    Image.new("RGB", (32, 32)).save(img1)
    Image.new("RGB", (32, 32)).save(img2)

    with (
        patch("image_evaluator.ssim_predictor.SSIMPredictor") as mock_ssim,
        patch("image_evaluator.psnr_predictor.PSNRPredictor") as mock_psnr,
    ):
        ssim_inst = MagicMock()
        ssim_inst.evaluate_ssim.return_value = float("nan")
        mock_ssim.return_value = ssim_inst

        psnr_inst = MagicMock()
        psnr_inst.evaluate_psnr.return_value = float("-inf")
        mock_psnr.return_value = psnr_inst

        res = main(
            [
                "--metrics",
                "ssim",
                "psnr",
                "--image",
                str(img1),
                "--reference",
                str(img2),
                "--format",
                "json",
            ]
        )

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["metrics"]["ssim"] is None
    assert data["metrics"]["psnr"] is None
    assert data["metrics"]["psnr_raw"] == "-inf"
    assert res == data
    assert res["metrics"]["ssim"] is None
    assert res["metrics"]["psnr"] is None


def test_cli_format_json_nested_non_finite_in_dataset_metric(
    tmp_path, capsys
):
    """Verify nested non-finite values in FID/KID dicts become null."""
    ref_dir = tmp_path / "ref"
    gen_dir = tmp_path / "gen"
    ref_dir.mkdir()
    gen_dir.mkdir()

    with patch("image_evaluator.fid_predictor.FIDPredictor") as mock_fid:
        fid_inst = MagicMock()
        fid_inst.evaluate_folder_fid.return_value = {
            "fid": float("nan"),
            "score": float("inf"),
            "backend": "clean-fid",
            "version": "0.1.35",
            "mode": "clean",
            "model": "inception_v3",
            "device": "cpu",
            "Nref": 20,
            "Ngen": 20,
        }
        mock_fid.return_value = fid_inst

        res = main(
            [
                "--metrics",
                "fid",
                "--image",
                str(gen_dir),
                "--reference",
                str(ref_dir),
                "--format",
                "json",
            ]
        )

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "success"
    assert data["metrics"]["fid"]["fid"] is None
    assert data["metrics"]["fid"]["score"] is None
    assert data["metrics"]["fid"]["backend"] == "clean-fid"
    assert res == data
    assert res["metrics"]["fid"]["fid"] is None


def test_cli_format_json_numpy_float32_finite_and_non_finite(
    tmp_path, capsys
):
    """Verify np.float32 scalars are converted to float or null in JSON mode.

    main() return value must match parsed stdout JSON, with no TypeError
    from json.dumps(..., allow_nan=False).
    """
    ref_dir = tmp_path / "ref"
    gen_dir = tmp_path / "gen"
    ref_dir.mkdir()
    gen_dir.mkdir()

    with patch("image_evaluator.fid_predictor.FIDPredictor") as mock_fid:
        fid_inst = MagicMock()
        fid_inst.evaluate_folder_fid.return_value = {
            "fid": np.float32(12.34),
            "nan_metric": np.float32("nan"),
            "inf_metric": np.float32("inf"),
            "ninf_metric": np.float32("-inf"),
            "backend": "clean-fid",
        }
        mock_fid.return_value = fid_inst

        res = main(
            [
                "--metrics",
                "fid",
                "--image",
                str(gen_dir),
                "--reference",
                str(ref_dir),
                "--format",
                "json",
            ]
        )

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert res == data
    fid_data = data["metrics"]["fid"]
    assert isinstance(fid_data["fid"], float)
    assert fid_data["fid"] == pytest.approx(12.34, abs=1e-4)
    assert fid_data["nan_metric"] is None
    assert fid_data["inf_metric"] is None
    assert fid_data["ninf_metric"] is None


def test_cli_format_text_preserves_raw_return(tmp_path, capsys):
    """Verify non-JSON text mode returns raw un-normalized results."""
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    Image.new("RGB", (32, 32)).save(img1)
    Image.new("RGB", (32, 32)).save(img2)

    with patch("image_evaluator.ssim_predictor.SSIMPredictor") as mock_ssim:
        ssim_inst = MagicMock()
        ssim_inst.evaluate_ssim.return_value = float("nan")
        mock_ssim.return_value = ssim_inst

        res = main(
            [
                "--metrics",
                "ssim",
                "--image",
                str(img1),
                "--reference",
                str(img2),
                "--format",
                "text",
            ]
        )

    # In text mode, the raw return dict retains the float("nan") object
    assert math.isnan(res["metrics"]["ssim"])


def test_cli_format_json_boolean_semantics(tmp_path, capsys):
    """Verify boolean values and NumPy booleans retain JSON boolean type.

    Ensures bool is not converted to int (due to bool subclassing int),
    and exact types are bool in both returned dict and parsed JSON stdout.
    """
    ref_dir = tmp_path / "ref"
    gen_dir = tmp_path / "gen"
    ref_dir.mkdir()
    gen_dir.mkdir()

    with patch("image_evaluator.fid_predictor.FIDPredictor") as mock_fid:
        fid_inst = MagicMock()
        fid_inst.evaluate_folder_fid.return_value = {
            "flag_py_true": True,
            "flag_py_false": False,
            "flag_np_true": np.bool_(True),
            "flag_np_false": np.bool_(False),
            "count": 1,
            "backend": "clean-fid",
        }
        mock_fid.return_value = fid_inst

        res = main(
            [
                "--metrics",
                "fid",
                "--image",
                str(gen_dir),
                "--reference",
                str(ref_dir),
                "--format",
                "json",
            ]
        )

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert res == data

    metrics_res = res["metrics"]["fid"]
    metrics_json = data["metrics"]["fid"]

    for m in (metrics_res, metrics_json):
        assert m["flag_py_true"] is True
        assert type(m["flag_py_true"]) is bool
        assert m["flag_py_false"] is False
        assert type(m["flag_py_false"]) is bool
        assert m["flag_np_true"] is True
        assert type(m["flag_np_true"]) is bool
        assert m["flag_np_false"] is False
        assert type(m["flag_np_false"]) is bool
        assert m["count"] == 1
        assert type(m["count"]) is int

    # Verify JSON text literals in stdout
    assert '"flag_py_true": true' in captured.out
    assert '"flag_py_false": false' in captured.out
    assert '"flag_np_true": true' in captured.out
    assert '"flag_np_false": false' in captured.out
    assert '"count": 1' in captured.out
