"""Comprehensive CLI error handling test suite.

Validates ERR-01 through ERR-08, architectural invariants, Python API
semantics, process parity, and JSON success regression in accordance with
DESIGN.md Section 5.
"""

import json
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from image_evaluator.main import CLIInputError, cli, main


@pytest.fixture
def sample_images(tmp_path):
    """Create test image files, directories, and corrupt payload."""
    img_64 = tmp_path / "img_64.png"
    Image.new("RGB", (64, 64), color=(120, 120, 120)).save(img_64)

    ref_64 = tmp_path / "ref_64.png"
    Image.new("RGB", (64, 64), color=(120, 120, 120)).save(ref_64)

    img_128 = tmp_path / "img_128.png"
    Image.new("RGB", (128, 128), color=(200, 200, 200)).save(img_128)

    ref_128 = tmp_path / "ref_128.png"
    Image.new("RGB", (128, 128), color=(200, 200, 200)).save(ref_128)

    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_bytes(b"INVALID_HEADER_NOT_A_REAL_IMAGE_FILE_DATA")

    img_dir = tmp_path / "img_dir"
    img_dir.mkdir()
    Image.new("RGB", (64, 64)).save(img_dir / "a.png")

    ref_dir = tmp_path / "ref_dir"
    ref_dir.mkdir()
    Image.new("RGB", (64, 64)).save(ref_dir / "b.png")

    return {
        "img_64": img_64,
        "ref_64": ref_64,
        "img_128": img_128,
        "ref_128": ref_128,
        "corrupt_file": corrupt_file,
        "img_dir": img_dir,
        "ref_dir": ref_dir,
    }


@pytest.fixture
def mock_all_predictors():
    """Mock all 9 predictor classes to prevent weights or backend load."""
    targets = {
        "aesthetic": (
            "image_evaluator.laion_ai_aesthetic_predictor."
            "LaionAIAestheticPredictor"
        ),
        "clip": "image_evaluator.clip_score_predictor.ClipScorePredictor",
        "arcface": (
            "image_evaluator.arcface_dist_predictor.ArcFaceDistPredictor"
        ),
        "lpips": "image_evaluator.lpips_predictor.LPIPSPredictor",
        "ssim": "image_evaluator.ssim_predictor.SSIMPredictor",
        "psnr": "image_evaluator.psnr_predictor.PSNRPredictor",
        "fid": "image_evaluator.fid_predictor.FIDPredictor",
        "kid": "image_evaluator.kid_predictor.KIDPredictor",
        "pickscore": (
            "image_evaluator.pickscore_predictor.PickScorePredictor"
        ),
    }

    patches = {}
    mocks = {}
    for key, target in targets.items():
        p = patch(target)
        mocks[key] = p.start()
        patches[key] = p

    try:
        yield mocks
    finally:
        for p in patches.values():
            p.stop()


# -------------------------------------------------------------------------
# ERR-01 & ERR-02: Missing Image / Reference (Pairwise File & JSON Mode)
# -------------------------------------------------------------------------


def test_err01_missing_image_file_pairwise(
    mock_all_predictors, sample_images, capsys
):
    """ERR-01: Missing image path in pairwise file mode exits with code 1."""
    nonexistent = "nonexistent_image_file.png"
    ret = cli(
        [
            "--metrics",
            "ssim",
            "--image",
            nonexistent,
            "--reference",
            str(sample_images["ref_64"]),
        ]
    )

    assert ret == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("image-evaluator: error:")
    assert f"Image path does not exist: '{nonexistent}'" in lines[0]
    assert "Traceback" not in captured.err
    for mock in mock_all_predictors.values():
        assert not mock.called


def test_err01_missing_reference_file_pairwise(
    mock_all_predictors, sample_images, capsys
):
    """ERR-01: Missing reference path in pairwise file mode exits code 1."""
    nonexistent = "nonexistent_ref_file.png"
    ret = cli(
        [
            "--metrics",
            "ssim",
            "--image",
            str(sample_images["img_64"]),
            "--reference",
            nonexistent,
        ]
    )

    assert ret == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("image-evaluator: error:")
    assert f"Reference path does not exist: '{nonexistent}'" in lines[0]
    assert "Traceback" not in captured.err
    for mock in mock_all_predictors.values():
        assert not mock.called


def test_err02_missing_image_json_format(
    mock_all_predictors, sample_images, capsys
):
    """ERR-02: Missing image with --format json outputs 0 bytes on stdout."""
    nonexistent = "nonexistent_for_json.png"
    ret = cli(
        [
            "--metrics",
            "ssim",
            "--image",
            nonexistent,
            "--reference",
            str(sample_images["ref_64"]),
            "--format",
            "json",
        ]
    )

    assert ret == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("image-evaluator: error:")
    assert f"Image path does not exist: '{nonexistent}'" in lines[0]
    assert "Traceback" not in captured.err
    for mock in mock_all_predictors.values():
        assert not mock.called


def test_err02_missing_reference_json_format(
    mock_all_predictors, sample_images, capsys
):
    """ERR-02: Missing reference with --format json outputs 0 bytes."""
    nonexistent = "nonexistent_ref_for_json.png"
    ret = cli(
        [
            "--metrics",
            "ssim",
            "--image",
            str(sample_images["img_64"]),
            "--reference",
            nonexistent,
            "--format",
            "json",
        ]
    )

    assert ret == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("image-evaluator: error:")
    assert f"Reference path does not exist: '{nonexistent}'" in lines[0]
    assert "Traceback" not in captured.err
    for mock in mock_all_predictors.values():
        assert not mock.called


# -------------------------------------------------------------------------
# ERR-03: Missing --prompt for clip
# -------------------------------------------------------------------------


def test_err03_missing_prompt_for_clip(
    mock_all_predictors, sample_images, capsys
):
    """ERR-03: Missing --prompt for clip triggers argparse error (exit 2)."""
    with pytest.raises(SystemExit) as excinfo:
        cli(["--metrics", "clip", "--image", str(sample_images["img_64"])])

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert (
        "--prompt is required when 'clip' metric is selected" in captured.err
    )
    assert "Traceback" not in captured.err
    for mock in mock_all_predictors.values():
        assert not mock.called


@pytest.mark.parametrize("empty_prompt", ["", "   ", "\t\n"])
def test_err03_whitespace_prompt_for_clip(
    empty_prompt, mock_all_predictors, sample_images, capsys
):
    """ERR-03: Empty/whitespace --prompt for clip triggers exit code 2."""
    with pytest.raises(SystemExit) as excinfo:
        cli(
            [
                "--metrics",
                "clip",
                "--image",
                str(sample_images["img_64"]),
                "--prompt",
                empty_prompt,
            ]
        )

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert (
        "--prompt is required when 'clip' metric is selected" in captured.err
    )
    assert "Traceback" not in captured.err
    for mock in mock_all_predictors.values():
        assert not mock.called


# -------------------------------------------------------------------------
# ERR-04: Missing --reference for ssim and other reference-based metrics
# -------------------------------------------------------------------------


def test_err04_missing_reference_for_ssim(
    mock_all_predictors, sample_images, capsys
):
    """ERR-04: Missing --reference for ssim triggers exit code 2."""
    with pytest.raises(SystemExit) as excinfo:
        cli(["--metrics", "ssim", "--image", str(sample_images["img_64"])])

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert (
        "--reference is required when 'ssim' metric is selected"
        in captured.err
    )
    assert "Traceback" not in captured.err
    for mock in mock_all_predictors.values():
        assert not mock.called


@pytest.mark.parametrize("metric", ["psnr", "lpips", "arcface"])
def test_err04_missing_reference_for_other_pairwise_metrics(
    metric, mock_all_predictors, sample_images, capsys
):
    """ERR-04: Missing --reference for reference metrics triggers exit 2."""
    with pytest.raises(SystemExit) as excinfo:
        cli(["--metrics", metric, "--image", str(sample_images["img_64"])])

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert (
        f"--reference is required when '{metric}' metric is selected"
        in captured.err
    )
    assert "Traceback" not in captured.err
    for mock in mock_all_predictors.values():
        assert not mock.called


# -------------------------------------------------------------------------
# ERR-05: Dataset metric fid/kid file input instead of directory
# -------------------------------------------------------------------------


def test_err05_fid_image_is_file_instead_of_directory(
    mock_all_predictors, sample_images, capsys
):
    """ERR-05: File passed as --image for fid triggers argparse error."""
    with pytest.raises(SystemExit) as excinfo:
        cli(
            [
                "--metrics",
                "fid",
                "--image",
                str(sample_images["img_64"]),
                "--reference",
                str(sample_images["ref_dir"]),
            ]
        )

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert (
        "--image must be a directory when 'fid' metric is selected"
        in captured.err
    )
    assert "Traceback" not in captured.err
    for mock in mock_all_predictors.values():
        assert not mock.called


def test_err05_kid_reference_is_file_instead_of_directory(
    mock_all_predictors, sample_images, capsys
):
    """ERR-05: File passed as --reference for kid triggers argparse error."""
    with pytest.raises(SystemExit) as excinfo:
        cli(
            [
                "--metrics",
                "kid",
                "--image",
                str(sample_images["img_dir"]),
                "--reference",
                str(sample_images["ref_64"]),
            ]
        )

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert (
        "--reference must be a directory when 'kid' metric is selected"
        in captured.err
    )
    assert "Traceback" not in captured.err
    for mock in mock_all_predictors.values():
        assert not mock.called


def test_err05_fid_path_does_not_exist(
    mock_all_predictors, sample_images, capsys
):
    """ERR-05: Nonexistent path for fid triggers argparse error."""
    with pytest.raises(SystemExit) as excinfo:
        cli(
            [
                "--metrics",
                "fid",
                "--image",
                "nonexistent_dataset_dir",
                "--reference",
                str(sample_images["ref_dir"]),
            ]
        )

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--image path does not exist:" in captured.err
    assert "Traceback" not in captured.err
    for mock in mock_all_predictors.values():
        assert not mock.called


# -------------------------------------------------------------------------
# ERR-06: Corrupt / Non-image file
# -------------------------------------------------------------------------


def test_err06_corrupt_image_file(mock_all_predictors, sample_images, capsys):
    """ERR-06: Corrupt --image file triggers exit 1, no traceback or model."""
    ret = cli(
        [
            "--metrics",
            "ssim",
            "--image",
            str(sample_images["corrupt_file"]),
            "--reference",
            str(sample_images["ref_64"]),
        ]
    )

    assert ret == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("image-evaluator: error:")
    assert "Cannot identify or decode image file" in lines[0]
    assert "Traceback" not in captured.err
    for mock in mock_all_predictors.values():
        assert not mock.called


def test_err06_corrupt_reference_file(
    mock_all_predictors, sample_images, capsys
):
    """ERR-06: Corrupt --reference file triggers exit 1, no traceback."""
    ret = cli(
        [
            "--metrics",
            "ssim",
            "--image",
            str(sample_images["img_64"]),
            "--reference",
            str(sample_images["corrupt_file"]),
        ]
    )

    assert ret == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("image-evaluator: error:")
    assert "Cannot identify or decode image file" in lines[0]
    assert "Traceback" not in captured.err
    for mock in mock_all_predictors.values():
        assert not mock.called


def test_err06_corrupt_single_image_metric(
    mock_all_predictors, sample_images, capsys
):
    """ERR-06: Corrupt image on single-image metric aesthetic exits 1."""
    ret = cli(
        [
            "--metrics",
            "aesthetic",
            "--image",
            str(sample_images["corrupt_file"]),
        ]
    )

    assert ret == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("image-evaluator: error:")
    assert "Cannot identify or decode image file" in lines[0]
    assert "Traceback" not in captured.err
    for mock in mock_all_predictors.values():
        assert not mock.called


# -------------------------------------------------------------------------
# ERR-07: Image Size Mismatch for Dimension-Sensitive Metrics
# -------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["ssim", "psnr", "lpips"])
def test_err07_size_mismatch_dimension_sensitive(
    metric, mock_all_predictors, sample_images, capsys
):
    """ERR-07: Dimension mismatch for ssim/psnr/lpips shows both dimensions."""
    ret = cli(
        [
            "--metrics",
            metric,
            "--image",
            str(sample_images["img_128"]),
            "--reference",
            str(sample_images["ref_64"]),
        ]
    )

    assert ret == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("image-evaluator: error:")
    assert "Image size mismatch" in lines[0]
    assert "(64, 64)" in lines[0]
    assert "(128, 128)" in lines[0]
    assert "Traceback" not in captured.err
    for mock in mock_all_predictors.values():
        assert not mock.called


def test_err07_size_mismatch_multiple_metrics(
    mock_all_predictors, sample_images, capsys
):
    """ERR-07: Combined sensitive metrics report mismatch and sorted list."""
    ret = cli(
        [
            "--metrics",
            "ssim",
            "psnr",
            "lpips",
            "--image",
            str(sample_images["img_128"]),
            "--reference",
            str(sample_images["ref_64"]),
        ]
    )

    assert ret == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("image-evaluator: error:")
    assert "Image size mismatch" in lines[0]
    assert "(64, 64)" in lines[0]
    assert "(128, 128)" in lines[0]
    assert "['lpips', 'psnr', 'ssim']" in lines[0]
    assert "Traceback" not in captured.err
    for mock in mock_all_predictors.values():
        assert not mock.called


# -------------------------------------------------------------------------
# ArcFace Size Tolerance (Contrast with ERR-07)
# -------------------------------------------------------------------------


def test_arcface_size_tolerance_different_dimensions(
    mock_all_predictors, sample_images, capsys
):
    """ArcFace allows different dimensions and does not fail preflight."""
    arc_inst = MagicMock()
    arc_inst.evaluate_arcface_distance.return_value = 0.38
    mock_all_predictors["arcface"].return_value = arc_inst

    ret = cli(
        [
            "--metrics",
            "arcface",
            "--image",
            str(sample_images["img_128"]),
            "--reference",
            str(sample_images["ref_64"]),
        ]
    )

    assert ret == 0
    captured = capsys.readouterr()
    assert "ArcFace Distance: 0.38" in captured.out
    assert captured.err == ""
    assert mock_all_predictors["arcface"].called
    arc_inst.evaluate_arcface_distance.assert_called_once_with(
        str(sample_images["ref_64"]), str(sample_images["img_128"])
    )


# -------------------------------------------------------------------------
# ERR-08: Invalid --format choice
# -------------------------------------------------------------------------


def test_err08_invalid_format_choice(
    mock_all_predictors, sample_images, capsys
):
    """ERR-08: Invalid --format choice triggers exit code 2."""
    with pytest.raises(SystemExit) as excinfo:
        cli(
            [
                "--metrics",
                "ssim",
                "--image",
                str(sample_images["img_64"]),
                "--reference",
                str(sample_images["ref_64"]),
                "--format",
                "xml",
            ]
        )

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid choice: 'xml'" in captured.err
    assert "Traceback" not in captured.err
    for mock in mock_all_predictors.values():
        assert not mock.called


# -------------------------------------------------------------------------
# Internal Exception Sentinel: Unhandled Exceptions Preserve Traceback
# -------------------------------------------------------------------------


def test_internal_exception_sentinel_in_process(
    mock_all_predictors, sample_images
):
    """Unhandled internal exception is NOT masked as CLIInputError."""
    ssim_inst = MagicMock()
    ssim_inst.evaluate_ssim.side_effect = RuntimeError("unexpected cuda crash")
    mock_all_predictors["ssim"].return_value = ssim_inst

    with pytest.raises(RuntimeError, match="unexpected cuda crash"):
        cli(
            [
                "--metrics",
                "ssim",
                "--image",
                str(sample_images["img_64"]),
                "--reference",
                str(sample_images["ref_64"]),
            ]
        )


def test_internal_exception_sentinel_subprocess(sample_images):
    """Unhandled internal exception preserves full Traceback in subprocess."""
    img_path = str(sample_images["img_64"])
    ref_path = str(sample_images["ref_64"])

    code = f"""
import sys
from unittest.mock import MagicMock, patch
from image_evaluator.main import cli

with patch("image_evaluator.ssim_predictor.SSIMPredictor") as mock_cls:
    inst = MagicMock()
    inst.evaluate_ssim.side_effect = RuntimeError("sentinel unexpected fault")
    mock_cls.return_value = inst
    raise SystemExit(
        cli([
            "--metrics", "ssim",
            "--image", {img_path!r},
            "--reference", {ref_path!r},
        ])
    )
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "Traceback (most recent call last):" in proc.stderr
    assert "RuntimeError: sentinel unexpected fault" in proc.stderr
    assert "image-evaluator: error:" not in proc.stderr


# -------------------------------------------------------------------------
# Programmatic API Invariant: main() raises CLIInputError (ValueError)
# -------------------------------------------------------------------------


def test_programmatic_api_invariant_missing_image(
    mock_all_predictors, sample_images
):
    """Direct call to main() raises CLIInputError as a ValueError subclass."""
    with pytest.raises(CLIInputError) as excinfo:
        main(
            [
                "--metrics",
                "ssim",
                "--image",
                "nonexistent.png",
                "--reference",
                str(sample_images["ref_64"]),
            ]
        )

    assert isinstance(excinfo.value, ValueError)
    assert "Image path does not exist" in str(excinfo.value)
    for mock in mock_all_predictors.values():
        assert not mock.called


def test_programmatic_api_invariant_size_mismatch(
    mock_all_predictors, sample_images
):
    """Direct call to main() raises CLIInputError on size mismatch."""
    with pytest.raises(CLIInputError) as excinfo:
        main(
            [
                "--metrics",
                "ssim",
                "--image",
                str(sample_images["img_128"]),
                "--reference",
                str(sample_images["ref_64"]),
            ]
        )

    assert isinstance(excinfo.value, ValueError)
    assert "Image size mismatch" in str(excinfo.value)
    for mock in mock_all_predictors.values():
        assert not mock.called


def test_programmatic_api_invariant_corrupt_image(
    mock_all_predictors, sample_images
):
    """Direct call to main() raises CLIInputError on corrupt image."""
    with pytest.raises(CLIInputError) as excinfo:
        main(
            [
                "--metrics",
                "ssim",
                "--image",
                str(sample_images["corrupt_file"]),
                "--reference",
                str(sample_images["ref_64"]),
            ]
        )

    assert isinstance(excinfo.value, ValueError)
    assert "Cannot identify or decode image file" in str(excinfo.value)
    for mock in mock_all_predictors.values():
        assert not mock.called


# -------------------------------------------------------------------------
# Process Parity: subprocess matches cli() behavior exactly
# -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_id,args_builder",
    [
        (
            "ERR-01",
            lambda s: [
                "--metrics",
                "ssim",
                "--image",
                "missing_parity_img.png",
                "--reference",
                str(s["ref_64"]),
            ],
        ),
        (
            "ERR-02",
            lambda s: [
                "--metrics",
                "ssim",
                "--image",
                "missing_parity_json.png",
                "--reference",
                str(s["ref_64"]),
                "--format",
                "json",
            ],
        ),
        (
            "ERR-06",
            lambda s: [
                "--metrics",
                "ssim",
                "--image",
                str(s["corrupt_file"]),
                "--reference",
                str(s["ref_64"]),
            ],
        ),
        (
            "ERR-07",
            lambda s: [
                "--metrics",
                "ssim",
                "--image",
                str(s["img_128"]),
                "--reference",
                str(s["ref_64"]),
            ],
        ),
    ],
)
def test_process_parity_for_preflight_errors(
    case_id, args_builder, sample_images, capsys
):
    """Verify python -m image_evaluator.main matches cli() in return and IO."""
    args = args_builder(sample_images)

    # 1. In-process invocation via cli()
    ret = cli(args)
    captured = capsys.readouterr()

    # 2. Separate OS subprocess invocation
    cmd = [sys.executable, "-m", "image_evaluator.main", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    # Assert identical exit code
    assert proc.returncode == 1
    assert ret == 1

    # Assert empty stdout on both
    assert proc.stdout == ""
    assert captured.out == ""

    # Assert exact stderr equivalence
    assert proc.stderr.strip() == captured.err.strip()
    assert len(proc.stderr.strip().splitlines()) == 1
    assert proc.stderr.startswith("image-evaluator: error:")
    assert "Traceback" not in proc.stderr


# -------------------------------------------------------------------------
# JSON Success Regression
# -------------------------------------------------------------------------


def test_json_success_regression_in_process(sample_images, capsys):
    """Successful evaluation with --format json outputs valid JSON."""
    with patch("image_evaluator.ssim_predictor.SSIMPredictor") as mock_ssim:
        inst = MagicMock()
        inst.evaluate_ssim.return_value = 0.985
        mock_ssim.return_value = inst

        ret = cli(
            [
                "--metrics",
                "ssim",
                "--image",
                str(sample_images["img_64"]),
                "--reference",
                str(sample_images["ref_64"]),
                "--format",
                "json",
            ]
        )

    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "success"
    assert "ssim" in data["metrics"]
    assert data["metrics"]["ssim"] == pytest.approx(0.985)


def test_json_success_regression_subprocess(sample_images):
    """Subprocess evaluation with --format json succeeds and outputs JSON."""
    cmd = [
        sys.executable,
        "-m",
        "image_evaluator.main",
        "--metrics",
        "ssim",
        "--image",
        str(sample_images["img_64"]),
        "--reference",
        str(sample_images["ref_64"]),
        "--format",
        "json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["status"] == "success"
    assert "ssim" in data["metrics"]
    assert 0.0 <= data["metrics"]["ssim"] <= 1.0
