"""Targeted test suite for model asset download disclosure and gating.

Covers M9-02 requirements:
1. Cached asset without allow_download passes without download.
2. Uncached asset without allow_download raises DownloadNotAllowedError.
3. Uncached asset with allow_download emits disclosure and proceeds.
4. Disclosure emitted only once per asset per session.
5. Unknown estimated size formatting.
6. Python evaluate() / evaluate_detailed() parameter compatibility & summary.
7. CLI --allow-download flag parsing and default behavior.
8. CLI stream separation on DownloadNotAllowedError (stdout 0 bytes, exit 1).
9. Heavy import isolation (torch, transformers, etc. not loaded on import).
10. Optional extras metadata verification (vqa, preference, modern).
11. Package-level export verification for ModelAsset & DownloadNotAllowedError.
"""

import subprocess
import sys
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

import image_evaluator
from image_evaluator.core import evaluate, evaluate_detailed
from image_evaluator.main import cli
from image_evaluator.model_assets import (
    DownloadNotAllowedError,
    ModelAsset,
    check_asset_and_permit_download,
    format_download_disclosure,
)


def test_package_exports():
    """Verify ModelAsset and DownloadNotAllowedError exported from top."""
    assert hasattr(image_evaluator, "ModelAsset")
    assert hasattr(image_evaluator, "DownloadNotAllowedError")
    assert image_evaluator.ModelAsset is ModelAsset
    assert image_evaluator.DownloadNotAllowedError is DownloadNotAllowedError


def test_estimated_size_formatting():
    """Verify size formatting across bytes, KB, MB, GB, and None."""
    asset_bytes = ModelAsset(
        metric_id="test",
        model_id="test-model",
        source="huggingface",
        estimated_download_bytes=512,
    )
    assert asset_bytes.format_estimated_size() == "512 bytes"

    asset_kb = ModelAsset(
        metric_id="test",
        model_id="test-model",
        source="huggingface",
        estimated_download_bytes=2048,
    )
    assert asset_kb.format_estimated_size() == "~2.0 KB (2048 bytes)"

    asset_mb = ModelAsset(
        metric_id="test",
        model_id="test-model",
        source="huggingface",
        estimated_download_bytes=330 * 1024 * 1024,
    )
    assert "330.0 MB" in asset_mb.format_estimated_size()

    asset_gb = ModelAsset(
        metric_id="test",
        model_id="test-model",
        source="huggingface",
        estimated_download_bytes=int(3.2 * 1024 * 1024 * 1024),
    )
    assert "3.2 GB" in asset_gb.format_estimated_size()

    asset_unknown = ModelAsset(
        metric_id="test",
        model_id="test-model",
        source="huggingface",
        estimated_download_bytes=None,
    )
    assert asset_unknown.format_estimated_size() == "unknown"


def test_download_not_allowed_error_message():
    """Verify informative error message formatting with and without extra."""
    asset_with_extra = ModelAsset(
        metric_id="vqascore",
        model_id="clip-flant5-xl",
        source="huggingface",
        revision="v1.0",
        estimated_download_bytes=1024 * 1024 * 100,
        install_extra="vqa",
    )
    err = DownloadNotAllowedError(asset_with_extra)
    msg = str(err)
    assert "vqascore" in msg
    assert "clip-flant5-xl" in msg
    assert "revision: v1.0" in msg
    assert "--allow-download" in msg
    assert "allow_download=True" in msg
    assert "pip install 'image-evaluator[vqa]'" in msg

    asset_without_extra = ModelAsset(
        metric_id="dino",
        model_id="facebook/dinov2-base",
        source="huggingface",
    )
    err2 = DownloadNotAllowedError(asset_without_extra)
    msg2 = str(err2)
    assert "facebook/dinov2-base" in msg2
    assert "pip install" not in msg2


def test_cached_model_no_download():
    """Cached asset returns True and never triggers download callback."""
    asset = ModelAsset(
        metric_id="test",
        model_id="cached-model",
        source="huggingface",
    )
    cb = MagicMock()
    result = check_asset_and_permit_download(
        asset=asset,
        is_cached=True,
        allow_download=False,
        disclosure_callback=cb,
    )
    assert result is True
    cb.assert_not_called()


def test_uncached_model_download_disallowed():
    """Uncached asset raises error when allow_download is False."""
    asset = ModelAsset(
        metric_id="test_metric",
        model_id="uncached-model",
        source="huggingface",
        estimated_download_bytes=5000,
    )
    cb = MagicMock()
    with pytest.raises(DownloadNotAllowedError) as exc_info:
        check_asset_and_permit_download(
            asset=asset,
            is_cached=False,
            allow_download=False,
            disclosure_callback=cb,
        )
    assert exc_info.value.asset is asset
    cb.assert_not_called()


def test_uncached_model_download_allowed():
    """Uncached asset triggers disclosure callback when download allowed."""
    asset = ModelAsset(
        metric_id="test_metric",
        model_id="uncached-model",
        source="huggingface",
        estimated_download_bytes=1024 * 1024 * 50,
    )
    cb = MagicMock()
    result = check_asset_and_permit_download(
        asset=asset,
        is_cached=False,
        allow_download=True,
        disclosure_callback=cb,
    )
    assert result is False
    cb.assert_called_once()
    called_asset, called_msg = cb.call_args[0]
    assert called_asset is asset
    assert "[image-evaluator] Downloading model 'uncached-model'" in called_msg
    assert "50.0 MB" in called_msg


def test_disclosure_emitted_only_once():
    """Multiple checks for same asset in a session emit disclosure once."""
    asset = ModelAsset(
        metric_id="test_metric",
        model_id="repeated-model",
        source="huggingface",
    )
    tracker = set()
    cb = MagicMock()

    check_asset_and_permit_download(
        asset=asset,
        is_cached=False,
        allow_download=True,
        disclosure_callback=cb,
        disclosed_tracker=tracker,
    )
    assert cb.call_count == 1

    # Second invocation with same tracker should not call callback
    check_asset_and_permit_download(
        asset=asset,
        is_cached=False,
        allow_download=True,
        disclosure_callback=cb,
        disclosed_tracker=tracker,
    )
    assert cb.call_count == 1


def test_format_download_disclosure():
    """Test format_download_disclosure formatting."""
    asset = ModelAsset(
        metric_id="clip_i",
        model_id="vit_large_patch14_clip_224.openai",
        source="timm",
        revision="v1.2",
        estimated_download_bytes=3 * 1024 * 1024 * 1024,
    )
    disclosure = format_download_disclosure(asset)
    assert "clip_i" in disclosure
    assert "vit_large_patch14_clip_224.openai" in disclosure
    assert "timm" in disclosure
    assert "(revision: v1.2)" in disclosure
    assert "3.0 GB" in disclosure


def test_core_evaluate_allow_download_param(tmp_path):
    """Verify evaluate() accepts allow_download and records in summary."""
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(img_path)

    # Calling with allow_download=False (default)
    res_default = evaluate_detailed(
        image=str(img_path),
        metrics=["ssim"],
        reference=str(img_path),
    )
    assert res_default.inputs.get("allow_download") is False

    # Calling with explicit allow_download=True
    res_allowed = evaluate_detailed(
        image=str(img_path),
        metrics=["ssim"],
        reference=str(img_path),
        allow_download=True,
    )
    assert res_allowed.inputs.get("allow_download") is True

    # evaluate() accepts both kwargs without error
    cb = MagicMock()
    res_simple = evaluate(
        image=str(img_path),
        metrics=["ssim"],
        reference=str(img_path),
        allow_download=True,
        download_callback=cb,
    )
    assert "ssim" in res_simple


def test_cli_allow_download_flag_parsing():
    """Verify CLI parser parses --allow-download and defaults to False."""
    from image_evaluator.main import main

    with patch("image_evaluator.main._validate_runtime_inputs"), patch(
        "image_evaluator.main._verify_image_file", return_value=(64, 64)
    ), patch("os.path.exists", return_value=True), patch(
        "os.path.isfile", return_value=True
    ), patch(
        "os.path.isdir", return_value=False
    ), patch(
        "image_evaluator.ssim_predictor.SSIMPredictor"
    ) as mock_ssim:
        mock_ssim.return_value.evaluate.return_value = 1.0

        # Without flag
        res_default = main(
            ["--metrics", "ssim", "--image", "a.png", "--reference", "b.png"]
        )
        assert res_default["status"] == "success"

        # With flag
        res_flag = main(
            [
                "--metrics",
                "ssim",
                "--image",
                "a.png",
                "--reference",
                "b.png",
                "--allow-download",
            ]
        )
        assert res_flag["status"] == "success"


def test_cli_download_not_allowed_stream_separation(capsys):
    """Verify exit code is 1, stdout is 0 bytes, stderr has error."""
    asset = ModelAsset(
        metric_id="test_m",
        model_id="mock_model",
        source="huggingface",
    )

    with patch("image_evaluator.main._validate_runtime_inputs"), patch(
        "os.path.exists", return_value=True
    ), patch("os.path.isfile", return_value=True), patch(
        "os.path.isdir", return_value=False
    ), patch(
        "image_evaluator.ssim_predictor.SSIMPredictor"
    ) as mock_ssim:
        mock_ssim.side_effect = DownloadNotAllowedError(asset)

        exit_code = cli(
            [
                "--metrics",
                "ssim",
                "--image",
                "a.png",
                "--reference",
                "b.png",
                "--format",
                "json",
            ]
        )

        captured = capsys.readouterr()
        assert exit_code == 1
        assert captured.out == ""  # Exactly 0 bytes on stdout!
        assert "image-evaluator: error: Model 'mock_model'" in captured.err
        assert "--allow-download" in captured.err


def test_heavy_import_isolation():
    """Verify importing image_evaluator does not import torch/transformers."""
    code = (
        "import sys\n"
        "import image_evaluator\n"
        "from image_evaluator.registry import list_metrics\n"
        "specs = list_metrics()\n"
        "assert len(specs) > 0\n"
        "prohibited = {'torch', 'transformers', 'open_clip', "
        "'accelerate', 'timm'}\n"
        "loaded = prohibited.intersection(sys.modules.keys())\n"
        "if loaded:\n"
        "    print(f'FAILED: {loaded}', file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "sys.exit(0)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"Import isolation failed: {proc.stderr}"


def test_pyproject_extras_configuration():
    """Verify pyproject.toml defines vqa, preference, and modern extras."""
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    assert pyproject_path.exists()

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    extras = data.get("tool", {}).get("poetry", {}).get("extras", {})
    assert "vqa" in extras, "Missing 'vqa' extra"
    assert "preference" in extras, "Missing 'preference' extra"
    assert "modern" in extras, "Missing 'modern' extra"

    assert set(extras["vqa"]) == {"accelerate", "sentencepiece"}
    assert set(extras["preference"]) == {"timm"}
    assert set(extras["modern"]) == {"accelerate", "sentencepiece", "timm"}
