import contextlib
import io
import subprocess
import sys
from unittest.mock import MagicMock, patch

# Lightweight module mocks if optional heavy backbones are not installed
if "insightface" not in sys.modules:
    mock_insightface = MagicMock()
    sys.modules["insightface"] = mock_insightface
    sys.modules["insightface.app"] = mock_insightface.app
if "open_clip" not in sys.modules:
    sys.modules["open_clip"] = MagicMock()
if "torchvision" not in sys.modules:
    mock_torchvision = MagicMock()
    sys.modules["torchvision"] = mock_torchvision
    sys.modules["torchvision.transforms"] = mock_torchvision.transforms

import pytest

from image_evaluator.main import main


@pytest.fixture
def mock_predictors():
    """Mock predictor classes to avoid model loading, network, or GPU usage."""
    with patch(
        "image_evaluator.laion_ai_aesthetic_predictor."
        "LaionAIAestheticPredictor"
    ) as mock_aes_cls, patch(
        "image_evaluator.clip_score_predictor.ClipScorePredictor"
    ) as mock_clip_cls, patch(
        "image_evaluator.arcface_dist_predictor.ArcFaceDistPredictor"
    ) as mock_arc_cls, patch(
        "image_evaluator.lpips_predictor.LPIPSPredictor"
    ) as mock_lpips_cls, patch(
        "image_evaluator.ssim_predictor.SSIMPredictor"
    ) as mock_ssim_cls, patch(
        "image_evaluator.psnr_predictor.PSNRPredictor"
    ) as mock_psnr_cls:
        aes_instance = MagicMock()
        aes_instance.evaluate_aesthetic_score.return_value = 6.5
        aes_instance.evaluate_folder_aesthetic_score.return_value = 6.2
        mock_aes_cls.return_value = aes_instance

        clip_instance = MagicMock()
        clip_instance.evaluate_clip_score.return_value = 0.28
        mock_clip_cls.return_value = clip_instance

        arc_instance = MagicMock()
        arc_instance.evaluate_arcface_distance.return_value = 0.35
        arc_instance.evaluate_folder_arcface_distance.return_value = 0.40
        mock_arc_cls.return_value = arc_instance

        lpips_instance = MagicMock()
        lpips_instance.evaluate_lpips.return_value = 0.15
        lpips_instance.evaluate_folder_lpips.return_value = 0.18
        mock_lpips_cls.return_value = lpips_instance

        ssim_instance = MagicMock()
        ssim_instance.evaluate_ssim.return_value = 0.88
        ssim_instance.evaluate_folder_ssim.return_value = 0.82
        mock_ssim_cls.return_value = ssim_instance

        psnr_instance = MagicMock()
        psnr_instance.evaluate_psnr.return_value = 32.5
        psnr_instance.evaluate_folder_psnr.return_value = 31.0
        mock_psnr_cls.return_value = psnr_instance

        yield {
            "aes_cls": mock_aes_cls,
            "aes_inst": aes_instance,
            "clip_cls": mock_clip_cls,
            "clip_inst": clip_instance,
            "arc_cls": mock_arc_cls,
            "arc_inst": arc_instance,
            "lpips_cls": mock_lpips_cls,
            "lpips_inst": lpips_instance,
            "ssim_cls": mock_ssim_cls,
            "ssim_inst": ssim_instance,
            "psnr_cls": mock_psnr_cls,
            "psnr_inst": psnr_instance,
        }



@pytest.mark.parametrize(
    "metric,module_name,class_name,blocked_modules,extra_args,method_name",
    [
        (
            "aesthetic",
            "image_evaluator.laion_ai_aesthetic_predictor",
            "LaionAIAestheticPredictor",
            [
                "image_evaluator.clip_score_predictor",
                "image_evaluator.arcface_dist_predictor",
                "image_evaluator.lpips_predictor",
                "image_evaluator.ssim_predictor",
                "image_evaluator.psnr_predictor",
            ],
            [],
            "evaluate_aesthetic_score",
        ),
        (
            "clip",
            "image_evaluator.clip_score_predictor",
            "ClipScorePredictor",
            [
                "image_evaluator.laion_ai_aesthetic_predictor",
                "image_evaluator.arcface_dist_predictor",
                "image_evaluator.lpips_predictor",
                "image_evaluator.ssim_predictor",
                "image_evaluator.psnr_predictor",
            ],
            ["--prompt", "a prompt"],
            "evaluate_clip_score",
        ),
        (
            "arcface",
            "image_evaluator.arcface_dist_predictor",
            "ArcFaceDistPredictor",
            [
                "image_evaluator.laion_ai_aesthetic_predictor",
                "image_evaluator.clip_score_predictor",
                "image_evaluator.lpips_predictor",
                "image_evaluator.ssim_predictor",
                "image_evaluator.psnr_predictor",
            ],
            ["--reference", "reference.png"],
            "evaluate_arcface_distance",
        ),
        (
            "lpips",
            "image_evaluator.lpips_predictor",
            "LPIPSPredictor",
            [
                "image_evaluator.laion_ai_aesthetic_predictor",
                "image_evaluator.clip_score_predictor",
                "image_evaluator.arcface_dist_predictor",
                "image_evaluator.ssim_predictor",
                "image_evaluator.psnr_predictor",
            ],
            ["--reference", "reference.png"],
            "evaluate_lpips",
        ),
        (
            "ssim",
            "image_evaluator.ssim_predictor",
            "SSIMPredictor",
            [
                "image_evaluator.laion_ai_aesthetic_predictor",
                "image_evaluator.clip_score_predictor",
                "image_evaluator.arcface_dist_predictor",
                "image_evaluator.lpips_predictor",
                "image_evaluator.psnr_predictor",
            ],
            ["--reference", "reference.png"],
            "evaluate_ssim",
        ),
        (
            "psnr",
            "image_evaluator.psnr_predictor",
            "PSNRPredictor",
            [
                "image_evaluator.laion_ai_aesthetic_predictor",
                "image_evaluator.clip_score_predictor",
                "image_evaluator.arcface_dist_predictor",
                "image_evaluator.lpips_predictor",
                "image_evaluator.ssim_predictor",
            ],
            ["--reference", "reference.png"],
            "evaluate_psnr",
        ),
    ],
)
def test_unselected_predictor_modules_are_not_imported(
    metric,
    module_name,
    class_name,
    blocked_modules,
    extra_args,
    method_name,
):
    """Verify each CLI path imports only its selected predictor module."""
    code = f"""
import sys
import types
from unittest.mock import MagicMock

fake_module = types.ModuleType({module_name!r})
fake_class = MagicMock()
setattr(fake_class.return_value, {method_name!r}, MagicMock(return_value=0.0))
setattr(fake_module, {class_name!r}, fake_class)
sys.modules[{module_name!r}] = fake_module

from image_evaluator.main import main

main([
    "--metrics", {metric!r}, "--image", "image.png", *{extra_args!r}
])
assert fake_class.call_count == 1
for blocked_module in {blocked_modules!r}:
    assert blocked_module not in sys.modules, blocked_module
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_missing_required_arguments():
    """Verify missing --metrics or --image raises SystemExit."""
    with pytest.raises(SystemExit):
        main([])

    with pytest.raises(SystemExit):
        main(["--image", "test.png"])

    with pytest.raises(SystemExit):
        main(["--metrics", "aesthetic"])


def test_invalid_metric_choice():
    """Verify invalid metric name raises SystemExit."""
    with pytest.raises(SystemExit):
        main(["--metrics", "invalid_metric", "--image", "test.png"])


def test_clip_requires_prompt():
    """Verify selecting clip without --prompt raises SystemExit."""
    with pytest.raises(SystemExit):
        main(["--metrics", "clip", "--image", "test.png"])


@pytest.mark.parametrize("empty_prompt", ["", "   ", "\t\n"])
def test_clip_rejects_empty_or_whitespace_prompt(empty_prompt):
    """Verify selecting clip with empty/whitespace prompt raises SystemExit."""
    with pytest.raises(SystemExit):
        main(
            [
                "--metrics",
                "clip",
                "--image",
                "test.png",
                "--prompt",
                empty_prompt,
            ]
        )


def test_prompt_without_clip_rejected():
    """Verify providing --prompt without clip metric raises SystemExit."""
    with pytest.raises(SystemExit):
        main(
            [
                "--metrics",
                "aesthetic",
                "--image",
                "test.png",
                "--prompt",
                "a prompt",
            ]
        )


def test_arcface_requires_reference():
    """Verify selecting arcface without --reference raises SystemExit."""
    with pytest.raises(SystemExit):
        main(["--metrics", "arcface", "--image", "test.png"])


@pytest.mark.parametrize("empty_ref", ["", "   ", "\t\n"])
def test_arcface_rejects_empty_or_whitespace_reference(empty_ref):
    """Verify selecting arcface with empty/whitespace ref raises SystemExit."""
    with pytest.raises(SystemExit):
        main(
            [
                "--metrics",
                "arcface",
                "--image",
                "test.png",
                "--reference",
                empty_ref,
            ]
        )


def test_arcface_rejects_mixed_dir_and_file(mock_predictors, tmp_path):
    """Verify mixed file/directory inputs for arcface fail before init."""
    img_file = tmp_path / "img.png"
    img_file.write_text("dummy")
    ref_dir = tmp_path / "ref_dir"
    ref_dir.mkdir()

    # image is file, reference is directory -> fail
    with pytest.raises(SystemExit):
        main(
            [
                "--metrics",
                "arcface",
                "--image",
                str(img_file),
                "--reference",
                str(ref_dir),
            ]
        )
    assert not mock_predictors["arc_cls"].called

    # image is directory, reference is file -> fail
    img_dir = tmp_path / "img_dir"
    img_dir.mkdir()
    ref_file = tmp_path / "ref.png"
    ref_file.write_text("dummy")

    with pytest.raises(SystemExit):
        main(
            [
                "--metrics",
                "arcface",
                "--image",
                str(img_dir),
                "--reference",
                str(ref_file),
            ]
        )
    assert not mock_predictors["arc_cls"].called


def test_reference_without_arcface_rejected():
    """Verify passing --reference without arcface raises SystemExit."""
    with pytest.raises(SystemExit):
        main(
            [
                "--metrics",
                "aesthetic",
                "--image",
                "test.png",
                "--reference",
                "ref.png",
            ]
        )


def test_selective_execution_aesthetic_only(mock_predictors, tmp_path):
    """Verify only aesthetic predictor is initialized and output."""
    img_file = tmp_path / "sample.png"
    img_file.write_text("dummy")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(["--metrics", "aesthetic", "--image", str(img_file)])

    captured = buf.getvalue()
    assert "LAION AI Aesthetic Score: 6.5" in captured
    assert "CLIP Score:" not in captured
    assert "ArcFace Distance:" not in captured

    assert mock_predictors["aes_cls"].called
    assert not mock_predictors["clip_cls"].called
    assert not mock_predictors["arc_cls"].called


def test_selective_execution_aesthetic_folder(mock_predictors, tmp_path):
    """Verify aesthetic folder dispatch when image path is a directory."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(["--metrics", "aesthetic", "--image", str(img_dir)])

    captured = buf.getvalue()
    assert "LAION AI Aesthetic Score: 6.2" in captured
    assert mock_predictors["aes_inst"].evaluate_folder_aesthetic_score.called
    assert not mock_predictors["aes_inst"].evaluate_aesthetic_score.called


def test_selective_execution_clip_only(mock_predictors, tmp_path):
    """Verify only clip predictor is initialized and output."""
    img_file = tmp_path / "sample.png"
    img_file.write_text("dummy")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(
            [
                "--metrics",
                "clip",
                "--image",
                str(img_file),
                "--prompt",
                "a red square",
            ]
        )

    captured = buf.getvalue()
    assert "CLIP Score: 0.28" in captured
    assert "LAION AI Aesthetic Score:" not in captured
    assert "ArcFace Distance:" not in captured

    assert not mock_predictors["aes_cls"].called
    assert mock_predictors["clip_cls"].called
    assert not mock_predictors["arc_cls"].called


def test_selective_execution_arcface_only(mock_predictors, tmp_path):
    """Verify only arcface predictor is initialized and output."""
    img_file = tmp_path / "sample.png"
    img_file.write_text("dummy")
    ref_file = tmp_path / "ref.png"
    ref_file.write_text("dummy ref")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(
            [
                "--metrics",
                "arcface",
                "--image",
                str(img_file),
                "--reference",
                str(ref_file),
            ]
        )

    captured = buf.getvalue()
    assert "ArcFace Distance: 0.35" in captured
    assert "LAION AI Aesthetic Score:" not in captured
    assert "CLIP Score:" not in captured

    assert not mock_predictors["aes_cls"].called
    assert not mock_predictors["clip_cls"].called
    assert mock_predictors["arc_cls"].called
    assert mock_predictors["arc_inst"].evaluate_arcface_distance.called


def test_selective_execution_arcface_folder(mock_predictors, tmp_path):
    """Verify arcface folder dispatch when both paths are directories."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(
            [
                "--metrics",
                "arcface",
                "--image",
                str(img_dir),
                "--reference",
                str(ref_dir),
            ]
        )

    captured = buf.getvalue()
    assert "ArcFace Distance: 0.4" in captured
    assert mock_predictors["arc_inst"].evaluate_folder_arcface_distance.called


def test_multi_metric_execution(mock_predictors, tmp_path):
    """Verify multi-metric selection initializes all specified predictors."""
    img_file = tmp_path / "sample.png"
    img_file.write_text("dummy")
    ref_file = tmp_path / "ref.png"
    ref_file.write_text("dummy ref")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(
            [
                "--metrics",
                "aesthetic",
                "clip",
                "arcface",
                "--image",
                str(img_file),
                "--prompt",
                "portrait",
                "--reference",
                str(ref_file),
            ]
        )

    captured = buf.getvalue()
    assert "LAION AI Aesthetic Score: 6.5" in captured
    assert "CLIP Score: 0.28" in captured
    assert "ArcFace Distance: 0.35" in captured

    assert mock_predictors["aes_cls"].called
    assert mock_predictors["clip_cls"].called
    assert mock_predictors["arc_cls"].called


def test_lpips_requires_reference():
    """Verify selecting lpips without --reference raises SystemExit."""
    with pytest.raises(SystemExit):
        main(["--metrics", "lpips", "--image", "test.png"])


@pytest.mark.parametrize("empty_ref", ["", "   ", "\t\n"])
def test_lpips_rejects_empty_or_whitespace_reference(empty_ref):
    """Verify selecting lpips with empty reference raises SystemExit."""
    with pytest.raises(SystemExit):
        main(
            [
                "--metrics",
                "lpips",
                "--image",
                "test.png",
                "--reference",
                empty_ref,
            ]
        )


def test_lpips_rejects_mixed_dir_and_file(mock_predictors, tmp_path):
    """Verify mixed file/directory inputs for lpips fail before init."""
    img_file = tmp_path / "img.png"
    img_file.write_text("dummy")
    ref_dir = tmp_path / "ref_dir"
    ref_dir.mkdir()

    with pytest.raises(SystemExit):
        main(
            [
                "--metrics",
                "lpips",
                "--image",
                str(img_file),
                "--reference",
                str(ref_dir),
            ]
        )
    assert not mock_predictors["lpips_cls"].called


def test_selective_execution_lpips_only(mock_predictors, tmp_path):
    """Verify only lpips predictor is initialized and output."""
    img_file = tmp_path / "sample.png"
    img_file.write_text("dummy")
    ref_file = tmp_path / "ref.png"
    ref_file.write_text("dummy ref")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(
            [
                "--metrics",
                "lpips",
                "--image",
                str(img_file),
                "--reference",
                str(ref_file),
            ]
        )

    captured = buf.getvalue()
    assert "LPIPS Distance: 0.15" in captured
    assert "LAION AI Aesthetic Score:" not in captured
    assert "CLIP Score:" not in captured
    assert "ArcFace Distance:" not in captured

    assert mock_predictors["lpips_cls"].called
    assert not mock_predictors["aes_cls"].called
    assert not mock_predictors["clip_cls"].called
    assert not mock_predictors["arc_cls"].called


def test_selective_execution_lpips_folder(mock_predictors, tmp_path):
    """Verify lpips folder dispatch when image path is a directory."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(
            [
                "--metrics",
                "lpips",
                "--image",
                str(img_dir),
                "--reference",
                str(ref_dir),
            ]
        )

    captured = buf.getvalue()
    assert "LPIPS Distance: 0.18" in captured
    assert mock_predictors["lpips_inst"].evaluate_folder_lpips.called
    assert not mock_predictors["lpips_inst"].evaluate_lpips.called


def test_multi_metric_execution_with_lpips(mock_predictors, tmp_path):
    """Verify multi-metric execution with all 4 metrics."""
    img_file = tmp_path / "sample.png"
    img_file.write_text("dummy")
    ref_file = tmp_path / "ref.png"
    ref_file.write_text("dummy ref")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(
            [
                "--metrics",
                "aesthetic",
                "clip",
                "arcface",
                "lpips",
                "--image",
                str(img_file),
                "--prompt",
                "portrait",
                "--reference",
                str(ref_file),
            ]
        )

    captured = buf.getvalue()
    assert "LAION AI Aesthetic Score: 6.5" in captured
    assert "CLIP Score: 0.28" in captured
    assert "ArcFace Distance: 0.35" in captured
    assert "LPIPS Distance: 0.15" in captured

    assert mock_predictors["aes_cls"].called
    assert mock_predictors["clip_cls"].called
    assert mock_predictors["arc_cls"].called
    assert mock_predictors["lpips_cls"].called


def test_ssim_requires_reference(mock_predictors, tmp_path):
    """Verify ssim metric fails if --reference is omitted."""
    img_file = tmp_path / "sample.png"
    img_file.write_text("dummy")

    with pytest.raises(SystemExit):
        main(["--metrics", "ssim", "--image", str(img_file)])
    assert not mock_predictors["ssim_cls"].called


def test_ssim_rejects_empty_or_whitespace_reference(
    mock_predictors, tmp_path
):
    """Verify ssim rejects empty or whitespace-only reference argument."""
    img_file = tmp_path / "sample.png"
    img_file.write_text("dummy")

    with pytest.raises(SystemExit):
        main(
            [
                "--metrics",
                "ssim",
                "--image",
                str(img_file),
                "--reference",
                "   ",
            ]
        )
    assert not mock_predictors["ssim_cls"].called


def test_ssim_rejects_mixed_dir_and_file(mock_predictors, tmp_path):
    """Verify ssim rejects folder image with file reference and vice-versa."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    ref_file = tmp_path / "ref.png"
    ref_file.write_text("dummy")

    with pytest.raises(SystemExit):
        main(
            [
                "--metrics",
                "ssim",
                "--image",
                str(img_dir),
                "--reference",
                str(ref_file),
            ]
        )
    assert not mock_predictors["ssim_cls"].called

    img_file = tmp_path / "image.png"
    img_file.write_text("dummy")
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()

    with pytest.raises(SystemExit):
        main(
            [
                "--metrics",
                "ssim",
                "--image",
                str(img_file),
                "--reference",
                str(ref_dir),
            ]
        )
    assert not mock_predictors["ssim_cls"].called


def test_selective_execution_ssim_only(mock_predictors, tmp_path):
    """Verify only ssim predictor is initialized and output."""
    img_file = tmp_path / "sample.png"
    img_file.write_text("dummy")
    ref_file = tmp_path / "ref.png"
    ref_file.write_text("dummy ref")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(
            [
                "--metrics",
                "ssim",
                "--image",
                str(img_file),
                "--reference",
                str(ref_file),
            ]
        )

    captured = buf.getvalue()
    assert "SSIM: 0.88" in captured
    assert "LAION AI Aesthetic Score:" not in captured
    assert "CLIP Score:" not in captured
    assert "ArcFace Distance:" not in captured
    assert "LPIPS Distance:" not in captured

    assert mock_predictors["ssim_cls"].called
    assert not mock_predictors["aes_cls"].called
    assert not mock_predictors["clip_cls"].called
    assert not mock_predictors["arc_cls"].called
    assert not mock_predictors["lpips_cls"].called


def test_selective_execution_ssim_folder(mock_predictors, tmp_path):
    """Verify ssim folder dispatch when image path is a directory."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(
            [
                "--metrics",
                "ssim",
                "--image",
                str(img_dir),
                "--reference",
                str(ref_dir),
            ]
        )

    captured = buf.getvalue()
    assert "SSIM: 0.82" in captured
    assert mock_predictors["ssim_inst"].evaluate_folder_ssim.called
    assert not mock_predictors["ssim_inst"].evaluate_ssim.called


def test_ssim_cli_fails_for_unscorable_small_image(tmp_path):
    """Verify SSIM via main() raises ValueError for small images."""
    from PIL import Image

    img_file = tmp_path / "small.png"
    Image.new("RGB", (6, 6), (128, 128, 128)).save(img_file)

    with pytest.raises(
        ValueError, match="smaller than SSIM window_size"
    ):
        main(
            [
                "--metrics",
                "ssim",
                "--image",
                str(img_file),
                "--reference",
                str(img_file),
            ]
        )


def test_ssim_cli_subprocess_exits_nonzero_for_small_image(tmp_path):
    """Verify CLI subprocess exits non-zero for unscorable small images."""
    from PIL import Image

    small_img = tmp_path / "small.png"
    valid_img = tmp_path / "valid.png"
    Image.new("RGB", (6, 6), (128, 128, 128)).save(small_img)
    Image.new("RGB", (11, 11), (128, 128, 128)).save(valid_img)

    cmd_fail = [
        sys.executable,
        "-m",
        "image_evaluator.main",
        "--metrics",
        "ssim",
        "--reference",
        str(small_img),
        "--image",
        str(small_img),
    ]
    res_fail = subprocess.run(
        cmd_fail,
        capture_output=True,
        check=False,
        text=True,
    )
    assert res_fail.returncode != 0
    assert "smaller than SSIM window_size" in res_fail.stderr

    cmd_ok = [
        sys.executable,
        "-m",
        "image_evaluator.main",
        "--metrics",
        "ssim",
        "--reference",
        str(valid_img),
        "--image",
        str(valid_img),
    ]
    res_ok = subprocess.run(
        cmd_ok,
        capture_output=True,
        check=False,
        text=True,
    )
    assert res_ok.returncode == 0
    assert "SSIM: 1.0" in res_ok.stdout


def test_multi_metric_execution_all_5_metrics(mock_predictors, tmp_path):
    """Verify multi-metric execution with all 5 metrics."""
    img_file = tmp_path / "sample.png"
    img_file.write_text("dummy")
    ref_file = tmp_path / "ref.png"
    ref_file.write_text("dummy ref")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(
            [
                "--metrics",
                "aesthetic",
                "clip",
                "arcface",
                "lpips",
                "ssim",
                "--image",
                str(img_file),
                "--prompt",
                "portrait",
                "--reference",
                str(ref_file),
            ]
        )

    captured = buf.getvalue()
    assert "LAION AI Aesthetic Score: 6.5" in captured
    assert "CLIP Score: 0.28" in captured
    assert "ArcFace Distance: 0.35" in captured
    assert "LPIPS Distance: 0.15" in captured
    assert "SSIM: 0.88" in captured

    assert mock_predictors["aes_cls"].called
    assert mock_predictors["clip_cls"].called
    assert mock_predictors["arc_cls"].called
    assert mock_predictors["lpips_cls"].called
    assert mock_predictors["ssim_cls"].called


def test_psnr_requires_reference(mock_predictors, tmp_path):
    """Verify psnr metric fails if --reference is omitted."""
    img_file = tmp_path / "sample.png"
    img_file.write_text("dummy")

    with pytest.raises(SystemExit):
        main(["--metrics", "psnr", "--image", str(img_file)])
    assert not mock_predictors["psnr_cls"].called


def test_psnr_rejects_empty_or_whitespace_reference(
    mock_predictors, tmp_path
):
    """Verify psnr rejects empty or whitespace-only reference argument."""
    img_file = tmp_path / "sample.png"
    img_file.write_text("dummy")

    with pytest.raises(SystemExit):
        main(
            [
                "--metrics",
                "psnr",
                "--image",
                str(img_file),
                "--reference",
                "   ",
            ]
        )
    assert not mock_predictors["psnr_cls"].called


def test_psnr_rejects_mixed_dir_and_file(mock_predictors, tmp_path):
    """Verify psnr rejects folder image with file reference and vice-versa."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    ref_file = tmp_path / "ref.png"
    ref_file.write_text("dummy")

    with pytest.raises(SystemExit):
        main(
            [
                "--metrics",
                "psnr",
                "--image",
                str(img_dir),
                "--reference",
                str(ref_file),
            ]
        )
    assert not mock_predictors["psnr_cls"].called

    img_file = tmp_path / "image.png"
    img_file.write_text("dummy")
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()

    with pytest.raises(SystemExit):
        main(
            [
                "--metrics",
                "psnr",
                "--image",
                str(img_file),
                "--reference",
                str(ref_dir),
            ]
        )
    assert not mock_predictors["psnr_cls"].called


def test_selective_execution_psnr_only(mock_predictors, tmp_path):
    """Verify only psnr predictor is initialized and output."""
    img_file = tmp_path / "sample.png"
    img_file.write_text("dummy")
    ref_file = tmp_path / "ref.png"
    ref_file.write_text("dummy ref")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(
            [
                "--metrics",
                "psnr",
                "--image",
                str(img_file),
                "--reference",
                str(ref_file),
            ]
        )

    captured = buf.getvalue()
    assert "PSNR: 32.5" in captured
    assert "LAION AI Aesthetic Score:" not in captured
    assert "CLIP Score:" not in captured
    assert "ArcFace Distance:" not in captured
    assert "LPIPS Distance:" not in captured
    assert "SSIM:" not in captured

    assert mock_predictors["psnr_cls"].called
    assert not mock_predictors["aes_cls"].called
    assert not mock_predictors["clip_cls"].called
    assert not mock_predictors["arc_cls"].called
    assert not mock_predictors["lpips_cls"].called
    assert not mock_predictors["ssim_cls"].called


def test_selective_execution_psnr_folder(mock_predictors, tmp_path):
    """Verify psnr folder dispatch when image path is a directory."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(
            [
                "--metrics",
                "psnr",
                "--image",
                str(img_dir),
                "--reference",
                str(ref_dir),
            ]
        )

    captured = buf.getvalue()
    assert "PSNR: 31.0" in captured
    assert mock_predictors["psnr_inst"].evaluate_folder_psnr.called
    assert not mock_predictors["psnr_inst"].evaluate_psnr.called


def test_multi_metric_execution_all_6_metrics(mock_predictors, tmp_path):
    """Verify multi-metric execution with all 6 metrics."""
    img_file = tmp_path / "sample.png"
    img_file.write_text("dummy")
    ref_file = tmp_path / "ref.png"
    ref_file.write_text("dummy ref")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(
            [
                "--metrics",
                "aesthetic",
                "clip",
                "arcface",
                "lpips",
                "ssim",
                "psnr",
                "--image",
                str(img_file),
                "--prompt",
                "portrait",
                "--reference",
                str(ref_file),
            ]
        )

    captured = buf.getvalue()
    assert "LAION AI Aesthetic Score: 6.5" in captured
    assert "CLIP Score: 0.28" in captured
    assert "ArcFace Distance: 0.35" in captured
    assert "LPIPS Distance: 0.15" in captured
    assert "SSIM: 0.88" in captured
    assert "PSNR: 32.5" in captured

    assert mock_predictors["aes_cls"].called
    assert mock_predictors["clip_cls"].called
    assert mock_predictors["arc_cls"].called
    assert mock_predictors["lpips_cls"].called
    assert mock_predictors["ssim_cls"].called
    assert mock_predictors["psnr_cls"].called
