from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

import image_evaluator
from image_evaluator import (
    LaionAIAestheticPredictor,
    PickScorePredictor,
    evaluate,
)


def test_package_exports_and_version():
    """Verify package version, exports, and lazy loading."""
    assert image_evaluator.__version__ == "0.5.0"
    for name in [
        "evaluate",
        "SSIMPredictor",
        "PSNRPredictor",
        "LPIPSPredictor",
        "ClipScorePredictor",
        "LaionAIAestheticPredictor",
        "PickScorePredictor",
        "FIDPredictor",
        "KIDPredictor",
        "ArcFaceDistPredictor",
        "DirectionalClipPredictor",
    ]:
        assert hasattr(image_evaluator, name)
        assert name in dir(image_evaluator)


    with pytest.raises(
        AttributeError, match="has no attribute 'unknown_symbol'"
    ):
        _ = image_evaluator.unknown_symbol


def test_evaluate_ssim_and_psnr_in_memory_pil():
    """Verify evaluate() runs with in-memory PIL images."""
    img1 = Image.new("RGB", (64, 64), color=(100, 150, 200))
    img2 = Image.new("RGB", (64, 64), color=(100, 150, 200))

    results = evaluate(
        metrics=["ssim", "psnr"], image=img1, reference=img2, device="cpu"
    )
    assert "ssim" in results
    assert "psnr" in results
    assert results["ssim"] == pytest.approx(1.0, abs=1e-5)
    assert results["psnr"] == float("inf")


def test_evaluate_ssim_and_psnr_in_memory_tensor():
    """Verify evaluate() runs with in-memory PyTorch tensors."""
    t1 = torch.rand(3, 32, 32)
    t2 = t1.clone()

    results = evaluate(
        metrics=["ssim", "psnr"], image=t1, reference=t2, device="cpu"
    )
    assert results["ssim"] == pytest.approx(1.0, abs=1e-5)
    assert results["psnr"] == float("inf")


def test_evaluate_single_metric_string():
    """Verify evaluate() accepts single metric as string."""
    img = Image.new("RGB", (32, 32), color=(50, 50, 50))
    results = evaluate(metrics="ssim", image=img, reference=img, device="cpu")
    assert list(results.keys()) == ["ssim"]
    assert results["ssim"] == pytest.approx(1.0, abs=1e-5)


def test_evaluate_unsupported_metric_raises():
    img = Image.new("RGB", (32, 32))
    with pytest.raises(ValueError, match="Unsupported metric"):
        evaluate(metrics=["invalid_metric"], image=img)


def test_evaluate_prompt_required_for_clip_and_pickscore():
    img = Image.new("RGB", (32, 32))
    with pytest.raises(
        ValueError, match="prompt is required when 'clip' metric is selected"
    ):
        evaluate(metrics="clip", image=img)

    with pytest.raises(
        ValueError,
        match="prompt is required when 'pickscore' metric is selected",
    ):
        evaluate(metrics="pickscore", image=img)

    with pytest.raises(
        ValueError,
        match="prompt is required when prompt-based metrics are selected",
    ):
        evaluate(metrics=["clip", "pickscore"], image=img)


def test_evaluate_prompt_prohibited_when_no_prompt_metric():
    img = Image.new("RGB", (32, 32))
    with pytest.raises(
        ValueError,
        match="prompt was provided but no prompt-based metric was selected",
    ):
        evaluate(metrics="ssim", image=img, reference=img, prompt="a cat")


def test_evaluate_reference_required_for_pairwise():
    img = Image.new("RGB", (32, 32))
    with pytest.raises(
        ValueError,
        match="reference is required when 'ssim' metric is selected",
    ):
        evaluate(metrics="ssim", image=img)


def test_evaluate_reference_prohibited_when_not_selected():
    img = Image.new("RGB", (32, 32))
    with pytest.raises(
        ValueError,
        match=(
            "reference was provided but no reference-based metric was selected"
        ),
    ):
        with patch.object(
            LaionAIAestheticPredictor,
            "get_aesthetic_model",
            return_value=MagicMock(),
        ):
            evaluate(metrics="aesthetic", image=img, reference=img)


def test_evaluate_dataset_metrics_require_directories(tmp_path):
    file_path = tmp_path / "img.png"
    Image.new("RGB", (32, 32)).save(file_path)

    with pytest.raises(
        ValueError, match="image must be an existing directory when 'fid'"
    ):
        evaluate(metrics="fid", image=str(file_path), reference=str(tmp_path))

    with pytest.raises(
        ValueError, match="reference must be an existing directory when 'kid'"
    ):
        evaluate(metrics="kid", image=str(tmp_path), reference=str(file_path))


def test_evaluate_pairwise_mismatch_folder_and_file(tmp_path):
    file_path = tmp_path / "img.png"
    Image.new("RGB", (32, 32)).save(file_path)

    with pytest.raises(
        ValueError,
        match=(
            "image and reference must both be single images or both be"
            " directories"
        ),
    ):
        evaluate(
            metrics="ssim", image=str(file_path), reference=str(tmp_path)
        )


def test_evaluate_aesthetic_in_memory():
    """Verify evaluate() aesthetic score with mocked model."""
    img = Image.new("RGB", (64, 64))

    with patch(
        "image_evaluator.laion_ai_aesthetic_predictor."
        "open_clip.create_model_and_transforms"
    ) as mock_clip, patch.object(
        LaionAIAestheticPredictor, "get_aesthetic_model"
    ) as mock_aes:
        mock_model = MagicMock()
        mock_model.encode_image.return_value = torch.ones(1, 768)
        mock_clip.return_value = (
            mock_model,
            None,
            lambda x: torch.zeros(3, 224, 224),
        )

        mock_linear = MagicMock()
        mock_score = MagicMock()
        mock_score.item.return_value = 5.88
        mock_linear.return_value = [[mock_score]]
        mock_aes.return_value = mock_linear

        res = evaluate(metrics="aesthetic", image=img)
        assert res["aesthetic"] == pytest.approx(5.88)


def test_evaluate_pickscore_in_memory():
    """Verify evaluate() pickscore with mocked processor and model."""
    img = Image.new("RGB", (64, 64))

    with patch.object(PickScorePredictor, "_load_model") as mock_load:
        mock_model = MagicMock()
        mock_processor = MagicMock()

        mock_processor.return_value = {"input_ids": torch.zeros(1, 5)}
        mock_load.return_value = (mock_model, mock_processor)

        with patch.object(
            PickScorePredictor,
            "compute_score_from_features",
            return_value=19.55,
        ):
            res = evaluate(
                metrics="pickscore",
                image=img,
                prompt="a beautiful painting",
            )
            assert res["pickscore"] == pytest.approx(19.55)


def test_evaluate_clip_in_memory():
    """Verify evaluate() clip with mocked processor and model."""
    img = Image.new("RGB", (64, 64))

    with patch(
        "image_evaluator.clip_score_predictor.AutoModel.from_pretrained"
    ) as mock_model_cls, patch(
        "image_evaluator.clip_score_predictor.AutoProcessor.from_pretrained"
    ) as mock_proc_cls, patch(
        "image_evaluator.clip_score_predictor.AutoTokenizer.from_pretrained"
    ) as mock_tok_cls:
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        feat = torch.tensor([[1.0, 0.0]])
        mock_model.get_image_features.return_value = feat
        mock_model.get_text_features.return_value = feat
        mock_model_cls.return_value = mock_model

        mock_proc = MagicMock()
        mock_proc.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
        mock_proc_cls.return_value = mock_proc

        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": torch.zeros(1, 10)}
        mock_tok_cls.return_value = mock_tok

        res = evaluate(metrics="clip", image=img, prompt="a scenic mountain")
        assert res["clip"] == pytest.approx(1.0)


def test_evaluate_lpips_in_memory():
    """Verify evaluate() lpips with in-memory tensor."""
    t1 = torch.rand(3, 64, 64)
    t2 = t1.clone()

    res = evaluate(metrics="lpips", image=t1, reference=t2, device="cpu")
    assert res["lpips"] == pytest.approx(0.0, abs=1e-5)


def test_evaluate_arcface_in_memory():
    """Verify evaluate() arcface with mocked face analysis."""
    img1 = Image.new("RGB", (112, 112))
    img2 = Image.new("RGB", (112, 112))

    with patch(
        "image_evaluator.arcface_dist_predictor.FaceAnalysis"
    ) as mock_face:
        mock_app = MagicMock()
        face_obj = MagicMock()
        face_obj.embedding = np.array([1.0, 0.0, 0.0])
        mock_app.get.return_value = [face_obj]
        mock_face.return_value = mock_app

        res = evaluate(metrics="arcface", image=img1, reference=img2)
        assert res["arcface"] == pytest.approx(0.0, abs=1e-5)


def test_bare_import_lazy_loading_subprocess():
    """Verify in a clean Python subprocess that bare import does not load
    heavy dependencies into sys.modules.
    """
    import subprocess
    import sys

    code = """
import sys
import image_evaluator

heavy_modules = [
    "torch",
    "transformers",
    "insightface",
    "open_clip",
    "lpips",
    "cleanfid",
]
loaded = [m for m in heavy_modules if m in sys.modules]
if loaded:
    print(f"Loaded: {loaded}")
    sys.exit(1)
sys.exit(0)
"""
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"Module leakage detected: {res.stdout}"


def test_evaluate_float32_precision_preservation():
    """Verify full float32 precision is preserved without 8-bit
    quantization.
    """
    t0 = torch.zeros(3, 32, 32)
    t1 = torch.full((3, 32, 32), 0.001)

    res = evaluate(
        metrics=["ssim", "psnr", "lpips"],
        image=t0,
        reference=t1,
        device="cpu",
    )
    # PSNR should be approx 60 dB (10 * log10(1 / 1e-6) = 60)
    assert 59.9 < res["psnr"] < 60.1
    # SSIM should reflect difference and not be quantized to 1.0
    assert res["ssim"] < 1.0
    # LPIPS distance should be strictly greater than zero
    assert res["lpips"] > 0.0


def test_evaluate_tensor_channel_dimensions():
    """Verify 1, 3, and 4 channel tensor contracts and dimension bounds."""
    # 1-channel 2D and 3D
    t_2d = torch.zeros(32, 32)
    t_3d_1ch = torch.zeros(1, 32, 32)
    t_4d_1ch = torch.zeros(1, 1, 32, 32)
    for t in [t_2d, t_3d_1ch, t_4d_1ch]:
        res = evaluate(metrics="psnr", image=t, reference=t, device="cpu")
        assert res["psnr"] == float("inf")

    # 3-channel 3D and 4D
    t_3d = torch.zeros(3, 32, 32)
    t_4d = torch.zeros(1, 3, 32, 32)
    for t in [t_3d, t_4d]:
        res = evaluate(metrics="ssim", image=t, reference=t, device="cpu")
        assert res["ssim"] == pytest.approx(1.0)

    # 4-channel RGBA: discards alpha channel, compares RGB
    t_rgba = torch.zeros(4, 32, 32)
    t_rgba_ref = torch.zeros(4, 32, 32)
    t_rgba[3, :, :] = 1.0  # Differing alpha channel
    res = evaluate(
        metrics=["ssim", "psnr"],
        image=t_rgba,
        reference=t_rgba_ref,
        device="cpu",
    )
    assert res["ssim"] == pytest.approx(1.0)
    assert res["psnr"] == float("inf")

    # Invalid channels (e.g. 2 channels or 5 channels)
    with pytest.raises(ValueError, match="Expected 1, 3, or 4 channels"):
        evaluate(
            metrics="psnr",
            image=torch.zeros(2, 32, 32),
            reference=torch.zeros(2, 32, 32),
        )

    # Invalid batch size > 1
    with pytest.raises(
        ValueError, match="Expected single-image tensor with batch size 1"
    ):
        evaluate(
            metrics="psnr",
            image=torch.zeros(2, 3, 32, 32),
            reference=torch.zeros(2, 3, 32, 32),
        )


def test_evaluate_numpy_array_support():
    """Verify float32 and uint8 NumPy array support across HWC, CHW, and 2D."""
    # float32 [0.0, 1.0] HWC and CHW
    arr_hwc = np.zeros((32, 32, 3), dtype=np.float32)
    arr_chw = np.zeros((3, 32, 32), dtype=np.float32)
    res = evaluate(
        metrics="psnr", image=arr_hwc, reference=arr_chw, device="cpu"
    )
    assert res["psnr"] == float("inf")

    # uint8 [0, 255] HWC and 2D grayscale
    arr_uint8 = np.ones((32, 32, 3), dtype=np.uint8) * 128
    arr_2d = np.ones((32, 32), dtype=np.uint8) * 128
    res = evaluate(
        metrics="ssim", image=arr_uint8, reference=arr_2d, device="cpu"
    )
    assert res["ssim"] == pytest.approx(1.0)

    # RGBA numpy array
    arr_rgba = np.zeros((32, 32, 4), dtype=np.float32)
    res = evaluate(
        metrics="psnr", image=arr_rgba, reference=arr_hwc, device="cpu"
    )
    assert res["psnr"] == float("inf")


def test_evaluate_device_propagation(tmp_path):
    """Verify device parameter is propagated to FID, KID, and Aesthetic."""
    with (
        patch("image_evaluator.fid_predictor.FIDPredictor") as mock_fid,
        patch("image_evaluator.kid_predictor.KIDPredictor") as mock_kid,
        patch(
            "image_evaluator.laion_ai_aesthetic_predictor.LaionAIAestheticPredictor"
        ) as mock_aes,
    ):
        mock_fid_inst = MagicMock()
        mock_fid_inst.evaluate_folder_fid.return_value = {"fid": 10.0}
        mock_fid.return_value = mock_fid_inst

        mock_kid_inst = MagicMock()
        mock_kid_inst.evaluate_folder_kid.return_value = {"kid": 0.01}
        mock_kid.return_value = mock_kid_inst

        mock_aes_inst = MagicMock()
        mock_aes_inst.evaluate_aesthetic_score.return_value = 6.5
        mock_aes.return_value = mock_aes_inst

        d = tmp_path / "dummy_dir"
        d.mkdir()
        img = tmp_path / "img.png"
        Image.new("RGB", (32, 32)).save(img)

        # Test FID and KID with explicit device
        evaluate(
            metrics=["fid", "kid"],
            image=str(d),
            reference=str(d),
            device="cpu",
        )
        mock_fid.assert_called_once_with(device="cpu")
        mock_kid.assert_called_once_with(device="cpu")

        # Test Aesthetic with explicit device
        evaluate(metrics="aesthetic", image=str(img), device="cpu")
        mock_aes.assert_called_once_with(
            model_name="vit_l_14", device="cpu"
        )


def test_evaluate_directional_clip_in_memory_pil():
    """Verify evaluate() runs directional_clip with in-memory PIL images."""
    img_src = Image.new("RGB", (64, 64), color=(100, 100, 100))
    img_edit = Image.new("RGB", (64, 64), color=(200, 200, 200))

    with (
        patch(
            "image_evaluator.directional_clip_predictor."
            "AutoModel.from_pretrained"
        ) as mock_model_cls,
        patch(
            "image_evaluator.directional_clip_predictor."
            "AutoProcessor.from_pretrained"
        ) as mock_proc_cls,
        patch(
            "image_evaluator.directional_clip_predictor."
            "AutoTokenizer.from_pretrained"
        ) as mock_tok_cls,
    ):
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        feat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        mock_model.get_image_features.return_value = feat
        mock_model.get_text_features.return_value = feat
        mock_model_cls.return_value = mock_model

        mock_proc = MagicMock()
        mock_proc.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
        mock_proc_cls.return_value = mock_proc

        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": torch.zeros(1, 10)}
        mock_tok_cls.return_value = mock_tok

        res = evaluate(
            metrics="directional_clip",
            image=img_edit,
            reference=img_src,
            prompt="a white square",
            source_prompt="a gray square",
            device="cpu",
        )
        assert "directional_clip" in res
        assert isinstance(res["directional_clip"], float)


def test_evaluate_directional_clip_in_memory_tensor():
    """Verify evaluate() runs directional_clip with in-memory tensors."""
    t_src = torch.zeros(3, 32, 32)
    t_edit = torch.ones(3, 32, 32)

    with (
        patch(
            "image_evaluator.directional_clip_predictor."
            "AutoModel.from_pretrained"
        ) as mock_model_cls,
        patch(
            "image_evaluator.directional_clip_predictor."
            "AutoProcessor.from_pretrained"
        ) as mock_proc_cls,
        patch(
            "image_evaluator.directional_clip_predictor."
            "AutoTokenizer.from_pretrained"
        ) as mock_tok_cls,
    ):
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        feat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        mock_model.get_image_features.return_value = feat
        mock_model.get_text_features.return_value = feat
        mock_model_cls.return_value = mock_model

        mock_proc = MagicMock()
        mock_proc.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
        mock_proc_cls.return_value = mock_proc

        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": torch.zeros(1, 10)}
        mock_tok_cls.return_value = mock_tok

        res = evaluate(
            metrics=["directional_clip"],
            image=t_edit,
            reference=t_src,
            prompt="target prompt",
            source_prompt="source prompt",
            device="cpu",
        )
        assert "directional_clip" in res


def test_evaluate_directional_clip_validation_errors():
    """Verify directional_clip parameter validations and error messages."""
    img = Image.new("RGB", (32, 32))

    # Missing prompt
    with pytest.raises(
        ValueError,
        match="prompt is required when 'directional_clip' metric is selected",
    ):
        evaluate(
            metrics="directional_clip",
            image=img,
            reference=img,
            source_prompt="src",
        )

    # Missing reference
    with pytest.raises(
        ValueError,
        match="reference is required when 'directional_clip'",
    ):
        evaluate(
            metrics="directional_clip",
            image=img,
            prompt="tgt",
            source_prompt="src",
        )

    # Missing source_prompt
    with pytest.raises(
        ValueError,
        match=(
            "source_prompt is required when 'directional_clip' "
            "metric is selected"
        ),
    ):
        evaluate(
            metrics="directional_clip",
            image=img,
            reference=img,
            prompt="tgt",
        )

    # Source prompt provided when not selected
    with pytest.raises(
        ValueError,
        match=(
            "source_prompt was provided but 'directional_clip' "
            "metric was not selected"
        ),
    ):
        evaluate(
            metrics="ssim",
            image=img,
            reference=img,
            source_prompt="unused",
        )


def test_evaluate_directional_clip_rejects_directory(tmp_path):
    """Verify directional_clip rejects directory inputs."""
    d = tmp_path / "img_dir"
    d.mkdir()
    img = tmp_path / "single.png"
    Image.new("RGB", (32, 32)).save(img)

    with pytest.raises(
        ValueError,
        match=(
            "image and reference must be single images for "
            "'directional_clip'"
        ),
    ):
        evaluate(
            metrics="directional_clip",
            image=str(d),
            reference=str(img),
            prompt="tgt",
            source_prompt="src",
        )

    with pytest.raises(
        ValueError,
        match=(
            "image and reference must be single images for "
            "'directional_clip'"
        ),
    ):
        evaluate(
            metrics="directional_clip",
            image=str(img),
            reference=str(d),
            prompt="tgt",
            source_prompt="src",
        )


def test_evaluate_mixed_metrics_with_directional_clip():
    """Verify evaluate() runs directional_clip mixed with other metrics."""
    img_src = Image.new("RGB", (32, 32), color=(50, 50, 50))
    img_edit = Image.new("RGB", (32, 32), color=(150, 150, 150))

    with (
        patch(
            "image_evaluator.directional_clip_predictor."
            "AutoModel.from_pretrained"
        ) as mock_model_cls,
        patch(
            "image_evaluator.directional_clip_predictor."
            "AutoProcessor.from_pretrained"
        ) as mock_proc_cls,
        patch(
            "image_evaluator.directional_clip_predictor."
            "AutoTokenizer.from_pretrained"
        ) as mock_tok_cls,
    ):
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        feat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        mock_model.get_image_features.return_value = feat
        mock_model.get_text_features.return_value = feat
        mock_model_cls.return_value = mock_model

        mock_proc = MagicMock()
        mock_proc.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
        mock_proc_cls.return_value = mock_proc

        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": torch.zeros(1, 10)}
        mock_tok_cls.return_value = mock_tok

        res = evaluate(
            metrics=["ssim", "directional_clip"],
            image=img_edit,
            reference=img_src,
            prompt="target",
            source_prompt="source",
            device="cpu",
        )
        assert "ssim" in res
        assert "directional_clip" in res
        assert isinstance(res["ssim"], float)
        assert isinstance(res["directional_clip"], float)


def test_registry_capabilities_single_source_of_truth():
    """Verify core capability sets strictly match Registry specifications."""
    from image_evaluator.core import (
        DATASET_METRICS,
        PAIRWISE_METRICS,
        PROMPT_METRICS,
        REFERENCE_METRICS,
        SOURCE_PROMPT_METRICS,
        SUPPORTED_METRICS,
    )
    from image_evaluator.registry import list_metrics

    specs = list_metrics()
    assert SUPPORTED_METRICS == {s.id for s in specs}
    assert "directional_clip" in SUPPORTED_METRICS
    assert len(SUPPORTED_METRICS) == 10

    assert PROMPT_METRICS == {
        s.id for s in specs if "prompt" in s.inputs.required
    }
    assert "directional_clip" in PROMPT_METRICS

    assert SOURCE_PROMPT_METRICS == {"directional_clip"}
    assert "directional_clip" in REFERENCE_METRICS
    assert DATASET_METRICS == {"fid", "kid"}
    assert PAIRWISE_METRICS == {"arcface", "lpips", "ssim", "psnr"}
