"""Tests for Directional CLIP model asset download gate and caching checks.

Verifies:
1. Complete cache + allow_download=False: constructs cleanly, pinned revision
   and local_files_only=True passed to all 3 components.
2. Empty cache + allow_download=False: raises DownloadNotAllowedError, 0 calls
   to from_pretrained.
3. Partial cache + allow_download=False: raises DownloadNotAllowedError, proves
   full snapshot required.
4. Empty cache + allow_download=True: disclosure callback called exactly once,
   pinned revision and local_files_only=False passed to all 3 components.
5. Component failure during init leaves no half-initialized state.
6. core.evaluate(..., metrics=["directional_clip"]) passes allow_download
   and download_callback to DirectionalClipPredictor.
"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from image_evaluator.core import evaluate
from image_evaluator.directional_clip_predictor import (
    DIRECTIONAL_CLIP_ASSET,
    DIRECTIONAL_CLIP_REQUIRED_FILES,
    DirectionalClipPredictor,
    _is_directional_clip_cached,
)
from image_evaluator.model_assets import DownloadNotAllowedError


def test_directional_clip_asset_metadata():
    """Verify DIRECTIONAL_CLIP_ASSET metadata conforms to specification."""
    assert DIRECTIONAL_CLIP_ASSET.metric_id == "directional_clip"
    assert DIRECTIONAL_CLIP_ASSET.model_id == "openai/clip-vit-base-patch32"
    assert DIRECTIONAL_CLIP_ASSET.source == "huggingface"
    expected_rev = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
    assert DIRECTIONAL_CLIP_ASSET.revision == expected_rev
    assert len(DIRECTIONAL_CLIP_ASSET.revision) == 40
    assert DIRECTIONAL_CLIP_ASSET.estimated_download_bytes == 608863580
    assert DIRECTIONAL_CLIP_ASSET.install_extra is None


def test_is_directional_clip_cached_all_present():
    """_is_directional_clip_cached returns True only when all files exist."""

    def fake_try_to_load(repo_id, filename, revision=None):
        return f"/mock/cache/{filename}"

    with patch(
        "huggingface_hub.try_to_load_from_cache",
        side_effect=fake_try_to_load,
    ), patch("os.path.exists", return_value=True):
        assert _is_directional_clip_cached() is True


def test_is_directional_clip_cached_partial_missing():
    """_is_directional_clip_cached fails if even one file is missing."""
    for missing in DIRECTIONAL_CLIP_REQUIRED_FILES:

        def fake_try_to_load(repo_id, filename, revision=None):
            if filename == missing:
                return None
            return f"/mock/cache/{filename}"

        with patch(
            "huggingface_hub.try_to_load_from_cache",
            side_effect=fake_try_to_load,
        ), patch("os.path.exists", return_value=True):
            assert (
                _is_directional_clip_cached() is False
            ), f"Failed to reject missing {missing}"


def test_complete_cache_default_disallow_download():
    """Complete cache allows init with local_files_only=True and revision."""
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_proc = MagicMock()
    mock_tok = MagicMock()

    with patch(
        "image_evaluator.directional_clip_predictor._is_directional_clip_cached",
        return_value=True,
    ), patch(
        "image_evaluator.directional_clip_predictor.AutoModel.from_pretrained",
        return_value=mock_model,
    ) as mock_m, patch(
        "image_evaluator.directional_clip_predictor.AutoProcessor.from_pretrained",
        return_value=mock_proc,
    ) as mock_p, patch(
        "image_evaluator.directional_clip_predictor.AutoTokenizer.from_pretrained",
        return_value=mock_tok,
    ) as mock_t:
        pred = DirectionalClipPredictor(device="cpu")
        assert pred.model is mock_model
        assert pred.processor is mock_proc
        assert pred.tokenizer is mock_tok

        expected_kwargs = {
            "revision": DIRECTIONAL_CLIP_ASSET.revision,
            "local_files_only": True,
        }
        mock_m.assert_called_once_with(
            "openai/clip-vit-base-patch32", **expected_kwargs
        )
        mock_p.assert_called_once_with(
            "openai/clip-vit-base-patch32", **expected_kwargs
        )
        mock_t.assert_called_once_with(
            "openai/clip-vit-base-patch32", **expected_kwargs
        )


def test_empty_cache_disallow_download_blocks():
    """Uncached asset without allow_download raises error with 0 calls."""
    with patch(
        "image_evaluator.directional_clip_predictor._is_directional_clip_cached",
        return_value=False,
    ), patch(
        "image_evaluator.directional_clip_predictor.AutoModel.from_pretrained",
    ) as mock_m, patch(
        "image_evaluator.directional_clip_predictor.AutoProcessor.from_pretrained",
    ) as mock_p, patch(
        "image_evaluator.directional_clip_predictor.AutoTokenizer.from_pretrained",
    ) as mock_t:
        with pytest.raises(DownloadNotAllowedError) as exc_info:
            DirectionalClipPredictor(device="cpu", allow_download=False)

        assert exc_info.value.asset == DIRECTIONAL_CLIP_ASSET
        mock_m.assert_not_called()
        mock_p.assert_not_called()
        mock_t.assert_not_called()


def test_partial_cache_disallow_download_blocks():
    """Partial cache without allow_download raises error with 0 calls."""
    with patch(
        "image_evaluator.directional_clip_predictor._is_directional_clip_cached",
        return_value=False,
    ), patch(
        "image_evaluator.directional_clip_predictor.AutoModel.from_pretrained",
    ) as mock_m, patch(
        "image_evaluator.directional_clip_predictor.AutoProcessor.from_pretrained",
    ) as mock_p, patch(
        "image_evaluator.directional_clip_predictor.AutoTokenizer.from_pretrained",
    ) as mock_t:
        with pytest.raises(DownloadNotAllowedError):
            DirectionalClipPredictor(device="cpu", allow_download=False)

        mock_m.assert_not_called()
        mock_p.assert_not_called()
        mock_t.assert_not_called()


def test_uncached_with_allow_download_and_callback():
    """Uncached asset with allow_download=True calls callback once."""
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_proc = MagicMock()
    mock_tok = MagicMock()
    callback_calls = []

    def mock_cb(asset, msg):
        callback_calls.append((asset, msg))

    with patch(
        "image_evaluator.directional_clip_predictor._is_directional_clip_cached",
        return_value=False,
    ), patch(
        "image_evaluator.directional_clip_predictor.AutoModel.from_pretrained",
        return_value=mock_model,
    ) as mock_m, patch(
        "image_evaluator.directional_clip_predictor.AutoProcessor.from_pretrained",
        return_value=mock_proc,
    ) as mock_p, patch(
        "image_evaluator.directional_clip_predictor.AutoTokenizer.from_pretrained",
        return_value=mock_tok,
    ) as mock_t:
        _ = DirectionalClipPredictor(
            device="cpu",
            allow_download=True,
            download_callback=mock_cb,
        )
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == DIRECTIONAL_CLIP_ASSET
        assert "openai/clip-vit-base-patch32" in callback_calls[0][1]

        expected_kwargs = {
            "revision": DIRECTIONAL_CLIP_ASSET.revision,
            "local_files_only": False,
        }
        mock_m.assert_called_once_with(
            "openai/clip-vit-base-patch32", **expected_kwargs
        )
        mock_p.assert_called_once_with(
            "openai/clip-vit-base-patch32", **expected_kwargs
        )
        mock_t.assert_called_once_with(
            "openai/clip-vit-base-patch32", **expected_kwargs
        )


def test_initialization_failure_cleans_up_state():
    """Failed initialization resets instance model/processor/tokenizer."""
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_proc = MagicMock()

    created_instances = []
    orig_init = DirectionalClipPredictor.__init__

    def wrapped_init(self, *args, **kwargs):
        created_instances.append(self)
        orig_init(self, *args, **kwargs)

    with patch(
        "image_evaluator.directional_clip_predictor._is_directional_clip_cached",
        return_value=True,
    ), patch(
        "image_evaluator.directional_clip_predictor.AutoModel.from_pretrained",
        return_value=mock_model,
    ), patch(
        "image_evaluator.directional_clip_predictor.AutoProcessor.from_pretrained",
        return_value=mock_proc,
    ), patch(
        "image_evaluator.directional_clip_predictor.AutoTokenizer.from_pretrained",
        side_effect=RuntimeError("Corrupt tokenizer file"),
    ), patch.object(
        DirectionalClipPredictor, "__init__", wrapped_init
    ):
        with pytest.raises(RuntimeError, match="Corrupt tokenizer file"):
            DirectionalClipPredictor(device="cpu")

    assert len(created_instances) == 1
    failed_instance = created_instances[0]
    assert failed_instance.model is None
    assert failed_instance.processor is None
    assert failed_instance.tokenizer is None


def test_custom_clip_model_rejected_zero_probes_zero_calls():
    """Custom clip_model must be rejected with 0 cache probes and 0 calls."""
    custom_model = "custom/my-clip-model"
    cb = MagicMock()

    with patch(
        "image_evaluator.directional_clip_predictor._is_directional_clip_cached"
    ) as mock_probe, patch(
        "image_evaluator.directional_clip_predictor.check_asset_and_permit_download"
    ) as mock_gate, patch(
        "image_evaluator.directional_clip_predictor.AutoModel.from_pretrained"
    ) as mock_m, patch(
        "image_evaluator.directional_clip_predictor.AutoProcessor.from_pretrained"
    ) as mock_p, patch(
        "image_evaluator.directional_clip_predictor.AutoTokenizer.from_pretrained"
    ) as mock_t:
        with pytest.raises(ValueError, match="Unsupported clip_model"):
            DirectionalClipPredictor(
                clip_model=custom_model,
                device="cpu",
                allow_download=True,
                download_callback=cb,
            )

        mock_probe.assert_not_called()
        mock_gate.assert_not_called()
        mock_m.assert_not_called()
        mock_p.assert_not_called()
        mock_t.assert_not_called()
        cb.assert_not_called()


def test_core_evaluate_passes_download_gate_parameters():
    """core.evaluate forwards allow_download and download_callback."""
    src_img = Image.new("RGB", (32, 32), color="red")
    edit_img = Image.new("RGB", (32, 32), color="blue")
    cb = MagicMock()

    mock_predictor_instance = MagicMock()
    mock_predictor_instance.evaluate_directional_clip.return_value = 0.85

    with patch(
        "image_evaluator.directional_clip_predictor.DirectionalClipPredictor",
        return_value=mock_predictor_instance,
    ) as mock_pred_cls:
        res = evaluate(
            image=edit_img,
            reference=src_img,
            prompt="a blue square",
            source_prompt="a red square",
            metrics=["directional_clip"],
            allow_download=True,
            download_callback=cb,
        )

        assert res["directional_clip"] == 0.85
        mock_pred_cls.assert_called_once_with(
            clip_model="openai/clip-vit-base-patch32",
            device=None,
            allow_download=True,
            download_callback=cb,
        )


def test_core_evaluate_rejects_custom_clip_model():
    """core.evaluate rejects custom clip_model for directional_clip."""
    src_img = Image.new("RGB", (32, 32), color="red")
    edit_img = Image.new("RGB", (32, 32), color="blue")

    with pytest.raises(ValueError, match="Unsupported clip_model"):
        evaluate(
            image=edit_img,
            reference=src_img,
            prompt="a blue square",
            source_prompt="a red square",
            metrics=["directional_clip"],
            clip_model="unsupported/clip-model",
        )


def test_direct_construction_writes_nothing_to_stdout(capsys):
    """DWI-03: constructing the predictor as an SDK object is stdout-silent."""
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model

    with patch(
        "image_evaluator.directional_clip_predictor._is_directional_clip_cached",
        return_value=True,
    ), patch(
        "image_evaluator.directional_clip_predictor.AutoModel.from_pretrained",
        return_value=mock_model,
    ), patch(
        "image_evaluator.directional_clip_predictor.AutoProcessor.from_pretrained",
        return_value=MagicMock(),
    ), patch(
        "image_evaluator.directional_clip_predictor.AutoTokenizer.from_pretrained",
        return_value=MagicMock(),
    ):
        DirectionalClipPredictor(device="cpu")

    assert capsys.readouterr().out == ""
