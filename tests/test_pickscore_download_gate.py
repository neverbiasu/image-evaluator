"""Tests for PickScore model assets, cache verification, and download gating.

Verifies:
1. Complete cache + allow_download=False: loads with local_files_only=True.
2. Empty cache + allow_download=False: raises DownloadNotAllowedError.
3. Partial cache (model missing or processor missing) raises error.
4. Empty cache + allow_download=True: disclosure callback called without dupes.
5. 40-character revision correctly passed to AutoModel and AutoProcessor.
6. Custom remote repo ID without 40-hex revision is rejected with ValueError.
7. Local path for custom model/processor accepted without remote checks.
8. Component failure during init clears state and allows retry.
9. evaluate(..., metrics="pickscore") propagates allow_download.
10. CLI format=json produces clean JSON without stdout pollution.
"""

import json
import math
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from image_evaluator.core import evaluate
from image_evaluator.main import main as cli_main
from image_evaluator.model_assets import DownloadNotAllowedError
from image_evaluator.pickscore_predictor import (
    PICKSCORE_MODEL_ASSET,
    PICKSCORE_MODEL_REQUIRED_FILES,
    PICKSCORE_PROCESSOR_ASSET,
    PickScorePredictor,
    _is_hf_asset_cached,
    _is_pickscore_cached,
)


@pytest.fixture
def dummy_image(tmp_path):
    img_path = tmp_path / "test.png"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(img_path)
    return str(img_path)


def test_pickscore_assets_metadata():
    """Verify PICKSCORE_MODEL_ASSET and PICKSCORE_PROCESSOR_ASSET metadata."""
    assert PICKSCORE_MODEL_ASSET.metric_id == "pickscore"
    assert PICKSCORE_MODEL_ASSET.model_id == "yuvalkirstain/PickScore_v1"
    assert PICKSCORE_MODEL_ASSET.source == "huggingface"
    assert (
        PICKSCORE_MODEL_ASSET.revision
        == "a4e4367c6dfa7288a00c550414478f865b875800"
    )
    assert len(PICKSCORE_MODEL_ASSET.revision) == 40
    assert PICKSCORE_MODEL_ASSET.estimated_download_bytes == 3944552236

    assert PICKSCORE_PROCESSOR_ASSET.metric_id == "pickscore"
    assert (
        PICKSCORE_PROCESSOR_ASSET.model_id
        == "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    )
    assert PICKSCORE_PROCESSOR_ASSET.source == "huggingface"
    assert (
        PICKSCORE_PROCESSOR_ASSET.revision
        == "1c2b8495b28150b8a4922ee1c8edee224c284c0c"
    )
    assert len(PICKSCORE_PROCESSOR_ASSET.revision) == 40
    assert PICKSCORE_PROCESSOR_ASSET.estimated_download_bytes == 3616854


def test_is_hf_asset_cached_complete_and_partial():
    """_is_hf_asset_cached returns True only when all files exist."""
    captured_revisions = []

    def fake_try_load(repo_id, filename, revision=None):
        captured_revisions.append(revision)
        return f"/mock/cache/{filename}"

    with (
        patch(
            "huggingface_hub.try_to_load_from_cache", side_effect=fake_try_load
        ),
        patch("os.path.exists", return_value=True),
    ):
        assert (
            _is_hf_asset_cached(
                PICKSCORE_MODEL_ASSET.model_id,
                PICKSCORE_MODEL_REQUIRED_FILES,
                revision=PICKSCORE_MODEL_ASSET.revision,
            )
            is True
        )
        assert all(
            r == PICKSCORE_MODEL_ASSET.revision for r in captured_revisions
        )

    # Missing model.safetensors
    def fake_missing_load(repo_id, filename, revision=None):
        if filename == "model.safetensors":
            return None
        return f"/mock/cache/{filename}"

    with (
        patch(
            "huggingface_hub.try_to_load_from_cache",
            side_effect=fake_missing_load,
        ),
        patch("os.path.exists", return_value=True),
    ):
        assert (
            _is_hf_asset_cached(
                PICKSCORE_MODEL_ASSET.model_id,
                PICKSCORE_MODEL_REQUIRED_FILES,
                revision=PICKSCORE_MODEL_ASSET.revision,
            )
            is False
        )


def test_is_pickscore_cached():
    """_is_pickscore_cached checks both model and processor with revision."""
    with patch(
        "image_evaluator.pickscore_predictor._is_hf_asset_cached",
        side_effect=[True, True],
    ) as mock_cached:
        assert _is_pickscore_cached() is True
        assert mock_cached.call_count == 2
        assert (
            mock_cached.call_args_list[0][1]["revision"]
            == PICKSCORE_MODEL_ASSET.revision
        )
        assert (
            mock_cached.call_args_list[1][1]["revision"]
            == PICKSCORE_PROCESSOR_ASSET.revision
        )

    with patch(
        "image_evaluator.pickscore_predictor._is_hf_asset_cached",
        side_effect=[True, False],
    ):
        assert _is_pickscore_cached() is False

    with patch(
        "image_evaluator.pickscore_predictor._is_hf_asset_cached",
        side_effect=[False, True],
    ):
        assert _is_pickscore_cached() is False


def test_complete_cache_default_disallow_download(dummy_image):
    """Complete cache loads model and processor with local_files_only=True."""
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model
    mock_proc = MagicMock()

    with (
        patch(
            "image_evaluator.pickscore_predictor._is_hf_asset_cached",
            return_value=True,
        ),
        patch(
            "transformers.AutoProcessor.from_pretrained",
            return_value=mock_proc,
        ) as mock_p,
        patch(
            "transformers.AutoModel.from_pretrained", return_value=mock_model
        ) as mock_m,
    ):
        pred = PickScorePredictor(device="cpu", allow_download=False)
        m, p = pred._load_model()
        assert m is mock_model
        assert p is mock_proc

        mock_p.assert_called_once_with(
            PICKSCORE_PROCESSOR_ASSET.model_id,
            revision=PICKSCORE_PROCESSOR_ASSET.revision,
            local_files_only=True,
        )
        mock_m.assert_called_once_with(
            PICKSCORE_MODEL_ASSET.model_id,
            revision=PICKSCORE_MODEL_ASSET.revision,
            local_files_only=True,
        )


def test_empty_cache_disallow_download_raises_download_not_allowed():
    """Uncached asset with allow_download=False raises error with 0 loads."""
    with (
        patch(
            "image_evaluator.pickscore_predictor._is_hf_asset_cached",
            return_value=False,
        ),
        patch("transformers.AutoProcessor.from_pretrained") as mock_p,
        patch("transformers.AutoModel.from_pretrained") as mock_m,
    ):
        pred = PickScorePredictor(device="cpu", allow_download=False)
        with pytest.raises(DownloadNotAllowedError) as exc_info:
            pred._load_model()

        assert "not permitted" in str(exc_info.value)
        mock_p.assert_not_called()
        mock_m.assert_not_called()


def test_partial_cache_disallow_download_raises_for_model():
    """If processor cached but model missing, raises for model."""
    # processor cached (True), model missing (False)
    with (
        patch(
            "image_evaluator.pickscore_predictor._is_hf_asset_cached",
            side_effect=[True, False],
        ),
        patch("transformers.AutoProcessor.from_pretrained") as mock_p,
        patch("transformers.AutoModel.from_pretrained") as mock_m,
    ):
        pred = PickScorePredictor(device="cpu", allow_download=False)
        with pytest.raises(DownloadNotAllowedError) as exc_info:
            pred._load_model()

        assert PICKSCORE_MODEL_ASSET.model_id in str(exc_info.value)
        mock_p.assert_not_called()
        mock_m.assert_not_called()


def test_empty_cache_allow_download_invokes_disclosure():
    """Authorized download invokes disclosure callback exactly once."""
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_proc = MagicMock()
    disclosed_records = []

    def callback(asset, msg):
        disclosed_records.append((asset.model_id, msg))

    with (
        patch(
            "image_evaluator.pickscore_predictor._is_hf_asset_cached",
            return_value=False,
        ),
        patch(
            "transformers.AutoProcessor.from_pretrained",
            return_value=mock_proc,
        ) as mock_p,
        patch(
            "transformers.AutoModel.from_pretrained", return_value=mock_model
        ) as mock_m,
    ):
        pred = PickScorePredictor(
            device="cpu", allow_download=True, download_callback=callback
        )
        pred._load_model()

        assert len(disclosed_records) == 2
        assert disclosed_records[0][0] == PICKSCORE_PROCESSOR_ASSET.model_id
        assert disclosed_records[1][0] == PICKSCORE_MODEL_ASSET.model_id

        mock_p.assert_called_once_with(
            PICKSCORE_PROCESSOR_ASSET.model_id,
            revision=PICKSCORE_PROCESSOR_ASSET.revision,
            local_files_only=False,
        )
        mock_m.assert_called_once_with(
            PICKSCORE_MODEL_ASSET.model_id,
            revision=PICKSCORE_MODEL_ASSET.revision,
            local_files_only=False,
        )


def test_custom_remote_model_missing_or_invalid_metadata_rejected():
    """Custom remote model requires 40-hex revision, bytes, and files."""
    valid_rev = "a" * 40
    # Missing revision
    with pytest.raises(
        ValueError, match="verified 40-character hexadecimal commit hash"
    ):
        PickScorePredictor(
            model_name="custom/my-model",
            model_download_bytes=1000,
            model_required_files=("config.json",),
        )

    # Missing download bytes
    with pytest.raises(ValueError, match="positive integer download size"):
        PickScorePredictor(
            model_name="custom/my-model",
            model_revision=valid_rev,
            model_download_bytes=None,
            model_required_files=("config.json",),
        )

    # Missing required files
    with pytest.raises(
        ValueError, match="non-empty sequence of required filenames"
    ):
        PickScorePredictor(
            model_name="custom/my-model",
            model_revision=valid_rev,
            model_download_bytes=1000,
            model_required_files=None,
        )


def test_custom_remote_processor_missing_or_invalid_metadata_rejected():
    """Custom remote processor requires 40-hex revision, bytes, and files."""
    valid_rev = "b" * 40
    # Missing revision
    with pytest.raises(
        ValueError, match="verified 40-character hexadecimal commit hash"
    ):
        PickScorePredictor(
            processor_name="custom/my-proc",
            processor_download_bytes=500,
            processor_required_files=("preprocessor_config.json",),
        )

    # Missing download bytes
    with pytest.raises(ValueError, match="positive integer download size"):
        PickScorePredictor(
            processor_name="custom/my-proc",
            processor_revision=valid_rev,
            processor_download_bytes=None,
            processor_required_files=("preprocessor_config.json",),
        )

    # Missing required files
    with pytest.raises(
        ValueError, match="non-empty sequence of required filenames"
    ):
        PickScorePredictor(
            processor_name="custom/my-proc",
            processor_revision=valid_rev,
            processor_download_bytes=500,
            processor_required_files=None,
        )


def test_custom_remote_invalid_types_and_values_rejected():
    """Reject non-positive bytes, boolean bytes, and empty file lists."""
    valid_rev = "c" * 40
    # Zero bytes
    with pytest.raises(ValueError, match="positive integer download size"):
        PickScorePredictor(
            model_name="custom/m",
            model_revision=valid_rev,
            model_download_bytes=0,
            model_required_files=("f.bin",),
        )

    # Negative bytes
    with pytest.raises(ValueError, match="positive integer download size"):
        PickScorePredictor(
            model_name="custom/m",
            model_revision=valid_rev,
            model_download_bytes=-100,
            model_required_files=("f.bin",),
        )

    # Boolean bytes
    with pytest.raises(ValueError, match="positive integer download size"):
        PickScorePredictor(
            model_name="custom/m",
            model_revision=valid_rev,
            model_download_bytes=True,
            model_required_files=("f.bin",),
        )

    # Empty tuple
    with pytest.raises(
        ValueError, match="non-empty sequence of required filenames"
    ):
        PickScorePredictor(
            model_name="custom/m",
            model_revision=valid_rev,
            model_download_bytes=1000,
            model_required_files=(),
        )

    # Tuple with empty string filename
    with pytest.raises(ValueError, match="invalid required filename"):
        PickScorePredictor(
            model_name="custom/m",
            model_revision=valid_rev,
            model_download_bytes=1000,
            model_required_files=("config.json", ""),
        )

    # Tuple with whitespace filename
    with pytest.raises(ValueError, match="invalid required filename"):
        PickScorePredictor(
            model_name="custom/m",
            model_revision=valid_rev,
            model_download_bytes=1000,
            model_required_files=("   ",),
        )


def test_revision_strict_syntax_and_newline_rejection():
    """Revision validation must strictly match 40-char hex without newlines."""
    # 40-char hex with trailing newline
    with pytest.raises(
        ValueError, match="verified 40-character hexadecimal commit hash"
    ):
        PickScorePredictor(
            model_name="custom/m",
            model_revision=("a" * 40) + "\n",
            model_download_bytes=100,
            model_required_files=("f",),
        )

    # 40-char hex with leading whitespace
    with pytest.raises(
        ValueError, match="verified 40-character hexadecimal commit hash"
    ):
        PickScorePredictor(
            model_name="custom/m",
            model_revision=" " + ("a" * 40),
            model_download_bytes=100,
            model_required_files=("f",),
        )

    # Branch names and non-hex
    for invalid in [
        "main",
        "latest",
        "v1.0",
        "g" * 40,
        "z" * 40,
        "a" * 39,
        "a" * 41,
    ]:
        with pytest.raises(
            ValueError, match="verified 40-character hexadecimal commit hash"
        ):
            PickScorePredictor(
                model_name="custom/m",
                model_revision=invalid,
                model_download_bytes=100,
                model_required_files=("f",),
            )


def test_rejection_happens_before_cache_and_from_pretrained():
    """Validation failure occurs before cache query or from_pretrained."""
    with (
        patch(
            "image_evaluator.pickscore_predictor._is_hf_asset_cached"
        ) as mock_cache,
        patch("transformers.AutoModel.from_pretrained") as mock_m,
        patch("transformers.AutoProcessor.from_pretrained") as mock_p,
    ):
        with pytest.raises(ValueError):
            PickScorePredictor(
                model_name="custom/m",
                model_revision="invalid_rev",
                model_download_bytes=100,
                model_required_files=("f",),
            )

        mock_cache.assert_not_called()
        mock_m.assert_not_called()
        mock_p.assert_not_called()


def test_custom_local_paths_accepted_without_remote_metadata(tmp_path):
    """Local path accepted for model/processor without remote metadata."""
    local_model_dir = tmp_path / "local_model"
    local_model_dir.mkdir()
    local_proc_dir = tmp_path / "local_proc"
    local_proc_dir.mkdir()

    pred = PickScorePredictor(
        model_name=str(local_model_dir),
        processor_name=str(local_proc_dir),
    )
    assert pred.is_local_model is True
    assert pred.model_name == str(local_model_dir)
    assert pred.model_revision is None
    assert pred.model_download_bytes is None
    assert pred.model_required_files == ()

    assert pred.is_local_processor is True
    assert pred.processor_name == str(local_proc_dir)
    assert pred.processor_revision is None
    assert pred.processor_download_bytes is None
    assert pred.processor_required_files == ()


def test_custom_remote_id_with_valid_40_char_hex_and_metadata():
    """Complete metadata enables cache check, disclosure, and pinned rev."""
    valid_model_rev = "0123456789abcdef0123456789abcdef01234567"
    valid_proc_rev = "abcdef0123456789abcdef0123456789abcdef01"

    disclosed = []

    def cb(asset, msg):
        disclosed.append((asset, msg))

    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_proc = MagicMock()

    pred = PickScorePredictor(
        model_name="custom/my-model",
        model_revision=valid_model_rev,
        processor_name="custom/my-proc",
        processor_revision=valid_proc_rev,
        model_download_bytes=1000000,
        processor_download_bytes=50000,
        model_required_files=("model.bin",),
        processor_required_files=("vocab.json",),
        allow_download=True,
        download_callback=cb,
    )
    assert pred.is_local_model is False
    assert pred.model_revision == valid_model_rev
    assert pred.is_local_processor is False
    assert pred.processor_revision == valid_proc_rev

    with (
        patch(
            "image_evaluator.pickscore_predictor._is_hf_asset_cached",
            return_value=False,
        ),
        patch(
            "transformers.AutoProcessor.from_pretrained",
            return_value=mock_proc,
        ) as mock_p,
        patch(
            "transformers.AutoModel.from_pretrained", return_value=mock_model
        ) as mock_m,
    ):
        pred._load_model()

    assert len(disclosed) == 2
    proc_asset, proc_msg = disclosed[0]
    model_asset, model_msg = disclosed[1]

    assert proc_asset.model_id == "custom/my-proc"
    assert proc_asset.revision == valid_proc_rev
    assert proc_asset.estimated_download_bytes == 50000
    assert "custom/my-proc" in proc_msg

    assert model_asset.model_id == "custom/my-model"
    assert model_asset.revision == valid_model_rev
    assert model_asset.estimated_download_bytes == 1000000
    assert "custom/my-model" in model_msg

    mock_p.assert_called_once_with(
        "custom/my-proc",
        revision=valid_proc_rev,
        local_files_only=False,
    )
    mock_m.assert_called_once_with(
        "custom/my-model",
        revision=valid_model_rev,
        local_files_only=False,
    )


def test_failure_cleanup_allows_successful_retry():
    """If initial load fails (e.g. OOM), state is cleared and retried."""
    mock_proc = MagicMock()
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model

    pred = PickScorePredictor(device="cpu", allow_download=False)

    # First attempt: AutoModel throws RuntimeError
    with (
        patch(
            "image_evaluator.pickscore_predictor._is_hf_asset_cached",
            return_value=True,
        ),
        patch(
            "transformers.AutoProcessor.from_pretrained",
            return_value=mock_proc,
        ),
        patch(
            "transformers.AutoModel.from_pretrained",
            side_effect=RuntimeError("GPU OOM"),
        ),
    ):
        with pytest.raises(RuntimeError, match="GPU OOM"):
            pred._load_model()

        assert pred._processor is None
        assert pred._model is None

    # Second attempt (retry): succeeds cleanly
    with (
        patch(
            "image_evaluator.pickscore_predictor._is_hf_asset_cached",
            return_value=True,
        ),
        patch(
            "transformers.AutoProcessor.from_pretrained",
            return_value=mock_proc,
        ),
        patch(
            "transformers.AutoModel.from_pretrained", return_value=mock_model
        ),
    ):
        m, p = pred._load_model()
        assert m is mock_model
        assert p is mock_proc
        assert pred._model is mock_model
        assert pred._processor is mock_proc


def test_core_evaluate_propagates_allow_download(dummy_image):
    """core.evaluate() passes download params to PickScorePredictor."""
    callback = MagicMock()

    with patch("image_evaluator.core._run_with_predictor") as mock_runner:
        evaluate(
            metrics="pickscore",
            image=dummy_image,
            prompt="test prompt",
            allow_download=True,
            download_callback=callback,
            pickscore_model="custom/m",
            pickscore_model_revision="a" * 40,
            pickscore_model_download_bytes=12345,
            pickscore_model_required_files=("model.safetensors",),
        )
        assert mock_runner.call_count == 1
        factory = mock_runner.call_args[0][0]
        # Instantiate factory and verify predictor attributes
        pred = factory()
        assert pred.allow_download is True
        assert pred.download_callback is callback
        assert pred.model_name == "custom/m"
        assert pred.model_revision == "a" * 40
        assert pred.model_download_bytes == 12345
        assert pred.model_required_files == ("model.safetensors",)


def test_cli_pickscore_clean_json_output(dummy_image, capsys):
    """CLI pickscore execution under format=json outputs valid JSON."""
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model
    mock_model.get_image_features.return_value = torch.tensor([[1.0, 0.0]])
    mock_model.get_text_features.return_value = torch.tensor([[1.0, 0.0]])
    mock_model.logit_scale = torch.tensor(3.0)
    mock_proc = MagicMock()
    mock_proc.return_value = {"input_ids": torch.tensor([[1, 2]])}

    with (
        patch(
            "image_evaluator.pickscore_predictor._is_hf_asset_cached",
            return_value=True,
        ),
        patch(
            "transformers.AutoProcessor.from_pretrained",
            return_value=mock_proc,
        ),
        patch(
            "transformers.AutoModel.from_pretrained", return_value=mock_model
        ),
    ):
        cli_main(
            [
                "--metrics",
                "pickscore",
                "--image",
                dummy_image,
                "--prompt",
                "test",
                "--format",
                "json",
            ]
        )

    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert "metrics" in parsed
    assert "pickscore" in parsed["metrics"]
    assert math.isclose(
        parsed["metrics"]["pickscore"],
        float(torch.exp(torch.tensor(3.0)).item()),
        rel_tol=1e-4,
    )


def test_cli_pickscore_download_disclosure_on_stderr(dummy_image, capsys):
    """CLI emits download disclosure to stderr without corrupting JSON."""
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model
    mock_model.get_image_features.return_value = torch.tensor([[1.0, 0.0]])
    mock_model.get_text_features.return_value = torch.tensor([[1.0, 0.0]])
    mock_model.logit_scale = torch.tensor(3.0)
    mock_proc = MagicMock()
    mock_proc.return_value = {"input_ids": torch.tensor([[1, 2]])}

    with (
        patch(
            "image_evaluator.pickscore_predictor._is_hf_asset_cached",
            return_value=False,
        ),
        patch(
            "transformers.AutoProcessor.from_pretrained",
            return_value=mock_proc,
        ),
        patch(
            "transformers.AutoModel.from_pretrained", return_value=mock_model
        ),
    ):
        cli_main(
            [
                "--metrics",
                "pickscore",
                "--image",
                dummy_image,
                "--prompt",
                "test",
                "--format",
                "json",
                "--allow-download",
            ]
        )

    captured = capsys.readouterr()
    # Stdout must parse as pristine JSON
    parsed = json.loads(captured.out)
    assert "pickscore" in parsed["metrics"]

    # Stderr must contain the download disclosures for both model and processor
    assert (
        "[image-evaluator] Downloading model "
        "'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'" in captured.err
    )
    assert (
        "[image-evaluator] Downloading model 'yuvalkirstain/PickScore_v1'"
        in captured.err
    )
    assert "a4e4367c6dfa7288a00c550414478f865b875800" in captured.err
    assert "1c2b8495b28150b8a4922ee1c8edee224c284c0c" in captured.err
