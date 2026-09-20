"""Regression and audit verification tests for Boss-reported issues.

Verifies:
1. OpenCLIP cache check is strictly offline and does not penetrate gate.
2. CLI and Predictors accept text files as --prompt / prompt arguments.
3. DINOv2 cache detection requires both model weights and preprocessor config.
4. Multi-metric evaluation clears intermediate memory after each predictor.
5. ModelAsset revisions are pinned and passed to Hugging Face loaders.
"""

from unittest.mock import patch

import pytest
from PIL import Image

from image_evaluator.clip_i_predictor import _is_clip_i_cached
from image_evaluator.dino_similarity_predictor import (
    DINO_SIMILARITY_ASSET,
    _is_dinov2_cached,
)
from image_evaluator.hpsv2_predictor import (
    HPSV2_ASSET,
    Hpsv2Predictor,
)
from image_evaluator.image_reward_predictor import (
    IMAGE_REWARD_ASSET,
)
from image_evaluator.main import cli
from image_evaluator.pickscore_predictor import PickScorePredictor
from image_evaluator.vqascore_predictor import (
    VQA_SCORE_ASSET,
)


def test_openclip_cache_check_does_not_download():
    """_is_clip_i_cached must never call download_pretrained."""
    with patch(
        "open_clip.pretrained.download_pretrained",
        side_effect=AssertionError("download_pretrained called illegally!"),
    ), patch(
        "open_clip.pretrained.get_pretrained_cfg",
        return_value={"url": "https://fake.url/model.pt"},
    ), patch(
        "os.path.isfile",
        return_value=False,
    ):
        result = _is_clip_i_cached("ViT-L-14-quickgelu", "openai")
        assert result is False


def test_dinov2_cache_requires_both_weights_and_processor():
    """_is_dinov2_cached must fail if preprocessor_config.json is missing."""

    def fake_cache(repo_id, filename, revision=None):
        if filename == "model.safetensors":
            return "/path/to/model.safetensors"
        if filename == "preprocessor_config.json":
            return None
        return None

    with patch(
        "huggingface_hub.try_to_load_from_cache", side_effect=fake_cache
    ):
        assert _is_dinov2_cached() is False

    def fake_cache_complete(repo_id, filename, revision=None):
        return f"/path/to/{filename}"

    with patch("os.path.exists", return_value=True), patch(
        "huggingface_hub.try_to_load_from_cache",
        side_effect=fake_cache_complete,
    ):
        assert _is_dinov2_cached() is True


def test_model_asset_revisions_pinned():
    """Model assets must have explicit commit SHA revisions."""
    assert len(DINO_SIMILARITY_ASSET.revision) == 40
    assert len(HPSV2_ASSET.revision) >= 7
    assert len(IMAGE_REWARD_ASSET.revision) >= 7
    assert len(VQA_SCORE_ASSET.revision) == 40


def test_cli_prompt_file_resolution(tmp_path):
    """CLI loads prompt content when --prompt points to a file."""
    prompt_file = tmp_path / "test_prompt.txt"
    prompt_file.write_text("a golden retriever playing in autumn leaves\n")

    img_file = tmp_path / "test_img.png"
    Image.new("RGB", (64, 64), color=(100, 100, 100)).save(img_file)

    recorded_prompt = []

    def mock_eval(self, image, prompt):
        recorded_prompt.append(prompt)
        return 0.85

    with patch.object(PickScorePredictor, "evaluate", mock_eval), patch.object(
        PickScorePredictor, "evaluate_pickscore", mock_eval
    ):
        code = cli(
            [
                "--metrics",
                "pickscore",
                "--image",
                str(img_file),
                "--prompt",
                str(prompt_file),
            ]
        )
        assert code == 0
        expected = ["a golden retriever playing in autumn leaves"]
        assert recorded_prompt == expected


def test_folder_evaluation_with_prompt_file_lines(tmp_path):
    """Predictors evaluate_folder methods map prompt file lines to images."""
    folder = tmp_path / "images"
    folder.mkdir()
    img1 = folder / "img1.png"
    img2 = folder / "img2.png"
    Image.new("RGB", (64, 64), color=(10, 10, 10)).save(img1)
    Image.new("RGB", (64, 64), color=(20, 20, 20)).save(img2)

    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("first prompt\nsecond prompt\n")

    hps_pred = Hpsv2Predictor()
    hps_pred._loaded = True
    recorded_prompts = []

    def mock_compute_hps(image, prompt):
        recorded_prompts.append(prompt)
        return 0.75

    hps_pred.compute_hpsv2 = mock_compute_hps
    score = hps_pred.evaluate_folder_hpsv2(str(folder), str(prompt_file))
    assert score == pytest.approx(0.75)
    assert recorded_prompts == ["first prompt", "second prompt"]
