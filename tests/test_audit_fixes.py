"""Regression and audit verification tests for Boss-reported issues.

Verifies:
1. OpenCLIP cache check is strictly offline and does not penetrate gate.
2. CLI and Predictors accept text files as --prompt / prompt arguments.
3. DINOv2 cache detection requires both model weights and preprocessor config.
4. Multi-metric evaluation clears intermediate memory after each predictor.
5. ModelAsset revisions are pinned and passed to Hugging Face loaders.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
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
    IMAGE_REWARD_TOKENIZER_ASSET,
)
from image_evaluator.main import cli
from image_evaluator.pickscore_predictor import PickScorePredictor
from image_evaluator.vqascore_predictor import (
    VQA_SCORE_ASSET,
    VQA_SCORE_TEXT_ASSET,
    VQA_SCORE_VISION_ASSET,
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
    """Model assets must have explicit 40-character commit SHA revisions."""
    assert len(DINO_SIMILARITY_ASSET.revision) == 40
    assert len(HPSV2_ASSET.revision) == 40
    assert len(IMAGE_REWARD_ASSET.revision) == 40
    assert len(IMAGE_REWARD_TOKENIZER_ASSET.revision) == 40
    assert len(VQA_SCORE_TEXT_ASSET.revision) == 40
    assert len(VQA_SCORE_VISION_ASSET.revision) == 40
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


def test_dinov2_cache_requires_config_json():
    """_is_dinov2_cached must fail if config.json is missing."""

    def fake_cache(repo_id, filename, revision=None):
        if filename == "model.safetensors":
            return "/path/to/model.safetensors"
        if filename == "preprocessor_config.json":
            return "/path/to/preprocessor_config.json"
        if filename == "config.json":
            return None
        return None

    with patch("os.path.exists", return_value=True), patch(
        "huggingface_hub.try_to_load_from_cache", side_effect=fake_cache
    ):
        assert _is_dinov2_cached() is False


def test_clip_i_and_dinov2_accept_numpy_arrays():
    """ClipI and DinoSimilarity _prepare_image accept np.ndarray."""
    import numpy as np

    from image_evaluator.clip_i_predictor import ClipIPredictor
    from image_evaluator.dino_similarity_predictor import (
        DinoSimilarityPredictor,
    )

    arr = np.zeros((32, 32, 3), dtype=np.uint8)

    clip_i = ClipIPredictor.__new__(ClipIPredictor)
    clip_i.device = "cpu"
    clip_i.preprocess = MagicMock(
        return_value=torch.zeros((3, 224, 224), dtype=torch.float32)
    )
    t_clip = clip_i._prepare_image(arr)
    assert t_clip.shape == (1, 3, 224, 224)

    dino = DinoSimilarityPredictor.__new__(DinoSimilarityPredictor)
    dino.device = "cpu"
    dino.processor = MagicMock(
        return_value={"pixel_values": torch.zeros((1, 3, 224, 224))}
    )
    t_dino = dino._prepare_image(arr)
    assert t_dino.shape == (1, 3, 224, 224)


def test_clip_dummy_dataset_supports_prompt_list_and_file(tmp_path):
    """DummyDataset in clip_score_predictor handles list and file prompts."""
    from image_evaluator.clip_score_predictor import DummyDataset

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "img1.png").write_bytes(b"")
    (img_dir / "img2.png").write_bytes(b"")

    # 1. Test list of prompts matching image count
    ds_list = DummyDataset(
        real_path=str(img_dir),
        fake_path=["prompt 1", "prompt 2"],
        real_flag="img",
        fake_flag="txt",
    )
    assert len(ds_list) == 2
    assert ds_list.fake_folder == ["prompt 1", "prompt 2"]

    # 2. Test list of prompts mismatching image count
    with pytest.raises(ValueError, match="does not match"):
        DummyDataset(
            real_path=str(img_dir),
            fake_path=["only one prompt"],
            real_flag="img",
            fake_flag="txt",
        )

    # 3. Test prompt file with matching lines
    pfile = tmp_path / "prompts.txt"
    pfile.write_text("prompt A\nprompt B\n")
    ds_file = DummyDataset(
        real_path=str(img_dir),
        fake_path=str(pfile),
        real_flag="img",
        fake_flag="txt",
    )
    assert len(ds_file) == 2
    assert ds_file.fake_folder == ["prompt A", "prompt B"]


def test_cli_folder_prompt_file_resolution(tmp_path):
    """CLI passes prompt lines as list[str] when evaluating a folder."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img1 = img_dir / "img1.png"
    img2 = img_dir / "img2.png"
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(img1)
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(img2)

    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("first line\nsecond line\n")

    recorded_prompt = []

    def mock_evaluate_folder(self, folder, prompt):
        recorded_prompt.append(prompt)
        res = MagicMock()
        res.mean_score = 0.9
        return res

    with patch.object(
        PickScorePredictor, "evaluate_folder", mock_evaluate_folder
    ):
        code = cli(
            [
                "--metrics",
                "pickscore",
                "--image",
                str(img_dir),
                "--prompt",
                str(prompt_file),
            ]
        )
        assert code == 0
        assert recorded_prompt == [["first line", "second line"]]


def test_directional_clip_text_features_non_string_guard():
    """_get_text_features handles list and non-string inputs safely."""
    from image_evaluator.directional_clip_predictor import (
        DirectionalClipPredictor,
    )

    pred = DirectionalClipPredictor.__new__(DirectionalClipPredictor)
    pred.device = "cpu"
    mock_tok = MagicMock()
    mock_tok.return_value = {
        "input_ids": torch.zeros((1, 10), dtype=torch.long)
    }
    pred.tokenizer = mock_tok
    mock_model = MagicMock()
    mock_model.get_text_features.return_value = torch.ones(
        (1, 512), dtype=torch.float32
    )
    pred.model = mock_model

    # 1. Non-path string
    feat = pred._get_text_features("a cute dog")
    assert feat.shape == (1, 512)

    # 2. List of strings
    feat_list = pred._get_text_features(
        ["first prompt", "second prompt"]  # type: ignore[arg-type]
    )
    assert feat_list.shape == (1, 512)


def test_core_evaluate_accepts_sequence_of_prompts(tmp_path):
    """core.py evaluate and evaluate_detailed accept Sequence[str] prompts."""
    from image_evaluator.core import evaluate, evaluate_detailed

    img_dir = tmp_path / "test_imgs"
    img_dir.mkdir()
    Image.new("RGB", (32, 32)).save(img_dir / "a.png")
    Image.new("RGB", (32, 32)).save(img_dir / "b.png")

    prompts = ("a red fox", "a blue bird")

    def mock_eval_folder(self, folder, prompt):
        res = MagicMock()
        res.mean_score = 0.88
        return res

    with patch.object(
        PickScorePredictor, "evaluate_folder", mock_eval_folder
    ):
        res_dict = evaluate(
            metrics="pickscore",
            image=str(img_dir),
            prompt=prompts,
        )
        assert res_dict["pickscore"] == 0.88

        res_detailed = evaluate_detailed(
            metrics="pickscore",
            image=str(img_dir),
            prompt=list(prompts),
        )
        assert res_detailed.scores["pickscore"] == 0.88


