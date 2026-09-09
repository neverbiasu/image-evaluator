import os
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from image_evaluator.clip_score_predictor import (
    ClipScorePredictor,
    DummyDataset,
)


@pytest.fixture
def mock_clip_components():
    """Create mock components for CLIP model, processor, and tokenizer."""
    mock_model = MagicMock()
    # Return 2D embedding tensor (batch_size=1, hidden_dim=4)
    mock_model.get_image_features.side_effect = (
        lambda **kwargs: torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        )
    )
    mock_model.get_text_features.side_effect = (
        lambda **kwargs: torch.tensor(
            [[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32
        )
    )

    mock_processor = MagicMock()
    mock_processor.side_effect = lambda text=None, images=None: {
        "pixel_values": torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    }

    mock_tokenizer = MagicMock()
    mock_tokenizer.side_effect = lambda data, **kwargs: {
        "input_ids": torch.tensor([[101, 2054, 102]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }

    return mock_model, mock_processor, mock_tokenizer


@pytest.fixture
def predictor(mock_clip_components):
    """Instantiate ClipScorePredictor with mocked dependencies on CPU."""
    mock_model, mock_processor, mock_tokenizer = mock_clip_components
    with patch(
        "image_evaluator.clip_score_predictor.AutoModel.from_pretrained",
        return_value=mock_model,
    ), patch(
        "image_evaluator.clip_score_predictor.AutoProcessor.from_pretrained",
        return_value=mock_processor,
    ), patch(
        "image_evaluator.clip_score_predictor.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    ):
        instance = ClipScorePredictor(clip_model="mock-clip", device="cpu")
        instance.model = mock_model
        return instance


@pytest.fixture
def dummy_image_file(tmp_path):
    """Create a temporary dummy PNG image file."""
    img_path = tmp_path / "test_image.png"
    img = Image.new("RGB", (64, 64), color="red")
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def dummy_text_file(tmp_path):
    """Create a temporary dummy text file."""
    txt_path = tmp_path / "test_prompt.txt"
    txt_path.write_text("a red square", encoding="utf-8")
    return str(txt_path)


def test_combine_without_prefix_scalar_file_and_literal(
    dummy_image_file, tmp_path
):
    """Verify files and literal strings remain scalar strings."""
    dataset = DummyDataset(
        dummy_image_file, "a literal prompt", real_flag="img", fake_flag="txt"
    )
    # Existing image file remains a scalar string
    assert dataset.real_folder == dummy_image_file
    assert isinstance(dataset.real_folder, str)

    # Non-existent string prompt remains a scalar string
    assert dataset.fake_folder == "a literal prompt"
    assert isinstance(dataset.fake_folder, str)


def test_combine_without_prefix_directory_and_hidden_files(tmp_path):
    """Verify directory inputs become sorted lists and ignore hidden files."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "b.png").write_text("dummy b")
    (img_dir / "a.png").write_text("dummy a")
    (img_dir / ".DS_Store").write_text("hidden")
    (img_dir / ".hidden_file").write_text("hidden")

    dataset = DummyDataset(
        str(img_dir), "prompt", real_flag="img", fake_flag="txt"
    )
    assert isinstance(dataset.real_folder, list)
    assert len(dataset.real_folder) == 2
    # Check alphabetical sorting
    assert os.path.basename(dataset.real_folder[0]) == "a.png"
    assert os.path.basename(dataset.real_folder[1]) == "b.png"


def test_dummy_dataset_scalar_image_and_prompt_loading(
    dummy_image_file, dummy_text_file, mock_clip_components
):
    """Verify scalar image file and prompt text file load into sample dict."""
    _, mock_processor, mock_tokenizer = mock_clip_components
    dataset = DummyDataset(
        dummy_image_file,
        dummy_text_file,
        real_flag="img",
        fake_flag="txt",
        transform=mock_processor,
        tokenizer=mock_tokenizer,
    )
    assert len(dataset) == 1
    sample = dataset[0]
    assert "real" in sample and "fake" in sample
    assert "pixel_values" in sample["real"]
    assert "input_ids" in sample["fake"]


def test_single_file_modality_routing(
    predictor, mock_clip_components, dummy_image_file
):
    """Verify image and text features receive correct input tensors."""
    mock_model, _, _ = mock_clip_components
    mock_model.get_image_features.reset_mock()
    mock_model.get_text_features.reset_mock()

    prompt = "a photo of a red square"
    score = predictor.evaluate_clip_score(dummy_image_file, prompt)

    assert isinstance(score, float)
    assert mock_model.get_image_features.called
    assert mock_model.get_text_features.called

    img_kwargs = mock_model.get_image_features.call_args.kwargs
    assert "pixel_values" in img_kwargs
    assert "input_ids" not in img_kwargs

    txt_kwargs = mock_model.get_text_features.call_args.kwargs
    assert "input_ids" in txt_kwargs
    assert "pixel_values" not in txt_kwargs


def test_single_file_numeric_score_calculation(
    predictor, mock_clip_components, dummy_image_file
):
    """Verify cosine similarity calculation between normalized vectors."""
    mock_model, _, _ = mock_clip_components
    # Orthogonal vectors -> dot product = 0.0
    mock_model.get_image_features.side_effect = (
        lambda **kwargs: torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    )
    mock_model.get_text_features.side_effect = (
        lambda **kwargs: torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    )

    score = predictor._evaluate_single_file(
        dummy_image_file, "some prompt", "img", "txt"
    )
    assert pytest.approx(score, abs=1e-5) == 0.0

    # Parallel vectors -> dot product = 1.0
    mock_model.get_image_features.side_effect = (
        lambda **kwargs: torch.tensor([[2.0, 0.0]], dtype=torch.float32)
    )
    mock_model.get_text_features.side_effect = (
        lambda **kwargs: torch.tensor([[5.0, 0.0]], dtype=torch.float32)
    )

    score = predictor._evaluate_single_file(
        dummy_image_file, "some prompt", "img", "txt"
    )
    assert pytest.approx(score, abs=1e-5) == 1.0


def test_single_file_with_text_file_prompt(
    predictor, mock_clip_components, dummy_image_file, dummy_text_file
):
    """Verify single file evaluation when prompt is a text file path."""
    mock_model, _, _ = mock_clip_components
    mock_model.get_image_features.side_effect = (
        lambda **kwargs: torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    )
    mock_model.get_text_features.side_effect = (
        lambda **kwargs: torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    )

    score = predictor.evaluate_clip_score(dummy_image_file, dummy_text_file)
    assert isinstance(score, float)
    assert pytest.approx(score, abs=1e-5) == 1.0


def test_single_file_inverted_modality_flags(
    predictor, mock_clip_components, dummy_image_file
):
    """Verify _evaluate_single_file correctly handles inverted flag args."""
    mock_model, _, _ = mock_clip_components
    mock_model.get_image_features.reset_mock()
    mock_model.get_text_features.reset_mock()

    score = predictor._evaluate_single_file(
        "a photo prompt", dummy_image_file, image_flag="txt", text_flag="img"
    )

    assert isinstance(score, float)
    img_kwargs = mock_model.get_image_features.call_args.kwargs
    assert "pixel_values" in img_kwargs
    txt_kwargs = mock_model.get_text_features.call_args.kwargs
    assert "input_ids" in txt_kwargs


def test_single_file_invalid_modality_flags(predictor, dummy_image_file):
    """Verify ValueError is raised when invalid flags are provided."""
    with pytest.raises(
        ValueError, match="Must specify one 'img' and one 'txt'"
    ):
        predictor._evaluate_single_file(
            dummy_image_file, "prompt", image_flag="img", text_flag="img"
        )


def test_forward_modality_accepts_direct_tensor_features(predictor):
    """Verify image and text paths pass through direct Tensor returns."""
    img_input = {"pixel_values": torch.zeros((1, 3, 224, 224))}
    txt_input = {
        "input_ids": torch.tensor([[101, 2054, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    expected_img = torch.tensor([[1.0, 0.0]])
    expected_txt = torch.tensor([[0.0, 1.0]])
    predictor.model.get_image_features.side_effect = (
        lambda **kwargs: expected_img
    )
    predictor.model.get_text_features.side_effect = (
        lambda **kwargs: expected_txt
    )

    img_features = predictor._forward_modality(dict(img_input), "img")
    txt_features = predictor._forward_modality(dict(txt_input), "txt")

    assert isinstance(img_features, torch.Tensor)
    assert isinstance(txt_features, torch.Tensor)
    assert torch.equal(img_features, expected_img)
    assert torch.equal(txt_features, expected_txt)


def test_forward_modality_extracts_pooled_output_features(predictor):
    """Verify image and text paths extract Tensor from pooler_output."""
    from types import SimpleNamespace

    img_input = {"pixel_values": torch.zeros((1, 3, 224, 224))}
    txt_input = {
        "input_ids": torch.tensor([[101, 2054, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    expected_img = torch.tensor([[1.0, 0.0]])
    expected_txt = torch.tensor([[0.0, 1.0]])
    predictor.model.get_image_features.side_effect = (
        lambda **kwargs: SimpleNamespace(pooler_output=expected_img)
    )
    predictor.model.get_text_features.side_effect = (
        lambda **kwargs: SimpleNamespace(pooler_output=expected_txt)
    )

    img_features = predictor._forward_modality(dict(img_input), "img")
    txt_features = predictor._forward_modality(dict(txt_input), "txt")

    assert isinstance(img_features, torch.Tensor)
    assert isinstance(txt_features, torch.Tensor)
    assert torch.equal(img_features, expected_img)
    assert torch.equal(txt_features, expected_txt)


def test_forward_modality_unsqueezes_3d_pixel_values_single_file_regression(
    predictor,
):
    """Regression: _evaluate_single_file yields C,H,W;

    _forward_modality must add batch dim only for 3D image.
    """
    captured: dict = {}

    def capture_img(**kwargs):
        captured["shape"] = tuple(kwargs["pixel_values"].shape)
        captured["ndim"] = kwargs["pixel_values"].ndim
        return torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    predictor.model.get_image_features.side_effect = capture_img

    # Unbatched single-file case: C,H,W -> should be unsqueezed to 1,C,H,W
    img_3d = {"pixel_values": torch.zeros((3, 224, 224))}
    predictor._forward_modality(dict(img_3d), "img")
    assert captured["shape"] == (1, 3, 224, 224)
    assert captured["ndim"] == 4

    # Batched DataLoader case: B,C,H,W must stay unchanged (directory behavior)
    img_4d_single = {"pixel_values": torch.zeros((1, 3, 224, 224))}
    predictor._forward_modality(dict(img_4d_single), "img")
    assert captured["shape"] == (1, 3, 224, 224)

    img_4d_batch = {"pixel_values": torch.zeros((2, 3, 224, 224))}
    predictor._forward_modality(dict(img_4d_batch), "img")
    assert captured["shape"] == (2, 3, 224, 224)

    # Text modality must not be altered
    txt_1d = {
        "input_ids": torch.tensor([101, 102, 103]),
        "attention_mask": torch.tensor([1, 1, 1]),
    }
    # Ensure text path does not hit image unsqueeze logic
    predictor.model.get_text_features.side_effect = (
        lambda **kwargs: torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    )
    # Should not raise and should not modify image logic
    out = predictor._forward_modality(dict(txt_1d), "txt")
    assert isinstance(out, torch.Tensor)


def test_single_file_clip_evaluates_with_3d_pixel_values_end_to_end(
    predictor, mock_clip_components, dummy_image_file
):
    """End-to-end single-file regression: processor yields 3D after Dataset[0];

    score must compute without missing batch dimension.
    """
    mock_model, _, _ = mock_clip_components

    # Patch DummyDataset._load_img to emulate real processor slicing
    # to 3D (already does [0]). Ensure predictor uses mocked processor
    # that mimics real: returns batched then sliced. Here we verify
    # evaluate_clip_score does not raise ValueError from missing batch dim.
    mock_model.get_image_features.side_effect = (
        lambda **kwargs: torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    )
    mock_model.get_text_features.side_effect = (
        lambda **kwargs: torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    )

    # Capture shape seen by model to ensure fix applied
    shapes = {}

    def wrapped_img(**kwargs):
        shapes["pixel"] = tuple(kwargs["pixel_values"].shape)
        return torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    mock_model.get_image_features.side_effect = wrapped_img

    score = predictor.evaluate_clip_score(dummy_image_file, "a photo prompt")
    assert isinstance(score, float)
    # Must have been unsqueezed to 4D before model call
    assert len(shapes["pixel"]) == 4
    assert shapes["pixel"][1:] == (3, 224, 224)
    assert pytest.approx(score, abs=1e-5) == 1.0
