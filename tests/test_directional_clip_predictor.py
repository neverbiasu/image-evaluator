from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from image_evaluator.directional_clip_predictor import (
    DirectionalClipPredictor,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_clip_components():
    """Create minimal mock CLIP components."""
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_tokenizer = MagicMock()

    # Default: all features point in the same direction
    feat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    mock_model.get_image_features.return_value = feat
    mock_model.get_text_features.return_value = feat

    mock_processor.side_effect = lambda images=None, return_tensors=None: {
        "pixel_values": torch.zeros((1, 3, 224, 224))
    }
    mock_tokenizer.side_effect = (
        lambda text, padding=True, truncation=True, max_length=77,
        return_tensors=None: {
            "input_ids": torch.tensor([[101, 2054, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
    )
    return mock_model, mock_processor, mock_tokenizer


@pytest.fixture
def predictor(mock_clip_components):
    """DirectionalClipPredictor with mocked CLIP backbone on CPU."""
    mock_model, mock_processor, mock_tokenizer = mock_clip_components
    with patch(
        "image_evaluator.directional_clip_predictor.AutoModel"
        ".from_pretrained",
        return_value=mock_model,
    ), patch(
        "image_evaluator.directional_clip_predictor.AutoProcessor"
        ".from_pretrained",
        return_value=mock_processor,
    ), patch(
        "image_evaluator.directional_clip_predictor.AutoTokenizer"
        ".from_pretrained",
        return_value=mock_tokenizer,
    ):
        instance = DirectionalClipPredictor(
            clip_model="mock-clip", device="cpu"
        )
        instance.model = mock_model
        return instance


@pytest.fixture
def dummy_image(tmp_path):
    """Small 64x64 PNG on disk."""
    p = tmp_path / "img.png"
    Image.new("RGB", (64, 64), color="blue").save(p)
    return str(p)


@pytest.fixture
def dummy_image_small(tmp_path):
    """Small 32x32 PNG on disk for variation."""
    p = tmp_path / "img_small.png"
    Image.new("RGB", (32, 32), color="red").save(p)
    return str(p)


# ---------------------------------------------------------------------------
# Score ordering: positive edit should rank above negative edit
# ---------------------------------------------------------------------------


def test_positive_edit_ranks_above_negative_edit(
    predictor, mock_clip_components
):
    """Directional CLIP must prefer edits that move toward the target prompt.

    Positive edit: delta_img aligned with delta_txt -> high score.
    Negative edit: delta_img opposed to delta_txt -> low score.
    Uses PIL.Image inputs to avoid file I/O.
    """
    mock_model, _, _ = mock_clip_components

    txt_src = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    txt_tgt = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    img_src_pos = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    img_edit_pos = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    img_src_neg = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    img_edit_neg = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    src_pil = Image.new("RGB", (4, 4), color="blue")
    edit_pil = Image.new("RGB", (4, 4), color="red")

    # Positive edit
    img_call = [0]
    txt_call = [0]

    def img_side_pos(**kwargs):
        img_call[0] += 1
        return img_src_pos if img_call[0] % 2 == 1 else img_edit_pos

    def txt_side(**kwargs):
        txt_call[0] += 1
        return txt_src if txt_call[0] % 2 == 1 else txt_tgt

    mock_model.get_image_features.side_effect = img_side_pos
    mock_model.get_text_features.side_effect = txt_side
    score_pos = predictor.evaluate_directional_clip(
        image_src=src_pil,
        image_edit=edit_pil,
        prompt_src="a dog",
        prompt_target="a cat",
    )

    # Negative edit
    img_call[0] = 0
    txt_call[0] = 0

    def img_side_neg(**kwargs):
        img_call[0] += 1
        return img_src_neg if img_call[0] % 2 == 1 else img_edit_neg

    mock_model.get_image_features.side_effect = img_side_neg
    mock_model.get_text_features.side_effect = txt_side
    score_neg = predictor.evaluate_directional_clip(
        image_src=src_pil,
        image_edit=edit_pil,
        prompt_src="a dog",
        prompt_target="a cat",
    )

    assert score_pos > score_neg



# ---------------------------------------------------------------------------
# Numerical stability: near-zero delta returns 0.0
# ---------------------------------------------------------------------------


def test_zero_image_delta_returns_zero(predictor, mock_clip_components):
    """When src and edit images are identical, score must be 0.0."""
    mock_model, _, _ = mock_clip_components
    same_feat = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    mock_model.get_image_features.return_value = same_feat

    txt_src = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    txt_tgt = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    call_count = [0]

    def text_side_effect(**kwargs):
        call_count[0] += 1
        return txt_src if call_count[0] % 2 == 1 else txt_tgt

    mock_model.get_text_features.side_effect = text_side_effect

    same_pil = Image.new("RGB", (4, 4), color="green")
    score = predictor.evaluate_directional_clip(
        image_src=same_pil,
        image_edit=same_pil,
        prompt_src="a dog",
        prompt_target="a cat",
    )
    assert score == pytest.approx(0.0, abs=1e-6)



def test_zero_text_delta_returns_zero(predictor, mock_clip_components):
    """When src and target prompts are identical, score must be 0.0."""
    mock_model, _, _ = mock_clip_components
    same_txt = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    mock_model.get_text_features.return_value = same_txt

    img_src = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    img_edit = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    call_count = [0]

    def img_side_effect(**kwargs):
        call_count[0] += 1
        return img_src if call_count[0] % 2 == 1 else img_edit

    mock_model.get_image_features.side_effect = img_side_effect

    src_pil = Image.new("RGB", (4, 4), color="blue")
    edit_pil = Image.new("RGB", (4, 4), color="red")
    score = predictor.evaluate_directional_clip(
        image_src=src_pil,
        image_edit=edit_pil,
        prompt_src="same prompt",
        prompt_target="same prompt",
    )
    assert score == pytest.approx(0.0, abs=1e-6)




# ---------------------------------------------------------------------------
# Input form: file path
# ---------------------------------------------------------------------------


def test_evaluate_with_file_paths(predictor, dummy_image, dummy_image_small):
    """evaluate_directional_clip must accept str file paths."""
    result = predictor.evaluate_directional_clip(
        image_src=dummy_image,
        image_edit=dummy_image_small,
        prompt_src="a blue square",
        prompt_target="a red square",
    )
    assert isinstance(result, float)
    assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Input form: PIL.Image
# ---------------------------------------------------------------------------


def test_evaluate_with_pil_images(predictor):
    """evaluate_directional_clip must accept PIL.Image objects."""
    src = Image.new("RGB", (64, 64), color=(0, 0, 255))
    edit = Image.new("RGB", (64, 64), color=(255, 0, 0))
    result = predictor.evaluate_directional_clip(
        image_src=src,
        image_edit=edit,
        prompt_src="blue",
        prompt_target="red",
    )
    assert isinstance(result, float)
    assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# _extract_tensor: handles Tensor and pooler_output object
# ---------------------------------------------------------------------------


def test_extract_tensor_from_plain_tensor():
    """_extract_tensor must return a Tensor unchanged."""
    t = torch.tensor([[1.0, 2.0]])
    result = DirectionalClipPredictor._extract_tensor(t)
    assert torch.equal(result, t)


def test_extract_tensor_from_pooler_output():
    """_extract_tensor must pull .pooler_output from HuggingFace output."""
    expected = torch.tensor([[3.0, 4.0]])
    fake_output = SimpleNamespace(pooler_output=expected)
    result = DirectionalClipPredictor._extract_tensor(fake_output)
    assert torch.equal(result, expected)


# ---------------------------------------------------------------------------
# _cosine_similarity: basic numerical correctness
# ---------------------------------------------------------------------------


def test_cosine_similarity_parallel_vectors():
    """Parallel unit vectors should yield 1.0."""
    a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    score = DirectionalClipPredictor._cosine_similarity(a, a.clone())
    assert score == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_orthogonal_vectors():
    """Orthogonal unit vectors should yield 0.0."""
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    score = DirectionalClipPredictor._cosine_similarity(a, b)
    assert score == pytest.approx(0.0, abs=1e-5)


def test_cosine_similarity_antiparallel_vectors():
    """Antiparallel unit vectors should yield -1.0."""
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([-1.0, 0.0])
    score = DirectionalClipPredictor._cosine_similarity(a, b)
    assert score == pytest.approx(-1.0, abs=1e-5)


def test_cosine_similarity_near_zero_vector_returns_zero():
    """A near-zero vector must return 0.0, not NaN."""
    a = torch.tensor([0.0, 0.0])
    b = torch.tensor([1.0, 0.0])
    score = DirectionalClipPredictor._cosine_similarity(a, b)
    assert score == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Export: DirectionalClipPredictor reachable from package root
# ---------------------------------------------------------------------------


def test_package_exports_directional_clip_predictor():
    """DirectionalClipPredictor must be importable from image_evaluator."""
    from image_evaluator import DirectionalClipPredictor as DC

    assert DC is not None
    assert DC.__name__ == "DirectionalClipPredictor"
