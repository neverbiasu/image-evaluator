from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from image_evaluator.arcface_dist_predictor import ArcFaceDistPredictor


@pytest.fixture
def predictor():
    with patch(
        "image_evaluator.arcface_dist_predictor.FaceAnalysis"
    ) as face_analysis:
        instance = MagicMock()
        face_analysis.return_value = instance
        result = ArcFaceDistPredictor(device="cpu")
        return result


def test_init_selects_requested_device_context():
    with patch(
        "image_evaluator.arcface_dist_predictor.FaceAnalysis"
    ) as face_analysis:
        ArcFaceDistPredictor(device="cpu")
        face_analysis.return_value.prepare.assert_called_with(ctx_id=-1)

        ArcFaceDistPredictor(device="cuda")
        face_analysis.return_value.prepare.assert_called_with(ctx_id=0)


def test_get_face_embedding_handles_face_and_no_face(predictor, tmp_path):
    image_path = tmp_path / "face.png"
    Image.new("RGB", (32, 32)).save(image_path)
    embedding = np.array([0.5, 0.5], dtype=np.float32)

    predictor.app.get.return_value = [SimpleNamespace(embedding=embedding)]
    assert np.array_equal(
        predictor.get_face_embedding(str(image_path)), embedding
    )

    predictor.app.get.return_value = []
    assert predictor.get_face_embedding(str(image_path)) is None


def test_distance_semantics_and_missing_face(predictor):
    same = np.array([1.0, 0.0], dtype=np.float32)
    orthogonal = np.array([0.0, 1.0], dtype=np.float32)

    with patch.object(
        predictor, "get_face_embedding", side_effect=[same, same]
    ):
        assert predictor.evaluate_arcface_distance("ref", "gen") == 0.0

    with patch.object(
        predictor, "get_face_embedding", side_effect=[same, orthogonal]
    ):
        assert predictor.evaluate_arcface_distance("ref", "gen") == 1.0

    with patch.object(
        predictor, "get_face_embedding", side_effect=[same, None]
    ):
        assert predictor.evaluate_arcface_distance("ref", "gen") is None


def test_folder_distance_stem_matched_mean(predictor, tmp_path):
    reference = tmp_path / "reference"
    generated = tmp_path / "generated"
    reference.mkdir()
    generated.mkdir()
    for folder in (reference, generated):
        (folder / "01.png").touch()
        (folder / "02.png").touch()

    with patch.object(
        predictor, "evaluate_arcface_distance", side_effect=[0.2, 0.4]
    ):
        result = predictor.evaluate_folder_arcface_distance(
            str(reference), str(generated)
        )
    assert result == pytest.approx(0.3)

    with patch.object(
        predictor, "evaluate_arcface_distance", side_effect=[0.1, None]
    ):
        with pytest.raises(ValueError, match="Unscorable"):
            predictor.evaluate_folder_arcface_distance(
                str(reference), str(generated)
            )
