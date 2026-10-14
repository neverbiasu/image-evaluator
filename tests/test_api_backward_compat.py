"""Tests for backward-compatible aliases and formatting methods."""

from image_evaluator import (
    ArcFaceDistPredictor,
    ClipScorePredictor,
    FIDPredictor,
    KIDPredictor,
    LaionAIAestheticPredictor,
    LaionAiAestheticPredictor,
    PickScorePredictor,
)
from image_evaluator.fid_predictor import FIDResult
from image_evaluator.kid_predictor import KIDResult


def test_laion_aesthetic_alias():
    assert LaionAiAestheticPredictor is LaionAIAestheticPredictor


def test_arcface_aliases():
    assert hasattr(ArcFaceDistPredictor, "evaluate_arcface")
    assert hasattr(ArcFaceDistPredictor, "evaluate_folder_arcface")
    arc_eval = ArcFaceDistPredictor.evaluate_arcface
    arc_canonical = ArcFaceDistPredictor.evaluate_arcface_distance
    assert arc_eval == arc_canonical
    arc_folder = ArcFaceDistPredictor.evaluate_folder_arcface
    arc_folder_canonical = (
        ArcFaceDistPredictor.evaluate_folder_arcface_distance
    )
    assert arc_folder == arc_folder_canonical


def test_kid_aliases():
    assert hasattr(KIDPredictor, "evaluate_kid")
    assert KIDPredictor.evaluate_kid == KIDPredictor.evaluate_folder_kid


def test_pickscore_aliases():
    assert hasattr(PickScorePredictor, "evaluate_folder_pickscore")
    pick_folder = PickScorePredictor.evaluate_folder_pickscore
    pick_canonical = PickScorePredictor.evaluate_folder
    assert pick_folder == pick_canonical


def test_clip_and_fid_imports():
    assert hasattr(ClipScorePredictor, "evaluate_clip_score")
    assert hasattr(FIDPredictor, "evaluate_fid")


def test_fid_result_formatting():
    res = FIDResult(
        fid=12.34567,
        backend="clean-fid",
        version="0.1.35",
        mode="clean",
        model="clean-fid",
        device="cpu",
        Nref=10,
        Ngen=10,
    )
    assert f"{res:.2f}" == "12.35"
    assert f"{res:.4f}" == "12.3457"
    assert float(res) == 12.34567


def test_kid_result_formatting():
    res = KIDResult(
        kid=0.0123456,
        backend="clean-fid",
        version="0.1.35",
        mode="clean",
        model="clean-fid",
        device="cpu",
        num_subsets=100,
        max_subset_size=1000,
        seed=0,
        Nref=10,
        Ngen=10,
    )
    assert f"{res:.4f}" == "0.0123"
    assert float(res) == 0.0123456
