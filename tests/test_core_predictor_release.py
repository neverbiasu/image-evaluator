"""Lifecycle tests for predictor release in ``image_evaluator.core.evaluate``.

Every metric branch must drop its predictor and call
``_release_eval_memory`` in a ``finally`` path, on success and on failure,
without swallowing the original exception. All predictors are mocked; no
real model is instantiated or downloaded.
"""

import importlib.machinery
import importlib.util
import sys
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

if (
    "insightface" not in sys.modules
    and importlib.util.find_spec("insightface") is None
):
    mock_insightface = MagicMock()
    mock_insightface.__spec__ = importlib.machinery.ModuleSpec(
        "insightface", None
    )
    sys.modules["insightface"] = mock_insightface
    sys.modules["insightface.app"] = mock_insightface.app

from image_evaluator import core

# metric id -> (predictor module, class name, single-input scoring method)
# fid/kid are folder-only, so their method is the folder method.
METRIC_PREDICTORS = {
    "aesthetic": (
        "laion_ai_aesthetic_predictor",
        "LaionAIAestheticPredictor",
        "evaluate_aesthetic_score",
    ),
    "clip": (
        "clip_score_predictor",
        "ClipScorePredictor",
        "evaluate_clip_score",
    ),
    "arcface": (
        "arcface_dist_predictor",
        "ArcFaceDistPredictor",
        "evaluate_arcface_distance",
    ),
    "lpips": ("lpips_predictor", "LPIPSPredictor", "evaluate_lpips"),
    "ssim": ("ssim_predictor", "SSIMPredictor", "evaluate_ssim"),
    "psnr": ("psnr_predictor", "PSNRPredictor", "evaluate_psnr"),
    "fid": ("fid_predictor", "FIDPredictor", "evaluate_folder_fid"),
    "kid": ("kid_predictor", "KIDPredictor", "evaluate_folder_kid"),
    "pickscore": ("pickscore_predictor", "PickScorePredictor", "evaluate"),
    "directional_clip": (
        "directional_clip_predictor",
        "DirectionalClipPredictor",
        "evaluate_directional_clip",
    ),
    "clip_i": ("clip_i_predictor", "ClipIPredictor", "evaluate_clip_i"),
    "dino_similarity": (
        "dino_similarity_predictor",
        "DinoSimilarityPredictor",
        "evaluate_dino_similarity",
    ),
    "hpsv2": ("hpsv2_predictor", "Hpsv2Predictor", "evaluate_hpsv2"),
    "image_reward": (
        "image_reward_predictor",
        "ImageRewardPredictor",
        "evaluate_image_reward",
    ),
    "vqascore": (
        "vqascore_predictor",
        "VQAScorePredictor",
        "evaluate_vqascore",
    ),
}


def test_metric_table_covers_every_supported_metric():
    """The release table must not silently miss a registered metric."""
    assert set(METRIC_PREDICTORS) == core.SUPPORTED_METRICS


def _evaluate_kwargs(metric, tmp_path):
    """Build the minimal valid evaluate() arguments for one metric."""
    if metric in core.DATASET_METRICS:
        gen_dir = tmp_path / "gen"
        ref_dir = tmp_path / "ref"
        gen_dir.mkdir()
        ref_dir.mkdir()
        kwargs = {"image": str(gen_dir), "reference": str(ref_dir)}
    else:
        kwargs = {"image": Image.new("RGB", (32, 32), color="blue")}
        if metric in core.REFERENCE_METRICS:
            kwargs["reference"] = Image.new("RGB", (32, 32), color="red")
    if metric in core.PROMPT_METRICS:
        kwargs["prompt"] = "a red square"
    if metric in core.SOURCE_PROMPT_METRICS:
        kwargs["source_prompt"] = "a blue square"
    return kwargs


def _patch_predictor(metric, instance=None, ctor_error=None):
    module, cls_name, _ = METRIC_PREDICTORS[metric]
    target = f"image_evaluator.{module}.{cls_name}"
    if ctor_error is not None:
        return patch(target, side_effect=ctor_error)
    return patch(target, return_value=instance)


@pytest.mark.parametrize("metric", sorted(METRIC_PREDICTORS))
def test_release_runs_after_successful_scoring(metric, tmp_path):
    """Success path: score is returned, then release runs exactly once."""
    _, _, method = METRIC_PREDICTORS[metric]
    events = []
    instance = MagicMock()
    getattr(instance, method).side_effect = (
        lambda *a, **k: events.append("score") or 0.5
    )

    with _patch_predictor(metric, instance), patch(
        "image_evaluator.core._release_eval_memory",
        side_effect=lambda *a, **k: events.append("release"),
    ) as release:
        result = core.evaluate(
            metrics=[metric], **_evaluate_kwargs(metric, tmp_path)
        )

    assert metric in result
    release.assert_called_once_with(suppress_errors=False)
    assert events == ["score", "release"]


@pytest.mark.parametrize("metric", sorted(METRIC_PREDICTORS))
def test_release_runs_and_error_propagates_when_scoring_fails(
    metric, tmp_path
):
    """Failure path: original exception surfaces and release still runs."""
    _, _, method = METRIC_PREDICTORS[metric]
    error = RuntimeError(f"{metric} scoring failed")
    instance = MagicMock()
    getattr(instance, method).side_effect = error

    with _patch_predictor(metric, instance), patch(
        "image_evaluator.core._release_eval_memory"
    ) as release:
        with pytest.raises(RuntimeError) as exc_info:
            core.evaluate(
                metrics=[metric], **_evaluate_kwargs(metric, tmp_path)
            )

    assert exc_info.value is error
    release.assert_called_once_with(suppress_errors=True)


@pytest.mark.parametrize(
    "metric", ["aesthetic", "fid", "directional_clip", "hpsv2"]
)
def test_release_runs_when_construction_fails(metric, tmp_path):
    """Constructor failure: exception surfaces and release still runs."""
    error = RuntimeError(f"{metric} construction failed")

    with _patch_predictor(metric, ctor_error=error), patch(
        "image_evaluator.core._release_eval_memory"
    ) as release:
        with pytest.raises(RuntimeError) as exc_info:
            core.evaluate(
                metrics=[metric], **_evaluate_kwargs(metric, tmp_path)
            )

    assert exc_info.value is error
    release.assert_called_once_with(suppress_errors=True)


def test_multi_metric_releases_once_per_predictor():
    """Each selected metric releases its own predictor."""
    image = Image.new("RGB", (32, 32), color="blue")
    reference = Image.new("RGB", (32, 32), color="red")

    with _patch_predictor("ssim", MagicMock()), _patch_predictor(
        "psnr", MagicMock()
    ), _patch_predictor("clip_i", MagicMock()), patch(
        "image_evaluator.core._release_eval_memory"
    ) as release:
        core.evaluate(
            metrics=["ssim", "psnr", "clip_i"],
            image=image,
            reference=reference,
        )

    assert release.call_count == 3


@pytest.mark.parametrize("cuda_available", [True, False])
def test_release_helper_semantics(cuda_available):
    """Helper keeps gc.collect plus CUDA cache clearing only when available."""
    with patch("image_evaluator.core.gc.collect") as collect, patch(
        "image_evaluator.core.torch.cuda.is_available",
        return_value=cuda_available,
    ), patch("image_evaluator.core.torch.cuda.empty_cache") as empty_cache:
        core._release_eval_memory()

    collect.assert_called_once_with()
    assert empty_cache.call_count == (1 if cuda_available else 0)


def test_scoring_exception_preserved_when_gc_collect_fails(tmp_path):
    """Scoring exception propagates even if gc.collect raises."""
    error = RuntimeError("primary scoring failure")
    instance = MagicMock()
    instance.evaluate_aesthetic_score.side_effect = error

    with _patch_predictor("aesthetic", instance), patch(
        "image_evaluator.core.gc.collect",
        side_effect=RuntimeError("gc failure"),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            core.evaluate(
                metrics=["aesthetic"],
                image=Image.new("RGB", (32, 32), color="blue"),
            )

    assert exc_info.value is error


def test_scoring_exception_preserved_when_empty_cache_fails(tmp_path):
    """Scoring exception propagates even if torch.cuda.empty_cache raises."""
    error = RuntimeError("primary scoring failure")
    instance = MagicMock()
    instance.evaluate_aesthetic_score.side_effect = error

    with _patch_predictor("aesthetic", instance), patch(
        "image_evaluator.core.torch.cuda.is_available", return_value=True
    ), patch(
        "image_evaluator.core.torch.cuda.empty_cache",
        side_effect=RuntimeError("empty_cache failure"),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            core.evaluate(
                metrics=["aesthetic"],
                image=Image.new("RGB", (32, 32), color="blue"),
            )

    assert exc_info.value is error


def test_successful_scoring_cleanup_exception_propagates(tmp_path):
    """When scoring succeeds but cleanup fails, cleanup error propagates."""
    instance = MagicMock()
    instance.evaluate_aesthetic_score.return_value = 5.5
    cleanup_err = RuntimeError("cleanup failed unexpectedly")

    with _patch_predictor("aesthetic", instance), patch(
        "image_evaluator.core.gc.collect", side_effect=cleanup_err
    ):
        with pytest.raises(RuntimeError) as exc_info:
            core.evaluate(
                metrics=["aesthetic"],
                image=Image.new("RGB", (32, 32), color="blue"),
            )

    assert exc_info.value is cleanup_err


def test_caller_in_except_cleanup_propagates_on_success(tmp_path):
    """When caller is in except block and eval succeeds, cleanup propagates."""
    instance = MagicMock()
    instance.evaluate_aesthetic_score.return_value = 5.5
    cleanup_err = RuntimeError("cleanup failed unexpectedly")

    with _patch_predictor("aesthetic", instance), patch(
        "image_evaluator.core.gc.collect", side_effect=cleanup_err
    ):
        try:
            raise ValueError("caller outer error")
        except ValueError:
            with pytest.raises(RuntimeError) as exc_info:
                core.evaluate(
                    metrics=["aesthetic"],
                    image=Image.new("RGB", (32, 32), color="blue"),
                )

    assert exc_info.value is cleanup_err


def test_caller_in_except_cuda_error_propagates_on_success(tmp_path):
    """When caller in except and eval succeeds, cuda err propagates."""
    instance = MagicMock()
    instance.evaluate_aesthetic_score.return_value = 5.5
    cuda_err = RuntimeError("cuda check failed")

    with _patch_predictor("aesthetic", instance), patch(
        "image_evaluator.core.torch.cuda.is_available", side_effect=cuda_err
    ):
        try:
            raise ValueError("caller outer error")
        except ValueError:
            with pytest.raises(RuntimeError) as exc_info:
                core.evaluate(
                    metrics=["aesthetic"],
                    image=Image.new("RGB", (32, 32), color="blue"),
                )

    assert exc_info.value is cuda_err


def test_constructor_failure_suppresses_cleanup_error(tmp_path):
    """When ctor fails and cleanup also fails, ctor error is preserved."""
    ctor_error = RuntimeError("ctor failure")
    cleanup_error = RuntimeError("cleanup failure")

    with _patch_predictor("aesthetic", ctor_error=ctor_error), patch(
        "image_evaluator.core.gc.collect", side_effect=cleanup_error
    ):
        with pytest.raises(RuntimeError) as exc_info:
            core.evaluate(
                metrics=["aesthetic"],
                image=Image.new("RGB", (32, 32), color="blue"),
            )

    assert exc_info.value is ctor_error


def test_predictor_unreachable_when_cleanup_starts(tmp_path):
    """Predictor must be unreachable when cleanup starts."""
    import weakref

    ref_box = []
    predictor_alive_during_cleanup = []

    class DummyPredictor:
        def evaluate_aesthetic_score(self, img):
            return 4.2

    def factory(**kw):
        inst = DummyPredictor()
        ref_box.append(weakref.ref(inst))
        return inst

    def cleanup_spy(suppress_errors=False):
        wr = ref_box[0]
        predictor_alive_during_cleanup.append(wr() is not None)

    with patch(
        "image_evaluator.laion_ai_aesthetic_predictor.LaionAIAestheticPredictor",
        side_effect=factory,
    ), patch(
        "image_evaluator.core._release_eval_memory",
        side_effect=cleanup_spy,
    ):
        result = core.evaluate(
            metrics=["aesthetic"],
            image=Image.new("RGB", (32, 32), color="blue"),
        )

    assert result["aesthetic"] == 4.2
    assert predictor_alive_during_cleanup == [False]
    assert ref_box[0]() is None


def test_multi_metric_previous_predictor_dead_before_next_constructed():
    """Previous predictor must be dead before next predictor is built."""
    import weakref

    refs = {}
    ssim_alive_at_psnr_ctor = []

    class DummySSIM:
        def evaluate_ssim(self, ref, img):
            return 0.95

    class DummyPSNR:
        def evaluate_psnr(self, ref, img):
            return 32.0

    def ssim_factory(**kw):
        inst = DummySSIM()
        refs["ssim"] = weakref.ref(inst)
        return inst

    def psnr_factory(**kw):
        wr_ssim = refs.get("ssim")
        ssim_alive_at_psnr_ctor.append(
            wr_ssim() is not None if wr_ssim else None
        )
        inst = DummyPSNR()
        refs["psnr"] = weakref.ref(inst)
        return inst

    image = Image.new("RGB", (32, 32), color="blue")
    reference = Image.new("RGB", (32, 32), color="red")

    with patch(
        "image_evaluator.ssim_predictor.SSIMPredictor",
        side_effect=ssim_factory,
    ), patch(
        "image_evaluator.psnr_predictor.PSNRPredictor",
        side_effect=psnr_factory,
    ):
        res = core.evaluate(
            metrics=["ssim", "psnr"],
            image=image,
            reference=reference,
        )

    assert "ssim" in res and "psnr" in res
    assert ssim_alive_at_psnr_ctor == [False]
    assert refs["ssim"]() is None
    assert refs["psnr"]() is None


def test_scoring_failure_frames_do_not_hold_predictor(tmp_path):
    """On scoring failure, helper/evaluate frames must not hold predictor."""
    err = RuntimeError("scoring blow up")

    class DummyPredictor:
        def evaluate_aesthetic_score(self, img):
            raise err

    frames_had_pred = []

    def cleanup_spy(suppress_errors=False):
        import sys

        f = sys._getframe(0)
        had_pred = False
        while f:
            if f.f_code.co_name in ("_run_with_predictor", "evaluate"):
                if "pred" in f.f_locals or any(
                    k.startswith("pred_") for k in f.f_locals
                ):
                    had_pred = True
            f = f.f_back
        frames_had_pred.append(had_pred)

    with patch(
        "image_evaluator.laion_ai_aesthetic_predictor.LaionAIAestheticPredictor",
        side_effect=lambda **kw: DummyPredictor(),
    ), patch(
        "image_evaluator.core._release_eval_memory",
        side_effect=cleanup_spy,
    ):
        with pytest.raises(RuntimeError) as exc_info:
            core.evaluate(
                metrics=["aesthetic"],
                image=Image.new("RGB", (32, 32), color="blue"),
            )

    assert exc_info.value is err
    assert frames_had_pred == [False]
