import gc
import os
import time
from collections.abc import Sequence
from typing import Any, Callable

import torch

from image_evaluator.registry import get_metric, list_metrics
from image_evaluator.result import EvaluationResult

_ALL_SPECS = list_metrics()
SUPPORTED_METRICS = {spec.id for spec in _ALL_SPECS}
PROMPT_METRICS = {
    spec.id for spec in _ALL_SPECS if "prompt" in spec.inputs.required
}
REFERENCE_METRICS = {
    spec.id
    for spec in _ALL_SPECS
    if "reference_image" in spec.inputs.required
    or "reference_collection" in spec.inputs.required
}
DATASET_METRICS = {
    spec.id
    for spec in _ALL_SPECS
    if "reference_collection" in spec.inputs.required
}
PAIRWISE_METRICS = {
    spec.id
    for spec in _ALL_SPECS
    if "reference_image" in spec.inputs.required
    and "arithmetic_mean_for_directory_inputs" in spec.aggregation
}
SOURCE_PROMPT_METRICS = {
    spec.id for spec in _ALL_SPECS if "source_prompt" in spec.inputs.required
}


def _release_eval_memory(suppress_errors: bool = False) -> None:
    """Collect garbage and clear the CUDA cache after a predictor is done.

    Args:
        suppress_errors: When True, suppress errors during cleanup so an
            in-flight exception from predictor construction or scoring is
            not overwritten. When False, cleanup errors propagate normally.
    """
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        if suppress_errors:
            return
        raise


def _run_with_predictor(
    factory: Callable[[], Any],
    operation: Callable[[Any], Any],
) -> Any:
    """Execute an operation with a predictor, guaranteeing reference release.

    Constructs the predictor via factory, executes operation to compute the
    score, drops the reference to the predictor completely, and then runs
    _release_eval_memory.

    If construction or operation raises, cleanup errors are suppressed to
    preserve the primary failure. If scoring succeeds, cleanup errors
    propagate normally.
    """
    pred = None
    has_error = True
    try:
        pred = factory()
        result = operation(pred)
        has_error = False
        return result
    finally:
        del pred
        _release_eval_memory(suppress_errors=has_error)


def evaluate(
    metrics: str | Sequence[str],
    image: Any,
    reference: Any = None,
    prompt: str | Sequence[str] | None = None,
    source_prompt: str | None = None,
    device: str | torch.device | None = None,
    detailed: bool = False,
    allow_download: bool = False,
    download_callback: Any = None,
    **kwargs: Any,
) -> dict[str, Any] | EvaluationResult:
    """Evaluate one or more metrics across image(s), references, or prompts.

    Supports paths, PIL.Image instances, and torch.Tensors directly
    in memory without requiring disk I/O.

    Args:
        metrics: Single metric name or collection of metric names.
        image: Evaluated image (path, PIL Image, or torch.Tensor).
        reference: Reference image or folder for pairwise/distribution metrics.
        prompt: Text prompt string, prompt file path, or sequence of prompts
            required for 'clip', 'pickscore', 'hpsv2', 'image_reward',
            'vqascore', and 'directional_clip'.
        source_prompt: Source prompt string required for 'directional_clip'.
        device: Computing device ('cuda', 'cpu', 'mps', or None for auto).
        detailed: If True, return EvaluationResult object; if False, return
            plain dict.
        allow_download: If True, allow downloading uncached model weights.
        download_callback: Optional callback(asset, msg) on download.
        **kwargs: Additional parameters passed to specific predictors.

    Returns:
        dict[str, Any] | EvaluationResult: Mapping of metric names to scores,
        or an EvaluationResult instance when detailed=True.

    Raises:
        ValueError: If metric names or required parameters are invalid.
        FileNotFoundError: If any specified file or folder path does not exist.
    """
    start_time = time.perf_counter()
    if isinstance(metrics, str):
        metric_list = [metrics.lower().strip()]
    elif isinstance(metrics, Sequence):
        metric_list = [str(m).lower().strip() for m in metrics]
    else:
        raise ValueError(
            f"Invalid metrics argument type: {type(metrics).__name__}. "
            "Expected str or Sequence[str]."
        )

    for m in metric_list:
        if m not in SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric: '{m}'. "
                f"Supported metrics: {sorted(SUPPORTED_METRICS)}"
            )

    selected = set(metric_list)

    # Validate prompt-based metrics
    selected_prompt = selected & PROMPT_METRICS
    if selected_prompt:
        has_prompt = False
        if isinstance(prompt, str) and prompt.strip():
            has_prompt = True
        elif isinstance(prompt, (list, tuple)) or (
            isinstance(prompt, Sequence)
            and not isinstance(prompt, (str, bytes))
        ):
            has_prompt = len(prompt) > 0 and all(
                isinstance(p, str) and p.strip() for p in prompt
            )
        if not has_prompt:
            if len(selected_prompt) == 1:
                single_metric = next(iter(selected_prompt))
                raise ValueError(
                    f"prompt is required when '{single_metric}' "
                    "metric is selected."
                )
            raise ValueError(
                "prompt is required when prompt-based metrics are selected."
            )
    elif prompt is not None:
        raise ValueError(
            "prompt was provided but no prompt-based metric was selected."
        )

    # Validate reference-based metrics
    selected_reference = selected & REFERENCE_METRICS
    if selected_reference:
        if reference is None:
            if len(selected_reference) == 1:
                single_metric = next(iter(selected_reference))
                raise ValueError(
                    f"reference is required when '{single_metric}' "
                    "metric is selected."
                )
            raise ValueError(
                "reference is required when reference-based "
                "metrics are selected."
            )
    elif reference is not None:
        raise ValueError(
            "reference was provided but no reference-based "
            "metric was selected."
        )

    # Validate source_prompt for directional_clip
    selected_source_prompt = selected & SOURCE_PROMPT_METRICS
    if selected_source_prompt:
        if source_prompt is None or not (
            isinstance(source_prompt, str) and source_prompt.strip()
        ):
            if len(selected_source_prompt) == 1:
                single_metric = next(iter(selected_source_prompt))
                raise ValueError(
                    f"source_prompt is required when '{single_metric}' "
                    "metric is selected."
                )
            raise ValueError(
                "source_prompt is required when source-prompt-based "
                "metrics are selected."
            )
    elif source_prompt is not None:
        raise ValueError(
            "source_prompt was provided but 'directional_clip' "
            "metric was not selected."
        )

    # Determine if inputs are folders
    is_image_folder = (
        isinstance(image, (str, os.PathLike)) and os.path.isdir(str(image))
    )
    is_ref_folder = (
        isinstance(reference, (str, os.PathLike))
        and os.path.isdir(str(reference))
    )

    if "directional_clip" in selected and (is_image_folder or is_ref_folder):
        raise ValueError(
            "image and reference must be single images for "
            "'directional_clip' metric."
        )

    # Validate dataset metrics
    for d_metric in DATASET_METRICS:
        if d_metric in selected:
            if not is_image_folder:
                raise ValueError(
                    "image must be an existing directory when "
                    f"'{d_metric}' metric is selected, "
                    f"got {type(image).__name__}: '{image}'"
                )
            if not is_ref_folder:
                raise ValueError(
                    "reference must be an existing directory when "
                    f"'{d_metric}' metric is selected, "
                    f"got {type(reference).__name__}: '{reference}'"
                )

    # Validate pairwise folder pairing
    selected_pairwise = selected & PAIRWISE_METRICS
    if selected_pairwise and (is_image_folder or is_ref_folder):
        if is_image_folder != is_ref_folder:
            raise ValueError(
                "image and reference must both be single images "
                "or both be directories."
            )

    results: dict[str, Any] = {}

    single_prompt: str = ""
    folder_prompt: str | Sequence[str] = ""
    if prompt is not None:
        folder_prompt = prompt
        if isinstance(prompt, str):
            single_prompt = prompt
        elif isinstance(prompt, (list, tuple)) or (
            isinstance(prompt, Sequence)
            and not isinstance(prompt, (str, bytes))
        ):
            single_prompt = prompt[0] if len(prompt) > 0 else ""
        else:
            single_prompt = str(prompt)

    # 1. Aesthetic
    if "aesthetic" in selected:
        from image_evaluator.laion_ai_aesthetic_predictor import (
            LaionAIAestheticPredictor,
        )

        model_name = kwargs.get("aesthetic_model_name", "vit_l_14")
        results["aesthetic"] = _run_with_predictor(
            lambda: LaionAIAestheticPredictor(
                model_name=model_name, device=device
            ),
            lambda pred: (
                pred.evaluate_folder_aesthetic_score(str(image))
                if is_image_folder
                else pred.evaluate_aesthetic_score(image)
            ),
        )

    # 2. CLIP Score
    if "clip" in selected:
        from image_evaluator.clip_score_predictor import ClipScorePredictor

        clip_model = kwargs.get("clip_model", "openai/clip-vit-base-patch32")
        results["clip"] = _run_with_predictor(
            lambda: ClipScorePredictor(clip_model=clip_model, device=device),
            lambda pred: pred.evaluate_clip_score(
                image, folder_prompt if is_image_folder else single_prompt
            ),
        )

    # 3. ArcFace Distance
    if "arcface" in selected:
        from image_evaluator.arcface_dist_predictor import ArcFaceDistPredictor

        results["arcface"] = _run_with_predictor(
            lambda: ArcFaceDistPredictor(device=device),
            lambda pred: (
                pred.evaluate_folder_arcface_distance(
                    str(reference), str(image)
                )
                if is_image_folder
                else pred.evaluate_arcface_distance(reference, image)
            ),
        )

    # 4. LPIPS Distance
    if "lpips" in selected:
        from image_evaluator.lpips_predictor import LPIPSPredictor

        net = kwargs.get("lpips_net", "alex")
        results["lpips"] = _run_with_predictor(
            lambda: LPIPSPredictor(net=net, device=device),
            lambda pred: (
                pred.evaluate_folder_lpips(str(reference), str(image))
                if is_image_folder
                else pred.evaluate_lpips(reference, image)
            ),
        )

    # 5. SSIM
    if "ssim" in selected:
        from image_evaluator.ssim_predictor import SSIMPredictor

        window_size = kwargs.get("ssim_window_size", 11)
        sigma = kwargs.get("ssim_sigma", 1.5)
        results["ssim"] = _run_with_predictor(
            lambda: SSIMPredictor(
                window_size=window_size, sigma=sigma, device=device
            ),
            lambda pred: (
                pred.evaluate_folder_ssim(str(reference), str(image))
                if is_image_folder
                else pred.evaluate_ssim(reference, image)
            ),
        )

    # 6. PSNR
    if "psnr" in selected:
        from image_evaluator.psnr_predictor import PSNRPredictor

        results["psnr"] = _run_with_predictor(
            lambda: PSNRPredictor(device=device),
            lambda pred: (
                pred.evaluate_folder_psnr(str(reference), str(image))
                if is_image_folder
                else pred.evaluate_psnr(reference, image)
            ),
        )

    # 7. FID
    if "fid" in selected:
        from image_evaluator.fid_predictor import FIDPredictor

        fid_device = device if device is not None else "cpu"
        results["fid"] = _run_with_predictor(
            lambda: FIDPredictor(device=fid_device),
            lambda pred: pred.evaluate_folder_fid(str(reference), str(image)),
        )

    # 8. KID
    if "kid" in selected:
        from image_evaluator.kid_predictor import KIDPredictor

        kid_device = device if device is not None else "cpu"
        results["kid"] = _run_with_predictor(
            lambda: KIDPredictor(device=kid_device),
            lambda pred: pred.evaluate_folder_kid(str(reference), str(image)),
        )

    # 9. PickScore
    if "pickscore" in selected:
        from image_evaluator.pickscore_predictor import PickScorePredictor

        model_name = kwargs.get("pickscore_model", None)
        processor_name = kwargs.get("pickscore_processor", None)
        model_revision = kwargs.get("pickscore_model_revision", None)
        processor_revision = kwargs.get("pickscore_processor_revision", None)
        model_download_bytes = kwargs.get(
            "pickscore_model_download_bytes", None
        )
        processor_download_bytes = kwargs.get(
            "pickscore_processor_download_bytes", None
        )
        model_required_files = kwargs.get(
            "pickscore_model_required_files", None
        )
        processor_required_files = kwargs.get(
            "pickscore_processor_required_files", None
        )

        results["pickscore"] = _run_with_predictor(
            lambda: PickScorePredictor(
                device=device,
                model_name=model_name,
                processor_name=processor_name,
                allow_download=allow_download,
                download_callback=download_callback,
                model_revision=model_revision,
                processor_revision=processor_revision,
                model_download_bytes=model_download_bytes,
                processor_download_bytes=processor_download_bytes,
                model_required_files=model_required_files,
                processor_required_files=processor_required_files,
            ),
            lambda pred: (
                pred.evaluate_folder(str(image), folder_prompt).mean_score
                if is_image_folder
                else pred.evaluate(image, single_prompt)
            ),
        )

    # 10. Directional CLIP
    if "directional_clip" in selected:
        from image_evaluator.directional_clip_predictor import (
            DirectionalClipPredictor,
        )

        clip_model = kwargs.get("clip_model", "openai/clip-vit-base-patch32")
        results["directional_clip"] = _run_with_predictor(
            lambda: DirectionalClipPredictor(
                clip_model=clip_model,
                device=device,
                allow_download=allow_download,
                download_callback=download_callback,
            ),
            lambda pred: pred.evaluate_directional_clip(
                image_src=reference,
                image_edit=image,
                prompt_src=str(source_prompt),
                prompt_target=single_prompt,
            ),
        )

    # 11. CLIP-I
    if "clip_i" in selected:
        from image_evaluator.clip_i_predictor import ClipIPredictor

        results["clip_i"] = _run_with_predictor(
            lambda: ClipIPredictor(
                device=device,
                allow_download=allow_download,
                download_callback=download_callback,
            ),
            lambda pred: (
                pred.evaluate_folder_clip_i(str(reference), str(image))
                if is_image_folder
                else pred.evaluate_clip_i(reference, image)
            ),
        )

    # 12. DINO Similarity
    if "dino_similarity" in selected:
        from image_evaluator.dino_similarity_predictor import (
            DinoSimilarityPredictor,
        )

        results["dino_similarity"] = _run_with_predictor(
            lambda: DinoSimilarityPredictor(
                device=device,
                allow_download=allow_download,
                download_callback=download_callback,
            ),
            lambda pred: (
                pred.evaluate_folder_dino_similarity(
                    str(reference), str(image)
                )
                if is_image_folder
                else pred.evaluate_dino_similarity(reference, image)
            ),
        )

    # 13. HPS v2.1
    if "hpsv2" in selected:
        from image_evaluator.hpsv2_predictor import Hpsv2Predictor

        results["hpsv2"] = _run_with_predictor(
            lambda: Hpsv2Predictor(
                device=device,
                allow_download=allow_download,
                download_callback=download_callback,
            ),
            lambda pred: (
                pred.evaluate_folder_hpsv2(str(image), folder_prompt)
                if is_image_folder
                else pred.evaluate_hpsv2(image, single_prompt)
            ),
        )

    # 14. ImageReward
    if "image_reward" in selected:
        from image_evaluator.image_reward_predictor import (
            ImageRewardPredictor,
        )

        results["image_reward"] = _run_with_predictor(
            lambda: ImageRewardPredictor(
                device=device,
                allow_download=allow_download,
                download_callback=download_callback,
            ),
            lambda pred: (
                pred.evaluate_folder_image_reward(str(image), folder_prompt)
                if is_image_folder
                else pred.evaluate_image_reward(image, single_prompt)
            ),
        )

    # 15. VQAScore
    if "vqascore" in selected:
        from image_evaluator.vqascore_predictor import (
            VQAScorePredictor,
        )

        results["vqascore"] = _run_with_predictor(
            lambda: VQAScorePredictor(
                device=device,
                allow_download=allow_download,
                download_callback=download_callback,
            ),
            lambda pred: (
                pred.evaluate_folder_vqascore(str(image), folder_prompt)
                if is_image_folder
                else pred.evaluate_vqascore(image, single_prompt)
            ),
        )

    if detailed:
        duration_seconds = max(0.0, time.perf_counter() - start_time)
        specs = {m: get_metric(m) for m in results}
        inputs_summary: dict[str, Any] = {
            "image_type": type(image).__name__,
            "reference_type": (
                type(reference).__name__ if reference is not None else None
            ),
            "prompt": prompt,
            "source_prompt": source_prompt,
            "device": str(device) if device is not None else None,
            "allow_download": allow_download,
        }
        return EvaluationResult(
            scores=results,
            specs=specs,
            inputs=inputs_summary,
            duration_seconds=duration_seconds,
        )

    return results


def evaluate_detailed(
    metrics: str | Sequence[str],
    image: Any,
    reference: Any = None,
    prompt: str | Sequence[str] | None = None,
    source_prompt: str | None = None,
    device: str | torch.device | None = None,
    allow_download: bool = False,
    download_callback: Any = None,
    **kwargs: Any,
) -> EvaluationResult:
    """Evaluate metric(s) and return an EvaluationResult instance.

    Provides identical calculation capabilities to evaluate(), but packages
    the outcome in an EvaluationResult containing scores, Registry
    specifications, input metadata, and RFC 8259 JSON serialization.
    """
    res = evaluate(
        metrics=metrics,
        image=image,
        reference=reference,
        prompt=prompt,
        source_prompt=source_prompt,
        device=device,
        detailed=True,
        allow_download=allow_download,
        download_callback=download_callback,
        **kwargs,
    )
    assert isinstance(res, EvaluationResult)
    return res
