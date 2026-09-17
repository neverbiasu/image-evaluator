import os
import time
from collections.abc import Sequence
from typing import Any

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


def evaluate(
    metrics: str | Sequence[str],
    image: Any,
    reference: Any = None,
    prompt: str | None = None,
    source_prompt: str | None = None,
    device: str | torch.device | None = None,
    detailed: bool = False,
    **kwargs: Any,
) -> dict[str, Any] | EvaluationResult:
    """Evaluate one or more metrics across image(s), references, or prompts.

    Supports paths, PIL.Image instances, and torch.Tensors directly
    in memory without requiring disk I/O.

    Args:
        metrics: Single metric name or collection of metric names.
        image: Evaluated image (path, PIL Image, or torch.Tensor).
        reference: Reference image or folder for pairwise/distribution metrics.
        prompt: Text prompt string required for 'clip', 'pickscore', and
            'directional_clip'.
        source_prompt: Source prompt string required for 'directional_clip'.
        device: Computing device ('cuda', 'cpu', 'mps', or None for auto).
        detailed: If True, return an EvaluationResult instance instead of a
            plain dict.
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
        if prompt is None or not (isinstance(prompt, str) and prompt.strip()):
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

    # 1. Aesthetic
    if "aesthetic" in selected:
        from image_evaluator.laion_ai_aesthetic_predictor import (
            LaionAIAestheticPredictor,
        )

        model_name = kwargs.get("aesthetic_model_name", "vit_l_14")
        pred_aesthetic = LaionAIAestheticPredictor(
            model_name=model_name, device=device
        )
        if is_image_folder:
            score = pred_aesthetic.evaluate_folder_aesthetic_score(
                str(image)
            )
        else:
            score = pred_aesthetic.evaluate_aesthetic_score(image)
        results["aesthetic"] = score

    # 2. CLIP Score
    if "clip" in selected:
        from image_evaluator.clip_score_predictor import ClipScorePredictor

        clip_model = kwargs.get("clip_model", "openai/clip-vit-base-patch32")
        pred_clip = ClipScorePredictor(clip_model=clip_model, device=device)
        results["clip"] = pred_clip.evaluate_clip_score(image, prompt)

    # 3. ArcFace Distance
    if "arcface" in selected:
        from image_evaluator.arcface_dist_predictor import ArcFaceDistPredictor

        pred_arcface = ArcFaceDistPredictor(device=device)
        if is_image_folder:
            results["arcface"] = pred_arcface.evaluate_folder_arcface_distance(
                str(reference), str(image)
            )
        else:
            results["arcface"] = pred_arcface.evaluate_arcface_distance(
                reference, image
            )

    # 4. LPIPS Distance
    if "lpips" in selected:
        from image_evaluator.lpips_predictor import LPIPSPredictor

        net = kwargs.get("lpips_net", "alex")
        pred_lpips = LPIPSPredictor(net=net, device=device)
        if is_image_folder:
            results["lpips"] = pred_lpips.evaluate_folder_lpips(
                str(reference), str(image)
            )
        else:
            results["lpips"] = pred_lpips.evaluate_lpips(reference, image)

    # 5. SSIM
    if "ssim" in selected:
        from image_evaluator.ssim_predictor import SSIMPredictor

        window_size = kwargs.get("ssim_window_size", 11)
        sigma = kwargs.get("ssim_sigma", 1.5)
        pred_ssim = SSIMPredictor(
            window_size=window_size, sigma=sigma, device=device
        )
        if is_image_folder:
            results["ssim"] = pred_ssim.evaluate_folder_ssim(
                str(reference), str(image)
            )
        else:
            results["ssim"] = pred_ssim.evaluate_ssim(reference, image)

    # 6. PSNR
    if "psnr" in selected:
        from image_evaluator.psnr_predictor import PSNRPredictor

        pred_psnr = PSNRPredictor(device=device)
        if is_image_folder:
            results["psnr"] = pred_psnr.evaluate_folder_psnr(
                str(reference), str(image)
            )
        else:
            results["psnr"] = pred_psnr.evaluate_psnr(reference, image)

    # 7. FID
    if "fid" in selected:
        from image_evaluator.fid_predictor import FIDPredictor

        fid_device = device if device is not None else "cpu"
        pred_fid = FIDPredictor(device=fid_device)
        results["fid"] = pred_fid.evaluate_folder_fid(
            str(reference), str(image)
        )

    # 8. KID
    if "kid" in selected:
        from image_evaluator.kid_predictor import KIDPredictor

        kid_device = device if device is not None else "cpu"
        pred_kid = KIDPredictor(device=kid_device)
        results["kid"] = pred_kid.evaluate_folder_kid(
            str(reference), str(image)
        )

    # 9. PickScore
    if "pickscore" in selected:
        from image_evaluator.pickscore_predictor import PickScorePredictor

        pred_pick = PickScorePredictor(device=device)
        if is_image_folder:
            pick_res = pred_pick.evaluate_folder(str(image), prompt)
            results["pickscore"] = pick_res.mean_score
        else:
            results["pickscore"] = pred_pick.evaluate(image, prompt)

    # 10. Directional CLIP
    if "directional_clip" in selected:
        from image_evaluator.directional_clip_predictor import (
            DirectionalClipPredictor,
        )

        clip_model = kwargs.get("clip_model", "openai/clip-vit-base-patch32")
        pred_dir_clip = DirectionalClipPredictor(
            clip_model=clip_model, device=device
        )
        results["directional_clip"] = (
            pred_dir_clip.evaluate_directional_clip(
                image_src=reference,
                image_edit=image,
                prompt_src=source_prompt,  # type: ignore[arg-type]
                prompt_target=prompt,  # type: ignore[arg-type]
            )
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
    prompt: str | None = None,
    source_prompt: str | None = None,
    device: str | torch.device | None = None,
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
        **kwargs,
    )
    assert isinstance(res, EvaluationResult)
    return res
