import os
from collections.abc import Sequence
from typing import Any

import torch

SUPPORTED_METRICS = {
    "aesthetic",
    "clip",
    "arcface",
    "lpips",
    "ssim",
    "psnr",
    "fid",
    "kid",
    "pickscore",
}

PROMPT_METRICS = {"clip", "pickscore"}
PAIRWISE_METRICS = {"arcface", "lpips", "ssim", "psnr"}
DATASET_METRICS = {"fid", "kid"}
REFERENCE_METRICS = PAIRWISE_METRICS | DATASET_METRICS


def evaluate(
    metrics: str | Sequence[str],
    image: Any,
    reference: Any = None,
    prompt: str | None = None,
    device: str | torch.device | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Evaluate one or more metrics across image(s), references, or prompts.

    Supports paths, PIL.Image instances, and torch.Tensors directly
    in memory without requiring disk I/O.

    Args:
        metrics: Single metric name or collection of metric names.
        image: Evaluated image (path, PIL Image, or torch.Tensor).
        reference: Reference image or folder for pairwise/distribution metrics.
        prompt: Text prompt string required for 'clip' and 'pickscore'.
        device: Computing device ('cuda', 'cpu', 'mps', or None for auto).
        **kwargs: Additional parameters passed to specific predictors.

    Returns:
        dict[str, Any]: Mapping of metric names to calculated scores.

    Raises:
        ValueError: If metric names or required parameters are invalid.
        FileNotFoundError: If any specified file or folder path does not exist.
    """
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

    # Determine if inputs are folders
    is_image_folder = (
        isinstance(image, (str, os.PathLike)) and os.path.isdir(str(image))
    )
    is_ref_folder = (
        isinstance(reference, (str, os.PathLike))
        and os.path.isdir(str(reference))
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

    return results
