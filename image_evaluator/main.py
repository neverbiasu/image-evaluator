import argparse
import contextlib
import json
import math
import os
import sys
from typing import Any

import numpy as np


def _normalize_json_values(obj: Any) -> Any:
    """Recursively convert non-finite float values (NaN, Inf, -Inf) to None.

    Ensures strict RFC 8259 JSON compliance when serialized with
    allow_nan=False. Supports Python floats and NumPy floating scalars.
    """
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        if not math.isfinite(val):
            return None
        return val
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, np.ndarray) and obj.ndim == 0:
        return _normalize_json_values(obj.item())
    if isinstance(obj, dict):
        return {k: _normalize_json_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_json_values(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_normalize_json_values(v) for v in obj)
    return obj


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Evaluate images using selected metrics."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=[
            "aesthetic",
            "clip",
            "arcface",
            "lpips",
            "ssim",
            "psnr",
            "fid",
            "kid",
            "pickscore",
        ],
        required=True,
        help="Metrics to evaluate",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image file or folder",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=(
            "Path to the prompt file or text prompt "
            "(required for 'clip' and 'pickscore')"
        ),
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help=(
            "Path to reference image or folder "
            "(required for 'arcface', 'lpips', 'ssim', 'psnr', 'fid', 'kid')"
        ),
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format: 'text' (default) or 'json'",
    )
    parsed_args = parser.parse_args(args)

    selected_metrics = set(parsed_args.metrics)

    # Validate dependent and prohibited options
    prompt_metrics = {"clip", "pickscore"}
    selected_prompt = selected_metrics & prompt_metrics
    if selected_prompt:
        if parsed_args.prompt is None or not parsed_args.prompt.strip():
            if len(selected_prompt) == 1:
                single_metric = next(iter(selected_prompt))
                parser.error(
                    f"--prompt is required when '{single_metric}' "
                    "metric is selected."
                )
            parser.error(
                "--prompt is required when prompt-based metrics are selected."
            )
    elif parsed_args.prompt is not None:
        parser.error(
            "--prompt was provided but no prompt-based metric was selected."
        )

    pairwise_metrics = {"arcface", "lpips", "ssim", "psnr"}
    reference_metrics = pairwise_metrics | {"fid", "kid"}
    selected_reference = selected_metrics & reference_metrics
    selected_pairwise = selected_metrics & pairwise_metrics
    if selected_reference:
        if (
            parsed_args.reference is None
            or not parsed_args.reference.strip()
        ):
            if len(selected_reference) == 1:
                single_metric = next(iter(selected_reference))
                parser.error(
                    f"--reference is required when '{single_metric}' "
                    "metric is selected."
                )
            parser.error(
                "--reference is required when reference-based "
                "metrics are selected."
            )

        for dataset_metric in ("fid", "kid"):
            if dataset_metric in selected_metrics:
                if not os.path.exists(parsed_args.image):
                    parser.error(
                        f"--image path does not exist: '{parsed_args.image}'"
                    )
                if not os.path.isdir(parsed_args.image):
                    parser.error(
                        "--image must be a directory when "
                        f"'{dataset_metric}' metric is selected, "
                        f"got file: '{parsed_args.image}'"
                    )
                if not os.path.exists(parsed_args.reference):
                    parser.error(
                        "--reference path does not exist: "
                        f"'{parsed_args.reference}'"
                    )
                if not os.path.isdir(parsed_args.reference):
                    parser.error(
                        "--reference must be a directory when "
                        f"'{dataset_metric}' metric is selected, "
                        f"got file: '{parsed_args.reference}'"
                    )

        if selected_pairwise:
            is_image_folder = os.path.isdir(parsed_args.image)
            is_ref_folder = os.path.isdir(parsed_args.reference)
            if is_image_folder != is_ref_folder:
                parser.error(
                    "--image and --reference must both be files or both "
                    "be directories."
                )
    elif parsed_args.reference is not None:
        parser.error(
            "--reference was provided but no reference-based metric "
            "was selected."
        )

    is_folder = os.path.isdir(parsed_args.image)
    results = {
        "status": "success",
        "metrics": {},
    }

    def _execute_metrics():
        # LAION AI Aesthetic Score
        if "aesthetic" in selected_metrics:
            from image_evaluator.laion_ai_aesthetic_predictor import (
                LaionAIAestheticPredictor,
            )

            aesthetic_predictor = LaionAIAestheticPredictor()
            if is_folder:
                aesthetic_score = (
                    aesthetic_predictor.evaluate_folder_aesthetic_score(
                        parsed_args.image
                    )
                )
            else:
                aesthetic_score = aesthetic_predictor.evaluate_aesthetic_score(
                    parsed_args.image
                )
            results["metrics"]["aesthetic"] = aesthetic_score
            if parsed_args.format == "text":
                print(f"LAION AI Aesthetic Score: {aesthetic_score}")

        # CLIP Score Evaluation
        if "clip" in selected_metrics:
            from image_evaluator.clip_score_predictor import (
                ClipScorePredictor,
            )

            clip_predictor = ClipScorePredictor()
            clip_score = clip_predictor.evaluate_clip_score(
                parsed_args.image, parsed_args.prompt
            )
            results["metrics"]["clip"] = clip_score
            if parsed_args.format == "text":
                print(f"CLIP Score: {clip_score}")

        # ArcFace Distance Evaluation
        if "arcface" in selected_metrics:
            from image_evaluator.arcface_dist_predictor import (
                ArcFaceDistPredictor,
            )

            arcface_predictor = ArcFaceDistPredictor()
            if is_folder:
                arcface_distance = (
                    arcface_predictor.evaluate_folder_arcface_distance(
                        parsed_args.reference, parsed_args.image
                    )
                )
            else:
                arcface_distance = (
                    arcface_predictor.evaluate_arcface_distance(
                        parsed_args.reference, parsed_args.image
                    )
                )
            results["metrics"]["arcface"] = arcface_distance
            if parsed_args.format == "text":
                print(f"ArcFace Distance: {arcface_distance}")

        # LPIPS Distance Evaluation
        if "lpips" in selected_metrics:
            from image_evaluator.lpips_predictor import LPIPSPredictor

            lpips_predictor = LPIPSPredictor()
            if is_folder:
                lpips_distance = lpips_predictor.evaluate_folder_lpips(
                    parsed_args.reference, parsed_args.image
                )
            else:
                lpips_distance = lpips_predictor.evaluate_lpips(
                    parsed_args.reference, parsed_args.image
                )
            results["metrics"]["lpips"] = lpips_distance
            if parsed_args.format == "text":
                print(f"LPIPS Distance: {lpips_distance}")

        # SSIM Similarity Evaluation
        if "ssim" in selected_metrics:
            from image_evaluator.ssim_predictor import SSIMPredictor

            ssim_predictor = SSIMPredictor()
            if is_folder:
                ssim_score = ssim_predictor.evaluate_folder_ssim(
                    parsed_args.reference, parsed_args.image
                )
            else:
                ssim_score = ssim_predictor.evaluate_ssim(
                    parsed_args.reference, parsed_args.image
                )
            results["metrics"]["ssim"] = ssim_score
            if parsed_args.format == "text":
                print(f"SSIM: {ssim_score}")

        # PSNR Evaluation
        if "psnr" in selected_metrics:
            from image_evaluator.psnr_predictor import PSNRPredictor

            psnr_predictor = PSNRPredictor()
            if is_folder:
                psnr_score = psnr_predictor.evaluate_folder_psnr(
                    parsed_args.reference, parsed_args.image
                )
            else:
                psnr_score = psnr_predictor.evaluate_psnr(
                    parsed_args.reference, parsed_args.image
                )
            if math.isinf(psnr_score):
                results["metrics"]["psnr"] = None
                results["metrics"]["psnr_raw"] = (
                    "inf" if psnr_score > 0 else "-inf"
                )
            elif math.isnan(psnr_score):
                results["metrics"]["psnr"] = None
                results["metrics"]["psnr_raw"] = "nan"
            else:
                results["metrics"]["psnr"] = psnr_score
                results["metrics"]["psnr_raw"] = str(psnr_score)
            if parsed_args.format == "text":
                print(f"PSNR: {psnr_score}")

        # FID Evaluation
        if "fid" in selected_metrics:
            from image_evaluator.fid_predictor import FIDPredictor

            fid_predictor = FIDPredictor()
            fid_result = fid_predictor.evaluate_folder_fid(
                parsed_args.reference, parsed_args.image
            )
            results["metrics"]["fid"] = fid_result
            if parsed_args.format == "text":
                print(
                    f"FID: {fid_result['fid']} "
                    f"(backend={fid_result['backend']}, "
                    f"version={fid_result['version']}, "
                    f"mode={fid_result['mode']}, "
                    f"model={fid_result['model']}, "
                    f"device={fid_result['device']}, "
                    f"Nref={fid_result['Nref']}, "
                    f"Ngen={fid_result['Ngen']})"
                )

        # KID Evaluation
        if "kid" in selected_metrics:
            from image_evaluator.kid_predictor import KIDPredictor

            kid_predictor = KIDPredictor()
            kid_result = kid_predictor.evaluate_folder_kid(
                parsed_args.reference, parsed_args.image
            )
            results["metrics"]["kid"] = kid_result
            if parsed_args.format == "text":
                print(
                    f"KID: {kid_result['kid']} "
                    f"(backend={kid_result['backend']}, "
                    f"version={kid_result['version']}, "
                    f"mode={kid_result['mode']}, "
                    f"model={kid_result['model']}, "
                    f"device={kid_result['device']}, "
                    f"num_subsets={kid_result['num_subsets']}, "
                    f"max_subset_size={kid_result['max_subset_size']}, "
                    f"seed={kid_result['seed']}, "
                    f"Nref={kid_result['Nref']}, "
                    f"Ngen={kid_result['Ngen']})"
                )

        # PickScore Evaluation
        if "pickscore" in selected_metrics:
            from image_evaluator.pickscore_predictor import (
                PickScorePredictor,
            )

            pickscore_predictor = PickScorePredictor()
            if is_folder:
                pickscore_res = pickscore_predictor.evaluate_folder(
                    parsed_args.image, parsed_args.prompt
                )
                pickscore_score = pickscore_res.mean_score
            else:
                pickscore_score = pickscore_predictor.evaluate(
                    parsed_args.image, parsed_args.prompt
                )
            results["metrics"]["pickscore"] = pickscore_score
            if parsed_args.format == "text":
                print(f"PickScore: {pickscore_score}")

    if parsed_args.format == "json":
        with contextlib.redirect_stdout(sys.stderr):
            _execute_metrics()
        clean_results = _normalize_json_values(results)
        print(json.dumps(clean_results, indent=2, allow_nan=False))
        return clean_results

    _execute_metrics()
    return results


if __name__ == "__main__":
    main()
