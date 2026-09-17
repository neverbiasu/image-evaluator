import argparse
import contextlib
import json
import math
import os
import sys
from typing import Any

import numpy as np
from PIL import Image

from image_evaluator.registry import (
    filter_metrics,
    get_metric,
    list_metrics,
)
from image_evaluator.specifications import MetricSpec


class CLIInputError(ValueError):
    """Raised when expected CLI runtime inputs are invalid or missing."""


def _verify_image_file(path: str) -> tuple[int, int]:
    try:
        with Image.open(path) as img:
            size = img.size
            img.verify()
            return size
    except Exception as exc:
        raise CLIInputError(
            f"Cannot identify or decode image file '{path}': {exc}"
        ) from exc


def _validate_runtime_inputs(
    parsed_args: argparse.Namespace, selected_metrics: set[str]
) -> None:
    if not os.path.exists(parsed_args.image):
        raise CLIInputError(
            f"Image path does not exist: '{parsed_args.image}'"
        )

    is_folder = os.path.isdir(parsed_args.image)
    pairwise_metrics = {"arcface", "lpips", "ssim", "psnr"}
    selected_pairwise = selected_metrics & pairwise_metrics

    if is_folder:
        if (
            selected_pairwise
            and (
                parsed_args.reference is None
                or not os.path.exists(parsed_args.reference)
            )
        ):
            raise CLIInputError(
                f"Reference path does not exist: '{parsed_args.reference}'"
            )
    else:
        if not os.path.isfile(parsed_args.image):
            raise CLIInputError(
                f"Image path is not a regular file: '{parsed_args.image}'"
            )
        img_size = _verify_image_file(parsed_args.image)

        if selected_pairwise:
            if (
                parsed_args.reference is None
                or not os.path.exists(parsed_args.reference)
            ):
                raise CLIInputError(
                    f"Reference path does not exist: '{parsed_args.reference}'"
                )
            if not os.path.isfile(parsed_args.reference):
                raise CLIInputError(
                    f"Reference path is not a regular file: "
                    f"'{parsed_args.reference}'"
                )
            ref_size = _verify_image_file(parsed_args.reference)

            sensitive_metrics = selected_pairwise & {"lpips", "ssim", "psnr"}
            if sensitive_metrics and img_size != ref_size:
                raise CLIInputError(
                    f"Image size mismatch: reference has size {ref_size}, "
                    f"generated image has size {img_size}. "
                    f"Metrics {sorted(sensitive_metrics)} "
                    f"require identical dimensions."
                )

        if "directional_clip" in selected_metrics:
            if (
                parsed_args.reference is None
                or not os.path.exists(parsed_args.reference)
            ):
                raise CLIInputError(
                    f"Reference path does not exist: '{parsed_args.reference}'"
                )
            if not os.path.isfile(parsed_args.reference):
                raise CLIInputError(
                    f"Reference path is not a regular file: "
                    f"'{parsed_args.reference}'"
                )
            _verify_image_file(parsed_args.reference)



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


def _spec_to_dict(spec: MetricSpec) -> dict[str, Any]:
    return {
        "id": spec.id,
        "display_name": spec.display_name,
        "score_direction": spec.score_direction,
        "tasks": list(spec.tasks),
        "objectives": list(spec.objectives),
        "inputs": {
            "required": list(spec.inputs.required),
            "optional": list(spec.inputs.optional),
        },
        "implementation": {
            "backend": spec.implementation.backend,
            "protocol": spec.implementation.protocol,
            "model": spec.implementation.model,
            "backend_version": spec.implementation.backend_version,
            "model_revision": spec.implementation.model_revision,
        },
        "aggregation": list(spec.aggregation),
        "dependencies": list(spec.dependencies),
        "citations": list(spec.citations),
        "docs_path": spec.docs_path,
    }


def _is_list_command(args: list[str]) -> bool:
    if not args:
        return False
    first = args[0].lower().strip()
    if first == "list":
        return True
    if (
        first == "metrics"
        and len(args) > 1
        and args[1].lower().strip() == "list"
    ):
        return True
    if "--list-metrics" in args or "--list" in args:
        return True
    return False


def _is_show_command(args: list[str]) -> bool:
    if not args:
        return False
    first = args[0].lower().strip()
    if first == "show":
        return True
    if (
        first == "metrics"
        and len(args) > 1
        and args[1].lower().strip() == "show"
    ):
        return True
    if "--show-metric" in args or "--show" in args:
        return True
    return False


def _extract_list_args(args: list[str]) -> list[str]:
    res = list(args)
    if res and res[0].lower().strip() == "list":
        return res[1:]
    if len(res) > 1 and res[0].lower().strip() == "metrics":
        if res[1].lower().strip() == "list":
            return res[2:]
    if "--list-metrics" in res:
        res.remove("--list-metrics")
    elif "--list" in res:
        res.remove("--list")
    return res


def _extract_show_args(args: list[str]) -> list[str]:
    res = list(args)
    if res and res[0].lower().strip() == "show":
        return res[1:]
    if len(res) > 1 and res[0].lower().strip() == "metrics":
        if res[1].lower().strip() == "show":
            return res[2:]
    if "--show-metric" in res:
        res.remove("--show-metric")
    elif "--show" in res:
        res.remove("--show")
    return res


def _handle_list(cmd_args: list[str]) -> list[dict[str, Any]]:
    parser = argparse.ArgumentParser(
        prog="image-evaluator list",
        description=(
            "List registered metrics with optional task and objective filters."
        ),
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default=None,
        help="Filter by task (e.g. text_to_image, image_editing).",
    )
    parser.add_argument(
        "--objective",
        "-o",
        type=str,
        default=None,
        help="Filter by objective (e.g. fidelity, alignment, quality).",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["text", "json"],
        default="text",
        help="Output format: 'text' (default) or 'json'.",
    )

    parsed = parser.parse_args(cmd_args)
    specs = filter_metrics(task=parsed.task, objective=parsed.objective)

    if parsed.format == "json":
        data = [_spec_to_dict(s) for s in specs]
        print(json.dumps(data, indent=2))
        return data

    if not specs:
        filters = []
        if parsed.task:
            filters.append(f"task='{parsed.task}'")
        if parsed.objective:
            filters.append(f"objective='{parsed.objective}'")
        filt_desc = f" matching {', '.join(filters)}" if filters else ""
        print(f"No metrics found{filt_desc}.")
        return []

    filt_info = ""
    if parsed.task and parsed.objective:
        filt_info = f", task='{parsed.task}', objective='{parsed.objective}'"
    elif parsed.task:
        filt_info = f", task='{parsed.task}'"
    elif parsed.objective:
        filt_info = f", objective='{parsed.objective}'"

    print(f"Available Evaluation Metrics ({len(specs)} total{filt_info}):")
    header = (
        f"{'ID':<18} {'Display Name':<32} {'Direction':<10} {'Inputs':<30}"
    )
    print(header)
    print("-" * len(header))
    for s in specs:
        dir_str = (
            "higher" if s.score_direction == "higher_is_better" else "lower"
        )
        reqs = ", ".join(s.inputs.required)
        print(f"{s.id:<18} {s.display_name:<32} {dir_str:<10} {reqs:<30}")

    return [_spec_to_dict(s) for s in specs]


def _handle_show(cmd_args: list[str]) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        prog="image-evaluator show",
        description="Show details of a registered metric specification.",
    )
    parser.add_argument("metric_id", type=str, help="Metric ID to inspect")
    parser.add_argument(
        "--format",
        "-f",
        choices=["text", "json"],
        default="text",
        help="Output format: 'text' (default) or 'json'.",
    )

    parsed = parser.parse_args(cmd_args)
    try:
        spec = get_metric(parsed.metric_id)
    except LookupError:
        available = sorted([s.id for s in list_metrics()])
        raise CLIInputError(
            f"Unknown metric '{parsed.metric_id}'. "
            f"Available metrics: {available}"
        )

    data = _spec_to_dict(spec)
    if parsed.format == "json":
        print(json.dumps(data, indent=2))
        return data

    print(f"Metric: {spec.id}")
    print(f"  Display Name:    {spec.display_name}")
    print(f"  Score Direction: {spec.score_direction}")
    print(f"  Tasks:           {', '.join(spec.tasks)}")
    print(f"  Objectives:      {', '.join(spec.objectives)}")
    print(f"  Required Inputs: {', '.join(spec.inputs.required)}")
    opt = ", ".join(spec.inputs.optional) if spec.inputs.optional else "(none)"
    print(f"  Optional Inputs: {opt}")
    print(f"  Backend:         {spec.implementation.backend}")
    print(f"  Protocol:        {spec.implementation.protocol}")
    model = spec.implementation.model or "(none)"
    print(f"  Model:           {model}")
    docs = spec.docs_path or "(none)"
    print(f"  Docs Path:       {docs}")
    return data


def main(args=None):
    raw_args = list(sys.argv[1:]) if args is None else list(args)

    if _is_list_command(raw_args):
        return _handle_list(_extract_list_args(raw_args))

    if _is_show_command(raw_args):
        return _handle_show(_extract_show_args(raw_args))

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
            "directional_clip",
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
    parser.add_argument(
        "--prompt-src",
        type=str,
        default=None,
        dest="prompt_src",
        help=(
            "Source prompt for 'directional_clip' metric. "
            "Describes the original image before editing."
        ),
    )
    parsed_args = parser.parse_args(args)

    selected_metrics = set(parsed_args.metrics)

    # Validate dependent and prohibited options
    prompt_metrics = {"clip", "pickscore", "directional_clip"}
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
    reference_metrics = pairwise_metrics | {"fid", "kid", "directional_clip"}
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

    if "directional_clip" in selected_metrics:
        if (
            parsed_args.prompt_src is None
            or not parsed_args.prompt_src.strip()
        ):
            parser.error(
                "--prompt-src is required when 'directional_clip' "
                "metric is selected (pass the source prompt)."
            )
    elif parsed_args.prompt_src is not None:
        parser.error(
            "--prompt-src was provided but 'directional_clip' "
            "metric was not selected."
        )


    _validate_runtime_inputs(parsed_args, selected_metrics)


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

        # Directional CLIP Evaluation
        if "directional_clip" in selected_metrics:
            from image_evaluator.directional_clip_predictor import (
                DirectionalClipPredictor,
            )

            dir_clip_predictor = DirectionalClipPredictor()
            dir_clip_score = dir_clip_predictor.evaluate_directional_clip(
                image_src=parsed_args.reference,
                image_edit=parsed_args.image,
                prompt_src=parsed_args.prompt_src,
                prompt_target=parsed_args.prompt,
            )
            results["metrics"]["directional_clip"] = dir_clip_score
            if parsed_args.format == "text":
                print(f"Directional CLIP: {dir_clip_score}")

    if parsed_args.format == "json":
        with contextlib.redirect_stdout(sys.stderr):
            _execute_metrics()
        clean_results = _normalize_json_values(results)
        print(json.dumps(clean_results, indent=2, allow_nan=False))
        return clean_results

    _execute_metrics()
    return results


def cli(args=None) -> int:
    try:
        main(args)
    except CLIInputError as exc:
        print(f"image-evaluator: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
