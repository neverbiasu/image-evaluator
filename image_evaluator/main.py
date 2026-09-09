import argparse
import os


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Evaluate images using selected metrics."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["aesthetic", "clip", "arcface", "lpips", "ssim", "psnr"],
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
        help="Path to the prompt file or text prompt (required for 'clip')",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help=(
            "Path to reference image or folder "
            "(required for 'arcface', 'lpips', 'ssim', 'psnr')"
        ),
    )
    parsed_args = parser.parse_args(args)

    selected_metrics = set(parsed_args.metrics)

    # Validate dependent and prohibited options
    if "clip" in selected_metrics:
        if parsed_args.prompt is None or not parsed_args.prompt.strip():
            parser.error(
                "--prompt is required when 'clip' metric is selected."
            )
    elif parsed_args.prompt is not None:
        parser.error(
            "--prompt was provided but 'clip' metric was not selected."
        )

    pairwise_metrics = {"arcface", "lpips", "ssim", "psnr"}
    selected_pairwise = selected_metrics & pairwise_metrics
    if selected_pairwise:
        if (
            parsed_args.reference is None
            or not parsed_args.reference.strip()
        ):
            if "arcface" in selected_pairwise and len(selected_pairwise) == 1:
                parser.error(
                    "--reference is required when "
                    "'arcface' metric is selected."
                )
            elif "lpips" in selected_pairwise and len(selected_pairwise) == 1:
                parser.error(
                    "--reference is required when 'lpips' metric is selected."
                )
            elif "ssim" in selected_pairwise and len(selected_pairwise) == 1:
                parser.error(
                    "--reference is required when 'ssim' metric is selected."
                )
            elif "psnr" in selected_pairwise and len(selected_pairwise) == 1:
                parser.error(
                    "--reference is required when 'psnr' metric is selected."
                )
            else:
                parser.error(
                    "--reference is required when reference-based "
                    "metrics are selected."
                )
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
        print(f"LAION AI Aesthetic Score: {aesthetic_score}")

    # CLIP Score Evaluation
    if "clip" in selected_metrics:
        from image_evaluator.clip_score_predictor import ClipScorePredictor

        clip_predictor = ClipScorePredictor()
        clip_score = clip_predictor.evaluate_clip_score(
            parsed_args.image, parsed_args.prompt
        )
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
            arcface_distance = arcface_predictor.evaluate_arcface_distance(
                parsed_args.reference, parsed_args.image
            )
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
        print(f"PSNR: {psnr_score}")


if __name__ == "__main__":
    main()
