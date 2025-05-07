import argparse
import os
from image_evaluator.laion_ai_aesthetic_predictor import LaionAIAestheticPredictor
from image_evaluator.clip_score_predictor import ClipScorePredictor
from image_evaluator.arcface_dist_predictor import ArcFaceDistPredictor


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the aesthetic score of an image."
    )
    parser.add_argument("--image", type=str, help="Path to the image file or folder")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Path to the prompt file or text prompt",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to the reference image for ArcFace distance",
    )
    args = parser.parse_args()

    # Check if it's a file or folder
    is_folder = os.path.isdir(args.image) if args.image else False

    # LAION AI Aesthetic Score
    laion_ai_aesthetic_predictor = LaionAIAestheticPredictor()
    if is_folder:
        laion_ai_aesthetic_score = (
            laion_ai_aesthetic_predictor.evaluate_folder_aesthetic_score(args.image)
        )
    else:
        laion_ai_aesthetic_score = (
            laion_ai_aesthetic_predictor.evaluate_aesthetic_score(args.image)
        )

    # CLIP Score Evaluation
    clip_score_predictor = ClipScorePredictor()
    clip_score = clip_score_predictor.evaluate_clip_score(args.image, args.prompt)

    # ArcFace Distance Evaluation
    arcface_distance_predictor = ArcFaceDistPredictor()
    if is_folder and os.path.isdir(args.reference):
        arcface_distance = arcface_distance_predictor.evaluate_folder_arcface_distance(
            args.reference, args.image
        )
    elif args.reference:
        arcface_distance = arcface_distance_predictor.evaluate_arcface_distance(
            args.reference, args.image
        )
    else:
        arcface_distance = None

    print(f"LAION AI Aesthetic Score: {laion_ai_aesthetic_score}")
    print(f"CLIP Score: {clip_score}")
    if arcface_distance is not None:
        print(f"ArcFace Distance: {arcface_distance}")
    else:
        print("ArcFace Distance: Not evaluated (reference image required)")


if __name__ == "__main__":
    main()
