import argparse
from laion_ai_aesthetic_predictor import LaionAIAestheticPredictor
from clip_score_predictor import ClipScorePredictor
from arcface_dist_predictor import ArcFaceDistPredictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the aesthetic score of an image."
    )
    parser.add_argument("--image", type=str, help="Path to the image file")
    parser.add_argument(
        "--prompt", type=str, default=None, help="Path to the prompt file"
    )
    args = parser.parse_args()

    laion_ai_aesthetic_predictor = LaionAIAestheticPredictor()
    laion_ai_aesthetic_score = laion_ai_aesthetic_predictor.evaluate_aesthetic_score(
        args.image
    )

    clip_score_predictor = ClipScorePredictor()
    clip_score = clip_score_predictor.evaluate_clip_score(args.image, args.prompt)

    arcface_distance_predictor = ArcFaceDistPredictor()
    arcface_distance = arcface_distance_predictor.evaluate_arcface_distance(
        args.image, args.image
    )

    print(f"LAION AI Aesthetic Score: {laion_ai_aesthetic_score}")
    print(f"CLIP Score: {clip_score}")
    print(f"ArcFace Distance: {arcface_distance}")
