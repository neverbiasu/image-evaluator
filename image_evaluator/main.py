import argparse
from laion_ai_aesthetic_predictor import LaionAIAestheticPredictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the aesthetic score of an image.")
    parser.add_argument("--image", type=str, help="Path to the image file")
    args = parser.parse_args()

    predictor = LaionAIAestheticPredictor()
    score = predictor.evaluate_aesthetic_score(args.image)
    print(f"Aesthetic Score: {score}")