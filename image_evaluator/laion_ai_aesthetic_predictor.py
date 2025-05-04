# Taken from https://github.com/LAION-AI/aesthetic-predictor/blob/main/asthetics_predictor.ipynb

import os
import torch
import open_clip
import torch.nn as nn
from PIL import Image
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel


class LaionAIAestheticPredictor:
    def __init__(self, model_name="vit_l_14"):
        """Initialize the aesthetic predictor with a specified model."""
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.aes_model = self.get_aesthetic_model()

    def get_aesthetic_model(self):
        """Load the aesthetic model based on the model type defined in __init__."""
        home = expanduser("~")
        cache_folder = home + "/.cache/emb_reader"
        path_to_model = cache_folder + f"/sa_0_4_{self.model_name}_linear.pth"

        if not os.path.exists(path_to_model):
            os.makedirs(cache_folder, exist_ok=True)
            url_model = f"https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_{self.model_name}_linear.pth?raw=true"
            urlretrieve(url_model, path_to_model)

        if self.model_name == "vit_l_14":
            m = nn.Linear(768, 1)
        elif self.model_name == "vit_b_32":
            m = nn.Linear(512, 1)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        s = torch.load(path_to_model, map_location=self.device)
        m.load_state_dict(s)
        m.to(self.device)
        m.eval()
        return m

    def evaluate_aesthetic_score(self, image_path):
        """Evaluate the aesthetic score of a single image

        Args:
            image_path: Path to the image file

        Returns:
            float: Aesthetic score of the image
        """
        if self.aes_model is None:
            self.aes_model = self.get_aesthetic_model()

        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai'
        )
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                score = self.aes_model(image_features)

            return score[0][0].item()
        except Exception as e:
            print(f"Error evaluating image {image_path}: {e}")
            return None

    def evaluate_folder_aesthetic_score(self, folder_path):
        """Evaluate the average aesthetic score of all images in a folder

        Args:
            folder_path: Path to folder containing images

        Returns:
            float: Average aesthetic score of all images in the folder
        """
        if not os.path.isdir(folder_path):
            raise ValueError(f"{folder_path} is not a valid folder path")

        image_files = [
            f
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
        ]

        if not image_files:
            return None

        scores = []
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            score = self.evaluate_aesthetic_score(img_path)
            if score is not None:
                scores.append(score)

        return sum(scores) / len(scores) if scores else None
