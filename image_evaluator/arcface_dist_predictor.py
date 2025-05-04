import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from insightface.app import FaceAnalysis
import numpy as np


class ArcFaceDistPredictor:
    def __init__(self, model_name="buffalo_l", device=None):
        """Initialize ArcFace distance predictor"""
        if device is None:
            ctx_id = 0 if torch.cuda.is_available() else -1
        else:
            ctx_id = 0 if device == 'cuda' else -1

        # Initialize ArcFace model
        self.app = FaceAnalysis(model_name)
        self.app.prepare(ctx_id=ctx_id)

        # Image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def get_face_embedding(self, image_path):
        """Get face embedding vector

        Args:
            image_path: Image file path

        Returns:
            numpy.ndarray: Face embedding vector or None (if no face is detected)
        """
        # Read image and convert to NumPy array
        img = Image.open(image_path).convert("RGB")
        img = np.array(img)

        # Get face embedding
        faces = self.app.get(img)
        if len(faces) == 0:
            return None
        return faces[0].embedding

    def evaluate_arcface_distance(self, reference_path, generated_path):
        """Evaluate ArcFace distance between two images

        Args:
            reference_path: Reference image file path
            generated_path: Generated image file path

        Returns:
            float: ArcFace distance score or None (if face not detected in either image)
        """
        ref_embedding = self.get_face_embedding(reference_path)
        gen_embedding = self.get_face_embedding(generated_path)
        if ref_embedding is None or gen_embedding is None:
            return None
        return (
            1
            - F.cosine_similarity(
                torch.tensor(ref_embedding), torch.tensor(gen_embedding), dim=0
            ).item()
        )

    def evaluate_folder_arcface_distance(self, reference_folder, generated_folder):
        """Evaluate average ArcFace distance between images in two folders

        Args:
            reference_folder: Folder path containing reference images
            generated_folder: Folder path containing generated images

        Returns:
            float: Average ArcFace distance
        """
        reference_images = sorted(os.listdir(reference_folder))
        generated_images = sorted(os.listdir(generated_folder))
        distances = []
        for ref_img, gen_img in zip(reference_images, generated_images):
            ref_path = os.path.join(reference_folder, ref_img)
            gen_path = os.path.join(generated_folder, gen_img)
            dist = self.evaluate_arcface_distance(ref_path, gen_path)
            if dist is not None:
                distances.append(dist)
        return np.mean(distances) if distances else None
