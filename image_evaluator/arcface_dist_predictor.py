import numpy as np
import torch
import torch.nn.functional as F
from insightface.app import FaceAnalysis
from PIL import Image
from torchvision import transforms


class ArcFaceDistPredictor:
    def __init__(self, model_name="buffalo_l", device=None):
        """Initialize ArcFace distance predictor"""
        if device is None:
            ctx_id = 0 if torch.cuda.is_available() else -1
        else:
            ctx_id = 0 if device == "cuda" else -1

        # Initialize ArcFace model
        self.app = FaceAnalysis(model_name)
        self.app.prepare(ctx_id=ctx_id)

        # Image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

    def get_face_embedding(self, image_path):
        """Get face embedding vector

        Args:
            image_path: Image file path, PIL Image, Tensor, or numpy array.

        Returns:
            numpy.ndarray: Face embedding vector or None
            (if no face is detected)
        """
        import os

        # Read image and convert to NumPy array
        if isinstance(image_path, (str, os.PathLike)):
            img = Image.open(image_path).convert("RGB")
            img = np.array(img)
        elif isinstance(image_path, np.ndarray):
            img = image_path
        else:
            from image_evaluator._input_adapters import to_pil_image

            img = np.array(to_pil_image(image_path))

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
            float: ArcFace distance score or None
            (if face not detected in either image)
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

    def evaluate_folder_arcface_distance(
        self, reference_folder, generated_folder
    ):
        """Evaluate average ArcFace distance between images in two folders

        Stem-matched (M2-03): exact case-sensitive stems ignoring
        extension. Any unscorable pair fails the whole batch.

        Args:
            reference_folder: Folder path containing reference images
            generated_folder: Folder path containing generated images

        Returns:
            float: Average ArcFace distance

        Raises:
            FileNotFoundError: stem sets differ or dirs are empty.
            ValueError: duplicate stems or unsupported visible files.
            OSError: unreadable files.
            ValueError: any pair yields no face / cannot be scored.
        """
        from image_evaluator._stem_pairing import IMAGE_EXTS, pair_dirs

        pairs = pair_dirs(
            reference_folder, generated_folder, IMAGE_EXTS, IMAGE_EXTS
        )
        distances = []
        for ref_path, gen_path in pairs:
            dist = self.evaluate_arcface_distance(ref_path, gen_path)
            if dist is None:
                raise ValueError(f"Unscorable pair: {ref_path}, {gen_path}")
            distances.append(dist)
        return float(np.mean(distances))
