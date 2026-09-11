import os
from typing import Any

import numpy as np
import torch
from PIL import Image

PICKSCORE_SUPPORTED_EXTENSIONS = {
    "bmp",
    "jpg",
    "jpeg",
    "png",
    "webp",
}


def _validate_image_file(file_path: str) -> None:
    """Validate that an image file is readable and uncorrupted.

    Args:
        file_path: Path to the image file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is a directory or cannot be opened/decoded.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image not found: '{file_path}'")
    if os.path.isdir(file_path):
        raise ValueError(f"Expected image file, got directory: '{file_path}'")

    try:
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:
            img.convert("RGB")
    except Exception as e:
        raise ValueError(
            f"Cannot read or decode image '{file_path}': {e}"
        ) from e


def _discover_and_validate_images(folder: str) -> list[str]:
    """Discover and validate all supported images in a directory.

    Args:
        folder: Path to directory to scan.

    Returns:
        Sorted list of valid image file paths.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If the path is not a directory, has no valid images,
            or contains corrupted image files.
    """
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Directory not found: '{folder}'")
    if not os.path.isdir(folder):
        raise ValueError(f"Expected directory path, got file: '{folder}'")

    image_paths: list[str] = []
    for fname in os.listdir(folder):
        ext = os.path.splitext(fname)[1].lstrip(".").lower()
        if ext in PICKSCORE_SUPPORTED_EXTENSIONS:
            image_paths.append(os.path.join(folder, fname))

    image_paths.sort()

    if not image_paths:
        raise ValueError(f"No supported image files found in '{folder}'")

    for p in image_paths:
        _validate_image_file(p)

    return image_paths


class PickScoreResult(dict[str, Any]):
    """Container for PickScore evaluation result and protocol metadata."""

    def __init__(
        self,
        score: float,
        model: str,
        processor: str,
        device: str,
        prompt: str,
        sample_count: int = 1,
        scores: dict[str, float] | None = None,
    ) -> None:
        data: dict[str, Any] = {
            "pickscore": score,
            "score": score,
            "mean_score": score,
            "model": model,
            "processor": processor,
            "device": device,
            "prompt": prompt,
            "sample_count": sample_count,
        }
        if scores is not None:
            data["scores"] = scores
        super().__init__(data)
        self.score: float = score
        self.mean_score: float = score
        self.model: str = model
        self.processor: str = processor
        self.device: str = device
        self.prompt: str = prompt
        self.sample_count: int = sample_count
        self.scores: dict[str, float] | None = scores

    def __repr__(self) -> str:
        return (
            f"PickScoreResult(score={self.score:.4f}, model='{self.model}', "
            f"device='{self.device}', sample_count={self.sample_count})"
        )


class PickScorePredictor:
    """Evaluates human preference score for text-to-image synthesis."""

    MODEL_NAME: str = "yuvalkirstain/PickScore_v1"
    PROCESSOR_NAME: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

    def __init__(
        self,
        device: str | torch.device | None = None,
        model_name: str | None = None,
        processor_name: str | None = None,
    ) -> None:
        """Initialize PickScorePredictor with deferred model loading.

        Args:
            device: Target torch device ('cpu', 'cuda', 'mps', or
                torch.device). If None, auto-selects cuda -> mps -> cpu.
            model_name: Custom model name or path. Defaults to MODEL_NAME.
            processor_name: Custom processor name or path.
                Defaults to PROCESSOR_NAME.
        """
        if device is None:
            if torch.cuda.is_available():
                self.device: torch.device = torch.device("cuda")
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model_name: str = model_name or self.MODEL_NAME
        self.processor_name: str = processor_name or self.PROCESSOR_NAME
        self._model = None
        self._processor = None

    def _load_model(self) -> tuple[Any, Any]:
        """Load PickScore AutoModel and AutoProcessor on demand.

        Returns:
            Tuple of (model, processor).

        Raises:
            ImportError: If transformers is not installed.
        """
        if self._model is None or self._processor is None:
            try:
                from transformers import AutoModel, AutoProcessor
            except ImportError as e:
                raise ImportError(
                    "transformers is required to use PickScore. "
                    "Please install transformers."
                ) from e

            self._processor = AutoProcessor.from_pretrained(
                self.processor_name
            )
            self._model = (
                AutoModel.from_pretrained(self.model_name)
                .eval()
                .to(self.device)
            )
        return self._model, self._processor

    @staticmethod
    def compute_score_from_features(
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor | float,
    ) -> float:
        """Compute PickScore scalar from image and text feature embeddings.

        Formula:
            score = exp(logit_scale) * (text_norm @ image_norm.T)

        Args:
            image_features: Tensor of shape (1, D) or (D,).
            text_features: Tensor of shape (1, D) or (D,).
            logit_scale: Model logit scale parameter (tensor or float).

        Returns:
            float: PickScore scalar value.

        Raises:
            ValueError: If inputs have dimension mismatch or output is
                non-finite.
        """
        if image_features.ndim == 1:
            image_features = image_features.unsqueeze(0)
        if text_features.ndim == 1:
            text_features = text_features.unsqueeze(0)

        if image_features.shape[-1] != text_features.shape[-1]:
            raise ValueError(
                f"Feature dimension mismatch: image "
                f"{image_features.shape[-1]} vs text {text_features.shape[-1]}"
            )

        norm_img = image_features / image_features.norm(dim=-1, keepdim=True)
        norm_txt = text_features / text_features.norm(dim=-1, keepdim=True)

        if isinstance(logit_scale, torch.Tensor):
            scale = logit_scale.exp()
        else:
            scale = np.exp(logit_scale)

        score_tensor = scale * (norm_txt @ norm_img.T)
        score = float(score_tensor.squeeze().item())

        if not np.isfinite(score):
            raise ValueError(
                f"PickScore computation returned non-finite value: {score}"
            )

        return score

    def evaluate(self, image_path: str, prompt: str) -> float:
        """Evaluate PickScore between a single image and text prompt.

        Args:
            image_path: Path to the image file.
            prompt: Text prompt describing the image.

        Returns:
            float: PickScore scalar value (higher indicates higher
                human preference).

        Raises:
            ValueError: If prompt is empty or image cannot be read.
            FileNotFoundError: If image file does not exist.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty for PickScore evaluation."
            )

        _validate_image_file(image_path)

        model, processor = self._load_model()

        with Image.open(image_path) as img:
            pil_img = img.convert("RGB")

        raw_image_inputs = processor(
            images=pil_img,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in raw_image_inputs.items()
        }

        raw_text_inputs = processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in raw_text_inputs.items()
        }

        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            if hasattr(image_features, "pooler_output") and isinstance(
                image_features.pooler_output, torch.Tensor
            ):
                image_features = image_features.pooler_output

            text_features = model.get_text_features(**text_inputs)
            if hasattr(text_features, "pooler_output") and isinstance(
                text_features.pooler_output, torch.Tensor
            ):
                text_features = text_features.pooler_output

            logit_scale = model.logit_scale

            return self.compute_score_from_features(
                image_features=image_features,
                text_features=text_features,
                logit_scale=logit_scale,
            )

    def evaluate_folder(
        self, folder_path: str, prompt: str
    ) -> PickScoreResult:
        """Evaluate PickScore for all images in folder against a prompt.

        Args:
            folder_path: Path to folder containing images.
            prompt: Text prompt to evaluate each image against.

        Returns:
            PickScoreResult: Summary containing mean score, individual scores,
                and execution metadata.

        Raises:
            ValueError: If prompt is empty or folder contains no valid images.
            FileNotFoundError: If folder does not exist.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty for PickScore evaluation."
            )

        image_paths = _discover_and_validate_images(folder_path)

        # Pre-load model once for entire folder
        self._load_model()

        scores: dict[str, float] = {}
        for p in image_paths:
            fname = os.path.basename(p)
            scores[fname] = self.evaluate(p, prompt)

        mean_val = float(np.mean(list(scores.values())))

        return PickScoreResult(
            score=mean_val,
            model=self.model_name,
            processor=self.processor_name,
            device=str(self.device),
            prompt=prompt,
            sample_count=len(scores),
            scores=scores,
        )

    def evaluate_pickscore(self, image_or_folder: str, prompt: str) -> float:
        """Convenience method accepting either a single image or a folder.

        Args:
            image_or_folder: Path to image file or directory of images.
            prompt: Text prompt describing the image.

        Returns:
            float: PickScore scalar (for single image) or mean score
                (for folder).
        """
        if os.path.isdir(image_or_folder):
            res = self.evaluate_folder(image_or_folder, prompt)
            return res.mean_score
        return self.evaluate(image_or_folder, prompt)
