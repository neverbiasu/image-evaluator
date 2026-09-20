"""DINOv2 Image-to-Image similarity predictor using facebook/dinov2-base.

Computes visual structural and semantic fidelity between paired images by
evaluating cosine similarity across normalized final CLS-token embeddings.

References:
    - Oquab et al. (2023) "DINOv2: Learning Robust Visual Features without
      Supervision", arXiv:2304.07193.
"""

import os
from typing import Any

import numpy as np
import torch
from PIL import Image

from image_evaluator.model_assets import (
    ModelAsset,
    check_asset_and_permit_download,
)

DINO_SIMILARITY_ASSET = ModelAsset(
    metric_id="dino_similarity",
    model_id="facebook/dinov2-base",
    source="huggingface",
    revision="f9e44c814b77203eaa57a6bdbbd535f21ede1415",
    estimated_download_bytes=345942474,
    install_extra=None,
)


def _is_dinov2_cached(
    model_id: str = "facebook/dinov2-base",
    revision: str | None = DINO_SIMILARITY_ASSET.revision,
) -> bool:
    """Check if model weights and configs are cached locally in HF cache."""
    try:
        from huggingface_hub import try_to_load_from_cache

        model_path = try_to_load_from_cache(
            model_id, "model.safetensors", revision=revision
        )
        if not (isinstance(model_path, str) and os.path.exists(model_path)):
            model_path = try_to_load_from_cache(
                model_id, "pytorch_model.bin", revision=revision
            )
            if not (
                isinstance(model_path, str) and os.path.exists(model_path)
            ):
                return False

        proc_path = try_to_load_from_cache(
            model_id, "preprocessor_config.json", revision=revision
        )
        if not (isinstance(proc_path, str) and os.path.exists(proc_path)):
            return False

        config_path = try_to_load_from_cache(
            model_id, "config.json", revision=revision
        )
        return bool(
            isinstance(config_path, str) and os.path.exists(config_path)
        )
    except Exception:
        return False


class DinoSimilarityPredictor:
    """Predictor for DINOv2 pairwise image visual fidelity.

    Uses facebook/dinov2-base CLS-token cosine similarity.
    """

    def __init__(
        self,
        model_id: str = "facebook/dinov2-base",
        device: str | torch.device | None = None,
        allow_download: bool = False,
        download_callback: Any = None,
    ) -> None:
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model_id = model_id

        is_cached = _is_dinov2_cached(
            model_id, revision=DINO_SIMILARITY_ASSET.revision
        )
        check_asset_and_permit_download(
            asset=DINO_SIMILARITY_ASSET,
            is_cached=is_cached,
            allow_download=allow_download,
            disclosure_callback=download_callback,
        )

        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(
            model_id,
            revision=DINO_SIMILARITY_ASSET.revision,
            local_files_only=not allow_download,
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            revision=DINO_SIMILARITY_ASSET.revision,
            local_files_only=not allow_download,
        )
        self.model.to(self.device)
        self.model.eval()

    def _prepare_image(self, image: Any) -> torch.Tensor:
        """Transform input image into preprocessed tensor for DINOv2."""
        if isinstance(image, (str, os.PathLike)):
            img_path = str(image)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: '{img_path}'")
            if os.path.isdir(img_path):
                raise ValueError(
                    f"Expected single image file, got directory: '{img_path}'"
                )
            with Image.open(img_path) as pil_img:
                rgb_img = pil_img.convert("RGB")
                inputs = self.processor(images=rgb_img, return_tensors="pt")
                return inputs["pixel_values"].to(self.device)
        if isinstance(image, Image.Image):
            rgb_img = image.convert("RGB")
            inputs = self.processor(images=rgb_img, return_tensors="pt")
            return inputs["pixel_values"].to(self.device)
        if isinstance(image, (torch.Tensor, np.ndarray)):
            from image_evaluator._input_adapters import to_pil_image

            pil_img = to_pil_image(image)
            inputs = self.processor(images=pil_img, return_tensors="pt")
            return inputs["pixel_values"].to(self.device)
        raise TypeError(
            f"Unsupported image input type: {type(image).__name__}. "
            "Expected str path, PIL.Image.Image, np.ndarray, or torch.Tensor."
        )

    def compute_dino_similarity(self, image1: Any, image2: Any) -> float:
        """Compute cosine similarity between normalized CLS-token
        embeddings.
        """
        t1 = self._prepare_image(image1)
        t2 = self._prepare_image(image2)
        with torch.no_grad():
            out1 = self.model(pixel_values=t1)
            out2 = self.model(pixel_values=t2)
            cls1 = out1.last_hidden_state[:, 0, :]
            cls2 = out2.last_hidden_state[:, 0, :]
            cls1 = torch.nn.functional.normalize(cls1, p=2, dim=-1)
            cls2 = torch.nn.functional.normalize(cls2, p=2, dim=-1)
            sim = (cls1 * cls2).sum(dim=-1).item()
        return float(sim)

    def evaluate_dino_similarity(self, reference: Any, image: Any) -> float:
        """Evaluate DINOv2 similarity between reference and generated image."""
        return self.compute_dino_similarity(reference, image)

    def evaluate_folder_dino_similarity(
        self, reference_folder: str, image_folder: str
    ) -> float:
        """Evaluate average DINOv2 similarity between matching image pairs.

        Uses stem pairing adhering to M2-03 strict pairing contracts.
        """
        from image_evaluator._stem_pairing import IMAGE_EXTS, pair_dirs

        pairs = pair_dirs(
            reference_folder, image_folder, IMAGE_EXTS, IMAGE_EXTS
        )
        scores = [
            self.evaluate_dino_similarity(ref, img) for ref, img in pairs
        ]
        return float(np.mean(scores))

