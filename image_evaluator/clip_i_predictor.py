"""CLIP-I Image-to-Image similarity predictor using ViT-L/14.

Computes visual semantic and subject fidelity between paired images by
evaluating cosine similarity across normalized CLIP visual projection
embeddings.

References:
    - Radford et al. (2021) "Learning Transferable Visual Models From
      Natural Language Supervision", ICML 2021.
    - Ruiz et al. (2023) "DreamBooth: Fine Tuning Text-to-Image Diffusion
      Models for Subject-Driven Generation", CVPR 2023.
"""

import os
from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image

from image_evaluator.model_assets import (
    ModelAsset,
    check_asset_and_permit_download,
)

CLIP_I_ASSET = ModelAsset(
    metric_id="clip_i",
    model_id="openai/clip-vit-large-patch14",
    source="open_clip/openai",
    revision="openai",
    estimated_download_bytes=1711205010,
    install_extra="modern",
)


def _is_clip_i_cached(
    model_name: str = "ViT-L-14-quickgelu",
    pretrained: str = "openai",
) -> bool:
    """Check if model weights are cached locally without network calls."""
    try:
        cfg = open_clip.pretrained.get_pretrained_cfg(model_name, pretrained)
        if not cfg:
            cfg = open_clip.pretrained.get_pretrained_cfg(
                "ViT-L-14", pretrained
            )
        if not cfg:
            return False

        if "file" in cfg:
            return bool(os.path.exists(cfg["file"]))

        url = cfg.get("url", "")
        if url:
            filename = os.path.basename(url)
            cache_dir = os.path.expanduser("~/.cache/clip")
            target = os.path.join(cache_dir, filename)
            return bool(os.path.isfile(target))

        hf_hub = cfg.get("hf_hub", "")
        if hf_hub:
            from huggingface_hub import try_to_load_from_cache

            model_id, filename = os.path.split(hf_hub)
            filename = filename or "open_clip_pytorch_model.bin"
            cached = try_to_load_from_cache(model_id, filename)
            return bool(isinstance(cached, str) and os.path.exists(cached))

        return False
    except Exception:
        return False


class ClipIPredictor:
    """Predictor for CLIP-I pairwise image visual fidelity.

    Uses OpenAI ViT-L/14 with QuickGELU visual projection embeddings.
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14-quickgelu",
        pretrained: str = "openai",
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

        self.model_name = model_name
        self.pretrained = pretrained

        is_cached = _is_clip_i_cached(model_name, pretrained)
        check_asset_and_permit_download(
            asset=CLIP_I_ASSET,
            is_cached=is_cached,
            allow_download=allow_download,
            disclosure_callback=download_callback,
        )

        self.model, _, self.preprocess = (
            open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=self.device,
            )
        )
        self.model.eval()

    def _prepare_image(self, image: Any) -> torch.Tensor:
        """Transform input image into preprocessed tensor for ViT-L/14."""
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
                tensor = self.preprocess(rgb_img).unsqueeze(0)
                return tensor.to(self.device)
        if isinstance(image, Image.Image):
            rgb_img = image.convert("RGB")
            tensor = self.preprocess(rgb_img).unsqueeze(0)
            return tensor.to(self.device)
        if isinstance(image, (torch.Tensor, np.ndarray)):
            from image_evaluator._input_adapters import to_pil_image

            pil_img = to_pil_image(image)
            tensor = self.preprocess(pil_img).unsqueeze(0)
            return tensor.to(self.device)
        raise TypeError(
            f"Unsupported image input type: {type(image).__name__}. "
            "Expected str path, PIL.Image.Image, np.ndarray, or torch.Tensor."
        )

    def compute_clip_i_similarity(
        self, image1: Any, image2: Any
    ) -> float:
        """Compute cosine similarity between normalized CLIP embeddings."""
        t1 = self._prepare_image(image1)
        t2 = self._prepare_image(image2)
        with torch.no_grad():
            f1 = self.model.encode_image(t1, normalize=True)
            f2 = self.model.encode_image(t2, normalize=True)
            sim = (f1 * f2).sum(dim=-1).item()
        return float(sim)

    def evaluate_clip_i(self, reference: Any, image: Any) -> float:
        """Evaluate CLIP-I similarity between reference and generated image."""
        return self.compute_clip_i_similarity(reference, image)

    def evaluate_folder_clip_i(
        self, reference_folder: str, image_folder: str
    ) -> float:
        """Evaluate average CLIP-I similarity between matching image pairs.

        Uses stem pairing adhering to M2-03 strict pairing contracts.
        """
        from image_evaluator._stem_pairing import IMAGE_EXTS, pair_dirs

        pairs = pair_dirs(
            reference_folder, image_folder, IMAGE_EXTS, IMAGE_EXTS
        )
        scores = [self.evaluate_clip_i(ref, img) for ref, img in pairs]
        return float(np.mean(scores))
