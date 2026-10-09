import os
from typing import Any

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from image_evaluator.model_assets import (
    ModelAsset,
    check_asset_and_permit_download,
)

DIRECTIONAL_CLIP_ASSET = ModelAsset(
    metric_id="directional_clip",
    model_id="openai/clip-vit-base-patch32",
    source="huggingface",
    revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    estimated_download_bytes=608863580,
    install_extra=None,
)

DIRECTIONAL_CLIP_REQUIRED_FILES = (
    "config.json",
    "preprocessor_config.json",
    "pytorch_model.bin",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
)


def _is_directional_clip_cached(
    model_id: str = "openai/clip-vit-base-patch32",
    revision: str | None = DIRECTIONAL_CLIP_ASSET.revision,
) -> bool:
    """Check if model weights and configs are cached locally in HF cache."""
    try:
        from huggingface_hub import try_to_load_from_cache

        for filename in DIRECTIONAL_CLIP_REQUIRED_FILES:
            cached_path = try_to_load_from_cache(
                model_id, filename, revision=revision
            )
            if not (
                isinstance(cached_path, str) and os.path.exists(cached_path)
            ):
                return False
        return True
    except Exception:
        return False


class DirectionalClipPredictor:
    """Directional CLIP similarity for image editing evaluation.

    Measures whether an edit moved the image in the direction
    specified by the text prompts, independent of absolute alignment.

    Reference: Gal et al. (2022) "StyleGAN-NADA".
    """

    def __init__(
        self,
        clip_model: str = "openai/clip-vit-base-patch32",
        device=None,
        allow_download: bool = False,
        download_callback: Any = None,
    ):
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.clip_model = clip_model
        self.model = None
        self.processor = None
        self.tokenizer = None

        if clip_model != DIRECTIONAL_CLIP_ASSET.model_id:
            raise ValueError(
                f"Unsupported clip_model '{clip_model}' for Directional CLIP. "
                f"Only verified asset '{DIRECTIONAL_CLIP_ASSET.model_id}' "
                f"is supported."
            )

        revision = DIRECTIONAL_CLIP_ASSET.revision
        is_cached = _is_directional_clip_cached(
            clip_model, revision=revision
        )
        check_asset_and_permit_download(
            asset=DIRECTIONAL_CLIP_ASSET,
            is_cached=is_cached,
            allow_download=allow_download,
            disclosure_callback=download_callback,
        )

        model = None
        processor = None
        tokenizer = None
        try:
            model = AutoModel.from_pretrained(
                clip_model,
                revision=revision,
                local_files_only=not allow_download,
            ).to(self.device)
            processor = AutoProcessor.from_pretrained(
                clip_model,
                revision=revision,
                local_files_only=not allow_download,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                clip_model,
                revision=revision,
                local_files_only=not allow_download,
            )
            self.model = model
            self.processor = processor
            self.tokenizer = tokenizer
        except Exception:
            self.model = None
            self.processor = None
            self.tokenizer = None
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_directional_clip(
        self,
        image_src,
        image_edit,
        prompt_src: str,
        prompt_target: str,
    ) -> float:
        """Compute Directional CLIP similarity for a single image pair.

        The score measures how well the image edit direction (edit - src)
        aligns with the text direction (target - src) in CLIP embedding
        space.

        Args:
            image_src: Source image.  str path, PIL.Image, or
                torch.Tensor (C, H, W) normalized to [0, 1].
            image_edit: Edited image.  Same types accepted.
            prompt_src: Text description of the source image.
            prompt_target: Text description of the edit target.

        Returns:
            float in [-1, 1].  Higher values mean the edit direction
            better matches the text direction.  Returns 0.0 when the
            image or text delta is near-zero (numerically degenerate).
        """
        img_src_feat = self._get_image_features(image_src)
        img_edit_feat = self._get_image_features(image_edit)
        txt_src_feat = self._get_text_features(prompt_src)
        txt_tgt_feat = self._get_text_features(prompt_target)

        delta_img = img_edit_feat - img_src_feat
        delta_txt = txt_tgt_feat - txt_src_feat

        return self._cosine_similarity(delta_img, delta_txt)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_image_features(self, image) -> torch.Tensor:
        """Encode a single image into a unit-norm CLIP embedding."""
        if isinstance(image, (str, os.PathLike)):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            # torch.Tensor (C, H, W)
            from image_evaluator._input_adapters import to_pil_image

            pil_img = to_pil_image(image)

        inputs = self.processor(images=pil_img, return_tensors="pt")
        for k in inputs:
            v = inputs[k]
            if isinstance(v, torch.Tensor):
                if v.ndim == 3:
                    v = v.unsqueeze(0)
                inputs[k] = v.to(self.device)

        raw = self.model.get_image_features(**inputs)
        feat = self._extract_tensor(raw)
        return torch.nn.functional.normalize(feat, dim=-1)

    def _get_text_features(self, text: str) -> torch.Tensor:
        """Encode a text prompt into a unit-norm CLIP embedding."""
        if isinstance(text, (str, os.PathLike)) and os.path.isfile(text):
            try:
                with open(text, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            except UnicodeDecodeError:
                with open(text, "r", encoding="latin-1") as f:
                    text = f.read().strip()
        elif isinstance(text, list):
            text = text[0] if text else ""
        else:
            text = str(text)

        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        for k in inputs:
            if isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].to(self.device)

        raw = self.model.get_text_features(**inputs)
        feat = self._extract_tensor(raw)
        return torch.nn.functional.normalize(feat, dim=-1)

    @staticmethod
    def _extract_tensor(output) -> torch.Tensor:
        """Pull a plain Tensor from a HuggingFace model output."""
        if isinstance(output, torch.Tensor):
            return output
        pooler = getattr(output, "pooler_output", None)
        if isinstance(pooler, torch.Tensor):
            return pooler
        return output

    @staticmethod
    def _cosine_similarity(
        a: torch.Tensor,
        b: torch.Tensor,
        eps: float = 1e-6,
    ) -> float:
        """Cosine similarity between two (1, D) or (D,) tensors.

        Returns 0.0 when either vector is near-zero to avoid NaN.
        """
        a = a.flatten()
        b = b.flatten()
        norm_a = a.norm().item()
        norm_b = b.norm().item()
        if norm_a < eps or norm_b < eps:
            return 0.0
        return (a @ b).item() / (norm_a * norm_b)
