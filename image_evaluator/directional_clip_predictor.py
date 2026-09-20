import os

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer


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
    ):
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        print(f"Loading CLIP model: {clip_model}")
        self.model = AutoModel.from_pretrained(clip_model).to(self.device)
        self.processor = AutoProcessor.from_pretrained(clip_model)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_model)

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
        if os.path.isfile(text):
            try:
                with open(text, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            except UnicodeDecodeError:
                with open(text, "r", encoding="latin-1") as f:
                    text = f.read().strip()

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
