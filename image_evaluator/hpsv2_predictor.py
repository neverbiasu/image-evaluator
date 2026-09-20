"""Human Preference Score v2.1 (HPSv2) predictor.

Evaluates human preference score for text-to-image synthesis using
the official OpenCLIP ViT-H-14 backbone fine-tuned on human preference choices
with the single approved HPS_v2.1_compressed.pt checkpoint.

References:
    - Wu et al. (2023) "Human Preference Score v2: A Benchmark and Dataset for
      Human Preference Evaluation", NeurIPS 2023, arXiv:2306.09341.
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

HPSV2_ASSET = ModelAsset(
    metric_id="hpsv2",
    model_id="xswu/HPSv2",
    source="huggingface",
    revision="697403c",
    estimated_download_bytes=1972490005,
    install_extra="preference",
)

HPSV2_SUPPORTED_EXTENSIONS = {
    "bmp",
    "jpg",
    "jpeg",
    "png",
    "webp",
}


def _get_hpsv2_cached_path() -> str | None:
    """Find locally cached HPS_v2.1_compressed.pt file if available."""
    # 1. Custom explicit environment variable override
    env_cp = os.environ.get("HPS_CHECKPOINT_PATH")
    if env_cp and os.path.exists(env_cp):
        return env_cp

    # 2. Check standard ~/.cache/hpsv2 directory
    env_root = os.environ.get("HPS_ROOT")
    root_path = (
        os.path.expanduser("~/.cache/hpsv2") if not env_root else env_root
    )
    p2 = os.path.join(root_path, "HPS_v2.1_compressed.pt")
    if os.path.exists(p2):
        return p2

    # 3. Check Hugging Face hub cache
    try:
        from huggingface_hub import try_to_load_from_cache

        p = try_to_load_from_cache(
            "xswu/HPSv2",
            "HPS_v2.1_compressed.pt",
            revision=HPSV2_ASSET.revision,
        )
        if isinstance(p, str) and os.path.exists(p):
            return p
    except Exception:
        pass

    return None


def _is_hpsv2_cached() -> bool:
    """Return True if HPS v2.1 checkpoint is present in local cache."""
    return _get_hpsv2_cached_path() is not None


class Hpsv2Predictor:
    """Predictor for Human Preference Score v2.1 text-image alignment.

    Extracts normalized image and text representations using fine-tuned
    OpenCLIP ViT-H-14 weights and computes their cosine similarity.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str | torch.device | None = None,
        allow_download: bool = False,
        download_callback: Any = None,
    ) -> None:
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.checkpoint_path = checkpoint_path
        self.allow_download = allow_download
        self.download_callback = download_callback

        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import open_clip

        cp_path = self.checkpoint_path or _get_hpsv2_cached_path()
        is_cached = cp_path is not None and os.path.exists(cp_path)

        check_asset_and_permit_download(
            asset=HPSV2_ASSET,
            is_cached=is_cached,
            allow_download=self.allow_download,
            disclosure_callback=self.download_callback,
        )

        if not is_cached:
            from huggingface_hub import hf_hub_download

            cp_path = hf_hub_download(
                repo_id="xswu/HPSv2",
                filename="HPS_v2.1_compressed.pt",
                revision=HPSV2_ASSET.revision,
            )

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14",
            pretrained=None,
            output_dict=True,
        )

        state_dict = torch.load(cp_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()

        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer("ViT-H-14")
        self._loaded = True

    def _prepare_image(self, image: Any) -> torch.Tensor:
        """Convert input image into preprocessed tensor for ViT-H-14."""
        self._ensure_loaded()
        if isinstance(image, (str, os.PathLike)):
            img_path = str(image)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: '{img_path}'")
            if os.path.isdir(img_path):
                raise ValueError(
                    f"Expected single image file, got directory: '{img_path}'"
                )
            with Image.open(img_path) as pil_img:
                img_rgb = pil_img.convert("RGB")
                tensor = self._preprocess(img_rgb).unsqueeze(0)
                return tensor.to(self.device)

        if isinstance(image, Image.Image):
            img_rgb = image.convert("RGB")
            tensor = self._preprocess(img_rgb).unsqueeze(0)
            return tensor.to(self.device)

        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
            tensor = self._preprocess(pil_img).unsqueeze(0)
            return tensor.to(self.device)

        if isinstance(image, torch.Tensor):
            t = image.detach()
            if t.dim() == 4 and t.shape[0] == 1:
                t = t.squeeze(0)
            if t.dim() != 3 or t.shape[0] != 3:
                raise ValueError(
                    "Expected 3-channel image tensor, got shape "
                    f"{tuple(t.shape)}"
                )
            if t.min() < 0.0:
                t = (t * 0.5 + 0.5).clamp(0.0, 1.0)
            t = t.clamp(0.0, 1.0)
            arr = (t.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            pil_img = Image.fromarray(arr).convert("RGB")
            tensor = self._preprocess(pil_img).unsqueeze(0)
            return tensor.to(self.device)

        raise TypeError(
            f"Unsupported image input type: {type(image).__name__}. "
            "Expected str path, PIL.Image.Image, np.ndarray, or torch.Tensor."
        )

    def compute_hpsv2(self, image: Any, prompt: str) -> float:
        """Compute HPS v2.1 score between image and prompt."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        if os.path.isfile(prompt):
            try:
                with open(prompt, "r", encoding="utf-8") as f:
                    prompt = f.read().strip()
            except UnicodeDecodeError:
                with open(prompt, "r", encoding="latin-1") as f:
                    prompt = f.read().strip()

        self._ensure_loaded()
        img_tensor = self._prepare_image(image)
        tokens = self._tokenizer([prompt]).to(self.device)

        with torch.no_grad():
            outputs = self._model(img_tensor, tokens)
            img_feats = outputs["image_features"]
            txt_feats = outputs["text_features"]
            # Ensure normalized
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            score = (img_feats @ txt_feats.T).item()

        return float(score)

    def evaluate_hpsv2(self, image: Any, prompt: str) -> float:
        """Alias for compute_hpsv2."""
        return self.compute_hpsv2(image=image, prompt=prompt)

    def evaluate_folder_hpsv2(
        self, image_dir: str, prompt: str | list[str]
    ) -> float:
        """Evaluate arithmetic mean HPS v2.1 score across images in folder."""
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Directory not found: '{image_dir}'")
        if not os.path.isdir(image_dir):
            raise ValueError(
                f"Expected directory path, got file: '{image_dir}'"
            )

        entries = sorted(os.listdir(image_dir))
        image_paths: list[str] = []
        for name in entries:
            ext = os.path.splitext(name)[1].lstrip(".").lower()
            if ext in HPSV2_SUPPORTED_EXTENSIONS:
                image_paths.append(os.path.join(image_dir, name))

        if not image_paths:
            raise ValueError(
                f"No supported image files found in '{image_dir}'"
            )

        if isinstance(prompt, list):
            if len(prompt) != len(image_paths):
                raise ValueError(
                    f"Number of prompts ({len(prompt)}) does not match "
                    f"number of images ({len(image_paths)}) in '{image_dir}'"
                )
            scores = [
                self.compute_hpsv2(image=p, prompt=pr)
                for p, pr in zip(image_paths, prompt)
            ]
        elif isinstance(prompt, str) and os.path.isfile(prompt):
            with open(prompt, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            if len(lines) == len(image_paths):
                scores = [
                    self.compute_hpsv2(image=p, prompt=pr)
                    for p, pr in zip(image_paths, lines)
                ]
            else:
                prompt_text = " ".join(lines) if lines else prompt
                scores = [
                    self.compute_hpsv2(image=p, prompt=prompt_text)
                    for p in image_paths
                ]
        else:
            scores = [
                self.compute_hpsv2(image=p, prompt=prompt)
                for p in image_paths
            ]
        return float(np.mean(scores))


def compute_hpsv2(
    image: Any,
    prompt: str,
    device: str | torch.device | None = None,
    allow_download: bool = False,
    download_callback: Any = None,
) -> float:
    """Convenience function to compute HPS v2.1 score."""
    predictor = Hpsv2Predictor(
        device=device,
        allow_download=allow_download,
        download_callback=download_callback,
    )
    return predictor.compute_hpsv2(image=image, prompt=prompt)


def evaluate_hpsv2(
    image: Any,
    prompt: str,
    device: str | torch.device | None = None,
    allow_download: bool = False,
    download_callback: Any = None,
) -> float:
    """Convenience function to evaluate HPS v2.1 score."""
    return compute_hpsv2(
        image=image,
        prompt=prompt,
        device=device,
        allow_download=allow_download,
        download_callback=download_callback,
    )


def evaluate_folder_hpsv2(
    image_dir: str,
    prompt: str,
    device: str | torch.device | None = None,
    allow_download: bool = False,
    download_callback: Any = None,
) -> float:
    """Convenience function to evaluate HPS v2.1 across a folder."""
    predictor = Hpsv2Predictor(
        device=device,
        allow_download=allow_download,
        download_callback=download_callback,
    )
    return predictor.evaluate_folder_hpsv2(image_dir=image_dir, prompt=prompt)
