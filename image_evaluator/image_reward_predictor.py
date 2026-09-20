"""ImageReward v1.0 human preference reward predictor.

Evaluates human preference reward for text-to-image synthesis using
the official ImageReward-v1.0 BLIP-based cross-attention backbone
and scalar reward head.

References:
    - Xu et al. (2023) "ImageReward: Learning and Evaluating Human Preferences
      for Text-to-Image Generation", NeurIPS 2023, arXiv:2304.05977.
"""

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)

from image_evaluator.model_assets import (
    ModelAsset,
    check_asset_and_permit_download,
)

IMAGE_REWARD_ASSET = ModelAsset(
    metric_id="image_reward",
    model_id="THUDM/ImageReward",
    source="huggingface",
    revision="v1.0",
    estimated_download_bytes=1786880927,
    install_extra="preference",
)

IMAGE_REWARD_SUPPORTED_EXTENSIONS = {
    "bmp",
    "jpg",
    "jpeg",
    "png",
    "webp",
}

# Normalization constants from official ImageReward implementation
IMAGE_REWARD_MEAN = 0.16717362830052426
IMAGE_REWARD_STD = 1.0333394966054072


def _get_image_reward_cached_path() -> str | None:
    """Find locally cached ImageReward.pt file if available."""
    # 1. Custom explicit environment variable override
    env_cp = os.environ.get("IMAGE_REWARD_CHECKPOINT_PATH")
    if env_cp and os.path.exists(env_cp):
        return env_cp

    # 2. Check standard ~/.cache/ImageReward directory
    env_root = os.environ.get("IMAGE_REWARD_ROOT")
    root_path = (
        os.path.expanduser("~/.cache/ImageReward")
        if not env_root
        else env_root
    )
    p2 = os.path.join(root_path, "ImageReward.pt")
    if os.path.exists(p2):
        return p2

    # 3. Check Hugging Face hub cache
    try:
        from huggingface_hub import try_to_load_from_cache

        p = try_to_load_from_cache("THUDM/ImageReward", "ImageReward.pt")
        if isinstance(p, str) and os.path.exists(p):
            return p
    except Exception:
        pass

    return None


def _is_image_reward_cached() -> bool:
    """Return True if ImageReward checkpoint is present in local cache."""
    return _get_image_reward_cached_path() is not None


class _MLP(nn.Module):
    """Scalar reward head MLP architecture from official ImageReward."""

    def __init__(self, input_size: int = 768) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _BlipModule(nn.Module):
    """Container holding BLIP vision transformer and text encoder."""

    def __init__(
        self, visual_encoder: nn.Module, text_encoder: nn.Module
    ) -> None:
        super().__init__()
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder


class _ImageRewardModel(nn.Module):
    """Complete ImageReward model matching official state_dict layout."""

    def __init__(
        self, visual_encoder: nn.Module, text_encoder: nn.Module
    ) -> None:
        super().__init__()
        self.blip = _BlipModule(visual_encoder, text_encoder)
        self.mlp = _MLP(768)
        self.mean = IMAGE_REWARD_MEAN
        self.std = IMAGE_REWARD_STD

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar human preference reward score."""
        image_embeds = self.blip.visual_encoder.forward_features(image)
        image_atts = torch.ones(
            image_embeds.size()[:-1],
            dtype=torch.long,
            device=image.device,
        )
        text_output = self.blip.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        txt_features = text_output.last_hidden_state[:, 0, :].float()
        raw_rewards = self.mlp(txt_features)
        rewards = (raw_rewards - self.mean) / self.std
        return rewards


class ImageRewardPredictor:
    """Predictor for ImageReward v1.0 human preference scalar reward.

    Extracts multi-modal cross-attention representations from BLIP ViT-L
    and a fine-tuned text encoder, followed by a scalar regression head.
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

        from timm.models.vision_transformer import VisionTransformer
        from transformers import BertTokenizer
        from transformers.models.blip.modeling_blip_text import (
            BlipTextConfig,
            BlipTextModel,
        )

        cp_path = self.checkpoint_path or _get_image_reward_cached_path()
        is_cached = cp_path is not None and os.path.exists(cp_path)

        check_asset_and_permit_download(
            asset=IMAGE_REWARD_ASSET,
            is_cached=is_cached,
            allow_download=self.allow_download,
            disclosure_callback=self.download_callback,
        )

        if not is_cached:
            from huggingface_hub import hf_hub_download

            cp_path = hf_hub_download(
                repo_id="THUDM/ImageReward",
                filename="ImageReward.pt",
            )

        # 1. Instantiate visual encoder (ViT-large, 224x224)
        visual_encoder = VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=0,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path_rate=0.1,
        )

        # 2. Instantiate cross-attention text encoder
        text_cfg = BlipTextConfig(
            vocab_size=30524,
            hidden_size=768,
            encoder_hidden_size=1024,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=512,
            is_decoder=True,
            add_cross_attention=True,
        )
        text_encoder = BlipTextModel(text_cfg)

        # 3. Assemble complete model
        model = _ImageRewardModel(
            visual_encoder=visual_encoder,
            text_encoder=text_encoder,
        )

        checkpoint = torch.load(
            cp_path,  # type: ignore[arg-type]
            map_location=self.device,
            weights_only=True,
        )
        state_dict = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict, strict=False)

        model = model.to(self.device)
        model.eval()

        self._model = model
        self._preprocess = Compose(
            [
                Resize(224, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(224),
                lambda img: img.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
        self._tokenizer = tokenizer

    def _prepare_image(self, image: Any) -> torch.Tensor:
        """Normalize supported input image representations to model tensor."""
        self._ensure_loaded()

        if isinstance(image, (str, os.PathLike)):
            img_path = str(image)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: '{img_path}'")
            if os.path.isdir(img_path):
                raise ValueError(
                    f"Expected single image file, got directory: '{img_path}'"
                )
            with Image.open(img_path) as img:
                pil_img = img.convert("RGB")
                tensor = self._preprocess(pil_img).unsqueeze(0)
                return tensor.to(self.device)

        if isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
            tensor = self._preprocess(pil_img).unsqueeze(0)
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

    def compute_image_reward(self, image: Any, prompt: str) -> float:
        """Compute ImageReward score between image and prompt."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        self._ensure_loaded()
        img_tensor = self._prepare_image(image)
        text_inputs = self._tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)

        with torch.no_grad():
            score = self._model(img_tensor, input_ids, attention_mask)

        return float(score.detach().cpu().item())

    def evaluate_image_reward(self, image: Any, prompt: str) -> float:
        """Alias for compute_image_reward."""
        return self.compute_image_reward(image=image, prompt=prompt)

    def evaluate_folder_image_reward(
        self, image_dir: str, prompt: str
    ) -> float:
        """Evaluate arithmetic mean ImageReward score for images in folder."""
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
            if ext in IMAGE_REWARD_SUPPORTED_EXTENSIONS:
                image_paths.append(os.path.join(image_dir, name))

        if not image_paths:
            raise ValueError(
                f"No supported image files found in '{image_dir}'"
            )

        scores = [
            self.compute_image_reward(image=p, prompt=prompt)
            for p in image_paths
        ]
        return float(np.mean(scores))


def compute_image_reward(
    image: Any,
    prompt: str,
    device: str | torch.device | None = None,
    allow_download: bool = False,
    download_callback: Any = None,
) -> float:
    """Convenience function to compute ImageReward on a single image."""
    predictor = ImageRewardPredictor(
        device=device,
        allow_download=allow_download,
        download_callback=download_callback,
    )
    return predictor.compute_image_reward(image=image, prompt=prompt)


def evaluate_image_reward(
    image: Any,
    prompt: str,
    device: str | torch.device | None = None,
    allow_download: bool = False,
    download_callback: Any = None,
) -> float:
    """Alias for compute_image_reward."""
    return compute_image_reward(
        image=image,
        prompt=prompt,
        device=device,
        allow_download=allow_download,
        download_callback=download_callback,
    )


def evaluate_folder_image_reward(
    image_dir: str,
    prompt: str,
    device: str | torch.device | None = None,
    allow_download: bool = False,
    download_callback: Any = None,
) -> float:
    """Convenience function to evaluate folder ImageReward arithmetic mean."""
    predictor = ImageRewardPredictor(
        device=device,
        allow_download=allow_download,
        download_callback=download_callback,
    )
    return predictor.evaluate_folder_image_reward(
        image_dir=image_dir, prompt=prompt
    )
