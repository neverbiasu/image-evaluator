"""VQAScore evaluation predictor using CLIP-FlanT5-XL (3B).

Evaluates text-to-visual generation alignment and compositionality
by computing the question-answering posterior probability P(Yes | Image, Text)
under the official VQAScore formulation.

References:
    - Lin et al. (2024) "VQAScore: Evaluating Text-to-Visual Generation
      with Image-to-Text Generation", ECCV 2024, arXiv:2404.01291.
    - Lin et al. (2024) "GenAI-Bench: Evaluating and Improving Compositional
      Text-to-Visual Generation", CVPR 2024, arXiv:2406.13743.
"""

import os
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from image_evaluator.model_assets import (
    ModelAsset,
    check_asset_and_permit_download,
)

SYSTEM_MSG = (
    "A chat between a curious user and an artificial intelligence "
    "assistant. The assistant gives helpful, detailed, and polite "
    "answers to the user's questions."
)
DEFAULT_IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100

VQA_SCORE_ASSET = ModelAsset(
    metric_id="vqascore",
    model_id="zhiqiulin/clip-flant5-xl",
    source="huggingface",
    revision="3b4a6b1b618f4e286f5353b5b5147a3ae7d9ec55",
    estimated_download_bytes=6327057688,
    install_extra="vqa",
)

VQA_SCORE_SUPPORTED_EXTENSIONS = {
    "bmp",
    "jpg",
    "jpeg",
    "png",
    "webp",
}


def _get_vqascore_cached_paths() -> tuple[str | None, str | None]:
    """Find locally cached model snapshot directories if available."""
    env_xl = os.environ.get("VQASCORE_CHECKPOINT_PATH")
    env_clip = os.environ.get("VQASCORE_VISION_PATH")

    xl_path = env_xl if (env_xl and os.path.exists(env_xl)) else None
    clip_path = (
        env_clip if (env_clip and os.path.exists(env_clip)) else None
    )

    if xl_path is None or clip_path is None:
        try:
            from huggingface_hub import try_to_load_from_cache

            if xl_path is None:
                p1 = try_to_load_from_cache(
                    "zhiqiulin/clip-flant5-xl",
                    "pytorch_model.bin",
                    revision=VQA_SCORE_ASSET.revision,
                )
                if isinstance(p1, str) and os.path.exists(p1):
                    xl_path = os.path.dirname(p1)

            if clip_path is None:
                p2 = try_to_load_from_cache(
                    "openai/clip-vit-large-patch14-336", "pytorch_model.bin"
                )
                if isinstance(p2, str) and os.path.exists(p2):
                    clip_path = os.path.dirname(p2)
        except Exception:
            pass

    return xl_path, clip_path


def _is_vqascore_cached() -> bool:
    """Return True if both XL backbone and vision tower weights are cached."""
    xl_path, clip_path = _get_vqascore_cached_paths()
    return bool(
        xl_path
        and os.path.exists(xl_path)
        and clip_path
        and os.path.exists(clip_path)
    )


class CLIPVisionTower(nn.Module):
    """CLIP-ViT-L-14-336 vision encoder tower for VQAScore."""

    def __init__(self, vision_tower_name: str) -> None:
        super().__init__()
        from transformers import CLIPVisionConfig

        self.is_loaded = False
        self.vision_tower_name = vision_tower_name
        self.select_layer = -2
        self.select_feature = "patch"
        self.cfg_only = CLIPVisionConfig.from_pretrained(
            self.vision_tower_name
        )
        self.image_processor: Any = None
        self.vision_tower: Any = None

    def load_model(self) -> None:
        from transformers import CLIPImageProcessor, CLIPVisionModel

        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name
        )
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_tower_name
        )
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs: Any) -> torch.Tensor:
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(
                f"Unexpected select feature: {self.select_feature}"
            )
        return image_features

    @torch.no_grad()
    def forward(self, images: torch.Tensor | list[torch.Tensor]) -> Any:
        if isinstance(images, list):
            image_features = []
            for img in images:
                dev = self.vision_tower.device
                dt = self.vision_tower.dtype
                out = self.vision_tower(
                    img.to(device=dev, dtype=dt).unsqueeze(0),
                    output_hidden_states=True,
                )
                feat = self.feature_select(out).to(img.dtype)
                image_features.append(feat)
            return image_features

        dev = self.vision_tower.device
        dt = self.vision_tower.dtype
        outs = self.vision_tower(
            images.to(device=dev, dtype=dt), output_hidden_states=True
        )
        return self.feature_select(outs).to(images.dtype)

    @property
    def hidden_size(self) -> int:
        return int(self.cfg_only.hidden_size)


def _build_vision_projector(config: Any) -> nn.Sequential:
    """Build multi-layer perceptron vision-language projector."""
    projector_type = getattr(config, "mm_projector_type", "mlp2x_gelu")
    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules: list[nn.Module] = [
            nn.Linear(config.mm_hidden_size, config.d_model)
        ]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.d_model, config.d_model))
        return nn.Sequential(*modules)
    raise ValueError(f"Unsupported projector type: {projector_type}")


class _LazyCLIPT5Config:
    """Lazy class resolver for CLIPT5Config."""


def _get_clip_t5_classes() -> tuple[type, type]:
    """Define and return CLIPT5Config and CLIPT5ForConditionalGeneration."""
    from transformers import T5Config, T5ForConditionalGeneration

    class CLIPT5Config(T5Config):
        model_type = "clip_t5"

    class CLIPT5ForConditionalGeneration(T5ForConditionalGeneration):
        config_class = CLIPT5Config

        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self.embed_tokens = self.encoder.embed_tokens
            vision_tower_name = getattr(
                config, "mm_vision_tower", "openai/clip-vit-large-patch14-336"
            )
            self.vision_tower = CLIPVisionTower(vision_tower_name)
            self.mm_projector = _build_vision_projector(config)

        def get_vision_tower(self) -> CLIPVisionTower:
            return self.vision_tower

        def encode_images(self, images: torch.Tensor) -> torch.Tensor:
            image_features = self.get_vision_tower()(images)
            return self.mm_projector(image_features)

        def prepare_inputs_labels_for_multimodal(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.Tensor | None,
            decoder_attention_mask: torch.Tensor | None,
            past_key_values: Any,
            labels: torch.LongTensor | None,
            images: torch.Tensor | None,
        ) -> tuple[Any, ...]:
            if images is None:
                return (
                    None,
                    attention_mask,
                    decoder_attention_mask,
                    past_key_values,
                    self.embed_tokens(input_ids),
                    labels,
                )

            image_features = self.encode_images(images)
            new_input_embeds = []
            cur_image_idx = 0
            for _, cur_input_ids in enumerate(input_ids):
                if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                    raise NotImplementedError("Missing image token in inputs")
                image_token_indices = torch.where(
                    cur_input_ids == IMAGE_TOKEN_INDEX
                )[0]
                cur_new_input_embeds = []
                while image_token_indices.numel() > 0:
                    cur_image_features = image_features[cur_image_idx]
                    image_token_start = image_token_indices[0]
                    cur_new_input_embeds.append(
                        self.embed_tokens(cur_input_ids[:image_token_start])
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    cur_image_idx += 1
                    cur_input_ids = cur_input_ids[image_token_start + 1 :]
                    image_token_indices = torch.where(
                        cur_input_ids == IMAGE_TOKEN_INDEX
                    )[0]
                if cur_input_ids.numel() > 0:
                    cur_new_input_embeds.append(
                        self.embed_tokens(cur_input_ids)
                    )
                cur_new_input_embeds = [
                    x.to(device=self.device) for x in cur_new_input_embeds
                ]
                cat_embeds = torch.cat(cur_new_input_embeds, dim=0)
                new_input_embeds.append(cat_embeds)

            stacked_embeds = torch.stack(new_input_embeds, dim=0)
            if attention_mask is not None:
                pad_len = stacked_embeds.shape[1] - input_ids.shape[1]
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], pad_len),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    (new_attn_mask_pad_left, attention_mask), dim=1
                )

            return (
                None,
                attention_mask,
                decoder_attention_mask,
                past_key_values,
                stacked_embeds,
                labels,
            )

        def forward(  # type: ignore[override]
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            decoder_attention_mask: torch.Tensor | None = None,
            past_key_values: Any = None,
            inputs_embeds: torch.FloatTensor | None = None,
            labels: torch.LongTensor | None = None,
            use_cache: bool | None = None,
            output_attentions: bool | None = None,
            output_hidden_states: bool | None = None,
            images: torch.FloatTensor | None = None,
            return_dict: bool | None = None,
            **kwargs: Any,
        ) -> Any:
            if inputs_embeds is None and input_ids is not None:
                (
                    _,
                    attention_mask,
                    decoder_attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    attention_mask,
                    decoder_attention_mask,
                    past_key_values,
                    labels,
                    images,
                )

            return super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

    return CLIPT5Config, CLIPT5ForConditionalGeneration


def expand2square(
    pil_img: Image.Image, background_color: tuple[int, ...]
) -> Image.Image:
    """Pad PIL Image to square with background while preserving aspect."""
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result


def _t5_tokenizer_image_token(
    prompt: str, tokenizer: Any, image_token_index: int = IMAGE_TOKEN_INDEX
) -> torch.LongTensor:
    """Tokenize prompt while preserving image token placeholders."""
    prompt_chunks = [
        tokenizer(chunk).input_ids for chunk in prompt.split("<image>")
    ]

    def insert_separator(
        items: list[list[int]], sep: list[int]
    ) -> list[list[int]]:
        res: list[list[int]] = []
        for i, chunk in enumerate(items):
            res.append(chunk)
            if i < len(items) - 1:
                res.append(sep)
        return res

    input_ids: list[int] = []
    for x in insert_separator(prompt_chunks, [image_token_index]):
        input_ids.extend(x)
    return torch.tensor(input_ids, dtype=torch.long)


class VQAScorePredictor:
    """VQAScore evaluation predictor for text-to-visual generation.

    Computes official posterior probability P(Yes | Image, Prompt) using
    the fine-tuned CLIP-FlanT5-XL 3B architecture with frozen CLIP-ViT-L/14-336
    visual feature extraction.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        vision_path: str | None = None,
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

        self.checkpoint_path = checkpoint_path
        self.vision_path = vision_path
        self.allow_download = allow_download
        self.download_callback = download_callback

        self._model: Any = None
        self._tokenizer: Any = None
        self._image_processor: Any = None
        self._loss_fct = nn.CrossEntropyLoss(reduction="mean")

    def _ensure_loaded(self) -> None:
        """Lazily load model and tokenizer weights on first evaluation."""
        if self._model is not None:
            return

        xl_path = self.checkpoint_path
        clip_path = self.vision_path

        if xl_path is None or clip_path is None:
            c_xl, c_clip = _get_vqascore_cached_paths()
            if xl_path is None:
                xl_path = c_xl
            if clip_path is None:
                clip_path = c_clip

        is_cached = bool(
            xl_path
            and os.path.exists(xl_path)
            and clip_path
            and os.path.exists(clip_path)
        )

        check_asset_and_permit_download(
            asset=VQA_SCORE_ASSET,
            is_cached=is_cached,
            allow_download=self.allow_download,
            disclosure_callback=self.download_callback,
        )

        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        if not is_cached:
            if xl_path is None or not os.path.exists(xl_path):
                xl_path = snapshot_download(
                    "zhiqiulin/clip-flant5-xl",
                    revision=VQA_SCORE_ASSET.revision,
                    ignore_patterns=[
                        "*.msgpack",
                        "*.h5",
                        "trainer_state.json",
                        "training_args.bin",
                    ],
                )
            if clip_path is None or not os.path.exists(clip_path):
                clip_path = snapshot_download(
                    "openai/clip-vit-large-patch14-336",
                    ignore_patterns=["*.h5", "*.msgpack", "*.safetensors"],
                )
        else:
            if xl_path is None or not os.path.exists(xl_path):
                xl_path = snapshot_download(
                    "zhiqiulin/clip-flant5-xl",
                    revision=VQA_SCORE_ASSET.revision,
                    ignore_patterns=[
                        "*.msgpack",
                        "*.h5",
                        "trainer_state.json",
                        "training_args.bin",
                    ],
                    local_files_only=True,
                )
            if clip_path is None or not os.path.exists(clip_path):
                clip_path = snapshot_download(
                    "openai/clip-vit-large-patch14-336",
                    ignore_patterns=["*.h5", "*.msgpack", "*.safetensors"],
                    local_files_only=True,
                )

        self._tokenizer = AutoTokenizer.from_pretrained(
            xl_path, use_fast=False, model_max_length=2048
        )

        config_cls, model_cls = _get_clip_t5_classes()
        config = config_cls.from_pretrained(xl_path)
        config.mm_vision_tower = clip_path

        torch_dt = (
            torch.float16 if self.device.type == "cuda" else torch.bfloat16
        )

        self._model = model_cls.from_pretrained(
            xl_path,
            config=config,
            torch_dtype=torch_dt,
            low_cpu_mem_usage=True,
        )
        self._model.vision_tower.load_model()
        self._model = self._model.to(self.device)
        self._model.eval()
        self._model.requires_grad_(False)
        self._image_processor = self._model.vision_tower.image_processor

    def _prepare_image(self, image: Any) -> torch.Tensor:
        """Convert input image representation into preprocessed tensor."""
        if isinstance(image, (str, os.PathLike)):
            with Image.open(image) as img:
                pil_img = img.convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                pil_img = Image.fromarray(arr).convert("RGB")
            elif arr.ndim == 3:
                if arr.shape[2] == 4:
                    pil_img = Image.fromarray(arr).convert("RGB")
                elif arr.shape[2] == 3:
                    if arr.dtype in (np.float32, np.float64):
                        if arr.max() <= 1.0:
                            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                        else:
                            arr = arr.clip(0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(arr, mode="RGB")
                elif arr.shape[0] == 3:
                    # (C, H, W) transpose
                    arr = np.transpose(arr, (1, 2, 0))
                    if arr.dtype in (np.float32, np.float64):
                        if arr.max() <= 1.0:
                            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                        else:
                            arr = arr.clip(0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(arr, mode="RGB")
                else:
                    raise ValueError(
                        f"Unsupported numpy array shape for image: {arr.shape}"
                    )
            else:
                raise ValueError(
                    f"Unsupported numpy array dimension: {arr.ndim}"
                )
        elif isinstance(image, torch.Tensor):
            t = image.detach().cpu()
            if t.ndim == 4 and t.shape[0] == 1:
                t = t.squeeze(0)
            if t.ndim == 3 and t.shape[0] in (1, 3):
                if t.shape[0] == 1:
                    t = t.repeat(3, 1, 1)
                t_arr = t.permute(1, 2, 0).numpy()
                if t_arr.dtype in (np.float32, np.float64):
                    if t_arr.max() <= 1.0:
                        t_arr = (
                            (t_arr * 255.0).clip(0, 255).astype(np.uint8)
                        )
                    else:
                        t_arr = t_arr.clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(t_arr, mode="RGB")
            else:
                shape_str = str(tuple(t.shape))
                raise ValueError(
                    f"Unsupported torch tensor shape for image: {shape_str}"
                )
        else:
            raise TypeError(
                f"Unsupported image input type: {type(image).__name__}. "
                "Expected str path, PIL.Image.Image, np.ndarray, or "
                "torch.Tensor."
            )

        bg_color = (
            tuple(int(x * 255) for x in self._image_processor.image_mean)
            if self._image_processor is not None
            else (128, 128, 128)
        )
        padded_img = expand2square(pil_img, bg_color)
        pixel_values = self._image_processor.preprocess(
            padded_img, return_tensors="pt"
        )["pixel_values"]
        torch_dt = (
            torch.float16 if self.device.type == "cuda" else torch.bfloat16
        )
        return pixel_values.to(device=self.device, dtype=torch_dt)

    def compute_vqascore(self, image: Any, prompt: str) -> float:
        """Compute VQAScore posterior probability for image and prompt."""
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
        pixel_values = self._prepare_image(image)

        clean_prompt = prompt.strip()
        question = (
            f'Does this figure show "{clean_prompt}"? Please answer yes or no.'
        )
        formatted_question = (
            f"{SYSTEM_MSG} USER: {DEFAULT_IMAGE_TOKEN}\n{question} ASSISTANT: "
        )
        formatted_answer = "Yes"

        input_ids = _t5_tokenizer_image_token(
            formatted_question, self._tokenizer
        ).unsqueeze(0).to(self.device)
        labels = _t5_tokenizer_image_token(
            formatted_answer, self._tokenizer
        ).unsqueeze(0).to(self.device)

        attention_mask = input_ids.ne(self._tokenizer.pad_token_id)
        decoder_attention_mask = labels.ne(IGNORE_INDEX)

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                images=pixel_values,
                return_dict=True,
            )
            logits = outputs.logits
            loss = self._loss_fct(logits[0], labels[0])
            score = (-loss).exp().item()

        return float(score)

    def evaluate_vqascore(self, image: Any, prompt: str) -> float:
        """Alias for compute_vqascore."""
        return self.compute_vqascore(image=image, prompt=prompt)

    def evaluate_folder_vqascore(
        self, image_dir: str, prompt: str | Sequence[str]
    ) -> float:
        """Evaluate arithmetic mean VQAScore across images in directory."""
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
            if ext in VQA_SCORE_SUPPORTED_EXTENSIONS:
                image_paths.append(os.path.join(image_dir, name))

        if not image_paths:
            raise ValueError(
                f"No supported image files found in '{image_dir}'"
            )

        if isinstance(prompt, (list, tuple)) or (
            isinstance(prompt, Sequence)
            and not isinstance(prompt, (str, bytes, os.PathLike))
        ):
            prompt_seq = list(prompt)
            if len(prompt_seq) != len(image_paths):
                raise ValueError(
                    f"Number of prompts ({len(prompt_seq)}) does not match "
                    f"number of images ({len(image_paths)}) in '{image_dir}'"
                )
            scores = [
                self.compute_vqascore(image=p, prompt=pr)
                for p, pr in zip(image_paths, prompt_seq)
            ]
        elif isinstance(prompt, str) and os.path.isfile(prompt):
            with open(prompt, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            if len(lines) == len(image_paths):
                scores = [
                    self.compute_vqascore(image=p, prompt=pr)
                    for p, pr in zip(image_paths, lines)
                ]
            else:
                prompt_text = " ".join(lines) if lines else prompt
                scores = [
                    self.compute_vqascore(image=p, prompt=prompt_text)
                    for p in image_paths
                ]
        else:
            scores = [
                self.compute_vqascore(image=p, prompt=prompt)
                for p in image_paths
            ]
        return float(np.mean(scores))


def compute_vqascore(
    image: Any,
    prompt: str,
    device: str | torch.device | None = None,
    allow_download: bool = False,
    download_callback: Any = None,
) -> float:
    """Convenience function to compute VQAScore on a single image."""
    predictor = VQAScorePredictor(
        device=device,
        allow_download=allow_download,
        download_callback=download_callback,
    )
    return predictor.compute_vqascore(image=image, prompt=prompt)


def evaluate_vqascore(
    image: Any,
    prompt: str,
    device: str | torch.device | None = None,
    allow_download: bool = False,
    download_callback: Any = None,
) -> float:
    """Alias for compute_vqascore."""
    return compute_vqascore(
        image=image,
        prompt=prompt,
        device=device,
        allow_download=allow_download,
        download_callback=download_callback,
    )


def evaluate_folder_vqascore(
    image_dir: str,
    prompt: str,
    device: str | torch.device | None = None,
    allow_download: bool = False,
    download_callback: Any = None,
) -> float:
    """Convenience function to evaluate folder VQAScore arithmetic mean."""
    predictor = VQAScorePredictor(
        device=device,
        allow_download=allow_download,
        download_callback=download_callback,
    )
    return predictor.evaluate_folder_vqascore(
        image_dir=image_dir, prompt=prompt
    )
