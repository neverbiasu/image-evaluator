import os
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image

from image_evaluator.model_assets import (
    ModelAsset,
    check_asset_and_permit_download,
)

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


PICKSCORE_MODEL_ASSET = ModelAsset(
    metric_id="pickscore",
    model_id="yuvalkirstain/PickScore_v1",
    source="huggingface",
    revision="a4e4367c6dfa7288a00c550414478f865b875800",
    estimated_download_bytes=3944552236,
    install_extra=None,
)

PICKSCORE_PROCESSOR_ASSET = ModelAsset(
    metric_id="pickscore",
    model_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    source="huggingface",
    revision="1c2b8495b28150b8a4922ee1c8edee224c284c0c",
    estimated_download_bytes=3616854,
    install_extra=None,
)

PICKSCORE_MODEL_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
)

PICKSCORE_PROCESSOR_REQUIRED_FILES = (
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
)


def _is_hf_asset_cached(
    model_id: str,
    required_files: tuple[str, ...],
    revision: str | None = None,
) -> bool:
    """Check if all required files for an asset exist in local HF cache."""
    try:
        from huggingface_hub import try_to_load_from_cache

        for filename in required_files:
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


def _is_pickscore_cached() -> bool:
    """Check if both PickScore model and processor are cached locally."""
    return _is_hf_asset_cached(
        PICKSCORE_MODEL_ASSET.model_id,
        PICKSCORE_MODEL_REQUIRED_FILES,
        revision=PICKSCORE_MODEL_ASSET.revision,
    ) and _is_hf_asset_cached(
        PICKSCORE_PROCESSOR_ASSET.model_id,
        PICKSCORE_PROCESSOR_REQUIRED_FILES,
        revision=PICKSCORE_PROCESSOR_ASSET.revision,
    )


_HEX_40_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")


def _validate_revision(revision: Any, asset_type: str, name: str) -> str:
    """Strictly validate revision is a 40-character hexadecimal commit hash.

    Must reject:
    - Non-string types
    - Length != 40
    - Non-hex characters
    - Leading/trailing whitespace
    - Appended newlines (e.g. \\n)
    - Branch names, tags, 'main', 'latest'
    No silent stripping allowed.
    """
    if not isinstance(revision, str) or not re.fullmatch(
        r"[0-9a-fA-F]{40}", revision
    ):
        raise ValueError(
            f"Custom PickScore {asset_type} '{name}' is a remote repository "
            f"identifier but does not provide a verified 40-character "
            f"hexadecimal commit hash revision (got: {revision!r}). Pinned "
            f"revisions are strictly required for reproducibility."
        )
    return revision


def _validate_custom_remote_metadata(
    name: str,
    asset_type: str,
    revision: Any,
    download_bytes: Any,
    required_files: Any,
) -> tuple[str, int, tuple[str, ...]]:
    """Strictly validate custom remote asset contract.

    Custom remote asset must provide:
    - 40-character hexadecimal commit hash revision
    - Positive integer download bytes (> 0, not bool)
    - Non-empty sequence of non-empty string filenames

    Raises ValueError before any cache check or from_pretrained call.
    """
    valid_rev = _validate_revision(revision, asset_type, name)

    if (
        isinstance(download_bytes, bool)
        or not isinstance(download_bytes, int)
        or download_bytes <= 0
    ):
        raise ValueError(
            f"Custom PickScore {asset_type} '{name}' requires a positive "
            f"integer download size in bytes (got: {download_bytes!r})."
        )

    if not isinstance(required_files, (list, tuple, set)):
        raise ValueError(
            f"Custom PickScore {asset_type} '{name}' requires a non-empty "
            f"sequence of required filenames "
            f"(got: {type(required_files).__name__})."
        )

    files_tuple = tuple(required_files)
    if not files_tuple:
        raise ValueError(
            f"Custom PickScore {asset_type} '{name}' requires a non-empty "
            f"sequence of required filenames (got empty sequence)."
        )

    for item in files_tuple:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"Custom PickScore {asset_type} '{name}' contains an invalid "
                f"required filename: {item!r}. All filenames must be "
                f"non-empty strings."
            )

    return valid_rev, download_bytes, files_tuple


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

    MODEL_NAME: str = PICKSCORE_MODEL_ASSET.model_id
    PROCESSOR_NAME: str = PICKSCORE_PROCESSOR_ASSET.model_id

    def __init__(
        self,
        device: str | torch.device | None = None,
        model_name: str | None = None,
        processor_name: str | None = None,
        allow_download: bool = False,
        download_callback: Any = None,
        model_revision: str | None = None,
        processor_revision: str | None = None,
        model_download_bytes: int | None = None,
        processor_download_bytes: int | None = None,
        model_required_files: tuple[str, ...] | None = None,
        processor_required_files: tuple[str, ...] | None = None,
    ) -> None:
        """Initialize PickScorePredictor with deferred loading and gating.

        Args:
            device: Target torch device ('cpu', 'cuda', 'mps', or
                torch.device). If None, auto-selects cuda -> mps -> cpu.
            model_name: Custom model name or path. Defaults to MODEL_NAME.
            processor_name: Custom processor name or path.
                Defaults to PROCESSOR_NAME.
            allow_download: Whether download of uncached weights is permitted.
            download_callback: Optional callable(asset, msg) upon download.
            model_revision: Optional 40-char hex commit hash for custom model.
            processor_revision: Optional 40-char hex commit hash for custom
                processor.
            model_download_bytes: Optional estimated bytes for custom model
                disclosure.
            processor_download_bytes: Optional estimated bytes for custom
                processor disclosure.
            model_required_files: Optional tuple of required filenames for
                custom model cache check.
            processor_required_files: Optional tuple of required filenames for
                custom processor cache check.
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
        self.allow_download: bool = allow_download
        self.download_callback: Any = download_callback

        self.is_custom_model = self.model_name != self.MODEL_NAME
        self.is_custom_processor = self.processor_name != self.PROCESSOR_NAME

        self.is_local_model = os.path.exists(self.model_name)
        if self.is_local_model:
            self.model_revision = None
            self.model_download_bytes = None
            self.model_required_files = ()
        elif not self.is_custom_model:
            self.model_revision = PICKSCORE_MODEL_ASSET.revision
            self.model_download_bytes = (
                PICKSCORE_MODEL_ASSET.estimated_download_bytes
            )
            self.model_required_files = PICKSCORE_MODEL_REQUIRED_FILES
        else:
            (
                self.model_revision,
                self.model_download_bytes,
                self.model_required_files,
            ) = _validate_custom_remote_metadata(
                self.model_name,
                "model",
                model_revision,
                model_download_bytes,
                model_required_files,
            )

        self.is_local_processor = os.path.exists(self.processor_name)
        if self.is_local_processor:
            self.processor_revision = None
            self.processor_download_bytes = None
            self.processor_required_files = ()
        elif not self.is_custom_processor:
            self.processor_revision = PICKSCORE_PROCESSOR_ASSET.revision
            self.processor_download_bytes = (
                PICKSCORE_PROCESSOR_ASSET.estimated_download_bytes
            )
            self.processor_required_files = PICKSCORE_PROCESSOR_REQUIRED_FILES
        else:
            (
                self.processor_revision,
                self.processor_download_bytes,
                self.processor_required_files,
            ) = _validate_custom_remote_metadata(
                self.processor_name,
                "processor",
                processor_revision,
                processor_download_bytes,
                processor_required_files,
            )

        self._model = None
        self._processor = None

    def _load_model(self) -> tuple[Any, Any]:
        """Load PickScore AutoModel and AutoProcessor on demand.

        Validates cache availability and respects download authorization
        before calling transformers.

        Returns:
            Tuple of (model, processor).

        Raises:
            ImportError: If transformers is not installed.
            DownloadNotAllowedError: If assets are not cached and
                allow_download is False.
        """
        if self._model is not None and self._processor is not None:
            return self._model, self._processor

        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "transformers is required to use PickScore. "
                "Please install transformers."
            ) from e

        disclosed_tracker: set[str] = set()

        # 1. Gating & Disclosure for Processor
        if not self.is_local_processor:
            if not self.is_custom_processor:
                proc_cached = _is_hf_asset_cached(
                    self.processor_name,
                    PICKSCORE_PROCESSOR_REQUIRED_FILES,
                    revision=self.processor_revision,
                )
                proc_asset = PICKSCORE_PROCESSOR_ASSET
            else:
                proc_cached = _is_hf_asset_cached(
                    self.processor_name,
                    self.processor_required_files,
                    revision=self.processor_revision,
                )
                proc_asset = ModelAsset(
                    metric_id="pickscore",
                    model_id=self.processor_name,
                    source="huggingface",
                    revision=self.processor_revision,
                    estimated_download_bytes=self.processor_download_bytes,
                )

            check_asset_and_permit_download(
                asset=proc_asset,
                is_cached=proc_cached,
                allow_download=self.allow_download,
                disclosure_callback=self.download_callback,
                disclosed_tracker=disclosed_tracker,
            )

        # 2. Gating & Disclosure for Model
        if not self.is_local_model:
            if not self.is_custom_model:
                model_cached = _is_hf_asset_cached(
                    self.model_name,
                    PICKSCORE_MODEL_REQUIRED_FILES,
                    revision=self.model_revision,
                )
                model_asset = PICKSCORE_MODEL_ASSET
            else:
                model_cached = _is_hf_asset_cached(
                    self.model_name,
                    self.model_required_files,
                    revision=self.model_revision,
                )
                model_asset = ModelAsset(
                    metric_id="pickscore",
                    model_id=self.model_name,
                    source="huggingface",
                    revision=self.model_revision,
                    estimated_download_bytes=self.model_download_bytes,
                )

            check_asset_and_permit_download(
                asset=model_asset,
                is_cached=model_cached,
                allow_download=self.allow_download,
                disclosure_callback=self.download_callback,
                disclosed_tracker=disclosed_tracker,
            )

        # 3. Instantiate Processor and Model safely
        proc_kwargs: dict[str, Any] = {}
        if not self.is_local_processor:
            proc_kwargs["revision"] = self.processor_revision
            proc_kwargs["local_files_only"] = not self.allow_download

        model_kwargs: dict[str, Any] = {}
        if not self.is_local_model:
            model_kwargs["revision"] = self.model_revision
            model_kwargs["local_files_only"] = not self.allow_download

        loaded_proc = None
        loaded_model = None
        try:
            loaded_proc = AutoProcessor.from_pretrained(
                self.processor_name, **proc_kwargs
            )
            loaded_model = (
                AutoModel.from_pretrained(self.model_name, **model_kwargs)
                .eval()
                .to(self.device)
            )
            self._processor = loaded_proc
            self._model = loaded_model
            return self._model, self._processor
        except Exception:
            self._processor = None
            self._model = None
            raise

    @staticmethod
    def _extract_tensor(features: Any) -> torch.Tensor:
        """Extract embedding tensor from tensor, ModelOutput, or tuple."""
        if isinstance(features, torch.Tensor):
            return features
        pooler = getattr(features, "pooler_output", None)
        if isinstance(pooler, torch.Tensor):
            return pooler
        if isinstance(features, (tuple, list)):
            # Real Transformers BaseModelOutputWithPooling.to_tuple() layout:
            # index 0: last_hidden_state (3D: batch, seq_len, dim)
            # index 1: pooler_output (2D: batch, dim)
            if (
                len(features) >= 2
                and isinstance(features[1], torch.Tensor)
                and features[1].ndim in (1, 2)
            ):
                return features[1]
            if (
                len(features) == 1
                and isinstance(features[0], torch.Tensor)
                and features[0].ndim in (1, 2)
            ):
                return features[0]
            raise TypeError(
                f"Unsupported tuple structure for PickScore features: "
                f"expected pooler tensor at index 1 or single-element tuple, "
                f"got length {len(features)}"
            )
        raise TypeError(
            f"Expected torch.Tensor, ModelOutput with pooler_output, or "
            f"Transformers output tuple, got {type(features).__name__}"
        )

    @staticmethod
    def compute_score_from_features(
        image_features: Any,
        text_features: Any,
        logit_scale: torch.Tensor | float,
    ) -> float:
        """Compute PickScore scalar from image and text feature embeddings.

        Formula:
            score = exp(logit_scale) * (text_norm @ image_norm.T)

        Args:
            image_features: Tensor or ModelOutput of shape (1, D) or (D,).
            text_features: Tensor or ModelOutput of shape (1, D) or (D,).
            logit_scale: Model logit scale parameter (tensor or float).

        Returns:
            float: PickScore scalar value.

        Raises:
            ValueError: If inputs have dimension mismatch or output is
                non-finite.
            TypeError: If input features cannot be resolved to 1D/2D tensors.
        """
        image_features = PickScorePredictor._extract_tensor(image_features)
        text_features = PickScorePredictor._extract_tensor(text_features)

        if not isinstance(image_features, torch.Tensor) or not isinstance(
            text_features, torch.Tensor
        ):
            raise TypeError(
                f"Expected torch.Tensor or ModelOutput with pooler_output, "
                f"got image: {type(image_features).__name__}, "
                f"text: {type(text_features).__name__}"
            )

        if image_features.ndim == 1:
            image_features = image_features.unsqueeze(0)
        if text_features.ndim == 1:
            text_features = text_features.unsqueeze(0)

        if image_features.ndim != 2 or text_features.ndim != 2:
            raise ValueError(
                f"Expected 1D or 2D feature tensors, got image: "
                f"{image_features.ndim}D, text: {text_features.ndim}D"
            )

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

    def evaluate(self, image_path: Any, prompt: str) -> float:
        """Evaluate PickScore between a single image and text prompt.

        Args:
            image_path: Path to the image file, PIL Image, or Tensor.
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

        if os.path.isfile(prompt):
            try:
                with open(prompt, "r", encoding="utf-8") as f:
                    prompt = f.read().strip()
            except UnicodeDecodeError:
                with open(prompt, "r", encoding="latin-1") as f:
                    prompt = f.read().strip()

        if isinstance(image_path, (str, os.PathLike)):
            _validate_image_file(str(image_path))
            with Image.open(image_path) as img:
                pil_img = img.convert("RGB")
        else:
            from image_evaluator._input_adapters import to_pil_image

            pil_img = to_pil_image(image_path)

        model, processor = self._load_model()

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
            raw_img_feat = model.get_image_features(**image_inputs)
            raw_txt_feat = model.get_text_features(**text_inputs)
            image_features = self._extract_tensor(raw_img_feat)
            text_features = self._extract_tensor(raw_txt_feat)
            logit_scale = model.logit_scale

        return self.compute_score_from_features(
            image_features, text_features, logit_scale
        )

    def evaluate_folder(
        self, folder_path: str, prompt: str | Sequence[str]
    ) -> PickScoreResult:
        """Evaluate PickScore for all images in folder against a prompt.

        Args:
            folder_path: Path to folder containing images.
            prompt: Text prompt, sequence of prompts, or path to prompt file.

        Returns:
            PickScoreResult: Summary containing mean score, individual scores,
                and execution metadata.

        Raises:
            ValueError: If prompt is empty or folder contains no valid images.
            FileNotFoundError: If folder does not exist.
        """
        image_paths = _discover_and_validate_images(folder_path)

        # Pre-load model once for entire folder
        self._load_model()

        if isinstance(prompt, (list, tuple)) or (
            isinstance(prompt, Sequence)
            and not isinstance(prompt, (str, bytes, os.PathLike))
        ):
            prompt_seq = list(prompt)
            if len(prompt_seq) != len(image_paths):
                raise ValueError(
                    f"Number of prompts ({len(prompt_seq)}) does not match "
                    f"number of images ({len(image_paths)})"
                )
            prompts = prompt_seq
        elif isinstance(prompt, str) and os.path.isfile(prompt):
            with open(prompt, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            if len(lines) == len(image_paths):
                prompts = lines
            else:
                prompt_text = " ".join(lines) if lines else prompt
                prompts = [prompt_text] * len(image_paths)
        else:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    "Prompt cannot be empty for PickScore evaluation."
                )
            prompts = [prompt] * len(image_paths)

        scores: dict[str, float] = {}
        for p, pr in zip(image_paths, prompts):
            fname = os.path.basename(p)
            scores[fname] = self.evaluate(p, pr)

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

    def evaluate_pickscore(
        self,
        image_or_folder: str | None = None,
        prompt: str = "",
        *,
        image_path: str | None = None,
    ) -> float:
        """Convenience method accepting either a single image or a folder.

        Args:
            image_or_folder: Path to image file or directory of images.
            prompt: Text prompt describing the image.
            image_path: Backward-compatible keyword alias for image_or_folder.

        Returns:
            float: PickScore scalar (for single image) or mean score
                (for folder).
        """
        target = image_or_folder or image_path
        if target is None:
            raise ValueError("image_or_folder or image_path must be provided.")
        if os.path.isdir(target):
            res = self.evaluate_folder(target, prompt)
            return res.mean_score
        return self.evaluate(target, prompt)

    # Backward-compatible convenience alias
    evaluate_folder_pickscore = evaluate_folder
