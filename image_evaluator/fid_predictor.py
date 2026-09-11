import importlib.metadata
import os
import warnings
from typing import Any

import numpy as np
import torch
from PIL import Image

# Upstream supported extensions in clean-fid 0.1.35
CLEANFID_SUPPORTED_EXTENSIONS = {
    "bmp",
    "jpg",
    "jpeg",
    "pgm",
    "png",
    "ppm",
    "tif",
    "tiff",
    "webp",
    "npy",
}


def _validate_image_file(file_path: str) -> None:
    """Validate that an image or npy file is readable and uncorrupted.

    Args:
        file_path: Absolute or relative path to the image or numpy file.

    Raises:
        ValueError: If the file cannot be opened, verified, or decoded.
    """
    ext = os.path.splitext(file_path)[1].lstrip(".").lower()
    if ext == "npy":
        try:
            arr = np.load(file_path)
            if not isinstance(arr, np.ndarray):
                raise ValueError("Loaded object is not a numpy array")
        except Exception as e:
            raise ValueError(
                f"Unreadable numpy file '{file_path}': {e}"
            ) from e
    else:
        try:
            with Image.open(file_path) as img:
                img.verify()
            with Image.open(file_path) as img:
                img.convert("RGB")
        except Exception as e:
            raise ValueError(
                f"Unreadable image file '{file_path}': {e}"
            ) from e


def _discover_and_validate_images(folder: str) -> list[str]:
    """Recursively discover and validate images in a directory.

    Args:
        folder: Path to directory to scan.

    Returns:
        Sorted list of valid image file paths.

    Raises:
        FileNotFoundError: If the folder does not exist.
        ValueError: If the path is not a directory or contains unreadable
            images.
    """
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Directory not found: '{folder}'")
    if not os.path.isdir(folder):
        raise ValueError(f"Expected directory path, got file: '{folder}'")

    image_paths: list[str] = []
    for root, _, files in os.walk(folder):
        for fname in files:
            ext = os.path.splitext(fname)[1].lstrip(".").lower()
            if ext in CLEANFID_SUPPORTED_EXTENSIONS:
                image_paths.append(os.path.join(root, fname))

    image_paths.sort()

    for p in image_paths:
        _validate_image_file(p)

    return image_paths


class FIDResult(dict[str, Any]):
    """Container for FID evaluation result and reproducibility metadata."""

    def __init__(
        self,
        fid: float,
        backend: str,
        version: str,
        mode: str,
        model: str,
        device: str,
        Nref: int,
        Ngen: int,
    ) -> None:
        data: dict[str, Any] = {
            "fid": fid,
            "score": fid,
            "backend": backend,
            "version": version,
            "mode": mode,
            "model": model,
            "device": device,
            "Nref": Nref,
            "Ngen": Ngen,
        }
        super().__init__(data)
        self.fid: float = fid
        self.score: float = fid
        self.backend: str = backend
        self.version: str = version
        self.mode: str = mode
        self.model: str = model
        self.device: str = device
        self.Nref: int = Nref
        self.Ngen: int = Ngen

    def __repr__(self) -> str:
        return (
            f"FIDResult(fid={self.fid:.6f}, backend='{self.backend}', "
            f"version='{self.version}', mode='{self.mode}', "
            f"model='{self.model}', device='{self.device}', "
            f"Nref={self.Nref}, Ngen={self.Ngen})"
        )


class FIDPredictor:
    """FID distance predictor using clean-fid under fixed clean protocol."""

    BACKEND: str = "clean-fid"
    VERSION: str = "0.1.35"
    MODE: str = "clean"
    MODEL_NAME: str = "inception_v3"
    # Minimum samples required for covariance estimation (not a reliability
    # guarantee)
    MIN_SAMPLES: int = 2

    def __init__(
        self,
        device: str | torch.device | None = "cpu",
        num_workers: int = 0,
        batch_size: int = 32,
    ) -> None:
        """Initialize FID distance predictor.

        Args:
            device: Computing device ('cpu', 'cuda', or torch.device).
                Default is 'cpu'. The requested device is strictly respected
                and never silently switched.
            num_workers: Number of DataLoader worker threads. Defaults to 0
                for stability across platforms.
            batch_size: Batch size for feature extraction. Defaults to 32.
        """
        if device is None:
            self.device: torch.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.num_workers: int = num_workers
        self.batch_size: int = batch_size

    def evaluate_folder_fid(
        self, reference_folder: str, generated_folder: str
    ) -> FIDResult:
        """Evaluate FID score between reference and generated image folders.

        Directory role contract:
            reference_folder: Real / reference dataset directory.
            generated_folder: Model generated dataset directory.

        Args:
            reference_folder: Path to directory containing reference images.
            generated_folder: Path to directory containing generated images.

        Returns:
            FIDResult: Evaluation result containing FID score and protocol
                metadata (backend, version, mode, model, device, Nref, Ngen).

        Raises:
            FileNotFoundError: If either directory does not exist.
            ValueError: If either path is not a directory, has < 2 images,
                contains unreadable images, or returns a non-finite value.
            ImportError: If clean-fid is not installed.
        """
        # 1. Discover and validate images on both sides
        ref_files = _discover_and_validate_images(reference_folder)
        gen_files = _discover_and_validate_images(generated_folder)

        Nref = len(ref_files)
        Ngen = len(gen_files)

        if Nref < self.MIN_SAMPLES:
            raise ValueError(
                f"Reference directory '{reference_folder}' contains {Nref} "
                f"valid image(s); at least {self.MIN_SAMPLES} images are "
                "required to compute covariance."
            )
        if Ngen < self.MIN_SAMPLES:
            raise ValueError(
                f"Generated directory '{generated_folder}' contains {Ngen} "
                f"valid image(s); at least {self.MIN_SAMPLES} images are "
                "required to compute covariance."
            )

        # Emit sample sensitivity warning without asserting universal
        # sufficiency threshold
        warnings.warn(
            f"FID is sensitive to sample size (Nref={Nref}, Ngen={Ngen}). "
            "Statistical reliability requires task-specific convergence or "
            "repeated trials; Nref and Ngen are recorded for auditability.",
            UserWarning,
            stacklevel=2,
        )

        # 2. Delayed import and strict version verification of clean-fid
        try:
            actual_version = importlib.metadata.version("clean-fid")
        except Exception as e:
            raise ImportError(
                "clean-fid is not installed. Please install clean-fid==0.1.35 "
                "to evaluate FID."
            ) from e

        if actual_version != self.VERSION:
            raise RuntimeError(
                f"clean-fid version mismatch: expected strictly "
                f"'{self.VERSION}', but found '{actual_version}'. "
                f"FID protocol requires exact version clean-fid=="
                f"{self.VERSION} for reproducibility."
            )

        try:
            from cleanfid import fid as cleanfid_fid
        except ImportError as e:
            raise ImportError(
                "clean-fid is not installed. Please install clean-fid==0.1.35 "
                "to evaluate FID."
            ) from e

        # 3. Invoke clean-fid with fixed protocol parameters
        score = cleanfid_fid.compute_fid(
            fdir1=reference_folder,
            fdir2=generated_folder,
            mode=self.MODE,
            model_name=self.MODEL_NAME,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            device=self.device,
            verbose=False,
        )

        # 4. Check for non-finite return value
        if score is None or not np.isfinite(score):
            raise ValueError(
                f"FID computation returned non-finite value: {score}"
            )

        return FIDResult(
            fid=float(score),
            backend=self.BACKEND,
            version=actual_version,
            mode=self.MODE,
            model=self.MODEL_NAME,
            device=str(self.device),
            Nref=Nref,
            Ngen=Ngen,
        )

    def evaluate_fid(
        self, reference_folder: str, generated_folder: str
    ) -> FIDResult:
        """Alias for evaluate_folder_fid."""
        return self.evaluate_folder_fid(reference_folder, generated_folder)
