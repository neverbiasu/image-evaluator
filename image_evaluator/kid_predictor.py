import importlib.metadata
import os
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


class KIDResult(dict[str, Any]):
    """Container for KID evaluation result and reproducibility metadata."""

    def __init__(
        self,
        kid: float,
        backend: str,
        version: str,
        mode: str,
        model: str,
        device: str,
        num_subsets: int,
        max_subset_size: int,
        seed: int | None,
        Nref: int,
        Ngen: int,
    ) -> None:
        data: dict[str, Any] = {
            "kid": kid,
            "score": kid,
            "backend": backend,
            "version": version,
            "mode": mode,
            "model": model,
            "device": device,
            "num_subsets": num_subsets,
            "max_subset_size": max_subset_size,
            "seed": seed,
            "Nref": Nref,
            "Ngen": Ngen,
        }
        super().__init__(data)
        self.kid: float = kid
        self.score: float = kid
        self.backend: str = backend
        self.version: str = version
        self.mode: str = mode
        self.model: str = model
        self.device: str = device
        self.num_subsets: int = num_subsets
        self.max_subset_size: int = max_subset_size
        self.seed: int | None = seed
        self.Nref: int = Nref
        self.Ngen: int = Ngen

    def __repr__(self) -> str:
        return (
            f"KIDResult(kid={self.kid:.6f}, backend='{self.backend}', "
            f"version='{self.version}', mode='{self.mode}', "
            f"model='{self.model}', device='{self.device}', "
            f"num_subsets={self.num_subsets}, "
            f"max_subset_size={self.max_subset_size}, seed={self.seed}, "
            f"Nref={self.Nref}, Ngen={self.Ngen})"
        )


class KIDPredictor:
    """KID distance predictor using clean-fid under fixed clean protocol."""

    BACKEND: str = "clean-fid"
    VERSION: str = "0.1.35"
    MODE: str = "clean"
    MODEL_NAME: str = "inception_v3"
    NUM_SUBSETS: int = 100
    MAX_SUBSET_SIZE: int = 1000
    MIN_SAMPLES: int = 2

    def __init__(
        self,
        device: str | torch.device | None = "cpu",
        seed: int | None = 0,
        num_workers: int = 0,
        batch_size: int = 32,
    ) -> None:
        """Initialize KID distance predictor.

        Args:
            device: Computing device ('cpu', 'cuda', or torch.device).
                Default is 'cpu'. The requested device is strictly respected
                and never silently switched.
            seed: Random seed for subset sampling. Default is 0 for
                deterministic reproducibility. Pass None for non-deterministic
                sampling.
            num_workers: Number of DataLoader worker threads. Defaults to 0
                for stability across platforms.
            batch_size: Batch size for feature extraction. Defaults to 32.
        """
        if device is None:
            self.device: torch.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.seed: int | None = seed
        self.num_workers: int = num_workers
        self.batch_size: int = batch_size

    def evaluate_folder_kid(
        self, reference_folder: str, generated_folder: str
    ) -> KIDResult:
        """Evaluate KID score between reference and generated image folders.

        Directory role contract:
            - reference_folder: Reference distribution (fdir1)
            - generated_folder: Generated distribution (fdir2)

        Args:
            reference_folder: Path to directory of reference images.
            generated_folder: Path to directory of generated images.

        Returns:
            KIDResult with scalar kid value (can be negative due to unbiased
            U-statistic estimation) and protocol audit metadata.

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
                "required to compute polynomial kernel distance."
            )
        if Ngen < self.MIN_SAMPLES:
            raise ValueError(
                f"Generated directory '{generated_folder}' contains {Ngen} "
                f"valid image(s); at least {self.MIN_SAMPLES} images are "
                "required to compute polynomial kernel distance."
            )

        # 2. Delayed import and strict version verification of clean-fid
        try:
            actual_version = importlib.metadata.version("clean-fid")
        except Exception as e:
            raise ImportError(
                "clean-fid is not installed. Please install clean-fid==0.1.35 "
                "to evaluate KID."
            ) from e

        if actual_version != self.VERSION:
            raise RuntimeError(
                f"clean-fid version mismatch: expected strictly "
                f"'{self.VERSION}', but found '{actual_version}'. "
                f"KID protocol requires exact version clean-fid=="
                f"{self.VERSION} for reproducibility."
            )

        try:
            from cleanfid import fid as cleanfid_fid
        except ImportError as e:
            raise ImportError(
                "clean-fid is not installed. Please install clean-fid==0.1.35 "
                "to evaluate KID."
            ) from e

        # 3. Invoke clean-fid compute_kid under seed management
        if self.seed is not None:
            old_state = np.random.get_state()
            try:
                np.random.seed(self.seed)
                score = cleanfid_fid.compute_kid(
                    fdir1=reference_folder,
                    fdir2=generated_folder,
                    mode=self.MODE,
                    num_workers=self.num_workers,
                    batch_size=self.batch_size,
                    device=self.device,
                    verbose=False,
                )
            finally:
                np.random.set_state(old_state)
        else:
            score = cleanfid_fid.compute_kid(
                fdir1=reference_folder,
                fdir2=generated_folder,
                mode=self.MODE,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                device=self.device,
                verbose=False,
            )

        # 4. Check for non-finite return value (negative values are valid)
        if score is None or not np.isfinite(score):
            raise ValueError(
                f"KID computation returned non-finite value: {score}"
            )

        return KIDResult(
            kid=float(score),
            backend=self.BACKEND,
            version=self.VERSION,
            mode=self.MODE,
            model=self.MODEL_NAME,
            device=str(self.device),
            num_subsets=self.NUM_SUBSETS,
            max_subset_size=self.MAX_SUBSET_SIZE,
            seed=self.seed,
            Nref=Nref,
            Ngen=Ngen,
        )
