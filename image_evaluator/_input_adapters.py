import os
from typing import Any

import numpy as np
import torch
from PIL import Image


def _numpy_to_chw_float(arr: np.ndarray) -> np.ndarray:
    """Convert 2D/3D numpy array to float32 CHW in range [0.0, 1.0].

    Supports float32 [0.0, 1.0] and uint8 [0, 255].
    For 4-channel RGBA arrays, discards the alpha channel (:3) and
    retains RGB channels.

    Args:
        arr: Input numpy array.

    Returns:
        np.ndarray: float32 array with shape (3, H, W) in [0.0, 1.0].

    Raises:
        ValueError: If dimensions or channels are unsupported.
    """
    if arr.ndim == 2:
        chw = np.repeat(arr[np.newaxis, :, :], 3, axis=0)
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
            chw = arr
        elif arr.shape[2] in (1, 3, 4) and arr.shape[0] not in (1, 3, 4):
            chw = np.transpose(arr, (2, 0, 1))
        elif arr.shape[0] in (1, 3, 4):
            chw = arr
        elif arr.shape[2] in (1, 3, 4):
            chw = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError(
                "Expected 1, 3, or 4 channels in numpy array, "
                f"got shape {arr.shape}"
            )

        if chw.shape[0] == 1:
            chw = np.repeat(chw, 3, axis=0)
        elif chw.shape[0] == 4:
            chw = chw[:3, :, :]
        elif chw.shape[0] != 3:
            raise ValueError(
                f"Unsupported channel count {chw.shape[0]} in numpy array."
            )
    else:
        raise ValueError(
            f"Unsupported numpy array dimension {arr.ndim} "
            f"(shape: {arr.shape})."
        )

    chw = chw.astype(np.float32)
    if np.issubdtype(arr.dtype, np.integer) or chw.max() > 1.0:
        chw = chw / 255.0
    return np.clip(chw, 0.0, 1.0)


def to_torch_tensor(
    image_input: Any, device: torch.device | str | None = None
) -> torch.Tensor:
    """Normalize input to a float32 torch.Tensor of shape (1, 3, H, W).

    Output range is [0.0, 1.0]. Preserves full float32 precision
    without quantization for floating-point tensors and numpy arrays.
    For 4-channel RGBA inputs, discards the alpha channel (:3) and
    retains RGB channels.

    Args:
        image_input: Path string, PIL Image, torch.Tensor, or numpy array.
        device: Target torch device.

    Returns:
        torch.Tensor: Normalized tensor with shape (1, 3, H, W) in [0.0, 1.0].

    Raises:
        FileNotFoundError: If path string does not exist.
        ValueError: If input format, batch size, or dimensions are invalid.
    """
    if isinstance(image_input, torch.Tensor):
        tensor = image_input.detach().clone()
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        elif tensor.dim() == 3:
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1).unsqueeze(0)
            elif tensor.shape[0] == 3:
                tensor = tensor.unsqueeze(0)
            elif tensor.shape[0] == 4:
                tensor = tensor[:3, :, :].unsqueeze(0)
            else:
                raise ValueError(
                    "Expected 1, 3, or 4 channels in 3D tensor (C, H, W), "
                    f"got {tensor.shape[0]}."
                )
        elif tensor.dim() == 4:
            if tensor.shape[0] != 1:
                raise ValueError(
                    "Expected single-image tensor with batch size 1, "
                    f"got shape {tuple(tensor.shape)}."
                )
            if tensor.shape[1] == 1:
                tensor = tensor.repeat(1, 3, 1, 1)
            elif tensor.shape[1] == 3:
                pass
            elif tensor.shape[1] == 4:
                tensor = tensor[:, :3, :, :]
            else:
                raise ValueError(
                    "Expected 1, 3, or 4 channels in 4D tensor (1, C, H, W), "
                    f"got {tensor.shape[1]}."
                )
        else:
            raise ValueError(
                f"Unsupported tensor dimension {tensor.dim()} "
                f"(shape: {tuple(tensor.shape)})."
            )

        tensor = tensor.to(dtype=torch.float32)
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        tensor = tensor.clamp(0.0, 1.0)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    if isinstance(image_input, np.ndarray):
        chw = _numpy_to_chw_float(image_input)
        tensor = torch.from_numpy(chw).unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    if isinstance(image_input, (str, os.PathLike)):
        path_str = str(image_input)
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Image not found: '{path_str}'")
        if os.path.isdir(path_str):
            raise ValueError(
                f"Expected image file, got directory: '{path_str}'"
            )
        with Image.open(path_str) as img:
            rgb_img = img.convert("RGB")
            arr = np.array(rgb_img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            if device is not None:
                tensor = tensor.to(device)
            return tensor

    if isinstance(image_input, Image.Image):
        rgb_img = image_input.convert("RGB")
        arr = np.array(rgb_img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    raise ValueError(
        f"Unsupported image input type: {type(image_input).__name__}. "
        "Expected str, Path, PIL.Image.Image, torch.Tensor, or numpy.ndarray."
    )

def to_pil_image(image_input: Any) -> Image.Image:
    """Normalize file path, PIL Image, Tensor, or NumPy to an RGB PIL.Image.

    Args:
        image_input: Path, os.PathLike, PIL Image, Tensor, or array.

    Returns:
        PIL.Image.Image: Converted image in RGB mode.

    Raises:
        FileNotFoundError: If path string does not exist.
        ValueError: If input format or dimensions are unsupported.
    """
    if isinstance(image_input, (str, os.PathLike)):
        path_str = str(image_input)
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Image not found: '{path_str}'")
        if os.path.isdir(path_str):
            raise ValueError(
                f"Expected image file, got directory: '{path_str}'"
            )
        with Image.open(path_str) as img:
            return img.convert("RGB")

    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    if isinstance(image_input, (torch.Tensor, np.ndarray)):
        tensor = to_torch_tensor(image_input)
        hwc = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        arr_uint8 = (hwc * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr_uint8, mode="RGB")

    raise ValueError(
        f"Unsupported image input type: {type(image_input).__name__}. "
        "Expected str, Path, PIL.Image.Image, torch.Tensor, or numpy.ndarray."
    )
