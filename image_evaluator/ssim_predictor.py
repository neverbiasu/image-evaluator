import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


def _gaussian_window(window_size=11, sigma=1.5, channels=3):
    """Create a 2D Gaussian filter kernel for multi-channel convolution."""
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    w = g.unsqueeze(1) * g.unsqueeze(0)
    w = w.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    return w


class SSIMPredictor:
    def __init__(self, window_size=11, sigma=1.5, device=None):
        """Initialize SSIM predictor based on Wang et al. (2004).

        Args:
            window_size: Gaussian window size (default: 11).
            sigma: Standard deviation of Gaussian window (default: 1.5).
            device: Computing device ('cuda', 'cpu', or None for auto-detect).
        """
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.window_size = window_size
        self.sigma = sigma
        self.pad = window_size // 2
        self.window = _gaussian_window(window_size, sigma, channels=3).to(
            self.device
        )
        self.to_tensor = transforms.ToTensor()

    def compute_ssim_tensor(self, img1, img2, data_range=1.0):
        """Compute Wang et al. (2004) SSIM for float tensors in [0, 1].

        Args:
            img1: Tensor of shape (1, 3, H, W) in [0.0, 1.0].
            img2: Tensor of shape (1, 3, H, W) in [0.0, 1.0].
            data_range: Dynamic range of pixel values (default: 1.0).

        Returns:
            float: Channel-averaged SSIM score.

        Raises:
            ValueError: If tensor shapes do not match or if spatial dimensions
                are smaller than window_size.
        """
        if img1.shape != img2.shape:
            raise ValueError(
                f"Tensor shape mismatch: img1 shape {tuple(img1.shape)} != "
                f"img2 shape {tuple(img2.shape)}."
            )

        h, w = img1.shape[-2:]
        if h < self.window_size or w < self.window_size:
            raise ValueError(
                f"Image spatial dimensions ({w}x{h}, WxH) are smaller than "
                f"SSIM window_size ({self.window_size}x{self.window_size}). "
                f"Both width and height must be at least {self.window_size}."
            )

        channels = img1.shape[1]
        window = self.window.to(dtype=img1.dtype, device=img1.device)

        k1, k2 = 0.01, 0.03
        c1 = (k1 * data_range) ** 2
        c2 = (k2 * data_range) ** 2

        pad = self.pad
        p1 = F.pad(img1, (pad, pad, pad, pad), mode="reflect")
        p2 = F.pad(img2, (pad, pad, pad, pad), mode="reflect")

        mu1 = F.conv2d(p1, window, groups=channels)
        mu2 = F.conv2d(p2, window, groups=channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(p1 * p1, window, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(p2 * p2, window, groups=channels) - mu2_sq
        sigma12 = F.conv2d(p1 * p2, window, groups=channels) - mu1_mu2

        num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        ssim_map = num / den

        # Crop border region to match Wang et al. 2004 & skimage valid region
        ssim_cropped = ssim_map[:, :, pad:-pad, pad:-pad]
        return float(ssim_cropped.mean().item())

    def evaluate_ssim(self, reference_path, generated_path):
        """Evaluate SSIM similarity between two images.

        Args:
            reference_path: Reference image file path, PIL Image, or Tensor.
            generated_path: Generated image file path, PIL Image, or Tensor.

        Returns:
            float: SSIM score in [-1.0, 1.0] (1.0 for identical images).

        Raises:
            ValueError: If images have mismatched spatial dimensions or if
                dimensions are smaller than window_size.
        """
        import os

        from image_evaluator._input_adapters import to_torch_tensor

        ref_tensor = to_torch_tensor(reference_path, device=self.device)
        gen_tensor = to_torch_tensor(generated_path, device=self.device)

        ref_h, ref_w = ref_tensor.shape[-2:]
        gen_h, gen_w = gen_tensor.shape[-2:]

        if (ref_h, ref_w) != (gen_h, gen_w):
            ref_name = (
                str(reference_path)
                if isinstance(reference_path, (str, os.PathLike))
                else "<in-memory>"
            )
            gen_name = (
                str(generated_path)
                if isinstance(generated_path, (str, os.PathLike))
                else "<in-memory>"
            )
            raise ValueError(
                f"Image size mismatch: reference '{ref_name}' has size "
                f"({ref_w}, {ref_h}) (WxH), but generated '{gen_name}' "
                f"has size ({gen_w}, {gen_h}) (WxH). "
                "SSIM requires identical spatial dimensions. "
                "Please align image sizes beforehand (e.g. via high-quality "
                "downsampling or super-resolution)."
            )

        if ref_w < self.window_size or ref_h < self.window_size:
            raise ValueError(
                f"Image dimensions ({ref_w}, {ref_h}) (WxH) are smaller than "
                f"SSIM window_size ({self.window_size}x{self.window_size}). "
                f"Both width and height must be at least {self.window_size}."
            )

        with torch.no_grad():
            score = self.compute_ssim_tensor(ref_tensor, gen_tensor)
        return score

    def evaluate_folder_ssim(self, reference_folder, generated_folder):
        """Evaluate average SSIM between images in two folders.

        Stem-matched (M2-03): exact case-sensitive stems ignoring
        extension. Any unscorable pair or size mismatch fails the whole batch.

        Args:
            reference_folder: Folder path containing reference images.
            generated_folder: Folder path containing generated images.

        Returns:
            float: Average SSIM similarity.

        Raises:
            FileNotFoundError: stem sets differ or dirs are empty.
            ValueError: duplicate stems, unsupported visible files,
                size mismatch, or image dimensions smaller than window_size.
            OSError: unreadable files.
        """
        from image_evaluator._stem_pairing import IMAGE_EXTS, pair_dirs

        pairs = pair_dirs(
            reference_folder, generated_folder, IMAGE_EXTS, IMAGE_EXTS
        )
        scores = []
        for ref_path, gen_path in pairs:
            score = self.evaluate_ssim(ref_path, gen_path)
            scores.append(score)
        return float(np.mean(scores))
