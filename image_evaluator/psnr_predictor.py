import math

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class PSNRPredictor:
    def __init__(self, device=None):
        """Initialize PSNR predictor.

        Args:
            device: Computing device ('cuda', 'cpu', or None for auto-detect).
        """
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.to_tensor = transforms.ToTensor()

    def compute_psnr_tensor(self, img1, img2, data_range=1.0):
        """Compute Peak Signal-to-Noise Ratio (PSNR) for float tensors.

        Input tensors are assumed to be in [0, 1] range.
        PSNR = 10 * log10(data_range^2 / MSE)
             = 20 * log10(data_range / sqrt(MSE))



        Args:
            img1: Tensor of shape (1, 3, H, W) or (3, H, W) in [0.0, 1.0].
            img2: Tensor of shape (1, 3, H, W) or (3, H, W) in [0.0, 1.0].
            data_range: Dynamic range of pixel values (default: 1.0).

        Returns:
            float: PSNR in dB (float('inf') if MSE == 0).
        """
        mse = torch.mean((img1 - img2) ** 2).item()
        if mse == 0.0:
            return float("inf")
        return float(10.0 * math.log10((data_range**2) / mse))

    def evaluate_psnr(self, reference_path, generated_path):
        """Evaluate PSNR between two images.

        Args:
            reference_path: Reference image file path.
            generated_path: Generated image file path.

        Returns:
            float: PSNR score in dB (float('inf') for identical images).

        Raises:
            ValueError: If images have mismatched spatial dimensions.
        """
        ref_img = Image.open(reference_path).convert("RGB")
        gen_img = Image.open(generated_path).convert("RGB")

        if ref_img.size != gen_img.size:
            raise ValueError(
                f"Image size mismatch: reference '{reference_path}' has size "
                f"{ref_img.size} (WxH), but generated '{generated_path}' "
                f"has size {gen_img.size} (WxH). "
                "PSNR requires identical spatial dimensions. "
                "Please align image sizes beforehand (e.g. via high-quality "
                "downsampling or super-resolution)."
            )

        ref_tensor = self.to_tensor(ref_img).unsqueeze(0).to(self.device)
        gen_tensor = self.to_tensor(gen_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            score = self.compute_psnr_tensor(ref_tensor, gen_tensor)
        return score

    def evaluate_folder_psnr(self, reference_folder, generated_folder):
        """Evaluate average PSNR between images in two folders.

        Stem-matched (M2-03): exact case-sensitive stems ignoring
        extension. Any unscorable pair or size mismatch fails the whole batch.

        Args:
            reference_folder: Folder path containing reference images.
            generated_folder: Folder path containing generated images.

        Returns:
            float: Average PSNR in dB.

        Raises:
            FileNotFoundError: stem sets differ or dirs are empty.
            ValueError: duplicate stems, unsupported visible files, or
                size mismatch.
            OSError: unreadable files.
        """
        from image_evaluator._stem_pairing import IMAGE_EXTS, pair_dirs

        pairs = pair_dirs(
            reference_folder, generated_folder, IMAGE_EXTS, IMAGE_EXTS
        )
        scores = []
        for ref_path, gen_path in pairs:
            score = self.evaluate_psnr(ref_path, gen_path)
            scores.append(score)
        return float(np.mean(scores))
