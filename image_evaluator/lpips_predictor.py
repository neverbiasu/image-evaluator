import numpy as np
import torch
from PIL import Image


class LPIPSPredictor:
    def __init__(self, net="alex", version="0.1", device=None):
        """Initialize LPIPS distance predictor using official lpips package.

        Args:
            net: Backbone network ('alex', 'vgg', 'squeeze').
            version: LPIPS model version ('0.1'). Default is '0.1'.
            device: Computing device ('cuda', 'cpu', or None for auto-detect).
        """
        import lpips

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.loss_fn = lpips.LPIPS(net=net, version=version)
        self.loss_fn.to(self.device)
        self.loss_fn.eval()

    def evaluate_lpips(self, reference_path, generated_path):
        """Evaluate LPIPS perceptual distance between two images.

        Args:
            reference_path: Reference image file path.
            generated_path: Generated image file path.

        Returns:
            float: LPIPS distance score.

        Raises:
            ValueError: If images have mismatched spatial dimensions.
        """
        import lpips

        ref_img = Image.open(reference_path).convert("RGB")
        gen_img = Image.open(generated_path).convert("RGB")

        if ref_img.size != gen_img.size:
            raise ValueError(
                f"Image size mismatch: reference '{reference_path}' has size "
                f"{ref_img.size} (WxH), but generated '{generated_path}' "
                f"has size {gen_img.size} (WxH). "
                "LPIPS requires identical spatial dimensions. "
                "Please align image sizes beforehand (e.g. via high-quality "
                "downsampling or super-resolution)."
            )

        ref_tensor = (
            lpips.im2tensor(np.array(ref_img))
            .to(self.device)
            .to(torch.float32)
        )
        gen_tensor = (
            lpips.im2tensor(np.array(gen_img))
            .to(self.device)
            .to(torch.float32)
        )

        with torch.no_grad():
            dist = self.loss_fn(ref_tensor, gen_tensor)
        return float(dist.item())

    def evaluate_folder_lpips(self, reference_folder, generated_folder):
        """Evaluate average LPIPS distance between images in two folders.

        Stem-matched (M2-03): exact case-sensitive stems ignoring
        extension. Any unscorable pair or size mismatch fails the whole batch.

        Args:
            reference_folder: Folder path containing reference images.
            generated_folder: Folder path containing generated images.

        Returns:
            float: Average LPIPS distance.

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
        distances = []
        for ref_path, gen_path in pairs:
            dist = self.evaluate_lpips(ref_path, gen_path)
            distances.append(dist)
        return float(np.mean(distances))
