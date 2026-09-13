import numpy as np
import torch


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
            reference_path: Reference image file path, PIL Image, or Tensor.
            generated_path: Generated image file path, PIL Image, or Tensor.

        Returns:
            float: LPIPS distance score.

        Raises:
            ValueError: If images have mismatched spatial dimensions.
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
                "LPIPS requires identical spatial dimensions. "
                "Please align image sizes beforehand (e.g. via high-quality "
                "downsampling or super-resolution)."
            )

        ref_scaled = (2.0 * ref_tensor - 1.0).to(torch.float32)
        gen_scaled = (2.0 * gen_tensor - 1.0).to(torch.float32)

        with torch.no_grad():
            dist = self.loss_fn(ref_scaled, gen_scaled)
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
