"""image-evaluator: A lightweight, transparent, and reproducible
image generation evaluation toolkit.
"""

import importlib
from typing import Any

__version__ = "0.4.0"

_EXPORTS = {
    "evaluate": ("image_evaluator.core", "evaluate"),
    "SSIMPredictor": ("image_evaluator.ssim_predictor", "SSIMPredictor"),
    "PSNRPredictor": ("image_evaluator.psnr_predictor", "PSNRPredictor"),
    "LPIPSPredictor": ("image_evaluator.lpips_predictor", "LPIPSPredictor"),
    "ClipScorePredictor": (
        "image_evaluator.clip_score_predictor",
        "ClipScorePredictor",
    ),
    "LaionAIAestheticPredictor": (
        "image_evaluator.laion_ai_aesthetic_predictor",
        "LaionAIAestheticPredictor",
    ),
    "PickScorePredictor": (
        "image_evaluator.pickscore_predictor",
        "PickScorePredictor",
    ),
    "FIDPredictor": ("image_evaluator.fid_predictor", "FIDPredictor"),
    "KIDPredictor": ("image_evaluator.kid_predictor", "KIDPredictor"),
    "ArcFaceDistPredictor": (
        "image_evaluator.arcface_dist_predictor",
        "ArcFaceDistPredictor",
    ),
    "DirectionalClipPredictor": (
        "image_evaluator.directional_clip_predictor",
        "DirectionalClipPredictor",
    ),
}

__all__ = ["__version__", *list(_EXPORTS.keys())]


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = importlib.import_module(module_name)
        val = getattr(module, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
