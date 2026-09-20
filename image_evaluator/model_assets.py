"""Model asset metadata, download disclosure, and gating primitives.

This module provides lightweight data structures and gating logic for model
assets across existing and modern metrics. It strictly prevents unauthorized
network transfers and ensures explicit user disclosure prior to downloading.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelAsset:
    """Metadata describing a downloadable model checkpoint or asset."""

    metric_id: str
    model_id: str
    source: str
    revision: str | None = None
    estimated_download_bytes: int | None = None
    install_extra: str | None = None

    def format_estimated_size(self) -> str:
        """Format the estimated download size into human-readable text."""
        if self.estimated_download_bytes is None:
            return "unknown"
        bytes_val = self.estimated_download_bytes
        if bytes_val >= 1024**3:
            return f"~{bytes_val / (1024**3):.1f} GB ({bytes_val} bytes)"
        if bytes_val >= 1024**2:
            return f"~{bytes_val / (1024**2):.1f} MB ({bytes_val} bytes)"
        if bytes_val >= 1024:
            return f"~{bytes_val / 1024:.1f} KB ({bytes_val} bytes)"
        return f"{bytes_val} bytes"


class DownloadNotAllowedError(ValueError):
    """Raised when a model asset is not cached and download is disallowed."""

    def __init__(
        self, asset: ModelAsset, custom_message: str | None = None
    ) -> None:
        self.asset = asset
        if custom_message is not None:
            super().__init__(custom_message)
            return

        size_str = asset.format_estimated_size()
        rev_part = f", revision: {asset.revision}" if asset.revision else ""
        extra_hint = (
            f" (install with: pip install "
            f"'image-evaluator[{asset.install_extra}]')"
            if asset.install_extra
            else ""
        )
        msg = (
            f"Model '{asset.model_id}' for metric '{asset.metric_id}' is not "
            f"cached locally and automatic download is not permitted. "
            f"Source: {asset.source}{rev_part}, estimated size: {size_str}. "
            f"To allow download, re-run with '--allow-download' (CLI) or "
            f"'allow_download=True' (Python){extra_hint}."
        )
        super().__init__(msg)


def format_download_disclosure(asset: ModelAsset) -> str:
    """Format explicit disclosure message before downloading weights."""
    size_str = asset.format_estimated_size()
    rev_str = f" (revision: {asset.revision})" if asset.revision else ""
    return (
        f"[image-evaluator] Downloading model '{asset.model_id}'{rev_str} "
        f"for metric '{asset.metric_id}' from {asset.source} "
        f"[estimated size: {size_str}]..."
    )


def check_asset_and_permit_download(
    asset: ModelAsset,
    is_cached: bool,
    allow_download: bool = False,
    disclosure_callback: Callable[[ModelAsset, str], Any] | None = None,
    disclosed_tracker: set[str] | None = None,
) -> bool:
    """Validate asset cache state and verify/disclose download authorization.

    Args:
        asset: Metadata description of the model asset.
        is_cached: Boolean indicating if asset is cached locally.
        allow_download: Whether download of uncached weights is authorized.
        disclosure_callback: Optional callable(asset, msg) on download.
        disclosed_tracker: Optional set of asset keys already disclosed.

    Returns:
        bool: True if already cached (no download needed), False if download
            is permitted.

    Raises:
        DownloadNotAllowedError: If asset is uncached and download disallowed.
    """
    if is_cached:
        return True

    if not allow_download:
        raise DownloadNotAllowedError(asset)

    asset_key = f"{asset.metric_id}:{asset.model_id}:{asset.revision}"
    if disclosed_tracker is None or asset_key not in disclosed_tracker:
        msg = format_download_disclosure(asset)
        if disclosure_callback is not None:
            disclosure_callback(asset, msg)
        if disclosed_tracker is not None:
            disclosed_tracker.add(asset_key)

    return False
