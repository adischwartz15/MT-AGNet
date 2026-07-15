"""Non-biometric image-quality diagnostics.

Computes resolution, brightness, contrast, and blur statistics and turns
them into human-readable warnings -- "image may be too dark", "may be
blurry", etc. -- describing the raw image itself, independent of any
model prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image


@dataclass
class QualityDiagnostics:
    width: int
    height: int
    brightness: float
    contrast: float
    blur_score: float
    file_type: str
    file_size_bytes: int
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "blur_score": self.blur_score,
            "file_type": self.file_type,
            "file_size_bytes": self.file_size_bytes,
            "warnings": self.warnings,
        }


def compute_quality_diagnostics(
    image: Image.Image,
    file_type: str,
    file_size_bytes: int,
    min_resolution: int = 96,
    blur_threshold: float = 80.0,
    dark_threshold: float = 0.25,
    bright_threshold: float = 0.88,
) -> QualityDiagnostics:
    """Compute quality diagnostics and warnings for an uploaded image."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    gray = np.asarray(rgb.convert("L"), dtype=np.float64)

    brightness = float(gray.mean() / 255.0)
    contrast = float(gray.std() / 255.0)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    warnings: list[str] = []
    if width < min_resolution or height < min_resolution:
        warnings.append("Low resolution: image is smaller than the recommended minimum size.")
    if blur_score < blur_threshold:
        warnings.append("Image may be blurry: low edge-sharpness score detected.")
    if brightness < dark_threshold:
        warnings.append("Image may be too dark.")
    if brightness > bright_threshold:
        warnings.append("Image may be overexposed.")

    return QualityDiagnostics(
        width=width, height=height, brightness=brightness, contrast=contrast,
        blur_score=blur_score, file_type=file_type, file_size_bytes=file_size_bytes,
        warnings=warnings,
    )
