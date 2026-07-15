"""Manual image transforms (PIL/NumPy/PyTorch only -- no torchvision).

Kept dependency-light and dependency-explicit: everything here is a plain
function or small class operating on PIL images / NumPy arrays, so there is
no reliance on any prebuilt vision-transform library.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

IMAGENET_MEAN = (0.485, 0.456, 0.406)  # standard RGB normalization constants, not pretrained weights
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image to a CHW float tensor in [0, 1]."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def normalize(tensor: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    mean_t = torch.tensor(mean).view(-1, 1, 1)
    std_t = torch.tensor(std).view(-1, 1, 1)
    return (tensor - mean_t) / std_t


def resize(image: Image.Image, size: int, interpolation: int = Image.BILINEAR) -> Image.Image:
    return image.resize((size, size), interpolation)


def resize_and_center_crop(
    image: Image.Image, size: int, interpolation: int = Image.BILINEAR, crop_pct: float = 1.0,
) -> Image.Image:
    """Resize preserving aspect ratio, then center-crop to ``size x size``.

    Unlike a direct ``resize((size, size))`` squish, this avoids
    distorting the aspect ratio of non-square inputs (e.g. a
    portrait-oriented photo upload) before the model sees them.
    ``interpolation`` defaults to bilinear (this project's original
    behaviour); a pretrained-backbone experiment can pass its own
    resolved interpolation mode (e.g. bicubic) instead.

    ``crop_pct`` (default ``1.0``, i.e. every existing caller's behaviour is
    unchanged) implements the standard ImageNet-style "resize-then-crop"
    protocol some pretrained backbones' own preprocessing specifies: resize
    the shorter side to ``round(size / crop_pct)``, then center-crop to
    ``size``. With ``crop_pct == 1.0`` this reduces to resizing the shorter
    side directly to ``size``, identical to the pre-existing behaviour. A
    pretrained model's resolved ``crop_pct < 1.0`` (e.g. many torchvision
    configs) means the intermediate resize is *larger* than the
    final crop, matching what that backbone was actually trained/validated
    with -- using ``crop_pct=1.0`` unconditionally for such a model would
    silently feed it out-of-distribution preprocessing.
    """
    if not 0.0 < crop_pct <= 1.0:
        raise ValueError(f"crop_pct must be in (0, 1], got {crop_pct}.")
    resize_size = round(size / crop_pct)
    width, height = image.size
    if width < height:
        new_width = resize_size
        new_height = max(resize_size, round(height * resize_size / width))
    else:
        new_height = resize_size
        new_width = max(resize_size, round(width * resize_size / height))
    resized = image.resize((new_width, new_height), interpolation)
    left = (new_width - size) // 2
    top = (new_height - size) // 2
    return resized.crop((left, top, left + size, top + size))


def random_horizontal_flip(image: Image.Image, p: float = 0.5) -> Image.Image:
    if random.random() < p:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


def random_crop_resize(
    image: Image.Image, size: int, scale: tuple[float, float] = (0.8, 1.0), interpolation: int = Image.BILINEAR,
) -> Image.Image:
    width, height = image.size
    area = width * height
    for _ in range(10):
        target_area = random.uniform(*scale) * area
        aspect = random.uniform(0.9, 1.1)
        w = int(round((target_area * aspect) ** 0.5))
        h = int(round((target_area / aspect) ** 0.5))
        if w <= width and h <= height:
            x = random.randint(0, width - w)
            y = random.randint(0, height - h)
            return image.crop((x, y, x + w, y + h)).resize((size, size), interpolation)
    return resize_and_center_crop(image, size, interpolation)


def color_jitter(image: Image.Image, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2) -> Image.Image:
    if brightness:
        image = ImageEnhance.Brightness(image).enhance(1.0 + random.uniform(-brightness, brightness))
    if contrast:
        image = ImageEnhance.Contrast(image).enhance(1.0 + random.uniform(-contrast, contrast))
    if saturation:
        image = ImageEnhance.Color(image).enhance(1.0 + random.uniform(-saturation, saturation))
    return image


def random_gaussian_blur(image: Image.Image, p: float = 0.2, radius_range: tuple[float, float] = (0.1, 1.5)) -> Image.Image:
    if random.random() < p:
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(*radius_range)))
    return image


class EvalTransform:
    """Deterministic resize (aspect-ratio-preserving) + center-crop + normalize.

    Used for validation/test/inference. Uses resize_and_center_crop
    rather than a direct squish-to-square, so a non-square input (e.g. an
    arbitrary uploaded photo) is not distorted before the model sees it.

    ``mean``/``std``/``interpolation`` default to this project's original
    values (``IMAGENET_MEAN``/``IMAGENET_STD``/bilinear), so every existing
    caller that only passes ``image_size`` is unaffected. A pretrained
    backbone experiment (e.g. pretrained-ResNet via torchvision) can
    instead pass the exact size/mean/std/interpolation resolved from that
    backbone's own pretrained-model config, rather than this project's
    defaults.
    """

    def __init__(
        self,
        image_size: int = 128,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
        interpolation: int = Image.BILINEAR,
        crop_pct: float = 1.0,
    ) -> None:
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.interpolation = interpolation
        self.crop_pct = crop_pct

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = resize_and_center_crop(image, self.image_size, self.interpolation, self.crop_pct)
        tensor = to_tensor(image)
        return normalize(tensor, self.mean, self.std)


class TrainTransform:
    """Moderate augmentation pipeline used for supervised multi-task training.

    See :class:`EvalTransform` for the ``mean``/``std``/``interpolation``/
    ``crop_pct`` default-preserving rationale. ``crop_pct`` only affects the
    initial resize target of the random-crop-resize augmentation's fallback
    path (see :func:`resize_and_center_crop`); the primary augmented path
    (:func:`random_crop_resize`) already samples its own random crop region.
    """

    def __init__(
        self,
        image_size: int = 128,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
        interpolation: int = Image.BILINEAR,
        crop_pct: float = 1.0,
    ) -> None:
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.interpolation = interpolation
        self.crop_pct = crop_pct

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = random_crop_resize(image, self.image_size, interpolation=self.interpolation)
        image = random_horizontal_flip(image)
        image = color_jitter(image)
        tensor = to_tensor(image)
        return normalize(tensor, self.mean, self.std)


def resolve_eval_transform(model, config: dict | None = None) -> EvalTransform:
    """The single place every evaluation/calibration/robustness/feature-
    extraction/inference/prediction-export code path resolves its
    deterministic preprocessing from.

    If ``model`` declares its own ``build_transforms()`` (currently the
    pretrained-torchvision wrapper -- see ``src/models/pretrained_resnet.py``),
    returns exactly that model's own resolved eval transform (its own input size, mean,
    std, interpolation, and crop_pct) -- **never** this project's 128px/
    IMAGENET-constant default for such a model. Every core (from-scratch)
    model has no such method, so this falls back to
    ``EvalTransform(config["dataset"]["image_size"])`` for them, identical
    to previous behaviour.

    Centralizing this (rather than each script re-implementing
    ``hasattr(model, "build_transforms")``) is what makes it structurally
    impossible for calibration/robustness/k-NN/prediction-export to
    accidentally evaluate a pretrained-ResNet checkpoint with the
    wrong preprocessing -- previously several of these scripts hardcoded
    ``EvalTransform(config["dataset"]["image_size"])`` unconditionally,
    which is silently wrong (wrong resolution *and* wrong normalization
    constants) for such a checkpoint.
    """
    if hasattr(model, "build_transforms"):
        _, eval_transform = model.build_transforms()
        return eval_transform
    image_size = config["dataset"]["image_size"] if config else 128
    return EvalTransform(image_size)


def resolve_train_transform(model, config: dict | None = None) -> TrainTransform:
    """Train-time counterpart of :func:`resolve_eval_transform` -- same
    model-declares-its-own-transforms resolution, returning the train half
    of ``model.build_transforms()`` when present."""
    if hasattr(model, "build_transforms"):
        train_transform, _ = model.build_transforms()
        return train_transform
    image_size = config["dataset"]["image_size"] if config else 128
    return TrainTransform(image_size)


class SimCLRTransform:
    """Strong augmentation pipeline for SimCLR-style self-supervised pretraining.

    Produces two independently augmented views of the same image.
    """

    def __init__(self, image_size: int = 128) -> None:
        self.image_size = image_size

    def _view(self, image: Image.Image) -> torch.Tensor:
        image = random_crop_resize(image, self.image_size, scale=(0.5, 1.0))
        image = random_horizontal_flip(image)
        image = color_jitter(image, brightness=0.4, contrast=0.4, saturation=0.4)
        image = random_gaussian_blur(image, p=0.5)
        tensor = to_tensor(image)
        return normalize(tensor)

    def __call__(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        return self._view(image), self._view(image)
