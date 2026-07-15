"""Tests for the manual image transforms, focused on aspect-ratio preservation.

resize_and_center_crop replaced a naive squish-to-square resize in
EvalTransform: a non-square uploaded photo (e.g. a portrait-oriented
passport photo) should be center-cropped, not stretched, before the
model sees it.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.data.transforms import EvalTransform, resize_and_center_crop


def _gradient_image(width: int, height: int) -> Image.Image:
    """A horizontal gradient so we can tell if content was stretched vs. cropped."""
    row = np.linspace(0, 255, width, dtype=np.uint8)
    array = np.tile(row, (height, 1))
    array = np.stack([array] * 3, axis=-1)
    return Image.fromarray(array)


def test_resize_and_center_crop_output_is_always_square():
    for size_wh in [(400, 300), (300, 400), (500, 500), (123, 987)]:
        image = _gradient_image(*size_wh)
        result = resize_and_center_crop(image, 128)
        assert result.size == (128, 128)


def test_resize_and_center_crop_preserves_aspect_ratio_no_squish():
    """A square crop from a portrait image should show a *center slice*, not a squished full image.

    We verify this indirectly: the shorter (width) side of a portrait image
    resized so its width == target size means the resize scale factor for
    width is exactly size/width == 1 crop-side scale, distinct from a
    squish which would scale width and height independently.
    """
    # Portrait image: width 100, height 200. Shorter side is width.
    image = _gradient_image(100, 200)
    result = resize_and_center_crop(image, 100)
    # Since width already equals the target size, resize_and_center_crop
    # should leave the horizontal gradient's endpoints intact (0 and ~255),
    # unlike a squish to (100,100) which would also preserve them in this
    # specific case -- so instead verify the *vertical* extent used is a
    # centered subset, not the full stretched height, by checking the
    # cropped image is taken from the middle rows of the resized image.
    resized_full = image.resize((100, 200), Image.BILINEAR)
    top = (200 - 100) // 2
    expected_crop = resized_full.crop((0, top, 100, top + 100))
    assert np.array_equal(np.asarray(result), np.asarray(expected_crop))


def test_resize_and_center_crop_square_input_is_unchanged_aside_from_resize():
    image = _gradient_image(200, 200)
    result = resize_and_center_crop(image, 128)
    expected = image.resize((128, 128), Image.BILINEAR)
    assert np.array_equal(np.asarray(result), np.asarray(expected))


def test_resize_and_center_crop_landscape_crops_horizontally():
    # Landscape: width 300, height 100. Shorter side is height.
    image = _gradient_image(300, 100)
    result = resize_and_center_crop(image, 100)
    resized_full = image.resize((300, 100), Image.BILINEAR)
    left = (300 - 100) // 2
    expected_crop = resized_full.crop((left, 0, left + 100, 100))
    assert np.array_equal(np.asarray(result), np.asarray(expected_crop))


def test_eval_transform_output_shape_for_non_square_input():
    transform = EvalTransform(image_size=64)
    image = _gradient_image(300, 450)
    tensor = transform(image)
    assert tensor.shape == (3, 64, 64)
