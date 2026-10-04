"""Tests for the PAIP geometry helpers in flowdis.data.utils.

The invariants being pinned down come from the FlowDIS paper's Position-Aware
Instance Pairing: the blank region is the largest rectangle adjacent to the
object's bounding box without overlapping it, and padding the side it shares
with the image doubles that region.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flowdis.data.utils import (  # noqa: E402
    get_blank_region_proposals,
    get_minimum_bounding_box,
    reflection_pad,
)

DIS5K_ROOT = Path("/home/ubuntu/jeremy/dataset/DIS5K_extracted")


def make(h: int, w: int, rows: slice, cols: slice) -> tuple[np.ndarray, np.ndarray]:
    """A blank image plus a mask with one rectangular object."""
    mask = np.zeros((h, w), np.uint8)
    mask[rows, cols] = 1
    return np.zeros((h, w, 3), np.uint8), mask


def area(region: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = region
    return (x1 - x0) * (y1 - y0)


# --------------------------------------------------------------------------
# get_minimum_bounding_box
# --------------------------------------------------------------------------


def test_bbox_is_x_then_y_and_inclusive():
    """Returns (xmin, xmax, ymin, ymax) with x=column, bounds inclusive."""
    _, mask = make(100, 200, slice(10, 31), slice(150, 191))
    assert get_minimum_bounding_box(mask) == (150, 190, 10, 30)


def test_bbox_does_not_transpose_rows_and_columns():
    """A regression guard: mask.nonzero() yields (row, col), not (x, y)."""
    _, mask = make(100, 200, slice(0, 5), slice(100, 180))
    xmin, xmax, ymin, ymax = get_minimum_bounding_box(mask)
    assert (xmin, xmax) == (100, 179), "x bounds must come from columns"
    assert (ymin, ymax) == (0, 4), "y bounds must come from rows"


def test_bbox_single_pixel():
    _, mask = make(50, 50, slice(7, 8), slice(23, 24))
    assert get_minimum_bounding_box(mask) == (23, 23, 7, 7)


def test_bbox_full_frame():
    _, mask = make(40, 60, slice(None), slice(None))
    assert get_minimum_bounding_box(mask) == (0, 59, 0, 39)


def test_bbox_ignores_holes():
    """The box is the extent, so an interior hole must not shrink it."""
    _, mask = make(60, 60, slice(10, 51), slice(10, 51))
    mask[20:40, 20:40] = 0
    assert get_minimum_bounding_box(mask) == (10, 50, 10, 50)


def test_bbox_empty_mask_raises():
    with pytest.raises(ValueError, match="no foreground"):
        get_minimum_bounding_box(np.zeros((10, 10), np.uint8))


@pytest.mark.parametrize("value", [1, 255])
def test_bbox_indifferent_to_foreground_value(value):
    mask = np.zeros((30, 30), np.uint8)
    mask[5:10, 5:10] = value
    assert get_minimum_bounding_box(mask) == (5, 9, 5, 9)


def test_bbox_returns_plain_ints():
    """numpy scalars leak into slicing and f-strings awkwardly; keep them int."""
    _, mask = make(20, 20, slice(2, 5), slice(3, 6))
    assert all(type(v) is int for v in get_minimum_bounding_box(mask))


# --------------------------------------------------------------------------
# get_blank_region_proposals
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "h,w,rows,cols,expected",
    [
        (100, 200, slice(10, 31), slice(150, 191), (0, 0, 150, 100)),    # left
        (100, 200, slice(10, 31), slice(5, 46), (46, 0, 200, 100)),      # right
        (200, 100, slice(150, 191), slice(10, 31), (0, 0, 100, 150)),    # top
        (200, 100, slice(5, 46), slice(10, 31), (0, 46, 100, 200)),      # bottom
    ],
)
def test_picks_the_largest_strip(h, w, rows, cols, expected):
    image, mask = make(h, w, rows, cols)
    coords, _ = get_blank_region_proposals(image, get_minimum_bounding_box(mask))
    assert coords == expected


def test_region_excludes_the_object():
    """The returned crop is the placement area, so it must be free of foreground."""
    _, mask = make(100, 200, slice(10, 31), slice(150, 191))
    _, crop = get_blank_region_proposals(mask, get_minimum_bounding_box(mask))
    assert crop.size > 0, "crop must not be empty"
    assert crop.sum() == 0


def test_crop_matches_the_region_rectangle():
    """Guards the slice order: image[y0:y1, x0:x1], not image[x0:y0, x1:y1].

    Without the shape check an inverted slice yields an empty array, and every
    "no foreground in the crop" assertion passes vacuously.
    """
    image, mask = make(100, 200, slice(10, 31), slice(150, 191))
    coords, crop = get_blank_region_proposals(image, get_minimum_bounding_box(mask))
    x0, y0, x1, y1 = coords
    assert crop.size > 0
    assert crop.shape[:2] == (y1 - y0, x1 - x0)


def test_crop_returns_the_right_pixels():
    """Shape alone can coincide; check the crop is the actual region content."""
    h, w = 40, 60
    image = np.arange(h * w, dtype=np.int32).reshape(h, w)
    mask = np.zeros((h, w), np.uint8)
    mask[10:20, 50:60] = 1  # object on the right -> left strip wins
    coords, crop = get_blank_region_proposals(image, get_minimum_bounding_box(mask))
    x0, y0, x1, y1 = coords
    np.testing.assert_array_equal(crop, image[y0:y1, x0:x1])
    assert crop.shape == (h, 50)


def test_region_is_off_by_one_safe_on_the_far_side():
    """bbox bounds are inclusive, so the right strip must start at xmax + 1."""
    _, mask = make(50, 50, slice(10, 21), slice(5, 16))
    bbox = get_minimum_bounding_box(mask)
    coords, crop = get_blank_region_proposals(mask, bbox)
    assert coords[0] == bbox[1] + 1
    assert crop.sum() == 0


def test_accepts_2d_and_3d_arrays():
    image, mask = make(50, 50, slice(20, 30), slice(20, 30))
    bbox = get_minimum_bounding_box(mask)
    coords_3d, crop_3d = get_blank_region_proposals(image, bbox)
    coords_2d, crop_2d = get_blank_region_proposals(mask, bbox)
    assert coords_3d == coords_2d
    assert crop_3d.size > 0 and crop_2d.size > 0
    assert crop_3d.shape[:2] == crop_2d.shape


def test_full_frame_object_yields_empty_region():
    image, mask = make(50, 50, slice(None), slice(None))
    _, crop = get_blank_region_proposals(image, get_minimum_bounding_box(mask))
    assert crop.size == 0


def test_object_touching_an_edge_never_scores_negative():
    """Guards the max(0, ...) clamp: a flush object must not produce a negative area."""
    image, mask = make(50, 50, slice(0, 50), slice(40, 50))
    coords, _ = get_blank_region_proposals(image, get_minimum_bounding_box(mask))
    assert area(coords) >= 0
    assert coords == (0, 0, 40, 50)


@pytest.mark.parametrize("seed", range(20))
def test_region_properties_hold_on_random_masks(seed):
    rng = random.Random(seed)
    for _ in range(50):
        h, w = rng.randint(8, 120), rng.randint(8, 120)
        r0, c0 = rng.randrange(h), rng.randrange(w)
        r1, c1 = rng.randint(r0 + 1, h), rng.randint(c0 + 1, w)
        _, mask = make(h, w, slice(r0, r1), slice(c0, c1))

        bbox = get_minimum_bounding_box(mask)
        assert bbox == (c0, c1 - 1, r0, r1 - 1)

        coords, crop = get_blank_region_proposals(mask, bbox)
        x0, y0, x1, y1 = coords
        assert 0 <= x0 <= x1 <= w and 0 <= y0 <= y1 <= h, "region out of bounds"
        assert crop.shape[:2] == (y1 - y0, x1 - x0), "crop does not match region"
        assert crop.sum() == 0, "region overlaps the object"
        # it really is the largest of the four strips
        assert area(coords) == max(c0 * h, r0 * w, (w - c1) * h, (h - r1) * w)


# --------------------------------------------------------------------------
# reflection_pad
# --------------------------------------------------------------------------

ORIENTATIONS = [
    pytest.param(100, 200, slice(10, 31), slice(150, 191), id="left"),
    pytest.param(100, 200, slice(10, 31), slice(5, 46), id="right"),
    pytest.param(200, 100, slice(150, 191), slice(10, 31), id="top"),
    pytest.param(200, 100, slice(5, 46), slice(10, 31), id="bottom"),
]


@pytest.mark.parametrize("h,w,rows,cols", ORIENTATIONS)
def test_padding_doubles_the_placement_region(h, w, rows, cols):
    """The paper's stated invariant: padding 'effectively doubles' R_max."""
    image, mask = make(h, w, rows, cols)
    coords, _ = get_blank_region_proposals(image, get_minimum_bounding_box(mask))
    padded_image, padded_mask = reflection_pad(image, mask, coords)

    new_coords, _ = get_blank_region_proposals(
        padded_image, get_minimum_bounding_box(padded_mask)
    )
    assert area(new_coords) == 2 * area(coords)


@pytest.mark.parametrize("h,w,rows,cols", ORIENTATIONS)
def test_padding_grows_exactly_one_axis(h, w, rows, cols):
    """PAIP pads one side only, never all four."""
    image, mask = make(h, w, rows, cols)
    coords, _ = get_blank_region_proposals(image, get_minimum_bounding_box(mask))
    padded_image, _ = reflection_pad(image, mask, coords)

    grew_rows = padded_image.shape[0] != h
    grew_cols = padded_image.shape[1] != w
    assert grew_rows != grew_cols, "exactly one axis should change"


@pytest.mark.parametrize("h,w,rows,cols", ORIENTATIONS)
def test_padding_keeps_image_and_mask_registered(h, w, rows, cols):
    image, mask = make(h, w, rows, cols)
    coords, _ = get_blank_region_proposals(image, get_minimum_bounding_box(mask))
    padded_image, padded_mask = reflection_pad(image, mask, coords)

    assert padded_image.shape[:2] == padded_mask.shape
    assert padded_image.shape[2] == 3, "channels must survive"
    assert padded_mask.ndim == 2


@pytest.mark.parametrize("h,w,rows,cols", ORIENTATIONS)
def test_padding_does_not_mirror_the_object_into_the_blank_area(h, w, rows, cols):
    """Why symmetric, not reflect: reflect repeats the object's boundary column."""
    image, mask = make(h, w, rows, cols)
    coords, _ = get_blank_region_proposals(image, get_minimum_bounding_box(mask))
    _, padded_mask = reflection_pad(image, mask, coords)

    x0, y0, x1, y1 = coords
    thickness = (x1 - x0) if (y0 == 0 and y1 == h) else (y1 - y0)
    if padded_mask.shape[1] != w:  # padded horizontally
        strip = padded_mask[:, :thickness] if x0 == 0 else padded_mask[:, -thickness:]
    else:
        strip = padded_mask[:thickness, :] if y0 == 0 else padded_mask[-thickness:, :]
    assert strip.sum() == 0, "object leaked into the padded region"


def test_padding_mirrors_content():
    """Sanity-check the padding really is a mirror, not zeros or edge repeat."""
    image = np.arange(6 * 4 * 1, dtype=np.uint8).reshape(6, 4, 1)
    mask = np.zeros((6, 4), np.uint8)
    mask[:, 3] = 1  # object hugs the right edge -> left strip wins
    coords, _ = get_blank_region_proposals(image, get_minimum_bounding_box(mask))
    padded_image, _ = reflection_pad(image, mask, coords)

    pad = coords[2] - coords[0]
    assert pad == 3
    # symmetric: the first pad columns mirror columns [pad-1 .. 0]
    np.testing.assert_array_equal(
        padded_image[:, :pad, 0], image[:, :pad, 0][:, ::-1]
    )


def test_padding_accepts_2d_image():
    mask = np.zeros((40, 60), np.uint8)
    mask[10:20, 45:55] = 1
    coords, _ = get_blank_region_proposals(mask, get_minimum_bounding_box(mask))
    padded_a, padded_b = reflection_pad(mask, mask, coords)
    assert padded_a.ndim == 2 and padded_a.shape == padded_b.shape


def test_padding_rejects_mismatched_mask():
    image, mask = make(40, 60, slice(10, 20), slice(45, 55))
    coords, _ = get_blank_region_proposals(image, get_minimum_bounding_box(mask))
    with pytest.raises(ValueError, match="differ in size"):
        reflection_pad(image, np.zeros((10, 10), np.uint8), coords)


def test_padding_rejects_a_non_strip_region():
    image, mask = make(40, 60, slice(10, 20), slice(45, 55))
    with pytest.raises(ValueError, match="not a full strip"):
        reflection_pad(image, mask, (5, 5, 20, 20))


def test_padding_rejects_whole_image_region():
    image, mask = make(40, 60, slice(10, 20), slice(45, 55))
    with pytest.raises(ValueError, match="whole image"):
        reflection_pad(image, mask, (0, 0, 60, 40))


def test_zero_thickness_region_is_a_noop():
    """A full-frame object leaves no strip; padding by zero must not blow up."""
    image, mask = make(30, 30, slice(None), slice(None))
    coords, _ = get_blank_region_proposals(image, get_minimum_bounding_box(mask))
    padded_image, padded_mask = reflection_pad(image, mask, coords)
    assert padded_image.shape == image.shape
    assert padded_mask.shape == mask.shape


@pytest.mark.parametrize("seed", range(20))
def test_padding_invariants_hold_on_random_masks(seed):
    rng = random.Random(1000 + seed)
    for _ in range(50):
        h, w = rng.randint(8, 120), rng.randint(8, 120)
        r0, c0 = rng.randrange(h), rng.randrange(w)
        r1, c1 = rng.randint(r0 + 1, h), rng.randint(c0 + 1, w)
        image, mask = make(h, w, slice(r0, r1), slice(c0, c1))

        coords, _ = get_blank_region_proposals(image, get_minimum_bounding_box(mask))
        if area(coords) == 0:
            continue
        padded_image, padded_mask = reflection_pad(image, mask, coords)

        assert padded_image.shape[:2] == padded_mask.shape
        new_coords, _ = get_blank_region_proposals(
            padded_image, get_minimum_bounding_box(padded_mask)
        )
        assert area(new_coords) == 2 * area(coords)


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------


@pytest.mark.skipif(not DIS5K_ROOT.is_dir(), reason="DIS5K not extracted")
def test_pipeline_on_real_dis5k_masks():
    import cv2

    gt_dir = DIS5K_ROOT / "DIS-TR" / "gt"
    names = sorted(p.name for p in gt_dir.glob("*.png"))[:25]
    assert names, "no masks found"

    for name in names:
        mask = cv2.imread(str(gt_dir / name), cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        binary = (mask > 127).astype(np.uint8)

        bbox = get_minimum_bounding_box(binary)
        coords, crop = get_blank_region_proposals(binary, bbox)
        assert crop.size == 0 or crop.sum() == 0, f"{name}: region overlaps object"
        if area(coords) == 0:
            continue

        padded_image, padded_mask = reflection_pad(image, binary, coords)
        assert padded_image.shape[:2] == padded_mask.shape
        new_coords, _ = get_blank_region_proposals(
            padded_mask, get_minimum_bounding_box(padded_mask)
        )
        assert area(new_coords) == 2 * area(coords), f"{name}: region did not double"
