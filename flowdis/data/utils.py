import numpy as np

# Coordinates throughout are (x, y) with x = column and y = row, and region
# rectangles are (x0, y0, x1, y1) with x1/y1 exclusive so they slice directly.


# individual minimum bounding box for single instance
def get_minimum_bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Minimal bounding box of the mask foreground, as (xmin, xmax, ymin, ymax).

    Bounds are inclusive: the object occupies columns xmin..xmax and rows
    ymin..ymax. Raises ValueError if the mask is empty, which PAIP can produce
    when compositing (e.g. the "Mj AND NOT Mk" case fully cancels).
    """
    rows, cols = mask.nonzero()  # numpy returns (row, col), i.e. (y, x)
    if rows.size == 0:
        raise ValueError("mask has no foreground pixels")

    return int(cols.min()), int(cols.max()), int(rows.min()), int(rows.max())


# blank region proposals
def get_blank_region_proposals(
    image: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """Largest rectangle adjacent to `bbox` that does not overlap it.

    Considers the four full strips beside the box and returns the largest as
    ((x0, y0, x1, y1), crop). The strips overlap each other at the corners,
    which is fine: only the largest is used.

    The crop is empty when the box spans the image in both axes, so callers
    should check `crop.size` before compositing into it.
    """
    h, w = image.shape[:2]  # [:2] so this also accepts a 2D mask
    xmin, xmax, ymin, ymax = bbox

    # bbox bounds are inclusive; step past them to get exclusive strip edges
    right, bottom = xmax + 1, ymax + 1

    regions = (
        (0, 0, xmin, h),       # left   of the box
        (0, 0, w, ymin),       # above  the box
        (right, 0, w, h),      # right  of the box
        (0, bottom, w, h),     # below  the box
    )
    # bbox bounds are inclusive and inside the image, so every area is >= 0;
    # a box flush against an edge simply gives that strip zero area
    areas = [(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in regions]

    region_coords = regions[int(np.argmax(areas))]
    x0, y0, x1, y1 = region_coords
    region_image = image[y0:y1, x0:x1]

    return region_coords, region_image

def _side_of(region_coords: tuple[int, int, int, int], h: int, w: int) -> str:
    """Which image edge the blank region is anchored to.

    Every proposal from get_blank_region_proposals is a full strip, so it spans
    the image in exactly one axis; that plus the edge it touches names the side.
    """
    x0, y0, x1, y1 = region_coords
    spans_rows = y0 == 0 and y1 == h
    spans_cols = x0 == 0 and x1 == w

    if spans_rows and spans_cols:
        raise ValueError("region covers the whole image; no side to pad against")
    if spans_rows:
        return "left" if x0 == 0 else "right"
    if spans_cols:
        return "top" if y0 == 0 else "bottom"
    raise ValueError(f"region {region_coords} is not a full strip of a {h}x{w} image")


# reflection padding
def reflection_pad(
    image: np.ndarray, mask: np.ndarray, region_coords: tuple[int, int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Enlarge the blank region by padding the side it is anchored to.

    PAIP pads the reference image *and its mask* along the side shared with
    R_max, by that region's own thickness, which doubles the area available to
    place the pairing foreground into.

    Padding is `symmetric` rather than numpy's `reflect`: the strip ends exactly
    where the object begins, and `reflect` repeats that boundary column, mirroring
    a sliver of the object into the region meant to be empty. `symmetric` is also
    what cv2.BORDER_REFLECT does.

    Returns (padded_image, padded_mask). Accepts 2D or 3D arrays for either.
    """
    h, w = image.shape[:2]
    if mask.shape[:2] != (h, w):
        raise ValueError(
            f"image and mask differ in size: {image.shape[:2]} vs {mask.shape[:2]}"
        )

    x0, y0, x1, y1 = region_coords
    side = _side_of(region_coords, h, w)
    thickness = (x1 - x0) if side in ("left", "right") else (y1 - y0)

    pad_rows = {"top": (thickness, 0), "bottom": (0, thickness)}.get(side, (0, 0))
    pad_cols = {"left": (thickness, 0), "right": (0, thickness)}.get(side, (0, 0))

    def pad(array: np.ndarray) -> np.ndarray:
        # trailing (0, 0) per extra axis, so this takes HW and HWC alike
        widths = [pad_rows, pad_cols] + [(0, 0)] * (array.ndim - 2)
        return np.pad(array, widths, mode="symmetric")

    return pad(image), pad(mask)

if __name__ == "__main__":
    # image 100 rows x 200 cols, object at rows 10..30, cols 150..190
    h, w = 100, 200
    mask = np.zeros((h, w), np.uint8)
    mask[10:31, 150:191] = 255
    image = np.zeros((h, w, 3), np.uint8)

    bbox = get_minimum_bounding_box(mask)
    print(f"bbox (xmin, xmax, ymin, ymax): {bbox}")

    coords, region = get_blank_region_proposals(image, bbox)
    x0, y0, x1, y1 = coords
    print(f"largest blank region: {coords}  area {(x1 - x0) * (y1 - y0)}")
    print(f"crop shape: {region.shape}")

    # the same region taken from the mask must contain no foreground
    _, mask_region = get_blank_region_proposals(mask, bbox)
    print(f"foreground pixels inside the blank region: {int(mask_region.sum())}")

    # PAIP: padding the shared side should double the placement region
    padded_image, padded_mask = reflection_pad(image, mask, coords)
    new_coords, _ = get_blank_region_proposals(
        padded_image, get_minimum_bounding_box(padded_mask)
    )
    nx0, ny0, nx1, ny1 = new_coords
    before = (x1 - x0) * (y1 - y0)
    after = (nx1 - nx0) * (ny1 - ny0)
    print(f"padded image: {image.shape} -> {padded_image.shape}")
    print(f"padded mask:  {mask.shape} -> {padded_mask.shape}")
    print(f"R_max area:   {before} -> {after}  ({after / before:.2f}x)")
    print("ok")
