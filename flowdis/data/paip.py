from dataclasses import dataclass
from typing import Sequence
import random

import cv2
import numpy as np
import torch

from flowdis.data.utils import (
    get_blank_region_proposals,
    get_minimum_bounding_box,
    reflection_pad,
)

# (mask option, prompt option) from the paper, sampled uniformly
VARIANTS = ("reference", "pairing", "both")


@dataclass
class PAIPSample:
    """One (image, mask, prompt) triplet going into PAIP.

    image: HxW or HxWxC, any numeric dtype (normalised floats are fine --
           blending and resizing are affine, so they commute with normalisation)
    mask:  HxW in [0, 1]
    """

    image: np.ndarray
    mask: np.ndarray
    prompt: str = ""


@dataclass
class PAIPResult:
    image: np.ndarray
    mask: np.ndarray
    prompt: str
    variant: str
    mask_reference: np.ndarray  # M-hat_j, the padded reference mask
    mask_pairing: np.ndarray  # M-hat_k, the placed pairing mask
    mixed: bool = True  # False when the sample was passed through unchanged


def _as_float_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.dtype == np.uint8:
        return mask.astype(np.float32) / 255.0
    return mask.astype(np.float32)


def crop_foreground(
    image: np.ndarray, mask: np.ndarray, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Cut the object out of `image` using its mask's bounding box."""
    xmin, xmax, ymin, ymax = get_minimum_bounding_box((mask > threshold).astype(np.uint8))
    return image[ymin : ymax + 1, xmin : xmax + 1], mask[ymin : ymax + 1, xmin : xmax + 1]


def fit_within(
    image: np.ndarray, mask: np.ndarray, width: int, height: int, scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Resize to fit inside width x height, preserving the aspect ratio."""
    h, w = mask.shape[:2]
    factor = min(width / w, height / h) * scale
    new_w = min(width, max(1, int(round(w * factor))))
    new_h = min(height, max(1, int(round(h * factor))))
    interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
    return (
        cv2.resize(image, (new_w, new_h), interpolation=interp),
        cv2.resize(mask, (new_w, new_h), interpolation=interp),
    )


def place_with_minimal_overlap(
    reference_mask: np.ndarray,
    foreground_mask: np.ndarray,
    region: tuple[int, int, int, int],
    rng: random.Random,
    candidates: int = 8,
) -> tuple[int, int]:
    """Sample positions inside `region` and keep the one covering the least of
    the reference object.

    With a region from get_blank_region_proposals the overlap is effectively
    zero already: the region never intersects the reference box, and padding
    grows away from it. Measured over 200 real DIS-TR pairs, 195 had exactly
    zero and the rest a sub-pixel sliver of anti-aliased mask edge (max 7.3px).

    A known consequence: the paper's "Mj AND NOT Mk" option therefore equals Mj
    in practice, so that variant carries no occlusion signal. The paper writing
    it as a subtraction suggests it expects some real overlap, i.e. "minimal"
    may mean small-but-nonzero rather than none. Accepted as-is for now; the
    sampling below is what would find a good spot if a caller ever passes a
    region that does intersect the reference.
    """
    x0, y0, x1, y1 = region
    fh, fw = foreground_mask.shape[:2]
    max_dx, max_dy = max(0, (x1 - x0) - fw), max(0, (y1 - y0) - fh)

    best_overlap, best_xy = None, (x0, y0)
    for _ in range(max(1, candidates)):
        x = x0 + rng.randint(0, max_dx)
        y = y0 + rng.randint(0, max_dy)
        window = reference_mask[y : y + fh, x : x + fw]
        overlap = float((window * foreground_mask[: window.shape[0], : window.shape[1]]).sum())
        if best_overlap is None or overlap < best_overlap:
            best_overlap, best_xy = overlap, (x, y)
        if best_overlap == 0.0:
            break
    return best_xy


def alpha_blend(
    canvas: np.ndarray, foreground: np.ndarray, alpha: np.ndarray, x: int, y: int
) -> np.ndarray:
    """Composite `foreground` onto `canvas` at (x, y) weighted by `alpha`."""
    fh, fw = alpha.shape[:2]
    out = canvas.astype(np.float32, copy=True)
    roi = out[y : y + fh, x : x + fw]
    a = alpha[: roi.shape[0], : roi.shape[1]]
    if roi.ndim == 3:
        a = a[..., None]
    fg = foreground[: roi.shape[0], : roi.shape[1]].astype(np.float32)
    out[y : y + fh, x : x + fw] = roi * (1.0 - a) + fg * a

    if np.issubdtype(canvas.dtype, np.integer):
        info = np.iinfo(canvas.dtype)
        return np.clip(out, info.min, info.max).astype(canvas.dtype)
    return out.astype(canvas.dtype)


def combine_masks(
    mask_reference: np.ndarray,
    mask_pairing: np.ndarray,
    prompt_reference: str,
    prompt_pairing: str,
    variant: str,
) -> tuple[np.ndarray, str]:
    """The paper's three (mask, prompt) options.

    AND is pixel-wise multiplication and OR pixel-wise maximum, so these also
    behave sensibly on the soft edges that resizing produces.
    """
    mask_reference = np.clip(mask_reference, 0.0, 1.0)
    mask_pairing = np.clip(mask_pairing, 0.0, 1.0)

    if variant == "reference":
        # the pairing object is composited on top, so subtract what it hides
        return mask_reference * (1.0 - mask_pairing), prompt_reference
    if variant == "pairing":
        return mask_pairing, prompt_pairing
    if variant == "both":
        return np.maximum(mask_reference, mask_pairing), (
            f"{prompt_reference} and {prompt_pairing}"
            if prompt_reference and prompt_pairing
            else prompt_reference or prompt_pairing
        )
    raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")


def paip_mix(
    reference: PAIPSample,
    pairing: PAIPSample,
    rng: random.Random | None = None,
    scale_range: tuple[float, float] = (1.0, 1.0),
    placement_candidates: int = 8,
    variant: str | None = None,
    threshold: float = 0.5,
) -> PAIPResult:
    """Build one mixed training sample from a reference and a pairing triplet.

    `scale_range` multiplies the fit-to-area scale; the default (1, 1) is the
    paper's "resize to fit within the placement area". Widening it shrinks the
    pasted object and gives the random placement more freedom.

    Falls back to the unmixed reference when there is nowhere to place the
    object -- a full-frame reference object leaves no blank region, and an
    empty pairing mask has nothing to crop.
    """
    rng = rng or random.Random()
    ref_mask = _as_float_mask(reference.mask)
    pair_mask = _as_float_mask(pairing.mask)

    def passthrough() -> PAIPResult:
        return PAIPResult(
            image=reference.image,
            mask=ref_mask,
            prompt=reference.prompt,
            variant="reference",
            mask_reference=ref_mask,
            mask_pairing=np.zeros_like(ref_mask),
            mixed=False,
        )

    if (ref_mask > threshold).sum() == 0 or (pair_mask > threshold).sum() == 0:
        return passthrough()

    # steps 2-4: box, blank region, pad the shared side to double it
    bbox = get_minimum_bounding_box((ref_mask > threshold).astype(np.uint8))
    region, _ = get_blank_region_proposals(reference.image, bbox)
    x0, y0, x1, y1 = region
    if (x1 - x0) * (y1 - y0) == 0:
        return passthrough()

    padded_image, padded_mask = reflection_pad(reference.image, ref_mask, region)
    placement, _ = get_blank_region_proposals(
        padded_image, get_minimum_bounding_box((padded_mask > threshold).astype(np.uint8))
    )
    px0, py0, px1, py1 = placement
    if (px1 - px0) * (py1 - py0) == 0:
        return passthrough()

    # step 5: crop, resize to fit, place, blend
    fg_image, fg_mask = crop_foreground(pairing.image, pair_mask, threshold)
    fg_image, fg_mask = fit_within(
        fg_image, fg_mask, px1 - px0, py1 - py0, rng.uniform(*scale_range)
    )
    x, y = place_with_minimal_overlap(
        padded_mask, fg_mask, placement, rng, placement_candidates
    )
    image_mix = alpha_blend(padded_image, fg_image, fg_mask, x, y)

    mask_pairing = np.zeros_like(padded_mask)
    fh, fw = fg_mask.shape[:2]
    mask_pairing[y : y + fh, x : x + fw] = fg_mask[
        : mask_pairing.shape[0] - y, : mask_pairing.shape[1] - x
    ]

    # steps 6-7
    variant = variant or rng.choice(VARIANTS)
    mask_mix, prompt_mix = combine_masks(
        padded_mask, mask_pairing, reference.prompt, pairing.prompt, variant
    )
    return PAIPResult(
        image=image_mix,
        mask=mask_mix,
        prompt=prompt_mix,
        variant=variant,
        mask_reference=padded_mask,
        mask_pairing=mask_pairing,
    )


def resize_hw(
    image: np.ndarray, mask: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resize a mixed (or passthrough) pair to the model resolution.

    Downscales with INTER_AREA and upscales with INTER_CUBIC. Native DIS images
    are often larger than 1024, so the training path is almost always a downsample
    after PAIP.
    """
    if image.shape[0] == height and image.shape[1] == width:
        return image, np.clip(mask, 0.0, 1.0)
    shrinking = height < image.shape[0] or width < image.shape[1]
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return (
        cv2.resize(image, (width, height), interpolation=interp),
        np.clip(cv2.resize(mask, (width, height), interpolation=interp), 0.0, 1.0),
    )


def _to_chw(image: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1),
        torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0),
    )


def paip_collate(
    batch: Sequence[dict],
    rng: random.Random | None = None,
    probability: float = 1.0,
    scale_range: tuple[float, float] = (1.0, 1.0),
    resolution: int | None = None,
) -> dict:
    """torch collate_fn applying PAIP across a batch of DIS5KDataset samples.

    Each sample is paired with another drawn from the same batch, as the paper
    specifies. Mixing runs at the sample's native resolution; the composite is
    then resized once to `resolution` (or back to the reference size when
    `resolution` is None) so the batch can stack.

    A batch of one is returned unmixed: there is no other sample to pair with.
    """
    rng = rng or random.Random()
    images, masks, prompts, names, variants = [], [], [], [], []

    for index, sample in enumerate(batch):
        image_t, mask_t = sample["image"], sample["mask"]
        if resolution is None:
            target_h, target_w = image_t.shape[-2:]
        else:
            target_h = target_w = resolution

        if len(batch) < 2 or rng.random() >= probability:
            image, mask = resize_hw(
                image_t.permute(1, 2, 0).numpy(),
                mask_t[0].numpy(),
                target_w,
                target_h,
            )
            image_t, mask_t = _to_chw(image, mask)
            images.append(image_t)
            masks.append(mask_t)
            prompts.append(sample.get("prompt", ""))
            names.append(sample.get("name", ""))
            variants.append("reference")
            continue

        other = batch[rng.choice([i for i in range(len(batch)) if i != index])]
        result = paip_mix(
            PAIPSample(
                image_t.permute(1, 2, 0).numpy(),
                mask_t[0].numpy(),
                sample.get("prompt", ""),
            ),
            PAIPSample(
                other["image"].permute(1, 2, 0).numpy(),
                other["mask"][0].numpy(),
                other.get("prompt", ""),
            ),
            rng=rng,
            scale_range=scale_range,
        )

        image, mask = resize_hw(result.image, result.mask, target_w, target_h)
        image_t, mask_t = _to_chw(image, mask)
        images.append(image_t)
        masks.append(mask_t)
        prompts.append(result.prompt)
        names.append(sample.get("name", ""))
        variants.append(result.variant)

    return {
        "image": torch.stack(images),
        "mask": torch.stack(masks),
        "prompt": prompts,
        "name": names,
        "variant": variants,
    }
