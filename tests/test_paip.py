"""Tests for the PAIP compositing steps in flowdis.data.paip.

Invariants come from the paper: the pasted object keeps its aspect ratio and
fits the placement area, it barely overlaps the reference object, and the three
(mask, prompt) options are exactly
{Mj AND NOT Mk -> tj, Mk -> tk, Mj OR Mk -> "tj and tk"}.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flowdis.data.paip import (  # noqa: E402
    VARIANTS,
    PAIPSample,
    alpha_blend,
    combine_masks,
    crop_foreground,
    fit_within,
    paip_collate,
    paip_mix,
    place_with_minimal_overlap,
)

DIS5K_ROOT = Path("/home/ubuntu/jeremy/dataset/DIS5K_extracted")


@pytest.fixture
def reference() -> PAIPSample:
    """Object in the lower-left, so a large blank region sits above/right."""
    image = np.full((100, 200, 3), 120, np.uint8)
    mask = np.zeros((100, 200), np.float32)
    mask[60:95, 10:70] = 1.0
    return PAIPSample(image, mask, "a bench")


@pytest.fixture
def pairing() -> PAIPSample:
    image = np.full((80, 80, 3), 200, np.uint8)
    mask = np.zeros((80, 80), np.float32)
    mask[20:60, 20:60] = 1.0
    return PAIPSample(image, mask, "a red ball")


def rng() -> random.Random:
    return random.Random(0)


# --------------------------------------------------------------------------
# crop_foreground / fit_within
# --------------------------------------------------------------------------


def test_crop_foreground_returns_the_bounding_box_region(pairing):
    image, mask = crop_foreground(pairing.image, pairing.mask)
    assert mask.shape == (40, 40)
    assert image.shape == (40, 40, 3)
    assert mask.min() == 1.0, "crop should be exactly the object extent"


def test_fit_within_preserves_aspect_ratio():
    image = np.zeros((40, 80, 3), np.uint8)
    mask = np.ones((40, 80), np.float32)
    _, resized = fit_within(image, mask, width=40, height=40)
    assert resized.shape == (20, 40), "2:1 source must stay 2:1"


def test_fit_within_never_exceeds_the_area():
    image = np.zeros((300, 90, 3), np.uint8)
    mask = np.ones((300, 90), np.float32)
    for w, h in [(50, 50), (10, 200), (7, 7)]:
        _, resized = fit_within(image, mask, w, h)
        assert resized.shape[0] <= h and resized.shape[1] <= w


def test_fit_within_scale_shrinks_the_object():
    image = np.zeros((40, 40, 3), np.uint8)
    mask = np.ones((40, 40), np.float32)
    _, full = fit_within(image, mask, 40, 40, scale=1.0)
    _, half = fit_within(image, mask, 40, 40, scale=0.5)
    assert half.shape[0] < full.shape[0]


def test_fit_within_never_collapses_to_zero():
    image = np.zeros((300, 300, 3), np.uint8)
    mask = np.ones((300, 300), np.float32)
    _, resized = fit_within(image, mask, 1, 1)
    assert resized.shape == (1, 1)


# --------------------------------------------------------------------------
# placement and blending
# --------------------------------------------------------------------------


def test_placement_stays_inside_the_region():
    reference_mask = np.zeros((100, 100), np.float32)
    fg = np.ones((10, 10), np.float32)
    region = (20, 30, 80, 90)
    for seed in range(50):
        x, y = place_with_minimal_overlap(reference_mask, fg, region, random.Random(seed))
        assert 20 <= x <= 80 - 10
        assert 30 <= y <= 90 - 10


def test_placement_avoids_the_reference_object():
    """Given a region that does overlap the object, it should dodge it."""
    reference_mask = np.zeros((100, 100), np.float32)
    reference_mask[:, :50] = 1.0  # object fills the left half
    fg = np.ones((10, 10), np.float32)
    x, _ = place_with_minimal_overlap(
        reference_mask, fg, (0, 0, 100, 100), random.Random(0), candidates=64
    )
    assert x >= 50, "should land clear of the object"


def test_alpha_blend_replaces_where_alpha_is_one():
    canvas = np.zeros((20, 20, 3), np.uint8)
    fg = np.full((5, 5, 3), 255, np.uint8)
    alpha = np.ones((5, 5), np.float32)
    out = alpha_blend(canvas, fg, alpha, 3, 4)
    assert (out[4:9, 3:8] == 255).all()
    assert out.sum() == 5 * 5 * 3 * 255, "nothing outside the patch may change"


def test_alpha_blend_is_a_true_mix_at_half_alpha():
    canvas = np.zeros((10, 10), np.float32)
    fg = np.ones((4, 4), np.float32)
    out = alpha_blend(canvas, fg, np.full((4, 4), 0.5, np.float32), 0, 0)
    assert np.allclose(out[:4, :4], 0.5)


def test_alpha_blend_preserves_dtype_and_clips():
    canvas = np.full((8, 8, 3), 250, np.uint8)
    fg = np.full((4, 4, 3), 255, np.uint8)
    out = alpha_blend(canvas, fg, np.ones((4, 4), np.float32), 0, 0)
    assert out.dtype == np.uint8 and out.max() <= 255


# --------------------------------------------------------------------------
# combine_masks -- the paper's three options
# --------------------------------------------------------------------------


def test_reference_variant_subtracts_the_occluder():
    ref = np.zeros((10, 10), np.float32)
    ref[:, :6] = 1.0
    pair = np.zeros((10, 10), np.float32)
    pair[:, 4:] = 1.0  # covers columns 4 and 5 of the reference
    mask, prompt = combine_masks(ref, pair, "j", "k", "reference")
    assert prompt == "j"
    assert mask[:, :4].all() and not mask[:, 4:].any()


def test_pairing_variant_is_the_pasted_object_only():
    ref, pair = np.ones((5, 5), np.float32), np.zeros((5, 5), np.float32)
    pair[1:3, 1:3] = 1.0
    mask, prompt = combine_masks(ref, pair, "j", "k", "pairing")
    assert prompt == "k"
    np.testing.assert_array_equal(mask, pair)


def test_both_variant_is_the_union_and_joins_prompts():
    ref, pair = np.zeros((5, 5), np.float32), np.zeros((5, 5), np.float32)
    ref[0, 0] = 1.0
    pair[4, 4] = 1.0
    mask, prompt = combine_masks(ref, pair, "a bench", "a red ball", "both")
    assert prompt == "a bench and a red ball"
    assert mask[0, 0] == 1.0 and mask[4, 4] == 1.0 and mask.sum() == 2.0


def test_and_or_semantics_on_soft_masks():
    """AND is multiplication and OR is maximum, per the paper."""
    ref = np.array([[0.8]], np.float32)
    pair = np.array([[0.25]], np.float32)
    assert combine_masks(ref, pair, "", "", "reference")[0] == pytest.approx(0.8 * 0.75)
    assert combine_masks(ref, pair, "", "", "both")[0] == pytest.approx(0.8)


def test_combine_masks_rejects_unknown_variant():
    z = np.zeros((2, 2), np.float32)
    with pytest.raises(ValueError, match="unknown variant"):
        combine_masks(z, z, "", "", "nope")


def test_prompt_join_degrades_when_one_is_missing():
    z = np.zeros((2, 2), np.float32)
    assert combine_masks(z, z, "a bench", "", "both")[1] == "a bench"
    assert combine_masks(z, z, "", "a ball", "both")[1] == "a ball"


# --------------------------------------------------------------------------
# paip_mix
# --------------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_mix_shapes_agree(reference, pairing, variant):
    result = paip_mix(reference, pairing, rng(), variant=variant)
    assert result.image.shape[:2] == result.mask.shape
    assert result.mask_reference.shape == result.mask_pairing.shape == result.mask.shape
    assert result.variant == variant


@pytest.mark.parametrize("variant", VARIANTS)
def test_mix_mask_stays_in_range(reference, pairing, variant):
    result = paip_mix(reference, pairing, rng(), variant=variant)
    assert result.mask.min() >= 0.0 and result.mask.max() <= 1.0


def test_mix_enlarges_the_canvas(reference, pairing):
    """Padding happens, so the composite is bigger than the reference."""
    result = paip_mix(reference, pairing, rng())
    assert result.image.shape[0] >= reference.image.shape[0]
    assert result.image.shape[1] >= reference.image.shape[1]
    assert result.image.size > reference.image.size


def test_mix_actually_pastes_something(reference, pairing):
    result = paip_mix(reference, pairing, rng(), variant="pairing")
    assert result.mask_pairing.sum() > 0, "no pairing object was placed"


def test_mix_places_clear_of_the_reference_object(reference, pairing):
    """The placement region never intersects the reference box, so overlap is 0."""
    result = paip_mix(reference, pairing, rng())
    assert (result.mask_reference * result.mask_pairing).sum() == 0.0


def test_mix_prompts_match_their_variant(reference, pairing):
    assert paip_mix(reference, pairing, rng(), variant="reference").prompt == "a bench"
    assert paip_mix(reference, pairing, rng(), variant="pairing").prompt == "a red ball"
    assert (
        paip_mix(reference, pairing, rng(), variant="both").prompt
        == "a bench and a red ball"
    )


def test_both_variant_covers_the_other_two(reference, pairing):
    a = paip_mix(reference, pairing, random.Random(3), variant="reference")
    c = paip_mix(reference, pairing, random.Random(3), variant="both")
    assert c.mask.sum() >= a.mask.sum()


def test_mix_is_deterministic_for_a_given_seed(reference, pairing):
    a = paip_mix(reference, pairing, random.Random(7))
    b = paip_mix(reference, pairing, random.Random(7))
    np.testing.assert_array_equal(a.image, b.image)
    np.testing.assert_array_equal(a.mask, b.mask)
    assert a.variant == b.variant


def test_mix_changes_the_image_content(reference, pairing):
    """The pasted object must be visible, not blended into nothing."""
    result = paip_mix(reference, pairing, rng(), variant="pairing")
    placed = result.image[result.mask_pairing > 0.5]
    assert placed.size > 0
    assert not np.allclose(placed, 120), "pasted region still shows reference colour"


def test_mix_accepts_uint8_masks(reference, pairing):
    ref = PAIPSample(reference.image, (reference.mask * 255).astype(np.uint8), "j")
    pair = PAIPSample(pairing.image, (pairing.mask * 255).astype(np.uint8), "k")
    result = paip_mix(ref, pair, rng(), variant="both")
    assert result.mask.max() <= 1.0 and result.mask.sum() > 0


def test_mix_falls_back_when_reference_fills_the_frame(pairing):
    full = PAIPSample(np.zeros((40, 40, 3), np.uint8), np.ones((40, 40), np.float32), "j")
    result = paip_mix(full, pairing, rng())
    assert result.mixed is False
    assert result.prompt == "j"


def test_mix_falls_back_on_an_empty_pairing_mask(reference):
    empty = PAIPSample(np.zeros((40, 40, 3), np.uint8), np.zeros((40, 40), np.float32), "k")
    result = paip_mix(reference, empty, rng())
    assert result.mixed is False


def test_mix_normalised_float_images_are_supported(reference, pairing):
    """Blending is affine, so it commutes with the dataset's normalisation."""
    ref = PAIPSample((reference.image / 255.0 - 0.5) / 0.5, reference.mask, "j")
    pair = PAIPSample((pairing.image / 255.0 - 0.5) / 0.5, pairing.mask, "k")
    result = paip_mix(ref, pair, rng(), variant="both")
    assert result.image.dtype == np.float64 or result.image.dtype == np.float32
    assert np.isfinite(result.image).all()


@pytest.mark.parametrize("seed", range(25))
def test_mix_invariants_on_random_geometry(seed):
    r = random.Random(seed)
    h, w = r.randint(30, 90), r.randint(30, 90)
    mask = np.zeros((h, w), np.float32)
    r0, c0 = r.randrange(h // 2), r.randrange(w // 2)
    mask[r0 : r0 + r.randint(3, h // 2), c0 : c0 + r.randint(3, w // 2)] = 1.0
    ref = PAIPSample(np.zeros((h, w, 3), np.uint8), mask, "j")

    ph, pw = r.randint(10, 50), r.randint(10, 50)
    pmask = np.zeros((ph, pw), np.float32)
    pmask[2 : ph - 2, 2 : pw - 2] = 1.0
    pair = PAIPSample(np.full((ph, pw, 3), 255, np.uint8), pmask, "k")

    result = paip_mix(ref, pair, r)
    assert result.image.shape[:2] == result.mask.shape
    assert 0.0 <= result.mask.min() and result.mask.max() <= 1.0
    if result.mixed:
        assert (result.mask_reference * result.mask_pairing).sum() == 0.0


# --------------------------------------------------------------------------
# paip_collate
# --------------------------------------------------------------------------


def torch_sample(seed: int, size: int = 64) -> dict:
    import torch

    gen = np.random.default_rng(seed)
    image = torch.from_numpy((gen.random((3, size, size), dtype=np.float32) * 2 - 1))
    mask = torch.zeros(1, size, size)
    mask[0, size // 2 :, : size // 3] = 1.0
    return {"image": image, "mask": mask, "prompt": f"object {seed}", "name": f"{seed}.jpg"}


def test_collate_preserves_batch_shape():
    batch = [torch_sample(i) for i in range(4)]
    out = paip_collate(batch, rng=random.Random(0))
    assert tuple(out["image"].shape) == (4, 3, 64, 64)
    assert tuple(out["mask"].shape) == (4, 1, 64, 64)
    assert len(out["prompt"]) == len(out["variant"]) == 4


def test_collate_masks_stay_in_range():
    out = paip_collate([torch_sample(i) for i in range(6)], rng=random.Random(1))
    assert float(out["mask"].min()) >= 0.0
    assert float(out["mask"].max()) <= 1.0


def test_collate_prompt_matches_variant():
    batch = [torch_sample(i) for i in range(4)]
    out = paip_collate(batch, rng=random.Random(0))
    for prompt, variant in zip(out["prompt"], out["variant"]):
        if variant == "both":
            assert " and " in prompt
        else:
            assert " and " not in prompt


def test_collate_single_sample_is_passed_through():
    out = paip_collate([torch_sample(0)], rng=random.Random(0))
    assert out["variant"] == ["reference"]
    assert out["prompt"] == ["object 0"]


def test_collate_probability_zero_disables_mixing():
    batch = [torch_sample(i) for i in range(4)]
    out = paip_collate(batch, rng=random.Random(0), probability=0.0)
    assert out["variant"] == ["reference"] * 4
    # untouched samples must come through bit-identical
    for i, sample in enumerate(batch):
        assert out["image"][i].equal(sample["image"])


def test_collate_mixes_when_probability_is_one():
    batch = [torch_sample(i) for i in range(4)]
    out = paip_collate(batch, rng=random.Random(0), probability=1.0)
    changed = sum(
        not out["image"][i].equal(batch[i]["image"]) for i in range(len(batch))
    )
    assert changed == len(batch), "every sample should have been mixed"


def test_collate_is_usable_as_a_dataloader_collate_fn():
    import torch
    from torch.utils.data import DataLoader, Dataset

    class Tiny(Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, idx):
            return torch_sample(idx, size=32)

    loader = DataLoader(
        Tiny(), batch_size=4, collate_fn=lambda b: paip_collate(b, rng=random.Random(0))
    )
    batch = next(iter(loader))
    assert tuple(batch["image"].shape) == (4, 3, 32, 32)
    assert isinstance(batch["prompt"], list)


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------


@pytest.mark.skipif(not DIS5K_ROOT.is_dir(), reason="DIS5K not extracted")
def test_paip_on_real_dis5k_pairs():
    import cv2

    im_dir, gt_dir = DIS5K_ROOT / "DIS-TR" / "im", DIS5K_ROOT / "DIS-TR" / "gt"
    names = sorted(p.name for p in im_dir.glob("*.jpg"))[:60]
    samples = []
    for name in names:
        image = cv2.imread(str(im_dir / name), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(gt_dir / f"{Path(name).stem}.png"), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
        samples.append(PAIPSample(image, mask.astype(np.float32) / 255.0, name))

    r = random.Random(0)
    for i, ref in enumerate(samples):
        pair = samples[(i + 1) % len(samples)]
        result = paip_mix(ref, pair, r)
        assert result.image.shape[:2] == result.mask.shape
        assert 0.0 <= result.mask.min() and result.mask.max() <= 1.0
        if result.mixed:
            # Real masks are anti-aliased, so the two can share a sub-pixel
            # sliver of soft mass where they meet at the region boundary.
            # Exact zero holds for binary masks (see the synthetic test) but
            # not here; what matters is that no real object area is shared.
            overlap = float((result.mask_reference * result.mask_pairing).sum())
            assert overlap < 1.0, f"{ref.prompt}: objects overlap by {overlap:.2f}px"
