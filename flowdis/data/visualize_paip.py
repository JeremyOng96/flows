"""Draw the PAIP geometry steps on real DIS5K samples, using cv2.

Per row: the object's bounding box and the largest adjacent blank region
R_max, then the same scene after reflection padding, where R_max has doubled.

    python3 flowdis/data/visualize_paip.py --split DIS-TR -n 3
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flowdis.data.utils import (  # noqa: E402
    get_blank_region_proposals,
    get_minimum_bounding_box,
    reflection_pad,
)

FONT = cv2.FONT_HERSHEY_SIMPLEX
BBOX_COLOR = (60, 200, 255)    # amber, BGR
REGION_COLOR = (240, 170, 60)  # blue
MASK_COLOR = (90, 220, 110)    # green
PAD = 10
CAPTION_H = 46
DISPLAY = 420


def load_prompts(root: Path, split: str) -> dict:
    path = root / "language_prompts" / f"{split}.json"
    if not path.is_file():
        return {}
    with path.open() as f:
        # DIS-TR stores 4 prompts per image (2 GPT-4V types + 2 paraphrases);
        # the eval splits store one string. Keep both shapes as-is.
        return json.load(f)


def pick_prompt(value, rng, slot: int | None = None) -> str:
    """Resolve one prompt, matching how training samples them.

    The paper samples one of the four uniformly, so that is the default here;
    `slot` pins a specific one instead, for inspecting the four side by side.
    """
    if isinstance(value, list):
        if not value:
            return ""
        return str(value[slot % len(value)] if slot is not None else rng.choice(value))
    return str(value or "")


def load(image_path: Path, mask_path: Path, longest: int):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    scale = longest / max(image.shape[:2])
    size = (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale)))
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)
    return image, (mask > 127).astype(np.uint8)


def shade(canvas: np.ndarray, rect, color, alpha=0.35) -> None:
    """Translucent fill plus a solid outline, drawn in place."""
    x0, y0, x1, y1 = rect
    if x1 <= x0 or y1 <= y0:
        return
    patch = canvas[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = cv2.addWeighted(
        patch, 1 - alpha, np.full_like(patch, color, dtype=np.uint8), alpha, 0
    )
    cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), color, 2)


def draw_mask_edge(canvas: np.ndarray, mask: np.ndarray) -> None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, MASK_COLOR, 2)


def annotate(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, tuple, int]:
    bbox = get_minimum_bounding_box(mask)
    region, _ = get_blank_region_proposals(image, bbox)
    canvas = image.copy()
    shade(canvas, region, REGION_COLOR)
    draw_mask_edge(canvas, mask)
    xmin, xmax, ymin, ymax = bbox
    cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), BBOX_COLOR, 2)
    x0, y0, x1, y1 = region
    return canvas, region, (x1 - x0) * (y1 - y0)


def label(panel: np.ndarray, text: str) -> np.ndarray:
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (32, 32, 32), cv2.FILLED)
    cv2.putText(out, text, (7, 16), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def fit(panel: np.ndarray, height: int) -> np.ndarray:
    """Scale to a common row height, then centre on a white field."""
    scale = height / panel.shape[0]
    resized = cv2.resize(
        panel,
        (max(1, int(round(panel.shape[1] * scale))), height),
        interpolation=cv2.INTER_AREA,
    )
    return resized


def build_row(image: np.ndarray, mask: np.ndarray, name: str) -> np.ndarray:
    before, region, area_before = annotate(image, mask)
    padded_image, padded_mask = reflection_pad(image, mask, region)
    after, new_region, area_after = annotate(padded_image, padded_mask)

    x0, y0, x1, y1 = region
    side = "left/right" if (y0 == 0 and y1 == image.shape[0]) else "top/bottom"
    ratio = area_after / area_before if area_before else 0.0

    panels = [
        label(before, f"bbox + R_max   area {area_before}"),
        label(after, f"after reflection pad   area {area_after}  ({ratio:.2f}x)"),
        label(
            cv2.cvtColor(padded_mask * 255, cv2.COLOR_GRAY2BGR),
            "padded mask (stays registered)",
        ),
    ]
    height = max(p.shape[0] for p in panels)
    panels = [fit(p, height) for p in panels]

    gap = np.full((height, PAD, 3), 255, dtype=np.uint8)
    row = cv2.hconcat([panels[0], gap, panels[1], gap, panels[2]])

    caption = np.full((CAPTION_H, row.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(caption, name[:92], (4, 16), FONT, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(
        caption,
        f"{image.shape[1]}x{image.shape[0]} -> {padded_image.shape[1]}x{padded_image.shape[0]}"
        f"   padded on the {side} axis   R_max {area_before} -> {area_after}",
        (4, 34),
        FONT,
        0.44,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return cv2.vconcat([row, caption])


def mask_panel(mask: np.ndarray) -> np.ndarray:
    gray = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def outline(image: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    canvas = image.copy()
    binary = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, color, 2)
    return canvas


def build_mix_row(reference, pairing, rng, name: str) -> np.ndarray:
    """reference | pairing | I_mix | the three M_mix options."""
    from flowdis.data.paip import paip_mix

    results = {
        v: paip_mix(reference, pairing, random.Random(rng.randint(0, 10**6)), variant=v)
        for v in ("reference", "pairing", "both")
    }
    mixed = results["both"]

    panels = [
        label(outline(reference.image, reference.mask, MASK_COLOR), "reference I_j"),
        label(outline(pairing.image, pairing.mask, (90, 140, 250)), "pairing I_k"),
        label(mixed.image, "I_mix  (alpha blended)"),
        label(mask_panel(results["reference"].mask), "M_j AND NOT M_k"),
        label(mask_panel(results["pairing"].mask), "M_k"),
        label(mask_panel(results["both"].mask), "M_j OR M_k"),
    ]
    height = max(p.shape[0] for p in panels)
    panels = [fit(p, height) for p in panels]

    gap = np.full((height, PAD, 3), 255, dtype=np.uint8)
    row = cv2.hconcat([p for panel in panels for p in (panel, gap)][:-1])

    caption = np.full((CAPTION_H + 16, row.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(caption, name[:110], (4, 15), FONT, 0.40, (90, 90, 90), 1, cv2.LINE_AA)
    for i, v in enumerate(("reference", "pairing", "both")):
        cv2.putText(
            caption,
            f"{v:10s} -> {results[v].prompt[:96]}",
            (4, 32 + i * 15),
            FONT,
            0.42,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return cv2.vconcat([row, caption])


def build_batch_grid(
    im_dir: Path, gt_dir: Path, names: list[str], prompts: dict, size: int, rng,
    slot: int | None = None,
) -> np.ndarray:
    """Run paip_collate over real samples and draw the batch it produces.

    This is the training view: PAIP at native resolution, then one resize to
    `size`, with its randomly chosen variant and matching prompt.
    """
    import torch

    from flowdis.data.paip import paip_collate

    mean = std = 0.5
    batch = []
    for name in names:
        image = cv2.imread(str(im_dir / name), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(gt_dir / f"{Path(name).stem}.png"), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        batch.append(
            {
                "image": torch.from_numpy((rgb - mean) / std).permute(2, 0, 1),
                "mask": torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0),
                "prompt": pick_prompt(prompts.get(name, name), rng, slot),
                "name": name,
            }
        )

    out = paip_collate(batch, rng=rng, resolution=size)
    panels = []
    for i in range(len(out["image"])):
        rgb = out["image"][i].permute(1, 2, 0).numpy() * std + mean
        bgr = cv2.cvtColor(np.clip(rgb * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        m = out["mask"][i, 0].numpy()[..., None]
        tint = np.full_like(bgr, (120, 230, 130), dtype=np.uint8)
        overlay = np.clip(bgr * (1 - 0.45 * m) + tint * 0.45 * m, 0, 255).astype(np.uint8)

        cell = cv2.hconcat([label(bgr, f"I_mix   [{out['variant'][i]}]"), label(overlay, "M_mix")])
        caption = np.full((36, cell.shape[1], 3), 255, dtype=np.uint8)
        for j, line in enumerate(wrap_text(out["prompt"][i], cell.shape[1] - 10)):
            cv2.putText(caption, line, (4, 15 + j * 15), FONT, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        panels.append(cv2.vconcat([cell, caption]))

    gap_w = np.full((panels[0].shape[0], PAD, 3), 255, dtype=np.uint8)
    rows = []
    for i in range(0, len(panels), 2):
        pair = panels[i : i + 2]
        row = cv2.hconcat([pair[0], gap_w, pair[1]] if len(pair) == 2 else [pair[0]])
        rows.append(row)
    width = max(r.shape[1] for r in rows)
    rows = [
        cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(255,) * 3)
        for r in rows
    ]
    gap_h = np.full((PAD * 2, width, 3), 255, dtype=np.uint8)
    stacked = [x for r in rows for x in (r, gap_h)][:-1]
    return cv2.vconcat(stacked)


def wrap_text(text: str, width_px: int, scale: float = 0.42) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if cv2.getTextSize(trial, FONT, scale, 1)[0][0] > width_px and current:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines[:2]


def legend(width: int) -> np.ndarray:
    bar = np.full((34, width, 3), 255, dtype=np.uint8)
    x = 6
    for color, text in (
        (MASK_COLOR, "mask outline"),
        (BBOX_COLOR, "bounding box B_j"),
        (REGION_COLOR, "largest blank region R_max"),
    ):
        cv2.rectangle(bar, (x, 11), (x + 22, 25), color, cv2.FILLED)
        cv2.putText(bar, text, (x + 29, 23), FONT, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        x += 40 + cv2.getTextSize(text, FONT, 0.45, 1)[0][0]
    return bar


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/home/ubuntu/jeremy/dataset/DIS5K_extracted")
    ap.add_argument("--split", default="DIS-TR")
    ap.add_argument("-n", "--num-samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--longest", type=int, default=DISPLAY)
    ap.add_argument(
        "--mode",
        choices=("geometry", "mix", "batch"),
        default="geometry",
        help="geometry: bbox/region/padding. mix: I_mix and its three masks. "
        "batch: what paip_collate hands the model.",
    )
    ap.add_argument("--prompt-slot", type=int, default=None,
                    help="pin prompt 0-3 instead of sampling uniformly (DIS-TR only)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path(f"paip_{args.mode}.png")

    im_dir = Path(args.root) / args.split / "im"
    gt_dir = Path(args.root) / args.split / "gt"
    names = sorted(p.name for p in im_dir.glob("*.jpg"))
    rng = random.Random(args.seed)
    prompts = load_prompts(Path(args.root), args.split)

    picked = rng.sample(names, min(args.num_samples * 2, len(names)))

    if args.mode == "batch":
        grid = build_batch_grid(im_dir, gt_dir, picked, prompts, args.longest, rng, args.prompt_slot)
        cv2.imwrite(str(out), grid)
        print(f"wrote {out}  ({grid.shape[1]}x{grid.shape[0]})")
        return

    rows, gap = [], None
    for index in range(0, len(picked) - 1, 2):
        name, other = picked[index], picked[index + 1]
        image, mask = load(im_dir / name, gt_dir / f"{Path(name).stem}.png", args.longest)

        if args.mode == "geometry":
            row = build_row(image, mask, name)
        else:
            from flowdis.data.paip import PAIPSample

            pair_image, pair_mask = load(
                im_dir / other, gt_dir / f"{Path(other).stem}.png", args.longest
            )
            row = build_mix_row(
                PAIPSample(image, mask.astype(np.float32), pick_prompt(prompts.get(name, name), rng, args.prompt_slot)),
                PAIPSample(pair_image, pair_mask.astype(np.float32), pick_prompt(prompts.get(other, other), rng, args.prompt_slot)),
                rng,
                f"{name}   +   {other}",
            )

        if gap is None:
            gap = np.full((PAD * 2, row.shape[1], 3), 255, dtype=np.uint8)
        if rows and row.shape[1] != rows[0].shape[1]:
            row = cv2.resize(row, (rows[0].shape[1], row.shape[0]))
        rows.extend([row, gap])
        if len(rows) // 2 >= args.num_samples:
            break

    grid = cv2.vconcat(rows[:-1])
    if args.mode == "geometry":
        grid = cv2.vconcat([legend(grid.shape[1]), grid])
    cv2.imwrite(str(out), grid)
    print(f"wrote {out}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
