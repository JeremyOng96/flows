"""Render DIS5KDataset samples as an image grid, composed entirely with cv2.

Each row is image | mask | overlay, taken straight out of the dataset (so what
you see is what the model gets, after resize and normalisation), with the
language prompt drawn underneath.

    python3 flowdis/data/visualize.py --split DIS-TR -n 4
    python3 flowdis/data/visualize.py --split DIS-VD --resolution 512 --show
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flowdis.data.dataloader import SPLITS, DIS5KDataset, DIS5KParams  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_LABELS = ("image", "mask", "overlay")
MASK_TINT = (80, 235, 100)  # BGR green
PAD = 10
CAPTION_H = 52


def to_bgr(image: torch.Tensor, mean: float, std: float) -> np.ndarray:
    """Undo the dataset's normalisation and hand cv2 the BGR uint8 it expects."""
    rgb = image.permute(1, 2, 0).numpy() * std + mean
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def mask_to_bgr(mask: torch.Tensor) -> np.ndarray:
    gray = np.clip(mask[0].numpy() * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def make_overlay(image_bgr: np.ndarray, mask: torch.Tensor) -> np.ndarray:
    """Green wash over the masked region, so any misalignment is obvious."""
    alpha = mask[0].numpy()[..., None]
    tint = np.full_like(image_bgr, MASK_TINT, dtype=np.uint8)
    blended = cv2.addWeighted(image_bgr, 0.55, tint, 0.45, 0.0)
    return (image_bgr * (1 - alpha) + blended * alpha).astype(np.uint8)


def label(panel: np.ndarray, text: str) -> np.ndarray:
    """Draw a caption bar across the top of a panel."""
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (32, 32, 32), cv2.FILLED)
    cv2.putText(out, text, (8, 17), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def wrap(text: str, width_px: int, scale: float = 0.45) -> list[str]:
    """Greedy wrap using cv2's own text metrics."""
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


def build_row(sample: dict, mean: float, std: float) -> np.ndarray:
    image_bgr = to_bgr(sample["image"], mean, std)
    panels = [
        image_bgr,
        mask_to_bgr(sample["mask"]),
        make_overlay(image_bgr, sample["mask"]),
    ]
    panels = [label(p, t) for p, t in zip(panels, PANEL_LABELS)]
    # separators between panels
    gap = np.full((panels[0].shape[0], PAD, 3), 255, dtype=np.uint8)
    row = cv2.hconcat([panels[0], gap, panels[1], gap, panels[2]])

    caption = np.full((CAPTION_H, row.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(caption, sample["name"][:96], (4, 16), FONT, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
    for i, line in enumerate(wrap(sample["prompt"], row.shape[1] - 8)):
        cv2.putText(caption, line, (4, 33 + i * 15), FONT, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return cv2.vconcat([row, caption])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/home/ubuntu/jeremy/dataset/DIS5K_extracted")
    ap.add_argument("--split", default="DIS-VD", choices=SPLITS)
    ap.add_argument("-n", "--num-samples", type=int, default=4)
    ap.add_argument("--resolution", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("samples_cv2.png"))
    ap.add_argument("--show", action="store_true", help="cv2.imshow instead of writing a file")
    args = ap.parse_args()

    params = DIS5KParams(
        root=args.root,
        split=args.split,
        resolution=args.resolution,
        use_language_pairing=True,
    )
    dataset = DIS5KDataset(params)
    print(f"{args.split}: {len(dataset)} samples, showing {args.num_samples}")

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    gap = None
    rows = []
    for idx in indices:
        row = build_row(dataset[idx], params.mean, params.std)
        if gap is None:
            gap = np.full((PAD, row.shape[1], 3), 255, dtype=np.uint8)
        rows.extend([row, gap])
    grid = cv2.vconcat(rows[:-1])

    if args.show:
        cv2.imshow(f"DIS5K {args.split}", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    cv2.imwrite(str(args.out), grid)
    print(f"wrote {args.out}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
