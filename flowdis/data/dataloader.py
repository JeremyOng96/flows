from dataclasses import dataclass
from functools import partial
from pathlib import Path
import json
import random
import albumentations as A

import cv2
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from flowdis.data.paip import paip_collate


SPLITS = ("DIS-TR", "DIS-VD", "DIS-TE1", "DIS-TE2", "DIS-TE3", "DIS-TE4")
TRAIN_SPLIT = "DIS-TR"  # the paper trains only here; DIS-VD/TE are test-only

INTERPOLATIONS = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
}


@dataclass
class DIS5KParams:
    root: str
    split: str = "DIS-TR"
    resolution: int = 1024
    use_language_pairing: bool = False
    # DIS-TR ships ~4 paraphrases per image; the eval splits ship one string.
    # "random" resamples per access (so epochs differ), "first" is deterministic.
    prompt_strategy: str = "random"
    mean: float = 0.5
    std: float = 0.5


@dataclass
class DIS5KTransforms:
    """Settings for the albumentations pipeline applied jointly to image and mask.

    The paper uses no geometric augmentation -- its only augmentation is PAIP,
    which mixes pairs of samples within a batch (see flowdis.data.paip). The crop
    and flip knobs are therefore off by default; turn them on only when you mean
    to deviate from the paper.

    The resize always runs, in train and eval alike: it is what makes samples
    the same size and therefore collatable.
    """

    # None keeps the plain resize; a (min, max) area fraction switches training
    # to RandomResizedCrop, which tolerates any input size. A fixed RandomCrop
    # cannot be used here -- 38% of DIS-TR has a side under 2048.
    crop_scale: tuple[float, float] | None = None
    horizontal_flip: float = 0.0
    vertical_flip: float = 0.0
    # "area" antialiases on the way down, which matters for the thin structures
    # DIS is about; the reference implementation uses PIL bicubic at inference.
    interpolation: str = "area"

    @classmethod
    def from_config(cls, config: DictConfig | dict | None) -> "DIS5KTransforms":
        if config is None:
            return cls()
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        return cls(**config)

    def build(self, resolution: int, train: bool) -> A.Compose:
        if self.interpolation not in INTERPOLATIONS:
            raise ValueError(
                f"unknown interpolation {self.interpolation!r}, "
                f"expected one of {tuple(INTERPOLATIONS)}"
            )
        interp = INTERPOLATIONS[self.interpolation]

        if train and self.crop_scale is not None:
            ops = [
                A.RandomResizedCrop(
                    size=(resolution, resolution),
                    scale=tuple(self.crop_scale),
                    interpolation=interp,
                    mask_interpolation=interp,
                )
            ]
        else:
            ops = [
                A.Resize(
                    resolution, resolution,
                    interpolation=interp, mask_interpolation=interp,
                )
            ]

        if train:
            if self.horizontal_flip > 0:
                ops.append(A.HorizontalFlip(p=self.horizontal_flip))
            if self.vertical_flip > 0:
                ops.append(A.VerticalFlip(p=self.vertical_flip))
        return A.Compose(ops)


class DIS5KDataset(Dataset):
    """DIS5K image / binary-mask pairs, with optional language prompts.

    Expects the extracted layout:

        <root>/<split>/im/<name>.jpg
        <root>/<split>/gt/<name>.png
        <root>/language_prompts/<split>.json   # {"<name>.jpg": str | list[str]}

    Each item is a dict:
        image  float32 (3, res, res), normalised to (x/255 - mean) / std
        mask   float32 (1, res, res) in [0, 1]
        prompt str ("" when language pairing is off)
        name   str, the image filename
    """

    def __init__(
        self,
        params: DIS5KParams,
        use_transforms: bool = True,
        transforms: DIS5KTransforms | None = None,
    ):
        self.params = params
        if params.split not in SPLITS:
            raise ValueError(f"unknown split {params.split!r}, expected one of {SPLITS}")
        if params.prompt_strategy not in ("random", "first"):
            raise ValueError(f"unknown prompt_strategy {params.prompt_strategy!r}")

        # use_transforms toggles augmentation only; the resize runs either way
        self.transform_params = transforms or DIS5KTransforms()
        self.transforms = self.transform_params.build(params.resolution, train=use_transforms)

        root = Path(params.root)
        self.image_dir = root / params.split / "im"
        self.mask_dir = root / params.split / "gt"
        self.prompt_path = root / "language_prompts" / f"{params.split}.json"
        self.load_data()

    @classmethod
    def from_config(cls, config: DictConfig) -> "DIS5KDataset":
        data = OmegaConf.to_container(config.data, resolve=True)
        transforms = data.pop("transforms", None)
        use_transforms = data.pop("use_transforms", True)
        return cls(
            DIS5KParams(**data),
            use_transforms=use_transforms,
            transforms=DIS5KTransforms.from_config(transforms),
        )

    def load_data(self) -> None:
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"image directory not found: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"mask directory not found: {self.mask_dir}")

        # Sort so the order is stable, and derive each mask from its image stem
        # rather than globbing the two directories independently.
        self.names = sorted(p.name for p in self.image_dir.glob("*.jpg"))
        if not self.names:
            raise FileNotFoundError(f"no .jpg images under {self.image_dir}")

        missing = [n for n in self.names if not (self.mask_dir / f"{Path(n).stem}.png").is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} image(s) have no mask in {self.mask_dir}, "
                f"first: {missing[0]}"
            )

        self.prompts: dict[str, str | list[str]] = {}
        if self.params.use_language_pairing:
            if not self.prompt_path.is_file():
                raise FileNotFoundError(f"language prompts not found: {self.prompt_path}")
            with self.prompt_path.open() as f:
                self.prompts = json.load(f)

    def __len__(self) -> int:
        return len(self.names)

    def _prompt(self, name: str) -> str:
        if not self.params.use_language_pairing:
            return ""
        value = self.prompts.get(name, "")
        if isinstance(value, list):
            if not value:
                return ""
            if self.params.prompt_strategy == "random":
                return str(random.choice(value))
            return str(value[0])
        return str(value)

    def __getitem__(self, idx: int) -> dict[str, Tensor | str]:
        name = self.names[idx]
        image_path = self.image_dir / name
        mask_path = self.mask_dir / f"{Path(name).stem}.png"

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to decode image: {image_path}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to decode mask: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2 decodes BGR
        # one call, so any geometric op stays registered between image and mask
        augmented = self.transforms(image=image, mask=mask)
        image = augmented["image"].astype(np.float32) / 255.0
        mask = np.clip(augmented["mask"].astype(np.float32) / 255.0, 0.0, 1.0)

        image = (image - self.params.mean) / self.params.std
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask).unsqueeze(0).contiguous()

        return {"image": image, "mask": mask, "prompt": self._prompt(name), "name": name}



def build_dataloader(
    params: DIS5KParams,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool | None = None,
    use_paip: bool | None = None,
    transforms: DIS5KTransforms | None = None,
    seed: int | None = None,
) -> DataLoader:
    """Build a loader for one split.

    `shuffle` and `use_paip` both default to "on for DIS-TR only". PAIP is a
    training augmentation: running it over DIS-VD/DIS-TE would composite the
    evaluation images into each other and replace their ground truth, which
    silently invalidates any score computed from them.
    """
    is_train = params.split == TRAIN_SPLIT
    if shuffle is None:
        shuffle = is_train
    if use_paip is None:
        use_paip = is_train

    dataset = DIS5KDataset(params, use_transforms=is_train, transforms=transforms)

    collate_fn = None
    if use_paip:
        # partial, not a lambda: num_workers > 0 has to pickle this
        collate_fn = partial(paip_collate, rng=random.Random(seed) if seed is not None else None)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        collate_fn=collate_fn,
    )


if __name__ == "__main__":
    from dataclasses import asdict

    params = DIS5KParams(
        root="/home/ubuntu/jeremy/dataset/DIS5K_extracted",
        split="DIS-VD",
        resolution=256,
        use_language_pairing=True,
    )
    print("DIS5KParams:")
    for key, value in asdict(params).items():
        print(f"  {key}: {value!r}")

    dataset = DIS5KDataset(params)
    print(f"dataset length: {len(dataset)}")
    print(f"length again:   {len(dataset)}")

    sample = dataset[0]
    print(f"image shape: {tuple(sample['image'].shape)}  dtype: {sample['image'].dtype}")
    print(f"mask shape:  {tuple(sample['mask'].shape)}  dtype: {sample['mask'].dtype}")
    print(f"image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"mask range:  [{sample['mask'].min():.3f}, {sample['mask'].max():.3f}]")
    print(f"name:   {sample['name']}")
    print(f"prompt: {sample['prompt']}")

    print("\n--- DIS-VD (eval split: no PAIP, no augmentation) ---")
    loader = build_dataloader(params, batch_size=2, num_workers=0)
    batch = next(iter(loader))
    print(f"collate_fn: {loader.collate_fn}")
    print(f"batch image: {tuple(batch['image'].shape)}")
    print(f"batch mask:  {tuple(batch['mask'].shape)}")
    print(f"batch prompts: {batch['prompt']}")

    print("\n--- DIS-TR (train split: PAIP on) ---")
    train_params = DIS5KParams(
        root=params.root, split="DIS-TR", resolution=256, use_language_pairing=True
    )
    train_loader = build_dataloader(
        train_params, batch_size=4, num_workers=2, seed=0
    )
    train_batch = next(iter(train_loader))
    print(f"batch image: {tuple(train_batch['image'].shape)}")
    print(f"batch mask:  {tuple(train_batch['mask'].shape)}")
    print(f"variants: {train_batch['variant']}")
    for prompt in train_batch["prompt"]:
        print(f"  {prompt[:88]}")
    print("ok")
