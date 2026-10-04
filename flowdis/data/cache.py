"""Frozen VAE / T5 / CLIP cache for one DIS5K split.

Each sample stores the tensors `flow_matching_loss` reads when `use_cache` is set:
`z_img`, `z_mask`, and every prompt's `z_txt_t5` and `z_txt_clip`. The image and
mask are stored too, so a cached batch can still draw the validation panel.

Values are the bfloat16 encoder outputs, packed as uint16. PAIP is not applied:
it mixes pixels, and these latents are the resized originals.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from diffusers import AutoencoderKL
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from flowdis.data.dataloader import DIS5KParams, DIS5KDataset

ENCODING = "bfloat16_uint16"


def _to_bf16(tensor: Tensor) -> np.ndarray:
    packed = tensor.detach().to(dtype=torch.bfloat16).contiguous().view(torch.uint16)
    return packed.cpu().numpy()


def _from_bf16(array: np.ndarray) -> Tensor:
    raw = np.ascontiguousarray(array, dtype=np.uint16)
    return torch.from_numpy(raw).view(torch.bfloat16)


def _prompts_for(name: str, table: dict) -> list[str]:
    value = table.get(name, "")
    if isinstance(value, list):
        prompts = [str(item) for item in value if str(item)]
        return prompts or [""]
    text = str(value)
    return [text] if text else [""]


def _encode_latent(vae: AutoencoderKL, pixels: Tensor) -> Tensor:
    shift = vae.config.shift_factor
    scale = vae.config.scaling_factor
    latent = vae.encode(pixels).latent_dist.mode()
    return (latent - shift) * scale


def build_cache(
    data_root: Path,
    schnell_dir: Path,
    split: str,
    output: Path,
    resolution: int = 1024,
    device: str = "cuda",
    limit: int = 0,
) -> None:
    from flowdis.train import CLIPTextEncoder, T5TextEncoder

    params = DIS5KParams(
        root=str(data_root),
        split=split,
        resolution=resolution,
        use_language_pairing=False,
    )
    dataset = DIS5KDataset(params, use_transforms=False)
    prompt_path = data_root / "language_prompts" / f"{split}.json"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"language prompts not found: {prompt_path}")
    prompts = json.loads(prompt_path.read_text())

    output.parent.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKL.from_pretrained(schnell_dir / "vae", torch_dtype=torch.bfloat16).to(device).eval()
    t5 = T5TextEncoder(schnell_dir / "text_encoder_2", schnell_dir / "tokenizer_2").to(device).eval()
    clip = CLIPTextEncoder(schnell_dir / "text_encoder", schnell_dir / "tokenizer").to(device).eval()

    names = dataset.names if limit <= 0 else dataset.names[:limit]
    name_to_index = {sample_name: index for index, sample_name in enumerate(dataset.names)}
    started = time.perf_counter()
    with h5py.File(output, "a") as handle:
        handle.attrs["encoding"] = ENCODING
        handle.attrs["split"] = split
        handle.attrs["resolution"] = resolution
        for index, name in enumerate(names, start=1):
            if name in handle and handle[name].attrs.get("complete", False):
                continue
            if name in handle:
                del handle[name]

            sample = dataset[name_to_index[name]]
            image = sample["image"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            mask = sample["mask"].unsqueeze(0).repeat(1, 3, 1, 1)
            mask = ((mask - 0.5) / 0.5).to(device=device, dtype=torch.bfloat16)
            texts = _prompts_for(name, prompts)
            with torch.inference_mode():
                z_img = _encode_latent(vae, image)
                z_mask = _encode_latent(vae, mask)
                z_txt_t5 = t5(texts)
                z_txt_clip = clip(texts)

            group = handle.create_group(name)
            group.attrs["name"] = name
            group.create_dataset("z_img", data=_to_bf16(z_img[0]))
            group.create_dataset("z_mask", data=_to_bf16(z_mask[0]))
            group.create_dataset("z_txt_t5", data=_to_bf16(z_txt_t5))
            group.create_dataset("z_txt_clip", data=_to_bf16(z_txt_clip))
            group.create_dataset(
                "prompts",
                data=np.array(texts, dtype=object),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            rgb = (sample["image"].float() * 0.5 + 0.5).clamp(0, 1)
            group.create_dataset(
                "image",
                data=(rgb.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8),
            )
            group.create_dataset(
                "mask",
                data=(sample["mask"][0].numpy() * 255).round().astype(np.uint8),
            )
            group.attrs["complete"] = True
            handle.flush()
            if index == 1 or index % 25 == 0 or index == len(names):
                elapsed = time.perf_counter() - started
                print(f"{split} {index}/{len(names)} {name} ({elapsed:.0f}s)", flush=True)

    print(f"wrote {output}", flush=True)


class CachedDIS5K(Dataset):
    """Reads one cache file written by `build_cache`."""

    def __init__(self, path: Path | str, prompt_strategy: str = "random"):
        if prompt_strategy not in ("random", "first"):
            raise ValueError(f"unknown prompt_strategy {prompt_strategy!r}")
        self.path = Path(path)
        self.prompt_strategy = prompt_strategy
        if not self.path.is_file():
            raise FileNotFoundError(
                f"latent cache not found: {self.path}. "
                "Generate it with flowdis/scripts/cache_latents.sh."
            )
        with h5py.File(self.path, "r") as handle:
            if handle.attrs.get("encoding") != ENCODING:
                raise ValueError(f"{self.path} was not written as {ENCODING}")
            self.keys = sorted(key for key in handle if handle[key].attrs.get("complete", False))
        if not self.keys:
            raise RuntimeError(f"no complete samples in {self.path}")
        self._handle: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.keys)

    def _file(self) -> h5py.File:
        # Open after fork so DataLoader workers do not share one handle.
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        group = self._file()[self.keys[index]]
        prompts = [item.decode() if isinstance(item, bytes) else str(item) for item in group["prompts"][()]]
        choice = 0 if self.prompt_strategy == "first" else random.randrange(len(prompts))
        image = torch.from_numpy(group["image"][()]).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(group["mask"][()]).unsqueeze(0).float() / 255.0
        return {
            "z_img": _from_bf16(group["z_img"][()]),
            "z_mask": _from_bf16(group["z_mask"][()]),
            "z_txt_t5": _from_bf16(group["z_txt_t5"][()])[choice],
            "z_txt_clip": _from_bf16(group["z_txt_clip"][()])[choice],
            "image": (image - 0.5) / 0.5,
            "mask": mask,
            "prompt": prompts[choice],
            "name": str(group.attrs["name"]),
        }


def build_cached_dataloader(
    path: Path | str,
    prompt_strategy: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        CachedDIS5K(path, prompt_strategy=prompt_strategy),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode one DIS5K split into an HDF5 latent cache.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--schnell-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="Encode only the first N images. 0 means the whole split.")
    args = parser.parse_args()
    build_cache(
        data_root=args.data_root,
        schnell_dir=args.schnell_dir,
        split=args.split,
        output=args.output,
        resolution=args.resolution,
        device=args.device,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
