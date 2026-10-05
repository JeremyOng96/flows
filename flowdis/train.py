import os
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange, repeat
import hydra
from omegaconf import DictConfig
from torch import Tensor, nn
from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel
from diffusers import AutoencoderKL
from scipy.special import betaincinv # inverse of the incomplete beta function
from typing import List
from flowdis.data.cache import build_cached_dataloader
from flowdis.data.dataloader import DIS5KParams, build_dataloader
from flowdis.metrics.metric_utils import SegmentationMetrics
from flowdis.model.flux import load_flux



def pack_latent(latent: Tensor) -> Tensor:
    """Fold each 2x2 patch of a 16-channel VAE latent into one FLUX token (64 channels)."""
    return rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)


def unpack_latent(tokens: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        tokens,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=height // 2,
        w=width // 2,
        c=16,
        ph=2,
        pw=2,
    )

def image_position_ids(latent: Tensor) -> Tensor:
    _, _, height, width = latent.shape
    h, w = height // 2, width // 2
    ids = torch.zeros(h, w, 3, device=latent.device, dtype=latent.dtype)
    ids[..., 1] = ids[..., 1] + torch.arange(h, device=latent.device, dtype=latent.dtype)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(w, device=latent.device, dtype=latent.dtype)[None, :]
    return repeat(ids, "h w c -> b (h w) c", b=latent.shape[0])

class T5TextEncoder(nn.Module):
    """T5 encoder used by FLUX. Returns token states (B, 512, 4096)."""

    def __init__(self, model_dir: Path, tokenizer_dir: Path, max_length: int = 512):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, legacy=True)
        self.model = T5EncoderModel.from_pretrained(model_dir, torch_dtype=torch.bfloat16)

    def forward(self, text: list[str]) -> Tensor:
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self.model.device)
        return self.model(input_ids=input_ids).last_hidden_state


class CLIPTextEncoder(nn.Module):
    """CLIP text encoder used by FLUX. Returns the pooled vector (B, 768)."""

    def __init__(self, model_dir: Path, tokenizer_dir: Path, max_length: int = 77):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.model = CLIPTextModel.from_pretrained(model_dir, torch_dtype=torch.bfloat16)

    def forward(self, text: list[str]) -> Tensor:
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self.model.device)
        return self.model(input_ids=input_ids).pooler_output


def _freeze(module: nn.Module) -> None:
    module.eval()
    module.requires_grad_(False)


class FlowDIS(L.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.use_cache = self.config.use_cache
        self.beta = torch.distributions.Beta(self.config.beta_alpha, self.config.beta_beta)
        # Loaded on CPU. Lightning moves the module; do not shadow Lightning's `device`.
        self.model = load_flux(
            Path(self.config.schnell_dir) / "flux1-schnell.safetensors",
            device="cpu",
        )
        # Only Flux is a Lightning submodule (FSDP shards it). VAE/T5/CLIP stay
        # off the tree so they are not gathered onto every GPU.
        vae = AutoencoderKL.from_pretrained(self.config.vae_dir, torch_dtype=torch.bfloat16)
        t5 = T5TextEncoder(self.config.t5_dir, self.config.t5_tokenizer_dir)
        clip = CLIPTextEncoder(self.config.clip_dir, self.config.clip_tokenizer_dir)
        _freeze(vae)
        _freeze(t5)
        _freeze(clip)
        object.__setattr__(self, "vae", vae)
        object.__setattr__(self, "t5", t5.to("cpu"))
        object.__setattr__(self, "clip", clip.to("cpu"))

    def on_fit_start(self) -> None:
        # Flux is FSDP-sharded (~1.5B/GPU). Encoders fit beside it on a 40GB card.
        self.vae.to(device=self.device, dtype=torch.bfloat16)
        self.t5.to(device=self.device)
        self.clip.to(device=self.device)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        self.t5.eval()
        self.clip.eval()
        return self

    def _encode_latent(self, pixels: Tensor) -> Tensor:
        shift = self.vae.config.shift_factor
        scale = self.vae.config.scaling_factor
        return (self.vae.encode(pixels).latent_dist.mode() - shift) * scale

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        t5_txt: Tensor,
        t5_txt_ids: Tensor,
        timesteps: Tensor,
        clip_txt: Tensor,
    ) -> Tensor:
        return self.model(img=img, img_ids=img_ids, txt=t5_txt, txt_ids=t5_txt_ids, timesteps=timesteps, y=clip_txt)

    def flow_matching_loss(
        self,
        batch: dict
    ):
        if self.use_cache:
            z_img = batch["z_img"].to(device=self.device)
            z_mask = batch["z_mask"].to(device=self.device)
            z_txt_t5 = batch["z_txt_t5"].to(device=self.device)
            z_txt_clip = batch["z_txt_clip"].to(device=self.device)
        else:
            image = batch["image"].to(device=self.device, dtype=torch.bfloat16)
            mask = batch["mask"].repeat(1, 3, 1, 1).to(device=self.device, dtype=torch.bfloat16) # repeats in the channel axis
            mask = (mask - 0.5) / 0.5
            z_img = self._encode_latent(image)
            z_mask = self._encode_latent(mask)
            z_txt_t5 = self.t5(list(batch["prompt"])).to(device=self.device) # generates seq2seq tokens
            z_txt_clip = self.clip(list(batch["prompt"])).to(device=self.device) # generates global vector 

        time = self.beta.sample((z_img.shape[0],)).to(device=z_img.device, dtype=z_img.dtype)
        z_t = (1 - time.view(-1, 1, 1, 1)) * z_mask + time.view(-1, 1, 1, 1) * z_img
        img = torch.cat((pack_latent(z_t), pack_latent(z_img)), dim=-1)
        target = pack_latent(z_img - z_mask)
        img_ids = image_position_ids(z_img)
        t5_txt_ids = torch.zeros(z_txt_t5.shape[0], z_txt_t5.shape[1], 3, device=z_txt_t5.device)

        predictions = self.forward(
            img,
            img_ids,
            z_txt_t5,
            t5_txt_ids,
            time,
            z_txt_clip,
        )

        return F.mse_loss(predictions.float(), target.float())

    def training_step(self, batch: dict) -> Tensor:
        loss = self.flow_matching_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_metrics = SegmentationMetrics()

    @torch.no_grad()
    def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
        loss = self.flow_matching_loss(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        if self.trainer.sanity_checking:
            return loss
        image = (batch["image"].float() * 0.5 + 0.5).clamp(0, 1)
        prediction = self.flow_matching_integration(image, list(batch["prompt"]))
        self._val_metrics.update(prediction, batch["mask"])
        if batch_idx == 0:
            self._log_segmentation(image, prediction, batch)
        return loss

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking or not self._val_metrics._mae:
            return
        scores = self._validation_scores()
        for name, value in scores.items():
            self.log(f"val/{name}", value, prog_bar=True, sync_dist=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
            betas=tuple(self.config.optimizer.betas),
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(self.config.optimizer.milestones),
            gamma=self.config.optimizer.gamma,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def _loader_params(self, split: str, prompt_strategy: str) -> DIS5KParams:
        data = self.config.data
        return DIS5KParams(
            root=data.root,
            split=split,
            resolution=data.resolution,
            use_language_pairing=data.use_language_pairing,
            prompt_strategy=prompt_strategy,
            mean=data.mean,
            std=data.std,
        )

    def _cached_loader(self, split: str, prompt_strategy: str, shuffle: bool):
        return build_cached_dataloader(
            Path(self.config.cache_dir) / f"{split}.h5",
            prompt_strategy=prompt_strategy,
            batch_size=self.config.loader.batch_size,
            num_workers=self.config.loader.num_workers,
            shuffle=shuffle,
        )

    def _pixel_loader(self, split: str, prompt_strategy: str, *, shuffle: bool, use_paip: bool):
        return build_dataloader(
            self._loader_params(split, prompt_strategy),
            batch_size=self.config.loader.batch_size,
            num_workers=self.config.loader.num_workers,
            shuffle=shuffle,
            use_paip=use_paip,
            seed=self.config.loader.seed,
        )

    def _test_splits(self) -> list[str]:
        splits = self.config.data.get("test_splits")
        if splits:
            return list(splits)
        return [self.config.data.val_split]

    def train_dataloader(self):
        if self.use_cache:
            return self._cached_loader(
                self.config.data.train_split,
                self.config.data.train_prompt_strategy,
                shuffle=True,
            )
        return self._pixel_loader(
            self.config.data.train_split,
            self.config.data.train_prompt_strategy,
            shuffle=True,
            use_paip=True,
        )

    def val_dataloader(self):
        if self.use_cache:
            return self._cached_loader(
                self.config.data.val_split,
                self.config.data.val_prompt_strategy,
                shuffle=False,
            )
        return self._pixel_loader(
            self.config.data.val_split,
            self.config.data.val_prompt_strategy,
            shuffle=False,
            use_paip=False,
        )

    def test_dataloader(self):
        prompt_strategy = self.config.data.get("test_prompt_strategy", self.config.data.val_prompt_strategy)
        return [
            self._pixel_loader(split, prompt_strategy, shuffle=False, use_paip=False)
            for split in self._test_splits()
        ]

    def on_test_epoch_start(self) -> None:
        self._test_split_names = self._test_splits()
        self._test_metrics = [SegmentationMetrics() for _ in self._test_split_names]

    @torch.no_grad()
    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        image = (batch["image"].float() * 0.5 + 0.5).clamp(0, 1)
        prediction = self.flow_matching_integration(image, list(batch["prompt"]))
        self._test_metrics[dataloader_idx].update(prediction, batch["mask"])
        if batch_idx == 0:
            split = self._test_split_names[dataloader_idx]
            self._log_segmentation(image, prediction, batch, key=f"test/{split}/segmentation")

    def on_test_epoch_end(self) -> None:
        te_names = {"DIS-TE1", "DIS-TE2", "DIS-TE3", "DIS-TE4"}
        gathered = [self._gathered_metrics(metrics) for metrics in self._test_metrics]
        for split, metrics in zip(self._test_split_names, gathered):
            for name, value in metrics.compute().items():
                self.log(f"test/{split}/{name}", value, sync_dist=False)
        te = [metrics for split, metrics in zip(self._test_split_names, gathered) if split in te_names]
        if len(te) == 4:
            combined = SegmentationMetrics()
            for metrics in te:
                combined._mae.extend(metrics._mae)
                combined._weighted_f.extend(metrics._weighted_f)
                combined._structure.extend(metrics._structure)
                combined._f_curves.extend(metrics._f_curves)
                combined._e_curves.extend(metrics._e_curves)
            for name, value in combined.compute().items():
                self.log(f"test/DIS-TE/{name}", value, sync_dist=False)


    def flow_matching_integration(
        self,
        image: Tensor,
        prompt: List[str],
        ):
        """
        This method performs inference in latent space using flow matching.

        For i = N-1,... 1, 0. The Euler integration method decodes the latent with the VAE decoder
        to produce the predicted mask.

        """
        q = torch.linspace(0, 1, self.config.num_steps + 1)
        time = torch.as_tensor(
            betaincinv(self.config.beta_alpha, self.config.beta_beta, q.numpy()),
            device=self.device,
            dtype=torch.float32,
        )
        image = ((image - 0.5) / 0.5).to(device=self.device, dtype=self.vae.dtype)
        shift = self.vae.config.shift_factor
        scale = self.vae.config.scaling_factor
        latent_image = self._encode_latent(image)
        latent_prompt_t5 = self.t5(prompt).to(device=self.device)
        latent_prompt_clip = self.clip(prompt).to(device=self.device)
        mask = pack_latent(latent_image)
        image_tokens = pack_latent(latent_image)
        mask_ids = image_position_ids(latent_image)
        t5_txt_ids = torch.zeros(latent_prompt_t5.shape[0], latent_prompt_t5.shape[1], 3, device=self.device)

        for i in reversed(range(self.config.num_steps)):
            t_curr = time[i + 1].expand(mask.shape[0]).to(dtype=mask.dtype)
            velocity = self.model(
                torch.cat((mask, image_tokens), dim=-1),
                mask_ids,
                latent_prompt_t5,
                t5_txt_ids,
                t_curr,
                latent_prompt_clip,
            )
            mask = mask + velocity * (time[i] - time[i + 1])

        mask = unpack_latent(mask, latent_image.shape[-2], latent_image.shape[-1])
        mask = mask / scale + shift
        # Algorithm 1: decode RGB, map VAE [-1, 1] to [0, 1], average, then clip.
        mask = self.vae.decode(mask.to(dtype=self.vae.dtype)).sample
        return ((mask + 1) / 2).mean(dim=1).clamp(0, 1)

    def _gathered_metrics(self, metrics: SegmentationMetrics) -> SegmentationMetrics:
        if self.trainer.world_size == 1:
            return metrics
        gathered: list[SegmentationMetrics | None] = [None] * self.trainer.world_size
        dist.all_gather_object(gathered, metrics)
        merged = SegmentationMetrics()
        for item in gathered:
            merged._mae.extend(item._mae)
            merged._weighted_f.extend(item._weighted_f)
            merged._structure.extend(item._structure)
            merged._f_curves.extend(item._f_curves)
            merged._e_curves.extend(item._e_curves)
        return merged

    def _validation_scores(self) -> dict[str, float]:
        return self._gathered_metrics(self._val_metrics).compute()

    @torch.no_grad()
    def _log_segmentation(
        self, image: Tensor, prediction: Tensor, batch: dict, key: str = "val/segmentation"
    ) -> None:
        import wandb

        panels = []
        count = min(4, image.shape[0])
        for index, caption in enumerate(batch["prompt"][:count]):
            rgb = (image[index].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            ground_truth = np.stack([batch["mask"][index, 0].cpu().numpy()] * 3, axis=-1)
            predicted = np.stack([prediction[index].cpu().numpy()] * 3, axis=-1)
            side_by_side = np.concatenate(
                [rgb, (ground_truth * 255).astype("uint8"), (predicted * 255).astype("uint8")],
                axis=1,
            )
            panels.append(wandb.Image(side_by_side, caption=str(caption)[:160]))
        self.logger.experiment.log({key: panels})


def _accumulate_grad_batches(cfg: DictConfig) -> int:
    """Keep the paper's global batch of 32 across whatever GPU count is visible."""
    target = int(getattr(cfg.loader, "global_batch_size", 32))
    per_gpu = int(cfg.loader.batch_size)
    devices = cfg.trainer.devices
    if devices == "auto":
        n_dev = torch.cuda.device_count() or 1
    elif isinstance(devices, (list, tuple)):
        n_dev = len(devices)
    else:
        n_dev = int(devices)
    n_dev *= int(cfg.trainer.num_nodes)
    return max(1, target // (per_gpu * max(n_dev, 1)))


def _load_wandb_env(cfg: DictConfig) -> None:
    env_file = Path(cfg.wandb.env_file)
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            if not line.strip() or line.strip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
    # The credentials file is named `.wandb`. The client treats that path as its run directory.
    os.environ["WANDB_DIR"] = str(cfg.root)
    os.environ["WANDB_USE_DOT_WANDB"] = "false"
    os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"


def _trainer_strategy(cfg: DictConfig):
    name = str(cfg.trainer.strategy)
    if name != "fsdp":
        return name
    from lightning.pytorch.strategies import FSDPStrategy

    from flowdis.model.flux import DoubleStreamBlock, SingleStreamBlock

    blocks = {DoubleStreamBlock, SingleStreamBlock}
    return FSDPStrategy(
        sharding_strategy="FULL_SHARD",
        auto_wrap_policy=blocks,
        activation_checkpointing_policy=blocks,
    )


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    from lightning.pytorch.loggers import WandbLogger

    _load_wandb_env(cfg)
    model = FlowDIS(cfg)
    logger = False
    if not cfg.wandb.get("disabled", False):
        logger = WandbLogger(project=cfg.wandb.project, save_dir=cfg.root)
    trainer_kwargs = dict(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        strategy=_trainer_strategy(cfg),
        max_steps=cfg.trainer.max_steps,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        accumulate_grad_batches=_accumulate_grad_batches(cfg),
        logger=logger,
    )
    for key in ("num_sanity_val_steps", "limit_val_batches", "limit_train_batches"):
        if key in cfg.trainer:
            trainer_kwargs[key] = cfg.trainer[key]
    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(model)
    if cfg.trainer.get("run_test", False):
        trainer.test(model)


if __name__ == "__main__":
    main()
