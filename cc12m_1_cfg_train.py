#!/usr/bin/env python3

import argparse
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
import math
import random
from pathlib import Path, PosixPath
import sys

from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
import torch
import torch_xla.debug.metrics as met
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from torchvision.transforms import functional as TF
from tqdm import trange
import wandb
import json
from pprint import pprint
import re

# import os

from CLIP import clip

# Define utility functions

@contextmanager
def train_mode(model, mode=True):
    """A context manager that places a model into training mode and restores
    the previous mode on exit."""
    modes = [module.training for module in model.modules()]
    try:
        yield model.train(mode)
    finally:
        for i, module in enumerate(model.modules()):
            module.training = modes[i]


def eval_mode(model):
    """A context manager that places a model into evaluation mode and restores
    the previous mode on exit."""
    return train_mode(model, False)


@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


# Define the diffusion noise schedule


def get_alphas_sigmas(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


# Define the model (a residual U-Net)


class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResLinearBlock(ResidualBlock):
    def __init__(self, f_in, f_mid, f_out, is_last=False):
        skip = None if f_in == f_out else nn.Linear(f_in, f_out, bias=False)
        super().__init__(
            [
                nn.Linear(f_in, f_mid),
                nn.ReLU(inplace=True),
                nn.Linear(f_mid, f_out),
                nn.ReLU(inplace=True) if not is_last else nn.Identity(),
            ],
            skip,
        )


class Modulation2d(nn.Module):
    def __init__(self, state, feats_in, c_out):
        super().__init__()
        self.state = state
        self.layer = nn.Linear(feats_in, c_out * 2, bias=False)

    def forward(self, input):
        scales, shifts = self.layer(self.state["cond"]).chunk(2, dim=-1)
        return torch.addcmul(shifts[..., None, None], input, scales[..., None, None] + 1)


class ResModConvBlock(ResidualBlock):
    def __init__(self, state, feats_in, c_in, c_mid, c_out, is_last=False):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__(
            [
                nn.Conv2d(c_in, c_mid, 3, padding=1),
                nn.GroupNorm(1, c_mid, affine=False),
                Modulation2d(state, feats_in, c_mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_mid, c_out, 3, padding=1),
                nn.GroupNorm(1, c_out, affine=False) if not is_last else nn.Identity(),
                Modulation2d(state, feats_in, c_out) if not is_last else nn.Identity(),
                nn.ReLU(inplace=True) if not is_last else nn.Identity(),
            ],
            skip,
        )


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)
        self.weight.requires_grad_(False)
        # self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class SelfAttention2d(nn.Module):
    def __init__(self, c_in, n_head=1, dropout_rate=0.1):
        super().__init__()
        assert c_in % n_head == 0
        self.norm = nn.GroupNorm(1, c_in)
        self.n_head = n_head
        self.qkv_proj = nn.Conv2d(c_in, c_in * 3, 1)
        self.out_proj = nn.Conv2d(c_in, c_in, 1)
        self.dropout = nn.Identity()  # nn.Dropout2d(dropout_rate, inplace=True)

    def forward(self, input):
        n, c, h, w = input.shape
        qkv = self.qkv_proj(self.norm(input))
        qkv = qkv.view([n, self.n_head * 3, c // self.n_head, h * w]).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=1)
        scale = k.shape[3] ** -0.25
        att = ((q * scale) @ (k.transpose(2, 3) * scale)).softmax(3)
        y = (att @ v).transpose(2, 3).contiguous().view([n, c, h, w])
        return input + self.dropout(self.out_proj(y))


def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])


class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        c = 128  # The base channel count
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8, c * 8]

        self.mapping_timestep_embed = FourierFeatures(1, 128)
        self.mapping = nn.Sequential(
            ResLinearBlock(512 + 128, 1024, 1024),
            ResLinearBlock(1024, 1024, 1024, is_last=True),
        )

        with torch.no_grad():
            for param in self.mapping.parameters():
                param *= 0.5**0.5

        self.state = {}
        conv_block = partial(ResModConvBlock, self.state, 1024)

        self.timestep_embed = FourierFeatures(1, 16)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(
            scale_factor=2,
            mode="nearest",
            # align_corners=False
        )

        self.net = nn.Sequential(  # 256x256
            conv_block(3 + 16, cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            SkipBlock(
                [
                    self.down,  # 128x128
                    conv_block(cs[0], cs[1], cs[1]),
                    conv_block(cs[1], cs[1], cs[1]),
                    conv_block(cs[1], cs[1], cs[1]),
                    conv_block(cs[1], cs[1], cs[1]),
                    SkipBlock(
                        [
                            self.down,  # 64x64
                            conv_block(cs[1], cs[2], cs[2]),
                            conv_block(cs[2], cs[2], cs[2]),
                            conv_block(cs[2], cs[2], cs[2]),
                            conv_block(cs[2], cs[2], cs[2]),
                            SkipBlock(
                                [
                                    self.down,  # 32x32
                                    conv_block(cs[2], cs[3], cs[3]),
                                    conv_block(cs[3], cs[3], cs[3]),
                                    conv_block(cs[3], cs[3], cs[3]),
                                    conv_block(cs[3], cs[3], cs[3]),
                                    SkipBlock(
                                        [
                                            self.down,  # 16x16
                                            conv_block(cs[3], cs[4], cs[4]),
                                            SelfAttention2d(cs[4], cs[4] // 64),
                                            conv_block(cs[4], cs[4], cs[4]),
                                            SelfAttention2d(cs[4], cs[4] // 64),
                                            conv_block(cs[4], cs[4], cs[4]),
                                            SelfAttention2d(cs[4], cs[4] // 64),
                                            conv_block(cs[4], cs[4], cs[4]),
                                            SelfAttention2d(cs[4], cs[4] // 64),
                                            SkipBlock(
                                                [
                                                    self.down,  # 8x8
                                                    conv_block(cs[4], cs[5], cs[5]),
                                                    SelfAttention2d(cs[5], cs[5] // 64),
                                                    conv_block(cs[5], cs[5], cs[5]),
                                                    SelfAttention2d(cs[5], cs[5] // 64),
                                                    conv_block(cs[5], cs[5], cs[5]),
                                                    SelfAttention2d(cs[5], cs[5] // 64),
                                                    conv_block(cs[5], cs[5], cs[5]),
                                                    SelfAttention2d(cs[5], cs[5] // 64),
                                                    SkipBlock(
                                                        [
                                                            self.down,  # 4x4
                                                            conv_block(cs[5], cs[6], cs[6]),
                                                            SelfAttention2d(cs[6], cs[6] // 64),
                                                            conv_block(cs[6], cs[6], cs[6]),
                                                            SelfAttention2d(cs[6], cs[6] // 64),
                                                            conv_block(cs[6], cs[6], cs[6]),
                                                            SelfAttention2d(cs[6], cs[6] // 64),
                                                            conv_block(cs[6], cs[6], cs[6]),
                                                            SelfAttention2d(cs[6], cs[6] // 64),
                                                            conv_block(cs[6], cs[6], cs[6]),
                                                            SelfAttention2d(cs[6], cs[6] // 64),
                                                            conv_block(cs[6], cs[6], cs[6]),
                                                            SelfAttention2d(cs[6], cs[6] // 64),
                                                            conv_block(cs[6], cs[6], cs[6]),
                                                            SelfAttention2d(cs[6], cs[6] // 64),
                                                            conv_block(cs[6], cs[6], cs[5]),
                                                            SelfAttention2d(cs[5], cs[5] // 64),
                                                            self.up,
                                                        ]
                                                    ),
                                                    conv_block(cs[5] * 2, cs[5], cs[5]),
                                                    SelfAttention2d(cs[5], cs[5] // 64),
                                                    conv_block(cs[5], cs[5], cs[5]),
                                                    SelfAttention2d(cs[5], cs[5] // 64),
                                                    conv_block(cs[5], cs[5], cs[5]),
                                                    SelfAttention2d(cs[5], cs[5] // 64),
                                                    conv_block(cs[5], cs[5], cs[4]),
                                                    SelfAttention2d(cs[4], cs[4] // 64),
                                                    self.up,
                                                ]
                                            ),
                                            conv_block(cs[4] * 2, cs[4], cs[4]),
                                            SelfAttention2d(cs[4], cs[4] // 64),
                                            conv_block(cs[4], cs[4], cs[4]),
                                            SelfAttention2d(cs[4], cs[4] // 64),
                                            conv_block(cs[4], cs[4], cs[4]),
                                            SelfAttention2d(cs[4], cs[4] // 64),
                                            conv_block(cs[4], cs[4], cs[3]),
                                            SelfAttention2d(cs[3], cs[3] // 64),
                                            self.up,
                                        ]
                                    ),
                                    conv_block(cs[3] * 2, cs[3], cs[3]),
                                    conv_block(cs[3], cs[3], cs[3]),
                                    conv_block(cs[3], cs[3], cs[3]),
                                    conv_block(cs[3], cs[3], cs[2]),
                                    self.up,
                                ]
                            ),
                            conv_block(cs[2] * 2, cs[2], cs[2]),
                            conv_block(cs[2], cs[2], cs[2]),
                            conv_block(cs[2], cs[2], cs[2]),
                            conv_block(cs[2], cs[2], cs[1]),
                            self.up,
                        ]
                    ),
                    conv_block(cs[1] * 2, cs[1], cs[1]),
                    conv_block(cs[1], cs[1], cs[1]),
                    conv_block(cs[1], cs[1], cs[1]),
                    conv_block(cs[1], cs[1], cs[0]),
                    self.up,
                ]
            ),
            conv_block(cs[0] * 2, cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            conv_block(cs[0], cs[0], 3, is_last=True),
        )

        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5**0.5

    def forward(self, input, t, clip_embed):
        clip_embed = F.normalize(clip_embed, dim=-1) * clip_embed.shape[-1] ** 0.5
        mapping_timestep_embed = self.mapping_timestep_embed(t[:, None])
        self.state["cond"] = self.mapping(torch.cat([clip_embed, mapping_timestep_embed], dim=1))
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        out = self.net(torch.cat([input, timestep_embed], dim=1))
        self.state.clear()
        return out


@torch.no_grad()
def sample(model, x, steps, eta, extra_args, guidance_scale=1.0):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]
    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
        # with torch.cuda.amp.autocast():
        ## NOTE removed above cuda line with no changes...
        x_in = torch.cat([x, x])
        ts_in = torch.cat([ts, ts])
        clip_embed = extra_args["clip_embed"]
        clip_embed = torch.cat([clip_embed, torch.zeros_like(clip_embed)])
        ### NOTE This concat seems to make the dimensions wrong...
        ####
        v_uncond, v_cond = model(x_in, ts_in * t[i], clip_embed).float().chunk(2)
        #####

        v = v_uncond + guidance_scale * (v_cond - v_uncond)

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = (
                eta * (sigmas[i + 1] ** 2 / sigmas[i] ** 2).sqrt() * (1 - alphas[i] ** 2 / alphas[i + 1] ** 2).sqrt()
            )
            adjusted_sigma = (sigmas[i + 1] ** 2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    # return alphas
    return pred


class TokenizerWrapper:
    def __init__(self, max_len=None):
        self.tokenizer = clip.simple_tokenizer.SimpleTokenizer()
        self.sot_token = self.tokenizer.encoder["<|startoftext|>"]
        self.eot_token = self.tokenizer.encoder["<|endoftext|>"]
        self.context_length = 77
        self.max_len = self.context_length - 2 if max_len is None else max_len

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        result = torch.zeros([len(texts), self.context_length], dtype=torch.long)
        for i, text in enumerate(texts):
            tokens_trunc = self.tokenizer.encode(text)[: self.max_len]
            tokens = [self.sot_token, *tokens_trunc, self.eot_token]
            result[i, : len(tokens)] = torch.tensor(tokens)
        return result


class JsonCaptions(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.indexfile = Path(root) / "index.json"
        self.transform = transform
        self.target_transform = target_transform
        with open(self.indexfile, "r") as f:
            self.dataindex = json.loads(f.read())
            self.data = self.dataindex["data"]
        print(f"Captions Data: found {len(self.data)} images.", file=sys.stderr)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            try:
                text, image_path = self.data[index]
                image = Image.open(Path(self.root) / image_path)
                if self.transform is not None:
                    image = self.transform(image)
                if self.target_transform is not None:
                    text = self.target_transform(text)
                return image, text
            except (
                OSError,
                ValueError,
                Image.DecompressionBombError,
                Image.UnidentifiedImageError,
            ) as err:
                print(
                    f"Bad image, skipping: {index} {self.stems[index]} " f"{type(err).__name__}: {err!s}",
                    file=sys.stderr,
                )
                return self[random.randrange(len(self))]
        except Exception as err:
            print(f"{type(err).__name__}: {err!s}", file=sys.stderr)
            # return self[random.randrange(len(self))]
            raise


class JsonCaptions2(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.indexfile = Path(root) / "index.json"
        self.transform = transform
        self.target_transform = target_transform
        with open(self.indexfile, "r") as f:
            self.dataindex = json.loads(f.read())
            self.data = self.dataindex["data"]
        print(f"Captions Data: found {len(self.data)} images.", file=sys.stderr)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            try:
                datapoint = self.data[index]
                text = datapoint["text"]
                image = Image.open(Path(self.root) / datapoint["filename"])
                font = datapoint["font"]
                text = f"<<<FONT:{font}>>> {text}"

                if self.transform is not None:
                    image = self.transform(image)
                if self.target_transform is not None:
                    text = self.target_transform(text)
                return image, text
            except (
                OSError,
                ValueError,
                Image.DecompressionBombError,
                Image.UnidentifiedImageError,
            ) as err:
                print(
                    f"Bad image, skipping: {index} {image} " f"{type(err).__name__}: {err!s}",
                    file=sys.stderr,
                )
                return self[random.randrange(len(self))]
        except Exception as err:
            print(f"{type(err).__name__}: {err!s}", file=sys.stderr)
            # return self[random.randrange(len(self))]
            raise


class DanbooruCaptions(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.indexfile = Path(root) / "index.json"
        self.transform = transform
        self.target_transform = target_transform
        with open(self.indexfile, "r") as f:
            self.dataindex = json.loads(f.read())
            self.data = self.dataindex["data"]
        print(f"Captions Data: found {len(self.data)} images.", file=sys.stderr)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            try:
                datapoint = self.data[index]
                image = Image.open(Path(self.root) / datapoint["filename"])
                tags = [example["name"] for example in datapoint["tags"]]
                text = (
                    f"A drawing. Rating {datapoint['rating']}, score {datapoint['score']}, and tags {','.join(tags)}."
                )
                # TODO log this text?
                if self.transform is not None:
                    image = self.transform(image)
                if self.target_transform is not None:
                    text = self.target_transform(text)
                return image, text
            except (
                OSError,
                ValueError,
                Image.DecompressionBombError,
                Image.UnidentifiedImageError,
            ) as err:
                print(
                    f"Bad image, skipping: {index} {image} " f"{type(err).__name__}: {err!s}",
                    file=sys.stderr,
                )
                return self[random.randrange(len(self))]
        except Exception as err:
            print(f"{type(err).__name__}: {err!s}", file=sys.stderr)
            # return self[random.randrange(len(self))]
            raise


class ConceptualCaptions(data.Dataset):
    def __init__(self, root, stems_list, transform=None, target_transform=None):
        self.images_root = Path(root) / "images"
        self.texts_root = Path(root) / "texts"
        self.transform = transform
        self.target_transform = target_transform
        # self.stems = sorted(path.stem for path in self.images_root.glob('*/*.jpg'))
        self.stems = [line.rstrip() for line in open(stems_list).readlines()]
        print(f"Conceptual Captions: found {len(self.stems)} images.", file=sys.stderr)

    def _get_image_text(self, stem):
        image = self.images_root / stem[:5] / (stem + ".jpg")
        text = self.texts_root / stem[:5] / (stem + ".txt")
        return image, text

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, index):
        try:
            try:
                image_path, text_path = self._get_image_text(self.stems[index])
                image = Image.open(image_path)
                text = text_path.read_text()
                if self.transform is not None:
                    image = self.transform(image)
                if self.target_transform is not None:
                    text = self.target_transform(text)
                return image, text
            except (
                OSError,
                ValueError,
                Image.DecompressionBombError,
                Image.UnidentifiedImageError,
            ) as err:
                print(
                    f"Bad image, skipping: {index} {self.stems[index]} " f"{type(err).__name__}: {err!s}",
                    file=sys.stderr,
                )
                return self[random.randrange(len(self))]
        except Exception as err:
            print(f"{type(err).__name__}: {err!s}", file=sys.stderr)
            # return self[random.randrange(len(self))]
            raise


class ToMode:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


class LightningDiffusion(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = DiffusionModel()
        self.model_ema = deepcopy(self.model)
        self.clip_model = clip.load("ViT-B/16", "cpu", jit=False)[0].eval().requires_grad_(False)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.lr = 3e-5
        self.eps = 1e-5
        self.weight_decay = 0.01

    def forward(self, *args, **kwargs):
        if self.training:
            return self.model(*args, **kwargs)
        return self.model_ema(*args, **kwargs)

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=self.lr, eps=self.eps, weight_decay=self.weight_decay)
        # return optim.AdamW(self.model.parameters(), lr=5e-6, weight_decay=0.01)

    def eval_batch(self, batch):
        reals, captions = batch
        cond = self.clip_model.encode_text(captions)
        p = torch.rand([reals.shape[0], 1], device=reals.device)
        cond = torch.where(p > 0.2, cond, torch.zeros_like(cond))

        # Sample timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(reals)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None, None]
        sigmas = sigmas[:, None, None, None]
        noise = torch.randn_like(reals)
        noised_reals = reals * alphas + noise * sigmas
        targets = noise * alphas - reals * sigmas

        # Compute the model output and the loss.
        v = self(noised_reals, t, cond)
        return F.mse_loss(v, targets)

    def training_step(self, batch, batch_idx):
        loss = self.eval_batch(batch)
        log_dict = {"train/loss": loss.detach()}
        # task_f1 = pl.metrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes)
        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self.eval_batch(batch)
        log_dict = {"val/loss": loss.detach()}
        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    # def train_dataloader(self):
    #     return super().train_dataloader()

    def on_before_zero_grad(self, *args, **kwargs):
        if self.trainer.global_step < 20000:
            decay = 0.99
        elif self.trainer.global_step < 200000:
            decay = 0.999
        else:
            decay = 0.9999
        ema_update(self.model, self.model_ema, decay)


class DemoCallback(pl.Callback):
    def __init__(self, prompts, prompts_toks):
        super().__init__()
        self.prompts = prompts[:8]
        self.prompts_toks = prompts_toks[:8]

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx, unused=0):
        if trainer.global_step == 0 or trainer.global_step % 1 != 0:
            return

        lines = [f"({i // 4}, {i % 4}) {line}" for i, line in enumerate(self.prompts)]
        lines_text = "\n".join(lines)
        Path("demo_prompts_out.txt").write_text(lines_text)
        noise = torch.randn([16, 3, 256, 256], device=module.device)
        clip_embed = module.clip_model.encode_text(self.prompts_toks.to(module.device))
        with eval_mode(module):
            fakes = sample(module, noise, 1000, 1, {"clip_embed": clip_embed}, guidance_scale=3.0)

        grid = utils.make_grid(fakes, 4, padding=0).cpu()
        image = TF.to_pil_image(grid.add(1).div(2).clamp(0, 1))
        filename = f"demo_{trainer.global_step:08}.png"
        image.save(filename)
        log_dict = {
            "demo_grid": wandb.Image(image),
            "prompts": wandb.Html(f"<pre>{lines_text}</pre>"),
            #'metrics_report': wandb.Html(f'<pre>{metrics_report}</pre>')
        }
        trainer.logger.experiment.log(log_dict, step=trainer.global_step)


class MetricsCallback(pl.Callback):
    def __init__(self, prompts):
        super().__init__()

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx, unused=0):
        if trainer.global_step == 0 or trainer.global_step % 1000 != 0:
            return
        log_dict = {"metrics_report": wandb.Html(f"<pre>{met.metrics_report()}</pre>")}
        trainer.logger.experiment.log(log_dict, step=trainer.global_step)


class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f"{type(err).__name__}: {err!s}", file=sys.stderr)


def worker_init_fn(worker_id):
    random.seed(torch.initial_seed())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_set", type=Path, required=True, help="the training set location")
    p.add_argument("--val_set", type=Path, required=False, help="the val set location")
    p.add_argument("--test_set", type=Path, required=False, help="the test set location")
    p.add_argument("--demo_prompts", type=Path, required=True, help="the demo prompts")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        required=False,
        help="load checkpoint file path",
    )
    p.add_argument(
        "--lightningcheckpoint",
        type=Path,
        default=None,
        required=False,
        help="load lightning mode checkpoint file path",
    )
    p.add_argument(
        "--batchsize",
        type=int,
        default=2,
        required=False,
        help="batchsize for training",
    )
    p.add_argument(
        "--imgsize",
        type=int,
        default=256,
        required=False,
        help="Image size in pixels. Assumes square image",
    )
    p.add_argument(
        "--dataset_mode",
        type=str,
        default="json2",
        required=False,
        help='Dataset mode to use: "conceptual, json, json2, danbooru"',
    )
    p.add_argument(
        "--project_name",
        type=str,
        default="kat-diffusion",
        required=False,
        help="project name for logging",
    )
    args = p.parse_known_args()[0]
    print(f"Starting train on {args.train_set}")

    ### See https://github.com/wandb/client/issues/1994
    # os.environ['WANDB_CONSOLE'] = 'off'

    tf = transforms.Compose(
        [
            ToMode("RGB"),
            transforms.Resize(
                args.imgsize,
                # interpolation=transforms.InterpolationMode.LANCZOS
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.CenterCrop(args.imgsize),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    tok_wrap = TokenizerWrapper()

    def ttf(caption):
        return tok_wrap(caption).squeeze(0)
    ## Choose dataset loader mode.
    if args.dataset_mode == "conceptual":
        fulldata_set = ConceptualCaptions(args.train_set, "stems.txt", transform=tf, target_transform=ttf)
    elif args.dataset_mode == "json":
        fulldata_set = JsonCaptions(args.train_set, transform=tf, target_transform=ttf)
    elif args.dataset_mode == "json2":
        fulldata_set = JsonCaptions2(args.train_set, transform=tf, target_transform=ttf)
    elif args.dataset_mode == "danbooru":
        fulldata_set = DanbooruCaptions(args.train_set, transform=tf, target_transform=ttf)

    if not args.val_set:
        ## Split data 
        train_set, val_set = data.dataset.random_split(fulldata_set, [len(fulldata_set)-len(fulldata_set)//20, len(fulldata_set)//20])                                                                                                            
    else:
        ## Choose dataset loader mode.
        if args.dataset_mode == "conceptual":
            val_set = ConceptualCaptions(args.val_set, "stems.txt", transform=tf, target_transform=ttf)
        elif args.dataset_mode == "json":
            val_set = JsonCaptions(args.val_set, transform=tf, target_transform=ttf)
        elif args.dataset_mode == "json2":
            val_set = JsonCaptions2(args.val_set, transform=tf, target_transform=ttf)
        elif args.dataset_mode == "danbooru":
            val_set = DanbooruCaptions(args.val_set, transform=tf, target_transform=ttf)

    val_dl = data.DataLoader(
        val_set,
        args.batchsize,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        num_workers=96,
        persistent_workers=True,
        pin_memory=True,
    )
    train_dl = data.DataLoader(
        train_set,
        args.batchsize,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        num_workers=96,
        persistent_workers=True,
        pin_memory=True,
    )
    demo_prompts = [line.rstrip() for line in open(args.demo_prompts).readlines()]

    model = LightningDiffusion()
    wandb_logger = pl.loggers.WandbLogger(project=args.project_name)
    wandb_logger.watch(model.model)
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=2500, save_top_k=2, monitor="val/loss")
    demo_callback = DemoCallback(demo_prompts, tok_wrap(demo_prompts))
    lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')

    metrics_callback = MetricsCallback(demo_prompts)
    exc_callback = ExceptionCallback()
    pl.Trainer.add_argparse_args(p)
    p.set_defaults(
        tpu_cores=8,
        num_nodes=1,
        precision="bf16",
        callbacks=[ckpt_callback, exc_callback, metrics_callback, lr_monitor_callback],
        logger=wandb_logger,
        log_every_n_steps=100,
        val_check_interval=.5,
        profiler="simple",
        accumulate_grad_batches={2:2,4:4,8:16,16:32,32:64,64:128},
        max_epochs=10,
    )
    args = p.parse_args()
    trainer = pl.Trainer.from_argparse_args(args)
    wandb.init(config=vars(args), save_code=True, name="Diffusion Run tmp")
    # wandb.config.update(vars(args))

    if args.checkpoint:
        try:
            print(f"Trying torch state_dict model format")
            checkpoint_loaded = torch.load(args.checkpoint, map_location="cpu")
            lightning_model = torch.load(args.lightningcheckpoint, map_location="cpu")
            state_dict_modified = {
                re.sub("net.(.*)", r"model.net.\1", key): value for (key, value) in checkpoint_loaded.items()
            }
            ## Hacky fix for unexpected keys
            for k in ["mapping_timestep_embed.weight", "mapping.0.main.0.weight", "mapping.0.main.0.bias", "mapping.0.main.2.weight", "mapping.0.main.2.bias", "mapping.0.skip.weight", "mapping.1.main.0.weight", "mapping.1.main.0.bias", "mapping.1.main.2.weight", "mapping.1.main.2.bias", "timestep_embed.weight"]:
                _ = state_dict_modified.pop(k, None)
            lightning_state_dict = deepcopy(lightning_model["state_dict"])
            lightning_state_dict.update(state_dict_modified)
            del checkpoint_loaded
            del lightning_model
            model.load_state_dict(lightning_state_dict)
            trainer.fit(model, train_dl, val_dl)
        except RuntimeError:
            print(f"Trying lightning model format")
            trainer.fit(model, train_dl, val_dl, ckpt_path=args.checkpoint)
    else:
        trainer.fit(model, train_dl, val_dl)

if __name__ == "__main__":
    wandb.require(experiment="service")
    wandb.setup()
    main()
