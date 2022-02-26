#!/usr/bin/env python3

import argparse
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
import math
import random
from pathlib import Path
import sys
import os
import re
import shlex
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
from CLIP import clip
from diffusion import sampling
from diffusion import utils as diffusionutils
from dataloaders import DanbooruCaptions, DrawtextCaptions, ConceptualCaptions, GoodbotCaptions, JsonTextCaptions

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
def cfg_sample(model, steps, eta, method="ddim", batchsize=1):
    """Draws samples from a model given starting noise."""

    _, side_y, side_x = model.shape

    zero_embed = torch.zeros([1, model.clip_model.visual.output_dim], device=model.device)
    target_embeds, weights = [zero_embed], []

    # normalize = transforms.Normalize(
    #     mean=[0.48145466, 0.4578275, 0.40821073],
    #     std=[0.26862954, 0.26130258, 0.27577711],
    # )

    # for prompt in args.prompts:
    #     txt, weight = parse_prompt(prompt)
    #     target_embeds.append(clip_model.encode_text(clip.tokenize(txt).to(device)).float())
    #     weights.append(weight)

    # for prompt in args.images:
    #     path, weight = parse_prompt(prompt)
    #     img = Image.open(diffusionutils.fetch(path)).convert("RGB")
    #     clip_size = clip_model.visual.input_resolution
    #     img = resize_and_center_crop(img, (clip_size, clip_size))
    #     batch = TF.to_tensor(img)[None].to(device)
    #     embed = F.normalize(clip_model.encode_image(normalize(batch)).float(), dim=-1)
    #     target_embeds.append(embed)
    #     weights.append(weight)
    
    # torch.manual_seed(args.seed)

    def cfg_model_fn(x, t):
        n = x.shape[0]
        n_conds = len(target_embeds)
        x_in = x.repeat([n_conds, 1, 1, 1])
        t_in = t.repeat([n_conds])
        clip_embed_in = torch.cat([*target_embeds]).repeat_interleave(n, 0)
        vs = model(x_in, t_in, clip_embed_in).view([n_conds, n, *x.shape[1:]])
        v = vs.mul(weights[:, None, None, None, None]).sum(0)
        return v

    def run(x, steps):
        if method == "ddpm":
            return sampling.sample(cfg_model_fn, x, steps, 1.0, {})
        if method == "ddim":
            return sampling.sample(cfg_model_fn, x, steps, eta, {})
        if method == "prk":
            return sampling.prk_sample(cfg_model_fn, x, steps, {})
        if method == "plms":
            return sampling.plms_sample(cfg_model_fn, x, steps, {})
        assert False

    def run_all(n, batchsize):
        x = torch.randn([n, 3, side_y, side_x], device=model.device)
        t = torch.linspace(1, 0, steps + 1, device=model.device)[:-1]
        steps = diffusionutils.get_spliced_ddpm_cosine_schedule(t)
        for i in trange(0, n, batchsize):
            cur_batch_size = min(n - i, batchsize)
            outs = run(x[i : i + cur_batch_size], steps)
            # for j, out in enumerate(outs):
            #     img = diffusionutils.to_pil_image(out)
        return outs

    weights = torch.tensor([1 - sum(weights), *weights], device=model.device)
    outs = run_all(steps, batchsize)
    return outs


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


class ToMode:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


class LightningDiffusion(pl.LightningModule):
    def __init__(
        self, epochs=20, steps_per_epoch=10000, lr=3e-5, eps=1e-5, gamma=0.95, weight_decay=0.01, scheduler=None
    ):
        super().__init__()
        self.model = DiffusionModel()
        self.model_ema = deepcopy(self.model)
        self.clip_model = clip.load("ViT-B/16", "cpu", jit=False)[0].eval().requires_grad_(False)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.scheduler = scheduler

    def forward(self, *args, **kwargs):
        if self.training:
            return self.model(*args, **kwargs)
        return self.model_ema(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, eps=self.eps, weight_decay=self.weight_decay)
        if self.scheduler == "onecyclelr":
            lr_scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, self.lr * 25, epochs=self.epochs, steps_per_epoch=self.steps_per_epoch
            )
        elif self.scheduler == "cosineannealingwarmrestarts":
            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1, last_epoch=self.epochs)
        elif self.scheduler == "exponentiallr":
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        else:
            lr_scheduler = self.scheduler
        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "step",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 10,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "train/loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            # "name": "Reduce on Plateau Scheduler",
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }
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
        # TODO use val text

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx, unused=0):
        if trainer.global_step == 0 or trainer.global_step % 1 != 0:
            return
        print(f"Running Demo Sampling")
        lines = [f"({i // 4}, {i % 4}) {line}" for i, line in enumerate(self.prompts)]
        lines_text = "\n".join(lines)
        Path("demo_prompts_out.txt").write_text(lines_text)
        noise = torch.randn([16, 3, 256, 256], device=module.device)
        clip_embed = module.clip_model.encode_text(self.prompts_toks.to(module.device))
        with eval_mode(module):
            # fakes = sample(module, noise, 1000, 1, {"clip_embed": clip_embed}, guidance_scale=3.0)
            fakes = cfg_sample(module, 1000, 1, batchsize=16)

        grid = utils.make_grid(fakes, 4, padding=0).cpu()
        image = TF.to_pil_image(grid.add(1).div(2).clamp(0, 1))
        filename = f"demo_{trainer.global_step:08}.png"
        image.save(filename)
        print(f"Saved demo image to: {filename}")
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


def get_orig_cmd(max_width=80, full_python_path=False):
    """
    form
    Return the original command line string that can be replayed
    nicely and wrapped for 80 char width
    Args:
        - max_width: the width to wrap for. defaults to 80
        - full_python_path: whether to replicate the full path
          or just the last part (i.e. `python`). default to `False`
    """

    cmd = []

    # deal with critical env vars
    # env_keys = ["CUDA_VISIBLE_DEVICES"]
    env_keys = []
    for key in env_keys:
        val = os.environ.get(key, None)
        if val is not None:
            cmd.append(f"{key}={val}")

    # python executable (not always needed if the script is executable)
    python = sys.executable if full_python_path else sys.executable.split("/")[-1]
    cmd.append(python)

    # now the normal args
    cmd += list(map(shlex.quote, sys.argv))

    # split up into up to MAX_WIDTH lines with shell multi-line escapes
    lines = []
    current_line = ""
    while len(cmd) > 0:
        current_line += f"{cmd.pop(0)} "
        if len(cmd) == 0 or len(current_line) + len(cmd[0]) + 1 > max_width - 1:
            lines.append(current_line)
            current_line = ""
    return " ".join(lines)


def rename_lightning_checkpoint_keys(checkpoint, lightning_state_dict):
    state_dict_modified = {re.sub("net.(.*)", r"model.net.\1", key): value for (key, value) in checkpoint.items()}
    ## Hacky fix for unexpected keys
    for k in [
        "mapping_timestep_embed.weight",
        "mapping.0.main.0.weight",
        "mapping.0.main.0.bias",
        "mapping.0.main.2.weight",
        "mapping.0.main.2.bias",
        "mapping.0.skip.weight",
        "mapping.1.main.0.weight",
        "mapping.1.main.0.bias",
        "mapping.1.main.2.weight",
        "mapping.1.main.2.bias",
        "timestep_embed.weight",
    ]:
        _ = state_dict_modified.pop(k, None)
    lightning_state_dict.update(state_dict_modified)
    return lightning_state_dict


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
        "--batchsize",
        type=int,
        default=2,
        required=False,
        help="batchsize for training",
    )
    p.add_argument(
        "--scheduler_epochs",
        type=int,
        default=20,
        required=False,
        help="epochs to pass to lr scheduler",
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
        default="drawtext",
        const="drawtext",
        required=False,
        nargs="?",
        choices=("conceptual", "drawtext", "text", "danbooru", "goodbot"),
        help="choose dataset loader mode (default: %(default)s)",
    )
    p.add_argument(
        "--project_name",
        type=str,
        default="kat-diffusion",
        required=False,
        help="project name for logging",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        required=False,
        help="starting lr",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        required=False,
        help="exponential decay gamma for lr",
    )
    p.add_argument(
        "--scheduler",
        default=None,
        const=None,
        required=False,
        nargs="?",
        choices=("cosineannealingwarmrestarts", "exponentiallr", "onecyclelr"),
        help="choose dataset loader mode (default: %(default)s)",
    )
    p.add_argument(
        "--restore_train_state",
        action="store_true",
        default=False,
        required=False,
        help="restore lightning training state",
    )
    args = p.parse_known_args()[0]
    print(f"Starting train on {args.train_set}")

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
    elif args.dataset_mode == "drawing":
        fulldata_set = DrawtextCaptions(args.train_set, transform=tf, target_transform=ttf)
    elif args.dataset_mode == "text":
        fulldata_set = JsonTextCaptions(args.train_set, transform=tf, target_transform=ttf)
    elif args.dataset_mode == "danbooru":
        fulldata_set = DanbooruCaptions(args.train_set, transform=tf, target_transform=ttf)
    elif args.dataset_mode == "goodbot":
        fulldata_set = GoodbotCaptions(args.train_set, transform=tf, target_transform=ttf)

    if not args.val_set:
        ## Split data
        train_set, val_set = data.dataset.random_split(
            fulldata_set, [len(fulldata_set) - len(fulldata_set) // 20, len(fulldata_set) // 20]
        )
    else:
        train_set = fulldata_set
        ## Choose dataset loader mode.
        if args.dataset_mode == "conceptual":
            val_set = ConceptualCaptions(args.val_set, "stems.txt", transform=tf, target_transform=ttf)
        elif args.dataset_mode == "drawing":
            val_set = DrawtextCaptions(args.val_set, transform=tf, target_transform=ttf)
        elif args.dataset_mode == "text":
            val_set = JsonTextCaptions(args.val_set, transform=tf, target_transform=ttf)
        elif args.dataset_mode == "danbooru":
            val_set = DanbooruCaptions(args.val_set, transform=tf, target_transform=ttf)
        elif args.dataset_mode == "goodbot":
            val_set = GoodbotCaptions(args.val_set, transform=tf, target_transform=ttf)

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

    model = LightningDiffusion(
        epochs=args.scheduler_epochs,
        steps_per_epoch=len(train_dl),
        lr=args.lr,
        gamma=args.gamma,
        scheduler=args.scheduler,
    )
    wandb_logger = pl.loggers.WandbLogger(project=args.project_name)
    wandb_logger.watch(model.model)
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=2500, save_top_k=2, monitor="val/loss")
    # demo_callback = DemoCallback(demo_prompts, tok_wrap(demo_prompts))
    lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

    metrics_callback = MetricsCallback(demo_prompts)
    exc_callback = ExceptionCallback()
    ## Load lightning argparse args
    pl.Trainer.add_argparse_args(p)
    p.set_defaults(
        tpu_cores=8,
        num_nodes=1,
        precision="bf16",
        callbacks=[ckpt_callback, exc_callback, metrics_callback, lr_monitor_callback],
        logger=wandb_logger,
        log_every_n_steps=100,
        val_check_interval=0.5,
        accumulate_grad_batches={1: 1, 3: 2, 6: 4, 8: 8, 16: 32, 32: 64, 64: 128, 100: 256},
        max_epochs=10,
    )
    args = p.parse_args()
    trainer = pl.Trainer.from_argparse_args(args)
    wandb.init(config=vars(args), save_code=True, name="Diffusion Run")
    for k, v in vars(args).items():
        wandb.config[str(k)] = v
    wandb.config["command"] = get_orig_cmd()

    ### Load checkpoint. There are different naming schemes, so this handles different options
    if args.checkpoint:
        print(f"Loading checkpoint {args.checkpoint}")
        if args.restore_train_state:
            trainer.fit(model, train_dl, val_dl, ckpt_path=args.checkpoint)
        else:
            try:
                ## Try lightning model format
                model.load_from_checkpoint(args.checkpoint)
            except KeyError:
                print(f"Falling back to state_dict loading")
                checkpoint_loaded = torch.load(args.checkpoint, map_location="cpu")
                lightning_state_dict = rename_lightning_checkpoint_keys(checkpoint_loaded, model.state_dict())
                model.load_state_dict(lightning_state_dict)
            trainer.fit(model, train_dl, val_dl)
    else: 
        trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    # Fix crashes on multiple tpu cores, but breaks stdout logging
    ### See https://github.com/wandb/client/issues/1994
    # os.environ['WANDB_CONSOLE'] = 'off'
    wandb.require(experiment="service")
    wandb.setup()
    main()
