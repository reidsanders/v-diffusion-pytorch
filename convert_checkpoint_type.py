#!/usr/bin/env python3

"""Convert ."""

import argparse
from functools import partial
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import trange
import re
from copy import deepcopy
from diffusion import get_model, get_models, sampling, utils

MODULE_DIR = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("checkpoint", type=str, help="the checkpoint to use")
    p.add_argument("target", type=str, help="the checkpoint to rename to")
    p.add_argument("--mode", type=str, help="lightning or cfg")
    p.add_argument("--model", type=str, default="cc12m_1_cfg", choices=["cc12m_1_cfg"], help="the model to use")
    p.add_argument("--outdir", type=str, default="./checkpoints_mod/", help="Directory to save output files to")
    args = p.parse_args()

    checkpoint_loaded = torch.load(args.checkpoint, map_location="cpu")
    lightning_model = torch.load(args.target, map_location="cpu")

    state_dict_modified = {
        re.sub("net.(.*)", r"model.net.\1", key): value for (key, value) in checkpoint_loaded.items()
    }

    lightning_model["state_dict"] = state_dict_modified
    lightning_model.save(Path(args.outdir) / "checkpoint_lightning.pth")
    #import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()
