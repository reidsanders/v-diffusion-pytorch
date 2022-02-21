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

    model = get_model(args.model)()
    _, side_y, side_x = model.shape
    checkpoint = args.checkpoint
    if not checkpoint:
        checkpoint = MODULE_DIR / f"checkpoints/{args.model}.pth"

    checkpoint_loaded = torch.load(checkpoint, map_location="cpu")
    lightning_model = torch.load(target, map_location="cpu")
    lightning_model[""]
    import ipdb; ipdb.set_trace()


    state_dict_modified = {
        re.sub("model.(.*)", r"\1", key): value for (key, value) in checkpoint_loaded["state_dict"].items()
    }

    checkpoint_example = MODULE_DIR / f"checkpoints/{args.model}.pth"
    checkpoint_example_keys = torch.load(checkpoint_example, map_location="cpu").keys()
    checkpoint_modified = {
        key: value for (key, value) in checkpoint_modified.items() if key in checkpoint_example_keys
    }
    try:
        model.load_state_dict(checkpoint_modified)
    except RuntimeError:
        import ipdb
        ipdb.set_trace()

    try:
        model.load_state_dict(checkpoint_loaded)
    except RuntimeError:
        print("Runtime error loading state dict, Trying lightning naming schema")
        checkpoint_modified = {
            re.sub("model.(.*)", r"\1", key): value for (key, value) in checkpoint_loaded["state_dict"].items()
        }

        checkpoint_example = MODULE_DIR / f"checkpoints/{args.model}.pth"
        checkpoint_example_keys = torch.load(checkpoint_example, map_location="cpu").keys()
        checkpoint_modified = {
            key: value for (key, value) in checkpoint_modified.items() if key in checkpoint_example_keys
        }
        try:
            model.load_state_dict(checkpoint_modified)
        except RuntimeError:
            import ipdb
            ipdb.set_trace()

    # TODO convert cc12m to lightning schema


if __name__ == "__main__":
    main()
