#!/usr/bin/env python3
import argparse
import cfg_sample
import csv
import random


def main(args):
    data = list(import_proverb_dataset(args.dataset))
    print(f"tags: {args.tags}")
    print(f"Datalen {len(data)}. Example: {data[0]}")
    print(data[0:20])

    def gen_hint_combos(tags, chance):
        hint_prompt = ""
        for tag in tags:
            if random.random() < chance:
                hint_prompt = f"{hint_prompt} | {tag}"
        return hint_prompt

    tag_combinations = [gen_hint_combos(args.tags, args.tag_threshold) for i in range(1, 5)]
    for example in data:
        for extra_tags in tag_combinations:
            args.prompts = [f"{example}{extra_tags}:5"]
            print(f"Trying cfg_sample with: {args.prompts}")
            cfg_sample.main(args)


def import_proverb_dataset(dataset_path: str):
    with open(dataset_path, encoding="utf8") as fIn:
        reader = csv.reader(fIn, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        next(reader)  # Skip top column name
        for row in reader:
            yield row[1]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--images", type=str, default=[], nargs="*", metavar="IMAGE", help="the image prompts")
    p.add_argument("--batch-size", "-bs", type=int, default=1, help="the number of images per batch")
    p.add_argument("--checkpoint", type=str, help="the checkpoint to use")
    p.add_argument("--device", type=str, help="the device to use")
    p.add_argument("--eta", type=float, default=0.0, help="the amount of noise to add during sampling (0-1)")
    p.add_argument("--init", type=str, help="the init image")
    p.add_argument(
        "--method",
        type=str,
        default="plms",
        choices=["ddpm", "ddim", "prk", "plms", "pie", "plms2"],
        help="the sampling method to use",
    )
    p.add_argument("--model", type=str, default="cc12m_1_cfg", choices=["cc12m_1_cfg"], help="the model to use")
    p.add_argument("-n", type=int, default=1, help="the number of images to sample")
    p.add_argument("--seed", type=int, default=0, help="the random seed")
    p.add_argument("--size", type=int, nargs=2, help="the output image size")
    p.add_argument(
        "--starting-timestep", "-st", type=float, default=0.9, help="the timestep to start at (used with init images)"
    )
    p.add_argument("--steps", type=int, default=50, help="the number of timesteps")
    p.add_argument("--outdir", type=str, default="./generated-images/", help="Directory to save output files to")
    p.add_argument(
        "--tags", type=str, default=[], nargs="*", help="extra prompts to append in random combinations to prompt"
    )
    p.add_argument("--tag-threshold", type=float, default=0.4, help="chance to randomly append to prompt")
    p.add_argument("--dataset", type=str, default="datasets/", help="Directory to save output files to")

    args = p.parse_args()

    main(args)
