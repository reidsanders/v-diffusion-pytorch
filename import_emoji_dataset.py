#!/usr/bin/env python3

from copy import deepcopy
from functools import partial
from pathlib import Path
from tkinter import W
from PIL import Image
from tqdm import trange
import json
import typer
from pprint import pprint
import imgkit
import requests
import re
import base64

app = typer.Typer()


@app.command()
def preview(indexfile: str, outdir: str = "out/"):
    with open(indexfile, "r") as f:
        data = json.loads(f.read())

    pprint(data[1])

    # imgkit.from_url('http://google.com', 'out.jpg')
    # imgkit.from_file('test.html', 'out.jpg')
    options = {
        "format": "png",
        "crop-h": "256",
        "crop-w": "256",
        "crop-x": "0",
        "crop-y": "0",
        "encoding": "UTF-8",
    }
    emoji = data[1]
    # imgkit.from_string(data[1]['unicode'], 'emoji.jpg', options=options)
    code = re.sub("U+", "\\u000", emoji["unicode"])
    print(code)
    e = EmojiConverter()
    b64 = e.to_base64_png(emoji["name"])
    print(b64)
    with open("out.jpg", "wb") as f:
        img = base64.b64decode(b64)
        f.write(img)

    # imgkit.from_string(u"\U0001F482", 'emoji.jpg', options=options)


class EmojiConverter:
    def __init__(self):
        self.data = requests.get("https://unicode.org/emoji/charts/full-emoji-list.html").text
        self.skintones_data = requests.get("https://unicode.org/emoji/charts/full-emoji-modifiers.html").text

    def to_base64_png(self, emoji, version=0):
        """For different versions, you can set version = 0 for ,"""
        html_search_string = r"<img alt='{}' class='imga' src='data:image/png;base64,([^']+)'>"  #'
        try:
            matchlist = re.findall(html_search_string.format(emoji), self.data)
        except Exception:
            matchlist = re.findall(html_search_string.format(emoji), self.skintones_data)
        return matchlist[version]


if __name__ == "__main__":
    app()
