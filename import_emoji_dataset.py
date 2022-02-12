#!/usr/bin/env python3

from copy import deepcopy
from functools import partial
from pathlib import Path
from PIL import Image
from tqdm import trange
import json
import typer
from pprint import pprint
import imgkit

app = typer.Typer()

@app.command()
def preview(indexfile: 'str'):
    with open(indexfile, "r") as f:
        data = json.loads(f.read())
    
    pprint(data[1])

    # imgkit.from_url('http://google.com', 'out.jpg')
    # imgkit.from_file('test.html', 'out.jpg')
    imgkit.from_string('Hello!', 'out.jpg')
    

if __name__ == "__main__":
    app()

