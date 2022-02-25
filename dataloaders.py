import json
from PIL import Image
from pathlib import Path
import sys
import random
from torch.utils import data


class DrawtextCaptions(data.Dataset):
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
                    f"Bad image, skipping: {index} {image_path} " f"{type(err).__name__}: {err!s}",
                    file=sys.stderr,
                )
                return self[random.randrange(len(self))]
        except Exception as err:
            print(f"{type(err).__name__}: {err!s}", file=sys.stderr)
            # return self[random.randrange(len(self))]
            raise


class JsonTextCaptions(data.Dataset):
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
                image_path = Path(self.root) / datapoint["filename"]
                image = Image.open(image_path)

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
                    f"Bad image, skipping: {index} {image_path} " f"{type(err).__name__}: {err!s}",
                    file=sys.stderr,
                )
        except Exception as err:
            print(f"{type(err).__name__}: {err!s}", file=sys.stderr)
            # return self[random.randrange(len(self))]
            raise


class GoodbotCaptions(data.Dataset):
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
                image_path = Path(self.root) / datapoint["filename"]
                image = Image.open(image_path)

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
                    f"Bad image, skipping: {index} {image_path} " f"{type(err).__name__}: {err!s}",
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
                image_path = Path(self.root) / datapoint["filename"]
                image = Image.open(image_path)
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
                    f"Bad image, skipping: {index} {image_path} " f"{type(err).__name__}: {err!s}",
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
                    f"Bad image, skipping: {index} {image_path} " f"{type(err).__name__}: {err!s}",
                    file=sys.stderr,
                )
                return self[random.randrange(len(self))]
        except Exception as err:
            print(f"{type(err).__name__}: {err!s}", file=sys.stderr)
            # return self[random.randrange(len(self))]
            raise
