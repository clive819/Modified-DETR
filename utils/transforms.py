import random
from typing import Dict, Tuple, Union

import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor

from .boxOps import boxCxcywh2Xyxy, boxXyxy2Cxcywh


class RandomOrder(object):
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img: Image.Image, targets: dict) -> Tuple[Image.Image, dict]:
        random.shuffle(self.transforms)
        for transform in self.transforms:
            img, targets = transform(img, targets)
        return img, targets


class RandomSizeCrop(object):
    def __init__(self, numClass, minScale=0.8):
        self.minScale = minScale
        self.numClass = numClass

    def __call__(self, img: Image.Image, targets: Dict[str, Tensor]) -> Tuple[Image.Image, Dict[str, Tensor]]:
        imgW, imgH = img.size

        scaleW = random.uniform(self.minScale, 1)
        scaleH = random.uniform(self.minScale, 1)

        newW = int(imgW * scaleW)
        newH = int(imgH * scaleH)

        region = T.RandomCrop.get_params(img, (newH, newW))

        # fix bboxes
        i, j, h, w = region
        boxes = boxCxcywh2Xyxy(targets['boxes']) * torch.as_tensor([imgW, imgH, imgW, imgH], dtype=torch.float32)
        maxSize = torch.as_tensor([w, h], dtype=torch.float32)
        croppedBoxes = boxes - torch.as_tensor([j, i, j, i])
        croppedBoxes = torch.min(croppedBoxes.reshape(-1, 2, 2), maxSize)
        croppedBoxes = croppedBoxes.clamp(min=0)

        mask = ~torch.all(croppedBoxes[:, 1, :] > croppedBoxes[:, 0, :], dim=1)
        targets['labels'][mask] = self.numClass

        targets["boxes"] = boxXyxy2Cxcywh(
            croppedBoxes.reshape(-1, 4) / torch.as_tensor([newW, newH, newW, newH], dtype=torch.float32))

        return T.functional.crop(img, *region), targets


class Normalize(object):
    def __init__(self):
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self, img: Image.Image, targets: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        img = self.transform(img)
        boxes = boxCxcywh2Xyxy(targets['boxes']).clamp(0, 1)
        targets['boxes'] = boxXyxy2Cxcywh(boxes)

        return img, targets


class Resize(object):
    def __init__(self, *args, **kwargs):
        self.transform = T.Resize(*args, **kwargs)

    def __call__(self, img: Image.Image, targets: dict) -> Tuple[Image.Image, dict]:
        return self.transform(img), targets


class RandomVerticalFlip(object):
    def __init__(self, p=.5):
        self.p = p

    def __call__(self, img: Image.Image, targets: Dict[str, Tensor]) -> Tuple[Image.Image, Dict[str, Tensor]]:
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            targets['boxes'][..., 1] = 1 - targets['boxes'][..., 1]
        return img, targets


class RandomHorizontalFlip(object):
    def __init__(self, p=.5):
        self.p = p

    def __call__(self, img: Image.Image, targets: Dict[str, Tensor]) -> Tuple[Image.Image, Dict[str, Tensor]]:
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            targets['boxes'][..., 0] = 1 - targets['boxes'][..., 0]
        return img, targets


class ColorJitter(object):
    def __init__(self, *args, **kwargs):
        self.transform = T.ColorJitter(*args, **kwargs)

    def __call__(self, img: Image.Image, targets: dict) -> Tuple[Image.Image, dict]:
        return self.transform(img), targets


class Compose(object):
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img: Image.Image, targets: dict) -> Tuple[Union[Tensor, Image.Image], dict]:
        for transform in self.transforms:
            img, targets = transform(img, targets)
        return img, targets
