from typing import Tuple

import torch
from torch import Tensor


@torch.no_grad()
def boxCxcywh2Xyxy(box: Tensor) -> Tensor:
    cx, cy, w, h = box.unbind(-1)

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack([x1, y1, x2, y2], -1)


@torch.no_grad()
def boxXyxy2Cxcywh(box: Tensor) -> Tensor:
    x1, y1, x2, y2 = box.unbind(-1)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    return torch.stack([cx, cy, w, h], -1)


@torch.no_grad()
def boxIoU(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    boxes1Area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2Area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    intersectArea = wh[:, :, 0] * wh[:, :, 1]

    unionArea = boxes1Area[:, None] + boxes2Area - intersectArea

    iou = intersectArea / unionArea
    return iou, unionArea


@torch.no_grad()
def gIoU(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    iou, unionArea = boxIoU(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)

    enclosingArea = wh[:, :, 0] * wh[:, :, 1]

    return iou - (enclosingArea - unionArea) / enclosingArea
