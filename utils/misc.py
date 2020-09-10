from argparse import ArgumentParser, Namespace
from collections import defaultdict
from time import time
from typing import Dict, List, Any

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter

from .boxOps import boxCxcywh2Xyxy


def baseParser() -> ArgumentParser:
    parser = ArgumentParser('Detection Transformer', add_help=False)

    # MARK: - model parameters
    # backbone
    parser.add_argument('--numGroups', default=8, type=int)
    parser.add_argument('--growthRate', default=32, type=int)
    parser.add_argument('--numBlocks', default=[6] * 4, type=list)

    # transformer
    parser.add_argument('--hiddenDims', default=512, type=int)
    parser.add_argument('--numHead', default=8, type=int)
    parser.add_argument('--numEncoderLayer', default=6, type=int)
    parser.add_argument('--numDecoderLayer', default=6, type=int)
    parser.add_argument('--dimFeedForward', default=2048, type=int)
    parser.add_argument('--dropout', default=.1, type=float)
    parser.add_argument('--numQuery', default=30, type=int)
    parser.add_argument('--numClass', default=5, type=int)

    # MARK: - dataset
    parser.add_argument('--targetHeight', default=608, type=int)
    parser.add_argument('--targetWidth', default=608, type=int)

    # MARK: - miscellaneous
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--weight', default='checkpoint/mango.pt', type=str)
    parser.add_argument('--seed', default=1588390, type=int)

    return parser


class MetricsLogger(object):
    def __init__(self, folder: str = './logs'):
        self.writer = SummaryWriter(folder)
        self.cache = defaultdict(list)
        self.lastStep = time()

    def addScalar(self, tag: str, value: Any, step: int = None, wallTime: float = None):
        self.writer.add_scalar(tag, value, step, wallTime)

    def close(self):
        self.writer.close()

    def flush(self):
        self.writer.flush()

    def step(self, metrics: dict, epoch: int, batch: int):
        elapse = time() - self.lastStep
        print(f'Elapse: {elapse: .4f}s, {1 / elapse: .2f} steps/sec')
        self.lastStep = time()

        for key in metrics:
            self.cache[key].append(metrics[key].cpu().item())

    def epochEnd(self, epoch: int):
        losses = []
        for key in self.cache:
            avg = np.mean(self.cache[key])
            if 'loss' in key:
                losses.append(avg)
            self.writer.add_scalar(f'Average/{key}', avg, epoch)

        avg = np.mean(losses)
        self.writer.add_scalar('Average/loss', avg, epoch)
        self.cache.clear()


class PostProcess(nn.Module):
    def __init__(self):
        super(PostProcess, self).__init__()

    @torch.no_grad()
    def forward(self, x: dict, imgSize: Tensor) -> List[Dict[str, Tensor]]:
        logits, bboxes = x['class'], x['bbox']

        prob = nn.functional.softmax(logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        boxes = boxCxcywh2Xyxy(bboxes)

        imgW, imgH = imgSize.unbind(1)

        scale = torch.stack([imgW, imgH, imgW, imgH], 1).unsqueeze(1)
        boxes *= scale

        return [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]


def saveArguments(args: Namespace, name: str):
    arr = []
    for key, val in vars(args).items():
        arr.append(f'--{key} "{val}"')

    with open(f'{name}-args.txt', 'w') as f:
        f.write(' '.join(arr))


def logMetrics(metrics: Dict[str, Tensor]):
    log = '[ '
    log += ' ] [ '.join([f'{k} = {v.cpu().item():.4f}' for k, v in metrics.items()])
    log += ' ]'
    print(log)


def cast2Float(x):
    if isinstance(x, list):
        return [cast2Float(y) for y in x]
    elif isinstance(x, dict):
        return {k: cast2Float(v) for k, v in x.items()}
    return x.float()
