from typing import Dict, List, Tuple

import torch
from torch import nn, Tensor

from utils.boxOps import boxCxcywh2Xyxy, gIoU, boxIoU
from .matcher import HungarianMatcher


class SetCriterion(nn.Module):
    def __init__(self, args):
        super(SetCriterion, self).__init__()

        self.matcher = HungarianMatcher(args.classCost, args.bboxCost, args.giouCost)
        self.numClass = args.numClass

        self.classCost = args.classCost
        self.bboxCost = args.bboxCost
        self.giouCost = args.giouCost

        emptyWeight = torch.ones(args.numClass + 1)
        emptyWeight[-1] = args.eosCost
        self.register_buffer('emptyWeight', emptyWeight)

    def forward(self, x: Dict[str, Tensor], y: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        ans = self.computeLoss(x, y)

        for i, aux in enumerate(x['aux']):
            ans.update({f'{k}_aux{i}': v for k, v in self.computeLoss(aux, y).items()})

        return ans

    def computeLoss(self, x: Dict[str, Tensor], y: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """
        :param x: a dictionary containing:
            'class': a tensor of shape [batchSize, numQuery * numDecoderLayer, numClass + 1]
            'bbox': a tensor of shape [batchSize, numQuery * numDecoderLayer, 4]

        :param y: a list of dictionaries containing:
            'labels': a tensor of shape [numObjects] that stores the ground-truth classes of objects
            'boxes': a tensor of shape [numObjects, 4] that stores the ground-truth bounding boxes of objects
            represented as [centerX, centerY, w, h]

        :return: a dictionary containing classification loss, bbox loss, and gIoU loss
        """
        ids = self.matcher(x, y)
        idx = self.getPermutationIdx(ids)

        # MARK: - classification loss
        logits = x['class']

        targetClassO = torch.cat([t['labels'] for t, (_, J) in zip(y, ids)])
        targetClass = torch.full(logits.shape[:2], self.numClass, dtype=torch.int64, device=logits.device)
        targetClass[idx] = targetClassO

        classificationLoss = nn.functional.cross_entropy(logits.transpose(1, 2), targetClass, self.emptyWeight)
        classificationLoss *= self.classCost

        # MARK: - bbox loss
        # ignore boxes that has no object
        mask = targetClassO != self.numClass
        boxes = x['bbox'][idx][mask]
        targetBoxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(y, ids)], 0)[mask]

        numBoxes = len(targetBoxes) + 1e-6

        bboxLoss = nn.functional.l1_loss(boxes, targetBoxes, reduction='none')
        bboxLoss = bboxLoss.sum() / numBoxes
        bboxLoss *= self.bboxCost

        # MARK: - giou loss
        giouLoss = 1 - torch.diag(gIoU(boxCxcywh2Xyxy(boxes), boxCxcywh2Xyxy(targetBoxes)))
        giouLoss = giouLoss.sum() / numBoxes
        giouLoss *= self.giouCost

        # MARK: - compute statistics
        with torch.no_grad():
            predClass = nn.functional.softmax(logits[idx], -1).max(-1)[1]
            classMask = (predClass == targetClassO)[mask]
            iou = torch.diag(boxIoU(boxCxcywh2Xyxy(boxes), boxCxcywh2Xyxy(targetBoxes))[0])

            ap = []
            for threshold in range(50, 100, 5):
                ap.append(((iou >= threshold / 100) * classMask).sum() / numBoxes)

            ap = torch.mean(torch.stack(ap))

        return {'classification loss': classificationLoss,
                'bbox loss': bboxLoss,
                'gIoU loss': giouLoss,
                'mAP': ap}

    @staticmethod
    def getPermutationIdx(indices: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        batchIdx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        srcIdx = torch.cat([src for (src, _) in indices])
        return batchIdx, srcIdx
