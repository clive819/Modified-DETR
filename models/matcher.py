from typing import Dict, List, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn, Tensor

from utils.boxOps import gIoU, boxCxcywh2Xyxy


class HungarianMatcher(nn.Module):
    def __init__(self, classCost: float = 1, bboxCost: float = 5, giouCost: float = 2):
        super(HungarianMatcher, self).__init__()

        self.classCost = classCost
        self.bboxCost = bboxCost
        self.giouCost = giouCost

    def forward(self, x: Dict[str, Tensor], y: List[Dict[str, Tensor]]) -> List[Tuple[Tensor, Tensor]]:
        batchSize, numQuery = x['class'].shape[:2]

        outProb = x['class'].flatten(0, 1).softmax(-1)
        outBbox = x['bbox'].flatten(0, 1)

        tgtIds = torch.cat([t['labels'] for t in y])
        tgtBbox = torch.cat([t['boxes'] for t in y])

        classLoss = -outProb[:, tgtIds]
        bboxLoss = torch.cdist(outBbox, tgtBbox, p=1)
        giouLoss = -gIoU(boxCxcywh2Xyxy(outBbox), boxCxcywh2Xyxy(tgtBbox))

        costMatrix = self.bboxCost * bboxLoss + self.classCost * classLoss + self.giouCost * giouLoss
        costMatrix = costMatrix.view(batchSize, numQuery, -1).cpu().detach()

        sizes = [len(t['boxes']) for t in y]

        ids = [linear_sum_assignment(c[i]) for i, c in enumerate(costMatrix.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in ids]
