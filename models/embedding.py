import math
from typing import Tuple

import torch
from torch import nn, Tensor


class PositionEmbeddingSine(nn.Module):
    def __init__(self, numPositionFeatures: int = 64, temperature: int = 10000, normalize: bool = True,
                 scale: float = None):
        super(PositionEmbeddingSine, self).__init__()

        self.numPositionFeatures = numPositionFeatures
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        N, _, H, W = x.shape

        mask = torch.zeros(N, H, W, dtype=torch.bool, device=x.device)
        notMask = ~mask

        yEmbed = notMask.cumsum(1)
        xEmbed = notMask.cumsum(2)

        if self.normalize:
            epsilon = 1e-6
            yEmbed = yEmbed / (yEmbed[:, -1:, :] + epsilon) * self.scale
            xEmbed = xEmbed / (xEmbed[:, :, -1:] + epsilon) * self.scale

        dimT = torch.arange(self.numPositionFeatures, dtype=torch.float32, device=x.device)
        dimT = self.temperature ** (2 * (dimT // 2) / self.numPositionFeatures)

        posX = xEmbed.unsqueeze(-1) / dimT
        posY = yEmbed.unsqueeze(-1) / dimT

        posX = torch.stack((posX[:, :, :, 0::2].sin(), posX[:, :, :, 1::2].cos()), -1).flatten(3)
        posY = torch.stack((posY[:, :, :, 0::2].sin(), posY[:, :, :, 1::2].cos()), -1).flatten(3)

        return torch.cat((posY, posX), 3).permute(0, 3, 1, 2), mask
