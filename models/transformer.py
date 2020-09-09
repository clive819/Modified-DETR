from copy import deepcopy
from typing import Optional

import torch
from torch import nn, Tensor


class Transformer(nn.Module):
    def __init__(self, hiddenDims: int, numHead: int, numEncoderLayer: int, numDecoderLayer: int, dimFeedForward: int,
                 dropout: float):
        super(Transformer, self).__init__()

        encoderLayer = TransformerEncoderLayer(hiddenDims, numHead, dimFeedForward, dropout)
        self.encoder = TransformerEncoder(encoderLayer, numEncoderLayer)

        decoderLayer = TransformerDecoderLayer(hiddenDims, numHead, dimFeedForward, dropout)
        self.decoder = TransformerDecoder(decoderLayer, numDecoderLayer)

        self.resetParameters()

    def resetParameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor, mask: Tensor, query: Tensor, pos: Tensor) -> Tensor:
        """
        :param src: tensor of shape [batchSize, hiddenDims, imageHeight // 32, imageWidth // 32]

        :param mask: tensor of shape [batchSize, imageHeight // 32, imageWidth // 32]
                     Please refer to detr.py for more detailed description.

        :param query: object queries, tensor of shape [numQuery, hiddenDims].

        :param pos: positional encoding, the same shape as src.

        :return: tensor of shape [batchSize, numQuery * numDecoderLayer, hiddenDims]
        """
        N = src.shape[0]

        src = src.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        pos = pos.flatten(2).permute(2, 0, 1)
        query = query.unsqueeze(1).repeat(1, N, 1)
        tgt = torch.zeros_like(query)

        memory = self.encoder(src, srcKeyPaddingMask=mask, pos=pos)
        out = self.decoder(tgt, memory, memoryKeyPaddingMask=mask, pos=pos, queryPos=query).transpose(1, 2)

        return out


class TransformerEncoder(nn.Module):
    def __init__(self, encoderLayer: nn.Module, numLayers: int):
        super(TransformerEncoder, self).__init__()

        self.layers = getClones(encoderLayer, numLayers)

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, srcKeyPaddingMask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None) -> Tensor:
        out = src

        for layer in self.layers:
            out = layer(out, mask, srcKeyPaddingMask, pos)

        return out


class TransformerDecoder(nn.Module):
    def __init__(self, decoderLayer: nn.Module, numLayers: int):
        super(TransformerDecoder, self).__init__()

        self.layers = getClones(decoderLayer, numLayers)

    def forward(self, tgt: Tensor, memory: Tensor, tgtMask: Optional[Tensor] = None,
                memoryMask: Optional[Tensor] = None, tgtKeyPaddingMask: Optional[Tensor] = None,
                memoryKeyPaddingMask: Optional[Tensor] = None, pos: Optional[Tensor] = None,
                queryPos: Optional[Tensor] = None) -> Tensor:
        out = tgt

        intermediate = []

        for layer in self.layers:
            out = layer(out, memory, tgtMask, memoryMask, tgtKeyPaddingMask, memoryKeyPaddingMask, pos, queryPos)
            intermediate.append(out)

        return torch.stack(intermediate)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hiddenDims: int, numHead: int, dimFeedForward: int, dropout: float):
        super(TransformerEncoderLayer, self).__init__()

        self.attention = nn.MultiheadAttention(hiddenDims, numHead, dropout=dropout)

        self.linear1 = nn.Linear(hiddenDims, dimFeedForward)
        self.linear2 = nn.Linear(dimFeedForward, hiddenDims)

        self.norm1 = nn.LayerNorm(hiddenDims)
        self.norm2 = nn.LayerNorm(hiddenDims)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, srcKeyPaddingMask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None) -> Tensor:
        q = k = withPosEmbed(src, pos)
        src2 = self.attention(q, k, value=src, attn_mask=mask, key_padding_mask=srcKeyPaddingMask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, hiddenDims: int, numHead: int, dimFeedForward: int, dropout: float):
        super(TransformerDecoderLayer, self).__init__()

        self.attention1 = nn.MultiheadAttention(hiddenDims, numHead, dropout=dropout)
        self.attention2 = nn.MultiheadAttention(hiddenDims, numHead, dropout=dropout)

        self.linear1 = nn.Linear(hiddenDims, dimFeedForward)
        self.linear2 = nn.Linear(dimFeedForward, hiddenDims)

        self.norm1 = nn.LayerNorm(hiddenDims)
        self.norm2 = nn.LayerNorm(hiddenDims)
        self.norm3 = nn.LayerNorm(hiddenDims)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, tgt: Tensor, memory: Tensor, tgtMask: Optional[Tensor] = None,
                memoryMask: Optional[Tensor] = None, tgtKeyPaddingMask: Optional[Tensor] = None,
                memoryKeyPaddingMask: Optional[Tensor] = None, pos: Optional[Tensor] = None,
                queryPos: Optional[Tensor] = None) -> Tensor:
        q = k = withPosEmbed(tgt, queryPos)
        tgt2 = self.attention1(q, k, value=tgt, attn_mask=tgtMask, key_padding_mask=tgtKeyPaddingMask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.attention2(query=withPosEmbed(tgt, queryPos), key=withPosEmbed(memory, pos),
                               value=memory, attn_mask=memoryMask, key_padding_mask=memoryKeyPaddingMask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def withPosEmbed(tensor: Tensor, pos: Optional[Tensor] = None) -> Tensor:
    return tensor + pos if pos is not None else tensor


def getClones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([deepcopy(module) for _ in range(N)])
