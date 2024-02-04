# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""

from .block import (C1, C2, C3, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, GhostBottleneck,
                    HGBlock, HGStem, Proto, RepC3, C2fOD, BottleneckOD, CABlock, C2fFaster, SEBlock, SKBlock, C2fBoT,
                    AFPNC2f, AFPNPConv, FasterBlocks, C2fCondConv, CPCA, ECA)
from .conv import (CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d, Focus,
                   GhostConv, LightConv, RepConv, SpatialAttention, ConvOD, ODConv, PartialConv, PConv, CondConv)
from .head import Classify, Detect, Pose, RTDETRDecoder, Segment
from .transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer)
from .FasterNet import FasterNet, FasterNetBlock, PatchEmbedding, PatchMerging, BasicStage as FasterBasicStage
from .MobileViT import MobileViTBlock, MV2Block
from .PoolFormer import PoolFormerBlocks, PatchEmbedding as PoolEmbedding
__all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
           'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'TransformerLayer',
           'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
           'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
           'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP',
           'ConvOD', 'ODConv', 'C2fOD', 'BottleneckOD', 'CABlock', 'FasterNet', 'C2fFaster',
           'FasterNetBlock', 'PatchEmbedding', 'PatchMerging', 'FasterBasicStage', 'SKBlock', 'SEBlock', 'C2fBoT',
           'MobileViTBlock', 'MV2Block', 'AFPNC2f', 'AFPNPConv', 'PConv', 'FasterBlocks', 'CondConv',
           'C2fCondConv', 'CPCA', 'ECA', 'PoolEmbedding', 'PoolFormerBlocks'
           )
